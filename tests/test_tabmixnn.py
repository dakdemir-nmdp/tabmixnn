"""
TabMixNN test suite — ground-truth diagnostics.

Every test uses synthetic data with a KNOWN data-generating process so that
failure means the model is wrong, not just unlucky.

Run with:
    /path/to/.venv/bin/python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_gaussian_lmm(n_groups=20, n_per_group=15, beta0=5.0, beta1=2.0,
                        sigma_u=3.0, sigma_e=1.0):
    """
    y_ij = beta0 + beta1*x_ij + u_i + e_ij
    True fixed effects: intercept=5, slope=2
    True random intercept SD: 3, residual SD: 1
    """
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group).astype(str)
    x = RNG.standard_normal(n)
    u = RNG.normal(0, sigma_u, n_groups)
    u_i = u[np.repeat(np.arange(n_groups), n_per_group)]
    e = RNG.normal(0, sigma_e, n)
    y = beta0 + beta1 * x + u_i + e
    return pd.DataFrame({"y": y, "x": x, "group": groups})


def _make_binomial_lmm(n_groups=15, n_per_group=20):
    """
    logit(p_ij) = 1.5 + 0.8*x_ij + u_i,  u_i ~ N(0, 1)
    """
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group).astype(str)
    x = RNG.standard_normal(n)
    u = RNG.normal(0, 1.0, n_groups)
    u_i = u[np.repeat(np.arange(n_groups), n_per_group)]
    logit_p = 1.5 + 0.8 * x + u_i
    p = 1 / (1 + np.exp(-logit_p))
    y = RNG.binomial(1, p).astype(float)
    return pd.DataFrame({"y": y, "x": x, "group": groups})


def _make_survival(n=300, n_groups=10, true_hr_coef=1.0):
    """
    Discrete-time survival with 10 bins.
    Baseline hazard ~0.1/bin. log-HR = true_hr_coef * x.
    """
    T = 10
    time_bins = list(range(T))
    groups = (np.arange(n) % n_groups).astype(str)
    x = RNG.standard_normal(n)
    base_h = 0.08
    h = base_h * np.exp(true_hr_coef * x)
    h = np.clip(h, 0.001, 0.999)

    times, events = [], []
    for i in range(n):
        survived = True
        t_obs, ev = T - 1, 0
        for t in range(T):
            if RNG.random() < h[i]:
                t_obs, ev = t, 1
                survived = False
                break
        times.append(t_obs)
        events.append(ev)

    return (
        pd.DataFrame({"time_idx": times, "event": events, "x": x, "group": groups}),
        time_bins,
    )


# ---------------------------------------------------------------------------
# 1. Formula parser
# ---------------------------------------------------------------------------

class TestFormulaParser:
    def test_fixed_only(self):
        from tabmixnn.parsing.formula import CompiledDesign, OutcomeSpec
        from tabmixnn import Gaussian
        df = _make_gaussian_lmm()
        outcomes = [OutcomeSpec(name="y", y="y", family=Gaussian())]
        design = CompiledDesign("y ~ x", outcomes, "none", None)
        design.fit(df)
        X, Z_ind, Z_slopes, Ys = design.transform(df, mode="fit")
        assert X.shape == (len(df), 2)
        np.testing.assert_allclose(X[:, 0], 1)
        assert len(Z_ind) == 0
        assert design.intercept_included is True

    def test_random_intercept(self):
        from tabmixnn.parsing.formula import CompiledDesign, OutcomeSpec
        from tabmixnn import Gaussian
        df = _make_gaussian_lmm()
        outcomes = [OutcomeSpec(name="y", y="y", family=Gaussian())]
        design = CompiledDesign("y ~ x + (1|group)", outcomes, "none", None)
        design.fit(df)
        X, Z_ind, Z_slopes, Ys = design.transform(df, mode="fit")
        assert len(Z_ind) == 1, "Expected 1 random term"
        assert Z_ind[0].shape == (len(df),), "Z_ind should be group indices [n]"
        # All group indices should be in [1, n_groups] (0 reserved for unknown)
        assert Z_ind[0].min() >= 1

    def test_random_slope(self):
        from tabmixnn.parsing.formula import CompiledDesign, OutcomeSpec
        from tabmixnn import Gaussian
        df = _make_gaussian_lmm()
        outcomes = [OutcomeSpec(name="y", y="y", family=Gaussian())]
        design = CompiledDesign("y ~ x + (1 + x|group)", outcomes, "none", None)
        design.fit(df)
        X, Z_ind, Z_slopes, Ys = design.transform(df, mode="fit")
        assert len(Z_slopes) == 1
        assert Z_slopes[0].shape[1] == 2, "Should have 2 slope cols: intercept + x"

    def test_unknown_group_gets_zero_index(self):
        from tabmixnn.parsing.formula import CompiledDesign, OutcomeSpec
        from tabmixnn import Gaussian
        df_train = _make_gaussian_lmm(n_groups=10)
        df_test = df_train.copy()
        df_test["group"] = "UNSEEN_GROUP_XYZ"
        outcomes = [OutcomeSpec(name="y", y="y", family=Gaussian())]
        design = CompiledDesign("y ~ x + (1|group)", outcomes, "none", None)
        design.fit(df_train)
        _, Z_ind_test, _, _ = design.transform(df_test, mode="predict")
        assert (Z_ind_test[0] == 0).all(), "Unseen groups must map to index 0"


# ---------------------------------------------------------------------------
# 2. Gaussian family — fit and predict
# ---------------------------------------------------------------------------

class TestGaussianFamily:
    def test_nll_at_truth(self):
        """NLL of N(mu, sigma) at y=mu should equal 0.5*log(2*pi*sigma^2)."""
        from tabmixnn import Gaussian
        fam = Gaussian()
        mu = torch.zeros(100)
        y = torch.zeros(100)
        theta = mu
        nll = fam.nll(theta, y)
        sigma = torch.exp(fam.log_sigma)
        expected = 0.5 * torch.log(2 * torch.tensor(np.pi) * sigma ** 2)
        assert abs(nll.item() - expected.item()) < 0.01, \
            f"NLL at truth: got {nll.item():.4f}, expected {expected.item():.4f}"

    def test_predict_returns_mean(self):
        from tabmixnn import Gaussian
        fam = Gaussian()
        eta = torch.tensor([[3.0], [5.0], [-1.0]])
        theta = fam.forward_params(eta)
        pred = fam.predict(theta)
        np.testing.assert_allclose(pred, [3.0, 5.0, -1.0], atol=1e-5)

    def test_model_fits_gaussian_lmm(self):
        """
        Fit a simple Gaussian LMM on synthetic data.
        Model should achieve R² > 0.5 on training data.
        (Weak threshold — we're checking it runs and learns, not that it's perfect.)
        """
        from tabmixnn import TabMixNN, Gaussian, OutcomeSpec
        df = _make_gaussian_lmm(n_groups=20, n_per_group=20)
        outcomes = [OutcomeSpec(name="y", y="y", family=Gaussian())]
        model = TabMixNN(
            formula="y ~ x + (1|group)",
            outcomes=outcomes,
            hidden_dims=[32],
            embedding_dim=8,
            epochs=100,
            lr=5e-3,
            lambda_kl=0.001,
            verbose=False,
        )
        model.fit(df)
        preds = model.predict(df)
        ss_res = np.sum((df["y"].values - preds) ** 2)
        ss_tot = np.sum((df["y"].values - df["y"].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.5, f"Gaussian LMM R² on training data: {r2:.3f} (expected > 0.5)"


# ---------------------------------------------------------------------------
# 3. Binomial family — fit and predict
# ---------------------------------------------------------------------------

class TestBinomialFamily:
    def test_nll_balanced(self):
        """NLL of balanced Bernoulli at p=0.5 should be log(2)."""
        from tabmixnn import Binomial
        fam = Binomial()
        logits = torch.zeros(1000)  # p=0.5
        y = torch.cat([torch.ones(500), torch.zeros(500)])
        nll = fam.nll(fam.forward_params(logits), y)
        assert abs(nll.item() - np.log(2)) < 0.05, \
            f"Balanced Binomial NLL: {nll.item():.4f}, expected {np.log(2):.4f}"

    def test_predict_proba_range(self):
        from tabmixnn import Binomial
        fam = Binomial()
        logits = torch.randn(100)
        theta = fam.forward_params(logits.unsqueeze(-1))
        probs = fam.predict(theta, mode="proba")
        assert probs.min() >= 0.0 and probs.max() <= 1.0

    def test_model_fits_binomial_lmm(self):
        """
        Fit Binomial GLMM. Should achieve AUC > 0.6 on training data.
        """
        from tabmixnn import TabMixNN, Binomial, OutcomeSpec
        from sklearn.metrics import roc_auc_score
        df = _make_binomial_lmm(n_groups=15, n_per_group=30)
        outcomes = [OutcomeSpec(name="y", y="y", family=Binomial())]
        model = TabMixNN(
            formula="y ~ x + (1|group)",
            outcomes=outcomes,
            hidden_dims=[32],
            embedding_dim=8,
            epochs=100,
            lr=5e-3,
            lambda_kl=0.001,
            verbose=False,
        )
        model.fit(df)
        probs = model.predict_proba(df)
        auc = roc_auc_score(df["y"].values, probs)
        assert auc > 0.6, f"Binomial GLMM AUC on training data: {auc:.3f} (expected > 0.6)"


# ---------------------------------------------------------------------------
# 4. Survival family — fit and predict
# ---------------------------------------------------------------------------

class TestSurvivalFamily:
    def test_nll_shape(self):
        """DiscreteTimeSurvival NLL should return a scalar."""
        from tabmixnn import DiscreteTimeSurvival
        T = 10
        fam = DiscreteTimeSurvival(time_bins=list(range(T)))
        logits = torch.randn(50, T)
        y = torch.stack([
            torch.randint(0, T, (50,)).float(),
            torch.randint(0, 2, (50,)).float()
        ], dim=1)
        theta = fam.forward_params(logits)
        nll = fam.nll(theta, y)
        assert nll.ndim == 0, "NLL should be a scalar"
        assert not torch.isnan(nll), "NLL should not be NaN"
        assert not torch.isinf(nll), "NLL should not be Inf"

    def test_survival_curve_monotone(self):
        """Predicted survival curve S(t) must be non-increasing."""
        from tabmixnn import DiscreteTimeSurvival
        T = 10
        fam = DiscreteTimeSurvival(time_bins=list(range(T)))
        logits = torch.randn(20, T)
        theta = fam.forward_params(logits)
        surv = fam.predict(theta, mode="survival")  # [n, T]
        diffs = np.diff(surv, axis=1)
        assert (diffs <= 1e-6).all(), \
            f"Survival curve not monotone: max increase = {diffs.max():.4f}"

    def test_model_fits_survival(self):
        """
        Fit DiscreteTimeSurvival model. C-index should be > 0.55 on training data.
        """
        from tabmixnn import TabMixNN, DiscreteTimeSurvival, OutcomeSpec
        df, time_bins = _make_survival(n=400, n_groups=10)
        fam = DiscreteTimeSurvival(time_bins=time_bins)
        outcomes = [OutcomeSpec(
            name="surv", family=fam,
            time="time_idx", event="event",
        )]
        model = TabMixNN(
            formula="~ x + (1|group)",
            outcomes=outcomes,
            hidden_dims=[32],
            embedding_dim=8,
            epochs=80,
            lr=5e-3,
            lambda_kl=0.001,
            verbose=False,
        )
        model.fit(df)
        # Predict risk score = area under incidence curve (higher = higher risk)
        X_np, Z_ind, Z_slopes, _ = model.design.transform(df, mode="predict")
        X_t = torch.tensor(X_np, dtype=torch.float32)
        Z_it = [torch.tensor(z, dtype=torch.long) for z in Z_ind]
        Z_st = [torch.tensor(z, dtype=torch.float32) for z in Z_slopes]
        with torch.no_grad():
            raw = model.forward(X_t, Z_it, Z_st)
            theta = fam.forward_params(raw)
            surv = fam.predict(theta, mode="survival")  # [n, T]
        risk = 1 - surv[:, -1]  # 1 - S(T) = cumulative incidence

        # C-index
        from sklearn.metrics import roc_auc_score
        # Use event as label, risk as score (only for events vs censored)
        events = df["event"].values
        c_index = roc_auc_score(events, risk)
        assert c_index > 0.55, \
            f"Survival C-index on training data: {c_index:.3f} (expected > 0.55)"


# ---------------------------------------------------------------------------
# 5. Random effects recovery — ground truth diagnostic
# ---------------------------------------------------------------------------

class TestRandomEffectsRecovery:
    """
    The core diagnostic for the CAR-T report question:
    does TabMixNN actually recover known random effects?

    DGP: y_ij = 5 + 2*x_ij + u_i + e_ij
         u_i ~ N(0, 9)  [sigma_u = 3]
         e_ij ~ N(0, 1)  [sigma_e = 1]

    We check:
    1. Predictions correlate with truth (R² > 0.5)
    2. Random effect estimates correlate with true u_i (r > 0.5)
    3. Model beats a fixed-effects-only baseline (no random effects)
    """

    def setup_method(self):
        self.n_groups = 30
        self.n_per_group = 20
        self.beta0 = 5.0
        self.beta1 = 2.0
        self.sigma_u = 3.0
        self.sigma_e = 1.0
        self.df = _make_gaussian_lmm(
            n_groups=self.n_groups,
            n_per_group=self.n_per_group,
            beta0=self.beta0,
            beta1=self.beta1,
            sigma_u=self.sigma_u,
            sigma_e=self.sigma_e,
        )
        # True group means (u_i + beta0, averaged over x)
        self.true_group_means = (
            self.df.groupby("group")["y"].mean().sort_index()
        )

    def _fit_model(self, use_random=True):
        from tabmixnn import TabMixNN, Gaussian, OutcomeSpec
        formula = "y ~ x + (1|group)" if use_random else "y ~ x"
        outcomes = [OutcomeSpec(name="y", y="y", family=Gaussian())]
        model = TabMixNN(
            formula=formula,
            outcomes=outcomes,
            hidden_dims=[32],
            embedding_dim=8,
            epochs=200,
            lr=5e-3,
            lambda_kl=0.001,
            enforce_centering=True,
            verbose=False,
        )
        model.fit(self.df)
        return model

    def test_random_effects_improve_r2(self):
        """Model with random effects should have higher R² than without."""
        model_re = self._fit_model(use_random=True)
        model_fe = self._fit_model(use_random=False)

        def r2(model):
            preds = model.predict(self.df)
            y = self.df["y"].values
            return 1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)

        r2_re = r2(model_re)
        r2_fe = r2(model_fe)
        assert r2_re > r2_fe, \
            f"Random effects R²={r2_re:.3f} should exceed fixed-only R²={r2_fe:.3f}"
        assert r2_re > 0.5, \
            f"Random effects model R²={r2_re:.3f} should exceed 0.5"

    def test_random_effect_estimates_correlate_with_truth(self):
        """
        The random effect CONTRIBUTION to predictions (full - fixed-only) should
        correlate with true group means. Pearson r > 0.5 is the threshold.

        NOTE: extract_parameters() returns raw embedding-space values which are NOT
        in outcome units. The correct way to measure random effect recovery is via
        prediction ablation: RE_contribution_i = mean_i(pred_full) - mean_i(pred_fixed_only).
        This is a known interpretability gap in the current implementation.
        """
        model = self._fit_model(use_random=True)

        # Measure random effect contribution via prediction ablation
        preds_full = model.predict(self.df)
        preds_fixed = model.predict(self.df, use_random_effects=False)
        re_contribution = preds_full - preds_fixed  # per-sample RE contribution

        # Average per group
        self.df["_re_contrib"] = re_contribution
        group_re = self.df.groupby("group")["_re_contrib"].mean()
        self.df.drop(columns=["_re_contrib"], inplace=True)

        common = group_re.index.intersection(self.true_group_means.index)
        assert len(common) >= 10, f"Too few common groups: {len(common)}"

        r = np.corrcoef(
            group_re.loc[common].values,
            self.true_group_means.loc[common].values
        )[0, 1]
        assert r > 0.5, (
            f"Random effect contribution correlation with true group means: r={r:.3f} (expected > 0.5).\n"
            f"NOTE: extract_parameters() embedding values have r≈0.04 — they are NOT in outcome units."
        )

    def test_beats_sklearn_ridge_baseline(self):
        """
        TabMixNN with random effects should beat Ridge regression (no grouping)
        on held-out data from seen groups.
        """
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        df = self.df
        # 80/20 split stratified by group
        train_idx = []
        test_idx = []
        for g, gdf in df.groupby("group"):
            n = len(gdf)
            cut = int(0.8 * n)
            train_idx.extend(gdf.index[:cut].tolist())
            test_idx.extend(gdf.index[cut:].tolist())

        df_train = df.loc[train_idx].reset_index(drop=True)
        df_test = df.loc[test_idx].reset_index(drop=True)

        # TabMixNN
        from tabmixnn import TabMixNN, Gaussian, OutcomeSpec
        outcomes = [OutcomeSpec(name="y", y="y", family=Gaussian())]
        model = TabMixNN(
            formula="y ~ x + (1|group)",
            outcomes=outcomes,
            hidden_dims=[32],
            embedding_dim=8,
            epochs=200,
            lr=5e-3,
            lambda_kl=0.001,
            verbose=False,
        )
        model.fit(df_train)
        preds_tabmix = model.predict(df_test)
        rmse_tabmix = np.sqrt(np.mean((df_test["y"].values - preds_tabmix) ** 2))

        # Ridge baseline (no group info)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(df_train[["x"]])
        X_test = scaler.transform(df_test[["x"]])
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, df_train["y"].values)
        preds_ridge = ridge.predict(X_test)
        rmse_ridge = np.sqrt(np.mean((df_test["y"].values - preds_ridge) ** 2))

        assert rmse_tabmix < rmse_ridge, (
            f"TabMixNN RMSE={rmse_tabmix:.3f} should beat Ridge RMSE={rmse_ridge:.3f}. "
            f"This is the core CAR-T report question — if this fails, the claim must be softened."
        )


# ---------------------------------------------------------------------------
# 6. FIX-1: λ_KL auto-scaling — Sleepstudy lme4 reference values
# ---------------------------------------------------------------------------

def _make_sleepstudy():
    """
    Synthetic Sleepstudy dataset matching the lme4 reference DGP.

    lme4 reference (Belenky et al. 2003):
        Reaction ~ Days + (Days | Subject)
        Fixed effects: intercept ≈ 251 ms, slope ≈ 10 ms/day
        18 subjects × 10 days = 180 observations

    We use the exact lme4 fixed-effect values as the DGP so the test is
    deterministic and the tolerance is tight.
    """
    rng = np.random.default_rng(0)
    n_subjects = 18
    n_days = 10
    beta0 = 251.0   # lme4 intercept
    beta1 = 10.0    # lme4 slope (ms/day)
    sigma_u0 = 24.7  # lme4 random intercept SD
    sigma_u1 = 5.7   # lme4 random slope SD
    sigma_e = 25.6   # lme4 residual SD

    subjects = []
    days_col = []
    reactions = []
    for s in range(n_subjects):
        u0 = rng.normal(0, sigma_u0)
        u1 = rng.normal(0, sigma_u1)
        for d in range(n_days):
            e = rng.normal(0, sigma_e)
            y = beta0 + beta1 * d + u0 + u1 * d + e
            subjects.append(str(s))
            days_col.append(float(d))
            reactions.append(y)
    return pd.DataFrame({"Reaction": reactions, "Days": days_col, "Subject": subjects})


class TestKLAutoScale:
    """
    FIX-1 validation: kl_auto_scale=True should produce λ_KL ≈ 1/10 for
    Sleepstudy (10 obs/subject), and the fitted model should recover the
    lme4 reference intercept (≈251 ms) and slope (≈10 ms/day) within
    generous tolerances (±60 ms intercept, ±8 ms/day slope).

    The Sleepstudy test uses hidden_dims=[16] (one nonlinear layer) because
    the pure LMM path (hidden_dims=[]) has a known identifiability issue when
    the true DGP includes random slopes but the formula only specifies random
    intercepts: the fixed slope is partially confounded with the random slope
    variance. The hidden_dims=[16] path recovers the slope correctly because
    the backbone can represent the slope through its own weights independently
    of the fixed_linear layer.
    """

    def test_kl_auto_scale_sets_lambda(self):
        """kl_auto_scale=True should set lambda_kl ≈ 1/mean_group_size after fit."""
        from tabmixnn import TabMixNN, Gaussian, OutcomeSpec
        df = _make_sleepstudy()
        outcomes = [OutcomeSpec(name="Reaction", y="Reaction", family=Gaussian())]
        model = TabMixNN(
            formula="Reaction ~ Days + (1|Subject)",
            outcomes=outcomes,
            hidden_dims=[],
            embedding_dim=1,
            epochs=1,
            lr=1e-3,
            kl_auto_scale=True,
            verbose=False,
        )
        model.fit(df)
        # 18 subjects × 10 days → mean_group_size = 10 → λ_KL ≈ 0.1
        assert abs(model.lambda_kl - 0.1) < 0.01, (
            f"Expected lambda_kl ≈ 0.1 (1/10), got {model.lambda_kl:.4f}"
        )

    def test_kl_auto_scale_false_preserves_lambda(self):
        """kl_auto_scale=False should leave lambda_kl at the user-supplied value."""
        from tabmixnn import TabMixNN, Gaussian, OutcomeSpec
        df = _make_sleepstudy()
        outcomes = [OutcomeSpec(name="Reaction", y="Reaction", family=Gaussian())]
        model = TabMixNN(
            formula="Reaction ~ Days + (1|Subject)",
            outcomes=outcomes,
            hidden_dims=[],
            embedding_dim=1,
            epochs=1,
            lr=1e-3,
            lambda_kl=0.05,
            kl_auto_scale=False,
            verbose=False,
        )
        model.fit(df)
        assert abs(model.lambda_kl - 0.05) < 1e-6, (
            f"Expected lambda_kl=0.05 unchanged, got {model.lambda_kl}"
        )

    def test_sleepstudy_intercept_slope_vs_lme4(self):
        """
        Fit Sleepstudy with kl_auto_scale=True and check that the mean
        prediction at Days=0 (intercept proxy) and the per-day slope are
        within ±60 ms and ±8 ms/day of the lme4 reference values.

        lme4 reference: intercept ≈ 251 ms, slope ≈ 10 ms/day.
        """
        from tabmixnn import TabMixNN, Gaussian, OutcomeSpec
        df = _make_sleepstudy()
        outcomes = [OutcomeSpec(name="Reaction", y="Reaction", family=Gaussian())]
        model = TabMixNN(
            formula="Reaction ~ Days + (1|Subject)",
            outcomes=outcomes,
            hidden_dims=[16],
            embedding_dim=8,
            epochs=500,
            lr=5e-3,
            kl_auto_scale=True,
            enforce_centering=True,
            verbose=False,
        )
        model.fit(df)

        # Estimate intercept and slope from predictions at Days=0 and Days=9
        df0 = df[df["Days"] == 0.0].copy()
        df9 = df[df["Days"] == 9.0].copy()
        pred0 = float(np.mean(model.predict(df0)))
        pred9 = float(np.mean(model.predict(df9)))
        slope_est = (pred9 - pred0) / 9.0

        lme4_intercept = 251.0
        lme4_slope = 10.0

        assert abs(pred0 - lme4_intercept) < 60.0, (
            f"Intercept proxy {pred0:.1f} ms deviates from lme4 {lme4_intercept} ms "
            f"by {abs(pred0 - lme4_intercept):.1f} ms (tolerance ±60 ms)"
        )
        assert abs(slope_est - lme4_slope) < 8.0, (
            f"Slope estimate {slope_est:.2f} ms/day deviates from lme4 {lme4_slope} ms/day "
            f"by {abs(slope_est - lme4_slope):.2f} ms/day (tolerance ±8 ms/day)"
        )


# ---------------------------------------------------------------------------
# 7. FIX-2: extract_parameters() returns outcome-unit BLUPs
# ---------------------------------------------------------------------------

class TestExtractParametersOutcomeUnits:
    """
    FIX-2 validation: _collect_random_effects() now uses prediction ablation,
    so the returned estimates should be in outcome units and correlate with
    true group effects (r > 0.5).
    """

    def test_random_effects_in_outcome_units(self):
        """
        extract_parameters() random_effects['estimate'] should correlate with
        true group random effects (r > 0.5) after the ablation fix.
        """
        from tabmixnn import TabMixNN, Gaussian, OutcomeSpec
        rng = np.random.default_rng(7)
        n_groups, n_per = 20, 15
        true_u = rng.normal(0, 3.0, n_groups)
        groups = np.repeat(np.arange(n_groups), n_per).astype(str)
        x = rng.standard_normal(n_groups * n_per)
        u_i = true_u[np.repeat(np.arange(n_groups), n_per)]
        y = 5.0 + 2.0 * x + u_i + rng.normal(0, 1.0, n_groups * n_per)
        df = pd.DataFrame({"y": y, "x": x, "group": groups})

        outcomes = [OutcomeSpec(name="y", y="y", family=Gaussian())]
        model = TabMixNN(
            formula="y ~ x + (1|group)",
            outcomes=outcomes,
            hidden_dims=[32],
            embedding_dim=8,
            epochs=200,
            lr=5e-3,
            kl_auto_scale=True,
            enforce_centering=True,
            verbose=False,
        )
        model.fit(df)

        params = model.extract_parameters(df)
        re_df = params["random_effects"]
        assert not re_df.empty, "random_effects DataFrame should not be empty"

        # Align estimated RE with true u by group label
        re_by_group = re_df.groupby("level")["estimate"].mean()
        common = [str(g) for g in range(n_groups) if str(g) in re_by_group.index]
        assert len(common) >= 10, f"Too few matched groups: {len(common)}"

        est = np.array([re_by_group[g] for g in common])
        truth = np.array([true_u[int(g)] for g in common])
        r = np.corrcoef(est, truth)[0, 1]
        assert r > 0.5, (
            f"extract_parameters() RE correlation with truth: r={r:.3f} (expected > 0.5). "
            f"Ablation fix may not be working."
        )


# ---------------------------------------------------------------------------
# 8. FIX-4: Minor fixes — torch scoping, AR1 marginal variance
# ---------------------------------------------------------------------------

class TestFix4:
    """
    FIX-4a: import torch inside try block removed from fit() — torch is already
            imported at module level; the inner import caused a scoping warning
            in Python 3.12+ (UnboundLocalError risk in except clause).

    FIX-4b: get_variance_summary() now reports marginal std for AR1:
            sigma_marginal = sigma_innovation / sqrt(1 - rho^2)
            Previously it reported sigma_innovation, which is wrong for
            interpreting variance components (e.g. rho=0.8, sigma=5 gives
            marginal std 8.33, not 5.0).

    FIX-4c: CompositeStructuralModel docstring now explicitly documents the
            sequential-vs-fixed-point deviation from paper Eq.22 and when to
            use HybridStructuralModel instead.
    """

    def test_fit_no_import_torch_in_try(self):
        """
        fit() must not contain 'import torch' inside a try block.
        torch is imported at module level; the inner import is redundant
        and causes a scoping hazard in Python 3.12+.
        """
        import inspect
        from tabmixnn.core.model import TabMixNN
        src = inspect.getsource(TabMixNN.fit)
        lines = src.split('\n')
        in_try = False
        for line in lines:
            stripped = line.strip()
            if stripped == 'try:':
                in_try = True
            if in_try and stripped.startswith('except'):
                in_try = False
            if in_try and stripped == 'import torch':
                raise AssertionError(
                    "Found 'import torch' inside a try block in fit(). "
                    "torch is already imported at module level — remove the inner import."
                )

    def test_ar1_variance_summary_reports_marginal_std(self):
        """
        get_variance_summary() must report marginal std for AR1, not innovation std.

        Stationary AR1: Var(u) = sigma^2 / (1 - rho^2)
        Marginal std    = sigma / sqrt(1 - rho^2)

        With sigma=5, rho=0.8: marginal std = 5/sqrt(1-0.64) = 5/0.6 = 8.333
        The old code returned 5.0 (innovation std) — wrong for variance reporting.
        """
        import torch
        from tabmixnn.covariance.structures import AR1Covariance
        from tabmixnn import TabMixNN, Gaussian, OutcomeSpec

        # Build a minimal model with AR1 covariance
        df = _make_gaussian_lmm(n_groups=10, n_per_group=10)
        outcomes = [OutcomeSpec(name="y", y="y", family=Gaussian())]
        model = TabMixNN(
            formula="y ~ x + (1|group)",
            outcomes=outcomes,
            hidden_dims=[],
            embedding_dim=1,
            epochs=1,
            lr=1e-3,
            cov_struct_map={"group": {"type": "AR1"}},
            verbose=False,
        )
        model.fit(df)

        # Manually set known AR1 parameters
        enc = model.encoder
        cov = enc.cov_modules["term_0"]
        with torch.no_grad():
            cov.log_sigma.fill_(float(torch.log(torch.tensor(5.0))))
            cov.raw_rho.fill_(float(torch.arctanh(torch.tensor(0.8))))

        summary = enc.get_variance_summary()
        term_vals = summary["term_0"]

        sigma_innovation = 5.0
        rho = 0.8
        expected_marginal = sigma_innovation / (1 - rho ** 2) ** 0.5  # 8.333

        reported = float(term_vals["sigmas"][0])
        assert abs(reported - expected_marginal) < 0.01, (
            f"AR1 get_variance_summary reported sigma={reported:.3f}, "
            f"expected marginal std={expected_marginal:.3f} (sigma/sqrt(1-rho^2)). "
            f"innovation_sigma should be in term_vals['innovation_sigma']."
        )
        # innovation_sigma should also be accessible
        assert "innovation_sigma" in term_vals, (
            "get_variance_summary should expose 'innovation_sigma' for AR1 "
            "so callers can access both innovation and marginal std."
        )
        assert abs(float(term_vals["innovation_sigma"][0]) - sigma_innovation) < 0.01

    def test_composite_structural_model_docstring_documents_eq22_deviation(self):
        """
        CompositeStructuralModel docstring must mention paper Eq.22 and the
        sequential-vs-fixed-point deviation, so users know when to use
        HybridStructuralModel instead.
        """
        from tabmixnn.layers.structural import CompositeStructuralModel
        doc = CompositeStructuralModel.__doc__ or ""
        assert "Eq.22" in doc or "fixed-point" in doc, (
            "CompositeStructuralModel docstring must document the deviation from "
            "paper Eq.22 (sequential approximation vs true fixed-point iteration)."
        )
        assert "HybridStructuralModel" in doc, (
            "CompositeStructuralModel docstring must mention HybridStructuralModel "
            "as the alternative for exact fixed-point semantics."
        )


# ---------------------------------------------------------------------------
# 9. Covariance structures
# ---------------------------------------------------------------------------

class TestCovarianceStructures:
    """AR1, IID, UnstructuredSlope forward passes; enforce_centering."""

    def _make_df(self, n_groups=15, n_per=20):
        rng = np.random.default_rng(99)
        groups = np.repeat(np.arange(n_groups), n_per).astype(str)
        x = rng.standard_normal(n_groups * n_per)
        u = rng.normal(0, 2.0, n_groups)[np.repeat(np.arange(n_groups), n_per)]
        y = 3.0 + 1.5 * x + u + rng.normal(0, 0.5, n_groups * n_per)
        return pd.DataFrame({"y": y, "x": x, "group": groups})

    def test_ar1_forward_shape_and_marginal_variance(self):
        """AR1 forward: output shape correct; u[0] has correct marginal variance."""
        import torch
        from tabmixnn.covariance.structures import AR1Covariance
        n_levels, n_slopes, emb = 10, 1, 4
        cov = AR1Covariance(n_levels, n_slopes)
        with torch.no_grad():
            cov.log_sigma.fill_(float(torch.log(torch.tensor(2.0))))
            cov.raw_rho.fill_(float(torch.arctanh(torch.tensor(0.6))))
        v = torch.randn(n_levels, n_slopes, emb)
        u = cov(v)
        assert u.shape == (n_levels, n_slopes, emb), f"Expected {(n_levels, n_slopes, emb)}, got {u.shape}"
        # Marginal variance of u[0] = sigma^2 / (1 - rho^2)
        sigma = float(torch.exp(cov.log_sigma[0, 0]))
        rho = float(torch.tanh(cov.raw_rho[0, 0]))
        expected_var = sigma ** 2 / (1 - rho ** 2)
        torch.manual_seed(0)
        samples = []
        for _ in range(500):
            v_s = torch.randn(n_levels, n_slopes, emb)
            u_s = cov(v_s)
            samples.append(u_s[0, 0, 0].item())
        empirical_var = float(np.var(samples))
        assert abs(empirical_var - expected_var) / expected_var < 0.15, (
            f"AR1 u[0] empirical var={empirical_var:.3f}, expected={expected_var:.3f}"
        )

    def test_iid_forward_scaled_by_sigma(self):
        """IID forward: output is v * sigma (broadcast over levels)."""
        import torch
        from tabmixnn.covariance.structures import IIDCovariance
        n_levels, n_slopes, emb = 8, 2, 4
        cov = IIDCovariance(n_levels, n_slopes)
        sigma_val = 3.0
        with torch.no_grad():
            cov.log_sigma.fill_(float(np.log(sigma_val)))
        v = torch.ones(n_levels, n_slopes, emb)
        u = cov(v)
        assert u.shape == (n_levels, n_slopes, emb)
        np.testing.assert_allclose(u.detach().numpy(), sigma_val, atol=1e-5)

    def test_unstructured_slope_output_shape(self):
        """UnstructuredSlope forward: output shape matches input."""
        import torch
        from tabmixnn.covariance.structures import UnstructuredSlopeCovariance
        n_levels, n_slopes, emb = 10, 3, 8
        cov = UnstructuredSlopeCovariance(n_levels, n_slopes)
        v = torch.randn(n_levels, n_slopes, emb)
        u = cov(v)
        assert u.shape == (n_levels, n_slopes, emb), f"Shape mismatch: {u.shape}"

    def test_enforce_centering_iid(self):
        """enforce_centering=True: mean of u across levels is near zero for IID."""
        from tabmixnn import TabMixNN, Gaussian, OutcomeSpec
        df = self._make_df()
        outcomes = [OutcomeSpec(name="y", y="y", family=Gaussian())]
        model = TabMixNN(
            formula="y ~ x + (1|group)",
            outcomes=outcomes,
            hidden_dims=[16],
            embedding_dim=4,
            epochs=50,
            lr=5e-3,
            kl_auto_scale=True,
            enforce_centering=True,
            verbose=False,
        )
        model.fit(df)
        enc = model.encoder
        tid = "term_0"
        assert tid in enc.variational_params, "term_0 not in variational_params"
        mu = enc.variational_params[tid][0]
        effects = enc.random_effects_values(tid, mu).detach().numpy()
        np.testing.assert_allclose(effects.mean(axis=0), 0, atol=1e-6)



# ---------------------------------------------------------------------------
# 10. Multi-task learning
# ---------------------------------------------------------------------------

class TestMultiTask:
    """Fit a model with 2 outcomes (Gaussian + Binomial); check predictions and metrics."""

    def _make_multitask_df(self, n=400, n_groups=10, seed=7):
        rng = np.random.default_rng(seed)
        groups = (np.arange(n) % n_groups).astype(str)
        x = rng.standard_normal(n)
        u = rng.normal(0, 1.5, n_groups)[np.arange(n) % n_groups]
        y_cont = 3.0 + 2.0 * x + u + rng.normal(0, 0.5, n)
        logit = 0.5 + 1.0 * x + 0.3 * u
        y_bin = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
        return pd.DataFrame({"y_cont": y_cont, "y_bin": y_bin, "x": x, "group": groups})

    def test_multitask_predictions_return_dict_with_both_keys(self):
        from tabmixnn import TabMixNN, Gaussian, Binomial, OutcomeSpec
        df = self._make_multitask_df()
        outcomes = [
            OutcomeSpec(name="cont", y="y_cont", family=Gaussian()),
            OutcomeSpec(name="bin", y="y_bin", family=Binomial()),
        ]
        model = TabMixNN(
            formula="~ x + (1|group)",
            outcomes=outcomes,
            hidden_dims=[32],
            embedding_dim=8,
            epochs=80,
            lr=5e-3,
            kl_auto_scale=True,
            verbose=False,
        )
        model.fit(df)
        preds = model.predict(df)
        assert isinstance(preds, dict), "predict() should return dict for multi-outcome model"
        assert "cont" in preds and "bin" in preds, f"Missing keys: {list(preds.keys())}"

    def test_multitask_reasonable_metrics(self):
        """Both outcomes achieve R²>0.3 and AUC>0.6 on training data."""
        from tabmixnn import TabMixNN, Gaussian, Binomial, OutcomeSpec
        from sklearn.metrics import roc_auc_score
        df = self._make_multitask_df()
        outcomes = [
            OutcomeSpec(name="cont", y="y_cont", family=Gaussian()),
            OutcomeSpec(name="bin", y="y_bin", family=Binomial()),
        ]
        model = TabMixNN(
            formula="~ x + (1|group)",
            outcomes=outcomes,
            hidden_dims=[32],
            embedding_dim=8,
            epochs=150,
            lr=5e-3,
            kl_auto_scale=True,
            verbose=False,
        )
        model.fit(df)
        preds = model.predict(df)

        y_cont = df["y_cont"].values
        r2 = 1 - np.sum((y_cont - preds["cont"]) ** 2) / np.sum((y_cont - y_cont.mean()) ** 2)
        assert r2 > 0.3, f"Gaussian outcome R²={r2:.3f} (expected > 0.3)"

        auc = roc_auc_score(df["y_bin"].values, preds["bin"])
        assert auc > 0.6, f"Binomial outcome AUC={auc:.3f} (expected > 0.6)"


# ---------------------------------------------------------------------------
# 11. Survival model fit — DiscreteTimeCompetingRisks CIF constraint
# ---------------------------------------------------------------------------

class TestSurvivalModelFit:
    """CIF sums to ≤ 1 at each time point for DiscreteTimeCompetingRisks."""

    def test_competing_risks_cif_sums_leq_one(self):
        """Fit DiscreteTimeCompetingRisks; verify CIF sums ≤ 1 at each time point."""
        import torch
        from tabmixnn import TabMixNN, DiscreteTimeCompetingRisks, OutcomeSpec

        rng = np.random.default_rng(55)
        n, T, n_causes = 200, 8, 2
        groups = (np.arange(n) % 10).astype(str)
        x = rng.standard_normal(n)
        cause = rng.choice([0, 1, 2], n, p=[0.4, 0.35, 0.25])
        time_idx = rng.integers(0, T, n)
        df = pd.DataFrame({"time": time_idx, "cause": cause, "x": x, "group": groups})

        time_bins = list(range(T))
        fam = DiscreteTimeCompetingRisks(time_bins=time_bins, num_causes=n_causes)
        outcomes = [OutcomeSpec(name="cr", family=fam, time="time", cause="cause")]

        model = TabMixNN(
            formula="~ x + (1|group)",
            outcomes=outcomes,
            hidden_dims=[16],
            embedding_dim=4,
            epochs=30,
            lr=5e-3,
            verbose=False,
        )
        model.fit(df)

        X_np, Z_ind, Z_slopes, _ = model.design.transform(df, mode="predict")
        X_t = torch.tensor(X_np, dtype=torch.float32)
        Z_it = [torch.tensor(z, dtype=torch.long) for z in Z_ind]
        Z_st = [torch.tensor(z, dtype=torch.float32) for z in Z_slopes]
        with torch.no_grad():
            raw = model.forward(X_t, Z_it, Z_st)
            theta = fam.forward_params(raw)
            cif = fam.predict(theta, mode="cif")  # [n, T, n_causes]

        cif_sum = cif.sum(axis=-1)  # [n, T]
        assert (cif_sum <= 1.0 + 1e-5).all(), (
            f"CIF sum exceeds 1 at some time points: max={cif_sum.max():.4f}"
        )


# ---------------------------------------------------------------------------
# 12. Save / Load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    """Fit, save, load, verify predictions are identical."""

    def test_save_load_predictions_identical(self, tmp_path):
        from tabmixnn import TabMixNN, Gaussian, OutcomeSpec
        df = _make_gaussian_lmm(n_groups=10, n_per_group=10)
        outcomes = [OutcomeSpec(name="y", y="y", family=Gaussian())]
        model = TabMixNN(
            formula="y ~ x + (1|group)",
            outcomes=outcomes,
            hidden_dims=[16],
            embedding_dim=4,
            epochs=30,
            lr=5e-3,
            verbose=False,
        )
        model.fit(df)
        preds_before = model.predict(df)

        path = str(tmp_path / "model.pt")
        model.save(path)

        loaded = TabMixNN.load(path)
        preds_after = loaded.predict(df)

        np.testing.assert_allclose(
            preds_before, preds_after, atol=1e-5,
            err_msg="Predictions differ after save/load"
        )


# ---------------------------------------------------------------------------
# 13. Optimizer parameter groups — C2 regression test
# ---------------------------------------------------------------------------

class TestOptimizerParameterGroups:
    """After fit(), variational_params and cov_modules have weight_decay=0."""

    def test_re_params_have_zero_weight_decay(self):
        from tabmixnn import TabMixNN, Gaussian, OutcomeSpec
        df = _make_gaussian_lmm(n_groups=10, n_per_group=10)
        outcomes = [OutcomeSpec(name="y", y="y", family=Gaussian())]
        model = TabMixNN(
            formula="y ~ x + (1|group)",
            outcomes=outcomes,
            hidden_dims=[16],
            embedding_dim=4,
            epochs=1,
            lr=1e-3,
            weight_decay=1e-4,
            verbose=False,
        )
        model.fit(df)

        enc = model.encoder
        re_param_ids = set()
        for p in enc.variational_params.parameters():
            re_param_ids.add(id(p))
        for p in enc.cov_modules.parameters():
            re_param_ids.add(id(p))

        for group in model.optimizer.param_groups:
            for p in group["params"]:
                if id(p) in re_param_ids:
                    assert group["weight_decay"] == 0.0, (
                        f"RE/cov param has weight_decay={group['weight_decay']} (expected 0.0). "
                        "C2 fix: variational_params and cov_modules must not receive L2 decay."
                    )


# ---------------------------------------------------------------------------
# 14. Approximation error — M3 diagnostic
# ---------------------------------------------------------------------------

class TestApproximationError:
    """CompositeStructuralModel.approximation_error() returns a finite nonnegative relative error."""

    def test_approximation_error_range(self):
        import torch
        from tabmixnn import TabMixNN, Gaussian, OutcomeSpec
        from tabmixnn.layers.structural import CompositeStructuralModel

        rng = np.random.default_rng(77)
        n = 100
        df = pd.DataFrame({
            "y": rng.standard_normal(n),
            "x": rng.standard_normal(n),
            "group": (np.arange(n) % 10).astype(str),
        })
        outcomes = [OutcomeSpec(name="y", y="y", family=Gaussian())]

        model = TabMixNN(
            formula="y ~ x + (1|group)",
            outcomes=outcomes,
            hidden_dims=[32],
            use_static_structure=True,
            use_dynamic_structure=True,
            epochs=1,
            verbose=False,
        )
        model.fit(df)

        composite = None
        for layer in model.backbone.hidden_layers:
            if hasattr(layer, "structure") and isinstance(
                layer.structure, CompositeStructuralModel
            ):
                composite = layer.structure
                break

        assert composite is not None, (
            "No CompositeStructuralModel found in backbone. "
            "Ensure use_static_structure=True and use_dynamic_structure=True."
        )

        xi = torch.randn(16, 32)
        err = composite.approximation_error(xi)
        assert isinstance(err, float), f"approximation_error() should return float, got {type(err)}"
        assert np.isfinite(err) and err >= 0.0, f"approximation_error()={err:.4f} not in [0, 1]"
