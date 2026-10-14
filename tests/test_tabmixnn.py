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
        # The design uses encoder bias for intercept (not an explicit column).
        # So 'y ~ x' produces X with 1 column (just x), not 2.
        # This is a deliberate design choice: intercept_included=False always.
        assert X.shape == (len(df), 1), f"Expected (n, 1) for x-only design, got {X.shape}"
        assert len(Z_ind) == 0
        # Document the known limitation: no way to get explicit intercept column
        # via formula — it always comes from encoder bias.
        assert design.intercept_included is False, (
            "intercept_included is hardcoded False — intercept comes from encoder bias. "
            "This means fixed_effects in extract_parameters() will never show an intercept "
            "in outcome units. Known interpretability gap."
        )

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
