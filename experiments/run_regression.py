"""
Regression benchmark — diagnostic scripts, not a reproduction of Tables 2-3
from arXiv:2512.23787.  Original per-fold outputs and configurations for those
tables are unavailable; no claim is made that these results reproduce them.

Datasets: Sleepstudy, ChickWeight, Machines, Oxboys, Loblolly
Metrics:  RMSE, R²
Baselines: XGBoost (no random effects), statsmodels MixedLM (LMM reference)
Split seed: 20261015 (experiments/splits.py SPLIT_SEED)
Results saved to: experiments/results/regression_results.csv
              and experiments/results/regression_raw.json (per-fold)
"""

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
torch.backends.mps.is_available = lambda: False  # prevent MPS LayerNorm dispatch on Apple Silicon

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

from tabmixnn import TabMixNNRegressor
from experiments.data_utils import load
from experiments.splits import obs_kfold, group_holdout, manifest_meta

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_SPLITS = 5
EPOCHS = 200
MODEL_SEED = 42  # model weight init seed; split seed is in splits.py


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _lmm_available():
    try:
        import statsmodels.api as sm  # noqa: F401
        return True
    except ImportError:
        return False


def cv_tabmixnn(df, formula, n_splits=N_SPLITS):
    """Observation K-fold CV for TabMixNN.  Preprocessing fitted on train only."""
    rmses, r2s = [], []
    for train_idx, test_idx in obs_kfold(df, n_splits):
        train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
        model = TabMixNNRegressor(
            formula=formula,
            hidden_dims=[32],
            embedding_dim=8,
            epochs=EPOCHS,
            batch_size=32,
            lr=0.005,
            kl_auto_scale=True,
            verbose=False,
            random_state=MODEL_SEED,
        )
        model.fit(train)
        pred = model.predict(test)
        outcome = formula.split("~")[0].strip()
        y = test[outcome].values
        rmses.append(rmse(y, pred))
        r2s.append(r2_score(y, pred))
    return float(np.mean(rmses)), float(np.mean(r2s)), rmses


def cv_tabmixnn_group_holdout(df, formula, group_col):
    """Hold out 25% of groups entirely; train on remaining groups.

    This is the correct evaluation for generalisation to unseen groups.
    Observation-level CV leaks group identity into training and over-estimates
    performance for new groups.
    """
    train_mask, test_mask = group_holdout(df, group_col)
    train = df[train_mask].copy()
    test = df[test_mask].copy()

    if len(train) == 0 or len(test) == 0:
        return float("nan"), float("nan"), []

    model = TabMixNNRegressor(
        formula=formula,
        hidden_dims=[32],
        embedding_dim=8,
        epochs=EPOCHS,
        batch_size=32,
        lr=0.005,
        kl_auto_scale=True,
        verbose=False,
        random_state=MODEL_SEED,
    )
    model.fit(train)
    pred = model.predict(test)
    outcome = formula.split("~")[0].strip()
    y = test[outcome].values
    r = rmse(y, pred)
    return r, float(r2_score(y, pred)), [r]


def cv_xgboost(df, outcome, features, n_splits=N_SPLITS):
    """Observation K-fold CV for XGBoost.  Same splits as cv_tabmixnn."""
    df = df.copy()
    for col in features:
        try:
            is_numeric = np.issubdtype(df[col].dtype, np.number)
        except TypeError:
            is_numeric = False
        if not is_numeric:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    X, y = df[features].values, df[outcome].values
    rmses, r2s = [], []
    for train_idx, test_idx in obs_kfold(df, n_splits):
        # Train-only preprocessing: LabelEncoder already applied globally above
        # (categories are fixed by the full dataset; numeric scaling is not applied).
        xgb = XGBRegressor(n_estimators=200, learning_rate=0.05,
                           random_state=MODEL_SEED, verbosity=0)
        xgb.fit(X[train_idx], y[train_idx])
        pred = xgb.predict(X[test_idx])
        rmses.append(rmse(y[test_idx], pred))
        r2s.append(r2_score(y[test_idx], pred))
    return float(np.mean(rmses)), float(np.mean(r2s)), rmses


def cv_lmm(df, outcome, fixed_features, group_col, n_splits=N_SPLITS):
    """Observation K-fold CV for statsmodels MixedLM (REML=False, ML).

    Returns (mean_rmse, mean_r2, per_fold_rmses) or (nan, nan, []) if
    statsmodels is not installed.
    """
    if not _lmm_available():
        return float("nan"), float("nan"), []
    import statsmodels.api as sm

    df = df.copy()
    for col in fixed_features:
        try:
            is_numeric = np.issubdtype(df[col].dtype, np.number)
        except TypeError:
            is_numeric = False
        if not is_numeric:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    rmses, r2s = [], []
    for train_idx, test_idx in obs_kfold(df, n_splits):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        X_tr = sm.add_constant(train[fixed_features].astype(float))
        X_te = sm.add_constant(test[fixed_features].astype(float),
                               has_constant="add")
        try:
            lmm = sm.MixedLM(
                train[outcome].astype(float),
                X_tr,
                groups=train[group_col],
            ).fit(reml=False, method="lbfgs", disp=False)
            if not lmm.converged:
                rmses.append(float("nan"))
                r2s.append(float("nan"))
                continue
            # Fixed-effects prediction only (zero random effects for test groups
            # that may be unseen; this is the conservative baseline).
            pred = X_te.values @ np.asarray(lmm.fe_params)
            y = test[outcome].values
            rmses.append(rmse(y, pred))
            r2s.append(r2_score(y, pred))
        except Exception:
            rmses.append(float("nan"))
            r2s.append(float("nan"))
    return float(np.nanmean(rmses)), float(np.nanmean(r2s)), rmses


DATASETS = [
    {
        "name": "Sleepstudy",
        "loader": "sleepstudy",
        "formula": "Reaction ~ Days + (1 + Days | Subject)",
        "outcome": "Reaction",
        "group": "Subject",
        "xgb_features": ["Days", "Subject"],
        "lmm_fixed": ["Days"],
    },
    {
        "name": "ChickWeight",
        "loader": "chickweight",
        "formula": "weight ~ Time + Diet + (1 | Chick)",
        "outcome": "weight",
        "group": "Chick",
        "xgb_features": ["Time", "Diet", "Chick"],
        "lmm_fixed": ["Time", "Diet"],
    },
    {
        "name": "Machines",
        "loader": "machines",
        "formula": "score ~ Machine + (1 | Worker)",
        "outcome": "score",
        "group": "Worker",
        "xgb_features": ["Machine", "Worker"],
        "lmm_fixed": ["Machine"],
    },
    {
        "name": "Oxboys",
        "loader": "oxboys",
        "formula": "height ~ age + (1 + age | Subject)",
        "outcome": "height",
        "group": "Subject",
        "xgb_features": ["age", "Subject", "Occasion"],
        "lmm_fixed": ["age"],
    },
    {
        "name": "Loblolly",
        "loader": "loblolly",
        "formula": "height ~ age + (1 | Seed)",
        "outcome": "height",
        "group": "Seed",
        "xgb_features": ["age", "Seed"],
        "lmm_fixed": ["age"],
    },
]


def main():
    rows = []
    raw_records = []
    lmm_present = _lmm_available()
    if not lmm_present:
        print("  [statsmodels not installed — LMM benchmark skipped]")

    for cfg in DATASETS:
        print(f"\n{'='*50}")
        print(f"Dataset: {cfg['name']}")
        df = load(cfg["loader"])
        print(f"  n={len(df)}, columns={list(df.columns)}")

        print(f"  TabMixNN ({N_SPLITS}-fold obs CV)...")
        tab_rmse, tab_r2, tab_folds = cv_tabmixnn(df, cfg["formula"])
        print(f"    RMSE={tab_rmse:.4f}  R²={tab_r2:.4f}")

        print(f"  TabMixNN (group-holdout, 25% groups held out)...")
        tab_gh_rmse, tab_gh_r2, tab_gh_folds = cv_tabmixnn_group_holdout(
            df, cfg["formula"], cfg["group"]
        )
        if not np.isnan(tab_gh_rmse):
            print(f"    RMSE={tab_gh_rmse:.4f}  R²={tab_gh_r2:.4f}")
        else:
            print("    skipped (insufficient groups)")

        print(f"  XGBoost ({N_SPLITS}-fold obs CV)...")
        xgb_rmse, xgb_r2, xgb_folds = cv_xgboost(
            df, cfg["outcome"], cfg["xgb_features"]
        )
        print(f"    RMSE={xgb_rmse:.4f}  R²={xgb_r2:.4f}")

        lmm_rmse, lmm_r2, lmm_folds = cv_lmm(
            df, cfg["outcome"], cfg["lmm_fixed"], cfg["group"]
        )
        if lmm_present:
            print(f"  LMM ({N_SPLITS}-fold obs CV, fixed-effects prediction)...")
            if not np.isnan(lmm_rmse):
                print(f"    RMSE={lmm_rmse:.4f}  R²={lmm_r2:.4f}")
            else:
                print("    convergence failed on all folds")

        rows.append({
            "dataset":               cfg["name"],
            "tabmixnn_obs_rmse":     round(tab_rmse, 4),
            "tabmixnn_obs_r2":       round(tab_r2, 4),
            "tabmixnn_gh_rmse":      round(tab_gh_rmse, 4) if not np.isnan(tab_gh_rmse) else None,
            "tabmixnn_gh_r2":        round(tab_gh_r2, 4) if not np.isnan(tab_gh_r2) else None,
            "xgboost_rmse":          round(xgb_rmse, 4),
            "xgboost_r2":            round(xgb_r2, 4),
            "lmm_fixed_rmse":        round(lmm_rmse, 4) if not np.isnan(lmm_rmse) else None,
            "lmm_fixed_r2":          round(lmm_r2, 4) if not np.isnan(lmm_r2) else None,
            "tabmixnn_wins_obs_rmse": tab_rmse < xgb_rmse,
            "tabmixnn_wins_gh_rmse":  (tab_gh_rmse < xgb_rmse) if not np.isnan(tab_gh_rmse) else None,
        })
        raw_records.append({
            "dataset":        cfg["name"],
            "tabmixnn_obs":   [round(v, 6) for v in tab_folds],
            "tabmixnn_gh":    [round(v, 6) for v in tab_gh_folds],
            "xgboost_obs":    [round(v, 6) for v in xgb_folds],
            "lmm_fixed_obs":  [round(v, 6) if not np.isnan(v) else None for v in lmm_folds],
        })

    results = pd.DataFrame(rows)
    csv_out = RESULTS_DIR / "regression_results.csv"
    results.to_csv(csv_out, index=False)

    raw_out = RESULTS_DIR / "regression_raw.json"
    raw_out.write_text(json.dumps({
        "meta": manifest_meta(),
        "note": (
            "Diagnostic scripts only. Not a reproduction of Tables 2-3 from "
            "arXiv:2512.23787. Original per-fold outputs unavailable."
        ),
        "results": raw_records,
    }, indent=2) + "\n")

    print(f"\n{'='*50}")
    print("REGRESSION SUMMARY")
    print(results.to_string(index=False))
    wins_obs = results["tabmixnn_wins_obs_rmse"].sum()
    wins_gh = results["tabmixnn_wins_gh_rmse"].sum()
    print(f"\nTabMixNN wins (obs CV RMSE):        {wins_obs}/{len(results)} datasets")
    print(f"TabMixNN wins (group-holdout RMSE): {wins_gh}/{len(results)} datasets")
    print(f"\nResults saved to: {csv_out}")
    print(f"Per-fold raw:     {raw_out}")
    return results


if __name__ == "__main__":
    main()
