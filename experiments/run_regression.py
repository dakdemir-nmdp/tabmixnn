"""
Regression benchmark — reproduces Tables 2-3 from arXiv:2512.23787.

Datasets: Sleepstudy, ChickWeight, Machines, Oxboys, Loblolly
Metrics:  RMSE, R²
Baseline: XGBoost (no random effects)
Results saved to: experiments/results/regression_results.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
torch.backends.mps.is_available = lambda: False  # prevent MPS LayerNorm dispatch on Apple Silicon

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")

from tabmixnn import TabMixNNRegressor
from experiments.data_utils import load

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_SPLITS = 5
EPOCHS = 200
SEED = 42


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def cv_tabmixnn(df, formula, group_col, n_splits=N_SPLITS):
    """K-fold CV for TabMixNN; returns mean RMSE and R²."""
    groups = df[group_col].values
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rmses, r2s = [], []
    for train_idx, test_idx in kf.split(df):
        train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
        model = TabMixNNRegressor(
            formula=formula,
            hidden_dims=[32],
            embedding_dim=8,
            epochs=EPOCHS,
            batch_size=32,
            learning_rate=0.005,
            kl_auto_scale=True,
            verbose=False,
        )
        model.fit(train)
        pred = model.predict(test)
        outcome = formula.split("~")[0].strip()
        y = test[outcome].values
        rmses.append(rmse(y, pred))
        r2s.append(r2_score(y, pred))
    return float(np.mean(rmses)), float(np.mean(r2s))


def cv_xgboost(df, outcome, features, n_splits=N_SPLITS):
    """K-fold CV for XGBoost baseline; returns mean RMSE and R²."""
    df = df.copy()
    for col in features:
        try:
            is_numeric = np.issubdtype(df[col].dtype, np.number)
        except TypeError:
            is_numeric = False
        if not is_numeric:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    X, y = df[features].values, df[outcome].values
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rmses, r2s = [], []
    for train_idx, test_idx in kf.split(X):
        xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=SEED, verbosity=0)
        xgb.fit(X[train_idx], y[train_idx])
        pred = xgb.predict(X[test_idx])
        rmses.append(rmse(y[test_idx], pred))
        r2s.append(r2_score(y[test_idx], pred))
    return float(np.mean(rmses)), float(np.mean(r2s))


DATASETS = [
    {
        "name": "Sleepstudy",
        "loader": "sleepstudy",
        "formula": "Reaction ~ Days + (1 + Days | Subject)",
        "outcome": "Reaction",
        "group": "Subject",
        "xgb_features": ["Days", "Subject"],
    },
    {
        "name": "ChickWeight",
        "loader": "chickweight",
        "formula": "weight ~ Time + Diet + (1 | Chick)",
        "outcome": "weight",
        "group": "Chick",
        "xgb_features": ["Time", "Diet", "Chick"],
    },
    {
        "name": "Machines",
        "loader": "machines",
        "formula": "score ~ Machine + (1 | Worker)",
        "outcome": "score",
        "group": "Worker",
        "xgb_features": ["Machine", "Worker"],
    },
    {
        "name": "Oxboys",
        "loader": "oxboys",
        "formula": "height ~ age + (1 + age | Subject)",
        "outcome": "height",
        "group": "Subject",
        "xgb_features": ["age", "Subject", "Occasion"],
    },
    {
        "name": "Loblolly",
        "loader": "loblolly",
        "formula": "height ~ age + (1 | Seed)",
        "outcome": "height",
        "group": "Seed",
        "xgb_features": ["age", "Seed"],
    },
]


def main():
    rows = []
    for cfg in DATASETS:
        print(f"\n{'='*50}")
        print(f"Dataset: {cfg['name']}")
        df = load(cfg["loader"])
        print(f"  n={len(df)}, columns={list(df.columns)}")

        print(f"  TabMixNN ({N_SPLITS}-fold CV)...")
        tab_rmse, tab_r2 = cv_tabmixnn(df, cfg["formula"], cfg["group"])
        print(f"    RMSE={tab_rmse:.4f}  R²={tab_r2:.4f}")

        print(f"  XGBoost ({N_SPLITS}-fold CV)...")
        xgb_rmse, xgb_r2 = cv_xgboost(df, cfg["outcome"], cfg["xgb_features"])
        print(f"    RMSE={xgb_rmse:.4f}  R²={xgb_r2:.4f}")

        rows.append({
            "dataset": cfg["name"],
            "tabmixnn_rmse": round(tab_rmse, 4),
            "tabmixnn_r2":   round(tab_r2, 4),
            "xgboost_rmse":  round(xgb_rmse, 4),
            "xgboost_r2":    round(xgb_r2, 4),
            "tabmixnn_wins_rmse": tab_rmse < xgb_rmse,
        })

    results = pd.DataFrame(rows)
    out = RESULTS_DIR / "regression_results.csv"
    results.to_csv(out, index=False)

    print(f"\n{'='*50}")
    print("REGRESSION SUMMARY")
    print(results.to_string(index=False))
    wins = results["tabmixnn_wins_rmse"].sum()
    print(f"\nTabMixNN wins (RMSE): {wins}/{len(results)} datasets")
    print(f"Results saved to: {out}")
    return results


if __name__ == "__main__":
    main()
