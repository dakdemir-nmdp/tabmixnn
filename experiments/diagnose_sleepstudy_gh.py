"""Diagnose Sleepstudy group-holdout collapse.

Reruns the group-holdout split at the validated protocol (2000 epochs, full
batch, lr=0.01, UN covariance, no kl_auto_scale) and compares to XGBoost
with and without the group column.

Run from repo root:
    .venv/bin/python experiments/diagnose_sleepstudy_gh.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
torch.backends.mps.is_available = lambda: False

import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from tabmixnn import TabMixNNRegressor
from experiments.data_utils import load
from experiments.splits import group_holdout

SEEDS = [42, 123, 456]


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def main():
    df = load("sleepstudy")
    train_mask, test_mask = group_holdout(df, "Subject")
    train, test = df[train_mask].copy(), df[test_mask].copy()
    print(f"Sleepstudy group-holdout: n_train={len(train)}, n_test={len(test)}")
    print(f"  Train groups: {sorted(train.Subject.unique())}")
    print(f"  Test groups:  {sorted(test.Subject.unique())}")

    # --- XGBoost with group column (obs-CV baseline, unfair for group-holdout) ---
    le = LabelEncoder().fit(df["Subject"].astype(str))
    train_xgb = train.copy()
    test_xgb = test.copy()
    train_xgb["Subject_enc"] = le.transform(train["Subject"].astype(str))
    test_xgb["Subject_enc"] = le.transform(test["Subject"].astype(str))

    xgb_group = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbosity=0)
    xgb_group.fit(train_xgb[["Days", "Subject_enc"]], train["Reaction"])
    pred_xgb_group = xgb_group.predict(test_xgb[["Days", "Subject_enc"]])
    print(f"\nXGBoost (with Subject):  RMSE={rmse(test.Reaction, pred_xgb_group):.4f}"
          f"  R²={r2_score(test.Reaction, pred_xgb_group):.4f}")

    # --- XGBoost without group column (fair comparison for unseen groups) ---
    xgb_nogroup = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbosity=0)
    xgb_nogroup.fit(train[["Days"]], train["Reaction"])
    pred_xgb_ng = xgb_nogroup.predict(test[["Days"]])
    print(f"XGBoost (no Subject):    RMSE={rmse(test.Reaction, pred_xgb_ng):.4f}"
          f"  R²={r2_score(test.Reaction, pred_xgb_ng):.4f}")

    # --- TabMixNN at validated protocol ---
    print(f"\nTabMixNN validated protocol (2000 epochs, full batch, lr=0.01, UN, no kl_auto_scale):")
    for seed in SEEDS:
        model = TabMixNNRegressor(
            formula="Reaction ~ Days + (1 + Days | Subject)",
            hidden_dims=[32],
            embedding_dim=8,
            epochs=2000,
            batch_size=len(train),
            lr=0.01,
            kl_auto_scale=False,
            verbose=False,
            random_state=seed,
            cov_struct_map={"Subject": {"type": "UN"}},
        )
        model.fit(train)
        pred = model.predict(test)
        print(f"  seed={seed}: RMSE={rmse(test.Reaction, pred):.4f}"
              f"  R²={r2_score(test.Reaction, pred):.4f}")

    # --- TabMixNN at benchmark protocol (200 epochs, kl_auto_scale) for comparison ---
    print(f"\nTabMixNN benchmark protocol (200 epochs, batch=32, lr=0.005, kl_auto_scale):")
    for seed in SEEDS:
        model = TabMixNNRegressor(
            formula="Reaction ~ Days + (1 + Days | Subject)",
            hidden_dims=[32],
            embedding_dim=8,
            epochs=200,
            batch_size=32,
            lr=0.005,
            kl_auto_scale=True,
            verbose=False,
            random_state=seed,
        )
        model.fit(train)
        pred = model.predict(test)
        print(f"  seed={seed}: RMSE={rmse(test.Reaction, pred):.4f}"
              f"  R²={r2_score(test.Reaction, pred):.4f}")


if __name__ == "__main__":
    main()
