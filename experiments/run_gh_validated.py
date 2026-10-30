"""Group-holdout benchmark at the validated protocol.

Protocol: 2000 epochs, full batch, lr=0.01, UN covariance, no kl_auto_scale.
Three seeds per dataset.  XGBoost-nogroup is the fair comparison baseline.

Run from repo root:
    .venv/bin/python experiments/run_gh_validated.py
"""
import json
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
from experiments.splits import group_holdout, manifest_meta

SEEDS = [42, 123, 456]
EPOCHS = 2000
LR = 0.01

DATASETS = [
    {
        "name": "Sleepstudy",
        "loader": "sleepstudy",
        "formula": "Reaction ~ Days + (1 + Days | Subject)",
        "outcome": "Reaction",
        "group": "Subject",
        "xgb_nogroup_features": ["Days"],
        "cov_struct": {"Subject": {"type": "UN"}},
    },
    {
        "name": "ChickWeight",
        "loader": "chickweight",
        "formula": "weight ~ Time + Diet + (1 | Chick)",
        "outcome": "weight",
        "group": "Chick",
        "xgb_nogroup_features": ["Time", "Diet"],
        "cov_struct": {"Chick": {"type": "UN"}},
    },
    {
        "name": "Machines",
        "loader": "machines",
        "formula": "score ~ Machine + (1 | Worker)",
        "outcome": "score",
        "group": "Worker",
        "xgb_nogroup_features": ["Machine"],
        "cov_struct": {"Worker": {"type": "UN"}},
    },
    {
        "name": "Oxboys",
        "loader": "oxboys",
        "formula": "height ~ age + (1 + age | Subject)",
        "outcome": "height",
        "group": "Subject",
        "xgb_nogroup_features": ["age", "Occasion"],
        "cov_struct": {"Subject": {"type": "UN"}},
    },
    {
        "name": "Loblolly",
        "loader": "loblolly",
        "formula": "height ~ age + (1 | Seed)",
        "outcome": "height",
        "group": "Seed",
        "xgb_nogroup_features": ["age"],
        "cov_struct": {"Seed": {"type": "UN"}},
    },
]

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def xgb_nogroup(train, test, outcome, features):
    df_tr = train.copy()
    df_te = test.copy()
    for col in features:
        try:
            is_num = np.issubdtype(df_tr[col].dtype, np.number)
        except TypeError:
            is_num = False
        if not is_num:
            le = LabelEncoder().fit(df_tr[col].astype(str))
            df_tr[col] = le.transform(df_tr[col].astype(str))
            df_te[col] = le.transform(df_te[col].astype(str))
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbosity=0)
    xgb.fit(df_tr[features], train[outcome])
    pred = xgb.predict(df_te[features])
    y = test[outcome].values
    return rmse(y, pred), float(r2_score(y, pred))


def main():
    rows = []
    for cfg in DATASETS:
        print(f"\n{'='*52}\n{cfg['name']}")
        df = load(cfg["loader"])
        train_mask, test_mask = group_holdout(df, cfg["group"])
        train, test = df[train_mask].copy(), df[test_mask].copy()
        print(f"  n_train={len(train)}, n_test={len(test)}, "
              f"held-out groups={test[cfg['group']].nunique()}")

        xgb_r, xgb_r2 = xgb_nogroup(train, test, cfg["outcome"], cfg["xgb_nogroup_features"])
        print(f"  XGBoost-nogroup:  RMSE={xgb_r:.4f}  R²={xgb_r2:.4f}")

        seed_rmses, seed_r2s = [], []
        for seed in SEEDS:
            model = TabMixNNRegressor(
                formula=cfg["formula"],
                hidden_dims=[32],
                embedding_dim=8,
                epochs=EPOCHS,
                batch_size=len(train),
                lr=LR,
                kl_auto_scale=False,
                verbose=False,
                random_state=seed,
                cov_struct_map=cfg["cov_struct"],
            )
            model.fit(train)
            pred = model.predict(test)
            y = test[cfg["outcome"]].values
            r = rmse(y, pred)
            r2 = float(r2_score(y, pred))
            seed_rmses.append(r)
            seed_r2s.append(r2)
            print(f"  seed={seed}: RMSE={r:.4f}  R²={r2:.4f}")

        mean_r = float(np.mean(seed_rmses))
        mean_r2 = float(np.mean(seed_r2s))
        print(f"  mean:   RMSE={mean_r:.4f}  R²={mean_r2:.4f}  "
              f"({'WIN' if mean_r < xgb_r else 'LOSS'} vs XGB-nogroup)")
        rows.append({
            "dataset": cfg["name"],
            "xgb_nogroup_rmse": round(xgb_r, 4),
            "xgb_nogroup_r2": round(xgb_r2, 4),
            "tabmixnn_mean_rmse": round(mean_r, 4),
            "tabmixnn_mean_r2": round(mean_r2, 4),
            "tabmixnn_seed_rmses": [round(v, 4) for v in seed_rmses],
            "tabmixnn_seed_r2s": [round(v, 4) for v in seed_r2s],
            "wins": mean_r < xgb_r,
        })

    print(f"\n{'='*52}")
    print("GROUP-HOLDOUT VALIDATED PROTOCOL SUMMARY")
    print(f"  (2000 epochs, full batch, lr={LR}, UN, no kl_auto_scale, mean of seeds {SEEDS})\n")
    header = f"{'Dataset':<12} {'XGB-ng RMSE':>12} {'TabMixNN RMSE':>14} {'TabMixNN R²':>12} {'Result':>8}"
    print(header)
    print("-" * len(header))
    wins = 0
    for r in rows:
        result = "WIN" if r["wins"] else "loss"
        wins += r["wins"]
        print(f"  {r['dataset']:<10} {r['xgb_nogroup_rmse']:>12.4f} "
              f"{r['tabmixnn_mean_rmse']:>14.4f} {r['tabmixnn_mean_r2']:>12.4f} {result:>8}")
    print(f"\nTabMixNN wins (GH, validated protocol): {wins}/{len(rows)}")

    out = RESULTS_DIR / "gh_validated_results.json"
    out.write_text(json.dumps({
        "meta": manifest_meta(),
        "protocol": {"epochs": EPOCHS, "lr": LR, "batch": "full_train",
                     "cov": "UN", "kl_auto_scale": False, "seeds": SEEDS},
        "results": rows,
    }, indent=2) + "\n")
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
