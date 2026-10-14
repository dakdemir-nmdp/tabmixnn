"""
Classification benchmark — reproduces Table 4 from arXiv:2512.23787.

Datasets: Aids2, Males, VerbAgg, Train
Metrics:  AUC, Accuracy
Baseline: XGBoost (no random effects)
Results saved to: experiments/results/classification_results.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
torch.backends.mps.is_available = lambda: False  # prevent MPS LayerNorm dispatch on Apple Silicon

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

from tabmixnn import TabMixNNClassifier
from experiments.data_utils import load

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_SPLITS = 5
EPOCHS = 200
SEED = 42


def cv_tabmixnn(df, formula, group_col, outcome, n_splits=N_SPLITS):
    """K-fold CV for TabMixNNClassifier; returns mean AUC and accuracy."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    aucs, accs = [], []
    for train_idx, test_idx in kf.split(df):
        train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
        model = TabMixNNClassifier(
            formula=formula,
            hidden_dims=[32],
            embedding_dim=8,
            epochs=EPOCHS,
            batch_size=64,
            learning_rate=0.005,
            kl_auto_scale=True,
            verbose=False,
        )
        model.fit(train)
        prob = model.predict_proba(test)
        # predict_proba may return (n,) or (n,2); take positive class
        if hasattr(prob, "ndim") and prob.ndim == 2:
            prob = prob[:, 1]
        y = test[outcome].values
        try:
            aucs.append(roc_auc_score(y, prob))
        except Exception:
            aucs.append(float("nan"))
        pred = (np.asarray(prob) >= 0.5).astype(int)
        accs.append(accuracy_score(y, pred))
    return float(np.nanmean(aucs)), float(np.mean(accs))


def cv_xgboost(df, outcome, features, n_splits=N_SPLITS):
    """K-fold CV for XGBoost baseline; returns mean AUC and accuracy."""
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
    aucs, accs = [], []
    for train_idx, test_idx in kf.split(X):
        xgb = XGBClassifier(n_estimators=200, learning_rate=0.05,
                             random_state=SEED, verbosity=0, eval_metric="logloss")
        xgb.fit(X[train_idx], y[train_idx])
        prob = xgb.predict_proba(X[test_idx])[:, 1]
        try:
            aucs.append(roc_auc_score(y[test_idx], prob))
        except Exception:
            aucs.append(float("nan"))
        accs.append(accuracy_score(y[test_idx], (prob >= 0.5).astype(int)))
    return float(np.nanmean(aucs)), float(np.mean(accs))


def prep_aids2(df):
    """Binary: status D=1 (died), A=0 (alive). Group: state."""
    df = df.copy()
    df["status_bin"] = (df["status"] == "D").astype(int)
    df["sex_bin"] = (df["sex"] == "M").astype(int)
    df["T_categ_enc"] = LabelEncoder().fit_transform(df["T.categ"].astype(str))
    df["state_enc"] = LabelEncoder().fit_transform(df["state"].astype(str))
    return df, "status_bin ~ age + sex_bin + T_categ_enc + (1 | state)", "state", \
           ["age", "sex_bin", "T_categ_enc", "state_enc"]


def prep_males(df):
    """Binary: union membership (yes=1). Group: nr (individual)."""
    df = df.copy()
    df["union_bin"] = (df["union"] == "yes").astype(int)
    df["ethn_enc"] = LabelEncoder().fit_transform(df["ethn"].astype(str))
    df["industry_enc"] = LabelEncoder().fit_transform(df["industry"].astype(str))
    return df, "union_bin ~ school + exper + ethn_enc + industry_enc + (1 | nr)", "nr", \
           ["school", "exper", "ethn_enc", "industry_enc", "nr"]


def prep_verbagg(df):
    """Binary: r2 (Y=1, N=0). Group: id (subject)."""
    df = df.copy()
    df["r2_bin"] = (df["r2"] == "Y").astype(int)
    df["Gender_bin"] = (df["Gender"] == "M").astype(int)
    df["btype_enc"] = LabelEncoder().fit_transform(df["btype"].astype(str))
    df["situ_enc"] = LabelEncoder().fit_transform(df["situ"].astype(str))
    df["mode_enc"] = LabelEncoder().fit_transform(df["mode"].astype(str))
    return df, "r2_bin ~ Anger + Gender_bin + btype_enc + situ_enc + mode_enc + (1 | id)", "id", \
           ["Anger", "Gender_bin", "btype_enc", "situ_enc", "mode_enc"]


def prep_train(df):
    """Binary: choice (train=1, car=0). Group: id (individual)."""
    df = df.copy()
    df["choice_bin"] = (df["choice"] == "train").astype(int)
    return df, \
           "choice_bin ~ price1 + time1 + change1 + comfort1 + price2 + time2 + change2 + comfort2 + (1 | id)", \
           "id", \
           ["price1", "time1", "change1", "comfort1", "price2", "time2", "change2", "comfort2", "id"]


DATASETS = [
    ("Aids2",   "aids2",   prep_aids2),
    ("Males",   "males",   prep_males),
    ("VerbAgg", "verbagg", prep_verbagg),
    ("Train",   "train",   prep_train),
]


def main():
    rows = []
    for display_name, loader_key, prep_fn in DATASETS:
        print(f"\n{'='*50}")
        print(f"Dataset: {display_name}")
        raw = load(loader_key)
        df, formula, group_col, xgb_features = prep_fn(raw)
        outcome = formula.split("~")[0].strip()
        print(f"  n={len(df)}, outcome={outcome}, group={group_col}")

        print(f"  TabMixNN ({N_SPLITS}-fold CV)...")
        tab_auc, tab_acc = cv_tabmixnn(df, formula, group_col, outcome)
        print(f"    AUC={tab_auc:.4f}  Acc={tab_acc:.4f}")

        print(f"  XGBoost ({N_SPLITS}-fold CV)...")
        xgb_auc, xgb_acc = cv_xgboost(df, outcome, xgb_features)
        print(f"    AUC={xgb_auc:.4f}  Acc={xgb_acc:.4f}")

        rows.append({
            "dataset":           display_name,
            "tabmixnn_auc":      round(tab_auc, 4),
            "tabmixnn_acc":      round(tab_acc, 4),
            "xgboost_auc":       round(xgb_auc, 4),
            "xgboost_acc":       round(xgb_acc, 4),
            "tabmixnn_wins_auc": tab_auc > xgb_auc,
        })

    results = pd.DataFrame(rows)
    out = RESULTS_DIR / "classification_results.csv"
    results.to_csv(out, index=False)

    print(f"\n{'='*50}")
    print("CLASSIFICATION SUMMARY")
    print(results.to_string(index=False))
    wins = results["tabmixnn_wins_auc"].sum()
    print(f"\nTabMixNN wins (AUC): {wins}/{len(results)} datasets")
    print(f"Results saved to: {out}")
    return results


if __name__ == "__main__":
    main()
