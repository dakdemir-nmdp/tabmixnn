#!/usr/bin/env python3
"""
Run each benchmark dataset in a fresh subprocess to avoid MPS memory segfaults.
Usage: python experiments/run_all.py [--regression-only] [--classification-only]

Split seed: 20261015 (matches experiments/splits.py SPLIT_SEED).
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

REGRESSION_DATASETS = [
    ("Sleepstudy",  "sleepstudy",  "Reaction ~ Days + (1 + Days | Subject)",  "Reaction", "Subject", "Days,Subject"),
    ("ChickWeight", "chickweight", "weight ~ Time + Diet + (1 | Chick)",       "weight",   "Chick",   "Time,Diet,Chick"),
    ("Machines",    "machines",    "score ~ Machine + (1 | Worker)",            "score",    "Worker",  "Machine,Worker"),
    ("Oxboys",      "oxboys",      "height ~ age + (1 + age | Subject)",        "height",   "Subject", "age,Subject,Occasion"),
    ("Loblolly",    "loblolly",    "height ~ age + (1 | Seed)",                 "height",   "Seed",    "age,Seed"),
]

CLASSIFICATION_DATASETS = [
    ("Aids2",   "aids2"),
    ("Males",   "males"),
    ("VerbAgg", "verbagg"),
    ("Train",   "train"),
]

REG_WORKER = """
import logging, sys, warnings, json
logging.getLogger("tabmixnn").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")
sys.path.insert(0, {root!r})

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from experiments.data_utils import load
from tabmixnn import TabMixNNRegressor

df = load({loader!r})
formula = {formula!r}
outcome = {outcome!r}
group   = {group!r}
xgb_features = {xgb_features!r}

for col in xgb_features:
    if df[col].dtype == object or str(df[col].dtype) == "category":
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

kf = KFold(n_splits=5, shuffle=True, random_state=20261015)
tab_rmses, tab_r2s, xgb_rmses, xgb_r2s = [], [], [], []

for tr_idx, te_idx in kf.split(df):
    train, test = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
    m = TabMixNNRegressor(formula=formula, hidden_dims=[32], embedding_dim=8,
        epochs=100, batch_size=32, learning_rate=0.005, kl_auto_scale=True, verbose=False)
    m.fit(train)
    pred = m.predict(test)
    y = test[outcome].values
    tab_rmses.append(float(np.sqrt(np.mean((pred - y)**2))))
    tab_r2s.append(float(r2_score(y, pred)))
    del m

    X_tr, y_tr = df.iloc[tr_idx][xgb_features].values, df.iloc[tr_idx][outcome].values
    X_te, y_te = df.iloc[te_idx][xgb_features].values, df.iloc[te_idx][outcome].values
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=20261015, verbosity=0)
    xgb.fit(X_tr, y_tr)
    xp = xgb.predict(X_te)
    xgb_rmses.append(float(np.sqrt(np.mean((xp - y_te)**2))))
    xgb_r2s.append(float(r2_score(y_te, xp)))

print(json.dumps({{
    "tab_rmse": round(float(np.mean(tab_rmses)), 4),
    "tab_r2":   round(float(np.mean(tab_r2s)), 4),
    "xgb_rmse": round(float(np.mean(xgb_rmses)), 4),
    "xgb_r2":   round(float(np.mean(xgb_r2s)), 4),
}}))
"""

CLS_WORKER = """
import logging, sys, warnings, json
logging.getLogger("tabmixnn").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")
sys.path.insert(0, {root!r})

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from experiments.data_utils import load
from tabmixnn import TabMixNNClassifier

# --- dataset-specific prep ---
name = {name!r}
df = load({loader!r})

if name == "Aids2":
    df["status_bin"]  = (df["status"] == "D").astype(int)
    df["sex_bin"]     = (df["sex"] == "M").astype(int)
    df["T_categ_enc"] = LabelEncoder().fit_transform(df["T.categ"].astype(str))
    df["state_enc"]   = LabelEncoder().fit_transform(df["state"].astype(str))
    outcome  = "status_bin"
    formula  = "status_bin ~ age + sex_bin + T_categ_enc + (1 | state)"
    xgb_feat = ["age", "sex_bin", "T_categ_enc", "state_enc"]

elif name == "Males":
    df["union_bin"]    = (df["union"] == "yes").astype(int)
    df["ethn_enc"]     = LabelEncoder().fit_transform(df["ethn"].astype(str))
    df["industry_enc"] = LabelEncoder().fit_transform(df["industry"].astype(str))
    outcome  = "union_bin"
    formula  = "union_bin ~ school + exper + ethn_enc + industry_enc + (1 | nr)"
    xgb_feat = ["school", "exper", "ethn_enc", "industry_enc", "nr"]

elif name == "VerbAgg":
    df["r2_bin"]     = (df["r2"] == "Y").astype(int)
    df["Gender_bin"] = (df["Gender"] == "M").astype(int)
    df["btype_enc"]  = LabelEncoder().fit_transform(df["btype"].astype(str))
    df["situ_enc"]   = LabelEncoder().fit_transform(df["situ"].astype(str))
    df["mode_enc"]   = LabelEncoder().fit_transform(df["mode"].astype(str))
    outcome  = "r2_bin"
    formula  = "r2_bin ~ Anger + Gender_bin + btype_enc + situ_enc + mode_enc + (1 | id)"
    xgb_feat = ["Anger", "Gender_bin", "btype_enc", "situ_enc", "mode_enc"]

elif name == "Train":
    # "choice1" is the train-choice level in the Ecdat Train dataset.
    df["choice_bin"] = (df["choice"] == "choice1").astype(int)
    outcome  = "choice_bin"
    formula  = "choice_bin ~ price1 + time1 + change1 + comfort1 + price2 + time2 + change2 + comfort2 + choiceid + (1 | id)"
    xgb_feat = ["price1","time1","change1","comfort1","price2","time2","change2","comfort2","choiceid","id"]

kf = KFold(n_splits=5, shuffle=True, random_state=20261015)
tab_aucs, tab_accs, xgb_aucs, xgb_accs = [], [], [], []

for tr_idx, te_idx in kf.split(df):
    train, test = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
    m = TabMixNNClassifier(formula=formula, hidden_dims=[32], embedding_dim=8,
        epochs=100, batch_size=64, learning_rate=0.005, kl_auto_scale=True, verbose=False)
    m.fit(train)
    prob = m.predict_proba(test)
    if hasattr(prob, "ndim") and prob.ndim == 2: prob = prob[:, 1]
    y = test[outcome].values
    try: tab_aucs.append(float(roc_auc_score(y, prob)))
    except: tab_aucs.append(float("nan"))
    tab_accs.append(float(accuracy_score(y, (np.asarray(prob) >= 0.5).astype(int))))
    del m

    X_tr = df.iloc[tr_idx][xgb_feat].values
    y_tr = df.iloc[tr_idx][outcome].values
    X_te = df.iloc[te_idx][xgb_feat].values
    y_te = df.iloc[te_idx][outcome].values
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, random_state=20261015,
                        verbosity=0, eval_metric="logloss")
    xgb.fit(X_tr, y_tr)
    xp = xgb.predict_proba(X_te)[:, 1]
    try: xgb_aucs.append(float(roc_auc_score(y_te, xp)))
    except: xgb_aucs.append(float("nan"))
    xgb_accs.append(float(accuracy_score(y_te, (xp >= 0.5).astype(int))))

print(json.dumps({{
    "tab_auc": round(float(np.nanmean(tab_aucs)), 4),
    "tab_acc": round(float(np.mean(tab_accs)), 4),
    "xgb_auc": round(float(np.nanmean(xgb_aucs)), 4),
    "xgb_acc": round(float(np.mean(xgb_accs)), 4),
}}))
"""


def run_subprocess(code: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    # last non-empty line should be the JSON result
    lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    if not lines:
        raise RuntimeError(f"No output.\nSTDERR:\n{result.stderr[-2000:]}")
    return json.loads(lines[-1])


def main():
    reg_only = "--regression-only" in sys.argv
    cls_only = "--classification-only" in sys.argv

    reg_rows, cls_rows = [], []

    if not cls_only:
        print("\n" + "="*60)
        print("REGRESSION BENCHMARK (5-fold CV, split_seed=20261015)")
        print("="*60)
        for display, loader, formula, outcome, group, xgb_cols in REGRESSION_DATASETS:
            print(f"\n  {display}...", flush=True)
            code = REG_WORKER.format(
                root=str(ROOT),
                loader=loader,
                formula=formula,
                outcome=outcome,
                group=group,
                xgb_features=xgb_cols.split(","),
            )
            r = run_subprocess(code)
            print(f"    TabMixNN  RMSE={r['tab_rmse']:.4f}  R²={r['tab_r2']:.4f}")
            print(f"    XGBoost   RMSE={r['xgb_rmse']:.4f}  R²={r['xgb_r2']:.4f}")
            reg_rows.append({"dataset": display, **{f"tabmixnn_{k[4:]}": v for k, v in r.items() if k.startswith("tab")},
                             **{f"xgboost_{k[4:]}": v for k, v in r.items() if k.startswith("xgb")},
                             "tabmixnn_wins_rmse": r["tab_rmse"] < r["xgb_rmse"]})

        import pandas as pd
        from pathlib import Path
        results_dir = ROOT / "experiments" / "results"
        results_dir.mkdir(exist_ok=True)
        reg_df = pd.DataFrame(reg_rows)
        reg_df.to_csv(results_dir / "regression_results.csv", index=False)
        wins = reg_df["tabmixnn_wins_rmse"].sum()
        print(f"\nREGRESSION SUMMARY")
        print(reg_df.to_string(index=False))
        print(f"\nTabMixNN wins (RMSE): {wins}/{len(reg_df)}")

    if not reg_only:
        print("\n" + "="*60)
        print("CLASSIFICATION BENCHMARK (5-fold CV, split_seed=20261015)")
        print("="*60)
        for display, loader in CLASSIFICATION_DATASETS:
            print(f"\n  {display}...", flush=True)
            code = CLS_WORKER.format(root=str(ROOT), name=display, loader=loader)
            r = run_subprocess(code)
            print(f"    TabMixNN  AUC={r['tab_auc']:.4f}  Acc={r['tab_acc']:.4f}")
            print(f"    XGBoost   AUC={r['xgb_auc']:.4f}  Acc={r['xgb_acc']:.4f}")
            cls_rows.append({"dataset": display,
                             "tabmixnn_auc": r["tab_auc"], "tabmixnn_acc": r["tab_acc"],
                             "xgboost_auc":  r["xgb_auc"], "xgboost_acc":  r["xgb_acc"],
                             "tabmixnn_wins_auc": r["tab_auc"] > r["xgb_auc"]})

        import pandas as pd
        from pathlib import Path
        results_dir = ROOT / "experiments" / "results"
        results_dir.mkdir(exist_ok=True)
        cls_df = pd.DataFrame(cls_rows)
        cls_df.to_csv(results_dir / "classification_results.csv", index=False)
        wins = cls_df["tabmixnn_wins_auc"].sum()
        print(f"\nCLASSIFICATION SUMMARY")
        print(cls_df.to_string(index=False))
        print(f"\nTabMixNN wins (AUC): {wins}/{len(cls_df)}")


if __name__ == "__main__":
    main()
