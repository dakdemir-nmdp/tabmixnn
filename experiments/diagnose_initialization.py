"""Fixed-budget, exploratory before/after diagnostic; no model selection.

Run before and after a package change with separate --output paths.
Uses the existing group split and UN protocol, shortened to 200 epochs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tabmixnn import TabMixNNRegressor
from experiments.run_gh_validated import DATASETS, SEEDS, load, rmse, xgb_nogroup
from experiments.splits import group_holdout, manifest_meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    result = dict(meta=manifest_meta(),
                  git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                  source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                                 for p in ['core/model.py', 'core/encoder.py', 'covariance/structures.py']},
                  protocol=dict(epochs=args.epochs, batch='full_train', lr=.01,
                                lambda_kl=1., kl_auto_scale=False, cov='UN', seeds=SEEDS,
                                hidden_dims=[32], embedding_dim=8, threads=1),
                  versions=dict(torch=torch.__version__, numpy=np.__version__), results=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for cfg in DATASETS:
        df = load(cfg['loader'])
        train_mask, test_mask = group_holdout(df, cfg['group'])
        train, test = df[train_mask].copy(), df[test_mask].copy()
        baseline, _ = xgb_nogroup(train, test, cfg['outcome'], cfg['xgb_nogroup_features'])
        row = dict(dataset=cfg['name'], xgb_nogroup_rmse=baseline,
                   data_sha256=hashlib.sha256(df.to_csv(index=False).encode()).hexdigest(),
                   train_positions=np.flatnonzero(train_mask).tolist(),
                   test_positions=np.flatnonzero(test_mask).tolist(), runs=[])
        for seed in SEEDS:
            start = time.monotonic()
            model = TabMixNNRegressor(cfg['formula'], hidden_dims=[32], embedding_dim=8,
                                     epochs=args.epochs, batch_size=len(train), lr=.01,
                                     random_state=seed, cov_struct_map=cfg['cov_struct'])
            model.fit(train)
            prediction = model.predict(test)
            row['runs'].append(dict(seed=seed, rmse=rmse(test[cfg['outcome']], prediction),
                                    predictions=prediction.tolist(), loss=model.history,
                                    seconds=time.monotonic()-start))
        row['mean_rmse'] = float(np.mean([r['rmse'] for r in row['runs']]))
        result['results'].append(row)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(f"{cfg['name']}: {row['mean_rmse']:.4f} (XGB {baseline:.4f})", flush=True)


if __name__ == '__main__':
    main()
