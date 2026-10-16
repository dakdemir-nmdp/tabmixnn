"""Canonical two-regime benchmark. No test-driven tuning; aggregate outputs only.

Run .venv/bin/python experiments/run_manuscript_benchmarks.py
Rscript with lme4/jsonlite is required for independent classical baselines.
Cached source records and per-row predictions stay in ignored experiments/data.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss
from xgboost import XGBRegressor, XGBClassifier
from tabmixnn import TabMixNN, Gaussian, Binomial, disable_logging
from tabmixnn.core.model import TabMixDataset
from experiments.data_utils import load, _URLS, DATA_DIR
from experiments.splits import SPLIT_SEED, group_holdout

SEEDS = [42, 123, 456]
DATASETS = [
 dict(name='Sleepstudy', loader='sleepstudy', task='regression', outcome='Reaction', group='Subject', numeric=['Days'], categorical=[], slope='Days', unit='ms'),
 dict(name='ChickWeight', loader='chickweight', task='regression', outcome='weight', group='Chick', numeric=['Time'], categorical=['Diet'], slope=None, unit='g'),
 dict(name='Machines', loader='machines', task='regression', outcome='score', group='Worker', numeric=[], categorical=['Machine'], slope=None, unit='score'),
 dict(name='Oxboys', loader='oxboys', task='regression', outcome='height', group='Subject', numeric=['age'], categorical=[], slope='age', unit='cm'),
 dict(name='Loblolly', loader='loblolly', task='regression', outcome='height', group='Seed', numeric=['age'], categorical=[], slope=None, unit='ft'),
 dict(name='Aids2', loader='aids2', task='classification', outcome='status', positive='D', group='state', numeric=['age'], categorical=['sex','T.categ'], slope=None, unit='probability'),
 dict(name='Males', loader='males', task='classification', outcome='union', positive='yes', group='nr', numeric=['school','exper'], categorical=['ethn','industry'], slope=None, unit='probability'),
 dict(name='VerbAgg', loader='verbagg', task='classification', outcome='r2', positive='Y', group='id', numeric=['Anger'], categorical=['Gender','btype','situ','mode'], slope=None, unit='probability'),
 dict(name='Train', loader='train', task='classification', outcome='choice', positive='choice1', group='id', numeric=['price1','time1','change1','comfort1','price2','time2','change2','comfort2'], categorical=[], slope=None, unit='probability'),
]

class FullBatchModel(TabMixNN):
    """Use the package training loop with a cached, source-order full batch.

    Avoids repeated per-row DataLoader collation, without changing the objective
    or optimizer. Full-batch row order is fixed; randomness is only model/VI draws.
    """
    def _make_dataloader(self, data, shuffle=False):
        if isinstance(data, TabMixDataset) and self.batch_size >= len(data):
            return [(data.X, data.Z_ind, data.Z_slopes, data.Ys)], len(data)
        return super()._make_dataloader(data, shuffle=shuffle)


def within_group_holdout(df, group, frac=.25, seed=SPLIT_SEED):
    rng = np.random.default_rng(seed)
    test = np.zeros(len(df), dtype=bool)
    for positions in df.groupby(group, sort=False).indices.values():
        if len(positions) > 1:
            n = min(len(positions)-1, max(1, int(len(positions)*frac)))
            test[rng.choice(positions, n, replace=False)] = True
    if not test.any():
        raise ValueError('No repeated groups for regime A')
    return ~test, test


def prepare(cfg, df, train_mask, test_mask):
    """Learn category levels and numeric scales using training rows only."""
    tr, te = df.loc[train_mask].copy(), df.loc[test_mask].copy()
    for col in cfg['categorical']:
        tr[col], te[col] = tr[col].astype(str), te[col].astype(str)
    transformers = []
    if cfg['numeric']:
        transformers.append(('numeric', StandardScaler(), cfg['numeric']))
    if cfg['categorical']:
        transformers.append(('category', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cfg['categorical']))
    transform = ColumnTransformer(transformers)
    X = transform.fit_transform(tr)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Found unknown categories')
        Q = transform.transform(te)
    names = [f'x{i}' for i in range(X.shape[1])]
    train, query = pd.DataFrame(X, columns=names), pd.DataFrame(Q, columns=names)
    # Remap source IDs locally. Only row positions/counts are exported.
    groups = pd.factorize(df[cfg['group']], sort=False)[0]
    train['g'], query['g'] = groups[train_mask].astype(str), groups[test_mask].astype(str)
    y = df[cfg['outcome']].to_numpy()
    if cfg['task'] == 'classification':
        y = (y == cfg['positive']).astype(float)
        if np.unique(y[train_mask]).size != 2 or np.unique(y[test_mask]).size != 2:
            raise ValueError('Both classes required in train and query')
        ym, ys = 0., 1.
    else:
        y = y.astype(float)
        ym, ys = float(y[train_mask].mean()), float(y[train_mask].std())
    if ys <= 0 or not np.isfinite(X).all() or not np.isfinite(Q).all():
        raise ValueError('Degenerate scale or nonfinite design')
    train['y'], query['y'] = (y[train_mask]-ym)/ys, (y[test_mask]-ym)/ys
    fixed = 'y ~ ' + ' + '.join(names)
    slope = None if cfg['slope'] is None else f"x{cfg['numeric'].index(cfg['slope'])}"
    mixed = fixed + (' + (1 | g)' if slope is None else f' + (1 + {slope} | g)')
    return train, query, fixed, mixed, ym, ys, transform


def metrics(actual, pred, task):
    actual, pred = np.asarray(actual), np.asarray(pred).reshape(-1)
    if actual.shape != pred.shape or not np.isfinite(pred).all():
        raise ValueError('Invalid prediction shape or nonfinite prediction')
    if task == 'regression':
        return dict(rmse=float(np.sqrt(mean_squared_error(actual,pred))),
                    mae=float(mean_absolute_error(actual,pred)), r2=float(r2_score(actual,pred)))
    if np.any((pred < 0) | (pred > 1)):
        raise ValueError('Probability outside [0,1]')
    return dict(auc=float(roc_auc_score(actual,pred)), accuracy=float(accuracy_score(actual,pred>=.5)),
                brier=float(brier_score_loss(actual,pred)), log_loss=float(log_loss(actual,pred,labels=[0,1])))


def make_model(formula, task, seed, epochs, n_train, linear=False, structure=None):
    return FullBatchModel(formula, family=Gaussian() if task=='regression' else Binomial(),
        hidden_dims=[] if linear else [32], embedding_dim=1 if linear else 8,
        use_identity_head=linear, cov_struct_map={'g':{'type':'UN'}},
        epochs=epochs, batch_size=n_train, lr=.01, weight_decay=0., lambda_kl=1.,
        kl_auto_scale=False, kl_normalize=False, enforce_centering=False,
        use_static_structure=structure in ('static','hybrid'),
        use_dynamic_structure=structure in ('dynamic','hybrid'),
        dynamic_structure_config={'feature_mixing': True} if structure else None,
        lambda_sparse=0., lambda_contraction=0., hidden_dropout=0., backbone_dropout=0.,
        random_state=seed, device='cpu', verbose=False)


def classical(train, query, formula, task, regime, work):
    work.mkdir(parents=True, exist_ok=True)
    train.to_csv(work/'train.csv',index=False)
    query.drop(columns='y').to_csv(work/'query.csv',index=False)
    run = subprocess.run(['Rscript',str(ROOT/'experiments/mixed_reference.R'),str(work/'train.csv'),
        str(work/'query.csv'),formula,task,regime,str(work/'reference.json')],
        capture_output=True,text=True,check=True)
    ref = json.loads((work/'reference.json').read_text())
    pred = np.asarray(ref.pop('prediction'))
    ref['stderr'] = run.stderr.strip()
    return pred, ref


def run_one(cfg, regime, epochs, seeds, output):
    df = load(cfg['loader'])
    cols = cfg['numeric']+cfg['categorical']+[cfg['group'],cfg['outcome']]
    if df[cols].isna().any().any():
        raise ValueError(f"Missing model input in {cfg['name']}; define a training-only policy first")
    tm,qm = within_group_holdout(df,cfg['group']) if regime=='A' else group_holdout(df,cfg['group'])
    train,query,fixed,mixed,ym,ys,transform = prepare(cfg,df,tm,qm)
    known=set(train.g); seen=set(query.g)
    assert seen <= known if regime=='A' else seen.isdisjoint(known)
    actual=query.y.to_numpy()*ys+ym
    records=[]
    local = DATA_DIR/'manuscript_runs'/f"{cfg['loader']}_{regime}_{epochs}"
    local.mkdir(parents=True,exist_ok=True)
    def record(name, seed, pred, **extra):
        m=metrics(actual,pred,cfg['task'])
        records.append(dict(model=name,seed=seed,metrics=m,**extra))
        # Local-only numeric predictions; no source IDs, ignored by git.
        np.savez(local/f'{name}_{seed}.npz', row=np.flatnonzero(qm), prediction=pred)
        print(cfg['name'],regime,name,seed,m,flush=True)
    start=time.perf_counter()
    p,meta=classical(train,query,mixed,cfg['task'],regime,local/'mixed')
    record('Mixed',None,p*ys+ym,diagnostics=meta,seconds=time.perf_counter()-start)
    features=[c for c in train if c.startswith('x')]
    X,Q=train[features].to_numpy(),query[features].to_numpy()
    if regime=='A':
        group_enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
        X=np.column_stack([X,group_enc.fit_transform(train[['g']])])
        Q=np.column_stack([Q,group_enc.transform(query[['g']])])
    for name in ['XGBoost','RandomForest']:
        cls = (XGBRegressor if cfg['task']=='regression' else XGBClassifier) if name=='XGBoost' else (RandomForestRegressor if cfg['task']=='regression' else RandomForestClassifier)
        kw=dict(n_estimators=200,random_state=42,n_jobs=1)
        if name=='XGBoost': kw.update(learning_rate=.05,max_depth=6,verbosity=0)
        else: kw.update(min_samples_leaf=2)
        model=cls(**kw); start=time.perf_counter(); model.fit(X,train.y)
        p=model.predict(Q) if cfg['task']=='regression' else model.predict_proba(Q)[:,1]
        record(name,42,p*ys+ym,seconds=time.perf_counter()-start)
    variants=[('TabMixNN',mixed,False),('FixedNN',fixed,False)]
    if cfg['name']=='Sleepstudy': variants.append(('LinearVI',mixed,True))
    for name,formula,linear in variants:
        for seed in seeds:
            start=time.perf_counter(); m=make_model(formula,cfg['task'],seed,epochs,len(train),linear)
            m.fit(train)
            p=m.predict(query.drop(columns='y'),mode='proba' if cfg['task']=='classification' else 'default')
            record(name,seed,np.asarray(p).reshape(-1)*ys+ym,seconds=time.perf_counter()-start,
                   training_loss_initial=float(m.history[0]), training_loss_final=float(m.history[-1]),
                   training_loss_last100_mean=float(np.mean(m.history[-100:])))
    return dict(dataset=cfg['name'],task=cfg['task'],regime=regime,unit=cfg['unit'],
        n=len(df),p=len(cfg['numeric'])+len(cfg['categorical']),G=int(df[cfg['group']].nunique()),
        n_train=int(tm.sum()),n_query=int(qm.sum()),G_train=len(known),G_query=len(seen),
        source=_URLS[cfg['loader']],source_sha256=hashlib.sha256((DATA_DIR/f"{cfg['loader']}.csv").read_bytes()).hexdigest(),
        features=cfg['numeric']+cfg['categorical'],numeric=cfg['numeric'],categorical=cfg['categorical'],
        fixed_formula=fixed,mixed_formula=mixed,design_width=len(features),
        design_names=transform.get_feature_names_out().tolist(),
        train_rows=np.flatnonzero(tm).tolist(),query_rows=np.flatnonzero(qm).tolist(),
        target_training_mean=ym,target_training_sd=ys,records=records)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--datasets',nargs='+',default=[d['name'] for d in DATASETS])
    ap.add_argument('--regimes',nargs='+',choices=['A','B'],default=['A','B'])
    ap.add_argument('--epochs',type=int,default=2000)
    ap.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    ap.add_argument('--output',type=Path,default=ROOT/'experiments/results/manuscript_benchmarks.json')
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args(); torch.set_num_threads(1); disable_logging()
    if set(args.datasets)-{d['name'] for d in DATASETS}: raise ValueError('Unknown dataset')
    protocol=dict(epochs=args.epochs,seeds=args.seeds,lr=.01,batch='full source-order training batch',
                  hidden=[32],embedding=8,covariance='UN',lambda_kl=1,weight_decay=0,
                  split_seed=SPLIT_SEED,split_version='manuscript-v1',test_fraction=.25,
                  tree_group_features='one-hot in A, absent in B',early_stopping=False)
    code_files=['experiments/run_manuscript_benchmarks.py','experiments/mixed_reference.R',
                'experiments/data_utils.py','experiments/splits.py']
    code_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in code_files}
    payload=dict(protocol=protocol,code_hashes=code_hashes,
                 versions={p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','scikit-learn','xgboost','tabmixnn']},results=[])
    if args.resume and args.output.exists():
        payload=json.loads(args.output.read_text())
        if payload['protocol'] != protocol or payload['code_hashes'] != code_hashes: raise ValueError('Resume protocol/code mismatch')
    for cfg in DATASETS:
        if cfg['name'] not in args.datasets: continue
        for regime in args.regimes:
            if any(r['dataset']==cfg['name'] and r['regime']==regime for r in payload['results']): continue
            payload['results'].append(run_one(cfg,regime,args.epochs,args.seeds,args.output))
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print('Saved',args.output,flush=True)

if __name__=='__main__': main()
