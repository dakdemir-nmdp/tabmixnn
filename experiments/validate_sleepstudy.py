"""Reproducible methods check, not a reproduction of the paper's tables.

Run from an editable install: .venv/bin/python experiments/validate_sleepstudy.py
Public input is cached but excluded from git. Outputs contain aggregates only.
"""
import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import torch

from tabmixnn import TabMixNN

URL = 'https://vincentarelbundock.github.io/Rdatasets/csv/lme4/sleepstudy.csv'


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual)-np.asarray(predicted))**2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    parser.add_argument('--output', type=Path, default=Path('experiments/results/methods_sleepstudy.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    cache = Path(__file__).parent/'data'/'sleepstudy.csv'
    if cache.exists():
        raw = cache.read_bytes()
    else:
        response = requests.get(URL, timeout=60)  # TLS verification enabled
        response.raise_for_status()
        raw = response.content
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(raw)
    df = pd.read_csv(io.BytesIO(raw))
    assert len(df) == 180 and df.Subject.nunique() == 18
    # Stable source-order split; no response-dependent selection.
    levels = df.Subject.drop_duplicates().tolist()
    split_rng = np.random.default_rng(20261015)
    seen_test = np.concatenate([split_rng.choice(np.flatnonzero(df.Subject == g),2,replace=False)
                                for g in levels])
    seen_mask = np.zeros(len(df),dtype=bool); seen_mask[seen_test] = True
    held = df.Subject.isin(levels[-6:]).to_numpy()
    protocols = [('full_fit',np.ones(len(df),bool),np.ones(len(df),bool),None),
                 ('seen_group_holdout',~seen_mask,seen_mask,None),
                 ('new_group_zero_shot',~held,held & (df.Days>=5).to_numpy(),None),
                 ('new_group_support_query',~held,held & (df.Days>=5).to_numpy(),
                  held & (df.Days<5).to_numpy())]
    results = []
    for name, trainmask, querymask, supportmask in protocols:
        train0, query0 = df.loc[trainmask].copy(), df.loc[querymask].copy()
        ym, ys = train0.Reaction.mean(), train0.Reaction.std(ddof=0)
        xm, xs = train0.Days.mean(), train0.Days.std(ddof=0)
        def prepare(frame):
            return pd.DataFrame({'x':(frame.Days-xm)/xs,
                'y':(frame.Reaction-ym)/ys,'g':frame.Subject.astype(str)},index=frame.index)
        train, query = prepare(train0), prepare(query0)
        support = prepare(df.loc[supportmask]) if supportmask is not None else None
        X = sm.add_constant(train.x).to_numpy()
        reference = sm.MixedLM(train.y.to_numpy(),X,groups=train.g,exog_re=X).fit(reml=False,method='lbfgs')
        assert reference.converged
        Xq = np.column_stack([np.ones(len(query)),query.x])
        base = Xq @ np.asarray(reference.fe_params)
        re = reference.random_effects
        D, sigma2 = np.asarray(reference.cov_re),float(reference.scale)
        if support is not None:
            for group, rows in support.groupby('g'):
                Z = np.column_stack([np.ones(len(rows)),rows.x])
                residual = rows.y.to_numpy()-Z @ reference.fe_params
                re[group] = np.linalg.solve(Z.T @ Z/sigma2 + np.linalg.inv(D), Z.T @ residual/sigma2)
        refpred = base + np.array([Xq[i] @ np.asarray(re[g]) if g in re else 0.
                                  for i,g in enumerate(query.g)])
        ref_rmse = rmse(query0.Reaction,refpred*ys+ym)
        # Covariance and coefficients transformed back to original units.
        transform = np.array([[ys,-ys*xm/xs],[0,ys/xs]])
        def original_beta(beta):
            return (transform @ np.asarray(beta)+[ym,0]).tolist()
        reference_record = {'rmse_ms':ref_rmse,'fixed_effects':original_beta(reference.fe_params),
            'random_covariance':(transform @ D @ transform.T).tolist(),
            'residual_sd_ms':float(np.sqrt(sigma2)*ys)}
        for seed in args.seeds:
            model = TabMixNN.lmm('y ~ x + (1+x|g)',cov_struct_map={'g':{'type':'UN'}},
                epochs=args.epochs,batch_size=len(train),lr=.01,random_state=seed,verbose=False)
            model.fit(train)
            if support is not None:
                prediction = model.predict_transductive(query.drop(columns='y'),support_df=support,
                                                         n_steps=1000,lr=.02)
            else:
                prediction = model.predict(query.drop(columns='y'))
            beta = model.encoder.fixed_linear.weight[0].detach().numpy()
            covariance = model.get_variance_summary()['term_0']['covariance']
            result = {'protocol':name,'seed':seed,'n_train':len(train),'n_query':len(query),
                'n_support':0 if support is None else len(support),
                'rmse_ms':rmse(query0.Reaction,prediction*ys+ym),
                'fixed_effects':original_beta(beta),
                'random_covariance':(transform @ covariance @ transform.T).tolist(),
                'residual_sd_ms':float(model.data_family('main').sigma.detach())*ys,
                'statsmodels_ml':reference_record}
            results.append(result)
            print(f"{name} seed={seed}: TabMixNN={result['rmse_ms']:.4f}, ML={ref_rmse:.4f}",flush=True)
    payload = {'source':URL,'sha256':hashlib.sha256(raw).hexdigest(),
        'versions':{p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','scipy','statsmodels','tabmixnn']},
        'settings':{'epochs':args.epochs,'lr':.01,'batch':'full training set','lambda_kl':1,
                    'centering':False,'covariance':'UN','split_seed':20261015},
        'notes':['Full-fit RMSE is descriptive, not predictive evaluation.',
                 'Train-only standardization; same rows/model structure for both methods.',
                 'New groups: first five days support, final five query; zero-shot uses same query.',
                 'Variational ELBO versus exact marginal ML; no REML equivalence claimed.'],
        'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+'\n')


if __name__ == '__main__':
    main()
