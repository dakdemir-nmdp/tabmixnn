"""Five-split descriptive sensitivity and primary-split calibration diagnostics.

No test-set selection or model recalibration is performed. Repeated holdouts
reuse observations, so their SD is descriptive and is not a standard error.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import hashlib
import json
import warnings
import numpy as np
import statsmodels.api as sm
from experiments.data_utils import load, DATA_DIR
from experiments.run_manuscript_benchmarks import DATASETS, metrics

MODELS=['TabMixNN','FixedNN','Mixed','XGBoost','RandomForest']
SPLITS=[20261015,20261016,20261017,20261018,20261019]


def calibration(actual, prediction):
    y=np.asarray(actual,dtype=float); p=np.asarray(prediction,dtype=float)
    if y.shape!=p.shape or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid probabilities')
    x=np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
    result=dict(mean_probability=float(p.mean()),observed_fraction=float(y.mean()),
                mean_probability_error=float((p-y).mean()),intercept=None,slope=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            fit=sm.GLM(y,sm.add_constant(x,has_constant='add'),family=sm.families.Binomial()).fit()
            ok=fit.converged and np.linalg.matrix_rank(fit.model.exog)==2 and np.isfinite(fit.params).all()
            result.update(converged=bool(ok),intercept=float(fit.params[0]) if ok else None,
                          slope=float(fit.params[1]) if ok else None)
        except (ValueError,np.linalg.LinAlgError) as exc:
            result.update(converged=False,error=str(exc))
        result['warnings']=[str(w.message) for w in caught]
    result['bins']=[]
    for lo,hi in zip(np.arange(0,1,.2),np.arange(.2,1.01,.2)):
        take=(p>=lo)&((p<hi) if hi<.999 else (p<=1))
        if take.any():
            result['bins'].append(dict(lower=float(lo),upper=float(hi),n=int(take.sum()),
                mean_prediction=float(p[take].mean()),observed_fraction=float(y[take].mean())))
    return result


def table(name,caption,label,columns,rows):
    header=' & '.join(columns)+r' \\'
    tex='\n'.join([r'\begin{table}[htbp]',r'\centering\small',r'\caption{'+caption+'}',
       r'\label{'+label+'}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{ll'+'r'*(len(columns)-2)+'}',
       r'\toprule',header,r'\midrule',*[' & '.join(row)+r' \\' for row in rows],
       r'\bottomrule',r'\end{tabular}}',r'\end{table}',''])
    (ROOT/'manuscript'/name).write_text(tex)


def main():
    paths=[ROOT/'experiments/results'/('manuscript_benchmarks.json' if s==SPLITS[0]
             else f'manuscript_split_{s}.json') for s in SPLITS]
    payloads=[json.loads(p.read_text()) for p in paths]
    for seed,payload in zip(SPLITS,payloads):
        assert payload['protocol']['split_seed']==seed and len(payload['results'])==18
        assert payload['protocol']['epochs']==2000 and payload['protocol']['seeds']==[42,123,456]
        assert payload['code_hashes']==payloads[0]['code_hashes']
    summary=[]; cal=[]; tables={'regression':[],'classification':[]}
    for cfg in DATASETS:
        for regime in ['A','B']:
            records=[next(r for r in p['results'] if r['dataset']==cfg['name'] and r['regime']==regime) for p in payloads]
            metric='rmse' if cfg['task']=='regression' else 'auc'
            label=cfg['name'] + (f" ({cfg['unit']})" if cfg['task']=='regression' else '')
            row=[label,regime]; by_model={}
            for model in MODELS:
                values=[float(np.mean([x['metrics'][metric] for x in r['records'] if x['model']==model])) for r in records]
                by_model[model]=dict(split_values=values,mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)))
                row.append(f"${np.mean(values):.3f} \\pm {np.std(values,ddof=1):.3f}$")
            summary.append(dict(dataset=cfg['name'],regime=regime,metric=metric,models=by_model,
                unique_query_splits=len({tuple(r['query_rows']) for r in records})))
            tables[cfg['task']].append(row)
            # Reconcile every saved metric with local source/predictions, all splits.
            df=load(cfg['loader'])
            for seed,r in zip(SPLITS,records):
                suffix='' if seed==SPLITS[0] else f'_split{seed}'
                local=DATA_DIR/'manuscript_runs'/f"{cfg['loader']}_{regime}_2000{suffix}"
                y=df.iloc[r['query_rows']][cfg['outcome']].to_numpy()
                if cfg['task']=='classification': y=(y==cfg['positive']).astype(float)
                else: y=y.astype(float)
                for rec in r['records']:
                    pred=np.load(local/f"{rec['model']}_{rec['seed']}.npz")
                    np.testing.assert_array_equal(pred['row'],r['query_rows'])
                    actual_metrics=metrics(y,pred['prediction'],cfg['task'])
                    for key,val in rec['metrics'].items():
                        np.testing.assert_allclose(actual_metrics[key],val,rtol=1e-10,atol=1e-10)
                    if seed==SPLITS[0] and cfg['task']=='classification':
                        cal.append(dict(dataset=cfg['name'],regime=regime,model=rec['model'],seed=rec['seed'],
                                        **calibration(y,pred['prediction'])))
    for task,rows in tables.items():
        metric='Raw-unit RMSE' if task=='regression' else 'AUROC'
        table(f'split_{task}_table.tex',metric+' over five fixed repeated holdouts. Each neural split value is the mean metric across three initialization seeds. Entries are mean $\\pm$ sample SD across split values. Reused observations and repeated group partitions make these descriptive SDs, not standard errors or confidence intervals.',
              f'tab:split_{task}',['Dataset','Regime','TabMixNN','Fixed NN','Mixed','XGBoost','RF'],rows)
    rows=[]
    for cfg in [c for c in DATASETS if c['task']=='classification']:
        for regime in ['A','B']:
            recs=[x for x in cal if x['dataset']==cfg['name'] and x['regime']==regime and x['model']=='TabMixNN']
            row=[cfg['name'],regime]
            for key in ['observed_fraction','mean_probability','mean_probability_error','intercept','slope']:
                vals=[x[key] for x in recs]
                row.append('--' if any(v is None for v in vals) else f'{np.mean(vals):.3f}')
            rows.append(row)
    table('calibration_table.tex','Primary-split TabMixNN probability diagnostics, averaged across three fitted seeds. The intercept and slope jointly fit $\\operatorname{logit}P(Y=1)=a+b\\operatorname{logit}(\\widehat p)$ on held-out rows. Their ideal values are 0 and 1. They are descriptive estimates, without independent-row uncertainty assumptions; no test-set recalibration is applied.',
          'tab:calibration',['Dataset','Regime','Observed','Predicted','Difference','$a$','$b$'],rows)
    output=dict(split_seeds=SPLITS,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sensitivity=summary,primary_calibration=cal,metric_reconstruction='passed for every model record in all 90 dataset/regime/split pairs')
    (ROOT/'experiments/results/manuscript_sensitivity.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print('Validated 90 pairs; wrote split sensitivity and calibration tables.')

if __name__=='__main__': main()
