"""Generate every reported table directly from canonical aggregate result files."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
MODELS=['TabMixNN','FixedNN','Mixed','XGBoost','RandomForest']


def cell(records, model, metric):
    values=[r['metrics'][metric] for r in records if r['model']==model]
    if not values: raise ValueError(f'Missing {model}/{metric}')
    mean=np.mean(values)
    return (f'${mean:.3f} \\pm {np.std(values,ddof=1):.3f}$' if len(values)>1 else f'{mean:.3f}')


def table(caption,label,header,rows,spec):
    return '\n'.join([r'\begin{table}[htbp]',r'\centering',r'\small',
        '\\caption{'+caption+'}', '\\label{'+label+'}',
        '\\begin{tabular}{'+spec+'}',r'\toprule',header+r' \\',r'\midrule',
        *[r+r' \\' for r in rows],r'\bottomrule',r'\end{tabular}',r'\end{table}',''])


def main():
    payload=json.loads((ROOT/'experiments/results/manuscript_benchmarks.json').read_text())
    results=payload['results']
    if len(results)!=20: raise ValueError('Require all ten datasets and both regimes')
    out=ROOT/'manuscript'; rows=[]
    for a in [r for r in results if r['regime']=='A']:
        b=next(r for r in results if r['dataset']==a['dataset'] and r['regime']=='B')
        rows.append(f"{a['dataset']} & {a['n']:,} & {a['p']} & {a['G']} & {'R' if a['task']=='regression' else 'C'} & {a['n_query']} & {b['n_query']} ({b['G_query']})")
    (out/'datasets_table.tex').write_text(table(
        r'Datasets and held-out samples. $p$ counts substantive predictors before one-hot encoding, excluding group identity and the intercept; $G$ counts groups. R: regression; C: classification. A and B give held-out rows; parentheses in B give held-out groups.',
        'tab:datasets',r'Dataset & $n$ & $p$ & $G$ & Task & A & B ($G$)',rows,'lrrrrrr'))
    for task in ['regression','classification']:
        for regime in ['A','B']:
            rows=[]
            for r in results:
                if r['task']!=task or r['regime']!=regime: continue
                for metric in (['rmse'] if task=='regression' else ['auc','brier']):
                    key=f"{r['dataset']} ({r['unit']})" if task=='regression' else f"{r['dataset']} & {metric.upper() if metric=='auc' else 'Brier'}"
                    rows.append(key+' & '+' & '.join(cell(r['records'],m,metric) for m in MODELS))
            text=('Regression RMSE in original outcome units (lower is better).' if task=='regression' else 'Classification AUROC (higher is better) and Brier score (lower is better).')
            text+=(' Regime A: observation holdout within known groups.' if regime=='A' else ' Regime B: entirely unseen groups, without labeled support.')
            text+=r' Neural entries are mean $\pm$ sample SD over three initialization seeds on one fixed split; other methods use one fit. These SDs are not confidence intervals.'
            header=('Dataset (unit)' if task=='regression' else 'Dataset & Metric')+r' & TabMixNN & Fixed NN & Mixed & XGBoost & RF'
            (out/f'{task}_{regime}_table.tex').write_text(table(text,f'tab:{task}_{regime}',header,rows,'l'*(2 if task=='classification' else 1)+'rrrrr'))
    rows=[]
    for r in results:
        if r['dataset']=='Sleepstudy':
            rows.append(r['regime']+' & '+cell(r['records'],'LinearVI','rmse')+' & '+cell(r['records'],'Mixed','rmse'))
    (out/'linear_check_table.tex').write_text(table(r'Sleepstudy linear-model check: raw RMSE (ms). Linear VI uses a scalar identity decoder and unstructured intercept/slope covariance; lme4 fits the same linear mean and random-effects structure by ML. Near agreement of predictions does not imply identical covariance estimation or REML equivalence.',
        'tab:linear_check','Regime & Linear VI & LMM (ML)',rows,'lrr'))
    ab=json.loads((ROOT/'experiments/results/manuscript_ablation.json').read_text())
    rows=[]
    for config in ['Intercept + slope','Intercept only','Fixed only','Linear decoder','Static','Dynamic','Gated static + dynamic']:
        rs=[dict(r,model=config) for r in ab['records'] if r['configuration']==config]
        if len(rs)!=3: raise ValueError(config)
        rows.append(config+' & '+cell(rs,config,'rmse')+' & '+cell(rs,config,'r2'))
    (out/'ablation_table.tex').write_text(table(r'Sleepstudy component comparisons on the same Regime-A split. RMSE is in milliseconds. Entries are mean $\pm$ sample SD over three seeds. Except for the scalar linear decoder, all variants use one hidden layer of width 32 and embedding dimension 8; all use the same 2,000-update budget. Structural variants use feature mixing without sparsity or contraction penalties. This is a component comparison, not a hyperparameter search.',
        'tab:ablation_comprehensive',r'Configuration & RMSE $\downarrow$ & $R^2$ $\uparrow$',rows,'lrr'))
    print('Generated seven tables from complete benchmark and ablation outputs.')

if __name__=='__main__': main()
