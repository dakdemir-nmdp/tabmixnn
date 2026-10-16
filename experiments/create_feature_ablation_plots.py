"""Rebuild Figures 1–3 from actual single-column ablations, not Shapley values.

Descriptive fit to all Sleepstudy observations. The background is all training
rows; random-effect indices and slope inputs are held fixed by the package API.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import hashlib
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tabmixnn import disable_logging
from experiments.run_manuscript_benchmarks import DATASETS, prepare, make_model
from experiments.data_utils import load, DATA_DIR


def main():
    torch.set_num_threads(1); disable_logging()
    df=load('sleepstudy'); allrows=np.ones(len(df),dtype=bool)
    train,_,_,formula,ym,ys,_=prepare(DATASETS[0],df,allrows,allrows)
    model=make_model(formula,'regression',42,2000,len(train)).fit(train)
    values=model.feature_ablation_values(train,background_size=len(train),seed=42)
    # Use authoritative compiled-design names; API display labels may be generic.
    j=model.design.design_info.column_names.index('x0')
    delta=values['ablation_values'][:,j]*ys
    assert np.isfinite(delta).all()
    # Independent check of the actual API perturbation and outcome rescaling.
    X,zi,zs,_=model.design.transform(train,mode='predict')
    xt,zit,zst,_=model._to_tensors(X,zi,zs,{})
    original=model._predict_from_tensors(xt,zit,zst)
    modified=xt.clone(); modified[:,j]=xt[:,j].mean()
    reference=(original-model._predict_from_tensors(modified,zit,zst)).numpy()*ys
    np.testing.assert_allclose(delta,reference,rtol=1e-4,atol=1e-3)
    # Anonymous plotting labels preserve group ordering, not source identifiers.
    group=pd.factorize(df.Subject,sort=False)[0]+1
    days=df.Days.to_numpy()
    slopes=[float(np.polyfit(days[group==g],delta[group==g],1)[0]) for g in range(1,19)]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    out=ROOT/'manuscript'
    hashes={}
    def save(fig,name):
        for ext in ['png','pdf']:
            path=out/f'{name}.{ext}'
            fig.savefig(path,dpi=220,bbox_inches='tight',metadata={'Creator':'TabMixNN reproducible ablation script'})
            hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(10,6),layout='constrained')
    for g,ax in enumerate(axes.flat,1):
        mask=group==g; x=days[mask]; y=delta[mask]; fit=np.polyfit(x,y,1)
        ax.scatter(x,y,s=22,color='#0072B2'); ax.plot(x,np.polyval(fit,x),color='#D55E00',lw=1.2)
        ax.axhline(0,color='.65',lw=.7); ax.set_title(f'Subject {g} | slope {fit[0]:.2f} ms/day')
        ax.set_xlabel('Days'); ax.set_ylabel('Feature ablation value (ms)')
    fig.suptitle('Feature Ablation Values: fixed-design Days column',fontsize=13)
    save(fig,'shap_days_effect_per_subject')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    order=np.argsort(slopes); axes[0].barh(np.arange(18),np.array(slopes)[order],color='#0072B2')
    axes[0].set_yticks(np.arange(18),[f'Subject {i+1}' for i in order]); axes[0].set_xlabel('Ablation slope (ms/day)')
    axes[0].set_title('Fixed-design Days ablation slopes')
    axes[1].hist(slopes,bins=6,color='#56B4E9',edgecolor='white')
    axes[1].set_xlabel('Ablation slope (ms/day)'); axes[1].set_ylabel('Number of subjects')
    axes[1].axvline(np.mean(slopes),color='#D55E00',label='Mean'); axes[1].legend(frameon=False)
    axes[1].set_title('Distribution across 18 subjects')
    save(fig,'shap_subject_effect_comparison')
    fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
    for g,ax in enumerate(axes.flat,1):
        mask=(group==g)&np.isin(days,[0,5,9]); x=days[mask]; y=delta[mask]
        assert x.tolist()==[0,5,9]
        ax.bar(np.arange(3),y,color=['#0072B2' if v>=0 else '#D55E00' for v in y])
        ax.set_xticks(np.arange(3),[f'Day {d}' for d in x]); ax.axhline(0,color='.5',lw=.7)
        ax.set_ylabel('Feature ablation value (ms)'); ax.set_title(f'Subject {g}: Ablation Breakdown')
    fig.suptitle('Days-column ablation at three observed time points',fontsize=13)
    save(fig,'shap_waterfall_subjects')
    meta=dict(source_sha256=hashlib.sha256((DATA_DIR/'sleepstudy.csv').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seed=42,epochs=2000,n_training=len(train),background_size=len(train),
        trained_on='all observations, descriptive only',unit='milliseconds',
        perturbation='fixed-design x0 (standardized Days); random slopes and group indices unchanged',
        example_days=[0,5,9],subject_labels='anonymous source-order numbers 1 through 18',
        slope_summary=dict(mean=float(np.mean(slopes)),minimum=min(slopes),maximum=max(slopes)),
        artifacts=hashes,api_perturbation_check='passed')
    (ROOT/'experiments/results/manuscript_figures.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))

if __name__=='__main__': main()
