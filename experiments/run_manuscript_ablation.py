"""Sleepstudy component comparisons under the same known-group split/protocol."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
import hashlib
import time
import numpy as np
import torch
from tabmixnn import disable_logging
from experiments.run_manuscript_benchmarks import DATASETS, within_group_holdout, prepare, make_model, metrics, SEEDS
from experiments.data_utils import load


def main():
    torch.set_num_threads(1); disable_logging()
    cfg=DATASETS[0]; df=load(cfg['loader'])
    tr,te=within_group_holdout(df,cfg['group'])
    train,query,fixed,mixed,ym,ys,_=prepare(cfg,df,tr,te)
    source=ROOT/'experiments/results/manuscript_benchmarks.json'
    benchmark=json.loads(source.read_text())
    base=next(r for r in benchmark['results'] if r['dataset']=='Sleepstudy' and r['regime']=='A')
    assert base['query_rows']==np.flatnonzero(te).tolist()
    assert benchmark['protocol']['epochs']==2000 and benchmark['protocol']['seeds']==SEEDS
    records=[dict(r,configuration={'TabMixNN':'Intercept + slope','FixedNN':'Fixed only','LinearVI':'Linear decoder'}[r['model']]) for r in base['records'] if r['model'] in ('TabMixNN','FixedNN','LinearVI')]
    for name,formula,structure in [('Intercept only',fixed+' + (1 | g)',None),
        ('Static',mixed,'static'),('Dynamic',mixed,'dynamic'),('Gated static + dynamic',mixed,'hybrid')]:
        for seed in SEEDS:
            start=time.perf_counter()
            m=make_model(formula,'regression',seed,2000,len(train),structure=structure).fit(train)
            pred=np.asarray(m.predict(query.drop(columns='y'))).reshape(-1)*ys+ym
            record=dict(configuration=name,seed=seed,metrics=metrics(query.y*ys+ym,pred,'regression'),
                        seconds=time.perf_counter()-start,training_loss_final=float(m.history[-1]))
            records.append(record); print(record,flush=True)
    out=dict(protocol=benchmark['protocol'],source_record_sha256=hashlib.sha256(json.dumps(base,sort_keys=True).encode()).hexdigest(),
             source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             train_rows=np.flatnonzero(tr).tolist(),query_rows=np.flatnonzero(te).tolist(),records=records)
    (ROOT/'experiments/results/manuscript_ablation.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')

if __name__=='__main__': main()
