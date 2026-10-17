"""Evaluation integrity checks using synthetic records only."""
import numpy as np
import pandas as pd
import torch
from experiments.run_manuscript_benchmarks import (
    within_group_holdout, prepare, metrics, make_model, DATASETS, FullBatchModel,
)
from experiments.splits import group_holdout
from tabmixnn import TabMixNN
from tabmixnn.core.model import TabMixDataset


def test_regimes_have_exact_group_semantics_and_reproducible_rows():
    df=pd.DataFrame({'g':np.repeat(np.arange(6),[1,2,3,4,5,6])})
    tr,te=within_group_holdout(df,'g')
    assert set(df.loc[te,'g']) <= set(df.loc[tr,'g'])
    assert tr[0] and not te[0]  # singleton cannot be a known-group test row
    assert np.all(tr ^ te)
    tr2,te2=within_group_holdout(df,'g')
    np.testing.assert_array_equal(te,te2)
    tr,te=group_holdout(df,'g')
    assert set(df.loc[te,'g']).isdisjoint(df.loc[tr,'g'])
    assert np.all(tr ^ te)


def test_query_labels_and_values_do_not_change_training_transform():
    cfg=dict(outcome='y',group='g',numeric=['x'],categorical=['c'],slope='x',task='regression')
    df=pd.DataFrame({'y':[1.,3.,5.,7.], 'x':[0.,2.,4.,6.], 'c':['a','b','a','b'], 'g':['a','b','a','b']})
    tr=np.array([True,True,False,False]); te=~tr
    first=prepare(cfg,df,tr,te)
    df.loc[te,'x']=1e6; df.loc[te,'y']=-1e6; df.loc[te,'c']='unseen'
    second=prepare(cfg,df,tr,te)
    pd.testing.assert_frame_equal(first[0],second[0])
    assert first[4:6] == second[4:6]
    np.testing.assert_array_equal(first[6].get_feature_names_out(),second[6].get_feature_names_out())
    assert np.isfinite(second[1].filter(like='x')).all().all()


def test_cached_full_batch_matches_package_objective_and_gradients():
    df=pd.DataFrame({'y':np.arange(12.)/10,'x':np.tile(np.arange(4.),3),'g':np.repeat(['a','b','c'],4)})
    model=make_model('y ~ x + (1+x|g)','regression',42,1,len(df)).fit(df)
    tensors=model.design.transform(df,mode='predict')
    ds=TabMixDataset(*tensors)
    cache,n=model._make_dataloader(ds,shuffle=False)
    standard,_=TabMixNN._make_dataloader(model,ds,shuffle=False)
    ordinary=next(iter(standard))  # materialize before seeding VI draws
    gradients=[]
    for batch in (cache[0],ordinary):
        model.train(); model.zero_grad(); torch.manual_seed(39)
        x,zi,zs,y=model._move_batch_to_device(batch)
        pred=model._ensure_dict_outputs(model.forward(x,zi,zs))
        loss=model._likelihood_loss(pred,y,n)+model.current_kl/n
        loss.backward()
        gradients.append([p.grad.clone() for p in model.parameters() if p.grad is not None])
    for a,b in zip(*gradients): torch.testing.assert_close(a,b)


def test_metric_units_and_probability_scores():
    result=metrics([0.,2.],[1.,3.],'regression')
    assert result['rmse']==1 and result['mae']==1
    prob=metrics([0.,1.],[.2,.8],'classification')
    assert prob['auc']==1 and np.isclose(prob['brier'],.04)


def test_matched_predictors_and_fixed_only_neural_control():
    ox=next(d for d in DATASETS if d['name']=='Oxboys')
    assert ox['numeric']==['age'] and ox['categorical']==[]
    df=pd.DataFrame({'y':[0.,1.,2.,3.], 'x0':[-1.,0.,1.,2.], 'g':['a','a','b','b']})
    model=make_model('y ~ x0','regression',42,1,len(df)).fit(df)
    assert len(model.encoder.config)==0
    assert len(model.encoder.variational_params)==0


def test_calibration_identity_and_signed_probability_bias():
    from experiments.summarize_split_sensitivity import calibration
    y=np.array([1,0,0,0,1,1,1,0.])
    p=np.array([.25]*4+[.75]*4)
    result=calibration(y,p)
    assert result['converged'] and abs(result['intercept'])<1e-6
    assert abs(result['slope']-1)<1e-6 and result['mean_probability_error']==0
    assert sum(b['n'] for b in result['bins'])==len(y)
