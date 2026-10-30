"""Public API and analytic likelihood contracts, using synthetic data only."""
import copy
import numpy as np
import pandas as pd
import pytest
import torch
from tabmixnn import (TabMixNN, OutcomeSpec, Gaussian, Binomial, Poisson,
    NegBinomial, Multinomial, MultivariateGaussian, MultiLabelBinomial,
    DiscreteTimeSurvival, DiscreteTimeCompetingRisks, ContinuousSurvival)


def data():
    return pd.DataFrame(dict(x=np.linspace(-1,1,12), g=np.repeat(['a','b','c'],4),
        y=np.linspace(1,3,12), binary=np.tile([0.,1.],6), count=np.tile([0.,1.,2.],4),
        time=np.tile([1.,2.,3.],4), event=np.tile([0.,1.,1.],4), cause=np.tile([0.,1.,2.],4)))


def model(spec):
    return TabMixNN('~ x + (1|g)', outcomes=[spec], hidden_dims=[8], embedding_dim=4,
                    epochs=2, batch_size=5, random_state=21, weight_decay=0)


@pytest.mark.parametrize('family,cols,extra,shape', [
    (Gaussian, 'y', {}, (12,)), (Binomial, 'binary', {}, (12,)),
    (Poisson, 'count', {}, (12,)), (NegBinomial, 'count', {}, (12,)),
    (lambda: Multinomial(3), 'count', {}, (12,)),
    (lambda: MultivariateGaussian(2), ['y','x'], {}, (12,2)),
    (lambda: MultiLabelBinomial(2), ['binary','event'], {}, (12,2)),
    (lambda: DiscreteTimeSurvival([1,2,3]), None, dict(time='time',event='event'), (12,3)),
    (lambda: DiscreteTimeCompetingRisks([1,2,3],2), None, dict(time='time',cause='cause'), (12,3,2)),
    (ContinuousSurvival, None, dict(time='time',event='event'), (12,)),
])
def test_all_ten_families_fit_predict_save_load(family, cols, extra, shape, tmp_path):
    m=model(OutcomeSpec(name='response',family=family(),y=cols,**extra)).fit(data())
    pred=m.predict(data()[['x','g']])
    assert pred.shape==shape and np.isfinite(pred).all()
    assert np.isfinite(m.history).all()
    m.batch_size=1
    np.testing.assert_allclose(pred,m.predict(data()),atol=2e-6)
    m.save(tmp_path/'model.pt')
    restored=TabMixNN.load(tmp_path/'model.pt')
    np.testing.assert_allclose(pred,restored.predict(data()),atol=2e-6)


def test_weibull_likelihood_and_curve_against_closed_form():
    family=ContinuousSurvival(init_log_shape=np.log(2.),init_log_scale=np.log(3.)).double()
    eta=torch.tensor([0.,np.log(2.)],dtype=torch.double,requires_grad=True)
    y=torch.tensor([[1.5,1.],[3.,0.]],dtype=torch.double)
    H=(y[:,0]/3).square()*eta.exp()
    h=2*y[:,0]/9*eta.exp()
    torch.testing.assert_close(family.nll(eta,y),(H-y[:,1]*h.log()).mean())
    times=np.array([0.,1.5,3.])
    expected=np.exp(-(times[None,:]/3)**2*np.exp(eta.detach().numpy()[:,None]))
    np.testing.assert_allclose(family.predict(eta,mode='survival',times=times),expected)
    med=family.predict(eta,mode='median')
    np.testing.assert_allclose(med,3*np.sqrt(np.log(2)/np.exp(eta.detach().numpy())))
    assert torch.autograd.gradcheck(lambda e:family.nll(e,y),(eta,))
    with pytest.raises(ValueError,match='times'): family.predict(eta,mode='survival')


def test_survival_times_reach_public_api_and_preserve_axes():
    m=model(OutcomeSpec(name='s',family=ContinuousSurvival(),time='time',event='event')).fit(data())
    times=[0.,1.,2.,5.]
    pred=m.predict(data(),mode='survival',times=times)
    assert pred.shape==(12,4) and (pred[:,0]==1).all() and (np.diff(pred,axis=1)<=0).all()
    m.batch_size=1
    np.testing.assert_allclose(pred,m.predict(data(),mode='survival',times=times),atol=1e-6)


def test_discrete_censoring_uses_only_completed_intervals_and_mass_is_conserved():
    df=data().iloc[:4].copy()
    df['time']=[.5,1.5,1.5,2.]; df['event']=[0.,0.,1.,1.]
    m=model(OutcomeSpec(name='s',family=DiscreteTimeSurvival([1,2,3]),time='time',event='event'))
    m.fit(df)
    y=m.design.transform(df,mode='fit')[3]['s']
    np.testing.assert_array_equal(y[:,0],[-1,0,1,1])
    eta=torch.zeros(4,3,dtype=torch.double,requires_grad=True)
    expected=np.log(2)*np.array([0,1,2,2]).mean()
    assert m.families['s'].nll(eta,torch.tensor(y)).item()==pytest.approx(expected)
    extreme=torch.tensor([[-100.,100.]],requires_grad=True)
    loss=DiscreteTimeSurvival([1,2]).nll(extreme,torch.tensor([[0.,1.]]))
    loss.backward()
    assert loss.item()==pytest.approx(100.) and extreme.grad[0,0].item()==pytest.approx(-1.)
    cr=DiscreteTimeCompetingRisks([1,2,3],2)
    logits=torch.zeros(4,3,2,dtype=torch.double)
    cause=torch.tensor([[-1.,0.],[0.,0.],[1.,1.],[1.,2.]])
    assert cr.nll(logits,cause).item()==pytest.approx(np.log(3)*np.array([0,1,2,2]).mean())
    np.testing.assert_allclose(cr.predict(logits).sum(2)+cr.predict(logits,mode='survival'),1.)


@pytest.mark.parametrize('family,col,value',[
    (Binomial,'binary',.5),(Poisson,'count',1.5),(NegBinomial,'count',-1),
    (lambda:Multinomial(3),'count',3),(lambda:Multinomial(3),'count',1.5),
    (Gaussian,'y',np.inf),
])
def test_invalid_observed_targets_rejected(family,col,value):
    df=data(); df.loc[0,col]=value
    with pytest.raises(ValueError): model(OutcomeSpec(name='r',family=family(),y=col)).fit(df)


def test_missing_targets_and_uncovered_followup_are_not_silent():
    m=model(OutcomeSpec(name='r',family=Gaussian(),y='y'))
    with pytest.raises(ValueError,match='Missing target'): m.fit(data().drop(columns='y'))
    with pytest.raises(ValueError,match='fully observed'): m.fit(data().assign(y=np.nan))
    with pytest.raises(ValueError,match='Missing target'): m.fit(data(),val_df=data().drop(columns='y'))
    with pytest.raises(ValueError,match='cover'): model(OutcomeSpec(name='s',family=DiscreteTimeSurvival([1,2]),time='time',event='event')).fit(data())
    with pytest.raises(ValueError,match='strictly increasing'): DiscreteTimeSurvival([2,1])
    with pytest.raises(ValueError,match='integers'):
        DiscreteTimeCompetingRisks([1,2],2).nll(torch.zeros(1,2,2),torch.tensor([[0.,1.5]]))


def test_prediction_ignores_query_targets_and_does_not_change_preprocessing():
    m=model(OutcomeSpec(name='r',family=Gaussian(),y='y')).fit(data())
    before=copy.deepcopy(m.design.numeric_transformer_params)
    np.testing.assert_array_equal(m.predict(data()),m.predict(data().assign(y='unused',extra=42.)))
    assert m.design.numeric_transformer_params==before
    with pytest.raises(ValueError,match='Unknown outcome'): m.predict(data(),outcome='typo')


def test_invalid_predictor_rows_cannot_be_silently_dropped():
    with pytest.raises(ValueError,match='finite'):
        model(OutcomeSpec(name='r',family=Gaussian(),y='y')).fit(data().assign(x=np.nan))


def test_ablation_names_reference_and_rng_state():
    m=model(OutcomeSpec(name='r',family=Gaussian(),y='y')).fit(data())
    m.train(); np.random.seed(19); expected=np.random.rand(4); np.random.seed(19)
    first=m.feature_ablation_values(data().iloc[:3],background_df=data(),background_size=12)
    np.testing.assert_array_equal(np.random.rand(4),expected)
    assert m.training and first['feature_names']==m.design.fixed_feature_names
    second=m.feature_ablation_values(data().iloc[:1],background_df=data(),background_size=12)
    np.testing.assert_allclose(first['ablation_values'][0],second['ablation_values'][0],atol=1e-6)


def test_latent_rescaling_preserves_predictions_and_elbo_not_covariance():
    m=model(OutcomeSpec(name='r',family=Gaussian(),y='y')).fit(data())
    X,zi,zs,Y=m.design.transform(data(),mode='fit')
    xt,zit,zst,yt=m._to_tensors(X,zi,zs,Y)
    m.train(); torch.manual_seed(812)
    original=m._ensure_dict_outputs(m(xt,zit,zst))
    old_loss=m._likelihood_loss(original,yt,len(X))+m.current_kl/len(X)
    old_sigma=m.encoder.cov_modules['term_0'].log_sigma.detach().exp().clone()
    with torch.no_grad():
        m.encoder.fixed_linear.weight.mul_(3.)
        m.encoder.cov_modules['term_0'].log_sigma.add_(np.log(3.))
        m.input_proj.weight.div_(3.)
    torch.manual_seed(812)
    new=m._ensure_dict_outputs(m(xt,zit,zst))
    new_loss=m._likelihood_loss(new,yt,len(X))+m.current_kl/len(X)
    torch.testing.assert_close(original['r'],new['r'])
    torch.testing.assert_close(old_loss,new_loss)
    torch.testing.assert_close(m.encoder.cov_modules['term_0'].log_sigma.exp().square(),9*old_sigma.square())


def test_gaussian_mean_field_gap_and_predictive_variance_counterexample():
    # Exact posterior precision P; optimal diagonal q has D_jj=1/P_jj.
    P=torch.tensor([[2.,1.],[1.,2.]],dtype=torch.double)
    exact=torch.linalg.inv(P); D=torch.diag(1/P.diag()); zero=torch.zeros(2,dtype=torch.double)
    kl=torch.distributions.kl_divergence(
        torch.distributions.MultivariateNormal(zero,D),
        torch.distributions.MultivariateNormal(zero,exact))
    torch.testing.assert_close(kl,.5*(P.diag().log().sum()-torch.logdet(P)))
    assert kl.item()==pytest.approx(.5*np.log(4/3))
    plus=torch.ones(2,dtype=torch.double); minus=torch.tensor([1.,-1.],dtype=torch.double)
    assert (plus@D@plus) > (plus@exact@plus)
    assert (minus@D@minus) < (minus@exact@minus)


def test_vector_ablation_never_sums_class_probabilities_into_constant():
    m=model(OutcomeSpec(name='c',family=Multinomial(3),y='count')).fit(data())
    with pytest.raises(NotImplementedError,match='scalar'): m.feature_ablation_values(data())
    m=TabMixNN('~ x + (1|g)',outcomes=[OutcomeSpec(name='a',family=Gaussian(),y='y'),
        OutcomeSpec(name='b',family=Binomial(),y='binary')],hidden_dims=[8],embedding_dim=4,epochs=1).fit(data())
    with pytest.raises(ValueError,match='Select outcome'): m.feature_ablation_values(data())
    a=m.feature_ablation_values(data(),outcome='b')
    np.testing.assert_allclose(a['predictions'],m.predict(data(),outcome='b',mode='proba'),atol=1e-6)
    terms=m.random_effect_ablations(data(),outcome='b')
    np.testing.assert_allclose(terms.prediction,m.predict(data(),outcome='b',mode='proba'),atol=1e-6)
