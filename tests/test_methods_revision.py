"""Analytic regression tests for the methods revision; no downloaded data."""
import numpy as np
import pandas as pd
import pytest
import torch

from tabmixnn import TabMixNN, Gaussian, OutcomeSpec
from tabmixnn.core.encoder import MixedEffectsRepresentation
from tabmixnn.core.families import NegBinomial, Poisson, Binomial
from tabmixnn.covariance.structures import (
    KCovariance, GPCovariance, KroneckerCovariance,
    CompoundSymmetryCovariance, LowRankCovariance,
)
from tabmixnn.layers.noise import SPDEKernel


def toy():
    return pd.DataFrame({'x': np.tile(np.arange(4.), 3),
                         'g': np.repeat(['b', 'a', 'c'], 4),
                         'y': np.arange(12.) / 5})


def linear(df=None, **kw):
    return TabMixNN('y ~ x + (1|g)', family='gaussian', hidden_dims=[],
                    embedding_dim=1, epochs=1, verbose=False, random_state=7,
                    **kw).fit(toy() if df is None else df)


def test_nb2_matches_closed_form_mean_variance_and_logpmf():
    mu = torch.tensor([.3, 2., 8.], dtype=torch.float64)
    alpha = torch.tensor([.2, .7, 1.5], dtype=torch.float64)
    y = torch.tensor([0., 3., 10.], dtype=torch.float64)
    r = 1 / alpha
    expected = (torch.lgamma(y+r) - torch.lgamma(r) - torch.lgamma(y+1)
                + r*torch.log(r/(r+mu)) + y*torch.log(mu/(r+mu)))
    actual = NegBinomial().nll((mu.log(), alpha.log()), y)
    torch.testing.assert_close(actual, -expected.mean())
    dist = torch.distributions.NegativeBinomial(r, logits=(mu*alpha).log())
    torch.testing.assert_close(dist.mean, mu)
    torch.testing.assert_close(dist.variance, mu + alpha*mu.square())


def test_count_constants_and_singleton_binomial():
    eta, y = torch.tensor([.2, 1.4]), torch.tensor([0., 4.])
    expected = (eta.exp() - y*eta + torch.lgamma(y+1)).mean()
    torch.testing.assert_close(Poisson().nll(eta, y), expected)
    assert torch.isfinite(Binomial().nll(torch.zeros(1, 1), torch.ones(1, 1)))


def test_elbo_partition_and_missing_outcomes_value_and_gradient():
    model = TabMixNN('a ~ x', outcomes=[OutcomeSpec(name='a', y='a', family=Gaussian()),
                     OutcomeSpec(name='b', y='b', family=Gaussian())], epochs=1, verbose=False)
    model.fit(pd.DataFrame({'x':[0.,1.,2.,3.], 'a':[0.,1.,2.,3.],
                            'b':[2.,1.,0.,1.]}))
    pred = torch.tensor([[.2], [.4], [.1], [.9]], requires_grad=True)
    outputs = {'a':pred, 'b':pred*2}
    targets = {'a':torch.tensor([[1.],[float('nan')],[0.],[2.]]),
               'b':torch.tensor([[float('nan')],[3.],[1.],[float('nan')]])}
    kl = pred.square().sum()  # A shared global penalty, not one per row.
    whole = model._likelihood_loss(outputs, targets, 4) + kl/4
    parts = []
    for sl, size in [(slice(0,1),1),(slice(1,4),3)]:
        parts.append(size/4 * (model._likelihood_loss(
            {k:v[sl] for k,v in outputs.items()}, {k:v[sl] for k,v in targets.items()}, size)+kl/4))
    combined = sum(parts)
    torch.testing.assert_close(whole, combined)
    g1 = torch.autograd.grad(whole, pred, retain_graph=True)[0]
    g2 = torch.autograd.grad(combined, pred)[0]
    torch.testing.assert_close(g1, g2)


@pytest.mark.parametrize('ctype', ['IID', 'K', 'GP', 'KRON'])
def test_noncentered_kl_and_no_phantom_level(ctype):
    levels = ['b','a','c']
    spec = {'type':ctype, 'levels':levels, 'matrix_tensor':torch.eye(3),
            'K_group':torch.eye(3), 'coords':[[0.],[1.],[2.]]}
    enc = MixedEffectsRepresentation(1, 1, [{'term_id':'t', 'n_levels':3,
        'n_slopes':1, 'group':'g', 'train_levels':levels}], {'g':spec},
        use_fixed_bias=False)
    mu, logs = enc.variational_params['t']
    assert mu.shape == (3,1,1)
    with torch.no_grad():
        mu.copy_(torch.tensor([.2,.4,.8]).reshape(3,1,1)); logs.fill_(-.3)
    enc.eval()
    _, kl = enc(torch.zeros(3,1), [torch.tensor([1,2,3])], [torch.ones(3,1)])
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(mu,logs.exp()), torch.distributions.Normal(0.,1.)).sum()
    torch.testing.assert_close(kl, expected)


@pytest.mark.parametrize('kind', ['K','GP','KRON'])
def test_dense_covariance_transform_and_unseen_conditional_mean(kind):
    full, train = ['a','b','c'], ['c','a']
    K = torch.tensor([[2.,.3,.2],[.3,1.5,.4],[.2,.4,1.]])
    kw = dict(n_levels=3,n_slopes=2,full_levels=full,train_levels=train)
    if kind == 'K':
        cov = KCovariance(**kw,matrix_tensor=K)
    elif kind == 'GP':
        cov = GPCovariance(**kw,coords=[[0.],[1.],[2.]])
        K = cov._kernel_full(cov.log_lengthscale)[0]
    else:
        S = torch.tensor([[1.5,.2],[.2,.7]])
        cov = KroneckerCovariance(**kw,K_group=K,K_slope=S)
    v = torch.tensor([[[.1],[.3]],[[.5],[.7]]])
    idx = torch.tensor([2,0])
    Ktt = K[idx][:,idx] + (0 if kind == 'GP' else 1e-4)*torch.eye(2)
    expected_train = torch.linalg.cholesky(Ktt) @ v[:,:,0]
    if kind == 'KRON':
        expected_train = expected_train @ torch.linalg.cholesky(S).T
    result = cov(v)
    torch.testing.assert_close(result[idx,:,0],expected_train,atol=1e-6,rtol=1e-5)
    expected_new = K[1,idx] @ torch.linalg.solve(Ktt,expected_train)
    torch.testing.assert_close(result[1,:,0],expected_new,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(cov(v,torch.tensor([0,3]))[0],torch.zeros(2,1))
    result.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in cov.parameters())


@pytest.mark.parametrize('raw', [-30.,0.,30.])
def test_compound_symmetry_stays_psd(raw):
    cov = CompoundSymmetryCovariance(8,1)
    with torch.no_grad(): cov.raw_rho.fill_(raw)
    rho = cov.correlation.item()
    assert -1/7 < rho < 1
    assert torch.isfinite(cov(torch.eye(8).reshape(8,1,8))).all()


def test_dense_spde_samples_inverse_precision():
    torch.manual_seed(3)
    cov = SPDEKernel((2,2),kappa_init=1.2,tau_init=1.7)
    with torch.no_grad():
        draws = cov.sample(50000)
        empirical = torch.cov(draws.T)
        expected = torch.linalg.inv(cov.get_precision())
    torch.testing.assert_close(empirical,expected,rtol=.035,atol=.002)


def test_formula_intercept_slopes_and_centering_in_effect_space():
    df = toy()
    m = TabMixNN('y ~ 0 + x + (1+x|g)', family='gaussian', hidden_dims=[],
        embedding_dim=1, epochs=1, enforce_centering=True, verbose=False).fit(df)
    X, inds, slopes, _ = m.design.transform(df, mode='predict')
    assert X.shape[1] == 1 and m.encoder.fixed_linear.bias is None
    np.testing.assert_allclose(slopes[0][:,1],df.x)
    mu = m.encoder.variational_params['term_0'][0]
    with torch.no_grad(): mu.normal_()
    values = m.encoder.random_effects_values('term_0',mu)
    torch.testing.assert_close(values.mean(0),torch.zeros_like(values[0]),atol=1e-6,rtol=0)
    extracted = m.extract_parameters(df)['random_effects']
    assert set(extracted['kind']) == {'linear_coefficient'}
    assert len(extracted) == 6


def test_conditional_map_matches_gaussian_solution_without_query_leakage():
    m = linear()
    with torch.no_grad():
        m.encoder.fixed_linear.weight.zero_()
        m.encoder.cov_modules['term_0'].log_sigma.fill_(np.log(2.))
        m.data_family('main').log_sigma.fill_(0.)
    support = pd.DataFrame({'x':[0.,0.,0.], 'g':['new']*3,'y':[2.,3.,4.]})
    query = pd.DataFrame({'x':[0.,1.], 'g':['new','absent']})
    m.train()
    state = {k:v.clone() for k,v in m.state_dict().items()}
    flags = [p.requires_grad for p in m.parameters()]
    override = m.adapt_unseen_groups(support,query_df=query,n_steps=400,lr=.05)
    assert m.training and [p.requires_grad for p in m.parameters()] == flags
    for k,v in m.state_dict().items(): torch.testing.assert_close(v,state[k])
    # b_hat = sum(y)/(n + sigma_e^2/sigma_b^2).
    assert override['term_0'][0,0,0].item() == pytest.approx(9/3.25,abs=2e-4)
    assert override['term_0'][1,0,0].item() == 0
    p1 = m.predict_transductive(query,support_df=support,n_steps=400,lr=.05)
    p2 = m.predict_transductive(query.assign(y=1e9),support_df=support,n_steps=400,lr=.05)
    np.testing.assert_array_equal(p1,p2)
    with pytest.raises(ValueError,match='support_df'):
        m.predict_transductive(query)


def test_prediction_partition_and_nondefault_save_load(tmp_path):
    m = linear(enforce_centering=True,cov_struct_map={'g':{'type':'UN'}},batch_size=5)
    pred = m.predict(toy())
    m.batch_size = 1
    np.testing.assert_allclose(pred,m.predict(toy()),atol=1e-6)
    path = tmp_path/'model.pt'
    m.save(path)
    restored = TabMixNN.load(path)
    np.testing.assert_allclose(restored.predict(toy()),pred,atol=1e-6)
    assert restored.enforce_centering and restored.cov_struct_map == m.cov_struct_map
    assert restored.AIC is None if hasattr(restored,'AIC') else True


@pytest.mark.parametrize('kind', ['K','GP','KRON'])
def test_full_level_order_training_subset_and_centering(kind):
    spec = {'type':kind, 'levels':['c','new','a','b'],
            'matrix_tensor':torch.eye(4), 'K_group':torch.eye(4),
            'coords':[[0.],[1.],[2.],[3.]]}
    m = linear(cov_struct_map={'g':spec},enforce_centering=True)
    enc = m.encoder
    assert enc.variational_params['term_0'][0].shape == (3,1,1)
    cov = enc.cov_modules['term_0']
    assert cov.train_levels == ['c','a','b']
    with torch.no_grad(): enc.variational_params['term_0'][0].copy_(torch.tensor([1.,2.,4.]).reshape(3,1,1))
    values = enc.random_effects_values('term_0',enc.variational_params['term_0'][0])
    torch.testing.assert_close(values[cov.train_idx].mean(0),torch.zeros(1,1),atol=1e-6,rtol=0)
    q = pd.DataFrame({'x':[0.,0.,0.],'g':['b','new','unknown']})
    pred = m.predict(q)
    separate = np.concatenate([m.predict(q.iloc[[i]]) for i in range(len(q))])
    np.testing.assert_allclose(pred,separate,atol=1e-6)


def test_linearized_effect_does_not_shrink_with_sample_count_or_modify_gradients():
    from tabmixnn.utils.summary import _linearized_fixed_effects
    m = linear()
    with torch.no_grad(): m.encoder.fixed_linear.weight.copy_(torch.tensor([[2.,3.]]))
    for p in m.parameters(): p.grad = torch.ones_like(p)
    m.train()
    first = _linearized_fixed_effects(m,toy(),m.design.fixed_feature_names,0)
    second = _linearized_fixed_effects(m,pd.concat([toy(),toy()]),m.design.fixed_feature_names,0)
    np.testing.assert_allclose(first.Estimate,[2.,3.],atol=1e-6)
    np.testing.assert_allclose(second.Estimate,first.Estimate,atol=1e-6)
    assert m.training and all(torch.equal(p.grad,torch.ones_like(p)) for p in m.parameters())


def test_training_overrides_are_honored():
    m = TabMixNN.lmm('y ~ x + (1|g)',epochs=10,verbose=False)
    m.fit(toy(),epochs=2,batch_size=4,lr=.02)
    assert len(m.history) == 2 and m.batch_size == 4 and m.lr == .02
    with pytest.raises(TypeError,match='Unsupported fit'):
        m.fit(toy(),epoch=99)


def test_vector_predictions_preserve_row_and_time_axes():
    from tabmixnn.core.families import DiscreteTimeSurvival
    df = toy().assign(time=np.tile([0,1,2,2],3),event=np.tile([1,1,0,1],3))
    m = TabMixNN('time ~ x',outcomes=[OutcomeSpec(name='s',family=DiscreteTimeSurvival([0,1,2]),
                    time='time',event='event')],hidden_dims=[],epochs=1,batch_size=5,verbose=False).fit(df)
    pred = m.predict(df,mode='survival')
    assert pred.shape == (len(df),3)
    assert (np.diff(pred,axis=1) <= 0).all()
    m.batch_size = 1
    np.testing.assert_allclose(pred,m.predict(df,mode='survival'),atol=1e-6)


def test_spde_factory_and_single_slope_low_rank():
    from tabmixnn.layers.noise import create_spde_kernel, KroneckerSPDEKernel
    assert isinstance(create_spde_kernel((2,2),use_kronecker=False),SPDEKernel)
    assert isinstance(create_spde_kernel((2,2),use_kronecker=True),KroneckerSPDEKernel)
    cov = LowRankCovariance(3,1,rank=2)
    assert cov(torch.ones(3,1,4)).shape == (3,1,4)


def test_diagnostic_rejects_nonconverged_reference_and_restores_mode():
    from tabmixnn.layers.structural import CompositeStructuralModel
    class Expanding(torch.nn.Module):
        def forward(self,x): return 2*x
        def solve(self,x): return -x
    model = CompositeStructuralModel(2,use_dynamic=False)
    model.B_static = Expanding()
    model.train()
    with pytest.raises(RuntimeError,match='did not converge'):
        model.approximation_error(torch.ones(3,2),hybrid_max_iter=5)
    assert model.training


def test_fixed_covariance_linear_mean_satisfies_henderson_equations():
    df = toy()
    m = linear(weight_decay=0)
    X, inds, slopes, targets = m.design.transform(df,mode='fit')
    Z = np.eye(3)[inds[0]-1]
    sigma_b, sigma_e = 1.2, .7
    A = np.column_stack([X,Z])
    penalty = np.diag([0.,0.,1/sigma_b**2,1/sigma_b**2,1/sigma_b**2])
    solution = np.linalg.solve(A.T @ A/sigma_e**2+penalty,A.T @ df.y.to_numpy()/sigma_e**2)
    with torch.no_grad():
        m.encoder.fixed_linear.weight.copy_(torch.tensor(solution[:2]).reshape(1,2))
        m.encoder.variational_params['term_0'][0].copy_(torch.tensor(solution[2:]/sigma_b).reshape(3,1,1))
        m.encoder.cov_modules['term_0'].log_sigma.fill_(np.log(sigma_b))
        m.data_family('main').log_sigma.fill_(np.log(sigma_e))
    m.eval()
    xt,zi,zs,yt = m._to_tensors(X,inds,slopes,targets)
    outputs = m._ensure_dict_outputs(m.forward(xt,zi,zs))
    _, kl = m.encoder(xt,zi,zs)
    loss = m._likelihood_loss(outputs,yt,len(df)) + kl/len(df)
    grads = torch.autograd.grad(loss,[m.encoder.fixed_linear.weight,m.encoder.variational_params['term_0'][0]])
    for grad in grads:
        torch.testing.assert_close(grad,torch.zeros_like(grad),atol=2e-6,rtol=0)
    np.testing.assert_allclose(m.predict(df),A @ solution,atol=1e-6)


def test_validation_labels_do_not_change_fitted_weights_without_selection():
    df = toy()
    validation_rows = df.sample(frac=.25,random_state=7).index
    altered = df.copy()
    altered.loc[validation_rows,'y'] = 1e6
    def fit(frame):
        return TabMixNN.lmm('y ~ x + (1|g)',epochs=3,random_state=7,verbose=False).fit(frame,val_fraction=.25)
    first, second = fit(df),fit(altered)
    for key,value in first.state_dict().items():
        torch.testing.assert_close(value,second.state_dict()[key],atol=0,rtol=0)
    assert len(first._train_data) == 9


def test_training_loop_uses_global_kl_over_n_for_every_batch(monkeypatch):
    m = linear(batch_size=5)
    mu, logs = m.encoder.variational_params['term_0']
    with torch.no_grad(): mu.fill_(1.); logs.zero_()
    captured = []
    class CaptureOptimizer:
        def zero_grad(self): m.zero_grad()
        def step(self): captured.append(mu.grad.clone())
    m.optimizer = CaptureOptimizer()
    monkeypatch.setattr(m,'_likelihood_loss',lambda outputs,targets,n:outputs['main'].sum()*0)
    X, zi, zs, y = m.design.transform(toy(),mode='fit')
    m._train_loop(m._to_tensors(X,zi,zs,y),None)
    assert len(captured) == 3  # Batch sizes 5, 5, 2.
    for grad in captured:
        torch.testing.assert_close(grad,torch.full_like(mu,1/12))
    assert m.history[-1] == pytest.approx(1.5/12)


def test_composite_is_not_one_iteration_even_with_zero_operators():
    from tabmixnn.layers.structural import CompositeStructuralModel
    class ZeroAttention(torch.nn.Module):
        def forward(self,x,return_weights=False): return (torch.zeros_like(x),None)
    model = CompositeStructuralModel(4,use_static=False,use_dynamic=False,
                                    learnable_fusion=False,fusion_alpha=.5)
    model.use_dynamic = True
    model.B_dynamic = ZeroAttention()
    # Coupled fixed point is xi. Composite is .5*xi: a 50% difference.
    assert model.approximation_error(torch.ones(3,4)) == pytest.approx(.5)


@pytest.mark.parametrize('kind', ['K','GP','KRON'])
def test_related_level_covariance_roundtrip(kind,tmp_path):
    spec = {'type':kind,'levels':['b','new','a','c'],
            'matrix_tensor':torch.eye(4),'K_group':torch.eye(4),
            'coords':[[0.],[1.],[2.],[3.]]}
    model = linear(cov_struct_map={'g':spec},enforce_centering=True)
    query = pd.DataFrame({'x':[0.,1.,2.],'g':['b','new','absent']})
    expected = model.predict(query)
    model.save(tmp_path/'related.pt')
    loaded = TabMixNN.load(tmp_path/'related.pt')
    np.testing.assert_allclose(loaded.predict(query),expected,atol=1e-6)


def test_intercept_only_persistence(tmp_path):
    model = TabMixNN.lmm('y ~ 1 + (1|g)',epochs=2,verbose=False).fit(toy())
    expected = model.predict(toy())
    model.save(tmp_path/'intercept.pt')
    loaded = TabMixNN.load(tmp_path/'intercept.pt')
    np.testing.assert_allclose(loaded.predict(toy()),expected,atol=1e-6)


def test_all_missing_minibatch_has_zero_likelihood_gradient():
    df = toy()
    df.loc[[0,2,5,7],'y'] = np.nan
    model = TabMixNN.lmm('y ~ x',epochs=2,batch_size=1,verbose=False).fit(df)
    pred = torch.tensor([[.7]],requires_grad=True)
    loss = model._likelihood_loss({'main':pred},{'main':torch.tensor([[float('nan')]])},1)
    assert loss.item() == 0
    loss.backward()
    assert pred.grad.item() == 0
    assert np.isfinite(model.predict(df)).all()


def test_dag_penalty_is_zero_and_sparsity_penalty_is_not():
    """
    dag_penalty must be identically zero for any B values because B is strictly
    lower-triangular by construction (tril_mask zeros diagonal and upper triangle),
    making it a DAG unconditionally.  The old NOTEARS formula trace(exp(B*B))-dim
    was also zero for the same reason: diag(B*B)=0 always.

    sparsity_penalty must be positive whenever B has nonzero entries, confirming
    it is the only active structural regulariser.
    """
    from tabmixnn.layers.structural import StaticStructureModel
    model = StaticStructureModel(4, sparse_penalty=1.0)
    # Fill B_raw with nonzero values so any non-trivial penalty would fire.
    with torch.no_grad():
        model.B_raw.fill_(2.0)
    assert model.dag_penalty().item() == 0.0
    assert model.sparsity_penalty().item() > 0.0
    # Confirm B is strictly lower-triangular regardless of B_raw values.
    B = model.B
    assert torch.equal(B, torch.tril(B, diagonal=-1))


def test_lambda_dag_removed_from_model():
    """lambda_dag must not appear as a constructor parameter or stored attribute."""
    import inspect
    sig = inspect.signature(TabMixNN.__init__)
    assert 'lambda_dag' not in sig.parameters
    m = TabMixNN.lmm('y ~ x', epochs=1, verbose=False)
    assert not hasattr(m, 'lambda_dag')


def test_methods_revision_contains_no_reml_equivalence_claim():
    """
    The methods revision doc must not contain phrases that assert or imply
    REML equivalence.  Permitted: explaining what REML is and why this
    implementation differs.  Forbidden: any claim that VI recovers, equals,
    approximates, or is equivalent to REML.
    """
    import pathlib, re
    doc = pathlib.Path(__file__).parent.parent / 'docs' / 'methods-revision-2026-10-15.md'
    text = doc.read_text().lower()
    # Strip lines that enumerate claims to remove (they list forbidden phrases
    # as targets, not assertions).
    lines = [l for l in text.splitlines()
             if not re.search(r'remove or qualify claims of|claims to remove', l)]
    body = '\n'.join(lines)
    forbidden = [
        'recovers reml',
        'equivalent to reml',
        'reml equivalent',
        'approximates reml',
        'reml approximation',
        r'equal[^s].*reml',
        r'reml.*equal[^s]',
        'reml estimate',   # "REML estimates" as a claim this produces them
    ]
    hits = [p for p in forbidden if re.search(p, body)]
    assert not hits, f"Forbidden REML-equivalence phrases found: {hits}"
