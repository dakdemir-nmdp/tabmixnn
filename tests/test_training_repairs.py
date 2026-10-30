"""Training lifecycle and covariance regressions using synthetic data only."""
import numpy as np
import pandas as pd
import pytest
import torch

from tabmixnn import TabMixNN
from tabmixnn.covariance.structures import ARMACovariance
from tabmixnn.utils.callbacks import Callback


def data(offset=300.):
    x = np.tile(np.linspace(-1, 1, 8), 4)
    return pd.DataFrame(dict(x=x, g=np.repeat(['a', 'b', 'c', 'd'], 8),
                             y=offset + 2*x + np.sin(np.arange(32))))


def model(**kwargs):
    return TabMixNN('y ~ x + (1|g)', family='gaussian', hidden_dims=[8],
                    embedding_dim=4, epochs=2, random_state=17, **kwargs)


class InitialState(Callback):
    def on_train_begin(self, logs=None):
        self.bias = self.trainer.heads['main'].bias.detach().clone()
        self.sigma = self.trainer.families['main'].sigma.item()


def test_gaussian_head_starts_at_training_mean_and_ignores_validation_labels():
    df = data()
    df.loc[3, 'y'] = np.nan
    val = data(1e6)
    callback = InitialState()
    model().fit(df, val_df=val, callbacks=[callback])
    assert callback.bias.item() == pytest.approx(df.y.mean(), rel=1e-6)
    assert callback.sigma == pytest.approx(df.y.std(ddof=0), rel=1e-6)


def test_repeated_fit_matches_fresh_fit_and_honors_new_learning_rate():
    first, second = data(), data(50.)
    second['g'] = 'new_' + second.g
    second['x'] += 20
    reused = model().fit(first).fit(second, lr=.02)
    fresh = model(lr=.02).fit(second)
    assert reused.optimizer.param_groups[0]['lr'] == .02
    assert len(reused.history) == 2
    assert set(reused.design.random_terms[0].level_map) == set(second.g)
    np.testing.assert_allclose(reused.predict(second), fresh.predict(second), atol=1e-6)


def test_validation_split_uses_positions_with_duplicate_index():
    df = data()
    df.index = np.zeros(len(df), dtype=int)
    fitted = model().fit(df, val_fraction=.25)
    assert len(fitted._train_data) == 24
    assert len(fitted.val_history) == 2


def test_arma_ar2_impulse_does_not_explode():
    cov = ARMACovariance(100, 1, p=2, q=0).double()
    with torch.no_grad():
        cov.phi_raw.fill_(np.arctanh(.9))
    impulse = torch.zeros(100, 1, 1, dtype=torch.double)
    impulse[0] = 1
    assert cov(impulse).abs().max() < 2


@pytest.mark.parametrize('p', [1, 2, 3, 5])
def test_arma_coefficients_have_stable_companion_roots(p):
    cov = ARMACovariance(10, 3, p=p, q=1).double()
    generator = torch.Generator().manual_seed(7)
    for _ in range(8):
        with torch.no_grad():
            cov.phi_raw.copy_(torch.randn(p, 3, 1, generator=generator, dtype=torch.double))
        coeff = cov.ar_coefficients.detach().numpy()[..., 0]
        for slope in range(3):
            roots = np.roots(np.r_[1., -coeff[:, slope]])
            assert np.abs(roots).max() < 1


def test_arma_transform_gradient_matches_finite_difference():
    cov = ARMACovariance(8, 1, p=3, q=1).double()
    raw = torch.tensor([.2, -.3, .4], dtype=torch.double, requires_grad=True).reshape(3, 1, 1)
    assert torch.autograd.gradcheck(cov._ar_from_partial_correlations, (raw,))


def test_arma_invalid_orders_fail_early():
    for p, q in [(-1, 0), (0, -1), (1.5, 0)]:
        with pytest.raises(ValueError, match='nonnegative integers'):
            ARMACovariance(3, 1, p=p, q=q)


def test_output_bias_is_not_penalized_but_matrix_weights_are():
    fitted = model(weight_decay=.1).fit(data())
    decays = {id(p): group['weight_decay'] for group in fitted.optimizer.param_groups
              for p in group['params']}
    assert decays[id(fitted.heads['main'].bias)] == 0
    assert decays[id(fitted.heads['main'].weight)] == .1


def test_arma_save_load_and_legacy_parameterization_guard(tmp_path):
    fitted = model(cov_struct_map={'g': {'type': 'ARMA', 'p': 2}}).fit(data())
    path = tmp_path / 'arma.pt'
    fitted.save(path)
    loaded = TabMixNN.load(path)
    np.testing.assert_allclose(fitted.predict(data()), loaded.predict(data()), atol=1e-6)
    saved = torch.load(path, weights_only=False)
    saved['format_version'] = 2
    torch.save(saved, path)
    with pytest.raises(ValueError, match='direct AR coefficients'):
        TabMixNN.load(path)


def test_non_arma_version2_checkpoint_still_loads(tmp_path):
    fitted = model().fit(data())
    path = tmp_path / 'iid.pt'
    fitted.save(path)
    saved = torch.load(path, weights_only=False)
    saved['format_version'] = 2
    saved.pop('initial_family_states')
    torch.save(saved, path)
    np.testing.assert_allclose(fitted.predict(data()), TabMixNN.load(path).predict(data()), atol=1e-6)


def test_arma_first_order_and_white_noise_limits():
    white = ARMACovariance(6, 1, p=0, q=0).double()
    noise = torch.arange(6., dtype=torch.double).reshape(6, 1, 1)
    torch.testing.assert_close(white(noise), noise)
    ar = ARMACovariance(6, 1, p=1, q=0).double()
    with torch.no_grad():
        ar.phi_raw.fill_(np.arctanh(.4))
    expected, previous = [], 0.
    for innovation in noise:
        previous = .4*previous + innovation
        expected.append(previous)
    torch.testing.assert_close(ar(noise), torch.stack(expected))


def test_arma_summary_reports_actual_ar_coefficients():
    fitted = model(cov_struct_map={'g': {'type': 'AR', 'p': 2}}).fit(data())
    tid = fitted.encoder.config[0]['term_id']
    cov = fitted.encoder.cov_modules[tid]
    summary = fitted.encoder.get_variance_summary()[tid]
    np.testing.assert_allclose(summary['phi'], cov.ar_coefficients.detach().numpy())
    assert summary['scale_kind'] == 'innovation'


def test_reuse_design_is_preprocessing_only():
    fitted = model().fit(data())
    design = fitted.design
    old_encoder = fitted.encoder
    fitted.fit(data(), reuse_design=design)
    assert fitted.design is design
    assert fitted.encoder is not old_encoder


def test_multitask_gaussian_initialization_uses_each_observed_mean():
    from tabmixnn import OutcomeSpec
    df = data()
    df['z'] = -20 + df.x
    df.loc[0, 'z'] = np.nan

    class Capture(Callback):
        def on_train_begin(self, logs=None):
            self.biases = {name: head.bias.item() for name, head in self.trainer.heads.items()}

    callback = Capture()
    TabMixNN('~ x', outcomes=[OutcomeSpec(name='y', y='y', family='gaussian'),
                              OutcomeSpec(name='z', y='z', family='gaussian')],
             hidden_dims=[8], epochs=1).fit(df, callbacks=[callback])
    assert callback.biases['y'] == pytest.approx(df.y.mean(), rel=1e-6)
    assert callback.biases['z'] == pytest.approx(df.z.mean(), rel=1e-6)
