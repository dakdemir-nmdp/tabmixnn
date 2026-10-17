import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..utils.logging_config import get_logger

class Family(nn.Module):
    """Base class for outcome families. Now inherits from nn.Module to support learnable parameters."""
    def __init__(self):
        super().__init__()

    @property
    def n_params(self):
        raise NotImplementedError

    @property
    def primary_param_index(self) -> int:
        """
        Index of the primary parameter that participates in output-level SEM.
        For multitask learning, only primary parameters (location/mean) are correlated;
        auxiliary parameters (dispersion, shape) remain independent.
        Default: 0 (first parameter).
        """
        return 0

    @property
    def primary_param_name(self) -> str:
        """
        Semantic name of the primary parameter for documentation and interpretation.
        Must be implemented by each family.

        Examples:
            - Gaussian: "mean"
            - NegBinomial: "log_mean"
            - Binomial: "log_odds"
            - ContinuousSurvival: "log_hazard_ratio"
        """
        raise NotImplementedError

    def forward_params(self, eta):
        """Transform raw neural output eta to distribution parameters theta."""
        raise NotImplementedError

    def nll(self, theta, y, **kwargs):
        """Compute Negative Log Likelihood."""
        raise NotImplementedError

    def predict(self, theta, mode="default", **kwargs):
        raise NotImplementedError

    def validate_targets(self, df, target_cols):
        """Validate observed targets before optimization; NaNs denote missingness."""
        expected = (self.n_tasks if isinstance(self, MultivariateGaussian) else
                    self.n_labels if isinstance(self, MultiLabelBinomial) else
                    2 if isinstance(self, (DiscreteTimeSurvival, DiscreteTimeCompetingRisks,
                                          ContinuousSurvival)) else 1)
        if len(target_cols) != expected:
            raise ValueError(f"{type(self).__name__} requires {expected} target columns")
        try:
            y = df[target_cols].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Targets must be numeric; encode class labels explicitly") from exc
        if np.isinf(y).any():
            raise ValueError("Targets must be finite or NaN for missing values")
        observed = y[~np.isnan(y).any(axis=1)]
        if not len(observed):
            raise ValueError(f"{type(self).__name__} needs at least one fully observed target row")
        if isinstance(self, (Binomial, MultiLabelBinomial)):
            if not np.isin(observed, [0, 1]).all():
                raise ValueError("Binary targets must be 0 or 1")
        elif isinstance(self, (Poisson, NegBinomial, Multinomial)):
            if (observed < 0).any() or (observed != np.floor(observed)).any():
                raise ValueError("Count/class targets must be nonnegative integers")
            if isinstance(self, Multinomial) and (observed >= self.n_classes).any():
                raise ValueError("Class targets must be smaller than n_classes")
        elif isinstance(self, (DiscreteTimeSurvival, DiscreteTimeCompetingRisks, ContinuousSurvival)):
            times, labels = observed.T
            upper = self.num_causes if isinstance(self, DiscreteTimeCompetingRisks) else 1
            if (labels < 0).any() or (labels > upper).any() or (labels != np.floor(labels)).any():
                raise ValueError(f"Event/cause targets must be integers from 0 to {upper}")
            if (times < 0).any() or (isinstance(self, ContinuousSurvival) and (times == 0).any()):
                raise ValueError("Survival times must be positive (nonnegative for discrete time)")
            if hasattr(self, 'time_bins') and (times > self.time_bins[-1]).any():
                raise ValueError("time_bins must cover all observed follow-up; no silent clipping")


def _time_grid(time_bins):
    grid = np.asarray(time_bins, dtype=float)
    if (grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all()
            or (grid < 0).any() or (np.diff(grid) <= 0).any()):
        raise ValueError("time_bins must be finite, nonnegative and strictly increasing")
    return grid


def _discrete_targets(y, n_times, n_causes):
    if y.ndim != 2 or y.shape[1] != 2 or not torch.isfinite(y).all():
        raise ValueError("Discrete survival targets must be finite [time_idx, event/cause] rows")
    if not torch.equal(y, y.floor()):
        raise ValueError("Discrete survival indices and event/cause labels must be integers")
    idx, cause = y[:, 0].long(), y[:, 1].long()
    if ((idx < -1) | (idx >= n_times) | (cause < 0) | (cause > n_causes)
            | ((idx == -1) & (cause != 0))).any():
        raise ValueError("Invalid discrete survival index or event/cause label")
    return idx, cause

class Gaussian(Family):
    def __init__(self, link="identity", init_log_sigma=0.0):
        """
        Gaussian family with global learned residual variance.

        Args:
            link: Link function (only 'identity' supported)
            init_log_sigma: Initial value for log(sigma). Default 0.0 -> sigma=1.0
        """
        super().__init__()
        self.link = link
        # Global learned residual standard deviation (more stable than per-sample)
        self.log_sigma = nn.Parameter(torch.tensor(init_log_sigma))

    @property
    def n_params(self):
        # Only predict mean μ; σ is global and learned separately
        return 1

    @property
    def primary_param_name(self) -> str:
        return "mean"

    @property
    def sigma(self):
        # Convenience accessor for std deviation
        return torch.exp(self.log_sigma)

    def forward_params(self, eta):
        # eta is [batch, 1] -> mu only
        mu = eta.squeeze(-1)
        return mu

    def nll(self, theta, y, **kwargs):
        # theta is just mu (1D tensor)
        mu = theta.squeeze() if isinstance(theta, torch.Tensor) else theta
        y = y.squeeze()

        # Use global learned sigma (clamped for numerical stability)
        from ..utils.constants import MIN_LOG_SIGMA, MAX_LOG_SIGMA, EPSILON
        log_sigma = torch.clamp(self.log_sigma, min=MIN_LOG_SIGMA, max=MAX_LOG_SIGMA)
        sigma = torch.exp(log_sigma) + EPSILON

        return -torch.distributions.Normal(mu, sigma).log_prob(y).mean()

    def predict(self, theta, mode="default", **kwargs):
        # theta is just mu
        mu = theta.squeeze() if isinstance(theta, torch.Tensor) else theta
        return mu.cpu().numpy() if isinstance(mu, torch.Tensor) else mu

class Binomial(Family):
    def __init__(self, link="logit"):
        super().__init__()
        self.link = link

    @property
    def n_params(self):
        return 1

    @property
    def primary_param_name(self) -> str:
        return "log_odds"

    def forward_params(self, eta):
        return eta.squeeze(-1) # Logits

    def nll(self, theta, y, **kwargs):
        # theta is logits
        # y is 0/1
        return F.binary_cross_entropy_with_logits(theta.reshape(-1), y.float().reshape(-1), reduction='mean')

    def predict(self, theta, mode="default", **kwargs):
        probs = torch.sigmoid(theta)
        if mode == "proba":
            return probs.cpu().numpy()
        return (probs > 0.5).long().cpu().numpy()

    def predict_proba(self, theta, **kwargs):
        return torch.sigmoid(theta).cpu().numpy()

class Poisson(Family):
    def __init__(self):
        super().__init__()

    @property
    def n_params(self):
        return 1

    @property
    def primary_param_name(self) -> str:
        return "log_rate"

    def forward_params(self, eta):
        # eta is log_rate
        return eta.squeeze(-1)

    def nll(self, theta, y, **kwargs):
        # theta is log_rate
        return -torch.distributions.Poisson(theta.reshape(-1).exp()).log_prob(y.reshape(-1)).mean()

    def predict(self, theta, mode="default", **kwargs):
        return torch.exp(theta).cpu().numpy()

class NegBinomial(Family):
    def __init__(self):
        super().__init__()
        from ..utils.logging_config import get_logger
        self.logger = get_logger("families")

    @property
    def n_params(self):
        return 2 # log_mu, log_alpha

    @property
    def primary_param_name(self) -> str:
        return "log_mean"

    def forward_params(self, eta):
        return eta[:, 0], eta[:, 1]

    def nll(self, theta, y, **kwargs):
        # Based on NB2 parameterization
        # mu = exp(eta_mu), alpha = exp(eta_alpha)
        # Var = mu + alpha * mu^2
        if isinstance(theta, (tuple, list)):
            log_mu, log_alpha = theta
        else:
            log_mu = theta[:, 0]
            log_alpha = theta[:, 1]
        # Clamp to avoid overflow/underflow and keep gradients flowing.
        from ..utils.constants import MIN_LOG_PARAM, MAX_LOG_PARAM, LARGE_PENALTY, EPSILON
        log_mu = torch.clamp(log_mu, min=MIN_LOG_PARAM, max=MAX_LOG_PARAM)
        log_alpha = torch.clamp(log_alpha, min=MIN_LOG_PARAM, max=MAX_LOG_PARAM)

        mu = torch.exp(log_mu) + EPSILON
        alpha = torch.exp(log_alpha) + EPSILON

        # PyTorch counts successes before r failures: mean = r*exp(logits).
        # NB2 therefore has r=1/alpha and logits=log(mu*alpha).
        dist = torch.distributions.NegativeBinomial(
            total_count=torch.exp(-log_alpha), logits=log_mu + log_alpha)
        return -dist.log_prob(y.reshape(-1)).mean()

    def predict(self, theta, mode="default", **kwargs):
        if isinstance(theta, (tuple, list)):
            log_mu = theta[0]
        else:
            log_mu = theta[:, 0]
        return torch.exp(log_mu).cpu().numpy()

class DiscreteTimeSurvival(Family):
    def __init__(self, time_bins):
        super().__init__()
        self.time_bins = _time_grid(time_bins)
        self._warned_invalid = False
        self.logger = get_logger("families.discrete_survival")

    @property
    def n_params(self):
        return len(self.time_bins)

    @property
    def primary_param_name(self) -> str:
        return "logit_hazard_t0"

    def forward_params(self, eta):
        # eta: [n, T] logits for hazard
        return eta

    def nll(self, theta, y, **kwargs):
        idx, event = _discrete_targets(y, theta.shape[1], 1)
        interval = torch.arange(theta.shape[1], device=theta.device)[None, :]
        before = interval < idx[:, None]
        at = interval == idx[:, None]
        # log-sigmoid avoids clipping probabilities and preserves tail gradients.
        ll = (before * F.logsigmoid(-theta)).sum(dim=1)
        ll = ll + (at * (event[:, None] * F.logsigmoid(theta)
                        + (1-event[:, None]) * F.logsigmoid(-theta))).sum(dim=1)
        return -ll.mean()

    def predict(self, theta, mode="default", **kwargs):
        if mode in ("default", "survival"):
            return torch.exp(F.logsigmoid(-theta).cumsum(dim=1)).detach().cpu().numpy()
        if mode == "hazard":
            return torch.sigmoid(theta).detach().cpu().numpy()
        raise ValueError("DiscreteTimeSurvival mode must be survival or hazard")


class DiscreteTimeCompetingRisks(Family):
    def __init__(self, time_bins, num_causes):
        super().__init__()
        self.time_bins = _time_grid(time_bins)
        if not isinstance(num_causes, (int, np.integer)) or num_causes < 1:
            raise ValueError("num_causes must be a positive integer")
        self.num_causes = num_causes # K
        self._warned_invalid = False
        self.logger = get_logger("families.competing_risks")

    @property
    def n_params(self):
        # logits for each cause at each time
        return len(self.time_bins) * self.num_causes

    @property
    def primary_param_name(self) -> str:
        return "logit_hazard_t0_cause1"

    def forward_params(self, eta):
        # reshape eta to [batch, T, K]
        # Blueprint 9.2: eta_{itk}
        n_bins = len(self.time_bins)
        return eta.reshape(-1, n_bins, self.num_causes)

    def nll(self, theta, y, **kwargs):
        idx, cause = _discrete_targets(y, theta.shape[1], self.num_causes)
        logits = torch.cat([theta.new_zeros(*theta.shape[:2], 1), theta], dim=2)
        logp = F.log_softmax(logits, dim=2)
        interval = torch.arange(theta.shape[1], device=theta.device)[None, :]
        before = interval < idx[:, None]
        at = interval == idx[:, None]
        selected = logp.gather(2, cause[:, None, None].expand(-1, theta.shape[1], 1)).squeeze(2)
        return -((before * logp[:, :, 0]).sum(1) + (at * selected).sum(1)).mean()

    def predict(self, theta, mode="default", **kwargs):
        logits = torch.cat([theta.new_zeros(*theta.shape[:2], 1), theta], dim=2)
        probs = F.softmax(logits, dim=2)
        survival = probs[:, :, 0].cumprod(dim=1)
        if mode == "survival":
            result = survival
        elif mode == "hazard":
            result = probs[:, :, 1:]
        elif mode in ("default", "cif"):
            previous = torch.cat([theta.new_ones(len(theta), 1), survival[:, :-1]], dim=1)
            result = (previous[:, :, None] * probs[:, :, 1:]).cumsum(dim=1)
        else:
            raise ValueError("Competing risks mode must be cif, survival or hazard")
        return result.detach().cpu().numpy()


class MultivariateGaussian(Family):
    """
    Multivariate Gaussian family for correlated multi-task outcomes.

    Models: Y ~ N(μ, Σ)

    Σ can be:
    - diagonal: Σ = diag(exp(log_var))
    - low_rank: Σ = diag(exp(log_diag)) + V @ V^T
    - full: Σ = L @ L^T (Cholesky)
    """
    def __init__(self, n_tasks, covariance_type='low_rank', covariance_rank=5):
        super().__init__()
        self.n_tasks = n_tasks
        self.covariance_type = covariance_type

        # Mean component: μ
        # Note: TabMixNN head produces μ directly

        # Covariance component: Σ
        if covariance_type == 'diagonal':
            self.log_var = nn.Parameter(torch.zeros(n_tasks))
        elif covariance_type == 'low_rank':
            self.log_diag = nn.Parameter(torch.zeros(n_tasks))
            self.V = nn.Parameter(torch.randn(n_tasks, covariance_rank) * 0.1)
        elif covariance_type == 'full':
            self.L_raw = nn.Parameter(torch.eye(n_tasks))
        else:
            raise ValueError(f"Unknown covariance_type: {covariance_type}")

    @property
    def n_params(self):
        # We only predict the mean μ through the neural network.
        # Covariance parameters are global and learned as part of the family.
        return self.n_tasks

    @property
    def primary_param_name(self) -> str:
        return "means"

    def forward_params(self, eta):
        # eta is [batch, n_tasks] -> mu
        return eta

    def get_covariance(self):
        """Compute Σ from parameters."""
        if self.covariance_type == 'diagonal':
            return torch.diag(torch.exp(self.log_var))
        elif self.covariance_type == 'low_rank':
            D = torch.diag(torch.exp(self.log_diag))
            return D + self.V @ self.V.T
        elif self.covariance_type == 'full':
            L = torch.tril(self.L_raw)
            return L @ L.T + 1e-6 * torch.eye(self.n_tasks, device=L.device)
        return torch.eye(self.n_tasks)

    def nll(self, theta, y, **kwargs):
        # theta is mu: [batch, n_tasks]
        # y: [batch, n_tasks]
        mu = theta
        cov = self.get_covariance()
        
        # Using MultivariateNormal. Note: If covariance is large, this is slow.
        # For low-rank, we could use Woodbury, but PyTorch's MultivariateNormal 
        # is reasonably optimized for many cases.
        try:
            dist = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
            return -dist.log_prob(y).mean()
        except ValueError:
            # Fallback for numerical stability if cov becomes non-PSD
            from ..utils.constants import EPSILON, LARGE_PENALTY
            jitter = EPSILON * torch.eye(self.n_tasks, device=cov.device)
            try:
                dist = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov + jitter)
                return -dist.log_prob(y).mean()
            except:
                return torch.tensor(LARGE_PENALTY, device=mu.device)

    def predict(self, theta, mode="default", **kwargs):
        # Returns mu
        return theta.detach().cpu().numpy()


class MultiLabelBinomial(Family):
    """
    Multi-label binomial family for independent binary tasks.
    """
    def __init__(self, n_labels: int):
        super().__init__()
        self.n_labels = n_labels

    @property
    def n_params(self):
        return self.n_labels

    @property
    def primary_param_name(self) -> str:
        return "log_odds_label0"

    def forward_params(self, eta):
        return eta

    def nll(self, theta, y, **kwargs):
        # theta are logits: [batch, n_labels]
        # y: [batch, n_labels]
        return F.binary_cross_entropy_with_logits(theta, y.float(), reduction='none').sum(dim=-1).mean()

    def predict(self, theta, mode="default", **kwargs):
        probs = torch.sigmoid(theta)
        if mode == "proba":
            return probs.detach().cpu().numpy()
        return (probs > 0.5).long().detach().cpu().numpy()


class ContinuousSurvival(Family):
    """
    Continuous-time survival family.
    Currently supports Weibull parametric hazard.

    Hazard: h(t; λ, k) = (k/λ) * (t/λ)^(k-1)
    S(t; λ, k) = exp(-(t/λ)^k)
    f(t; λ, k) = h(t) * S(t)

    Log-likelihood (observed): log(h(t)) + log(S(t))
    Log-likelihood (censored): log(S(t))
    """
    def __init__(self, init_log_shape=0.0, init_log_scale=0.0):
        super().__init__()
        # Global learned baseline parameters (Weibull baseline)
        self.log_shape = nn.Parameter(torch.tensor(init_log_shape)) # log k
        self.log_scale = nn.Parameter(torch.tensor(init_log_scale)) # log λ_baseline
        self._warned_invalid = False
        self.logger = get_logger("families.continuous_survival")

    @property
    def n_params(self):
        # We predict a per-sample log-hazard ratio (log_hr)
        # λ_i = λ_baseline * exp(-log_hr / k)  # This makes h(t) = h0(t) * exp(log_hr)
        return 1

    @property
    def primary_param_name(self) -> str:
        return "log_hazard_ratio"

    def forward_params(self, eta):
        # eta is log_hr: [batch, 1]
        return eta.squeeze(-1)

    def nll(self, theta, y, **kwargs):
        if (y.ndim != 2 or y.shape[1] != 2 or not torch.isfinite(y).all()
                or (y[:, 0] <= 0).any() or not ((y[:, 1] == 0) | (y[:, 1] == 1)).all()):
            raise ValueError("Continuous survival requires positive finite times and binary events")
        log_t, event = y[:, 0].log(), y[:, 1]
        log_H = self.log_shape.exp() * (log_t - self.log_scale) + theta.reshape(-1)
        log_h = self.log_shape - log_t + log_H
        return (log_H.exp() - event * log_h).mean()

    def predict(self, theta, mode="default", **kwargs):
        theta = theta.reshape(-1)
        if mode in ("default", "log_hazard_ratio"):
            result = theta
        elif mode == "risk":
            result = theta.exp()
        elif mode == "median":
            result = (self.log_scale + (np.log(np.log(2.0)) - theta)
                      / self.log_shape.exp()).exp()
        elif mode in ("survival", "cumulative_hazard"):
            if kwargs.get("times") is None:
                raise ValueError("times is required for survival or cumulative_hazard prediction")
            times = torch.as_tensor(kwargs['times'], device=theta.device, dtype=theta.dtype)
            if times.ndim != 1 or not len(times) or not torch.isfinite(times).all() or (times < 0).any():
                raise ValueError("times must be a nonempty vector of finite nonnegative values")
            # At t=0 log(t)=-inf, hence H(0)=0 and S(0)=1 exactly.
            H = (self.log_shape.exp() * (times.log()[None, :] - self.log_scale)
                 + theta[:, None]).exp()
            result = (-H).exp() if mode == "survival" else H
        else:
            raise ValueError("Unknown ContinuousSurvival prediction mode")
        return result.detach().cpu().numpy()


class Multinomial(Family):
    """
    Multinomial (Categorical) family for multi-class classification.
    """
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    @property
    def n_params(self):
        return self.n_classes

    @property
    def primary_param_name(self) -> str:
        return "logit_class0"

    def forward_params(self, eta):
        # eta is [batch, n_classes] logits
        return eta

    def nll(self, theta, y, **kwargs):
        # theta is logits, y is labels [batch]
        # Ensure y is 1D [batch] (remove potential column dim from dataframe extraction)
        return F.cross_entropy(theta, y.long().flatten())

    def predict(self, theta, mode="default", **kwargs):
        # theta is logits
        if mode == "proba":
            return F.softmax(theta, dim=-1).detach().cpu().numpy()
        return torch.argmax(theta, dim=-1).detach().cpu().numpy()
