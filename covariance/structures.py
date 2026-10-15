import torch
import torch.nn as nn
import numpy as np
from .kernels import rbf_kernel

import torch.nn.functional as F

def _covariance_matrix(value, size, name):
    matrix = torch.as_tensor(value, dtype=torch.float32)
    if matrix.shape != (size, size) or not torch.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite {size} by {size} matrix")
    if not torch.allclose(matrix, matrix.T, atol=1e-6):
        raise ValueError(f"{name} must be symmetric")
    if torch.linalg.eigvalsh(matrix).min() < -1e-5:
        raise ValueError(f"{name} must be positive semidefinite")
    return matrix


def _gather_levels(values, indices):
    if indices is None:
        return values
    padding = values.new_zeros(1, *values.shape[1:])
    return torch.cat([padding, values], dim=0).index_select(0, indices)


class CovarianceStructure(nn.Module):
    """
    Abstract base class for covariance structures L_t(theta).
    Forward input: v ~ N(0, I)
    Forward output: u = L v
    """
    def __init__(self, n_levels, n_slopes):
        super().__init__()
        self.n_levels = n_levels
        self.n_slopes = n_slopes
        
    def forward(self, v):
        """
        v: [n_levels, n_slopes, embedding_dim]
        Returns u of same shape.
        """
        raise NotImplementedError
        
    def penalty(self):
        return 0.0

class IIDCovariance(CovarianceStructure):
    """
    u_{g, r} ~ N(0, sigma_r^2)
    L is diagonal scaling per slope dimension (broadcast over levels).
    """
    def __init__(self, n_levels, n_slopes):
        super().__init__(n_levels, n_slopes)
        # Log sigma per slope
        self.log_sigma = nn.Parameter(torch.zeros(n_slopes, 1))

    def forward(self, v):
        # v: [n_levels, n_slopes, emb]
        # sigma: [n_slopes, 1] -> [1, n_slopes, 1]
        sigma = torch.exp(self.log_sigma).unsqueeze(0)
        return v * sigma


class ARMACovariance(CovarianceStructure):
    """
    ARMA(p, q) along the level dimension for each slope independently.
    Supports p >= 0, q >= 0 (recommended small: p<=2, q<=2). If p=1,q=0 this reduces to AR(1).
    """
    def __init__(self, n_levels, n_slopes, p=1, q=0):
        super().__init__(n_levels, n_slopes)
        self.p = p
        self.q = q
        self.log_sigma = nn.Parameter(torch.zeros(n_slopes, 1))
        if p > 0:
            self.phi_raw = nn.Parameter(torch.zeros(p, n_slopes, 1))
        else:
            self.register_parameter("phi_raw", None)
        if q > 0:
            self.theta_raw = nn.Parameter(torch.zeros(q, n_slopes, 1))
        else:
            self.register_parameter("theta_raw", None)

    def forward(self, v):
        # v: [n_levels, n_slopes, emb]
        sigma = torch.exp(self.log_sigma).unsqueeze(0)  # [1, n_slopes, 1]
        phi = torch.tanh(self.phi_raw) if self.phi_raw is not None else None
        theta = torch.tanh(self.theta_raw) if self.theta_raw is not None else None

        # Initialize buffers for previous u and eps
        u_prev = [torch.zeros_like(v[0:1]) for _ in range(self.p)]
        eps_prev = [torch.zeros_like(v[0:1]) for _ in range(self.q)]

        outputs = []
        for t in range(self.n_levels):
            eps_t = v[t:t+1]
            eps_eff = eps_t
            if theta is not None:
                for i in range(min(self.q, len(eps_prev))):
                    eps_eff = eps_eff + theta[i].unsqueeze(0) * eps_prev[i]

            autoreg = 0.0
            if phi is not None:
                for i in range(min(self.p, len(u_prev))):
                    autoreg = autoreg + phi[i].unsqueeze(0) * u_prev[i]

            u_t = autoreg + sigma * eps_eff
            outputs.append(u_t)

            # Update histories
            if self.p > 0:
                u_prev = [u_t] + u_prev[: max(0, self.p - 1)]
            if self.q > 0:
                eps_prev = [eps_t] + eps_prev[: max(0, self.q - 1)]

        return torch.cat(outputs, dim=0)


class CompoundSymmetryCovariance(CovarianceStructure):
    """
    Compound symmetry (CS): Cov = sigma^2 * ((1-rho) I + rho 11^T) across levels, per slope.
    """
    def __init__(self, n_levels, n_slopes):
        super().__init__(n_levels, n_slopes)
        self.log_sigma = nn.Parameter(torch.zeros(n_slopes, 1))
        self.raw_rho = nn.Parameter(torch.zeros(n_slopes, 1))

    @property
    def correlation(self):
        if self.n_levels <= 1:
            return self.raw_rho * 0
        lower = -1.0 / (self.n_levels - 1)
        # Shift makes raw_rho=0 correspond to rho=0.
        shift = -np.log(self.n_levels - 1)
        unit = torch.sigmoid(self.raw_rho + shift).clamp(1e-6, 1 - 1e-6)
        return lower + (1 - lower) * unit

    def forward(self, v):
        # v: [levels, slopes, emb]
        sigma = torch.exp(self.log_sigma)  # [slopes, 1]
        rho = self.correlation

        levels = self.n_levels
        device = v.device
        ones = torch.ones(levels, levels, device=device, dtype=v.dtype)
        eye = torch.eye(levels, device=device, dtype=v.dtype)

        out = []
        for s in range(self.n_slopes):
            C = (1 - rho[s]) * eye + rho[s] * ones
            try:
                L = torch.linalg.cholesky(C + 1e-5 * eye)
            except RuntimeError:
                L = torch.linalg.cholesky(C + 1e-3 * eye)
            V = v[:, s, :]  # [levels, emb]
            U = torch.matmul(L, V) * sigma[s]
            out.append(U.unsqueeze(1))
        return torch.cat(out, dim=1)


class UnstructuredSlopeCovariance(CovarianceStructure):
    """
    Unstructured covariance across slopes (per level), shared across levels, with learnable Cholesky.
    """
    def __init__(self, n_levels, n_slopes):
        super().__init__(n_levels, n_slopes)
        # Lower-triangular raw parameters for slope covariance
        self.raw_lower = nn.Parameter(torch.zeros(n_slopes, n_slopes))

    def forward(self, v):
        # v: [levels, slopes, emb]
        # Build Cholesky with positive diagonals
        tril = torch.tril(self.raw_lower)
        diag = torch.diag(F.softplus(torch.diag(self.raw_lower)) + 1e-3)
        L = tril - torch.diag(torch.diag(tril)) + diag  # replace diag with softplus

        levels, _, emb = v.shape
        V = v.permute(0, 2, 1)  # [levels, emb, slopes]
        V_flat = V.reshape(levels * emb, self.n_slopes)  # [levels*emb, slopes]
        U_flat = torch.matmul(V_flat, L.t())  # apply slope covariance
        U = U_flat.view(levels, emb, self.n_slopes).permute(0, 2, 1)  # [levels, slopes, emb]
        return U


class KroneckerCovariance(CovarianceStructure):
    """
    Separable covariance: Cov = K_group ⊗ K_slope.
    Handles Train/Test splitting via GBLUP extrapolation on the group dimension.
    """
    def __init__(self, n_levels, n_slopes, K_group, full_levels, train_levels, K_slope=None, jitter=1e-4):
        super().__init__(n_levels, n_slopes)
        K_group = _covariance_matrix(K_group, len(full_levels), "K_group")
        self.register_buffer("K_full", K_group)
        self.jitter = jitter

        # Levels handling
        self.full_levels = list(full_levels)
        self.train_levels = list(train_levels)
        self.level_to_idx = {lev: i for i, lev in enumerate(self.full_levels)}
        train_idx = torch.tensor([self.level_to_idx[l] for l in self.train_levels], dtype=torch.long)
        
        self.register_buffer("train_idx", train_idx)
        # Subsets
        self.register_buffer("K_train", K_group.index_select(0, train_idx).index_select(1, train_idx))
        self.register_buffer("K_cross", K_group.index_select(1, train_idx)) # All rows, train cols

        if K_slope is None:
            self.raw_lower_slope = nn.Parameter(torch.zeros(n_slopes, n_slopes))
        else:
            K_slope = _covariance_matrix(K_slope, n_slopes, "K_slope")
            chol = torch.linalg.cholesky(K_slope)
            diagonal = (chol.diag() - jitter).clamp_min(1e-8)
            raw_diag = diagonal + torch.log(-torch.expm1(-diagonal))
            self.raw_lower_slope = nn.Parameter(torch.tril(chol, -1) + torch.diag(raw_diag))

    def forward(self, v, indices=None):
        # v: [n_train, slopes, emb] - variational params for training set

        # L_slope
        tril = torch.tril(self.raw_lower_slope)
        diag = torch.diag(F.softplus(torch.diag(self.raw_lower_slope)) + self.jitter)
        L_slope = tril - torch.diag(torch.diag(tril)) + diag  # [slopes, slopes]

        # L_group (Train)
        eye = torch.eye(self.K_train.shape[0], device=v.device, dtype=v.dtype)
        try:
            L_group = torch.linalg.cholesky(self.K_train + self.jitter * eye)
        except RuntimeError:
            L_group = torch.linalg.cholesky(self.K_train + 10 * self.jitter * eye)

        # Generate u_train = L_slope @ v @ L_group.T
        # Shapes: v [N_train, Slopes, Emb]
        # We process each Emb channel
        n_train, n_slopes, emb = v.shape
        u_train = torch.zeros_like(v)
        
        for e in range(emb):
            V_e = v[:, :, e].t() # [Slopes, N_train]
            # Y = L_slope @ V @ L_group.T
            # L_group.T is [N_train, N_train] (Upper?)
            # torch.linalg.cholesky returns Lower L. L @ L.T = K.
            # We want u ~ N(0, K). u = L @ v_white.
            # Here K_total = K_train \otimes (L_slope L_slope.T).
            # u = (L_group \otimes L_slope) vec(V).
            # Matrix form: U = L_slope @ V @ L_group.T.
            Y = L_slope @ V_e @ L_group.t() # [Slopes, N_train]
            u_train[:, :, e] = Y.t()

        # Extrapolate to u_all via Henderson BLUP: u_all = K_cross @ K_train^-1 @ u_train
        # Solve K_train @ alpha = u_train
        # We can reuse L_group: K_train = L L.T
        # alpha = K^-1 u = L.T^-1 L^-1 u
        # rhs: u_train [N_train, Slopes, Emb] - reshape to 2D
        u_train_flat = u_train.reshape(n_train, -1) # [N, S*E]
        # alpha = solve(K_train, u_train)
        # Using cholesky solve is faster
        alpha_flat = torch.cholesky_solve(u_train_flat, L_group)
        
        # u_all = K_cross @ alpha
        # K_cross [N_full, N_train]
        u_all_flat = self.K_cross @ alpha_flat # [N_full, S*E]
        u_all = u_all_flat.reshape(-1, n_slopes, emb)

        # Interpolation applies only to held-out levels, never shrink training values.
        u_all = u_all.index_copy(0, self.train_idx, u_train)
        return _gather_levels(u_all, indices)


class GPCovariance(CovarianceStructure):
    """
    RBF kernel GP-style covariance over levels, slope-specific lengthscale and amplitude.
    Similar to KCovariance but kernel is learned via lengthscale.
    """
    def __init__(self, n_levels, n_slopes, full_levels, train_levels, coords, lambda_k=1e-4):
        super().__init__(n_levels, n_slopes)
        coords = np.asarray(coords)
        if coords.ndim == 1:
            coords = coords[:, None]
        if coords.shape[0] != len(full_levels):
            raise ValueError("coords must align with full_levels length.")
        self.register_buffer("coords", torch.tensor(coords, dtype=torch.float32))
        self.full_levels = list(full_levels)
        self.train_levels = list(train_levels)
        self.lambda_k = lambda_k

        self.level_to_idx = {lev: i for i, lev in enumerate(self.full_levels)}
        self.train_to_idx = {lev: i for i, lev in enumerate(self.train_levels)}
        train_idx = torch.tensor([self.level_to_idx[l] for l in self.train_levels], dtype=torch.long)
        self.register_buffer("train_idx", train_idx)

        self.log_sigma = nn.Parameter(torch.zeros(n_slopes, 1))
        self.log_lengthscale = nn.Parameter(torch.zeros(n_slopes, 1))

        # Precompute pairwise squared distances over full levels
        diff = self.coords.unsqueeze(1) - self.coords.unsqueeze(0)  # [L, L, d]
        dist2 = (diff ** 2).sum(dim=2)  # [L, L]
        self.register_buffer("dist2", dist2)

    def _kernel_full(self, lengthscale):
        # lengthscale: [n_slopes, 1]
        out = []
        for s in range(self.n_slopes):
            l = torch.exp(lengthscale[s])
            K = rbf_kernel(self.coords, self.coords, l, jitter=self.lambda_k)
            out.append(K)
        return torch.stack(out)

    def forward(self, v_train, indices=None):
        """
        v_train: [n_train_levels, n_slopes, emb] variational means for training levels
        indices: [batch] integer codes from design (0 is unknown; 1..n_levels map into full_levels)
                 If None, returns embeddings for all full_levels.
        """
        sigma = torch.exp(self.log_sigma).view(self.n_slopes)  # [slopes]
        K_full = self._kernel_full(self.log_lengthscale)  # [slopes, L, L]

        # Build train/full kernels per slope
        train_idx = self.train_idx
        emb = v_train.shape[2]

        L_full = self.coords.shape[0]
        u_all = v_train.new_zeros(L_full, self.n_slopes, emb)
        for s in range(self.n_slopes):
            K_full_s = K_full[s]
            K_train = K_full_s.index_select(0, train_idx).index_select(1, train_idx)
            K_cross = K_full_s.index_select(0, torch.arange(L_full, device=v_train.device))
            K_cross = K_cross.index_select(1, train_idx)

            # Kernel helper already adds diagonal jitter to a self-kernel.
            L = torch.linalg.cholesky(K_train)
            u_train = sigma[s] * (L @ v_train[:, s, :])
            alpha = torch.cholesky_solve(u_train, L)
            u_pred = K_cross @ alpha
            u_all[:, s, :] = u_pred.index_copy(0, train_idx, u_train)
        return _gather_levels(u_all, indices)


class AR1Covariance(CovarianceStructure):
    """
    Autoregressive AR(1) over LEVELS (e.g. time points).
    Assumes levels are ordered 0..N-1.
    Cov(u_i, u_j) = sigma^2 * rho^|i-j|
    Cholesky L for AR1 is structured. 
    L_{i,j} = 0 if j > i
    L_{i,i} = sqrt(1-rho^2) for i>0, 1 for i=0 ? 
    Wait, AR1 inverse (precision) is tridiagonal.
    Explicit L construction:
    L is lower triangular.
    u_t = rho * u_{t-1} + noise
    Recursive: u_0 = eps_0 / sqrt(1-rho^2), u_t = rho u_{t-1} + eps_t
    
    This implementation applies AR1 over the 'n_levels' dimension, indep per slope.
    """
    def __init__(self, n_levels, n_slopes):
        super().__init__(n_levels, n_slopes)
        self.log_sigma = nn.Parameter(torch.zeros(n_slopes, 1))
        # Rho per slope or shared? Let's say rho per slope.
        self.raw_rho = nn.Parameter(torch.zeros(n_slopes, 1)) # tanh -> rho

    def forward(self, v):
        # v: [n_levels, n_slopes, emb]
        # Construct AR1 transform explicitly or recursively.
        # Recursive is O(N).
        # u[0] = v[0] * sigma / sqrt(1-rho^2)  (stationary variance)
        # u[t] = rho * u[t-1] + sigma * v[t]
        
        rho = torch.tanh(self.raw_rho).unsqueeze(0) # [1, n_slopes, 1]
        sigma = torch.exp(self.log_sigma).unsqueeze(0) # [1, n_slopes, 1]
        
        u_list = []
        # Stationary initialization
        # Var(u) = sigma^2 / (1-rho^2)
        # u0 = v0 * sigma / sqrt(1-rho^2)
        
        sqrt_stat = torch.sqrt(1 - rho**2 + 1e-6)
        u_prev = (v[0:1] * sigma) / sqrt_stat
        u_list.append(u_prev)
        
        for t in range(1, self.n_levels):
            # v[t:t+1]
            # u_t = rho * u_{t-1} + sigma * v_t
            # Note: usually AR1 noise is sigma_epsilon. The marginal variance is sigma^2/(1-rho^2).
            # If we interpret sigma parameter as the marginal std, then conditional std is sigma*sqrt(1-rho^2).
            # Let's interpret self.log_sigma as the innovation noise std.
            
            u_curr = rho * u_prev + sigma * v[t:t+1]
            u_list.append(u_curr)
            u_prev = u_curr
            
        return torch.cat(u_list, dim=0)

class KCovariance(CovarianceStructure):
    """Known group covariance: u_train = sigma * chol(K_train+jitter I) v.

    The encoder uses q(v) and a standard-normal prior. Other supplied levels
    return Gaussian conditional means K_cross (K_train+jitter I)^-1 u_train.
    This point-prediction operator omits conditional residual uncertainty.
    """
    def __init__(self, n_levels, n_slopes, full_levels, train_levels, matrix_tensor, lambda_k=1e-4):
        super().__init__(n_levels, n_slopes)
        matrix = _covariance_matrix(matrix_tensor, len(full_levels), "K")
        self.full_levels, self.train_levels = list(full_levels), list(train_levels)
        self.lambda_k = lambda_k
        self.level_to_idx = {lev: i for i, lev in enumerate(full_levels)}
        self.train_to_idx = {lev: i for i, lev in enumerate(train_levels)}
        idx = torch.tensor([self.level_to_idx[x] for x in train_levels], dtype=torch.long)
        self.register_buffer("train_idx", idx)
        self.register_buffer("K_full", matrix)
        self.register_buffer("K_train", matrix.index_select(0, idx).index_select(1, idx))
        self.register_buffer("K_cross", matrix.index_select(1, idx))
        chol = torch.linalg.cholesky(self.K_train + lambda_k * torch.eye(len(idx)))
        self.register_buffer("K_chol", chol)
        self.log_sigma = nn.Parameter(torch.zeros(n_slopes, 1))

    def forward(self, v_train, indices=None):
        flat = v_train.reshape(len(self.train_levels), -1)
        u_train = (self.K_chol @ flat).reshape_as(v_train) * self.log_sigma.exp()[None]
        alpha = torch.cholesky_solve(u_train.reshape(len(self.train_levels), -1), self.K_chol)
        values = (self.K_cross @ alpha).reshape(-1, self.n_slopes, v_train.shape[-1])
        values = values.index_copy(0, self.train_idx, u_train)
        return _gather_levels(values, indices)

class LowRankCovariance(CovarianceStructure):
    """
    u = (D + VV^T)^1/2 v
    Σ = D + V V^T
    """
    def __init__(self, n_levels, n_slopes, rank=5):
        super().__init__(n_levels, n_slopes)
        self.rank = rank
        self.log_diag = nn.Parameter(torch.zeros(n_slopes, 1))
        self.V = nn.Parameter(torch.randn(n_slopes, rank) * 0.1)

    def forward(self, v):
        L = self.get_cholesky() 
        levels, slopes, emb = v.shape
        v_flat = v.permute(0, 2, 1).reshape(-1, slopes) 
        u_flat = v_flat @ L.T
        return u_flat.view(levels, emb, slopes).permute(0, 2, 1)

    def get_matrix(self):
        D = torch.diag(torch.exp(self.log_diag).squeeze(-1))
        return D + self.V @ self.V.T

    def get_cholesky(self):
        Sigma = self.get_matrix()
        return torch.linalg.cholesky(Sigma + 1e-6 * torch.eye(self.n_slopes, device=Sigma.device))

class FullCovariance(CovarianceStructure):
    """
    Unstructured covariance matrix Σ = L L^T.
    """
    def __init__(self, n_levels, n_slopes):
        super().__init__(n_levels, n_slopes)
        self.L_raw = nn.Parameter(torch.eye(n_slopes))

    def forward(self, v):
        L = torch.tril(self.L_raw)
        levels, slopes, emb = v.shape
        v_flat = v.permute(0, 2, 1).reshape(-1, slopes)
        u_flat = v_flat @ L.T
        return u_flat.view(levels, emb, slopes).permute(0, 2, 1)
