import torch
import torch.nn as nn
import numpy as np

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

    def forward(self, v):
        # v: [levels, slopes, emb]
        sigma = torch.exp(self.log_sigma)  # [slopes, 1]
        rho = torch.tanh(self.raw_rho).clamp(-0.99, 0.99)  # stability

        levels = self.n_levels
        device = v.device
        ones = torch.ones(levels, levels, device=device)
        eye = torch.eye(levels, device=device)

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
        diag = torch.diag(torch.softplus(torch.diag(self.raw_lower)) + 1e-3)
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
    K_group is provided; K_slope is learnable (unstructured PSD).
    """
    def __init__(self, n_levels, n_slopes, K_group, K_slope=None, jitter=1e-4):
        super().__init__(n_levels, n_slopes)
        if isinstance(K_group, np.ndarray):
            K_group = torch.tensor(K_group, dtype=torch.float32)
        self.register_buffer("K_group", K_group)
        self.jitter = jitter

        if K_slope is None:
            self.raw_lower_slope = nn.Parameter(torch.zeros(n_slopes, n_slopes))
        else:
            if isinstance(K_slope, np.ndarray):
                K_slope = torch.tensor(K_slope, dtype=torch.float32)
            # Store as parameter to allow fine-tuning
            self.raw_lower_slope = nn.Parameter(torch.linalg.cholesky(K_slope + jitter * torch.eye(K_slope.shape[0])))

    def forward(self, v):
        # v: [levels, slopes, emb]
        # Compute L_slope
        tril = torch.tril(self.raw_lower_slope)
        diag = torch.diag(torch.softplus(torch.diag(self.raw_lower_slope)) + self.jitter)
        L_slope = tril - torch.diag(torch.diag(tril)) + diag  # [slopes, slopes]

        # Cholesky of K_group
        eye = torch.eye(self.K_group.shape[0], device=self.K_group.device)
        L_group = torch.linalg.cholesky(self.K_group + self.jitter * eye)  # [levels, levels]

        levels, slopes, emb = v.shape
        out = torch.zeros_like(v)
        for e in range(emb):
            V = v[:, :, e].t()  # [slopes, levels]
            Y = L_slope @ V @ L_group.t()  # [slopes, levels]
            out[:, :, e] = Y.t()
        return out


class GPCovariance(CovarianceStructure):
    """
    RBF kernel GP-style covariance over levels, slope-specific lengthscale and amplitude.
    Similar to KCovariance but kernel is learned via lengthscale.
    """
    def __init__(self, n_levels, n_slopes, full_levels, train_levels, coords, lambda_k=1e-4):
        super().__init__(n_levels, n_slopes)
        coords = np.asarray(coords)
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
        l2 = torch.exp(2 * lengthscale).unsqueeze(-1).unsqueeze(-1)  # [n_slopes, 1, 1]
        K = torch.exp(-0.5 * self.dist2.unsqueeze(0) / l2)  # [n_slopes, L, L]
        return K

    def forward(self, v_train, indices):
        """
        v_train: [n_train_levels, n_slopes, emb] variational means for training levels
        indices: [batch] integer codes from design (0 is unknown; 1..n_levels map into full_levels)
        """
        sigma = torch.exp(self.log_sigma).view(self.n_slopes)  # [slopes]
        K_full = self._kernel_full(self.log_lengthscale)  # [slopes, L, L]

        # Build train/full kernels per slope
        train_idx = self.train_idx
        batch_size = indices.shape[0]
        emb = v_train.shape[2]

        L_full = self.coords.shape[0]
        u_all = torch.zeros(L_full, self.n_slopes, emb, device=v_train.device)
        for s in range(self.n_slopes):
            K_full_s = K_full[s]
            K_train = K_full_s.index_select(0, train_idx).index_select(1, train_idx)
            K_cross = K_full_s.index_select(0, torch.arange(L_full, device=v_train.device))
            K_cross = K_cross.index_select(1, train_idx)

            A = K_train + self.lambda_k * torch.eye(K_train.shape[0], device=K_train.device)
            rhs = (v_train[:, s, :] * sigma[s])  # [n_train, emb]
            alpha = torch.linalg.solve(A, rhs)  # [n_train, emb]

            u_pred = K_cross @ alpha  # [n_levels, emb]
            u_all[:, s, :] = u_pred

        # Gather batch rows with UNK padding at 0
        idx_adj = torch.clamp(indices, min=0)
        unknown_row = u_all.mean(dim=0, keepdim=True)
        u_all_padded = torch.cat([unknown_row, u_all], dim=0)
        gathered = u_all_padded[idx_adj]
        return gathered

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
    """
    Known Kinship/Similarity Matrix K (n_levels x n_levels).
    u ~ N(0, sigma^2 K).
    L = Cholesky(K). 
    u = sigma * L * v
    K is passed via kwargs probably or set later?
    Construct with a fixed K.
    """
    def __init__(self, n_levels, n_slopes, full_levels, train_levels, matrix_tensor, lambda_k=1e-4):
        super().__init__(n_levels, n_slopes)
        # matrix_tensor: Tensor [L, L] for all levels in full_levels order
        if isinstance(matrix_tensor, np.ndarray):
            matrix_tensor = torch.tensor(matrix_tensor, dtype=torch.float32)
        self.register_buffer("K_full", matrix_tensor)
        self.full_levels = list(full_levels)
        self.train_levels = list(train_levels)
        self.lambda_k = lambda_k

        # Map levels to indices
        self.level_to_idx = {lev: i for i, lev in enumerate(self.full_levels)}
        self.train_to_idx = {lev: i for i, lev in enumerate(self.train_levels)}

        # Build K_train and K_cross (all levels vs train)
        train_idx = torch.tensor([self.level_to_idx[l] for l in self.train_levels], dtype=torch.long)
        K_train = matrix_tensor.index_select(0, train_idx).index_select(1, train_idx)
        K_cross = matrix_tensor.index_select(0, torch.arange(len(self.full_levels)))
        K_cross = K_cross.index_select(1, train_idx)

        self.register_buffer("K_train", K_train)
        self.register_buffer("K_cross", K_cross)

        self.log_sigma = nn.Parameter(torch.zeros(n_slopes, 1))

    def forward(self, v_train, indices):
        """
        v_train: [n_train_levels, n_slopes, emb] variational means for training levels only
        indices: [batch] integers from design (0 is unknown; 1..n_levels map into full_levels)
        """
        sigma = torch.exp(self.log_sigma).view(1, self.n_slopes, 1)
        # v_train scaled per slope
        u_train = v_train * sigma  # [n_train, n_slopes, emb]

        # Solve (K_train + lambda I) alpha = u_train for each slope/emb jointly
        n_train = self.K_train.shape[0]
        A = self.K_train + self.lambda_k * torch.eye(n_train, device=self.K_train.device)
        rhs = u_train.reshape(n_train, -1)  # [n_train, n_slopes*emb]
        alpha = torch.linalg.solve(A, rhs)  # [n_train, n_slopes*emb]
        alpha = alpha.reshape(n_train, self.n_slopes, u_train.shape[2])

        # Predict all levels (including train) via K_cross @ alpha
        # K_cross: [n_full, n_train]
        u_all = torch.einsum("lt, tse -> lse", self.K_cross, alpha)

        # Gather batch rows; indices 0 are unknown -> zero row
        idx_adj = indices.clone()
        # clamp negatives
        idx_adj = torch.clamp(idx_adj, min=0)
        # build padded to account for unknown at 0; our u_all corresponds to full_levels starting at 0 offset
        # we insert a zero row at top
        unknown_row = u_all.mean(dim=0, keepdim=True)
        u_all_padded = torch.cat([unknown_row, u_all], dim=0)
        gathered = u_all_padded[idx_adj]
        return gathered
