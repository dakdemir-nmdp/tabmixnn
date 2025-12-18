import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SEMLayer(nn.Module):
    """
    Structural Equation Model (SEM) Layer for causal coordination.
    Implements Lavaan-style directed acyclic graph (DAG): y = (I-B)^{-1} x
    where B is a learned lower triangular adjacency matrix.
    
    This is NOT spatial attention - it models directed dependencies between neurons.
    """
    def __init__(self, dim, sam_alpha_init=0.0, max_offdiag=0.9, residual=True, residual_scale=1.0):
        super().__init__()
        self.dim = dim
        self.max_offdiag = max_offdiag
        self.use_residual = residual
        self.residual_scale = residual_scale
        
        # Unconstrained weights B_tilde
        # We only need lower triangle (strictly).
        # We can store full matrix and mask, or just parameters.
        # Mask: 1 if i > j, 0 else.
        
        # Manual mask creation to avoid torch.tril segfault on some platforms
        r = torch.arange(dim).long().unsqueeze(1)
        c = torch.arange(dim).long().unsqueeze(0)
        tril_ones = (r > c).float()
        
        self.register_buffer('mask', tril_ones)
        
        self.B_tilde = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.alpha_logits = nn.Parameter(torch.ones(dim, dim) * sam_alpha_init)
        
    def get_B(self):
        """Get gated adjacency matrix B."""
        gates = torch.sigmoid(self.alpha_logits)
        B = gates * torch.tanh(self.B_tilde) * self.max_offdiag
        return B * self.mask

    def forward(self, x):
        # x: [batch, dim]
        B = self.get_B()
        I = torch.eye(self.dim, device=x.device)
        
        # Solve (I - B) y = x  => y = (I-B)^-1 x
        # B is strictly lower triangular, creating a DAG.
        # (I-B) is lower triangular with 1s on diagonal. Invertible.
        # torch.linalg.solve_triangular is best if available, else solve.
        # Since B is strictly lower, I-B is lower triangular.
        # shape of (I-B) is [dim, dim]. x is [batch, dim].
        # We want y such that (I-B) y^T = x^T
        
        A = I - B
        # solve A Y^T = X^T
        # torch.linalg.solve_triangular expects A to be triangular.
        # upper=False (lower).
        
        # x is [batch, dim], transpose to [dim, batch]
        x_t = x.t()
        # Ensure device compatibility
        y_t = torch.linalg.solve_triangular(A, x_t, upper=False)
        y = y_t.t()
        if self.use_residual:
            y = y + self.residual_scale * x
        return y

    def penalty(self):
        # L1-like penalty on gates: sum sigmoid(alpha)
        # Only strict lower triangle
        sigs = torch.sigmoid(self.alpha_logits) * self.mask
        return sigs.sum()


class PositionalSEMLayer(nn.Module):
    """
    SEM layer with position-aware prior for 1D/2D neuron arrays.
    Neurons at positions i and j have dependency strength based on distance |i-j|.
    
    Uses RBF kernel to encourage nearby neurons to have stronger dependencies.
    """
    def __init__(self, dim, max_offdiag=0.9, residual=True, residual_scale=1.0,
                 lengthscale_init=3.0, grid_shape=None):
        """
        Args:
            dim: Number of neurons
            max_offdiag: Maximum off-diagonal value in B
            residual: Whether to add residual connection
            residual_scale: Scale for residual connection
            lengthscale_init: Initial lengthscale for RBF kernel
            grid_shape: Optional (U, V) tuple for 2D grid. If None, assumes 1D layout.
        """
        super().__init__()
        self.dim = dim
        self.max_offdiag = max_offdiag
        self.use_residual = residual
        self.residual_scale = residual_scale
        self.grid_shape = grid_shape
        
        # Neuron positions
        if grid_shape is not None:
            # 2D grid: positions are (row, col) coordinates
            U, V = grid_shape
            assert U * V == dim, f"Grid shape {grid_shape} doesn't match dim {dim}"
            rows = torch.arange(U, dtype=torch.float32).unsqueeze(1).repeat(1, V)
            cols = torch.arange(V, dtype=torch.float32).unsqueeze(0).repeat(U, 1)
            positions = torch.stack([rows.flatten(), cols.flatten()], dim=1)  # [dim, 2]
        else:
            # 1D: positions are 0, 1, 2, ..., dim-1
            positions = torch.arange(dim, dtype=torch.float32).unsqueeze(1)  # [dim, 1]
        
        self.register_buffer("positions", positions)
        
        # Learnable lengthscale for position kernel
        self.log_lengthscale = nn.Parameter(torch.tensor(lengthscale_init).log())
        
        # Base adjacency weights (unconstrained)
        self.B_tilde = nn.Parameter(torch.randn(dim, dim) * 0.01)
        
        # Gating logits
        self.alpha_logits = nn.Parameter(torch.zeros(dim, dim))
        
        # Lower triangular mask (DAG constraint)
        mask = torch.tril(torch.ones(dim, dim), diagonal=-1)
        self.register_buffer("mask", mask)
    
    def get_position_prior(self):
        """
        Compute position-based prior: K[i,j] = exp(-||pos_i - pos_j||^2 / (2*l^2))
        This encourages nearby neurons to have stronger dependencies.
        """
        l = torch.exp(self.log_lengthscale)
        # Compute pairwise distances
        diff = self.positions.unsqueeze(0) - self.positions.unsqueeze(1)  # [dim, dim, d]
        dist_sq = (diff ** 2).sum(dim=2)  # [dim, dim]
        K = torch.exp(-0.5 * dist_sq / (l ** 2))
        return K
    
    def get_B(self):
        """
        Adjacency matrix with position-aware gating.
        B[i,j] = gate[i,j] * tanh(B_tilde[i,j]) * K[i,j] * mask[i,j]
        """
        gates = torch.sigmoid(self.alpha_logits)
        K = self.get_position_prior()
        
        # Combine: base weights * position kernel * gate * DAG mask
        B = gates * torch.tanh(self.B_tilde) * K * self.max_offdiag
        return B * self.mask
    
    def forward(self, x):
        """
        Solve (I - B)y = x for y.
        """
        B = self.get_B()
        I = torch.eye(self.dim, device=x.device)
        A = I - B
        
        # Solve via triangular solver
        y = torch.linalg.solve_triangular(A, x.T, upper=False).T
        
        if self.use_residual:
            y = y + self.residual_scale * x
        return y
    
    def penalty(self):
        """
        L1 penalty on gates + lengthscale regularization.
        """
        gate_penalty = torch.sigmoid(self.alpha_logits).sum()
        # Encourage moderate lengthscales (not too local, not too global)
        l = torch.exp(self.log_lengthscale)
        length_penalty = (l - 3.0) ** 2  # Soft prior around l=3
        return gate_penalty + 0.01 * length_penalty


class SpatialCovarianceLayer(nn.Module):
    """
    Stochastic layer that adds spatial covariance noise to activations.
    Implements: h_out = h_in + ε, where ε ~ N(0, Σ_spatial)
    
    Supports 1D and 2D neuron grids with Kronecker factorization for efficiency.
    """
    def __init__(self, dim, kernel_type="rbf", lengthscale_init=3.0, 
                 amplitude_init=0.1, grid_shape=None, jitter=1e-4):
        """
        Args:
            dim: Number of neurons
            kernel_type: Type of kernel ("rbf", "matern")
            lengthscale_init: Initial lengthscale
            amplitude_init: Initial noise amplitude
            grid_shape: Optional (U, V) for 2D grid with Kronecker factorization
            jitter: Numerical stability parameter
        """
        super().__init__()
        self.dim = dim
        self.kernel_type = kernel_type
        self.grid_shape = grid_shape
        self.jitter = jitter
        
        if grid_shape is not None:
            # 2D grid: use Kronecker factorization Σ = Σ_rows ⊗ Σ_cols
            U, V = grid_shape
            assert U * V == dim, f"Grid shape {grid_shape} doesn't match dim {dim}"
            self.U = U
            self.V = V
            
            # Positions for rows and columns
            pos_rows = torch.arange(U, dtype=torch.float32).unsqueeze(1)
            pos_cols = torch.arange(V, dtype=torch.float32).unsqueeze(1)
            self.register_buffer("pos_rows", pos_rows)
            self.register_buffer("pos_cols", pos_cols)
            
            # Separate lengthscales for rows and columns
            self.log_lengthscale_rows = nn.Parameter(torch.tensor(lengthscale_init).log())
            self.log_lengthscale_cols = nn.Parameter(torch.tensor(lengthscale_init).log())
        else:
            # 1D: standard covariance
            positions = torch.arange(dim, dtype=torch.float32).unsqueeze(1)
            self.register_buffer("positions", positions)
            self.log_lengthscale = nn.Parameter(torch.tensor(lengthscale_init).log())
        
        # Amplitude (noise scale)
        self.log_amplitude = nn.Parameter(torch.tensor(amplitude_init).log())
    
    def _rbf_kernel(self, positions, lengthscale):
        """Compute RBF kernel matrix."""
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=2)
        K = torch.exp(-0.5 * dist_sq / (lengthscale ** 2))
        return K
    
    def get_covariance_1d(self):
        """Get covariance matrix for 1D layout."""
        l = torch.exp(self.log_lengthscale)
        K = self._rbf_kernel(self.positions, l)
        # Add jitter for numerical stability
        K = K + self.jitter * torch.eye(self.dim, device=K.device)
        return K
    
    def get_covariance_2d(self):
        """Get Kronecker factorized covariance for 2D grid."""
        l_rows = torch.exp(self.log_lengthscale_rows)
        l_cols = torch.exp(self.log_lengthscale_cols)
        
        K_rows = self._rbf_kernel(self.pos_rows, l_rows)
        K_cols = self._rbf_kernel(self.pos_cols, l_cols)
        
        # Add jitter
        K_rows = K_rows + self.jitter * torch.eye(self.U, device=K_rows.device)
        K_cols = K_cols + self.jitter * torch.eye(self.V, device=K_cols.device)
        
        return K_rows, K_cols
    
    def sample_noise(self, batch_size, device):
        """Sample ε ~ N(0, Σ_spatial)."""
        amplitude = torch.exp(self.log_amplitude)
        
        if self.grid_shape is not None:
            # 2D: Kronecker sampling
            K_rows, K_cols = self.get_covariance_2d()
            
            # Cholesky decomposition
            L_rows = torch.linalg.cholesky(K_rows)
            L_cols = torch.linalg.cholesky(K_cols)
            
            # Sample standard normal
            z = torch.randn(batch_size, self.U, self.V, device=device)
            
            # Apply Kronecker structure: (L_rows ⊗ L_cols) z
            # Efficient: z_out = L_rows @ z @ L_cols^T
            noise = torch.einsum('ij,bjk,kl->bil', L_rows, z, L_cols.T)
            noise = noise.reshape(batch_size, self.dim)
        else:
            # 1D: standard sampling
            K = self.get_covariance_1d()
            L = torch.linalg.cholesky(K)
            
            z = torch.randn(batch_size, self.dim, device=device)
            noise = torch.matmul(z, L.T)
        
        return amplitude * noise
    
    def forward(self, x):
        """
        Add spatial noise during training, deterministic during eval.
        """
        if self.training:
            noise = self.sample_noise(x.shape[0], x.device)
            return x + noise
        else:
            return x
    
    def penalty(self):
        """Regularization on amplitude to prevent excessive noise."""
        amplitude = torch.exp(self.log_amplitude)
        # Soft constraint: prefer small but non-zero amplitude
        return (amplitude - 0.1) ** 2

from .covariance import (
    IIDCovariance,
    AR1Covariance,
    ARMACovariance,
    CompoundSymmetryCovariance,
    UnstructuredSlopeCovariance,
    KroneckerCovariance,
    KCovariance,
    GPCovariance,
)

class MixedEffectsRepresentation(nn.Module):
    def __init__(self, fixed_input_dim, embedding_dim, random_terms_config, cov_struct_map=None, kl_normalize=True, scale_random_by_terms=False, unknown_strategy="learned"):
        super().__init__()
        # Fixed part
        self.fixed_linear = nn.Linear(fixed_input_dim, embedding_dim)
        self.kl_normalize = kl_normalize
        self.scale_random_by_terms = scale_random_by_terms
        self.unknown_strategy = unknown_strategy
        
        self.random_terms = nn.ModuleDict()
        self.variational_params = nn.ParameterDict() # For mu, log_scale
        # Deprecate simple variance_components dict, use Covariance modules
        self.cov_modules = nn.ModuleDict()
        self.unknown_params = nn.ParameterDict()
        
        self.embedding_dim = embedding_dim
        self.config = random_terms_config # List of dicts
        
        for term in random_terms_config:
            tid = term['term_id']
            # e.g. term_0
            # group_col needed to lookup cov type?
            # We pass cov_struct_map which maps group_col -> type.
            # term['group_col'] needs to be in random_terms_config passed from design?
            # Currently model init only gets minimal config.
            # We assume default IID.
            
            n_levels = term['n_levels']
            n_slopes = term['n_slopes']
            group_name = term.get('group', 'unknown') # Add group name to config in next step
            train_levels = term.get('train_levels', [])
            
            # Select Covariance
            # Default IID
            ctype = "IID"
            cargs = {}
            if cov_struct_map and group_name in cov_struct_map:
                spec = cov_struct_map[group_name]
                ctype = spec.get("type", "IID").upper()
                cargs = spec
            else:
                ctype = "IID"
                cargs = {}

            if ctype == "AR1":
                self.variational_params[tid] = nn.ParameterList([
                    nn.Parameter(torch.zeros(n_levels, n_slopes, embedding_dim)), # mu
                    nn.Parameter(torch.ones(n_levels, n_slopes, embedding_dim) * -2.0) # log_s
                ])
                self.cov_modules[tid] = AR1Covariance(n_levels, n_slopes)

            elif ctype in ["AR", "ARMA"]:
                p = int(cargs.get("p", 1))
                q = int(cargs.get("q", 0))
                self.variational_params[tid] = nn.ParameterList([
                    nn.Parameter(torch.zeros(n_levels, n_slopes, embedding_dim)), # mu
                    nn.Parameter(torch.ones(n_levels, n_slopes, embedding_dim) * -2.0) # log_s
                ])
                self.cov_modules[tid] = ARMACovariance(n_levels, n_slopes, p=p, q=q)

            elif ctype in ["CS", "COMPOUND_SYMMETRY"]:
                self.variational_params[tid] = nn.ParameterList([
                    nn.Parameter(torch.zeros(n_levels, n_slopes, embedding_dim)), # mu
                    nn.Parameter(torch.ones(n_levels, n_slopes, embedding_dim) * -2.0) # log_s
                ])
                self.cov_modules[tid] = CompoundSymmetryCovariance(n_levels, n_slopes)

            elif ctype in ["UN", "UNSTRUCTURED"]:
                self.variational_params[tid] = nn.ParameterList([
                    nn.Parameter(torch.zeros(n_levels, n_slopes, embedding_dim)), # mu
                    nn.Parameter(torch.ones(n_levels, n_slopes, embedding_dim) * -2.0) # log_s
                ])
                self.cov_modules[tid] = UnstructuredSlopeCovariance(n_levels, n_slopes)

            elif ctype == "KRON":
                self.variational_params[tid] = nn.ParameterList([
                    nn.Parameter(torch.zeros(n_levels, n_slopes, embedding_dim)), # mu
                    nn.Parameter(torch.ones(n_levels, n_slopes, embedding_dim) * -2.0) # log_s
                ])
                K_group = cargs.get("K_group") or cargs.get("matrix_tensor")
                K_slope = cargs.get("K_slope", None)
                jitter = cargs.get("lambda_k", 1e-4)
                if K_group is None:
                    raise ValueError(f"KRON covariance for group {group_name} requires 'K_group'.")
                self.cov_modules[tid] = KroneckerCovariance(
                    n_levels=n_levels,
                    n_slopes=n_slopes,
                    K_group=K_group,
                    K_slope=K_slope,
                    jitter=jitter
                )

            elif ctype == "K":
                # For K, we maintain variational params only for training levels; unseen levels are predicted via Henderson BLUP.
                train_levels = cargs.get("train_levels", train_levels)
                n_train_levels = len(train_levels)
                self.variational_params[tid] = nn.ParameterList([
                    nn.Parameter(torch.zeros(n_train_levels, n_slopes, embedding_dim)), # mu for train
                    nn.Parameter(torch.ones(n_train_levels, n_slopes, embedding_dim) * -2.0) # log_s for train
                ])
                matrix_tensor = cargs.get("matrix_tensor", None)
                all_levels = cargs.get("levels", train_levels)
                lambda_k = cargs.get("lambda_k", 1e-4)
                self.cov_modules[tid] = KCovariance(
                    n_levels=n_levels,
                    n_slopes=n_slopes,
                    full_levels=all_levels,
                    train_levels=train_levels,
                    matrix_tensor=matrix_tensor,
                    lambda_k=lambda_k
                )

            elif ctype == "GP":
                train_levels = cargs.get("train_levels", train_levels)
                all_levels = cargs.get("levels", train_levels)
                coords = cargs.get("coords", None)
                if coords is None:
                    raise ValueError(f"GP covariance for group {group_name} requires 'coords'.")
                n_train_levels = len(train_levels)
                self.variational_params[tid] = nn.ParameterList([
                    nn.Parameter(torch.zeros(n_train_levels, n_slopes, embedding_dim)), # mu for train
                    nn.Parameter(torch.ones(n_train_levels, n_slopes, embedding_dim) * -2.0) # log_s for train
                ])
                lambda_k = cargs.get("lambda_k", 1e-4)
                self.cov_modules[tid] = GPCovariance(
                    n_levels=n_levels,
                    n_slopes=n_slopes,
                    full_levels=all_levels,
                    train_levels=train_levels,
                    coords=coords,
                    lambda_k=lambda_k
                )

            else:
                # Default: IID covariance
                self.variational_params[tid] = nn.ParameterList([
                    nn.Parameter(torch.zeros(n_levels, n_slopes, embedding_dim)), # mu
                    nn.Parameter(torch.ones(n_levels, n_slopes, embedding_dim) * -2.0) # log_s
                ])
                self.cov_modules[tid] = IIDCovariance(n_levels, n_slopes)

            if self.unknown_strategy == "learned":
                self.unknown_params[tid] = nn.Parameter(torch.zeros(1, n_slopes, embedding_dim))

    def forward(self, X, Z_indices, Z_slopes):
        # X: [batch, fixed_dim]
        # Z_indices: list of [batch] indices
        # Z_slopes: list of [batch, n_slopes]
        
        # Fixed part
        fixed_out = self.fixed_linear(X)
        
        # Random parts
        random_out = torch.zeros_like(fixed_out)
        kl_sum = 0.0
        kl_elems = 0
        n_terms = max(len(self.config), 1)
        
        for i, term in enumerate(self.config):
            tid = term['term_id']
            indices = Z_indices[i] # [batch]
            slopes_val = Z_slopes[i] # [batch, n_slopes]
            
            mu, log_s = self.variational_params[tid]
            
            # KL divergence: q(v) vs N(0, I)
            s = torch.exp(log_s)
            var = s**2
            kl = 0.5 * (var + mu**2 - 1 - 2*log_s).sum()
            kl_sum = kl_sum + kl
            kl_elems += (mu.numel() + log_s.numel())
            
            if self.training:
                noise = torch.randn_like(mu)
                v_input = mu + noise * torch.exp(log_s)
            else:
                # Use mean at evaluation time for deterministic predictions
                v_input = mu
            
            # Apply Covariance Structure L
            cov_module = self.cov_modules[tid]
            if isinstance(cov_module, (KCovariance, GPCovariance)):
                u_expected = cov_module(v_input, indices)  # already [batch, slopes, emb]
                if self.unknown_strategy == "learned":
                    unknown_row = self.unknown_params[tid]
                else:
                    unknown_row = u_expected.mean(dim=0, keepdim=True)
                if (indices == 0).any():
                    u_expected = u_expected.clone()
                    u_expected[indices == 0] = unknown_row
                gathered_u = u_expected
            else:
                u_expected = cov_module(v_input) # [levels, slopes, emb]
                # Reserve index 0 for unseen groups -> configurable strategy
                if self.unknown_strategy == "learned":
                    unknown_row = self.unknown_params[tid]
                else:
                    unknown_row = u_expected.mean(dim=0, keepdim=True)
                u_padded = torch.cat([unknown_row, u_expected], dim=0)
                gathered_u = u_padded.index_select(0, indices.clamp(min=0))
            
            # Weighted sum
            term_contrib = (gathered_u * slopes_val.unsqueeze(-1)).sum(dim=1)
            random_out += term_contrib
            
        kl_out = kl_sum / max(kl_elems, 1) if self.kl_normalize else kl_sum
        random_out_scaled = random_out / n_terms if self.scale_random_by_terms else random_out
        return fixed_out + random_out_scaled, kl_out

    def get_variance_summary(self):
        """
        Returns a dict of term_id -> {slope_idx: sigma_value, type: cov_type}
        """
        summary = {}
        for term_id, mod in self.cov_modules.items():
            # Assume mod has log_sigma or similar
            # IID and AR1 have log_sigma
            # K has log_sigma
            
            # Access params directly? Or rely on convention.
            vals = {}
            if hasattr(mod, 'log_sigma'):
                sigmas = torch.exp(mod.log_sigma).detach().cpu().numpy().flatten()
                vals['sigmas'] = sigmas
            
            if hasattr(mod, 'raw_rho'): # AR1
                rho = torch.tanh(mod.raw_rho).detach().cpu().numpy().flatten()
                vals['rho'] = rho
                
            vals['type'] = mod.__class__.__name__
            summary[term_id] = vals
        return summary

class GEGLULayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class TabMixNNModule(nn.Module):
    def __init__(self, 
                 fixed_input_dim, 
                 compiled_design,
                 embedding_dim=64, 
                 hidden_dims=[64],
                 sam_location="output",
                 sam_dim=None,
                 cov_struct_map=None,
                 mixed_activation="relu",
                 kl_normalize=True,
                 scale_random_by_terms=False,
                 unknown_strategy="learned",
                 sam_residual=True,
                 sam_residual_scale=1.0,
                 sam_max_offdiag=0.9,
                 hidden_dropout=0.0,
                 hidden_layer_norm=False,
                 hidden_residual=True
                 ):
        super().__init__()
        self.sam_location = sam_location
        self.outcomes = compiled_design.outcomes
        self.sam_dim = sam_dim
        self.mixed_activation_name = mixed_activation
        self.mixed_activation = self._resolve_activation(mixed_activation)
        self.kl_normalize = kl_normalize
        self.scale_random_by_terms = scale_random_by_terms
        self.hidden_dropout_rate = hidden_dropout
        self.hidden_layer_norm = hidden_layer_norm
        self.hidden_residual = hidden_residual
        
        # 0. Mixed Effects Block
        # Needs random term config from compiled_design
        random_config = []
        for term in compiled_design.random_terms:
            random_config.append({
                'term_id': term.term_id,
                'n_levels': max(term.level_map.values()) + 1 if term.level_map else 1,
                'n_slopes': len(term.slope_cols),
                'group': term.group_col, # Pass group name for cov struct lookup
                'train_levels': getattr(term, "train_levels", [])
            })
            
        self.mixed_block = MixedEffectsRepresentation(
             fixed_input_dim=fixed_input_dim,
             embedding_dim=embedding_dim,
             random_terms_config=random_config,
             cov_struct_map=cov_struct_map,
             kl_normalize=kl_normalize,
             scale_random_by_terms=scale_random_by_terms,
             unknown_strategy=unknown_strategy
        )

        # Track current representation width
        curr_dim = embedding_dim
        self.dropout = nn.Dropout(hidden_dropout) if hidden_dropout > 0 else None
        
        # SAM First (optional projection to sam_dim)
        self.sam_first = None
        self.first_proj = None
        if sam_location in ["first", "both", "all"] and curr_dim > 1:
            sam_width = sam_dim if sam_dim is not None else curr_dim
            if sam_width != curr_dim:
                self.first_proj = nn.Linear(curr_dim, sam_width)
                curr_dim = sam_width
            self.sam_first = SEMLayer(curr_dim, max_offdiag=sam_max_offdiag, residual=sam_residual, residual_scale=sam_residual_scale)

        # Hidden Blocks
        self.hidden_layers = nn.ModuleList()
        self.hidden_meta = []  # track type and dim
        self.hidden_norms = nn.ModuleList()
        self.sam_hidden = nn.ModuleList()
        
        for h_dim in hidden_dims:
            self.hidden_layers.append(GEGLULayer(curr_dim, h_dim))
            self.hidden_meta.append(("linear", h_dim))
            if hidden_layer_norm:
                self.hidden_norms.append(nn.LayerNorm(curr_dim))
            else:
                self.hidden_norms.append(nn.Identity())
            curr_dim = h_dim
            # SAM All
            if sam_location == "all" and curr_dim > 1:
                sam_layer = SEMLayer(curr_dim, max_offdiag=sam_max_offdiag, residual=sam_residual, residual_scale=sam_residual_scale)
                self.hidden_layers.append(sam_layer)
                self.hidden_meta.append(("sam", curr_dim))
                self.sam_hidden.append(sam_layer)
        
        # Optionally project before output SAM
        self.output_proj = None

        self.sam_output = None
        if sam_location in ["output", "both", "all"] and curr_dim > 1:
            sam_width_out = sam_dim if sam_dim is not None else curr_dim
            if sam_width_out != curr_dim:
                self.output_proj = nn.Linear(curr_dim, sam_width_out)
                curr_dim = sam_width_out
            self.sam_output = SEMLayer(curr_dim, max_offdiag=sam_max_offdiag, residual=sam_residual, residual_scale=sam_residual_scale)
        
        self.last_dim = curr_dim

        # Heads and Families
        self.heads = nn.ModuleDict()
        self.families = nn.ModuleDict()
        for outcome in self.outcomes:
            family = outcome.family
            out_dim = family.n_params
            self.heads[outcome.name] = nn.Linear(self.last_dim, out_dim)
            # Register family as module so its parameters (e.g., Gaussian.log_sigma) are tracked
            self.families[outcome.name] = family

    def get_sam_layer(self, layer_name):
        if layer_name == "first":
            return self.sam_first
        elif layer_name == "output":
            return self.sam_output
        elif layer_name.startswith("hidden_"):
            idx = int(layer_name.split("_")[1])
            if 0 <= idx < len(self.sam_hidden):
                return self.sam_hidden[idx]
        return None
        
    def get_variance_summary(self):
        return self.mixed_block.get_variance_summary()

    def forward(self, X, Z_indices, Z_slopes):
        # 1. Mixed Effects
        x, kl = self.mixed_block(X, Z_indices, Z_slopes)
        if self.mixed_activation is not None:
            x = self.mixed_activation(x)
        
        # 2. SAM First
        if self.sam_first:
            if self.first_proj is not None:
                x = self.first_proj(x)
            x = self.sam_first(x)
            
        # 3. Hidden Layers
        lin_norm_idx = 0
        for i, layer in enumerate(self.hidden_layers):
            meta = self.hidden_meta[i] if i < len(self.hidden_meta) else ("sam", None)
            
            if isinstance(layer, SEMLayer) or meta[0] == "sam":
                # SAM layers handle their own valid residual logic if configured
                x = layer(x)
            else:
                # Dense Block: Pre-Norm -> GEGLU -> Dropout -> Residual
                residual = x if (self.hidden_residual and x.shape[1] == meta[1]) else None
                
                z = x
                if self.hidden_layer_norm:
                    z = self.hidden_norms[lin_norm_idx](z)
                
                # Apply layer (GEGLU usually handles activation internally, but check definition)
                # GEGLULayer includes F.gelu(gate). 
                # If layer is GEGLU, we just call it.
                if isinstance(layer, GEGLULayer):
                     z = layer(z)
                else:
                     z = F.gelu(layer(z)) # Switch to GELU for legacy linear layers
                
                if self.dropout is not None:
                    z = self.dropout(z)
                
                if residual is not None:
                    x = residual + z
                else:
                    x = z
                lin_norm_idx += 1
                
        # 4. Optional projection then SAM Output (Latent)
        if self.output_proj is not None:
            x = self.output_proj(x)
        if self.sam_output:
            x = self.sam_output(x)
            
        # 5. Heads
        outputs = {}
        for name, head in self.heads.items():
            outputs[name] = head(x)
            
        return outputs, kl

    @staticmethod
    def _resolve_activation(name):
        if name is None:
            return None
        name = str(name).lower()
        if name in ["identity", "none", "linear"]:
            return None
        if name == "relu":
            return F.relu
        if name == "gelu":
            return F.gelu
        raise ValueError(f"Unknown mixed_activation '{name}'. Supported: identity/none, relu, gelu.")

    def get_sparsity_loss(self, lambda_sparse=None):
        """
        Compute sparsity penalty with optional layer-specific weights.
        lambda_sparse: float or dict with keys first/hidden/output.
        """
        if lambda_sparse is None:
            lambda_sparse = {"first": 1.0, "hidden": 1.0, "output": 1.0}
        elif not isinstance(lambda_sparse, dict):
            lambda_sparse = {"first": float(lambda_sparse),
                             "hidden": float(lambda_sparse),
                             "output": float(lambda_sparse)}

        loss = 0.0
        if self.sam_first:
            loss += lambda_sparse.get("first", 0.0) * self.sam_first.penalty()
        if self.sam_output:
            loss += lambda_sparse.get("output", 0.0) * self.sam_output.penalty()
        for l in self.hidden_layers:
            if isinstance(l, SEMLayer):
                loss += lambda_sparse.get("hidden", 0.0) * l.penalty()
        return loss
