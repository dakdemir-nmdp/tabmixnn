import torch
import torch.nn as nn
import numpy as np
from ..covariance.kernels import rbf_kernel

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
    
    def _kernel(self, positions, lengthscale):
        """Compute kernel matrix."""
        return rbf_kernel(positions, positions, lengthscale, jitter=self.jitter)

    def _safe_cholesky(self, K):
        """Robust Cholesky with jitter fallback."""
        try:
            return torch.linalg.cholesky(K)
        except RuntimeError:
            jitter = self.jitter
            for _ in range(5):
                jitter *= 10
                try:
                    return torch.linalg.cholesky(K + jitter * torch.eye(K.shape[0], device=K.device))
                except RuntimeError:
                    continue
            # Final fallback: eigen-based PSD projection
            eigvals, eigvecs = torch.linalg.eigh(K)
            eigvals = torch.clamp(eigvals, min=1e-6)
            K_psd = eigvecs @ torch.diag(eigvals) @ eigvecs.T
            return torch.linalg.cholesky(K_psd + 1e-6 * torch.eye(K.shape[0], device=K.device))
    
    def get_covariance_1d(self):
        """Get covariance matrix for 1D layout."""
        l = torch.exp(self.log_lengthscale)
        return self._kernel(self.positions, l)
    
    def get_covariance_2d(self):
        """Get Kronecker factorized covariance for 2D grid."""
        l_rows = torch.exp(self.log_lengthscale_rows)
        l_cols = torch.exp(self.log_lengthscale_cols)
        
        K_rows = self._kernel(self.pos_rows, l_rows)
        K_cols = self._kernel(self.pos_cols, l_cols)
        
        return K_rows, K_cols
    
    def sample_noise(self, batch_size, device):
        """Sample ε ~ N(0, Σ_spatial)."""
        amplitude = torch.exp(self.log_amplitude)
        
        if self.grid_shape is not None:
            # 2D: Kronecker sampling
            K_rows, K_cols = self.get_covariance_2d()
            
            # Cholesky decomposition
            L_rows = self._safe_cholesky(K_rows)
            L_cols = self._safe_cholesky(K_cols)
            
            # Sample standard normal
            z = torch.randn(batch_size, self.U, self.V, device=device)
            
            # Apply Kronecker structure: (L_rows ⊗ L_cols) z
            # Efficient: z_out = L_rows @ z @ L_cols^T
            noise = torch.einsum('ij,bjk,kl->bil', L_rows, z, L_cols.T)
            noise = noise.reshape(batch_size, self.dim)
        else:
            # 1D: standard sampling
            K = self.get_covariance_1d()
            L = self._safe_cholesky(K)
            
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


class SPDEKernel(nn.Module):
    """
    SPDE-based spatial kernel using precision matrix representation.
    
    Implements Matérn-like fields via:
    Q = τ(L + κ²I)^α
    
    where:
    - L is the graph Laplacian (encodes 2D grid topology)
    - κ controls the range (inverse lengthscale)
    - τ controls the marginal variance
    - α controls smoothness (α=1 for exponential, α=2 for Matérn ν=1)
    """
    
    def __init__(self, grid_shape, alpha=2, kappa_init=0.3, tau_init=1.0, jitter=1e-4):
        """
        Args:
            grid_shape: (U, V) tuple for 2D grid, or int for 1D
            alpha: Smoothness parameter (1 or 2)
            kappa_init: Initial range parameter (inverse lengthscale)
            tau_init: Initial precision scaling
            jitter: Numerical stability
        """
        super().__init__()
        
        if isinstance(grid_shape, int):
            grid_shape = (grid_shape, 1)
        
        self.grid_shape = grid_shape
        self.U, self.V = grid_shape
        self.dim = self.U * self.V
        self.alpha = alpha
        self.jitter = jitter
        
        # Learnable parameters (log scale for positivity)
        self.log_kappa = nn.Parameter(torch.tensor(kappa_init).log())
        self.log_tau = nn.Parameter(torch.tensor(tau_init).log())
        
        # Build graph Laplacian for 2D grid
        L = self._build_laplacian_2d()
        self.register_buffer("L", L)
        
    def _build_laplacian_2d(self):
        """
        Build graph Laplacian for 2D grid with 4-connectivity.
        L = D - A where D is degree matrix, A is adjacency.
        """
        U, V = self.U, self.V
        n = U * V
        
        # Build adjacency matrix for 2D grid
        A = torch.zeros(n, n)
        
        for i in range(U):
            for j in range(V):
                idx = i * V + j
                
                # Right neighbor
                if j < V - 1:
                    neighbor = i * V + (j + 1)
                    A[idx, neighbor] = 1
                    A[neighbor, idx] = 1
                
                # Down neighbor
                if i < U - 1:
                    neighbor = (i + 1) * V + j
                    A[idx, neighbor] = 1
                    A[neighbor, idx] = 1
        
        # Degree matrix
        D = torch.diag(A.sum(dim=1))
        
        # Laplacian
        L = D - A
        return L
    
    def get_precision(self):
        """
        Compute precision matrix Q = τ(L + κ²I)^α
        """
        kappa = torch.exp(self.log_kappa)
        tau = torch.exp(self.log_tau)
        
        I = torch.eye(self.dim, device=self.L.device)
        
        # Q = (L + κ²I)
        Q_base = self.L + (kappa ** 2) * I
        
        if self.alpha == 1:
            Q = tau * Q_base
        elif self.alpha == 2:
            Q = tau * torch.matmul(Q_base, Q_base)
        else:
            # General case: matrix power (expensive)
            Q = tau * torch.linalg.matrix_power(Q_base, self.alpha)
        
        # Add jitter for numerical stability
        Q = Q + self.jitter * I
        
        return Q
    
    def get_covariance(self):
        """
        Compute covariance matrix Σ = Q^{-1}
        """
        Q = self.get_precision()
        Sigma = torch.linalg.inv(Q)
        return Sigma
    
    def sample(self, batch_size=1, device=None):
        """
        Sample from the SPDE field: ε ~ N(0, Σ) where Σ = Q^{-1}
        
        Uses precision-based sampling for efficiency:
        1. Sample z ~ N(0, I)
        2. Solve chol(Q).T ε = z for ε
        """
        if device is None:
            device = self.L.device
        
        Q = self.get_precision()
        
        # Sample standard normal
        z = torch.randn(batch_size, self.dim, device=device)
        
        # Solve Qε = z (more efficient than computing Σ explicitly)
        # Q is symmetric positive definite, use Cholesky
        try:
            L_chol = torch.linalg.cholesky(Q)
            # One triangular solve: Cov(L^-T z) = Q^-1.
            epsilon = torch.linalg.solve_triangular(L_chol.T, z.T, upper=True)
            epsilon = epsilon.T
        except RuntimeError:
            # Fallback: direct inverse (slower but more stable)
            Sigma = self.get_covariance()
            L_chol = torch.linalg.cholesky(Sigma)
            epsilon = torch.matmul(z, L_chol.T)
        
        return epsilon
    
    def forward(self, batch_size=1):
        """Sample from the field."""
        return self.sample(batch_size)
    
    def penalty(self):
        """
        Regularization penalty to prevent extreme parameters.
        Encourages moderate range and variance.
        """
        kappa = torch.exp(self.log_kappa)
        tau = torch.exp(self.log_tau)
        
        # Soft constraints
        kappa_penalty = (kappa - 0.3) ** 2
        tau_penalty = (tau - 1.0) ** 2
        
        return 0.01 * (kappa_penalty + tau_penalty)


class KroneckerSPDEKernel(nn.Module):
    """
    Efficient SPDE kernel using N-dimensional Kronecker factorization.
    
    For an N-dimensional grid (d_1, ..., d_n), the precision is approximated as:
    Q = τ * [ (Q_1 ⊗ ... ⊗ Q_n) ]
    where Q_i = (L_i + κ² I) is the 1D precision for dimension i.
    
    This allows sampling with complexity O(Σ d_i³) instead of O((Π d_i)³),
    by performing sequential tensor contractions (Tucker product).
    """
    
    def __init__(self, grid_shape, alpha=2, kappa_init=0.3, tau_init=1.0, jitter=1e-4):
        super().__init__()
        
        if isinstance(grid_shape, int):
            grid_shape = (grid_shape,)
        
        self.grid_shape = tuple(grid_shape)
        self.dims = self.grid_shape
        self.n_dims = len(self.dims)
        self.total_dim = int(np.prod(self.dims))
        self.dim = self.total_dim # Alias for interface compatibility
        
        # Legacy aliases for 2D
        if self.n_dims == 2:
            self.U, self.V = self.dims
        else:
            self.U, self.V = None, None # Or raise error if accessed?
        
        self.alpha = alpha
        self.jitter = jitter
        
        # Learnable parameters
        self.log_kappa = nn.Parameter(torch.tensor(kappa_init).log())
        self.log_tau = nn.Parameter(torch.tensor(tau_init).log())
        
        # Build 1D Laplacians for each dimension
        # Store as buffers named L_0, L_1, ... to ensure state_dict compatibility
        for i, d in enumerate(self.dims):
            L_1d = self._build_laplacian_1d(d)
            self.register_buffer(f"L_{i}", L_1d)
    
    def _build_laplacian_1d(self, n):
        """Build 1D chain Laplacian."""
        if n == 1:
            return torch.zeros(1, 1)
            
        L = torch.zeros(n, n)
        for i in range(n):
            L[i, i] = 2 if 0 < i < n - 1 else 1
            if i > 0:
                L[i, i-1] = -1
            if i < n - 1:
                L[i, i+1] = -1
        return L
    
    def sample(self, batch_size=1, device=None):
        """
        N-dimensional Kronecker sampling using sequential contractions.
        
        ε = z ×_1 L_1 ×_2 L_2 ... ×_n L_n
        where z ~ N(0, I) and L_i are Cholesky factors of Σ_i.
        """
        if device is None:
            device = getattr(self, "L_0").device
        
        kappa = torch.exp(self.log_kappa)
        tau = torch.exp(self.log_tau)
        
        # 1. Compute Cholesky factor for each dimension
        L_factors = []
        for i, d in enumerate(self.dims):
            L_base = getattr(self, f"L_{i}")
            I = torch.eye(d, device=device)
            
            # 1D Precision Q_i = (L + κ²I)
            # Note: For Kronecker product of precision matrices to sum correctly to 2D operator,
            # strict interpretation varies. Here we follow the separable covariance assumption:
            # Cov(x, y) = Π K_i(x_i, y_i)
            # which implies Q = Q_1 ⊗ ... ⊗ Q_n (inverse of tensor product is tensor product of inverses)
            
            Q_i = L_base + (kappa ** 2) * I
            
            Q_i = torch.linalg.matrix_power(Q_i, self.alpha)
            
            # Distribute tau equally across dimensions (geometric mean)
            # Or just apply to final result. Let's apply to final result for simplicity.
            # But we need Q_i to be inverted.
            
            # Add jitter before inversion
            Q_i = Q_i + self.jitter * I
            
            # Invert to get Covariance Σ_i
            # Note: For small 1D dims (e.g. 8x8), direct inverse is fast.
            Sigma_i = torch.linalg.inv(Q_i)
            
            # Cholesky L_i
            try:
                L_i = torch.linalg.cholesky(Sigma_i)
            except RuntimeError:
                # Fallback for numerical stability
                L_i = torch.linalg.cholesky(Sigma_i + 1e-4 * I)
                
            L_factors.append(L_i)
            
        # 2. Sample standard normal tensor
        # Shape: [batch, d_1, d_2, ..., d_n]
        z_shape = (batch_size,) + self.dims
        z = torch.randn(z_shape, device=device)
        
        # 3. Sequential Tensor Contractions (Tucker Product)
        # We process dimensions 1 to N (skipping batch dim 0)
        # z is [batch, d1, d2, ...]
        
        curr = z
        for i, L_i in enumerate(L_factors):
            # We want to contract along dimension (i + 1) of curr
            # Mode-n product: Y = X ×_n U  =>  Y_...j... = sum_k X_...k... * U_jk
            # In PyTorch: tensordot or matmul with permutations.
            # Efficient way: move target dim to last, matmul, move back.
            
            # Permute: move dim (i+1) to last
            # dims: 0, 1, ..., i+1, ..., n_dims
            # perm: 0, 1, ..., i, i+2, ..., n_dims, i+1
            
            target_dim = i + 1
            
            # Move target dim to back
            curr = curr.transpose(target_dim, -1)
            
            # Apply linear map L_i (which is d_i x d_i)
            # We want: y = L_i @ x (where x is column vector)
            # But here x is in last dim as row? 
            # If z ~ N(0, I), and we want Cov = L L^T, then x = L z.
            # L_i shape: [d_i, d_i]. 
            # We perform: curr @ L_i.T
            curr = torch.matmul(curr, L_i.T)
            
            # Move dim back
            curr = curr.transpose(target_dim, -1)
            
        # 4. Apply scale tau (sqrt(tau) to match variance scaling)
        # Q_total = tau * Q_kron  => Sigma_total = (1/tau) * Sigma_kron
        # So std dev scale is 1/sqrt(tau).
        # We defined log_tau as precision scaling, so lower std dev.
        scale = torch.exp(-0.5 * self.log_tau)
        epsilon = curr.reshape(batch_size, self.total_dim) * scale
        
        return epsilon
    
    def forward(self, batch_size=1):
        return self.sample(batch_size)
    
    def penalty(self):
        kappa = torch.exp(self.log_kappa)
        tau = torch.exp(self.log_tau)
        return 0.01 * ((kappa - 0.3) ** 2 + (tau - 1.0) ** 2)


def create_spde_kernel(grid_shape, use_kronecker=True, **kwargs):
    """
    Factory function to create appropriate SPDE kernel.
    
    Args:
        grid_shape: tuple of ints (d_1, ..., d_n), or scalar int
        use_kronecker: If True, use efficient Kronecker factorization (recommended)
        **kwargs: Passed to kernel constructor
    
    Returns:
        SPDEKernel or KroneckerSPDEKernel
    """
    if isinstance(grid_shape, int):
        grid_shape = (grid_shape,)
    
    grid_shape = tuple(grid_shape)
    
    if use_kronecker:
        return KroneckerSPDEKernel(grid_shape, **kwargs)
    if len(grid_shape) == 1:
        grid_shape = (grid_shape[0], 1)
    if len(grid_shape) != 2:
        raise ValueError("Dense SPDE supports one- or two-dimensional grids")
    return SPDEKernel(grid_shape, **kwargs)
