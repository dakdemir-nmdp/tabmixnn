"""
SPDE (Stochastic Partial Differential Equation) Kernels for SAM-NN.

Implements Matérn field approximations via precision matrices:
Q = τ(L + κ²I)^α where L is the graph Laplacian.

This provides a mathematically rigorous spatial prior for 2D latent manifolds.
"""

import torch
import torch.nn as nn
import numpy as np


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
        2. Solve Qε = z for ε
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
            # Solve L L^T ε = z
            # First solve L y = z
            y = torch.linalg.solve_triangular(L_chol, z.T, upper=False)
            # Then solve L^T ε = y
            epsilon = torch.linalg.solve_triangular(L_chol.T, y, upper=True)
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
    Efficient SPDE kernel using Kronecker factorization for 2D grids.
    
    Q = τ(L_rows ⊗ I_V + I_U ⊗ L_cols + κ²I)^α
    
    This is much more efficient than full SPDE for large grids.
    Complexity: O(U³ + V³) instead of O((UV)³)
    """
    
    def __init__(self, grid_shape, alpha=2, kappa_init=0.3, tau_init=1.0, jitter=1e-4):
        super().__init__()
        
        self.grid_shape = grid_shape
        self.U, self.V = grid_shape
        self.dim = self.U * self.V
        self.alpha = alpha
        self.jitter = jitter
        
        # Learnable parameters
        self.log_kappa = nn.Parameter(torch.tensor(kappa_init).log())
        self.log_tau = nn.Parameter(torch.tensor(tau_init).log())
        
        # Build 1D Laplacians
        L_rows = self._build_laplacian_1d(self.U)
        L_cols = self._build_laplacian_1d(self.V)
        
        self.register_buffer("L_rows", L_rows)
        self.register_buffer("L_cols", L_cols)
    
    def _build_laplacian_1d(self, n):
        """Build 1D chain Laplacian."""
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
        Efficient Kronecker sampling.
        
        For Σ = Σ_rows ⊗ Σ_cols:
        ε = (L_rows ⊗ L_cols) z
        where L_rows, L_cols are Cholesky factors.
        """
        if device is None:
            device = self.L_rows.device
        
        kappa = torch.exp(self.log_kappa)
        tau = torch.exp(self.log_tau)
        
        # Build precision matrices for rows and cols
        I_U = torch.eye(self.U, device=device)
        I_V = torch.eye(self.V, device=device)
        
        Q_rows = self.L_rows + (kappa ** 2) * I_U
        Q_cols = self.L_cols + (kappa ** 2) * I_V
        
        if self.alpha == 2:
            Q_rows = torch.matmul(Q_rows, Q_rows)
            Q_cols = torch.matmul(Q_cols, Q_cols)
        
        Q_rows = tau * Q_rows + self.jitter * I_U
        Q_cols = tau * Q_cols + self.jitter * I_V
        
        # Invert to get covariances
        Sigma_rows = torch.linalg.inv(Q_rows)
        Sigma_cols = torch.linalg.inv(Q_cols)
        
        # Cholesky factors
        L_rows = torch.linalg.cholesky(Sigma_rows)
        L_cols = torch.linalg.cholesky(Sigma_cols)
        
        # Sample standard normal
        z = torch.randn(batch_size, self.U, self.V, device=device)
        
        # Apply Kronecker structure: ε = (L_rows ⊗ L_cols) z
        # Efficient: ε = L_rows @ z @ L_cols^T
        epsilon = torch.einsum('ij,bjk,kl->bil', L_rows, z, L_cols.T)
        epsilon = epsilon.reshape(batch_size, self.dim)
        
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
        grid_shape: (U, V) tuple or int
        use_kronecker: If True and 2D, use efficient Kronecker factorization
        **kwargs: Passed to kernel constructor
    
    Returns:
        SPDEKernel or KroneckerSPDEKernel
    """
    if isinstance(grid_shape, int):
        grid_shape = (grid_shape, 1)
    
    U, V = grid_shape
    
    # Use Kronecker if 2D and requested
    if use_kronecker and V > 1:
        return KroneckerSPDEKernel(grid_shape, **kwargs)
    else:
        return SPDEKernel(grid_shape, **kwargs)


__all__ = ['SPDEKernel', 'KroneckerSPDEKernel', 'create_spde_kernel']
