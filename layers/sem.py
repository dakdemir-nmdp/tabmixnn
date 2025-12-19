"""
Lightweight SEM layers for spatial and manifold architectures.

This module provides position-aware SEM layers optimized for ManifoldSAMNN and encoders.
These layers implement y = (I-B)^{-1}x with learned DAG structure.

When to use:
- ManifoldBackbone with spatial structure (PositionalSEMLayer)
- Simple SEM constraints in encoders (SEMLayer)
- Position-aware dependencies in neuron grids

For advanced attention-based structural models (GM-SEM architecture),
see `structural.py` which provides HybridStructuralModel with multi-head attention.

Classes:
    SEMLayer: Basic SEM with gated lower-triangular B matrix
    PositionalSEMLayer: SEM with RBF kernel prior for spatial/grid layouts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
    def get_B(self, deterministic=False): # deterministic arg kept for API stability
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
        # Use generic solve to avoid potential solve_triangular segfaults on some platforms
        y_t = torch.linalg.solve(A, x_t)
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
