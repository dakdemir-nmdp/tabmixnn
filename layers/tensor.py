"""
Tensor-Structured Output Layers

Reshapes flat layer outputs into tensors with separable covariance structure.

Key idea: A d-dimensional layer output is reshaped into a tensor of shape (d₁, d₂, ..., dₖ)
where d = d₁ × d₂ × ... × dₖ, and we model covariance as:

Vec(H) ~ N(Vec(μ), Ω₁ ⊗ Ω₂ ⊗ ... ⊗ Ωₖ)

where each Ωᵢ (size dᵢ × dᵢ) captures correlations along dimension i.

Benefits:
- Structured sparsity in covariance
- Interpretable organization of neurons
- Efficient: O(∑dᵢ²) parameters instead of O(d²)
- Learnable structure (AR, factor, etc.)

Example:
    64 neurons → reshape to (8, 8) → model row and column correlations separately
    128 neurons → reshape to (4, 4, 8) → 3-way structure
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union
import numpy as np
from .activations import get_activation


class TensorStructuredLayer(nn.Module):
    """
    Layer with tensor-structured output and separable covariance.
    
    Output: h ∈ ℝ^d is reshaped to H ∈ ℝ^(d₁ × d₂ × ... × dₖ)
    Covariance: Vec(H) ~ N(Vec(μ), Ω₁ ⊗ Ω₂ ⊗ ... ⊗ Ωₖ)
    
    Each Ωᵢ can be:
    - 'identity': Independent along this dimension
    - 'diagonal': Different variance per position
    - 'ar1': AR(1) structure (adjacent positions correlated)
    - 'factor': Low-rank factor structure
    - 'full': Unrestricted covariance
    
    Example:
        >>> # 64 neurons organized as 8×8 with row/column structure
        >>> layer = TensorStructuredLayer(
        ...     input_dim=32,
        ...     output_shape=(8, 8),
        ...     mode_structures=['ar1', 'factor'],
        ... )
        >>> H, cov_loss = layer(X)  # H is [batch, 8, 8]
    """
    
    def __init__(
        self,
        input_dim: int,
        output_shape: Tuple[int, ...],  # (d₁, d₂, ..., dₖ)
        mode_structures: List[str],  # Structure for each dimension
        mode_ranks: Optional[List[int]] = None,  # Rank for factor structures
        activation: str = 'gelu',
        use_covariance_loss: bool = True,
        covariance_weight: float = 0.01,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_shape = output_shape
        self.output_dim = int(np.prod(output_shape))
        self.n_modes = len(output_shape)
        self.mode_structures = mode_structures
        self.use_covariance_loss = use_covariance_loss
        self.covariance_weight = covariance_weight
        
        assert len(mode_structures) == self.n_modes, \
            f"Need {self.n_modes} structures for shape {output_shape}"
        
        # Main linear layer
        self.linear = nn.Linear(input_dim, self.output_dim)
        
        # Activation
        self.activation = get_activation(activation)
        
        # Covariance structures for each mode
        self.mode_covariances = nn.ModuleList()
        for i, (dim_size, structure) in enumerate(zip(output_shape, mode_structures)):
            rank = mode_ranks[i] if mode_ranks and i < len(mode_ranks) else max(1, dim_size // 4)
            cov = self._create_mode_covariance(dim_size, structure, rank)
            self.mode_covariances.append(cov)
    

    
    def _create_mode_covariance(self, dim: int, structure: str, rank: int):
        """Create covariance structure for one mode."""
        return ModeCovariance(dim, structure, rank)
    
    def forward(
        self,
        X: torch.Tensor,  # [batch, input_dim]
        return_flat: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with tensor-structured output.
        
        Args:
            X: Input [batch, input_dim]
            return_flat: If True, return flattened output
        
        Returns:
            H: Output [batch, d₁, d₂, ..., dₖ] or [batch, d] if return_flat
            cov_loss: Covariance regularization loss (if use_covariance_loss)
        """
        batch_size = X.shape[0]
        
        # Linear + activation
        h_flat = self.activation(self.linear(X))  # [batch, d]
        
        # Reshape to tensor
        H = h_flat.reshape(batch_size, *self.output_shape)
        
        # Compute covariance regularization loss
        if self.use_covariance_loss:
            cov_loss = self._compute_covariance_loss(H)
            
            if return_flat:
                return h_flat, cov_loss
            else:
                return H, cov_loss
        else:
            if return_flat:
                return h_flat
            else:
                return H
    
    def _compute_covariance_loss(self, H: torch.Tensor) -> torch.Tensor:
        """
        Encourage empirical covariance to match the structured model.
        
        For each mode i, compare empirical vs. model covariance.
        """
        batch_size = H.shape[0]
        total_loss = 0.0
        
        # For each mode, compute marginal covariance
        for mode_idx, cov_module in enumerate(self.mode_covariances):
            # Flatten all dimensions except current mode
            # H shape: [batch, d₁, ..., dₖ]
            # Move mode_idx to position 1: [batch, dᵢ, other_dims]
            perm = [0, mode_idx + 1] + [j + 1 for j in range(self.n_modes) if j != mode_idx]
            H_perm = H.permute(*perm)
            
            # Flatten other dimensions
            mode_dim = self.output_shape[mode_idx]
            H_mode = H_perm.reshape(batch_size, mode_dim, -1)  # [batch, dᵢ, prod(other)]
            
            # Average over other dimensions and batch
            H_mode_avg = H_mode.mean(dim=-1)  # [batch, dᵢ]
            
            # Compute empirical covariance
            H_centered = H_mode_avg - H_mode_avg.mean(dim=0, keepdim=True)
            emp_cov = H_centered.T @ H_centered / batch_size  # [dᵢ, dᵢ]
            
            # Model covariance
            model_cov = cov_module.get_matrix()
            
            # Frobenius norm loss
            loss_i = torch.norm(emp_cov - model_cov, p='fro') ** 2
            total_loss += loss_i
        
        return self.covariance_weight * total_loss / self.n_modes
    
    def get_full_covariance(self) -> torch.Tensor:
        """
        Compute full d×d covariance matrix as Kronecker product.
        
        Warning: Can be expensive for large d!
        
        Returns:
            Full covariance [d, d] where d = ∏dᵢ
        """
        # Start with first mode
        full_cov = self.mode_covariances[0].get_matrix()
        
        # Kroneck er product with remaining modes
        for cov_module in self.mode_covariances[1:]:
            mode_cov = cov_module.get_matrix()
            full_cov = torch.kron(full_cov, mode_cov)
        
        return full_cov
    
    def sample_structured(
        self,
        mu: torch.Tensor,  # [batch, d] or [batch, d₁, ..., dₖ]
    ) -> torch.Tensor:
        """
        Sample from the structured distribution.
        
        H ~ N(μ, Ω₁ ⊗ ... ⊗ Ωₖ)
        
        Args:
            mu: Mean (flat or tensor)
        
        Returns:
            Sample with same shape as mu
        """
        batch_size = mu.shape[0]
        
        # Ensure tensor shape
        if mu.dim() == 2:
            mu = mu.reshape(batch_size, *self.output_shape)
        
        # Sample standard normal tensor
        Z = torch.randn_like(mu)
        
        # Apply Cholesky factors via mode products
        sample = Z
        for mode_idx, cov_module in enumerate(self.mode_covariances):
            # Get Cholesky factor
            L = cov_module.get_cholesky()
            
            # Apply along current mode
            # Move mode to last position
            perm_to_end = list(range(sample.dim()))
            perm_to_end[mode_idx + 1], perm_to_end[-1] = perm_to_end[-1], perm_to_end[mode_idx + 1]
            sample = sample.permute(*perm_to_end)
            
            # Apply L
            sample = sample @ L.T
            
            # Move back
            perm_back = list(range(sample.dim()))
            perm_back[mode_idx + 1], perm_back[-1] = perm_back[-1], perm_back[mode_idx + 1]
            sample = sample.permute(*perm_back)
        
        return mu + sample


class ModeCovariance(nn.Module):
    """
    Covariance structure for one mode/dimension.
    
    Supports:
    - identity: Ω = I
    - diagonal: Ω = diag(exp(log_var))
    - ar1: Ω_ij = ρ^|i-j|
    - factor: Ω = D + FF' (low-rank)
    - full: Ω = LL' (Cholesky)
    """
    
    def __init__(self, dim: int, structure: str, rank: Optional[int] = None):
        super().__init__()
        
        self.dim = dim
        self.structure = structure
        self.rank = min(rank if rank is not None else max(1, dim // 4), dim)
        
        if structure == 'identity':
            pass  # No parameters
        
        elif structure == 'diagonal':
            self.log_var = nn.Parameter(torch.zeros(dim))
        
        elif structure == 'ar1':
            self.rho_raw = nn.Parameter(torch.tensor(0.0))
        
        elif structure == 'factor':
            self.log_diag = nn.Parameter(torch.zeros(dim))
            self.factor = nn.Parameter(torch.randn(dim, self.rank) * 0.1)
        
        elif structure == 'full':
            self.L_raw = nn.Parameter(torch.eye(dim))
        
        else:
            raise ValueError(f"Unknown structure: {structure}")
    
    @property
    def rho(self) -> Optional[torch.Tensor]:
        """AR(1) correlation parameter."""
        if hasattr(self, 'rho_raw'):
            return torch.tanh(self.rho_raw)
        return None
    
    def get_matrix(self) -> torch.Tensor:
        """Return covariance matrix."""
        device = next(self.parameters()).device if self.structure != 'identity' else None
        
        if self.structure == 'identity':
            return torch.eye(self.dim, device=device) if device else torch.eye(self.dim)
        
        elif self.structure == 'diagonal':
            return torch.diag(torch.exp(self.log_var))
        
        elif self.structure == 'ar1':
            rho = self.rho
            idx = torch.arange(self.dim, dtype=torch.float32, device=rho.device)
            diff = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
            return rho ** diff
        
        elif self.structure == 'factor':
            D = torch.diag(torch.exp(self.log_diag))
            return D + self.factor @ self.factor.T
        
        elif self.structure == 'full':
            L = torch.tril(self.L_raw)
            return L @ L.T + 1e-6 * torch.eye(self.dim, device=L.device)
    
    def get_cholesky(self) -> torch.Tensor:
        """Return Cholesky factor L such that Ω = LL'."""
        Omega = self.get_matrix()
        try:
            return torch.linalg.cholesky(Omega + 1e-6 * torch.eye(self.dim, device=Omega.device))
        except:
            # Fallback: use eigendecomposition
            eigvals, eigvecs = torch.linalg.eigh(Omega)
            eigvals = torch.clamp(eigvals, min=1e-6)
            return eigvecs @ torch.diag(torch.sqrt(eigvals))


class MatrixStructuredLayer(TensorStructuredLayer):
    """
    Specialized 2D case: output as matrix with row/column structure.
    
    Useful for:
    - Similarity matrices
    - Attention-like outputs
    - Spatial grids
   
    Example:
        >>> # 64 neurons as 8×8 matrix
        >>> layer = MatrixStructuredLayer(
        ...     input_dim=32,
        ...     n_rows=8,
        ...     n_cols=8,
        ...     row_structure='ar1',
        ...     col_structure='factor',
        ... )
    """
    
    def __init__(
        self,
        input_dim: int,
        n_rows: int,
        n_cols: int,
        row_structure: str = 'ar1',
        col_structure: str = 'ar1',
        row_rank: int = 5,
        col_rank: int = 5,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            output_shape=(n_rows, n_cols),
            mode_structures=[row_structure, col_structure],
            mode_ranks=[row_rank, col_rank],
            **kwargs,
        )
        
        self.n_rows = n_rows
        self.n_cols = n_cols


class Tensor3DStructuredLayer(TensorStructuredLayer):
    """
    Specialized 3D case: output as 3D tensor.
    
    Useful for:
    - 3D conv-like outputs
    - Multi-channel grids
    - Volumetric data
    
    Example:
        >>> # 128 neurons as 4×4×8 tensor
        >>> layer = Tensor3DStructuredLayer(
        ...     input_dim=64,
        ...     shape=(4, 4, 8),
        ...     structures=['ar1', 'ar1', 'factor'],
        ... )
    """
    
    def __init__(
        self,
        input_dim: int,
        shape: Tuple[int, int, int],
        structures: List[str],
        ranks: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            output_shape=shape,
            mode_structures=structures,
            mode_ranks=ranks,
            **kwargs,
        )
