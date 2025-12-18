"""
Universal SAM-NN: Hierarchical Latent Gaussian Model with SPDE Manifolds.

This implements the most general form of SAM-NN as a compositional,
graph-based latent variable system with:
- SPDE-based 2D latent manifolds
- Persistent random effects across all layers
- Recursive connections (Layer L can see Layer 1)
- Causal SEM output header
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
from .spde import create_spde_kernel
from .model import SEMLayer, PositionalSEMLayer


class SAMBlock(nn.Module):
    """
    A single SAM manifold block combining:
    1. Fixed effects (structural map)
    2. Random effects (cluster-specific)
    3. SPDE topology (spatial correlation)
    4. Optional SEM coordination
    
    Implements: h^(l) = g(B^(l))^{-1} [F^(l)(h^(<l)) + Z^(l)b + ε^(l)]
    """
    
    def __init__(
        self,
        grid_shape: Tuple[int, int],
        input_dim: int,
        random_effects_config: Optional[List[Dict]] = None,
        use_spde: bool = True,
        use_sem: bool = False,
        activation: str = "relu",
        spde_alpha: int = 2,
        spde_kappa_init: float = 0.3,
        use_kronecker: bool = True,
        dropout: float = 0.0,
    ):
        """
        Args:
            grid_shape: (U, V) for 2D manifold
            input_dim: Dimension of input from previous layers
            random_effects_config: List of random effect specifications
            use_spde: Whether to add SPDE spatial noise
            use_sem: Whether to add SEM coordination
            activation: Activation function name
            spde_alpha: SPDE smoothness (1 or 2)
            spde_kappa_init: SPDE range parameter
            use_kronecker: Use efficient Kronecker SPDE
            dropout: Dropout rate
        """
        super().__init__()
        
        self.grid_shape = grid_shape
        self.U, self.V = grid_shape
        self.manifold_dim = self.U * self.V
        self.input_dim = input_dim
        self.use_spde = use_spde
        self.use_sem = use_sem
        
        # Fixed effects: map input to manifold
        self.W_fixed = nn.Linear(input_dim, self.manifold_dim)
        
        # Random effects (persistent across layers)
        self.random_effects_config = random_effects_config or []
        self.random_projections = nn.ModuleDict()
        
        for config in self.random_effects_config:
            term_id = config['term_id']
            n_levels = config['n_levels']
            # Embedding for each level
            self.random_projections[term_id] = nn.Embedding(
                n_levels, self.manifold_dim
            )
        
        # SPDE kernel for spatial correlation
        if use_spde:
            self.spde_kernel = create_spde_kernel(
                grid_shape=grid_shape,
                use_kronecker=use_kronecker,
                alpha=spde_alpha,
                kappa_init=spde_kappa_init,
            )
        else:
            self.spde_kernel = None
        
        # Optional SEM coordination
        if use_sem:
            self.sem = PositionalSEMLayer(
                dim=self.manifold_dim,
                grid_shape=grid_shape,
                lengthscale_init=3.0
            )
        else:
            self.sem = None
        
        # Activation
        self.activation = self._resolve_activation(activation)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Layer norm (optional, for stability)
        self.layer_norm = nn.LayerNorm(self.manifold_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        random_indices: Optional[Dict[str, torch.Tensor]] = None,
        return_manifold: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through SAM block.
        
        Args:
            x: Input tensor [batch, input_dim]
            random_indices: Dict of {term_id: indices [batch]} for random effects
            return_manifold: If True, return both flat and 2D manifold
        
        Returns:
            h: Output tensor [batch, manifold_dim]
            h_2d: (optional) 2D manifold [batch, U, V]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # 1. Fixed effects
        mu = self.W_fixed(x)
        
        # 2. Random effects (persistent)
        if random_indices:
            for term_id, indices in random_indices.items():
                if term_id in self.random_projections:
                    random_contrib = self.random_projections[term_id](indices)
                    mu = mu + random_contrib
        
        # 3. SPDE spatial noise (stochastic topology)
        if self.spde_kernel is not None and self.training:
            epsilon = self.spde_kernel.sample(batch_size, device)
            mu = mu + epsilon
        
        # 4. Activation
        if self.activation is not None:
            h = self.activation(mu)
        else:
            h = mu
        
        # 5. Layer norm
        h = self.layer_norm(h)
        
        # 6. Dropout
        if self.dropout is not None:
            h = self.dropout(h)
        
        # 7. Optional SEM coordination
        if self.sem is not None:
            h = self.sem(h)
        
        if return_manifold:
            # Reshape to 2D manifold
            h_2d = h.view(batch_size, self.U, self.V)
            return h, h_2d
        else:
            return h
    
    def penalty(self):
        """Regularization penalty from SPDE and SEM."""
        penalty = 0.0
        if self.spde_kernel is not None:
            penalty += self.spde_kernel.penalty()
        if self.sem is not None:
            penalty += self.sem.penalty()
        return penalty
    
    @staticmethod
    def _resolve_activation(name):
        if name is None or name == "none":
            return None
        name = str(name).lower()
        if name == "relu":
            return F.relu
        if name == "gelu":
            return F.gelu
        if name == "tanh":
            return torch.tanh
        raise ValueError(f"Unknown activation: {name}")


class UniversalSAMNN(nn.Module):
    """
    Universal SAM-NN: Hierarchical Latent Gaussian Model.
    
    Architecture:
    1. LME4-style input layer (mixed effects)
    2. Stack of SAM manifolds with recursive connections
    3. Causal SEM output header
    
    General form:
    h^(l) = g(B^(l))^{-1} [F^(l)(h^(<l)) + Z^(l)b + ε^(l)]
    """
    
    def __init__(
        self,
        input_dim: int,
        manifold_configs: List[Dict],
        random_effects_config: List[Dict],
        output_sem_spec: Optional[str] = None,
        use_recursive: bool = True,
        aggregation: str = "concat",  # concat, sum, attention
    ):
        """
        Args:
            input_dim: Dimension of fixed input features
            manifold_configs: List of dicts specifying each SAM manifold
                Each dict should have: grid_shape, use_spde, use_sem, etc.
            random_effects_config: Specification of random effects (persistent)
            output_sem_spec: Lavaan-style formula for output SEM
            use_recursive: If True, each layer sees all previous layers
            aggregation: How to combine previous layers (concat/sum/attention)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.random_effects_config = random_effects_config
        self.use_recursive = use_recursive
        self.aggregation = aggregation
        self.n_manifolds = len(manifold_configs)
        
        # Build SAM manifolds
        self.manifolds = nn.ModuleList()
        current_dim = input_dim
        
        for i, config in enumerate(manifold_configs):
            # If recursive, input includes all previous manifolds
            if use_recursive and i > 0:
                if aggregation == "concat":
                    # Concatenate all previous outputs
                    prev_dims = sum(m.manifold_dim for m in self.manifolds)
                    current_dim = input_dim + prev_dims
                elif aggregation == "sum":
                    # Sum requires same dimensions
                    current_dim = input_dim
                # For attention, we'll handle in forward pass
            
            manifold = SAMBlock(
                grid_shape=config['grid_shape'],
                input_dim=current_dim,
                random_effects_config=random_effects_config,
                use_spde=config.get('use_spde', True),
                use_sem=config.get('use_sem', False),
                activation=config.get('activation', 'relu'),
                spde_alpha=config.get('spde_alpha', 2),
                dropout=config.get('dropout', 0.0),
            )
            
            self.manifolds.append(manifold)
        
        # Output SEM header (causal coordination)
        # For now, simple implementation - can be extended with lavaan parser
        self.output_sem = None
        if output_sem_spec:
            # Parse lavaan-style spec and build SEM
            # For now, placeholder
            pass
        
        # Collect final dimension
        if aggregation == "concat":
            self.final_dim = sum(m.manifold_dim for m in self.manifolds)
        else:
            self.final_dim = self.manifolds[-1].manifold_dim
    
    def forward(
        self,
        x: torch.Tensor,
        random_indices: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through all manifolds.
        
        Args:
            x: Fixed input features [batch, input_dim]
            random_indices: Dict of random effect indices
        
        Returns:
            h_final: Final latent representation [batch, final_dim]
            h_all: List of all manifold outputs (for visualization/analysis)
        """
        batch_size = x.shape[0]
        h_all = []
        
        # Initial input
        h_current = x
        
        for i, manifold in enumerate(self.manifolds):
            if self.use_recursive and i > 0:
                if self.aggregation == "concat":
                    # Concatenate input + all previous manifolds
                    h_current = torch.cat([x] + h_all, dim=1)
                elif self.aggregation == "sum":
                    # Sum all previous (requires same dims)
                    h_current = x + sum(h_all)
            
            # Pass through manifold
            h = manifold(h_current, random_indices)
            h_all.append(h)
            
            # Update current for next layer
            h_current = h
        
        # Aggregate final representation
        if self.aggregation == "concat":
            h_final = torch.cat(h_all, dim=1)
        elif self.aggregation == "sum":
            h_final = sum(h_all)
        else:
            h_final = h_all[-1]
        
        return h_final, h_all
    
    def get_manifold_2d(self, layer_idx: int, x: torch.Tensor, 
                        random_indices: Optional[Dict] = None) -> torch.Tensor:
        """
        Get 2D manifold representation for visualization.
        
        Args:
            layer_idx: Which manifold to visualize
            x: Input features
            random_indices: Random effect indices
        
        Returns:
            h_2d: 2D manifold [batch, U, V]
        """
        # Forward through layers up to layer_idx
        h_current = x
        for i in range(layer_idx + 1):
            if self.use_recursive and i > 0:
                # Simplified: just use previous output
                pass
            
            if i == layer_idx:
                h, h_2d = self.manifolds[i](h_current, random_indices, return_manifold=True)
                return h_2d
            else:
                h_current = self.manifolds[i](h_current, random_indices)
        
        raise ValueError(f"Invalid layer_idx: {layer_idx}")
    
    def penalty(self):
        """Total regularization penalty from all manifolds."""
        return sum(m.penalty() for m in self.manifolds)


__all__ = ['SAMBlock', 'UniversalSAMNN']
