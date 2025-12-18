"""
Universal SAM-NN Model: TabMixNN with SAMBlock hidden layers.

This extends TabMixNNModule to support Universal SAM-NN architecture
where hidden layers are replaced with SAMBlock manifolds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .universal import SAMBlock, UniversalSAMNN
from .model import MixedEffectsRepresentation


class UniversalSAMNNModule(nn.Module):
    """
    TabMixNN with Universal SAM-NN hidden layers.
    
    Architecture:
    1. Mixed Effects Entry Layer (LME4-style) - same as TabMixNN
    2. SAMBlock manifolds instead of MLP hidden layers
    3. Outcome-Specific Output Heads - same as TabMixNN
    """
    
    def __init__(self,
                 fixed_input_dim,
                 compiled_design,
                 embedding_dim=64,
                 hidden_dims=[64],
                 manifold_configs=None,
                 cov_struct_map=None,
                 mixed_activation="relu",
                 kl_normalize=True,
                 scale_random_by_terms=False,
                 unknown_strategy="learned",
                 use_recursive=True,
                 aggregation="concat",
                 hidden_dropout=0.0):
        """
        Args:
            fixed_input_dim: Dimension of fixed effects
            compiled_design: Compiled design from formula
            embedding_dim: Dimension of mixed effects embedding
            hidden_dims: List of hidden dimensions (used if manifold_configs is None)
            manifold_configs: List of SAMBlock configs (overrides hidden_dims)
            cov_struct_map: Covariance structure map for random effects
            mixed_activation: Activation after mixed effects layer
            kl_normalize: Whether to normalize KL divergence
            scale_random_by_terms: Whether to scale random effects by number of terms
            unknown_strategy: Strategy for unknown levels ("learned" or "mean")
            use_recursive: Whether SAMBlocks see all previous layers
            aggregation: How to aggregate previous layers ("concat", "sum", "last")
            hidden_dropout: Dropout rate for SAMBlocks
        """
        super().__init__()
        
        self.outcomes = compiled_design.outcomes
        self.mixed_activation_name = mixed_activation
        self.mixed_activation = self._resolve_activation(mixed_activation)
        self.kl_normalize = kl_normalize
        self.scale_random_by_terms = scale_random_by_terms
        self.use_recursive = use_recursive
        self.aggregation = aggregation
        
        # 1. Mixed Effects Block (same as TabMixNN)
        random_config = []
        for term in compiled_design.random_terms:
            random_config.append({
                'term_id': term.term_id,
                'n_levels': max(term.level_map.values()) + 1 if term.level_map else 1,
                'n_slopes': len(term.slope_cols),
                'group': term.group_col,
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
        
        self.random_config = random_config
        
        # 2. Build manifold configs from hidden_dims if not provided
        if manifold_configs is None:
            manifold_configs = self._hidden_dims_to_manifolds(hidden_dims, hidden_dropout)
        
        self.manifold_configs = manifold_configs
        
        # 3. Build Universal SAM-NN
        self.universal_samnn = UniversalSAMNN(
            input_dim=embedding_dim,
            manifold_configs=manifold_configs,
            random_effects_config=random_config,
            use_recursive=use_recursive,
            aggregation=aggregation
        )
        
        self.last_dim = self.universal_samnn.final_dim
        
        # 4. Output Heads (same as TabMixNN)
        self.heads = nn.ModuleDict()
        self.families = nn.ModuleDict()
        for outcome in self.outcomes:
            family = outcome.family
            out_dim = family.n_params
            self.heads[outcome.name] = nn.Linear(self.last_dim, out_dim)
            self.families[outcome.name] = family
    
    @staticmethod
    def _hidden_dims_to_manifolds(hidden_dims, dropout=0.0):
        """
        Convert hidden_dims to manifold_configs.
        
        Maps each hidden dimension to a 2D grid shape:
        - 64 -> (8, 8)
        - 32 -> (6, 6) ≈ 36
        - 16 -> (4, 4)
        - etc.
        """
        manifold_configs = []
        
        for i, dim in enumerate(hidden_dims):
            # Find closest square grid
            grid_size = int(np.sqrt(dim))
            
            # Adjust to get close to target dim
            if grid_size ** 2 < dim:
                # Try rectangular grids
                for u in range(grid_size, grid_size + 3):
                    for v in range(grid_size, grid_size + 3):
                        if abs(u * v - dim) <= 2:  # Within 2 of target
                            grid_shape = (u, v)
                            break
                    else:
                        continue
                    break
                else:
                    # Fallback to square
                    grid_shape = (grid_size, grid_size)
            else:
                grid_shape = (grid_size, grid_size)
            
            # Use SPDE for all layers, SEM only for last layer
            config = {
                'grid_shape': grid_shape,
                'use_spde': True,
                'use_sem': (i == len(hidden_dims) - 1),  # SEM only on last layer
                'spde_alpha': 2,
                'activation': 'gelu',
                'dropout': dropout
            }
            
            manifold_configs.append(config)
        
        return manifold_configs
    
    def forward(self, X, Z_indices, Z_slopes):
        """
        Forward pass through Universal SAM-NN model.
        
        Args:
            X: Fixed effects [batch, fixed_dim]
            Z_indices: List of random effect indices
            Z_slopes: List of random effect slopes
        
        Returns:
            outputs: Dict of outcome predictions
            kl: KL divergence from mixed effects
        """
        # 1. Mixed Effects Entry Layer
        x, kl = self.mixed_block(X, Z_indices, Z_slopes)
        if self.mixed_activation is not None:
            x = self.mixed_activation(x)
        
        # 2. Convert Z_indices to dict format for SAMBlocks
        random_indices = {}
        for i, term in enumerate(self.random_config):
            term_id = term['term_id']
            random_indices[term_id] = Z_indices[i]
        
        # 3. Universal SAM-NN layers
        h_final, h_all = self.universal_samnn(x, random_indices)
        
        # 4. Output Heads
        outputs = {}
        for name, head in self.heads.items():
            outputs[name] = head(h_final)
        
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
    
    def get_variance_summary(self):
        """Get variance summary from mixed effects block."""
        return self.mixed_block.get_variance_summary()
    
    def get_sparsity_loss(self, lambda_sparse=None):
        """
        Compute sparsity penalty from Universal SAM-NN.
        
        This includes:
        - SPDE penalties from all SAMBlocks
        - SEM penalties from SAMBlocks with SEM enabled
        """
        if lambda_sparse is None:
            lambda_sparse = 1.0
        elif isinstance(lambda_sparse, dict):
            # Use 'hidden' key for Universal SAM-NN
            lambda_sparse = lambda_sparse.get('hidden', 1.0)
        
        return float(lambda_sparse) * self.universal_samnn.penalty()
    
    def get_manifold_2d(self, layer_idx, X, Z_indices, Z_slopes):
        """
        Get 2D manifold representation for visualization.
        
        Args:
            layer_idx: Which SAMBlock to visualize
            X: Fixed effects
            Z_indices: Random effect indices
            Z_slopes: Random effect slopes
        
        Returns:
            h_2d: 2D manifold [batch, U, V]
        """
        # Get mixed effects representation
        x, _ = self.mixed_block(X, Z_indices, Z_slopes)
        if self.mixed_activation is not None:
            x = self.mixed_activation(x)
        
        # Convert to dict format
        random_indices = {}
        for i, term in enumerate(self.random_config):
            term_id = term['term_id']
            random_indices[term_id] = Z_indices[i]
        
        # Get 2D manifold from Universal SAM-NN
        return self.universal_samnn.get_manifold_2d(layer_idx, x, random_indices)


__all__ = ['UniversalSAMNNModule']
