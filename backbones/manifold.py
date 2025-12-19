import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Union

from .base import TabMixNNBackbone
from ..layers import create_spde_kernel
from ..layers import PositionalSEMLayer, resolve_activation

class ManifoldBlock(nn.Module):
    """
    A single SAM manifold block for Manifold SAM-NN.
    """
    def __init__(self,
                 grid_shape,
                 input_dim,
                 random_config: Optional[List[Dict]] = None,
                 use_spde=True,
                 use_sem=False,
                 activation="relu",
                 spde_alpha=2,
                 spde_kappa_init=0.3,
                 use_kronecker=True,
                 dropout=0.0):
        super().__init__()
        
        if isinstance(grid_shape, int):
            grid_shape = (grid_shape,)
        self.grid_shape = tuple(grid_shape)
        self.manifold_dim = int(np.prod(self.grid_shape))
        self.input_dim = input_dim
        self.use_spde = use_spde
        self.use_sem = use_sem
        
        # Fixed effects map
        self.W_fixed = nn.Linear(input_dim, self.manifold_dim)
        
        # Random effects (Layer-specific embeddings for now)
        self.random_config = random_config or []
        self.random_projections = nn.ModuleDict()
        
        for term in self.random_config:
            term_id = term['term_id']
            n_levels = term['n_levels']
            # We use simple Embedding here for Universal capacity
            self.random_projections[term_id] = nn.Embedding(n_levels, self.manifold_dim)
            
        # SPDE Kernel
        if use_spde:
            self.spde_kernel = create_spde_kernel(
                grid_shape=grid_shape,
                use_kronecker=use_kronecker,
                alpha=spde_alpha,
                kappa_init=spde_kappa_init
            )
        else:
            self.spde_kernel = None
            
        # SEM
        if use_sem:
            self.sem = PositionalSEMLayer(
                dim=self.manifold_dim,
                grid_shape=grid_shape,
                lengthscale_init=3.0
            )
        else:
            self.sem = None
            
        self.activation = resolve_activation(activation)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        self.layer_norm = nn.LayerNorm(self.manifold_dim)
        
    def forward(self, x, context):
        # x: [batch, input_dim]
        # context: dict with 'Z_indices' (list matched to config) or dict {term_id: indices}
        
        # 1. Fixed
        mu = self.W_fixed(x)
        
        # 2. Random
        # Context extraction
        # If context provides a dict of tensors by term_id:
        random_indices_map = context.get('random_indices_map')
        
        if random_indices_map:
            for term_id, indices in random_indices_map.items():
                if term_id in self.random_projections:
                    mu = mu + self.random_projections[term_id](indices)
                    
        # 3. SPDE
        if self.spde_kernel is not None and self.training:
            epsilon = self.spde_kernel.sample(x.shape[0], x.device)
            mu = mu + epsilon
            
        # 4. Activation
        if self.activation:
            h = self.activation(mu)
        else:
            h = mu
            
        # 5. Norm + Dropout
        h = self.layer_norm(h)
        if self.dropout_layer:
            h = self.dropout_layer(h)
            
        # 6. SEM
        if self.sem:
            h = self.sem(h)
            
        return h
        
    def penalty(self):
        p = 0.0
        if self.spde_kernel:
            p += self.spde_kernel.penalty()
        if self.sem:
            p += self.sem.penalty()
        return p


class ManifoldBackbone(TabMixNNBackbone):
    """
    Manifold SAM-NN Backbone.
    Stack of Manifolds with Recursive Connections.
    """
    def __init__(self,
                 input_dim,
                 manifold_configs: List[Dict],
                 random_config: List[Dict],
                 use_recursive=True,
                 aggregation="concat"):
        super().__init__(input_dim)
        
        self.manifolds = nn.ModuleList()
        self.use_recursive = use_recursive
        self.aggregation = aggregation
        self.random_config = random_config
        
        current_dim = input_dim
        
        for i, config in enumerate(manifold_configs):
            # Calculate input dim based on recursion
            if use_recursive and i > 0:
                if aggregation == "concat":
                    prev_dims = sum(m.manifold_dim for m in self.manifolds)
                    block_input = input_dim + prev_dims
                elif aggregation == "sum":
                    block_input = input_dim
                else: 
                    block_input = self.manifolds[-1].manifold_dim
            else:
                block_input = current_dim if i==0 else self.manifolds[-1].manifold_dim
            
            # Create Block
            block = ManifoldBlock(
                grid_shape=config['grid_shape'],
                input_dim=block_input,
                random_config=random_config,
                use_spde=config.get('use_spde', True),
                use_sem=config.get('use_sem', False),
                activation=config.get('activation', 'relu'),
                spde_alpha=config.get('spde_alpha', 2),
                dropout=config.get('dropout', 0.0)
            )
            self.manifolds.append(block)
            current_dim = block.manifold_dim # update for next if sequential
            
        # Final dim
        if aggregation == "concat":
            self.output_dim = sum(m.manifold_dim for m in self.manifolds)
        elif aggregation == "sum":
            self.output_dim = self.manifolds[-1].manifold_dim # Assume matched
        else:
            self.output_dim = self.manifolds[-1].manifold_dim
            
    def forward(self, x, context=None):
        # x: [batch, input_dim]
        # context: dict with 'Z_indices' (list)
        
        # Prepare indices map for blocks
        # Context likely comes from TabMixNN which has Z_indices list.
        # We need to map this list to term_ids based on self.random_config order.
        if context and 'Z_indices' in context:
            z_indices = context['Z_indices'] # list
            indices_map = {}
            # Assuming random_config order matches Z_indices order from Encoder
            # The encoder builds config in same order as it outputs Z_indices
            for i, term in enumerate(self.random_config):
                if i < len(z_indices):
                    indices_map[term['term_id']] = z_indices[i]
            context['random_indices_map'] = indices_map
            
        aux_loss = 0.0
        h_all = []
        
        for i, manifold in enumerate(self.manifolds):
            # Form recursive input
            if self.use_recursive and i > 0:
                if self.aggregation == "concat":
                    inp = torch.cat([x] + h_all, dim=1)
                elif self.aggregation == "sum":
                    inp = x + sum(h_all) # Dangerous if dims mismatch
                else:
                    inp = h_all[-1]
            elif i == 0:
                inp = x
            else: # Sequential non-recursive
                inp = h_all[-1]
                
            h = manifold(inp, context)
            h_all.append(h)
            # aux_loss += manifold.penalty() # Used .penalty() now
            
        # Aggregation
        if self.aggregation == "concat":
            out = torch.cat(h_all, dim=1)
        elif self.aggregation == "sum":
            out = sum(h_all)
        else:
            out = h_all[-1]
            
        return out, 0.0
        
    def penalty(self):
        loss = 0.0
        for m in self.manifolds:
            loss += m.penalty()
        return loss

    def get_manifold_2d(self, layer_idx, x, context=None):
        """Helper to get reshaped 2D manifold for visualization."""
        if layer_idx >= len(self.manifolds):
            raise ValueError(f"Invalid layer_idx {layer_idx}")
            
        m = self.manifolds[layer_idx]
        grid = m.grid_shape
        if len(grid) != 2:
            raise ValueError("Manifold is not 2D")
            
        # We need to run forward pass up to this layer
        # This is inefficient but fine for visualization
        self.eval() 
        with torch.no_grad():
            # ... duplicated forward logic ...
            # To avoid duplication, we can just grab output?
            # But we need input to this layer.
            # Let's just create a dry run
            
            # Prepare indices map
            if context and 'Z_indices' in context:
                z_indices = context['Z_indices'] 
                indices_map = {}
                for i, term in enumerate(self.random_config):
                    if i < len(z_indices):
                        indices_map[term['term_id']] = z_indices[i]
                context['random_indices_map'] = indices_map
            
            h_all = []
            final_h = None
            
            for i, manifold in enumerate(self.manifolds):
                if self.use_recursive and i > 0:
                    if self.aggregation == "concat":
                        inp = torch.cat([x] + h_all, dim=1)
                    elif self.aggregation == "sum":
                        inp = x + sum(h_all) 
                    else:
                        inp = h_all[-1]
                elif i == 0:
                    inp = x
                else:
                    inp = h_all[-1]
                    
                h = manifold(inp, context)
                h_all.append(h)
                
                if i == layer_idx:
                    final_h = h
                    break
                    
            # Reshape
            return final_h.reshape(-1, grid[0], grid[1])
            
# Compatibility alias
class ManifoldSAMNN(ManifoldBackbone):
    """
    Backwards compatibility wrapper.
    In the new architecture, ManifoldSAMNN is just the Backbone.
    The outer Model (TabMixNN) handles the compilation and encoder.
    If someone tries to use this standalone, it behaves like the backbone.
    """
    def __init__(self,
                  input_dim,
                  manifold_configs: List[Dict],
                  random_effects_config: List[Dict], # Renamed from random_config
                  output_sem_spec=None, # Ignored/Deprecated
                  use_recursive=True,
                  aggregation="concat"):
        
        # random_effects_config to random_config
        super().__init__(
            input_dim=input_dim,
            manifold_configs=manifold_configs,
            random_config=random_effects_config,
            use_recursive=use_recursive,
            aggregation=aggregation
        )
        # self.final_dim is needed for tests using this class directly
        self.final_dim = self.output_dim

    def forward(self, x, random_indices=None):
        # The test passes 'random_indices' dict {term: indices} directly
        # The backbone expects context with 'Z_indices' list OR 'random_indices_map'
        context = {}
        if random_indices:
             context['random_indices_map'] = random_indices
        
        out, _ = super().forward(x, context)
        
        # Test expects (final, all_manifolds)
        # But super().forward returns (out, aux).
        # We need to hack it to return intermediate outputs if we want full compat?
        # The test checks: h_final, h_all = model(x, random_indices)
        
        # This is tough. The backbone abstraction hides intermediates.
        # But for the test script we need them.
        # Let's override forward here for the legacy class.
        
        # Re-implement simple forward returning h_all
        # Or attach h_all to self temporarily?
        
        # Better: Reuse code logic but expose internals
        
        h_all = []
        
        # Prepare context
        if random_indices:
             context['random_indices_map'] = random_indices
             
        for i, manifold in enumerate(self.manifolds):
            if self.use_recursive and i > 0:
                if self.aggregation == "concat":
                    inp = torch.cat([x] + h_all, dim=1)
                elif self.aggregation == "sum":
                    inp = x + sum(h_all)
                else:
                    inp = h_all[-1]
            elif i == 0:
                inp = x
            else:
                inp = h_all[-1]
                
            h = manifold(inp, context)
            h_all.append(h)
            
        if self.aggregation == "concat":
            out = torch.cat(h_all, dim=1)
        elif self.aggregation == "sum":
            out = sum(h_all)
        else:
            out = h_all[-1]
            
        return out, h_all

    def get_manifold_2d(self, layer_idx, x, random_indices=None):
        context = {}
        if random_indices:
             context['random_indices_map'] = random_indices
        return super().get_manifold_2d(layer_idx, x, context)
