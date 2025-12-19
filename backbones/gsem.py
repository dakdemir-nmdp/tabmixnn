"""
Complete GM-SEM Network Architecture

Implements the full pipeline: Input → [Hidden × L] → Output

Enables flexible layer composition with different structures at each level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any
from .base import TabMixNNBackbone
from ..layers.block import MixedInputLayer, GSEMHiddenLayer
from ..families import Gaussian, Binomial, Multinomial, MultivariateGaussian, ContinuousSurvival
from ..formula import OutcomeSpec


class GSEMBackbone(TabMixNNBackbone):
    """
    Backbone consisting of a stack of GSEM hidden layers.
    
    Architecture:
        [Hidden Layer × L]
    
    Each layer has its own structural (static/dynamic) and random effects configuration.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        use_static_structure: Union[bool, List[bool]] = False,
        use_dynamic_structure: Union[bool, List[bool]] = False,
        n_attention_heads: Union[int, List[int]] = 4,
        activations: Union[str, List[str]] = 'gelu',
        use_residual: Union[bool, List[bool]] = True,
        use_layer_norm: Union[bool, List[bool]] = True,
        dropout: Union[float, List[float]] = 0.0,
    ):
        super().__init__(input_dim)
        self.hidden_dims = hidden_dims
        n_layers = len(hidden_dims)
        
        # Broadcast scalar configs to all layers
        def broadcast(value, n):
            if isinstance(value, (list, tuple)):
                assert len(value) == n, f"Expected {n} values, got {len(value)}"
                return list(value)
            return [value] * n
        
        use_static = broadcast(use_static_structure, n_layers)
        use_dynamic = broadcast(use_dynamic_structure, n_layers)
        n_heads = broadcast(n_attention_heads, n_layers)
        acts = broadcast(activations, n_layers)
        use_res = broadcast(use_residual, n_layers)
        use_ln = broadcast(use_layer_norm, n_layers)
        drops = broadcast(dropout, n_layers)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        current_dim = input_dim
        for l in range(n_layers):
            out_dim = hidden_dims[l]
            layer = GSEMHiddenLayer(
                input_dim=current_dim,
                output_dim=out_dim,
                use_static_structure=use_static[l],
                use_dynamic_structure=use_dynamic[l],
                n_attention_heads=n_heads[l],
                activation=acts[l],
                use_residual=use_res[l],
                use_layer_norm=use_ln[l],
                dropout=drops[l],
            )
            self.hidden_layers.append(layer)
            current_dim = out_dim
            
        self.output_dim = current_dim

    def forward(self, x: torch.Tensor, context: Optional[Dict] = None, return_attention: bool = False):
        """
        Forward pass through hidden layers.
        
        Args:
            x: Input tensor [batch, input_dim]
            context: Optional dict containing 'group_indices' for random effects
            return_attention: Whether to return attention weights from structural layers
            
        Returns:
            output: Final hidden state
            aux_loss: Regularization penalties (always 0.0, use penalty() instead)
            attention: (Optional) List of attention weights per layer
        """
        H = x
        group_indices = context.get('group_indices') if context else None
        # Extract specific group if dict
        if isinstance(group_indices, dict) and 'main' in group_indices:
            group_indices = group_indices['main']
        # If list/tuple, take first as 'main'
        elif isinstance(group_indices, (list, tuple)) and len(group_indices) > 0:
            group_indices = group_indices[0]

        attns = []
        for layer in self.hidden_layers:
            if return_attention:
                H, attn = layer(H, group_indices, return_attention=True)
                attns.append(attn)
            else:
                H = layer(H, group_indices)
            
        if return_attention:
            return H, 0.0, attns
        return H, 0.0

    def penalty(self):
        loss = 0.0
        for layer in self.hidden_layers:
            penalties = layer.penalties()
            loss += sum(penalties.values())
        return loss


class GSEMModel(nn.Module):
    """
    Complete GM-SEM network with configurable layers (Legacy / Full-Pipeline version).
    
    Architecture:
        Input Layer → GSEMBackbone → Output Layer
    """
    
    def __init__(
        self,
        # Input specification
        continuous_dim: int = 0,
        categorical_specs: Optional[Dict[str, int]] = None,
        embedding_dim: int = 16,
        # Architecture
        hidden_dims: List[int] = [64, 32],
        use_static_structure: Union[bool, List[bool]] = False,
        use_dynamic_structure: Union[bool, List[bool]] = False,
        n_attention_heads: Union[int, List[int]] = 4,
        activations: Union[str, List[str]] = 'gelu',
        # Random effects
        random_effect_groups: Optional[Dict[str, int]] = None,
        # Residual and normalization
        use_residual: Union[bool, List[bool]] = True,
        use_layer_norm: Union[bool, List[bool]] = True,
        dropout: Union[float, List[float]] = 0.0,
        # Output specification
        output_type: str = 'regression',
        output_dim: int = 1,
        output_config: Optional[Dict[str, Any]] = None,
        spatial_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.output_type = output_type
        
        # 1. Input layer
        input_layer_out = hidden_dims[0] if hidden_dims else embedding_dim
        self.input_layer = MixedInputLayer(
            continuous_dim=continuous_dim,
            categorical_specs=categorical_specs,
            embedding_dim=embedding_dim,
            output_dim=input_layer_out,
            spatial_encoder=spatial_encoder,
            random_effect_groups=random_effect_groups,
            embed_as_random_effect=True,
            dropout=dropout if isinstance(dropout, float) else dropout[0] if hidden_dims else 0.0,
        )
        
        # 2. Backbone
        self.backbone = GSEMBackbone(
            input_dim=input_layer_out,
            hidden_dims=hidden_dims,
            use_static_structure=use_static_structure,
            use_dynamic_structure=use_dynamic_structure,
            n_attention_heads=n_attention_heads,
            activations=activations,
            use_residual=use_residual,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
        
        # 3. Output layer
        final_dim = self.backbone.output_dim
        output_config = output_config or {}
        
        # Determine family based on output_type
        if output_type == 'regression':
            self.family = Gaussian()
            self.head = nn.Linear(final_dim, output_dim)
        elif output_type == 'classification':
            task = output_config.get('task_type', 'multiclass')
            if task == 'binary':
                self.family = Binomial()
                self.head = nn.Linear(final_dim, 1)
            else:
                self.family = Multinomial(n_classes=output_dim)
                self.head = nn.Linear(final_dim, output_dim)
        elif output_type == 'multi_task':
            self.family = MultivariateGaussian(
                n_tasks=output_dim,
                covariance_type=output_config.get('covariance_type', 'low_rank'),
                covariance_rank=output_config.get('covariance_rank', 5)
            )
            self.head = nn.Linear(final_dim, output_dim)
        elif output_type == 'survival':
            self.family = ContinuousSurvival(
                baseline_hazard=output_config.get('baseline_hazard', 'weibull')
            )
            self.head = nn.Linear(final_dim, 1)
        else:
            raise ValueError(f"Unknown output_type: {output_type}")
            
        # Register for legacy access
        self.output_layer = self.head 
    
    def forward(
        self,
        X_cont: Optional[torch.Tensor] = None,
        X_cat: Optional[Dict[str, torch.Tensor]] = None,
        X_pos: Optional[torch.Tensor] = None,
        group_indices: Optional[Dict[str, torch.Tensor]] = None,
        return_hidden: bool = False,
    ):
        """Forward pass through entire pipeline."""
        H = self.input_layer(X_cont, X_cat, X_pos, group_indices)
        
        hidden_states = [H] if return_hidden else None
        
        # Extract main group index for hidden layers
        layer_group_idx = group_indices.get('main') if group_indices and 'main' in group_indices else None
        context = {'group_indices': layer_group_idx}
        
        # Backbone pass
        H, _ = self.backbone(H, context)
        
        if return_hidden:
            # Reconstruct hidden states if needed (only final for now in this wrapper)
            hidden_states.append(H)
        
        # Output
        out = self.head(H)
        
        if return_hidden:
            return {'output': out, 'hidden_states': hidden_states}
        return out

    def forward_from_embeddings(self, H, group_indices=None, return_hidden=False):
        """Compatibility method for TabMixNN."""
        layer_group_idx = group_indices.get('main') if group_indices and 'main' in group_indices else None
        H_out, _ = self.backbone(H, {'group_indices': layer_group_idx})
        if return_hidden:
            return {'hidden_states': [H, H_out], 'output': H_out}
        return H_out
    
    def compute_loss(
        self,
        X_cont: Optional[torch.Tensor] = None,
        X_cat: Optional[Dict[str, torch.Tensor]] = None,
        Y: Optional[torch.Tensor] = None,
        group_indices: Optional[Dict[str, torch.Tensor]] = None,
        X_pos: Optional[torch.Tensor] = None,
        lambda_structural: float = 0.01,
        lambda_kl: float = 0.001,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss with all regularization terms.
        
        Returns dictionary of losses:
        - task_loss: Main prediction loss
        - structural penalties: DAG, sparsity, contraction
        - kl_divergence: For random effects
        - total: Sum of all weighted losses
        """
        # Forward pass
        result = self.forward(
            X_cont, X_cat, X_pos, group_indices, 
            return_hidden=True
        )
        
        if isinstance(result, dict):
            output = result['output']
            hidden_states = result.get('hidden_states', [])
        else:
            output = result
            hidden_states = []
        
        losses = {}
        
        # Main task loss
        if Y is not None:
            if self.output_type == 'regression':
                losses['task_loss'] = F.mse_loss(output, Y)
            
            elif self.output_type == 'classification':
                losses['task_loss'] = self.output_layer.loss(hidden_states[-1] if hidden_states else output, Y)
            
            elif self.output_type == 'multi_task':
                losses['task_loss'] = -self.output_layer.log_likelihood(hidden_states[-1] if hidden_states else output, Y)
            
            elif self.output_type == 'survival':
                # User needs to provide custom survival loss
                losses['task_loss'] = torch.tensor(0.0, device=output.device)
        
        # Input layer regularization
        input_reg = self.input_layer.regularization_loss()
        for name, val in input_reg.items():
            losses[f'input_{name}'] = lambda_kl * val
        
        # Hidden layer penalties
        for l, layer in enumerate(self.hidden_layers):
            layer_penalties = layer.penalties()
            for name, val in layer_penalties.items():
                if 'kl' in name:
                    losses[f'layer{l}_{name}'] = lambda_kl * val
                else:
                    losses[f'layer{l}_{name}'] = lambda_structural * val
        
        # Output layer penalties
        if hasattr(self.output_layer, 'penalties'):
            output_penalties = self.output_layer.penalties()
            for name, val in output_penalties.items():
                losses[f'output_{name}'] = lambda_kl * val
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses
    
    def get_layer_summary(self) -> str:
        """Return a human-readable summary of the network architecture."""
        lines = ["GM-SEM Network Architecture", "=" * 50]
        
        # Input layer
        lines.append(f"\nInput Layer:")
        lines.append(f"  Output dim: {self.input_layer.output_dim}")
        if self.input_layer.embeddings:
            lines.append(f"  Categorical features: {list(self.input_layer.categorical_specs.keys())}")
        
        # Hidden layers
        for i, layer in enumerate(self.hidden_layers):
            lines.append(f"\nHidden Layer {i+1}:")
            lines.append(f"  Dimensions: {layer.input_dim} → {layer.output_dim}")
            
            structure_info = []
            if layer.structure:
                if layer.structure.use_static:
                    structure_info.append("Static (SEM)")
                if layer.structure.use_dynamic:
                    structure_info.append("Dynamic (Attention)")
            
            if structure_info:
                lines.append(f"  Structure: {', '.join(structure_info)}")
            else:
                lines.append(f"  Structure: None (Standard MLP)")
            
            lines.append(f"  Activation: {layer.activation.__class__.__name__}")
            lines.append(f"  Residual: {layer.use_residual}")
        
        # Output layer
        lines.append(f"\nOutput Layer:")
        lines.append(f"  Type: {self.output_type}")
        if self.output_type == 'multi_task' and hasattr(self.output_layer, 'covariance_type'):
            lines.append(f"  Covariance: {self.output_layer.covariance_type}")
        
        return "\n".join(lines)
