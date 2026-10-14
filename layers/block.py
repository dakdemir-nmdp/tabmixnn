"""
Building blocks for GSEM architectures.
Contains Input and Hidden layers.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from .structural import RandomEffectsModule, HybridStructuralModel
from .activations import get_activation

# ----------------------------------------------------------------------------
# Input Layer
# ----------------------------------------------------------------------------

class MixedInputLayer(nn.Module):
    """
    Input layer with mixed-type encoding and random effects.
    
    Implements: H⁽⁰⁾ = X_cont W_cont + Σ E_j[X_cat] + S(X_pos) + Z_0 u_0
    
    Unifies:
    - Continuous feature projection
    - Categorical embeddings (= random intercepts)
    - Spatial/positional encoding
    - Explicit random effects with prior
    """
    
    def __init__(
        self,
        continuous_dim: int,
        categorical_specs: Optional[Dict[str, int]] = None,  # {name: n_categories}
        embedding_dim: int = 16,
        output_dim: int = 64,
        spatial_encoder: Optional[nn.Module] = None,
        random_effect_groups: Optional[Dict[str, int]] = None,
        embed_as_random_effect: bool = True,  # Treat embeddings as random effects
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.embed_as_random_effect = embed_as_random_effect
        self.continuous_dim = continuous_dim
        
        # Continuous projection
        if continuous_dim > 0:
            self.W_cont = nn.Linear(continuous_dim, output_dim, bias=False)
        else:
            self.W_cont = None
        
        # Categorical embeddings
        self.categorical_specs = categorical_specs or {}
        if self.categorical_specs:
            self.embeddings = nn.ModuleDict({
                name: nn.Embedding(n_cat, embedding_dim)
                for name, n_cat in categorical_specs.items()
            })
            
            # Project concatenated embeddings to output dim
            total_embed_dim = len(categorical_specs) * embedding_dim
            self.embed_proj = nn.Linear(total_embed_dim, output_dim, bias=False)
            
            # Random effect prior variances (for regularization)
            if embed_as_random_effect:
                self.log_prior_var = nn.ParameterDict({
                    name: nn.Parameter(torch.zeros(embedding_dim))
                    for name in categorical_specs.keys()
                })
        else:
            self.embeddings = None
            self.embed_proj = None
        
        # Optional spatial encoder
        self.spatial_encoder = spatial_encoder
        
        # Explicit random effects (beyond embeddings)
        if random_effect_groups:
            self.explicit_random_effects = RandomEffectsModule(
                group_specs={name: (n, output_dim) for name, n in random_effect_groups.items()},
                covariance_type='diagonal'
            )
        else:
            self.explicit_random_effects = None
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(
        self, 
        X_cont: Optional[torch.Tensor] = None,
        X_cat: Optional[Dict[str, torch.Tensor]] = None,
        X_pos: Optional[torch.Tensor] = None,
        group_indices: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass through input layer.
        
        Args:
            X_cont: Continuous features [batch, continuous_dim]
            X_cat: Categorical features {name: [batch] indices}
            X_pos: Positional/spatial coordinates [batch, spatial_dim]
            group_indices: Group indices for random effects {name: [batch]}
        
        Returns:
            H: Initial hidden representation [batch, output_dim]
        """
        device = self.bias.device
        batch_size = None
        
        # Determine batch size from inputs
        if X_cont is not None:
            batch_size = X_cont.shape[0]
        elif X_cat is not None and len(X_cat) > 0:
            batch_size = list(X_cat.values())[0].shape[0]
        elif group_indices is not None and len(group_indices) > 0:
            batch_size = list(group_indices.values())[0].shape[0]
        else:
            raise ValueError("At least one input must be provided")
        
        # Initialize output
        h = torch.zeros(batch_size, self.output_dim, device=device)
        
        # Continuous features
        if X_cont is not None and self.W_cont is not None:
            h = h + self.W_cont(X_cont)
        
        # Categorical embeddings
        if X_cat is not None and self.embeddings is not None:
            embeds = []
            for name, indices in X_cat.items():
                if name in self.embeddings:
                    embeds.append(self.embeddings[name](indices.to(device)))
            
            if embeds:
                embed_concat = torch.cat(embeds, dim=-1)
                h = h + self.embed_proj(embed_concat)
        
        # Spatial/positional encoding
        if self.spatial_encoder is not None and X_pos is not None:
            h = h + self.spatial_encoder(X_pos)
        
        # Explicit random effects
        if self.explicit_random_effects is not None and group_indices is not None:
            h = h + self.explicit_random_effects(group_indices)
        
        # Bias
        h = h + self.bias
        
        # Dropout
        if self.dropout is not None:
            h = self.dropout(h)
        
        return h
    
    def embedding_kl_divergence(self) -> torch.Tensor:
        """KL divergence treating embeddings as random effects."""
        if not self.embed_as_random_effect or self.embeddings is None:
            return torch.tensor(0.0)
        
        kl = 0.0
        for name, emb in self.embeddings.items():
            prior_var = torch.exp(self.log_prior_var[name])
            kl += 0.5 * ((emb.weight ** 2) / prior_var.unsqueeze(0)).sum()
        
        return kl
    
    def regularization_loss(self) -> Dict[str, torch.Tensor]:
        """Compute all regularization terms."""
        losses = {}
        
        # Embedding KL
        losses['embedding_kl'] = self.embedding_kl_divergence()
        
        # Explicit random effects KL
        if self.explicit_random_effects is not None:
            losses['random_effects_kl'] = self.explicit_random_effects.kl_divergence()
        
        return losses

# ----------------------------------------------------------------------------
# Hidden Layer
# ----------------------------------------------------------------------------

class GSEMHiddenLayer(nn.Module):
    """
    Hidden layer implementing the GM-SEM transform.
    
    H_out = σ((I - B_s - B_d(H))^{-1} [W @ H_in + Zu + b])
    
    Each layer can have:
    - Static structure (B_s) for fixed dependencies
    - Dynamic structure (B_d) for attention-based dependencies
    - Random effects (Zu) for hierarchical variation
    - Residual connections
    - Layer normalization
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        # Structure configuration
        structure_type: str = 'composite', # 'composite' or 'iterative'
        use_static_structure: bool = False,
        use_dynamic_structure: bool = False,
        static_config: Optional[Dict] = None,
        dynamic_config: Optional[Dict] = None,
        n_attention_heads: int = 4,
        # Random effects
        random_effect_dim: Optional[int] = None,
        n_random_groups: Optional[int] = None,
        # Architectural features
        activation: str = 'gelu',
        use_residual: bool = True,
        use_layer_norm: bool = True,
        norm_position: str = 'pre_structure', # 'pre_structure' or 'post_structure'
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residual = use_residual and (input_dim == output_dim)
        self.use_layer_norm = use_layer_norm
        self.norm_position = norm_position
        
        # Main linear transform: W @ H_in
        self.W = nn.Linear(input_dim, output_dim, bias=True)
        
        # Structural components (applies to output space)
        if use_static_structure or use_dynamic_structure:
            from .structural import CompositeStructuralModel, HybridStructuralModel 
            
            # Re-import to be safe/clean
            if structure_type == 'iterative':
                ModelClass = HybridStructuralModel
            else:
                ModelClass = CompositeStructuralModel
                
            d_config = dynamic_config or {}
            if 'feature_mixing' not in d_config:
                d_config['feature_mixing'] = True
            if 'n_heads' not in d_config:
                d_config['n_heads'] = n_attention_heads

            self.structure = ModelClass(
                dim=output_dim,
                use_static=use_static_structure,
                use_dynamic=use_dynamic_structure,
                static_config=static_config or {},
                dynamic_config=d_config,
            )
        else:
            self.structure = None
        
        # Random effects within layer
        if random_effect_dim is not None and n_random_groups is not None:
            self.random_effects = nn.Embedding(n_random_groups, random_effect_dim)
            self.re_proj = nn.Linear(random_effect_dim, output_dim, bias=False)
            nn.init.normal_(self.random_effects.weight, std=0.01)
        else:
            self.random_effects = None
        
        # Activation
        self.activation = get_activation(activation)
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    

    
    def forward(
        self, 
        H_in: torch.Tensor,
        group_indices: Optional[torch.Tensor] = None,
        return_pre_activation: bool = False,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through hidden layer.
        
        Args:
            H_in: Input hidden state [batch, input_dim]
            group_indices: Group indices for random effects [batch]
            return_pre_activation: Return pre-activation values
            return_attention: Return attention weights
        
        Returns:
            H_out: Output hidden state [batch, output_dim]
            (optionally: pre_activation, attention_weights)
        """
        # Linear transform: xi = W @ H_in + b
        xi = self.W(H_in)
        
        # Add random effects: xi = xi + Zu
        if self.random_effects is not None and group_indices is not None:
            re = self.random_effects(group_indices.to(self.W.weight.device))
            xi = xi + self.re_proj(re)
        
        # Normalization: Pre-structure (Separation of Scales)
        # NOTE (m2): Pre-LN normalizes xi to unit scale before the structural solve,
        # but the residual below adds unnormalized H_in. If H_in has large scale early
        # in training, the structural transformation's contribution is swamped. The
        # standard Pre-LN Transformer pattern would apply LN to H_in before the
        # residual; this is a known suboptimality retained for backward compatibility.
        if self.layer_norm is not None and self.norm_position == 'pre_structure':
            xi = self.layer_norm(xi)

        # Apply structural transformation: H = (I - B_s - B_d)^{-1} xi
        attn_weights = None
        if self.structure is not None:
            if return_attention:
                H_out, attn_weights = self.structure(xi, return_attention=True)
            else:
                H_out = self.structure(xi)
        else:
            H_out = xi
        
        # Store pre-activation for potential use
        pre_act = H_out.clone()
        
        # Activation: σ(H)
        H_out = self.activation(H_out)
        
        # Normalization: Post-structure (Legacy / Standard MLP)
        if self.layer_norm is not None and self.norm_position == 'post_structure':
            H_out = self.layer_norm(H_out)
        
        # Residual connection: H_out = H_out + H_in
        if self.use_residual:
            H_out = H_out + H_in
        
        # Dropout
        if self.dropout is not None:
            H_out = self.dropout(H_out)
        
        # Return based on flags
        if return_pre_activation and return_attention:
            return H_out, pre_act, attn_weights
        elif return_pre_activation:
            return H_out, pre_act
        elif return_attention:
            return H_out, attn_weights
        else:
            return H_out
    
    def penalties(self) -> Dict[str, torch.Tensor]:
        """Return regularization penalties for this layer."""
        penalties = {}
        
        if self.structure is not None:
            penalties.update(self.structure.penalties())
        
        if self.random_effects is not None:
            # KL divergence for random effects
            penalties['random_effects_kl'] = 0.5 * (self.random_effects.weight ** 2).sum()
        
        return penalties

    def implied_covariance(self):
        """
        Compute model-implied covariance matrix Σ_eta for Static Structural Model.
        Assumes Psi = I (standardized residual variance).
        Sigma_eta = (I-B)^-1 (I-B)^-T
        """
        if self.structure is None or not self.structure.use_static:
            return None
            
        device = self.W.weight.device
        dim = self.output_dim
        I = torch.eye(dim, device=device)
        
        # B is assumed DAG (lower triangular if sorted)
        B = self.structure.B_static.B
        I_minus_B = I - B
        
        # Inv(I-B). using solve_triangular implies B is triangular.
        # If masked DAG but not strictly triangular, use generic inverse.
        # But HybridStructuralModel enforces DAG via acyclic constraint, not strict triangularity by default unless 'mask' type.
        # Safer to use generic inverse unless we know it's triangular.
        try:
             inv_I_minus_B = torch.linalg.inv(I_minus_B)
        except RuntimeError:
             # Fallback or error
             return None
             
        # Sigma_eta = (I-B)^-1 (I-B)^-T
        Sigma_eta = inv_I_minus_B @ inv_I_minus_B.T
        return Sigma_eta
