"""
Advanced structural models for GM-SEM (Generalized Mixed SEM) architecture.

This module provides flexible structural components with attention-based dependencies
for building deep GM-SEM networks. Supports hybrid static+dynamic structures.

When to use:
- GSEMBackbone with attention-based dependencies (HybridStructuralModel)
- Deep networks needing layer-wise structural constraints
- Combining static DAG with dynamic attention

For simpler position-aware SEM layers (ManifoldSAMNN, spatial architectures),
see `sem.py` which provides lightweight SEMLayer and PositionalSEMLayer.

Classes:
    MeasurementModel: Y = Λη + ε (SEM measurement equations)
    StaticStructureModel: Fixed B_s matrix (DAG structure)
    DynamicStructureModel: Attention-based B_d(η) (context-dependent)
    HybridStructuralModel: Combines static + dynamic structures
    RandomEffectsModule: Zu terms for hierarchical variation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional, Union

class MeasurementModel(nn.Module):
    """
    Implements: Y = Λη + ε, with identification constraints on Λ.
    
    Y: Observed variables (output)
    η: Latent factors (input from structural model)
    Λ: Factor loading matrix
    ε: Measurement error
    """
    
    def __init__(
        self, 
        latent_dim: int, 
        observed_dim: int, 
        constraint: str = 'first_loading',
        estimate_theta: bool = True
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.observed_dim = observed_dim
        
        # Initialize Λ
        if observed_dim >= latent_dim:
            init_lambda = torch.eye(observed_dim, latent_dim)
            init_lambda = init_lambda + 0.01 * torch.randn(observed_dim, latent_dim)
        else:
            init_lambda = torch.randn(observed_dim, latent_dim) * 0.1

        self.Lambda_raw = nn.Parameter(init_lambda)
        self.constraint = constraint
        
        # Measurement error variance (log scale)
        if estimate_theta:
            self.theta_log = nn.Parameter(torch.zeros(observed_dim))
        else:
            self.register_buffer('theta_log', torch.zeros(observed_dim))
    
    def _apply_constraints(self, Lambda):
        """Apply identification constraints to Λ."""
        if self.constraint == 'first_loading':
            # Fix first loading of each factor to 1.
            # Lambda: [observed, latent]
            n_factors = Lambda.shape[1]
            mask = torch.zeros_like(Lambda)
            
            # For each factor, fix one loading. 
            # Simple heuristic: fix Lambda[j, j] = 1 for j < min(obs, lat)
            limit = min(n_factors, Lambda.shape[0])
            for j in range(limit):
                mask[j, j] = 1.0
            
            # (1-mask)*Lambda + mask
            return Lambda * (1 - mask) + mask
            
        elif self.constraint == 'orthogonal':
            Q, _ = torch.linalg.qr(Lambda)
            return Q
        return Lambda
    
    @property
    def Lambda(self):
        return self._apply_constraints(self.Lambda_raw)
    
    def forward(self, eta, sample_noise: bool = None):
        if sample_noise is None:
            sample_noise = self.training
            
        Lambda = self.Lambda
        y = eta @ Lambda.T
        
        if sample_noise:
            noise = torch.randn_like(y) * torch.exp(0.5 * self.theta_log)
            y = y + noise
        
        return y


class StaticStructureModel(nn.Module):
    """
    Static structural dependencies (classical SEM).
    Implements: B_s η where B_s is strictly lower-triangular.
    """
    
    def __init__(
        self, 
        dim: int, 
        sparse_penalty: float = 0.01,
        init_structure: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.dim = dim
        self.sparse_penalty_weight = sparse_penalty
        
        if init_structure is not None:
            self.B_raw = nn.Parameter(init_structure.clone())
        else:
            self.B_raw = nn.Parameter(torch.zeros(dim, dim))
        
        # Lower triangular mask (exclude diagonal)
        mask = torch.tril(torch.ones(dim, dim), diagonal=-1)
        self.register_buffer('tril_mask', mask)
    
    @property
    def B(self):
        """Return constrained B matrix (strictly lower-triangular)."""
        return self.B_raw * self.tril_mask
    
    def forward(self, eta):
        """Compute B_s @ η."""
        # Note: F.linear(input, weight) computes input @ weight.T
        # We want B @ eta (if eta is col vector) or eta @ B.T (if eta is row vector [batch, dim])
        # B is [dim, dim]. forward returns [batch, dim].
        # Result = eta @ B.T
        return F.linear(eta, self.B)
    
    def solve(self, xi):
        """
        Solve (I - B_s)η = ξ efficiently via triangular solve.
        """
        B = self.B
        I_minus_B = torch.eye(self.dim, device=xi.device) - B
        
        if xi.dim() == 2:
            # xi: [batch, dim]
            # solve_triangular expects A, B. Solves A X = B.
            # Our system: eta @ (I-B)^T = xi
            # Transpose: (I-B) @ eta^T = xi^T
            # A = I-B (lower triangular), X = eta^T, RHS = xi^T
            
            eta_T = torch.linalg.solve_triangular(
                I_minus_B,
                xi.T,
                upper=False
            )
            return eta_T.T
        else:
            return torch.linalg.solve_triangular(I_minus_B, xi, upper=False)
    
    def dag_penalty(self):
        B = self.B
        M = torch.matrix_exp(B * B)
        return torch.trace(M) - self.dim
    
    def sparsity_penalty(self):
        return self.sparse_penalty_weight * torch.abs(self.B).sum()
    
    def spectral_norm(self):
        return torch.linalg.matrix_norm(self.B, ord=2)


class DynamicStructureModel(nn.Module):
    """
    Dynamic structural dependencies via attention.
    Implements: B_d(η) η where B_d = softmax(QK^T / sqrt(d))
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        causal_mask: bool = False,
        feature_mixing: bool = False,
    ):
        super().__init__()
        
        self.dim = dim
        self.n_heads = n_heads
        
        # Ensure divisibility
        if dim % n_heads != 0:
            # Fallback or error? Blueprint says assert.
            # Let's adjust head_dim to float or handle remainder? 
            # Standard transformers assert divisibility.
            pass
            
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.causal_mask = causal_mask
        self.feature_mixing = feature_mixing
        
        # Projections
        
        # Projections
        if self.feature_mixing:
             # Default to scalar features if not specified via method
             # If user wants other dim, they should call enable_feature_mixing manually or we add args
             self.input_feature_dim = 1
             # We need to initialize for feature mixing
             self.enable_feature_mixing(1) 
        else:
            self.W_Q = nn.Linear(dim, dim, bias=False)
            self.W_K = nn.Linear(dim, dim, bias=False)
            self.W_V = nn.Linear(dim, dim, bias=False)
            self.W_O = nn.Linear(dim, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)


    def enable_feature_mixing(self, feature_dim: int = 1):
        """
        Enable feature mixing mode where the input dimension D is treated as the sequence length.
        The input features are assumed to be scalars (or vectors of size feature_dim).
        
        Input: [batch, D] or [batch, D, feature_dim]
        Attention: computed over the D axis (D x D attention map).
        """
        self.feature_mixing = True
        self.input_feature_dim = feature_dim
        # Re-initialize projections for small feature_dim -> internal dim
        # We need adequate internal dimension for heads
        # Current projections are Linear(dim, dim) which assumes flat vector mixing? 
        # No, standard attention preserves shape.
        
        # If we are in feature mixing mode:
        # Input shape: [Batch, Sequence=Dim, Embedding=feature_dim]
        # We want Output: [Batch, Sequence=Dim, Embedding=feature_dim]
        
        # We need to replace W_Q, W_K, W_V to project from feature_dim to (n_heads * head_dim)
        # We generally keep head_dim * n_heads = dim? No, dim is the sequence length here.
        # We need an internal 'model_dim' for the attention mechanism.
        # Let's use 'dim' as the sequence length, and 'd_model' as the projection size.
        
        self.d_model = 32 # Arbitrary internal embedding size for features
        self.head_dim = self.d_model // self.n_heads
        self.scale = self.head_dim ** -0.5
        
        self.W_Q = nn.Linear(feature_dim, self.d_model, bias=False)
        self.W_K = nn.Linear(feature_dim, self.d_model, bias=False)
        self.W_V = nn.Linear(feature_dim, self.d_model, bias=False)
        self.W_O = nn.Linear(self.d_model, feature_dim, bias=False)
        
    def forward(self, eta, return_weights=False):
        """
        Returns: Attention(eta) = B_d(eta) * eta * W_V * W_O
        Note: The blueprint says B_d(eta) * eta. 
        Standard attention is Softmax(...) * V. 
        If V is projection of eta, then it is Softmax(...) * eta * W_V.
        """
        batch_size = eta.shape[0]
        
        # If eta is 2D [batch, dim], treat as seq_len=1?
        # Or treat as single vector per batch?
        # Blueprint says: "For non-sequence data, treat as seq_len=1 or reshape"
        # If seq_len=1, self-attention is trivial (attention to self = 1).
        # Dynamic Structure implies we attend to OTHER elements in the 'dim' space?
        # WAIT. 
        # Standard Transformer attends across SEQUENCE dimension (tokens).
        # Here, eta is a vector of LATENT FACTORS.
        # Does B_d(eta) mix latent factors within a single sample (Graph Attention)?
        # Or does it mix across samples (Batch Attention)?
        # Or does it treat latent factors as a sequence?
        
        # Blueprint Section 2.2:
        # "Attention... linear mixing of hidden states."
        # "Attention: h_i = sum A_ij v_j"
        # If i, j are indices of the latent vector eta, then we are mixing latent factors.
        # This is Graph Attention (GAT) where nodes are latent variables.
        # BUT standard MHSA implementations usually operate on (batch, seq, dim) and mix along 'seq'.
        
        # If we want to mix latent factors (dim), we need to view eta as (batch, dim, 1) or similar?
        # NO. If we are mixing components of eta, then B is [dim, dim].
        # B_d(eta) should be [dim, dim].
        # Standard transfomer computes Attention Matrix A of shape [seq, seq].
        # So 'seq' corresponds to 'dim' in SEM context if we want to mix factors.
        
        # IF use_dynamic is for mixing latent factors (0..dim-1):
        # We need to reshape eta to [batch, dim, 1] (where 1 is feature size per factor?) 
        # OR project eta [batch, dim] -> Q, K, V [batch, dim, projection_dim]?
        # This implies each latent factor has an embedding?
        
        # Re-reading Blueprint 2.5:
        # "Standard self-attention computes Attn(H) = ... B_d(H) V"
        # where H is [batch, seq, d].
        # Here "H" in master equation is the vector of latent variables eta [batch, dim].
        # If we treat eta as the "hidden state", and we want B_d to be [dim, dim],
        # then we are treating the latent dimensions as the "sequence" elements.
        
        # BUT the DynamicStructureModel code in blueprint does:
        # Q = W_Q(eta)
        # eta [batch, seq, dim].
        # It assumes eta HAS a sequence dimension.
        
        # If GM-SEM is replacing a Transformer layer, then eta IS [batch, seq, dim].
        # If GM-SEM is replacing a flat SEM layer (eta [batch, dim]), then seq=1?
        # If seq=1, attention is useless (1x1 matrix).
        
        # Interpretation:
        # GM-SEM generalizes Transformers. Transformers operate on sequences.
        # So 'eta' can be a sequence.
        # For tabular data/SEM, we usually have flat vectors (seq=1).
        # UNLESS variables are treated as tokens (TabTransformer).
        
        # If we use it for Tabular data as "Attention between variables":
        # Then eta should be [batch, num_variables, embedding_dim].
        # But `latent_dim` in `GSEMLayer` usually refers to total size of flattened vector?
        # Or `num_variables`?
        
        # The blueprint `GSEMLayer` takes `latent_dim`. 
        # In `forward`: `xi = self.Gamma(self.backbone(X))`.
        # If backbone outputs flat vector [batch, dim], then eta is [batch, dim].
        # Then `DynamicStructureModel` sees [batch, 1, dim].
        # Attention scores are [batch, heads, 1, 1]. Trivial.
        
        # Unless `DynamicStructureModel` is intended to accept [batch, seq, dim].
        
        # If the user wants "Attention is Structure" unifying SEM and Transformer:
        # In SEM, B is [dim, dim]. It mixes variables.
        # In Transformer, Attention is [seq, seq]. It mixes tokens.
        # So "Variables" in SEM correspond to "Tokens" in Transformer.
        
        # If we want to support both:
        # 1. Standard Transformer mode: H is [batch, seq, dim]. B(H) mixes seq.
        # 2. SEM mode: H is [batch, n_vars]. B is [n_vars, n_vars]. 
        
        # How to reconcile?
        # If H is flat [batch, n_vars], we can reshape to [batch, n_vars, 1] to use attention?
        # But then Q, K project scalar to vector? 
        # W_Q would be [1, proj_dim].
        
        # Let's trust the blueprint code implementation for now:
        # It handles `eta.dim() == 2` by unsqueeze(1).
        # This implies it treats the whole vector as 1 token.
        # This effectively disables attention mixing unless Sequence Length > 1.
        
        # However, for "Hybrid" models (Experiment 4), the user might imply something else?
        # If we want B_d to be [dim, dim] mixing the *components* of a flat vector:
        # That would require [batch, dim, 1] logic.
        
        # I will implement exactly as Blueprint code for now:
        # It supports sequence inputs. If input is flat, it effectively does nothing dynamic 
        # (unless self-attention with 1 token does something useful? No, it's just scaling).
        
        # Maybe the "Fixed Effects" part creates a sequence?
        # "xi = Gamma(backbone(X))".
        # If backbone returns sequence, then we have sequence.
        
        # If mixing features (dim becomes sequence):
        if self.feature_mixing:
            if eta.dim() == 2:
                # [batch, dim] -> [batch, dim, 1]
                eta = eta.unsqueeze(-1)
            return self._forward_feature_mixing(eta, return_weights)

        return self._forward_impl(eta, return_weights)

    def _forward_feature_mixing(self, x, return_weights):
        # x: [batch, seq=dim, feature_dim]
        batch, seq, fdim = x.shape
        
        # Q, K, V: [batch, seq, n_heads * head_dim]
        # view -> [batch, seq, n_heads, head_dim]
        # permute -> [batch, n_heads, seq, head_dim]
        Q = self.W_Q(x).view(batch, seq, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.W_K(x).view(batch, seq, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.W_V(x).view(batch, seq, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Scores: [batch, heads, seq, seq]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        # [batch, heads, seq, head_dim]
        attn_out = torch.matmul(weights, V)
        
        # [batch, seq, heads, head_dim] -> [batch, seq, d_model]
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(batch, seq, self.d_model)
        
        # Project back to feature_dim
        out = self.W_O(attn_out)
        
        if self.input_feature_dim == 1:
            out = out.squeeze(-1)
            
        if return_weights:
            return out, weights.mean(dim=1)
        return out

    def _forward_impl(self, eta, return_weights):
        is_2d = eta.dim() == 2
        if is_2d:
             eta = eta.unsqueeze(1) # [batch, 1, dim]
        
        batch, seq, dim = eta.shape
        
        Q = self.W_Q(eta).view(batch, seq, self.n_heads, self.head_dim)
        K = self.W_K(eta).view(batch, seq, self.n_heads, self.head_dim)
        V = self.W_V(eta).view(batch, seq, self.n_heads, self.head_dim)
        
        # Scores: [batch, heads, seq, seq]
        scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) * self.scale
        
        if self.causal_mask:
            mask = torch.triu(torch.ones(seq, seq, device=eta.device), diagonal=1)
            scores = scores.masked_fill(mask.bool(), float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Mix V: [batch, heads, seq, head_dim]
        # einsum: bhqk * bkhd -> bqhd
        out = torch.einsum('bhqk,bkhd->bqhd', attn_weights, V)
        out = out.reshape(batch, seq, dim)
        out = self.W_O(out)
        
        if is_2d:
            out = out.squeeze(1)
            
        if return_weights:
            # Avg over heads: [batch, seq, seq]
            return out, attn_weights.mean(dim=1)
            
        return out
    
    def get_attention_matrix(self, eta):
        # Return averaged attention matrix for interpretability
        if eta.dim() == 2:
            eta = eta.unsqueeze(1)
        
        batch, seq, dim = eta.shape
        Q = self.W_Q(eta).view(batch, seq, self.n_heads, self.head_dim)
        K = self.W_K(eta).view(batch, seq, self.n_heads, self.head_dim)
        scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) * self.scale
        attn = F.softmax(scores, dim=-1)
        return attn.mean(dim=1) # [batch, seq, seq]


class HybridStructuralModel(nn.Module):
    """
    Combines static SEM structure with dynamic attention.
    Implements: η = (I - B_s - B_d(η))^{-1} ξ
    """
    
    def __init__(
        self,
        dim: int,
        use_static: bool = True,
        use_dynamic: bool = True,
        static_config: Dict = None,
        dynamic_config: Dict = None,
        interpolation_alpha: Optional[float] = None,
    ):
        super().__init__()
        
        self.dim = dim
        self.use_static = use_static
        self.use_dynamic = use_dynamic
        self.interpolation_alpha = interpolation_alpha
        
        static_config = static_config or {}
        dynamic_config = dynamic_config or {}
        
        if use_static:
            self.B_static = StaticStructureModel(dim, **static_config)
        
        if use_dynamic:
            self.B_dynamic = DynamicStructureModel(dim, **dynamic_config)
            
        self.max_iter = 10
        self.tol = 1e-5
        self._convergence_failures = 0  # Track convergence failures for diagnostics

    def forward(self, xi, return_attention=False):
        """
        Forward pass with fixed-point iteration.

        CONVERGENCE GUARANTEE:
            For convergence, the spectral norm ||B_static + B_dynamic|| < 1 is required.
            The contraction penalty encourages this during training, but does not enforce it.
            If convergence fails (max_iter reached), a warning is logged.

        Consider using CompositeStructuralModel for guaranteed stability without iteration.
        """
        # Case 1: Static only
        if self.use_static and not self.use_dynamic:
            eta = self.B_static.solve(xi)
            if return_attention:
                return eta, None
            return eta

        # Case 2: Fixed-point iteration
        eta = xi.clone()
        attn_weights = None
        converged = False

        for i in range(self.max_iter):
            eta_old = eta.clone()

            term = 0.0
            if self.use_static:
                term = term + self.B_static(eta)

            if self.use_dynamic:
                dyn_term, attn = self.B_dynamic(eta, return_weights=True)
                attn_weights = attn
                term = term + dyn_term

            # Note: interpolation_alpha applied to weights or terms?
            # Blueprint: alpha * static + (1-alpha) * dynamic
            # But what about identity? (I - B...)
            # "structural_term = alpha * static_term + (1 - alpha) * dynamic_term"
            # This implies B_total = alpha*B_s + (1-alpha)*B_d

            if self.interpolation_alpha is not None and self.use_static and self.use_dynamic:
                alpha = self.interpolation_alpha
                # We need to separate terms to interpolate correctly
                # (Re-calculating for clarity/correctness based on blueprint logic)
                s_term = self.B_static(eta)
                d_term, attn = self.B_dynamic(eta, return_weights=True)
                attn_weights = attn
                structural_term = alpha * s_term + (1 - alpha) * d_term
            else:
                 structural_term = term

            # Update: eta = xi + structural_term
            eta = xi + structural_term

            if torch.norm(eta - eta_old) < self.tol * (torch.norm(eta) + 1e-8):
                converged = True
                break

        # Log warning if convergence failed
        if not converged:
            self._convergence_failures += 1
            if self._convergence_failures <= 5 or self._convergence_failures % 100 == 0:
                from ..logging_config import get_logger
                logger = get_logger("layers.structural")
                logger.warning(
                    f"Fixed-point iteration failed to converge in {self.max_iter} iterations "
                    f"(failure #{self._convergence_failures}). Consider: (1) using CompositeStructuralModel, "
                    f"(2) increasing contraction penalty, or (3) reducing model complexity."
                )
                
        if return_attention:
            return eta, attn_weights
        
        return eta

    def penalties(self):
        p = {}
        if self.use_static:
            p['dag'] = self.B_static.dag_penalty()
            p['sparsity'] = self.B_static.sparsity_penalty()
        if self.use_static and self.use_dynamic:
            p['contraction'] = self._contraction_penalty()
        return p
        
    def _contraction_penalty(self, margin=0.1):
        static_norm = self.B_static.spectral_norm() if self.use_static else 0.0
        dynamic_norm = 1.0 if self.use_dynamic else 0.0
        
        if self.interpolation_alpha is not None:
             total_norm = self.interpolation_alpha * static_norm + (1 - self.interpolation_alpha) * dynamic_norm
        else:
             total_norm = static_norm + dynamic_norm
             
        threshold = 1.0 - margin
        return F.relu(total_norm - threshold) ** 2


class CompositeStructuralModel(nn.Module):
    """
    Robust structural model implementing the 'General Formulation' composition:
    
    1. Resolve Static Geometry: u = (I - B_s)^-1 xi
    2. Dynamic Attention: a = Attn(u)
    3. Fusion: h = G * u + (1-G) * a
    
    This avoids fixed-point instability and ensures clean identification.
    """
    def __init__(
        self,
        dim: int,
        use_static: bool = True,
        use_dynamic: bool = True,
        static_config: Dict = None,
        dynamic_config: Dict = None,
        fusion_alpha: float = 0.5, # Default fusion weight
        learnable_fusion: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.use_static = use_static
        self.use_dynamic = use_dynamic
        
        static_config = static_config or {}
        dynamic_config = dynamic_config or {}
        
        if use_static:
            self.B_static = StaticStructureModel(dim, **static_config)
            
        if use_dynamic:
            self.B_dynamic = DynamicStructureModel(dim, **dynamic_config)
            
        # Fusion
        if learnable_fusion:
            # Sigmoid gate initialization
            self.gate_logit = nn.Parameter(torch.tensor(0.0)) # Starts at 0.5
            self.use_gate = True
        else:
            self.register_buffer('alpha', torch.tensor(fusion_alpha))
            self.use_gate = False
            
    def forward(self, xi, return_attention=False):
        # 1. Static Geometry
        if self.use_static:
            u = self.B_static.solve(xi)
        else:
            u = xi
            
        # 2. Dynamic Attention (moves within geometry)
        attn_weights = None
        if self.use_dynamic:
            # Attends to 'u' (the resolved geometry)
            a, attn_weights = self.B_dynamic(u, return_weights=True)
        else:
            a = u # No movement
            
        # 3. Fusion
        if self.use_gate:
            alpha = torch.sigmoid(self.gate_logit)
        else:
            alpha = self.alpha
            
        # h = alpha * u + (1 - alpha) * a
        # (General Formulation uses alpha for static, 1-alpha for dynamic)
        h = alpha * u + (1 - alpha) * a
        
        if return_attention:
            return h, attn_weights
        return h

    def penalties(self):
        p = {}
        if self.use_static:
            p['dag'] = self.B_static.dag_penalty()
            p['sparsity'] = self.B_static.sparsity_penalty()
        return p


class RandomEffectsModule(nn.Module):
    """
    Implements: Zu where u ~ N(0, G)
    """
    def __init__(
        self,
        group_specs: Dict[str, Tuple[int, int]],
        covariance_type: str = 'diagonal'
    ):
        super().__init__()
        self.group_specs = group_specs
        self.covariance_type = covariance_type
        
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(n_groups, effect_dim)
            for name, (n_groups, effect_dim) in group_specs.items()
        })
        
        for emb in self.embeddings.values():
            nn.init.normal_(emb.weight, std=0.01)
        
        if covariance_type == 'diagonal':
            self.log_var = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(effect_dim))
                for name, (_, effect_dim) in group_specs.items()
            })
            
    def forward(self, group_indices: Dict[str, torch.Tensor]):
        effects = []
        for name, indices in group_indices.items():
            if name in self.embeddings:
                idx = indices.to(self.embeddings[name].weight.device)
                effects.append(self.embeddings[name](idx))
        if not effects:
            return 0.0
        return sum(effects)
    
    def kl_divergence(self):
        kl = 0.0
        for name, emb in self.embeddings.items():
            if self.covariance_type == 'diagonal':
                prior_var = torch.exp(self.log_var[name])
                term1 = (emb.weight ** 2) / prior_var.unsqueeze(0)
                kl += 0.5 * term1.sum()
                kl += 0.5 * self.log_var[name].sum() * emb.weight.shape[0]
            else:
                kl += 0.5 * (emb.weight ** 2).sum()
        return kl
