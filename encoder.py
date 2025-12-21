import torch
from torch import nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from types import SimpleNamespace

from .formula import CompiledDesign
from .layers.sem import SEMLayer
from .covariance import (
    IIDCovariance,
    AR1Covariance,
    ARMACovariance,
    CompoundSymmetryCovariance,
    UnstructuredSlopeCovariance,
    KroneckerCovariance,
    KCovariance,
    GPCovariance,
)


class MixedEffectsRepresentation(nn.Module):
    def __init__(self, fixed_input_dim, embedding_dim, random_terms_config, cov_struct_map=None, kl_normalize=True, scale_random_by_terms=False, unknown_strategy="learned"):
        super().__init__()
        # Fixed part
        self.fixed_linear = nn.Linear(fixed_input_dim, embedding_dim)
        self.kl_normalize = kl_normalize
        self.scale_random_by_terms = scale_random_by_terms
        self.unknown_strategy = unknown_strategy
        
        self.random_terms = nn.ModuleDict()
        self.variational_params = nn.ParameterDict() # For mu, log_scale
        # Deprecate simple variance_components dict, use Covariance modules
        self.cov_modules = nn.ModuleDict()
        self.unknown_params = nn.ParameterDict()
        
        self.embedding_dim = embedding_dim
        self.config = random_terms_config # List of dicts
        
        for term in random_terms_config:
            tid = term['term_id']
            # e.g. term_0
            # group_col needed to lookup cov type?
            # We pass cov_struct_map which maps group_col -> type.
            # term['group_col'] needs to be in random_terms_config passed from design?
            # Currently model init only gets minimal config.
            # We assume default IID.
            
            n_levels = term['n_levels']
            n_slopes = term['n_slopes']
            group_name = term.get('group', 'unknown') # Add group name to config in next step
            train_levels = term.get('train_levels', [])
            
            # Select Covariance
            # Default IID
            ctype = "IID"
            cargs = {}
            if cov_struct_map and group_name in cov_struct_map:
                spec = cov_struct_map[group_name]
                ctype = spec.get("type", "IID").upper()
                cargs = spec
            else:
                ctype = "IID"
                cargs = {}

            if ctype == "AR1":
                self.variational_params[tid] = nn.ParameterList([
                    nn.Parameter(torch.zeros(n_levels, n_slopes, embedding_dim)), # mu
                    nn.Parameter(torch.ones(n_levels, n_slopes, embedding_dim) * -2.0) # log_s
                ])
                self.cov_modules[tid] = AR1Covariance(n_levels, n_slopes)

            elif ctype in ["AR", "ARMA"]:
                p = int(cargs.get("p", 1))
                q = int(cargs.get("q", 0))
                self.variational_params[tid] = nn.ParameterList([
                    nn.Parameter(torch.zeros(n_levels, n_slopes, embedding_dim)), # mu
                    nn.Parameter(torch.ones(n_levels, n_slopes, embedding_dim) * -2.0) # log_s
                ])
                self.cov_modules[tid] = ARMACovariance(n_levels, n_slopes, p=p, q=q)

            elif ctype in ["CS", "COMPOUND_SYMMETRY"]:
                self.variational_params[tid] = nn.ParameterList([
                    nn.Parameter(torch.zeros(n_levels, n_slopes, embedding_dim)), # mu
                    nn.Parameter(torch.ones(n_levels, n_slopes, embedding_dim) * -2.0) # log_s
                ])
                self.cov_modules[tid] = CompoundSymmetryCovariance(n_levels, n_slopes)

            elif ctype in ["UN", "UNSTRUCTURED"]:
                self.variational_params[tid] = nn.ParameterList([
                    nn.Parameter(torch.zeros(n_levels, n_slopes, embedding_dim)), # mu
                    nn.Parameter(torch.ones(n_levels, n_slopes, embedding_dim) * -2.0) # log_s
                ])
                self.cov_modules[tid] = UnstructuredSlopeCovariance(n_levels, n_slopes)

            elif ctype == "KRON":
                self.variational_params[tid] = nn.ParameterList([
                    nn.Parameter(torch.zeros(n_levels, n_slopes, embedding_dim)), # mu
                    nn.Parameter(torch.ones(n_levels, n_slopes, embedding_dim) * -2.0) # log_s
                ])
                K_group = cargs.get("K_group") or cargs.get("matrix_tensor")
                K_slope = cargs.get("K_slope", None)
                jitter = cargs.get("lambda_k", 1e-4)
                if K_group is None:
                    raise ValueError(f"KRON covariance for group {group_name} requires 'K_group'.")
                self.cov_modules[tid] = KroneckerCovariance(
                    n_levels=n_levels,
                    n_slopes=n_slopes,
                    K_group=K_group,
                    K_slope=K_slope,
                    jitter=jitter
                )

            elif ctype == "K":
                # For K, we maintain variational params only for training levels; unseen levels are predicted via Henderson BLUP.
                train_levels = cargs.get("train_levels", train_levels)
                n_train_levels = len(train_levels)
                self.variational_params[tid] = nn.ParameterList([
                    nn.Parameter(torch.zeros(n_train_levels, n_slopes, embedding_dim)), # mu for train
                    nn.Parameter(torch.ones(n_train_levels, n_slopes, embedding_dim) * -2.0) # log_s for train
                ])
                matrix_tensor = cargs.get("matrix_tensor", None)
                all_levels = cargs.get("levels", train_levels)
                lambda_k = cargs.get("lambda_k", 1e-4)
                self.cov_modules[tid] = KCovariance(
                    n_levels=n_levels,
                    n_slopes=n_slopes,
                    full_levels=all_levels,
                    train_levels=train_levels,
                    matrix_tensor=matrix_tensor,
                    lambda_k=lambda_k
                )

            elif ctype == "GP":
                train_levels = cargs.get("train_levels", train_levels)
                all_levels = cargs.get("levels", train_levels)
                coords = cargs.get("coords", None)
                if coords is None:
                    raise ValueError(f"GP covariance for group {group_name} requires 'coords'.")
                n_train_levels = len(train_levels)
                self.variational_params[tid] = nn.ParameterList([
                    nn.Parameter(torch.zeros(n_train_levels, n_slopes, embedding_dim)), # mu for train
                    nn.Parameter(torch.ones(n_train_levels, n_slopes, embedding_dim) * -2.0) # log_s for train
                ])
                lambda_k = cargs.get("lambda_k", 1e-4)
                self.cov_modules[tid] = GPCovariance(
                    n_levels=n_levels,
                    n_slopes=n_slopes,
                    full_levels=all_levels,
                    train_levels=train_levels,
                    coords=coords,
                    lambda_k=lambda_k
                )

            else:
                # Default: IID covariance
                self.variational_params[tid] = nn.ParameterList([
                    nn.Parameter(torch.zeros(n_levels, n_slopes, embedding_dim)), # mu
                    nn.Parameter(torch.ones(n_levels, n_slopes, embedding_dim) * -2.0) # log_s
                ])
                self.cov_modules[tid] = IIDCovariance(n_levels, n_slopes)

            if self.unknown_strategy == "learned":
                self.unknown_params[tid] = nn.Parameter(torch.zeros(1, n_slopes, embedding_dim))

            # Backward compatibility: expose variational means as embedding-like accessors
            self._register_compat_embedding(group_name, tid)

        # Default alias for first random effect if present
        if len(random_terms_config) > 0 and not hasattr(self, "random_intercept_embeddings"):
            first_tid = random_terms_config[0]['term_id']
            self.random_intercept_embeddings = SimpleNamespace(weight=self.variational_params[first_tid][0])

    def _register_compat_embedding(self, group_name: str, term_id: str):
        """Expose mu parameters with a .weight attribute for legacy tests."""
        if term_id not in self.variational_params:
            return
        mu_param = self.variational_params[term_id][0]
        wrapper = SimpleNamespace(weight=mu_param)
        setattr(self, f"random_intercept_embeddings_{group_name}", wrapper)

    def forward(self, X, Z_indices, Z_slopes, use_random_effects: bool = True):
        # X: [batch, fixed_dim]
        # Z_indices: list of [batch] indices
        # Z_slopes: list of [batch, n_slopes]
        
        # Fixed part
        fixed_out = self.fixed_linear(X)
        
        # Random parts
        random_out = torch.zeros_like(fixed_out)
        kl_sum = 0.0
        kl_elems = 0
        n_terms = max(len(self.config), 1)
        
        if use_random_effects:
            for i, term in enumerate(self.config):
                tid = term['term_id']
                indices = Z_indices[i] # [batch]
                slopes_val = Z_slopes[i] # [batch, n_slopes]
                
                mu, log_s = self.variational_params[tid]
                
                # KL divergence: q(v) vs N(0, I)
                s = torch.exp(log_s)
                var = s**2
                kl = 0.5 * (var + mu**2 - 1 - 2*log_s).sum()
                kl_sum = kl_sum + kl
                kl_elems += (mu.numel() + log_s.numel())
                
                if self.training:
                    noise = torch.randn_like(mu)
                    v_input = mu + noise * torch.exp(log_s)
                else:
                    # Use mean at evaluation time for deterministic predictions
                    v_input = mu
                
                # Apply Covariance Structure L
                cov_module = self.cov_modules[tid]
                if isinstance(cov_module, (KCovariance, GPCovariance)):
                    u_expected = cov_module(v_input, indices)  # already [batch, slopes, emb]
                    if self.unknown_strategy == "learned":
                        unknown_row = self.unknown_params[tid]
                    else:
                        unknown_row = u_expected.mean(dim=0, keepdim=True)
                    if (indices == 0).any():
                        u_expected = u_expected.clone()
                        u_expected[indices == 0] = unknown_row
                    gathered_u = u_expected
                else:
                    u_expected = cov_module(v_input) # [levels, slopes, emb]
                    # Reserve index 0 for unseen groups -> configurable strategy
                    if self.unknown_strategy == "learned":
                        unknown_row = self.unknown_params[tid]
                    else:
                        unknown_row = u_expected.mean(dim=0, keepdim=True)
                    u_padded = torch.cat([unknown_row, u_expected], dim=0)
                    gathered_u = u_padded.index_select(0, indices.clamp(min=0))
                
                # Weighted sum
                term_contrib = (gathered_u * slopes_val.unsqueeze(-1)).sum(dim=1)
                random_out += term_contrib
            
        kl_out = kl_sum / max(kl_elems, 1) if self.kl_normalize else kl_sum
        random_out_scaled = random_out / n_terms if self.scale_random_by_terms else random_out
        return fixed_out + random_out_scaled, kl_out

    def get_variance_summary(self):
        """
        Returns a dict of term_id -> {slope_idx: sigma_value, type: cov_type}
        """
        summary = {}
        for term_id, mod in self.cov_modules.items():
            # Assume mod has log_sigma or similar
            # IID and AR1 have log_sigma
            # K has log_sigma
            
            # Access params directly? Or rely on convention.
            vals = {}
            if hasattr(mod, 'log_sigma'):
                sigmas = torch.exp(mod.log_sigma).detach().cpu().numpy().flatten()
                vals['sigmas'] = sigmas
            
            if hasattr(mod, 'raw_rho'): # AR1
                rho = torch.tanh(mod.raw_rho).detach().cpu().numpy().flatten()
                vals['rho'] = rho
                
            vals['type'] = mod.__class__.__name__
            summary[term_id] = vals
        return summary


@dataclass
class EncodedBatch:
    """
    Container returned by MixedEffectsEncoder.encode_dataframe.
    """
    X: torch.Tensor
    Z_indices: List[torch.Tensor]
    Z_slopes: List[torch.Tensor]
    targets: Dict[str, torch.Tensor]


def _build_random_config(design: CompiledDesign) -> List[Dict[str, Any]]:
    cfg = []
    for term in design.random_terms:
        cfg.append(
            {
                "term_id": term.term_id,
                "n_levels": max(term.level_map.values()) + 1 if term.level_map else 1,
                "n_slopes": len(term.slope_cols),
                "group": term.group_col,
                "train_levels": getattr(term, "train_levels", []),
            }
        )
    return cfg


def _infer_fixed_input_dim(design: CompiledDesign) -> int:
    dim = 1 if design.intercept_included else 0
    for spec in design.fixed_cols:
        if spec["type"] == "numeric":
            dim += 1
        else:
            # Use the actual dummy columns emitted by the design (drops base if intercept)
            dummy_levels = design.categorical_dummy_levels.get(
                spec["name"], design.categorical_levels[spec["name"]]
            )
            dim += len(dummy_levels)
    return dim


class MixedEffectsEncoder(nn.Module):
    """
    Reusable tabular encoder that wraps the mixed-effects block (and optional SAM)
    so it can be dropped into arbitrary PyTorch models as the first layer.

    Typical usage:
        encoder = MixedEffectsEncoder.from_formula("~ x1 + (1|g)", df_train)
        batch = encoder.encode_dataframe(df_batch)
        h, kl = encoder(batch.X, batch.Z_indices, batch.Z_slopes)
        out = downstream_mlp(h)
        loss = task_loss(out, y) + lambda_kl * kl + lambda_sparse * encoder.sparsity_loss()
    """

    def __init__(
        self,
        compiled_design: CompiledDesign,
        embedding_dim: int = 64,
        cov_struct_map: Optional[Dict[str, Any]] = None,
        add_sam: bool = False,
        fixed_input_dim: Optional[int] = None,
    ):
        super().__init__()
        self.design = compiled_design

        fixed_dim = fixed_input_dim or _infer_fixed_input_dim(compiled_design)
        random_config = _build_random_config(compiled_design)

        self.mixed_block = MixedEffectsRepresentation(
            fixed_input_dim=fixed_dim,
            embedding_dim=embedding_dim,
            random_terms_config=random_config,
            cov_struct_map=cov_struct_map,
            kl_normalize=True,  # Hardcoded optimal default
            scale_random_by_terms=False,  # Hardcoded optimal default
        )

        self.mixed_activation = None  # Hardcoded: no activation after mixed block

        self.sam_proj = None
        self.sam = None
        self.output_dim = embedding_dim
        if add_sam and embedding_dim > 1:
            # SAM always matches embedding_dim (no separate sam_dim parameter)
            self.sam = SEMLayer(embedding_dim)
            self.output_dim = embedding_dim

    def forward(self, X, Z_indices, Z_slopes):
        h, kl = self.mixed_block(X, Z_indices, Z_slopes)
        if self.mixed_activation is not None:
            h = self.mixed_activation(h)
        if self.sam is not None:
            h = self.sam(h)
        return h, kl

    def encode_dataframe(self, df, device: Optional[torch.device] = None, include_targets: bool = False) -> EncodedBatch:
        """
        Transform a DataFrame into tensors aligned with the encoder.
        """
        X, Z_ind, Z_slopes, Ys = self.design.transform(df, mode="predict")
        device = device or self._infer_device()

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        Z_ind_t = [torch.tensor(z, dtype=torch.long, device=device) for z in Z_ind]
        Z_slopes_t = [torch.tensor(z, dtype=torch.float32, device=device) for z in Z_slopes]

        targets_t: Dict[str, torch.Tensor] = {}
        if include_targets:
            for k, v in Ys.items():
                targets_t[k] = torch.tensor(v, dtype=torch.float32, device=device)

        return EncodedBatch(X=X_t, Z_indices=Z_ind_t, Z_slopes=Z_slopes_t, targets=targets_t)

    def sparsity_loss(self, lambda_sparse: float = 1.0):
        if self.sam is None:
            return torch.tensor(0.0, device=self._infer_device())
        return lambda_sparse * self.sam.penalty()

    def _infer_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @classmethod
    def from_formula(
        cls,
        formula: str,
        df,
        embedding_dim: int = 64,
        cov_struct: Optional[Dict[str, Any]] = None,
        add_sam: bool = False,
    ):
        design = CompiledDesign(
            formula=formula,
            outcomes=[],
            sam_location="first",
            sam_dim=None,  # Always match layer width
            cov_struct_map=cov_struct,
        )
        design.fit(df)
        return cls(
            compiled_design=design,
            embedding_dim=embedding_dim,
            cov_struct_map=cov_struct,
            add_sam=add_sam,
        )


__all__ = ["MixedEffectsEncoder", "EncodedBatch", "MixedEffectsRepresentation"]
