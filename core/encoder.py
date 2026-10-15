import torch
from torch import nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from types import SimpleNamespace

from ..parsing.formula import CompiledDesign
from ..layers.sem import SEMLayer
from ..covariance import (
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
    """
    Mixed-effects encoder with variational random effects and covariance structures.

    IMPORTANT - Index 0 Semantics:
        Group index 0 is RESERVED for unknown/unseen groups in random effects.
        - Training data groups are indexed starting from 1
        - Test data with new groups receive index 0
        - Behavior controlled by `unknown_strategy` parameter:
            * "learned": Uses a learned embedding for unseen groups (self.unknown_params)
            * "zero": Uses mean of training group embeddings
        - This affects ALL random effect terms and is applied during forward()
    """
    def __init__(
        self,
        fixed_input_dim,
        embedding_dim,
        random_terms_config,
        cov_struct_map=None,
        kl_normalize=False,
        scale_random_by_terms=False,
        unknown_strategy="learned",
        use_fixed_bias=True,
        enforce_centering=True,
    ):
        """
        Args:
            fixed_input_dim: Dimension of fixed effects input
            embedding_dim: Dimension of embedding space
            random_terms_config: List of random effect term configurations
            cov_struct_map: Dict mapping term_id to covariance structure
            kl_normalize: Normalize KL by parameter count (default False). Setting
                True divides the KL sum by the total number of variational parameters,
                which destroys the ELBO's probabilistic interpretation and makes
                kl_auto_scale meaningless. Use False; the training loop's (B/N)
                mini-batch factor is the only normalization the ELBO requires.
            scale_random_by_terms: Scale random effects by number of terms
            unknown_strategy: Strategy for unseen groups - "learned" or "zero"
            enforce_centering: If True, enforce zero-mean constraint on random effects (orthogonal to intercept)
        """
        super().__init__()
        # Fixed part
        self.fixed_linear = nn.Linear(fixed_input_dim, embedding_dim, bias=use_fixed_bias)
        self.kl_normalize = kl_normalize
        self.scale_random_by_terms = scale_random_by_terms
        self.unknown_strategy = unknown_strategy
        self.enforce_centering = enforce_centering
        
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
                train_levels = cargs.get("train_levels", train_levels)
                n_train_levels = len(train_levels) if train_levels else n_levels
                
                self.variational_params[tid] = nn.ParameterList([
                    nn.Parameter(torch.zeros(n_train_levels, n_slopes, embedding_dim)), # mu
                    nn.Parameter(torch.ones(n_train_levels, n_slopes, embedding_dim) * -2.0) # log_s
                ])
                K_group = cargs.get("K_group")
                if K_group is None:
                    K_group = cargs.get("matrix_tensor")
                K_slope = cargs.get("K_slope", None)
                jitter = cargs.get("lambda_k", 1e-4)
                if K_group is None:
                    raise ValueError(f"KRON covariance for group {group_name} requires 'K_group'.")
                all_levels = cargs.get("levels", train_levels)
                self.cov_modules[tid] = KroneckerCovariance(
                    n_levels=n_levels,
                    n_slopes=n_slopes,
                    K_group=K_group,
                    full_levels=all_levels,
                    train_levels=train_levels,
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

    def forward(self, X, Z_indices, Z_slopes, use_random_effects: bool = True,
                 _transductive_override: Optional[Dict[str, torch.Tensor]] = None):
        # X: [batch, fixed_dim]
        # Z_indices: list of [batch] indices
        # Z_slopes: list of [batch, n_slopes]
        # _transductive_override: optional dict {term_id: [batch, n_slopes, emb]} with
        #   pre-computed per-sample random effect embeddings for unseen groups.
        #   Samples with index != 0 are ignored (their gathered_u is used as normal).
        
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
                
                # KL divergence: q(v) vs prior
                cov_module = self.cov_modules[tid]
                
                # Enforce centering (orthogonality to intercept)
                if self.enforce_centering and isinstance(cov_module, (IIDCovariance, UnstructuredSlopeCovariance,
                                                                       AR1Covariance, ARMACovariance,
                                                                       CompoundSymmetryCovariance)):
                     # mu: [n_levels, n_slopes, emb]
                     mu_mean = mu.mean(dim=0, keepdim=True)
                     mu = mu - mu_mean
                
                if isinstance(cov_module, KCovariance):
                    # KL between q = N(mu, diag(s^2)) and p = N(0, sigma^2 * K)
                    n_train = mu.shape[0]
                    s = torch.exp(log_s)  # [n_train, n_slopes, emb]
                    k_inv = cov_module.K_inv
                    diag_k_inv = cov_module.K_inv_diag
                    logdet_p_base = cov_module.K_logdet
                    
                    # Get learned sigma from covariance module
                    # sigma is [n_slopes, 1] usually
                    if hasattr(cov_module, 'log_sigma'):
                        sigma_sq = torch.exp(cov_module.log_sigma * 2.0).view(1, -1, 1) # [1, n_slopes, 1]
                        log_sigma_sq = cov_module.log_sigma * 2.0 # [n_slopes, 1]
                    else:
                        sigma_sq = torch.ones(1, n_slopes, 1, device=mu.device)
                        log_sigma_sq = torch.zeros(n_slopes, 1, device=mu.device)

                    # Ensure cache on the correct device (in case model was moved)
                    if k_inv.device != mu.device:
                        k_inv = k_inv.to(mu.device)
                        diag_k_inv = diag_k_inv.to(mu.device)
                        logdet_p_base = logdet_p_base.to(mu.device)

                    kl_term = 0.0
                    # Iterate slopes and embeddings
                    for sl in range(mu.shape[1]):
                        sig2 = sigma_sq[0, sl, 0]
                        nb_log_sig2 = n_train * log_sigma_sq[sl, 0]
                        # logdet_p = logdet(sig2 * K) = N * log(sig2) + logdet(K)
                        logdet_p = logdet_p_base + nb_log_sig2
                        
                        for e in range(mu.shape[2]):
                            mu_vec = mu[:, sl, e]  # [n_train]
                            s_vec = s[:, sl, e]    # stds
                            logdet_q = 2.0 * torch.log(torch.clamp(s_vec, min=1e-8)).sum()
                            
                            # trace(Sigma_p^{-1} Sigma_q) -> trace(K^{-1}/sig2 * s^2)
                            trace_term = (diag_k_inv * (s_vec ** 2)).sum() / sig2
                            
                            # mu^T Sigma_p^{-1} mu -> mu^T K^{-1} mu / sig2
                            quad_term = (mu_vec @ (k_inv @ mu_vec)) / sig2
                            
                            kl_term = kl_term + 0.5 * (trace_term + quad_term - n_train + logdet_p - logdet_q)
                    kl_sum = kl_sum + kl_term
                    kl_elems += (mu.numel() + log_s.numel())

                else:
                    # IID/other priors: q(v) vs N(0, I)
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
                if isinstance(cov_module, (KCovariance, KroneckerCovariance, GPCovariance)):
                    u_expected = cov_module(v_input, indices)  # already [batch, slopes, emb]
                    # Enforce centering for K/Kronecker/GP: subtract mean over the full
                    # level array so random effects remain mean-zero (identifiability).
                    # cov_module returns batch-gathered values; call without indices to
                    # get the full [n_levels, slopes, emb] array for mean computation.
                    if self.enforce_centering:
                        u_all_full = cov_module(v_input, None)  # [n_levels, slopes, emb]
                        u_mean = u_all_full.mean(dim=0, keepdim=True)  # [1, slopes, emb]
                        u_expected = u_expected - u_mean
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
                    # Transductive override: replace per-sample embeddings for unseen groups
                    # that have been adapted via E-step in adapt_unseen_groups().
                    if _transductive_override is not None and tid in _transductive_override:
                        unseen_mask = (indices == 0)
                        if unseen_mask.any():
                            gathered_u = gathered_u.clone()
                            gathered_u[unseen_mask] = _transductive_override[tid][unseen_mask]
                
                # Weighted sum
                term_contrib = (gathered_u * slopes_val.unsqueeze(-1)).sum(dim=1)
                random_out += term_contrib
            
        kl_out = kl_sum / max(kl_elems, 1) if self.kl_normalize else kl_sum
        random_out_scaled = random_out / n_terms if self.scale_random_by_terms else random_out
        return fixed_out + random_out_scaled, kl_out

    def get_variance_summary(self):
        """
        Returns a dict of term_id -> {sigmas, rho (if AR1), type}.

        sigmas always contains the MARGINAL standard deviation of the random effect
        in the embedding space, not the innovation std:
          - IID/K/GP: sigma (already marginal)
          - AR1/ARMA: sigma / sqrt(1 - rho^2)  (stationary marginal std)
          - Kronecker: sqrt(diag(L_slope @ L_slope.T))
        """
        summary = {}
        for term_id, mod in self.cov_modules.items():
            vals = {}

            if isinstance(mod, AR1Covariance):
                sigma = torch.exp(mod.log_sigma).detach().cpu()
                rho = torch.tanh(mod.raw_rho).detach().cpu()
                # Marginal std of stationary AR1: sigma / sqrt(1 - rho^2)
                marginal = (sigma / torch.sqrt(1.0 - rho ** 2 + 1e-8)).numpy().flatten()
                vals['sigmas'] = marginal
                vals['rho'] = rho.numpy().flatten()
                vals['innovation_sigma'] = sigma.numpy().flatten()

            elif isinstance(mod, ARMACovariance):
                # For ARMA, report innovation sigma; marginal variance is complex
                sigma = torch.exp(mod.log_sigma).detach().cpu().numpy().flatten()
                vals['sigmas'] = sigma
                if mod.phi_raw is not None:
                    phi = torch.tanh(mod.phi_raw).detach().cpu().numpy()
                    vals['phi'] = phi

            elif hasattr(mod, 'log_sigma'):
                sigmas = torch.exp(mod.log_sigma).detach().cpu().numpy().flatten()
                vals['sigmas'] = sigmas
                if hasattr(mod, 'raw_rho'):
                    rho = torch.tanh(mod.raw_rho).detach().cpu().numpy().flatten()
                    vals['rho'] = rho

            elif hasattr(mod, 'raw_lower_slope'):
                # KroneckerCovariance: marginal std from slope Cholesky
                import torch.nn.functional as F
                tril = torch.tril(mod.raw_lower_slope)
                jitter = getattr(mod, 'jitter', 1e-4)
                diag_vals = F.softplus(torch.diag(mod.raw_lower_slope)) + jitter
                diag_mat = torch.diag(diag_vals)
                L = tril - torch.diag(torch.diag(tril)) + diag_mat
                Sigma = L @ L.t()
                sigmas = torch.sqrt(torch.diag(Sigma)).detach().cpu().numpy()
                vals['sigmas'] = sigmas

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
    if getattr(design, "design_info", None) is not None:
        return design.design_info.matrix_shape[1]

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
        kl_normalize: bool = False,
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
            kl_normalize=kl_normalize,
            scale_random_by_terms=False,  # Hardcoded optimal default
            use_fixed_bias=not compiled_design.intercept_included,
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
