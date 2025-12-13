import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SAMLayer(nn.Module):
    def __init__(self, dim, sam_alpha_init=0.0):
        """
        SAM Layer: y = (I-B)^{-1} x
        B is lower triangular with zero diagonal.
        """
        super().__init__()
        self.dim = dim
        
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
        
    def get_B(self, deterministic=False):
        # Gate
        if deterministic:
            gates = torch.sigmoid(self.alpha_logits)
        else:
            gates = torch.sigmoid(self.alpha_logits)
            
        B = gates * self.B_tilde
        return B * self.mask

    def forward(self, x):
        # x: [batch, dim]
        B = self.get_B(deterministic=not self.training)
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
        y_t = torch.linalg.solve_triangular(A, x_t, upper=False)
        return y_t.t()

    def penalty(self):
        # L1-like penalty on gates: sum sigmoid(alpha)
        # Only strict lower triangle
        sigs = torch.sigmoid(self.alpha_logits) * self.mask
        return sigs.sum()

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
    def __init__(self, fixed_input_dim, embedding_dim, random_terms_config, cov_struct_map=None):
        super().__init__()
        # Fixed part
        self.fixed_linear = nn.Linear(fixed_input_dim, embedding_dim)
        
        self.random_terms = nn.ModuleDict()
        self.variational_params = nn.ParameterDict() # For mu, log_scale
        # Deprecate simple variance_components dict, use Covariance modules
        self.cov_modules = nn.ModuleDict()
        
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
                    nn.Parameter(torch.ones(n_levels, n_slopes, embedding_dim) * -3.0) # log_s
                ])
                self.cov_modules[tid] = AR1Covariance(n_levels, n_slopes)

            elif ctype in ["AR", "ARMA"]:
                p = int(cargs.get("p", 1))
                q = int(cargs.get("q", 0))
                self.variational_params[tid] = nn.ParameterList([
                    nn.Parameter(torch.zeros(n_levels, n_slopes, embedding_dim)), # mu
                    nn.Parameter(torch.ones(n_levels, n_slopes, embedding_dim) * -3.0) # log_s
                ])
                self.cov_modules[tid] = ARMACovariance(n_levels, n_slopes, p=p, q=q)

            elif ctype in ["CS", "COMPOUND_SYMMETRY"]:
                self.variational_params[tid] = nn.ParameterList([
                    nn.Parameter(torch.zeros(n_levels, n_slopes, embedding_dim)), # mu
                    nn.Parameter(torch.ones(n_levels, n_slopes, embedding_dim) * -3.0) # log_s
                ])
                self.cov_modules[tid] = CompoundSymmetryCovariance(n_levels, n_slopes)

            elif ctype in ["UN", "UNSTRUCTURED"]:
                self.variational_params[tid] = nn.ParameterList([
                    nn.Parameter(torch.zeros(n_levels, n_slopes, embedding_dim)), # mu
                    nn.Parameter(torch.ones(n_levels, n_slopes, embedding_dim) * -3.0) # log_s
                ])
                self.cov_modules[tid] = UnstructuredSlopeCovariance(n_levels, n_slopes)

            elif ctype == "KRON":
                self.variational_params[tid] = nn.ParameterList([
                    nn.Parameter(torch.zeros(n_levels, n_slopes, embedding_dim)), # mu
                    nn.Parameter(torch.ones(n_levels, n_slopes, embedding_dim) * -3.0) # log_s
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
                    nn.Parameter(torch.ones(n_train_levels, n_slopes, embedding_dim) * -3.0) # log_s for train
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
                    nn.Parameter(torch.ones(n_train_levels, n_slopes, embedding_dim) * -3.0) # log_s for train
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
                self.variational_params[tid] = nn.ParameterList([
                    nn.Parameter(torch.zeros(n_levels, n_slopes, embedding_dim)), # mu
                    nn.Parameter(torch.ones(n_levels, n_slopes, embedding_dim) * -3.0) # log_s
                ])
                self.cov_modules[tid] = IIDCovariance(n_levels, n_slopes)

    def forward(self, X, Z_indices, Z_slopes):
        # X: [batch, fixed_dim]
        # Z_indices: list of [batch] indices
        # Z_slopes: list of [batch, n_slopes]
        
        # Fixed part
        fixed_out = self.fixed_linear(X)
        
        # Random parts
        random_out = torch.zeros_like(fixed_out)
        kl_div = 0
        
        for i, term in enumerate(self.config):
            tid = term['term_id']
            indices = Z_indices[i] # [batch]
            slopes_val = Z_slopes[i] # [batch, n_slopes]
            
            mu, log_s = self.variational_params[tid]
            
            # KL divergence: q(v) vs N(0, I)
            s = torch.exp(log_s)
            var = s**2
            kl = 0.5 * (var + mu**2 - 1 - 2*log_s).sum()
            kl_div += kl
            
            v_input = mu
            
            # Apply Covariance Structure L
            cov_module = self.cov_modules[tid]
            if isinstance(cov_module, (KCovariance, GPCovariance)):
                u_expected = cov_module(v_input, indices)  # already [batch, slopes, emb]
                gathered_u = u_expected
            else:
                u_expected = cov_module(v_input) # [levels, slopes, emb]
                gathered_u = u_expected.index_select(0, indices)
            
            # Weighted sum
            term_contrib = (gathered_u * slopes_val.unsqueeze(-1)).sum(dim=1)
            random_out += term_contrib
            
        return fixed_out + random_out, kl_div

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

class TabMixNNModule(nn.Module):
    def __init__(self, 
                 fixed_input_dim, 
                 compiled_design,
                 embedding_dim=64, 
                 hidden_dims=[64],
                 sam_location="output",
                 sam_dim=None,
                 cov_struct_map=None
                 ):
        super().__init__()
        self.sam_location = sam_location
        self.outcomes = compiled_design.outcomes
        
        # 0. Mixed Effects Block
        # Needs random term config from compiled_design
        random_config = []
        for term in compiled_design.random_terms:
            random_config.append({
                'term_id': term.term_id,
                'n_levels': max(term.level_map.values()) + 1 if term.level_map else 1,
                'n_slopes': len(term.slope_cols),
                'group': term.group_col, # Pass group name for cov struct lookup
                'train_levels': getattr(term, "train_levels", [])
            })
            
        self.mixed_block = MixedEffectsRepresentation(
             fixed_input_dim=fixed_input_dim,
             embedding_dim=embedding_dim,
             random_terms_config=random_config,
             cov_struct_map=cov_struct_map
        )
        
        # SAM First
        self.sam_first = None
        if sam_location in ["first", "both", "all"] and embedding_dim > 1:
            self.sam_first = SAMLayer(embedding_dim)

        # Hidden Blocks
        self.hidden_layers = nn.ModuleList()
        curr_dim = embedding_dim
        
        for h_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(curr_dim, h_dim))
            curr_dim = h_dim
            # SAM All
            if sam_location == "all" and curr_dim > 1:
                self.hidden_layers.append(SAMLayer(curr_dim))
        
        self.last_dim = curr_dim
        
        # SAM Output (before heads or after?)
        # Blueprint 5.5: "SAM block can be applied either to latent predictors... or directly to linear predictors"
        # "Recommended... directly to linear predictors (logits) for multi-task"
        # Wait, if we apply to logits, we need to know total output dim.
        # Let's apply before heads for now as "Output SAM" usually means the latent representation before branching or mixing the heads is harder.
        # Actually, "SAM at output predictor layer only" -> "Directly to linear predictors... recommended".
        # This implies we generate ALL logits for ALL tasks, concat them, apply SAM, then split?
        # Or apply SAM on the shared representation just before heads?
        # "SAM at output predictor layer only (l=L)"
        # "r^(L,m) = g_m(h^(L-1))"
        # If SAM is on logits, it means we modify r^(L) = SAM( concat(r^(L,m)) ).
        
        # Implementation decision: Apply SAM on the shared latent `h^(L-1)` is easier ("latent predictors").
        # If we want task relatedness, applying on `h` is good. 
        # But if we want specific output correlations (e.g. failure cause 1 vs cause 2), SAM on logits is better.
        
        # Let's stick to SAM on the shared last hidden layer for simplicity in Phase A/B unless specified.
        # Blueprint 9.2 implies logits are independent unless correlated.
        # Let's do SAM on shared representation for "output" location for now.
        
        self.sam_output = None
        if sam_location == "output" or sam_location == "both":
            self.sam_output = SAMLayer(self.last_dim)
        else:
            self.sam_output = None
        
        self.sam_hidden = nn.ModuleList()
        # If location 'all', we would populate this. For now MVP only output/first/both.
            
        # Heads
        self.heads = nn.ModuleDict()
        for outcome in self.outcomes:
            family = outcome.family
            out_dim = family.n_params
            self.heads[outcome.name] = nn.Linear(self.last_dim, out_dim)

    def get_sam_layer(self, layer_name):
        if layer_name == "first":
            return self.sam_first
        elif layer_name == "output":
            return self.sam_output
        elif layer_name.startswith("hidden_"):
            idx = int(layer_name.split("_")[1])
            if 0 <= idx < len(self.sam_hidden):
                return self.sam_hidden[idx]
        return None
        
    def get_variance_summary(self):
        return self.mixed_block.get_variance_summary()

    def forward(self, X, Z_indices, Z_slopes):
        # 1. Mixed Effects
        x, kl = self.mixed_block(X, Z_indices, Z_slopes)
        x = F.relu(x) # Activation for first block?
        
        # 2. SAM First
        if self.sam_first:
            x = self.sam_first(x)
            
        # 3. Hidden Layers
        for i, layer in enumerate(self.hidden_layers):
            if isinstance(layer, SAMLayer):
                x = layer(x)
            else:
                x = F.relu(layer(x))
                
        # 4. SAM Output (Latent)
        if self.sam_output:
            x = self.sam_output(x)
            
        # 5. Heads
        outputs = {}
        for name, head in self.heads.items():
            outputs[name] = head(x)
            
        return outputs, kl

    def get_sparsity_loss(self):
        loss = 0
        if self.sam_first:
            loss += self.sam_first.penalty()
        if self.sam_output:
            loss += self.sam_output.penalty()
        for l in self.hidden_layers:
            if isinstance(l, SAMLayer):
                loss += l.penalty()
        return loss
