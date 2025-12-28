"""
Summary and parameter extraction utilities for TabMixNN models.

Provides lme4/sommer/lavaan-style summaries adapted for neural network mixed models.
"""

import torch
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List


def _design_feature_names(design) -> List[str]:
    """Return encoded fixed-effect column names in design order."""
    names: List[str] = []
    if design is None:
        return names
    for spec in getattr(design, "fixed_cols", []):
        if spec.get("type") == "numeric":
            names.append(spec["name"])
        elif spec.get("type") == "categorical":
            levels = design.categorical_dummy_levels.get(
                spec["name"], design.categorical_levels.get(spec["name"], [])
            )
            for lvl in levels:
                names.append(f"{spec['name']}[{lvl}]")
    return names


def _is_strictly_linear(model) -> bool:
    """Detect when the pipeline is linear so closed-form coefficients apply."""
    try:
        no_hidden = hasattr(model, "hidden_dims") and len(model.hidden_dims) == 0
        backbone_layers = getattr(getattr(model, "backbone", None), "hidden_layers", [])
        return model.architecture == "gsem" and no_hidden and (backbone_layers is None or len(backbone_layers) == 0)
    except Exception:
        return False


def _closed_form_fixed_effects(model, feature_names: List[str]) -> Optional[pd.DataFrame]:
    """Exact fixed-effect coefficients when the model reduces to an LMM/GLM."""
    try:
        enc = model.encoder.fixed_linear
        head_name = list(model.heads.keys())[0]
        head = model.heads[head_name]

        enc_w = enc.weight.detach().cpu().numpy()
        enc_b = enc.bias.detach().cpu().numpy() if enc.bias is not None else np.zeros(enc_w.shape[0])

        if hasattr(head, "weight"):
            head_w = head.weight.detach().cpu().numpy()
        else:
            head_w = np.ones((1, enc_w.shape[0]))
        head_b = head.bias.detach().cpu().numpy() if hasattr(head, "bias") and head.bias is not None else np.zeros(head_w.shape[0])

        eff_w = head_w @ enc_w  # [out_dim, fixed_dim]
        eff_b = head_b + head_w @ enc_b  # [out_dim]

        slopes = eff_w[0]
        intercept = eff_b[0]
        names = ["(Intercept)"] + (feature_names or [f"x{i}" for i in range(len(slopes))])
        coefficients = np.concatenate([[intercept], slopes])
        return pd.DataFrame({"Estimate": coefficients, "Method": "closed_form"}, index=names)
    except Exception:
        return None


def _zero_random_effects(model):
    """Temporarily zero random-effect means to isolate fixed effects."""
    backups = {"mu": {}, "unknown": {}}
    enc = getattr(model, "encoder", None)
    if enc is None or not hasattr(enc, "variational_params"):
        return backups
    for term_id, params in enc.variational_params.items():
        mu = params[0]
        backups["mu"][term_id] = mu.detach().clone()
        mu.data.zero_()
    for term_id, param in getattr(enc, "unknown_params", {}).items():
        backups["unknown"][term_id] = param.detach().clone()
        param.data.zero_()
    return backups


def _restore_random_effects(model, backups):
    enc = getattr(model, "encoder", None)
    if enc is None:
        return
    for term_id, buf in backups.get("mu", {}).items():
        if term_id in enc.variational_params:
            enc.variational_params[term_id][0].data.copy_(buf)
    for term_id, buf in backups.get("unknown", {}).items():
        if term_id in enc.unknown_params:
            enc.unknown_params[term_id].data.copy_(buf)


def _linearized_fixed_effects(model, df, feature_names: List[str], sample_size: int = 256) -> Optional[pd.DataFrame]:
    """First-order Taylor coefficients of the network output with random effects suppressed."""
    design = getattr(model, "design", None)
    if design is None or df is None:
        return None
    try:
        X_np, Z_ind, Z_slopes, _ = design.transform(df, mode="predict")
        if len(X_np) == 0:
            return None
        idx = np.arange(len(X_np))
        if sample_size and sample_size < len(X_np):
            idx = np.random.choice(len(X_np), size=sample_size, replace=False)

        device = getattr(model, "device", "cpu")
        Xt = torch.tensor(X_np[idx], dtype=torch.float32, device=device, requires_grad=True)

        # Zero-out random-effect inputs to isolate the fixed component
        Z_ind_zero = [torch.zeros(len(idx), dtype=torch.long, device=device) for _ in Z_ind]
        Z_slopes_zero = []
        for slopes in Z_slopes:
            z = torch.zeros((len(idx), slopes.shape[1]), dtype=torch.float32, device=device)
            Z_slopes_zero.append(z)

        backups = _zero_random_effects(model)
        try:
            model.eval()
            model.zero_grad(set_to_none=True)
            outputs = model.forward(Xt, Z_ind_zero, Z_slopes_zero)
            logits = outputs[list(outputs.keys())[0]] if isinstance(outputs, dict) else outputs
            if logits.ndim > 1:
                logits = logits[:, 0]
            mean_pred = logits.mean()
            mean_pred.backward()
            grads = Xt.grad.detach().cpu().mean(dim=0).numpy()
            x_mean = Xt.detach().cpu().mean(dim=0).numpy()
            intercept = float(mean_pred.detach().cpu().item() - np.dot(grads, x_mean))
            names = ["(Intercept)"] + (feature_names or [f"x{i}" for i in range(len(grads))])
            coefficients = np.concatenate([[intercept], grads])
            return pd.DataFrame({"Estimate": coefficients, "Method": "linearized"}, index=names)
        finally:
            _restore_random_effects(model, backups)
    except Exception:
        return None


def _collect_random_effects(model) -> pd.DataFrame:
    """Return BLUP-style random effect estimates as a DataFrame."""
    enc = getattr(model, "encoder", None)
    design = getattr(model, "design", None)
    if enc is None or not hasattr(enc, "variational_params"):
        return pd.DataFrame()

    from .covariance.structures import KCovariance, KroneckerCovariance, GPCovariance

    records: List[Dict[str, Any]] = []
    for term_cfg in getattr(enc, "config", []):
        tid = term_cfg.get("term_id")
        if tid not in enc.variational_params:
            continue
        mu_param, log_s_param = enc.variational_params[tid]
        
        cov = enc.cov_modules[tid] if tid in enc.cov_modules else None
        
        # Calculate expected Random Effects u = E[u|v]
        # For structured covariances (GBLUP), this performs extrapolation to all levels
        with torch.no_grad():
            if isinstance(cov, (KCovariance, KroneckerCovariance, GPCovariance)):
                n_total = getattr(cov, "n_levels", mu_param.shape[0])
                all_indices = torch.arange(n_total, device=mu_param.device)
                u_full = cov(mu_param, all_indices) # [N_total, Slopes, Emb]
            else:
                # IID, AR1, etc - apply transformation L if needed (usually L*v)
                # cov(v) returns u
                u_full = cov(mu_param) if cov else mu_param # [N_train, Slopes, Emb]

        u_np = u_full.detach().cpu().numpy()
        sd_np = np.exp(log_s_param.detach().cpu().numpy())
        
        # Covariance scale (if applicable)
        scale = None
        if cov is not None and hasattr(cov, "log_sigma"):
            scale = np.exp(cov.log_sigma.detach().cpu().numpy())

        term_info = None
        if design is not None:
            term_info = next((t for t in design.random_terms if t.term_id == tid), None)
        slope_names = term_info.slope_cols if term_info else [f"slope_{j}" for j in range(u_np.shape[1])]
        
        # Determine Level Labels
        # Prioritize covariance module's full_levels (handled GBLUP/Extrapolation correctly)
        cov_levels = getattr(cov, "full_levels", None)
        if cov_levels is not None:
            all_labels = cov_levels
        elif term_info and term_info.level_map:
            inv_map = {v: k for k, v in term_info.level_map.items()}
            # Ensure we cover the full range of output
            all_labels = [inv_map.get(i, f"level_{i}") for i in range(u_np.shape[0])]
        else:
            all_labels = [f"level_{i}" for i in range(u_np.shape[0])]
            
        group_name = term_info.group_col if term_info else term_cfg.get("group", tid)

        # Iterate over all produced levels
        for i, level_label in enumerate(all_labels):
            if i >= u_np.shape[0]: 
                break
                
            # Attempt to determine if observed or extrapolated
            # This is heuristics; strict checking would require checking indices mapping
            kind = "observed"
            sd_val_vec = None
            if i < sd_np.shape[0] and not isinstance(cov, (KCovariance, KroneckerCovariance, GPCovariance)):
                 # IID/AR1 case: 1-to-1 mapping
                 sd_val_vec = sd_np[i]
            elif isinstance(cov, (KCovariance, KroneckerCovariance, GPCovariance)):
                 # Check if this level index i was in training set?
                 # cov.train_indices maps train_idx -> full_idx
                 # But we don't have reverse map easily exposed publicly
                 # For now, label 'estimated'
                 kind = "estimated"
                 # We don't have posterior SD for extrapolated levels readily computed here without heavy lifting
                 # So we leave SD as NaN for extrapolated, or use training mean SD roughly
                 sd_val_vec = np.zeros_like(u_np[i]) * np.nan
            else:
                 sd_val_vec = np.zeros_like(u_np[i]) * np.nan

            for slope_idx, slope_name in enumerate(slope_names):
                est = u_np[i, slope_idx].mean()
                
                # Apply scale if it wasn't integrated into 'cov()' call
                # Most cov modules include sigma in forward? 
                # KCovariance forward: Y = L_slope @ V_e @ L_group.t(). 
                # It does NOT multiply by sigma usually, sigma is separate in KL?
                # Check structures.py: Scale might be needed.
                if scale is not None:
                     sigma_slice = scale[slope_idx] if scale.ndim > 1 else scale
                     # Check if cov forward already applied it?
                     # IID forward: v * noise -> no. 
                     # We are transforming MEAN parameter. 
                     # u = mu. Scaling by sigma happens at "gather" time in Encoder?
                     # In encoder.py: term_contrib = (gathered_u * slopes_val).
                     # gathered_u IS the effect.
                     # It comes from cov_module(v).
                     # So if cov_module includes sigma, it's there. 
                     # Most don't include scalar sigma in forward, they return standardized u?
                     # No, encoder.py calculates KL with sigma.
                     # But where is sigma applied to u?
                     pass # Revisit coherence of sigma application
                
                # For now assume u_np is the effect on the linear predictor scale
                
                sd_val = float(sd_val_vec[slope_idx].mean()) if sd_val_vec is not None else np.nan
                records.append(
                    {
                        "group": group_name,
                        "level": str(level_label),
                        "slope": slope_name,
                        "estimate": float(est),
                        "sd": float(sd_val),
                        "kind": kind,
                        "term_id": tid,
                    }
                )
    return pd.DataFrame(records)


def _collect_variance_components(model) -> pd.DataFrame:
    enc = getattr(model, "encoder", None)
    design = getattr(model, "design", None)
    if enc is None or not hasattr(enc, "get_variance_summary"):
        return pd.DataFrame()

    summary = enc.get_variance_summary()
    records: List[Dict[str, Any]] = []

    for term_cfg in getattr(enc, "config", []):
        tid = term_cfg.get("term_id")
        vals = summary.get(tid, {})
        term_info = None
        if design is not None:
            term_info = next((t for t in design.random_terms if t.term_id == tid), None)
        slope_names = term_info.slope_cols if term_info else [f"slope_{j}" for j in range(len(vals.get("sigmas", [])))]
        group_name = term_info.group_col if term_info else term_cfg.get("group", tid)
        sigmas = vals.get("sigmas", [])
        for j, s in enumerate(sigmas):
            slope = slope_names[j] if j < len(slope_names) else f"slope_{j}"
            records.append(
                {
                    "group": group_name,
                    "component": slope,
                    "type": vals.get("type", "Unknown"),
                    "sd": float(s),
                }
            )
        if "rho" in vals:
            records.append(
                {
                    "group": group_name,
                    "component": "rho",
                    "type": vals.get("type", "Unknown"),
                    "sd": float(vals["rho"][0]) if hasattr(vals["rho"], "__len__") else float(vals["rho"]),
                }
            )

    # Residual variance if available
    family = None
    if hasattr(model, "families") and model.families:
        # Use first outcome
        family = list(model.families.values())[0]
    elif hasattr(model, "design") and getattr(model.design, "outcomes", None):
        family = model.design.outcomes[0].family

    if family is not None and hasattr(family, "sigma"):
        sigma_e = float(family.sigma.item())
        records.append({"group": "Residual", "component": "eps", "type": "Residual", "sd": sigma_e})

    return pd.DataFrame(records)


def _collect_structural_edges(model, layer_idx: int = 0):
    if not hasattr(model, "extract_sam"):
        return None, pd.DataFrame()
    B, probs = model.extract_sam(layer_idx=layer_idx)
    if B is None:
        return None, pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            if i == j:
                continue
            weight = float(B[i, j])
            if abs(weight) < 1e-8:
                continue
            prob_val = float(probs[i, j]) if probs is not None else None
            rows.append({"from": j, "to": i, "weight": weight, "prob": prob_val})

    edges_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["from", "to", "weight", "prob"])
    return B, edges_df


def parameter_estimates(model, data=None, sample_size: int = 256, linearize: bool = True) -> Dict[str, Any]:
    """
    Extract interpretable parameter estimates from a fitted model.

    Args:
        model: TabMixNN model
        data: Optional dataframe for linearization; defaults to training data if available.
        sample_size: Subsample size for gradient-based linearization.
        linearize: If True, compute first-order Taylor coefficients when the model is non-linear.
    """
    df = data if data is not None else getattr(model, "_train_data", None)
    feature_names = _design_feature_names(getattr(model, "design", None))

    fixed_effects = None
    notes: List[str] = []
    if _is_strictly_linear(model):
        fixed_effects = _closed_form_fixed_effects(model, feature_names)
    elif linearize:
        fixed_effects = _linearized_fixed_effects(model, df, feature_names, sample_size=sample_size)
        if fixed_effects is None:
            notes.append("Linearization failed; falling back to encoder/head weights.")
    if fixed_effects is None:
        fixed_effects = _closed_form_fixed_effects(model, feature_names)
        if fixed_effects is not None and not _is_strictly_linear(model):
            notes.append("Model is non-linear; fixed effects are from the first linear layer.")

    random_effects = _collect_random_effects(model)
    variance_components = _collect_variance_components(model)
    B_matrix, structural_edges = _collect_structural_edges(model)

    return {
        "fixed_effects": fixed_effects,
        "random_effects": random_effects,
        "variance_components": variance_components,
        "structural_edges": structural_edges,
        "structural_matrix": B_matrix,
        "notes": notes,
    }


class TabMixNNSummary:
    """
    Summary object for TabMixNN models, similar to lme4/sommer/lavaan summaries.
    """
    
    def __init__(self, model, data=None, sample_size: int = 256, linearize: bool = True):
        self.model = model
        self.formula = model.formula
        self.architecture = model.architecture
        self.n_obs = len(model._train_data) if hasattr(model, '_train_data') and model._train_data is not None else None
        
        estimates = parameter_estimates(model, data=data, sample_size=sample_size, linearize=linearize)
        self.fixed_effects = estimates.get("fixed_effects", pd.DataFrame())
        self.random_effects = estimates.get("random_effects", pd.DataFrame())
        self.variance_components = estimates.get("variance_components", pd.DataFrame())
        self.structural_edges = estimates.get("structural_edges", pd.DataFrame())
        self.structural_matrix = estimates.get("structural_matrix", None)
        self.notes = estimates.get("notes", [])

        self._extract_model_fit()
    def _extract_model_fit(self):
        """Extract model fit statistics including LL, AIC, BIC, R²."""
        self.fit_stats = {}
        
        # Training history
        if hasattr(self.model, 'history') and self.model.history:
            self.fit_stats['final_loss'] = self.model.history[-1]
            self.fit_stats['n_epochs'] = len(self.model.history)
        
        # Validation history
        if hasattr(self.model, 'val_history') and self.model.val_history:
            self.fit_stats['final_val_loss'] = self.model.val_history[-1]
        
        # Model complexity
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.fit_stats['n_parameters'] = n_params
        
        # Calculate log-likelihood, AIC, BIC, R² if training data available
        if hasattr(self.model, '_train_data') and self.model._train_data is not None:
            try:
                df = self.model._train_data
                n = len(df)
                
                # Get predictions
                with torch.no_grad():
                    y_pred = self.model.predict(df)
                    if isinstance(y_pred, torch.Tensor):
                        y_pred = y_pred.cpu().numpy()
                    y_pred = y_pred.flatten()
                
                # Get true values
                if hasattr(self.model, 'design') and self.model.design:
                    outcome_name = self.model.design.outcomes[0].name
                    y_true = df[outcome_name].values
                else:
                    # Try to infer outcome from formula
                    import re
                    match = re.match(r'(\w+)\s*~', self.model.formula)
                    if match:
                        outcome_name = match.group(1)
                        y_true = df[outcome_name].values
                    else:
                        y_true = None
                
                if y_true is not None:
                    # R² (coefficient of determination)
                    ss_res = np.sum((y_true - y_pred)**2)
                    ss_tot = np.sum((y_true - y_true.mean())**2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    self.fit_stats['R2'] = r2
                    
                    # RMSE
                    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
                    self.fit_stats['RMSE'] = rmse
                    
                    # Log-likelihood (approximate for Gaussian)
                    # LL = -n/2 * log(2π) - n/2 * log(σ²) - 1/(2σ²) * Σ(y - ŷ)²
                    sigma2 = ss_res / n  # MLE estimate of variance
                    if sigma2 > 0:
                        log_lik = -0.5 * n * np.log(2 * np.pi) - 0.5 * n * np.log(sigma2) - 0.5 * ss_res / sigma2
                        self.fit_stats['logLik'] = log_lik
                        
                        # AIC = -2*LL + 2*k
                        aic = -2 * log_lik + 2 * n_params
                        self.fit_stats['AIC'] = aic
                        
                        # BIC = -2*LL + k*log(n)
                        bic = -2 * log_lik + n_params * np.log(n)
                        self.fit_stats['BIC'] = bic
            except Exception as e:
                # If calculation fails, just skip these metrics
                pass
    
    def __str__(self):
        """Print summary in lme4-style format."""
        lines = []
        lines.append("=" * 80)
        lines.append("TabMixNN Model Summary")
        lines.append("=" * 80)
        
        # Formula
        lines.append(f"\nFormula: {self.formula}")
        
        # Architecture
        lines.append(f"Architecture: {self.architecture}")
        if hasattr(self.model, 'hidden_dims'):
            lines.append(f"Hidden dimensions: {self.model.hidden_dims}")
        if hasattr(self.model, 'embedding_dim'):
            lines.append(f"Embedding dimension: {self.model.embedding_dim}")
        
        # Sample size
        if self.n_obs:
            lines.append(f"Number of observations: {self.n_obs}")
        
        # Fixed Effects
        lines.append("\n" + "-" * 80)
        lines.append("Fixed Effects:")
        lines.append("-" * 80)
        if isinstance(self.fixed_effects, pd.DataFrame) and not self.fixed_effects.empty:
            lines.append(self.fixed_effects.to_string())
        else:
            lines.append("  (fixed effects unavailable)")
        
        # Random Effects
        lines.append("\n" + "-" * 80)
        lines.append("Random Effects:")
        lines.append("-" * 80)
        if isinstance(self.random_effects, pd.DataFrame) and not self.random_effects.empty:
            summary_df = (
                self.random_effects.groupby(["group", "slope"])["estimate"]
                .agg(["mean", "std"])
                .rename(columns={"mean": "BLUP mean", "std": "BLUP sd"})
            )
            lines.append(summary_df.to_string())
        else:
            lines.append("  (random effects unavailable)")
        
        # Variance Components
        lines.append("\n" + "-" * 80)
        lines.append("Variance Components:")
        lines.append("-" * 80)
        if isinstance(self.variance_components, pd.DataFrame) and not self.variance_components.empty:
            lines.append(self.variance_components.to_string(index=False))
        else:
            lines.append("  (variance components unavailable)")

        # Structural Edges (SEM/SAM)
        lines.append("\n" + "-" * 80)
        lines.append("Structural Edges (B matrix):")
        lines.append("-" * 80)
        if isinstance(self.structural_edges, pd.DataFrame) and not self.structural_edges.empty:
            lines.append(self.structural_edges.to_string(index=False))
        else:
            lines.append("  (no structural edges extracted)")
        
        # Model Fit
        if self.fit_stats:
            lines.append("\n" + "-" * 80)
            lines.append("Model Fit:")
            lines.append("-" * 80)
            # Order: R2, RMSE, logLik, AIC, BIC, then others
            ordered_keys = ['R2', 'RMSE', 'logLik', 'AIC', 'BIC']
            for key in ordered_keys:
                if key in self.fit_stats:
                    val = self.fit_stats[key]
                    if isinstance(val, float):
                        lines.append(f"  {key}: {val:.4f}")
                    else:
                        lines.append(f"  {key}: {val}")
            # Add remaining keys
            for key, val in self.fit_stats.items():
                if key not in ordered_keys:
                    if isinstance(val, float):
                        lines.append(f"  {key}: {val:.4f}")
                    else:
                        lines.append(f"  {key}: {val}")

        # Notes
        if self.notes:
            lines.append("\n" + "-" * 80)
            lines.append("Notes:")
            lines.append("-" * 80)
            for note in self.notes:
                lines.append(f"  - {note}")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def __repr__(self):
        return self.__str__()


def summary(model, data=None, sample_size: int = 256, linearize: bool = True):
    """
    Create a summary of a fitted TabMixNN model.
    
    Args:
        model: A fitted TabMixNN model
        data: Optional dataframe to use for parameter extraction / linearization
        sample_size: Subsample size for linearization of non-linear backbones
        linearize: Whether to linearize non-linear models for approximate coefficients
        
    Returns:
        TabMixNNSummary object
        
    Example:
        >>> model = TabMixNN(formula='y ~ x + (1|group)', family=Gaussian())
        >>> model.fit(df)
        >>> summary(model)
    """
    return TabMixNNSummary(model, data=data, sample_size=sample_size, linearize=linearize)


# Add summary method to TabMixNN class
def add_summary_method():
    """Add summary() method to TabMixNN class."""
    from tabmixnn import TabMixNN
    TabMixNN.summary = lambda self, **kwargs: summary(self, **kwargs)
