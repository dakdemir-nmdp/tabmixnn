import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
from .formula import CompiledDesign, OutcomeSpec
from .model import TabMixNNModule
from .families import Binomial, Gaussian
from .logging_config import get_logger
from .constants import (
    VALID_SAM_LOCATIONS, VALID_DEVICE_PREFIXES, VALID_ACTIVATIONS, 
    VALID_UNKNOWN_STRATEGIES, DEFAULT_EMBEDDING_DIM, DEFAULT_HIDDEN_DIMS,
    DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_WEIGHT_DECAY,
    DEFAULT_LAMBDA_SPARSE, DEFAULT_LAMBDA_KL, DEFAULT_SAM_MAX_OFFDIAG,
    DEFAULT_SAM_RESIDUAL_SCALE
)


# Increment if serialization schema changes in a backward-incompatible way.
SCHEMA_VERSION = 1

class TabMixNN:
    def __init__(self,
                 formula=None,
                 outcomes=None, # List of OutcomeSpec
                 family=None, # For single outcome short-hand
                 sam_location="output",
                 sam_dim=None,
                 hidden_dims=None,
                 embedding_dim=16,
                 lr=0.001,
                 weight_decay=1e-4,
                 batch_size=256,
                 epochs=20,
                 lambda_sparse=1e-3,
                 lambda_kl=1e-2,
                 cov_struct=None, # dict: group -> {type: "IID"|"AR1"|"K", ...}
                 scale_random_by_terms=True,
                 mixed_activation="relu",
                 hidden_dropout=0.0,
                 hidden_layer_norm=False,
                 hidden_residual=True,
                 sam_residual=True,
                 sam_residual_scale=1.0,
                 sam_max_offdiag=0.9,
                 unknown_strategy="learned",
                 device="cpu",
                 verbose=False,
                 grad_clip_norm=1.0):  # NEW: gradient clipping
        
        # Initialize logger first
        self.logger = get_logger("estimator")
        self.verbose = verbose
        
        # ========== INPUT VALIDATION ==========
        
        # Validate numeric hyperparameters
        if epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {epochs}")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if lambda_sparse < 0:
            raise ValueError(f"lambda_sparse must be >= 0, got {lambda_sparse}")
        if lambda_kl < 0:
            raise ValueError(f"lambda_kl must be >= 0, got {lambda_kl}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {embedding_dim}")
        if hidden_dropout < 0 or hidden_dropout >= 1:
            raise ValueError(f"hidden_dropout must be in [0, 1), got {hidden_dropout}")
        if sam_max_offdiag <= 0 or sam_max_offdiag >= 1:
            raise ValueError(f"sam_max_offdiag must be in (0, 1), got {sam_max_offdiag}")
        if sam_residual_scale <= 0:
            raise ValueError(f"sam_residual_scale must be > 0, got {sam_residual_scale}")
        if grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm must be > 0, got {grad_clip_norm}")
        
        # Validate categorical hyperparameters
        if sam_location not in VALID_SAM_LOCATIONS:
            raise ValueError(
                f"sam_location must be one of {VALID_SAM_LOCATIONS}, got '{sam_location}'"
            )
        
        if not any(device.startswith(prefix) for prefix in VALID_DEVICE_PREFIXES):
            raise ValueError(
                f"device must start with one of {VALID_DEVICE_PREFIXES}, got '{device}'"
            )
        
        if mixed_activation not in VALID_ACTIVATIONS:
            raise ValueError(
                f"mixed_activation must be one of {VALID_ACTIVATIONS}, got '{mixed_activation}'"
            )
        
        if unknown_strategy not in VALID_UNKNOWN_STRATEGIES:
            raise ValueError(
                f"unknown_strategy must be one of {VALID_UNKNOWN_STRATEGIES}, got '{unknown_strategy}'"
            )
        
        # Validate hidden_dims
        if hidden_dims is not None:
            if not isinstance(hidden_dims, list):
                raise ValueError(f"hidden_dims must be a list, got {type(hidden_dims).__name__}")
            if not all(isinstance(d, int) and d > 0 for d in hidden_dims):
                raise ValueError(f"All values in hidden_dims must be positive integers, got {hidden_dims}")
        
        # Validate sam_dim if provided
        if sam_dim is not None and sam_dim <= 0:
            raise ValueError(f"sam_dim must be > 0 if provided, got {sam_dim}")
        
        # Log initialization
        self.logger.info(
            f"Initializing TabMixNN: sam_location={sam_location}, "
            f"epochs={epochs}, lr={lr}, embedding_dim={embedding_dim}"
        )
        
        # ========== STORE PARAMETERS ==========
        
        self.formula = formula
        self.cov_struct = cov_struct
        if outcomes is None:
            if family is None:
                raise ValueError("Must provide either 'outcomes' or 'family'.")
            self.single_outcome_family = family
            self.outcomes = [] # Will populate in fit
            self.logger.debug(f"Single-outcome mode with family: {family.__class__.__name__}")
        else:
            self.outcomes = outcomes
            self.logger.debug(f"Multi-outcome mode with {len(outcomes)} outcomes")

        self.sam_location = sam_location
        self.sam_dim = sam_dim
        self.hidden_dims = DEFAULT_HIDDEN_DIMS if hidden_dims is None else hidden_dims
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.lambda_sparse = lambda_sparse
        self.lambda_kl = lambda_kl
        self.scale_random_by_terms = scale_random_by_terms
        self.mixed_activation = mixed_activation
        self.hidden_dropout = hidden_dropout
        self.hidden_layer_norm = hidden_layer_norm
        self.hidden_residual = hidden_residual
        self.sam_residual = sam_residual
        self.sam_residual_scale = sam_residual_scale
        self.sam_max_offdiag = sam_max_offdiag
        self.unknown_strategy = unknown_strategy
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        
        self.design = None
        self.model = None
        self.history = []
        self.val_history = []

    def fit(self, df, y=None, state_dict=None, reuse_design=None,
            val_df=None, val_fraction=0.0, early_stopping=False,
            patience=5, min_delta=0.0):
        """Fit the TabMixNN model to training data."""
        
        # Reset histories
        self.history = []
        self.val_history = []
        
        # Log fit start
        self.logger.info(f"Starting fit with {len(df)} samples, val_fraction={val_fraction}")
        
        # Cache training data for inference methods
        train_df = df
        if val_df is None and val_fraction > 0.0:
            if val_fraction <= 0 or val_fraction >= 1:
                raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
            val_df = df.sample(frac=val_fraction, random_state=42)
            train_df = df.drop(val_df.index).reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
            self.logger.info(f"Split data: {len(train_df)} train, {len(val_df)} validation")
        self._train_data = train_df.copy()

        # 1. Prepare Outcome Specs if single
        if not self.outcomes:
            # Parse target from formula LHS
            if '~' in self.formula:
                lhs = self.formula.split('~')[0].strip()
                if not lhs:
                    raise ValueError("Formula missing LHS for single outcome mode.")
                # Basic spec
                self.outcomes = [OutcomeSpec(name="main", y=lhs, family=self.single_outcome_family)]
            else:
                # Assuming simple implicit specifiction not fully supported without LHS
                raise ValueError("Formula must include LHS 'y ~ ...' for single outcome shortcut.")
        
        self.logger.debug(f"Formula: {self.formula}")
        self.logger.debug(f"Outcomes: {[o.name for o in self.outcomes]}")
                
        # 2. Compile Design
        if reuse_design is not None:
            self.design = reuse_design
            self.logger.info("Reusing provided CompiledDesign")
        else:
            self.logger.info("Compiling design from formula...")
            self.design = CompiledDesign(
                formula=self.formula,
                outcomes=self.outcomes,
                sam_location=self.sam_location,
                sam_dim=self.sam_dim,  # Matches layer width if None
                cov_struct_map=self.cov_struct
            )
            self.design.fit(train_df)
            self.logger.info(
                f"Design compiled: {len(self.design.fixed_cols)} fixed effects, "
                f"{len(self.design.random_terms)} random terms"
            )
        
        # 3. Transform Data (OPTIMIZED: Cache tensors)
        X, Z_ind, Z_slopes, Ys = self.design.transform(train_df, mode='fit')
        self.logger.info(f"Transformed data: X shape {X.shape}, targets: {list(Ys.keys())}")
        
        if self.verbose:
            print(f"X shape: {X.shape}")
            print(f"Targets: {list(Ys.keys())}")
        
        # Convert to Tensors (cache these!)
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        Z_ind_t = [torch.tensor(z, dtype=torch.long).to(self.device) for z in Z_ind]
        Z_slopes_t = [torch.tensor(z, dtype=torch.float32).to(self.device) for z in Z_slopes]
        
        Ys_t = {}
        for k, v in Ys.items():
            Ys_t[k] = torch.tensor(v, dtype=torch.float32).to(self.device)
        if not Ys_t:
            raise ValueError("No targets found in training data for the specified outcomes.")
        
        self.logger.debug(f"Tensors created on device: {self.device}")
        
        # 4. Initialize Model
        input_dim = X.shape[1]
        
        self.logger.info(f"Initializing TabMixNNModule with input_dim={input_dim}...")
        self.model = TabMixNNModule(
            fixed_input_dim=input_dim,
            compiled_design=self.design,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            sam_location=self.sam_location,
            sam_dim=self.sam_dim,  # Matches layer width if None
            cov_struct_map=self.cov_struct,
            mixed_activation=self.mixed_activation,
            kl_normalize=True,  # Hardcoded: always normalize KL by # params
            scale_random_by_terms=self.scale_random_by_terms,
            unknown_strategy=self.unknown_strategy,
            sam_residual=self.sam_residual,
            sam_residual_scale=self.sam_residual_scale,
            sam_max_offdiag=self.sam_max_offdiag,
            hidden_dropout=self.hidden_dropout,
            hidden_layer_norm=self.hidden_layer_norm,
            hidden_residual=self.hidden_residual
        ).to(self.device)
        
        if state_dict is not None:
            self.model.load_state_dict(state_dict, strict=False)
            self.logger.info("Loaded state_dict for warm start")
        
        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model initialized with {n_params:,} trainable parameters")
        
        # 5. Training Loop
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Validation prep
        val_tensors = None
        if val_df is not None:
            Xv, Zv_ind, Zv_slopes, Yv = self.design.transform(val_df, mode='fit')
            if not Yv:
                raise ValueError("Validation data must include targets for early stopping.")
            Xv_t = torch.tensor(Xv, dtype=torch.float32).to(self.device)
            Zv_ind_t = [torch.tensor(z, dtype=torch.long).to(self.device) for z in Zv_ind]
            Zv_slopes_t = [torch.tensor(z, dtype=torch.float32).to(self.device) for z in Zv_slopes]
            Yv_t = {k: torch.tensor(v, dtype=torch.float32).to(self.device) for k, v in Yv.items()}
            val_tensors = (Xv_t, Zv_ind_t, Zv_slopes_t, Yv_t)
            self.logger.debug(f"Validation tensors prepared: {Xv_t.shape[0]} samples")

        # Batching
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))
        kl_weight = self.lambda_kl
        
        best_val = float("inf")
        best_state = None
        patience_ctr = 0
        
        self.logger.info(
            f"Starting training: {self.epochs} epochs, {n_batches} batches/epoch, "
            f"batch_size={self.batch_size}"
        )
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0
            perm = torch.randperm(n_samples)
            
            pbar_iter = range(n_batches)
            if self.verbose:
                pbar_iter = tqdm(pbar_iter, desc=f"Epoch {epoch}/{self.epochs}")
                
            for i in pbar_iter:
                idx = perm[i*self.batch_size : (i+1)*self.batch_size]
                
                # OPTIMIZED: Use advanced indexing instead of index_select
                batch_X = X_t[idx]
                batch_Z_ind = [z[idx] for z in Z_ind_t]
                batch_Z_slopes = [z[idx] for z in Z_slopes_t]
                batch_Ys = {k: v[idx] for k, v in Ys_t.items()}
                
                optimizer.zero_grad()
                outputs, kl = self.model(batch_X, batch_Z_ind, batch_Z_slopes) 
                batch_frac = float(batch_X.shape[0]) / float(n_samples)
                
                # NLL Loss across outcomes
                nll_total = 0.0
                nll_count = 0
                for out_spec in self.outcomes:
                    name = out_spec.name
                    if name not in batch_Ys:
                        continue  # allow partially labeled batches
                    target = batch_Ys[name]
                    raw_out = outputs[name]
                    # Build mask for rows with observed targets
                    if target.ndim == 1:
                        mask = ~torch.isnan(target)
                    else:
                        mask = ~torch.isnan(target).any(dim=1)
                    if mask.sum() == 0:
                        continue
                    family = self.model.families[name]
                    theta = family.forward_params(raw_out[mask])
                    nll_total = nll_total + family.nll(theta, target[mask])
                    nll_count += 1
                if nll_count == 0:
                    raise ValueError("No outcomes contributed to the loss; check targets, masks, and outcomes configuration.")
                nll_total = nll_total / nll_count

                # KL already normalized per-parameter in the module; scale by weight only
                kl_scaled = kl * kl_weight * batch_frac
                sparse_loss = self.model.get_sparsity_loss(self._lambda_sparse_weights())
                
                loss = nll_total + kl_scaled + sparse_loss
                loss.backward()
                
                # NEW: Gradient clipping for numerical stability
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                
            train_epoch_loss = epoch_loss / n_batches
            self.history.append(train_epoch_loss)
            
            val_epoch_loss = None
            if val_tensors is not None:
                self.model.eval()
                with torch.no_grad():
                    Xv_t, Zv_ind_t, Zv_slopes_t, Yv_t = val_tensors
                    outputs_v, kl_v = self.model(Xv_t, Zv_ind_t, Zv_slopes_t)
                    val_batch_frac = float(Xv_t.shape[0]) / float(n_samples)
                    nll_total_v = 0.0
                    nll_count_v = 0
                    for out_spec in self.outcomes:
                        name = out_spec.name
                        if name not in Yv_t:
                            continue
                        target_v = Yv_t[name]
                        raw_out_v = outputs_v[name]
                        if target_v.ndim == 1:
                            mask_v = ~torch.isnan(target_v)
                        else:
                            mask_v = ~torch.isnan(target_v).any(dim=1)
                        if mask_v.sum() == 0:
                            continue
                        family = self.model.families[name]
                        theta_v = family.forward_params(raw_out_v[mask_v])
                        nll_total_v = nll_total_v + family.nll(theta_v, target_v[mask_v])
                        nll_count_v += 1
                    nll_total_v = nll_total_v / max(nll_count_v, 1)
                    kl_scaled_v = kl_v * kl_weight * val_batch_frac
                    sparse_loss_v = self.model.get_sparsity_loss(self._lambda_sparse_weights())
                    val_epoch_loss = (nll_total_v + kl_scaled_v + sparse_loss_v).item()
                    self.val_history.append(val_epoch_loss)

                if early_stopping:
                    if val_epoch_loss + min_delta < best_val:
                        best_val = val_epoch_loss
                        best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                        patience_ctr = 0
                        self.logger.debug(f"Epoch {epoch}: New best validation loss: {best_val:.4f}")
                    else:
                        patience_ctr += 1
                        if patience_ctr >= patience:
                            self.logger.info(
                                f"Early stopping at epoch {epoch} (patience={patience}, "
                                f"best_val={best_val:.4f})"
                            )
                            if best_state is not None:
                                self.model.load_state_dict(best_state)
                            break
            
            # Log progress periodically
            if epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs:
                msg = f"Epoch {epoch}/{self.epochs}: train_loss={train_epoch_loss:.4f}"
                if val_epoch_loss is not None:
                    msg += f", val_loss={val_epoch_loss:.4f}"
                self.logger.info(msg)
            
            if self.verbose and val_epoch_loss is not None:
                print(f"Epoch {epoch}: Train={train_epoch_loss:.4f}, Val={val_epoch_loss:.4f}")

        if best_state is not None and not (patience_ctr >= patience):
            # Load best state if we didn't early stop
            self.model.load_state_dict(best_state)
            self.logger.info(f"Loaded best model state (val_loss={best_val:.4f})")
        
        self.logger.info(f"Training completed: {len(self.history)} epochs, final_loss={self.history[-1]:.4f}")
        return self

    def predict(self, df, outcome=None, mode='default'):
        self.model.eval()
        X, Z_ind, Z_slopes, _ = self.design.transform(df, mode='predict')
        
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        Z_ind_t = [torch.tensor(z, dtype=torch.long).to(self.device) for z in Z_ind]
        Z_slopes_t = [torch.tensor(z, dtype=torch.float32).to(self.device) for z in Z_slopes]
        
        with torch.no_grad():
            outputs, _ = self.model(X_t, Z_ind_t, Z_slopes_t)
            
        # If outcome specified, return that
        if outcome:
            out_spec = next((o for o in self.outcomes if o.name == outcome), None)
            if not out_spec:
                raise ValueError(f"Outcome {outcome} not found.")
            # Use family from model
            family = self.model.families[outcome]
            theta = family.forward_params(outputs[outcome])
            return family.predict(theta, mode=mode)

        # Return dict of all
        results = {}
        for spec in self.outcomes:
            # Use family from model
            family = self.model.families[spec.name]
            theta = family.forward_params(outputs[spec.name])
            results[spec.name] = family.predict(theta, mode=mode)
            
        # If single outcome mode and no outcome requested, return main
        if len(self.outcomes) == 1 and outcome is None:
             return results[self.outcomes[0].name]
             
        return results

    def predict_proba(self, df, outcome=None):
        # Only valid for classification
        return self.predict(df, outcome=outcome, mode='proba')

    # --- sklearn-style helpers ---

    def get_params(self, deep=True):
        """
        Return constructor parameters for compatibility with sklearn-style APIs.
        """
        return {
            "formula": self.formula,
            "outcomes": self.outcomes,
            # For single-outcome shortcut we still persist the family for compatibility.
            "family": getattr(self, "single_outcome_family", None),
            "sam_location": self.sam_location,
            "sam_dim": self.sam_dim,
            "hidden_dims": self.hidden_dims,
            "embedding_dim": self.embedding_dim,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lambda_sparse": self.lambda_sparse,
            "lambda_kl": self.lambda_kl,
            "scale_random_by_terms": self.scale_random_by_terms,
            "mixed_activation": self.mixed_activation,
            "hidden_dropout": self.hidden_dropout,
            "hidden_layer_norm": self.hidden_layer_norm,
            "hidden_residual": self.hidden_residual,
            "sam_residual": self.sam_residual,
            "sam_residual_scale": self.sam_residual_scale,
            "sam_max_offdiag": self.sam_max_offdiag,
            "unknown_strategy": self.unknown_strategy,
            "cov_struct": self.cov_struct,
            "device": self.device,
            "verbose": self.verbose,
        }

    def set_params(self, **params):
        """
        Set constructor parameters; does not re-fit the model.
        """
        for k, v in params.items():
            if not hasattr(self, k) and k != "family":
                raise ValueError(f"Unknown parameter '{k}'")
            if k == "family":
                # Preserve single-outcome family shortcut
                self.single_outcome_family = v
            else:
                setattr(self, k, v)
        return self

    def score(self, df, outcome=None):
        """
        Compute a simple default score on the provided DataFrame.
        - Binomial: ROC-AUC
        - Gaussian/Poisson/NegBinomial: R^2
        """
        family_name = self.outcomes[0].family.__class__.__name__ if self.outcomes else None
        # For classification we need probabilities, not thresholded labels.
        if family_name == "Binomial":
            preds = self.predict(df, outcome=outcome, mode="proba")
        else:
            preds = self.predict(df, outcome=outcome)
        y_true = df[self._get_outcome_col(outcome)].values

        if family_name == "Binomial":
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(y_true, preds))
        else:
            from sklearn.metrics import r2_score
            return float(r2_score(y_true, preds))

    # --- Introspection ---
    
    def summary(self):
        """
        Prints a summary of the fitted model.
        """
        if not self.model:
            print("Model is not fitted.")
            return

        print("="*60)
        print("TabMixNN Summary")
        print("="*60)
        print(f"Formula: {self.formula}")
        print(f"Outcomes: {[o.name for o in self.outcomes]}")
        print(f"SAM Location: {self.sam_location} (sparse_lambda={self.lambda_sparse})")
        
        # Variance Components
        print("\nRandom Effects Variance Components:")
        var_summary = self.model.get_variance_summary()
        # Map term_id back to name?
        # compiled_design.random_terms has mapping
        term_map = {t.term_id: t for t in self.design.random_terms}
        
        for tid, stats in var_summary.items():
            if tid in term_map:
                t = term_map[tid]
                print(f"  Group: {t.group_col}")
                print(f"    Type: {stats.get('type', 'Unknown')}")
                if 'sigmas' in stats:
                    # Match sigmas to slope names
                    for i, sval in enumerate(stats['sigmas']):
                        sname = t.slope_cols[i] if i < len(t.slope_cols) else str(i)
                        print(f"    Sigma[{sname}]: {sval:.4f}")
                if 'rho' in stats:
                    for i, rval in enumerate(stats['rho']):
                        sname = t.slope_cols[i] if i < len(t.slope_cols) else str(i)
                        print(f"    Rho[{sname}]: {rval:.4f} (AR1)")
        
        print("\nSAM Structure:")
        # Check edges
        locs = ["first", "output"] + [f"hidden_{i}" for i in range(len(self.hidden_dims))]
        for loc in locs:
            sam = self.model.get_sam_layer(loc)
            if sam is not None:
                # Expected edges
                probs = torch.sigmoid(sam.alpha_logits) * sam.mask
                probs_np = probs.detach().cpu().numpy()
                active_mask = probs_np > 0.5
                n_edges = active_mask.sum()
                total = int(sam.mask.sum().item())
                mean_prob = probs_np[sam.mask.bool().cpu().numpy()].mean() if total > 0 else 0.0
                print(f"  Layer '{loc}': {n_edges}/{total} active edges (p>0.5). Mean prob: {mean_prob:.4f}")

    def extract_sam(self, layer="output"):
        """
        Returns (weights_B, gate_probs) for the specified SAM layer.
        """
        if not self.model:
            raise RuntimeError("Not fitted")

        sam = self.model.get_sam_layer(layer)
        if sam is None:
            raise ValueError(f"No SAM layer found at '{layer}'")

        B = sam.get_B().detach().cpu().numpy()
        probs = torch.sigmoid(sam.alpha_logits).detach().cpu().numpy()
        return B, probs

    def prune_sam(self, layer="output", threshold=0.5):
        """
        Returns a binary adjacency matrix for the specified SAM layer, thresholding gate probabilities.
        """
        _, probs = self.extract_sam(layer)
        return (probs > threshold).astype(int)

    # --- Helper Methods ---

    def _clone(self):
        """Create new instance with same hyperparameters."""
        return self.__class__(
            formula=self.formula,
            outcomes=self.outcomes,
            sam_location=self.sam_location,
            sam_dim=self.sam_dim,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            lr=self.lr,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            epochs=self.epochs,
            lambda_sparse=self.lambda_sparse,
            lambda_kl=self.lambda_kl,
            scale_random_by_terms=self.scale_random_by_terms,
            mixed_activation=self.mixed_activation,
            hidden_dropout=self.hidden_dropout,
            hidden_layer_norm=self.hidden_layer_norm,
            hidden_residual=self.hidden_residual,
            sam_residual=self.sam_residual,
            sam_residual_scale=self.sam_residual_scale,
            sam_max_offdiag=self.sam_max_offdiag,
            unknown_strategy=self.unknown_strategy,
            cov_struct=self.cov_struct,
            device=self.device,
            verbose=self.verbose
        )

    def _get_outcome_col(self, outcome):
        """Get target column name for outcome."""
        if outcome is None:
            outcome = self.outcomes[0].name

        spec = next((o for o in self.outcomes if o.name == outcome), None)
        if spec is None:
            raise ValueError(f"Outcome '{outcome}' not found")

        return spec.y if hasattr(spec, 'y') else None

    def _get_feature_names(self):
        """Get feature names for fixed effects (for SHAP interpretability)."""
        names = []

        if self.design.intercept_included:
            names.append("Intercept")

        for spec in self.design.fixed_cols:
            col_name = spec['name']
            if spec['type'] == 'numeric':
                names.append(col_name)
            else:
                # Categorical - get all levels
                levels = self.design.categorical_dummy_levels.get(col_name, self.design.categorical_levels[col_name])
                for level in levels:
                    names.append(f"{col_name}_{level}")

        return names

    def _lambda_sparse_weights(self):
        """
        Resolve lambda_sparse to a uniform dict.
        lambda_sparse is now always a single float applied to all SAM layers.
        """
        val = float(self.lambda_sparse)
        return {"first": val, "hidden": val, "output": val}

    def _require_df(self, df, purpose="this method"):
        """
        Ensure a dataframe is available for inference methods that default to training data.
        When models are loaded from disk, _train_data may be absent; in that case the caller
        must supply df explicitly.
        """
        if df is not None:
            return df
        if getattr(self, "_train_data", None) is not None:
            return self._train_data
        raise ValueError(f"{purpose} requires 'df' because cached training data is unavailable.")

    def _evaluate_performance(self, df, outcome):
        """Evaluate model performance (for ablation studies)."""
        spec = next((o for o in self.outcomes if o.name == (outcome or self.outcomes[0].name)), None)
        if spec is None:
            raise ValueError(f"Outcome '{outcome}' not found")
        family_name = spec.family.__class__.__name__

        # Use probabilities for classification metrics to avoid AUC on hard labels.
        mode = "proba" if family_name == "Binomial" else "default"
        y_col = self._get_outcome_col(spec.name)
        if y_col is None:
            raise NotImplementedError(f"Metric not defined for outcome '{spec.name}' (non-scalar target).")

        preds = self.predict(df, outcome=spec.name, mode=mode)
        y_true = df[y_col].values
        mask = ~np.isnan(y_true)
        if mask.sum() == 0:
            raise ValueError(f"No observed targets for outcome '{spec.name}' to evaluate performance.")
        preds = preds[mask]
        y_true = y_true[mask]

        if family_name == 'Binomial':
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, preds)
        elif family_name in ['Gaussian', 'Poisson', 'NegBinomial']:
            from sklearn.metrics import r2_score
            return r2_score(y_true, preds)
        else:
            raise NotImplementedError(f"Metric not defined for {family_name}")

    def _predict_ablate_random(self, df, outcome, mode="default"):
        """Predict with all random effects set to zero."""
        original_params = {}
        with torch.no_grad():
            for tid, params in self.model.mixed_block.variational_params.items():
                mu = params[0]
                original_params[tid] = mu.detach().clone()
                mu.zero_()
        try:
            return self.predict(df, outcome=outcome, mode=mode)
        finally:
            with torch.no_grad():
                for tid, mu_orig in original_params.items():
                    self.model.mixed_block.variational_params[tid][0].copy_(mu_orig)

    def _predict_with_term(self, df, outcome, term_id, mode="default"):
        """Predict with only specified term active."""
        original_params = {}
        with torch.no_grad():
            for tid, params in self.model.mixed_block.variational_params.items():
                mu = params[0]
                original_params[tid] = mu.detach().clone()
                if tid != term_id:
                    mu.zero_()
        try:
            return self.predict(df, outcome=outcome, mode=mode)
        finally:
            with torch.no_grad():
                for tid, mu_orig in original_params.items():
                    self.model.mixed_block.variational_params[tid][0].copy_(mu_orig)

    def _predict_without_term(self, df, outcome, term_id, mode="default"):
        """Predict with specified term zeroed out."""
        original_params = {}
        with torch.no_grad():
            for tid, params in self.model.mixed_block.variational_params.items():
                if tid == term_id:
                    mu = params[0]
                    original_params[tid] = mu.detach().clone()
                    mu.zero_()
        try:
            return self.predict(df, outcome=outcome, mode=mode)
        finally:
            with torch.no_grad():
                for tid, mu_orig in original_params.items():
                    self.model.mixed_block.variational_params[tid][0].copy_(mu_orig)

    # --- Inference Methods ---

    def variance_decomposition(self, outcome=None, df=None):
        """Partition variance into fixed, random, and residual."""

        df = self._require_df(df, "variance_decomposition")

        spec = next((o for o in self.outcomes if o.name == (outcome or self.outcomes[0].name)), None)
        if spec is None:
            raise ValueError(f"Outcome '{outcome}' not found.")
        family_name = spec.family.__class__.__name__
        mode = "proba" if family_name == "Binomial" else "default"
        y_col = self._get_outcome_col(spec.name)
        if y_col is None:
            raise NotImplementedError("Variance decomposition currently supports scalar outcomes only.")

        # Get predictions with all components
        y_full = self.predict(df, outcome=spec.name, mode=mode)
        y_true = df[y_col].values
        mask = ~np.isnan(y_true)
        if mask.sum() == 0:
            raise ValueError("No observed targets available for variance decomposition.")
        y_true = y_true[mask]
        y_full = y_full[mask]

        eps = 1e-8

        # 1. Total variance
        var_total = np.var(y_true) + eps

        # 2. Residual variance (irreducible error)
        residuals = y_true - y_full
        var_residual = np.var(residuals)

        # 3. Fixed effects variance
        # Predict with random effects zeroed out
        y_fixed_only = self._predict_ablate_random(df, outcome=spec.name, mode=mode)[mask]
        var_fixed = np.var(y_fixed_only)

        # 4. Random effects variance per term
        var_random = {}
        icc = {}
        variance_explained = {}

        for term in self.design.random_terms:
            tid = term.term_id

            # Predict with only this random term active
            y_with_term = self._predict_with_term(df, spec.name, term_id=tid, mode=mode)[mask]
            y_without_term = self._predict_without_term(df, spec.name, term_id=tid, mode=mode)[mask]

            # Variance contribution of this term
            contrib = y_with_term - y_without_term
            var_random[tid] = np.var(contrib)

            # ICC: σ²_u / (σ²_u + σ²_e)
            sigma2_u = var_random[tid]
            sigma2_e = var_residual
            icc_raw = sigma2_u / max(sigma2_u + sigma2_e, eps)
            icc[tid] = float(np.clip(icc_raw, 0.0, 1.0))

            # Proportion of total variance
            variance_explained[tid] = var_random[tid] / var_total

        return {
            'total_var': var_total,
            'fixed_var': var_fixed,
            'random_var': var_random,
            'residual_var': var_residual,
            'icc': icc,
            'variance_explained': variance_explained
        }

    def sam_edge_stability(self, layer='output', n_boot=500, threshold=0.5,
                           method='case', seed=42, warm_start=True):
        """Bootstrap edge stability analysis."""

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Get current SAM layer
        sam = self.model.get_sam_layer(layer)
        if sam is None:
            raise ValueError(f"No SAM layer at '{layer}'")

        d = sam.dim
        edge_count = np.zeros((d, d))
        edge_weights = []

        # Store original training data
        train_df = self._require_df(getattr(self, "_train_data", None), "sam_edge_stability")
        n_samples = len(train_df)

        for b in tqdm(range(n_boot), desc="Bootstrap samples"):
            # Case bootstrap: resample rows with replacement
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_df = train_df.iloc[boot_idx].reset_index(drop=True)

            # Clone model architecture (reuse hyperparameters)
            boot_model = self._clone()

            boot_model.verbose = False
            if warm_start:
                boot_model.epochs = max(self.epochs // 4, 5)
                boot_model.fit(
                    boot_df,
                    state_dict=self.model.state_dict(),
                    reuse_design=self.design,
                )
            else:
                boot_model.epochs = max(self.epochs // 2, 10)
                boot_model.fit(boot_df)

            # Extract SAM adjacency
            B, probs = boot_model.extract_sam(layer)
            adj = (probs > threshold).astype(int)

            edge_count += adj
            edge_weights.append(B)

        # Aggregate results
        edge_prob = edge_count / n_boot
        edge_mean = np.mean(edge_weights, axis=0)
        edge_std = np.std(edge_weights, axis=0)

        # Optional: compare to ground truth if available
        precision, recall = None, None
        if hasattr(self, '_ground_truth_dag'):
            true_adj = self._ground_truth_dag
            pred_adj = (edge_prob > 0.5).astype(int)

            tp = ((pred_adj == 1) & (true_adj == 1)).sum()
            fp = ((pred_adj == 1) & (true_adj == 0)).sum()
            fn = ((pred_adj == 0) & (true_adj == 1)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return {
            'edge_prob': edge_prob,
            'edge_mean': edge_mean,
            'edge_std': edge_std,
            'precision': precision,
            'recall': recall,
            'n_boot': n_boot
        }

    def shapley_values(self, df, outcome=None, method='gradient',
                       background_size=100, seed=42):
        """Compute SHAP values for feature importance."""

        df = self._require_df(df, "shapley_values")

        if method == 'gradient':
            return self._shap_gradient(df, outcome, background_size, seed)
        elif method == 'kernel':
            raise NotImplementedError("KernelSHAP not yet implemented")
        else:
            raise ValueError(f"Unknown method: {method}")

    def _shap_gradient(self, df, outcome, background_size, seed):
        """GradientSHAP: fast approximation for neural networks."""

        self.model.eval()

        # Prefer cached training data for the background; otherwise use the provided df.
        background_df = getattr(self, "_train_data", None)
        if background_df is None:
            background_df = df
        background_df = self._require_df(background_df, "shapley_values (background)")

        # Sample background rows
        np.random.seed(seed)
        bg_idx = np.random.choice(
            len(background_df),
            size=min(background_size, len(background_df)),
            replace=False,
        )
        bg_df = background_df.iloc[bg_idx]

        # Get fixed design matrix (exclude random effects for SHAP)
        X_explain = self.design._transform_fixed(df)  # [n, p]
        X_bg = self.design._transform_fixed(bg_df)    # [bg_size, p]

        n_samples, n_features = X_explain.shape

        # Baseline prediction (expected value over background) using the same fixed-only path.
        X_bg_t = torch.tensor(X_bg, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            base_value = self._forward_fixed_only(X_bg_t, outcome).mean().item()

        # Compute SHAP values: φ_j ≈ (x_j - x̄_j) × E[∂f/∂x_j]
        shap_values = np.zeros((n_samples, n_features))

        X_explain_t = torch.tensor(X_explain, dtype=torch.float32).to(self.device)
        bg_mean = X_bg.mean(axis=0)
        X_bg_t = torch.tensor(X_bg, dtype=torch.float32).to(self.device)

        for i in range(n_samples):
            x_i = X_explain_t[i]  # [p]

            # Interpolate every background row with the current sample (midpoint alpha=0.5)
            x_interp = 0.5 * x_i.unsqueeze(0).expand_as(X_bg_t) + 0.5 * X_bg_t
            x_interp.requires_grad_(True)

            # Forward for all interpolated rows at once
            y_pred = self._forward_fixed_only(x_interp, outcome)

            # Gradient of summed outputs w.r.t. inputs, then average
            grads = torch.autograd.grad(
                outputs=y_pred.sum(),
                inputs=x_interp,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
            grad_mean = grads.mean(dim=0)

            # SHAP value: difference × expected gradient
            x_diff = X_explain[i] - bg_mean
            shap_values[i] = x_diff * grad_mean.detach().cpu().numpy()

        # Get feature names
        feature_names = self._get_feature_names()

        with torch.no_grad():
            preds_tensor = self._forward_fixed_only(
                torch.tensor(X_explain, dtype=torch.float32).to(self.device),
                outcome
            )
            shap_predictions = preds_tensor.detach().cpu().numpy()

        return {
            'shap_values': shap_values,
            'base_value': base_value,
            'feature_names': feature_names,
            'predictions': shap_predictions
        }

    def _forward_fixed_only(self, X_tensor, outcome):
        """Forward pass through fixed effects only (for SHAP)."""
        # Mirror the main forward pass but without random effects.
        h = self.model.mixed_block.fixed_linear(X_tensor)
        if self.model.mixed_activation is not None:
            h = self.model.mixed_activation(h)

        # SAM first block (respect optional projection)
        if getattr(self.model, "first_proj", None) is not None:
            h = self.model.first_proj(h)
        if self.model.sam_first:
            h = self.model.sam_first(h)

        # Hidden layers (Linear and possible SAMLayer entries)
        for layer in self.model.hidden_layers:
            if isinstance(layer, nn.Linear):
                h = F.relu(layer(h))
            else:
                h = layer(h)

        # Output projection + SAM
        if getattr(self.model, "output_proj", None) is not None:
            h = self.model.output_proj(h)
        if self.model.sam_output:
            h = self.model.sam_output(h)

        # Resolve outcome/family
        if outcome is None:
            outcome = self.outcomes[0].name
        out_spec = next((o for o in self.outcomes if o.name == outcome), None)
        if out_spec is None:
            raise ValueError(f"Outcome '{outcome}' not found")

        logits = self.model.heads[outcome](h)
        # Use family from model
        family = self.model.families[outcome]
        theta = family.forward_params(logits)

        # Return differentiable scalar prediction depending on family
        fam_name = family.__class__.__name__
        if fam_name == 'Binomial':
            return torch.sigmoid(theta).squeeze()
        if fam_name == 'Gaussian':
            # theta is just mu now (no longer tuple)
            return theta.squeeze()
        if fam_name == 'Poisson':
            return torch.exp(theta).squeeze()
        if fam_name == 'NegBinomial':
            log_mu = theta[0] if isinstance(theta, (tuple, list)) else theta[:, 0]
            return torch.exp(log_mu).squeeze()
        raise NotImplementedError(f"SHAP fixed-only path not implemented for family '{fam_name}'")

    def plot_shap_summary(self, df, outcome=None, max_features=20):
        """Beeswarm plot of SHAP values."""
        import matplotlib.pyplot as plt

        shap_result = self.shapley_values(df, outcome=outcome)
        shap_vals = shap_result['shap_values']
        feature_names = shap_result['feature_names']

        # Get feature values for coloring
        X = self.design._transform_fixed(df)

        # Sort features by mean |SHAP|
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[::-1][:max_features]
        n_top = len(top_idx)

        fig, ax = plt.subplots(figsize=(10, max(n_top, 1) * 0.4))

        for i, feat_idx in enumerate(top_idx):
            y_pos = n_top - i - 1
            shap_feat = shap_vals[:, feat_idx]
            x_feat = X[:, feat_idx]

            # Scatter with color = feature value
            scatter = ax.scatter(shap_feat, np.full_like(shap_feat, y_pos),
                                 c=x_feat, cmap='coolwarm', alpha=0.6, s=20)

        ax.set_yticks(range(n_top))
        ax.set_yticklabels([feature_names[i] for i in top_idx])
        ax.set_xlabel('SHAP value (impact on prediction)')
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_title(f'SHAP Feature Importance (outcome: {outcome})')

        plt.colorbar(scatter, label='Feature value')
        plt.tight_layout()

        return fig

    def shapley_random_effects(self, df, outcome=None, term_id=None):
        """Decompose predictions via Shapley values for random effects."""

        df = self._require_df(df, "shapley_random_effects")
        n_samples = len(df)

        spec = next((o for o in self.outcomes if o.name == (outcome or self.outcomes[0].name)), None)
        if spec is None:
            raise ValueError(f"Outcome '{outcome}' not found.")
        family_name = spec.family.__class__.__name__
        mode = "proba" if family_name == "Binomial" else "default"

        y_col = self._get_outcome_col(spec.name)
        if y_col is None:
            raise NotImplementedError("Shapley random effects currently supports scalar outcomes only.")

        # Baseline: prediction with all random effects at 0
        y_baseline = self._predict_ablate_random(df, spec.name, mode=mode)

        # Full prediction
        y_full = self.predict(df, outcome=spec.name, mode=mode)

        # Get Z_indices to map samples to groups
        _, Z_ind, _, _ = self.design.transform(df, mode='predict')

        results = []

        # Filter terms if specific term requested
        terms = [t for t in self.design.random_terms
                 if term_id is None or t.term_id == term_id]

        for term in terms:
            tid = term.term_id
            term_idx = [t.term_id for t in self.design.random_terms].index(tid)

            # Prediction with only this term
            y_with_term = self._predict_with_term(df, spec.name, tid, mode=mode)

            # Shapley value for this term
            shap_term = y_with_term - y_baseline

            # Map to group IDs
            indices = Z_ind[term_idx]

            # Reverse mapping: index → group_id
            idx_to_group = {idx: gid for gid, idx in term.level_map.items()}

            for i in range(n_samples):
                group_idx = indices[i]
                group_id = idx_to_group.get(group_idx, 'UNK')

                results.append({
                    'sample_id': i,
                    'term_id': tid,
                    'group_col': term.group_col,
                    'group_id': group_id,
                    'shap_value': shap_term[i],
                    'baseline': y_baseline[i],
                    'prediction': y_full[i]
                })

        # Verify additivity: sum of SHAP across terms ≈ prediction - baseline
        df_results = pd.DataFrame(results)

        if self.verbose:
            for sample_id in range(min(5, n_samples)):
                sample_shap_sum = df_results[df_results['sample_id'] == sample_id]['shap_value'].sum()
                expected = y_full[sample_id] - y_baseline[sample_id]
                print(f"Sample {sample_id}: SHAP sum = {sample_shap_sum:.4f}, "
                      f"Expected = {expected:.4f}, "
                      f"Error = {abs(sample_shap_sum - expected):.6f}")

        return df_results

    def shapley_sam_edges(self, layer='output', df=None, outcome=None,
                          n_samples=1000, seed=42):
        """Edge importance via Shapley values (ablation-based)."""

        df_base = self._require_df(df, "shapley_sam_edges")
        df = df_base.iloc[: min(len(df_base), 1000)]  # Subsample for efficiency

        np.random.seed(seed)

        # Get SAM layer
        sam = self.model.get_sam_layer(layer)
        if sam is None:
            raise ValueError(f"No SAM layer at '{layer}'")

        d = sam.dim
        B, probs = self.extract_sam(layer)

        # Identify active edges (pruned adjacency)
        active_edges = []
        for i in range(d):
            for j in range(i):  # Lower triangular
                if probs[i, j] > 0.5:
                    active_edges.append((i, j))

        n_edges = len(active_edges)

        # Shapley value approximation via sampling
        edge_shap = np.zeros((d, d))

        for _ in tqdm(range(n_samples), desc="Shapley sampling"):
            # Random subset of edges
            subset_size = np.random.randint(0, n_edges + 1)
            subset_idx = np.random.choice(n_edges, size=subset_size, replace=False)
            subset = [active_edges[i] for i in subset_idx]

            # Pick a random edge not in subset
            remaining = [e for e in active_edges if e not in subset]
            if not remaining:
                continue
            edge = remaining[np.random.randint(len(remaining))]

            # Performance without edge
            perf_without = self._evaluate_with_edges(df, outcome, subset, layer)

            # Performance with edge
            perf_with = self._evaluate_with_edges(df, outcome, subset + [edge], layer)

            # Marginal contribution
            contrib = perf_with - perf_without

            edge_shap[edge[0], edge[1]] += contrib

        # Average over samples
        edge_shap /= n_samples

        # Node importance: sum of edge importances
        node_shap = np.zeros(d)
        for i in range(d):
            # Out-edges: i → j
            node_shap[i] += np.abs(edge_shap[i, :]).sum()
            # In-edges: j → i
            node_shap[i] += np.abs(edge_shap[:, i]).sum()

        return {
            'edge_shap': edge_shap,
            'edge_mask': (probs > 0.5).astype(int),
            'node_shap': node_shap,
            'active_edges': active_edges,
            'n_samples': n_samples
        }

    def _evaluate_with_edges(self, df, outcome, edge_list, layer):
        """Evaluate performance with only specified edges active."""

        sam = self.model.get_sam_layer(layer)
        with torch.no_grad():
            original_B = sam.B_tilde.detach().clone()
            original_alpha = sam.alpha_logits.detach().clone()

            # Zero out all edges, then restore only those in edge_list
            sam.B_tilde.zero_()
            sam.alpha_logits.fill_(-10.0)  # sigmoid(-10) ≈ 0

            for (i, j) in edge_list:
                sam.B_tilde[i, j] = original_B[i, j]
                sam.alpha_logits[i, j] = original_alpha[i, j]

        try:
            # Evaluate
            spec = next((o for o in self.outcomes if o.name == (outcome or self.outcomes[0].name)), None)
            if spec is None:
                raise ValueError(f"Outcome '{outcome}' not found.")
            family_name = spec.family.__class__.__name__
            mode = "proba" if family_name == 'Binomial' else "default"
            y_col = self._get_outcome_col(spec.name)
            if y_col is None:
                raise NotImplementedError("Edge evaluation currently supports scalar outcomes only.")

            preds = self.predict(df, outcome=spec.name, mode=mode)
            y_true = df[y_col].values
            mask = ~np.isnan(y_true)
            if mask.sum() == 0:
                raise ValueError(f"No observed targets for outcome '{spec.name}' to evaluate performance.")
            preds = preds[mask]
            y_true = y_true[mask]

            # Performance metric (higher is better)
            if family_name == 'Binomial':
                from sklearn.metrics import roc_auc_score
                perf = roc_auc_score(y_true, preds)
            else:
                from sklearn.metrics import r2_score
                perf = r2_score(y_true, preds)
        finally:
            with torch.no_grad():
                sam.B_tilde.copy_(original_B)
                sam.alpha_logits.copy_(original_alpha)

        return perf

    def save(self, path):
        # Save design and model state
        state = {
            'schema_version': SCHEMA_VERSION,
            'class_name': self.__class__.__name__,
            'init_params': self.get_params(deep=False),
            'design_state': self.design, # Design is picklable
            'model_state': self.model.state_dict(),
            'history': self.history,
            'input_dim': self.model.mixed_block.fixed_linear.in_features if self.model else None,
            # Keep a small slice of training data to enable post-load inference utilities.
            'train_data_sample': getattr(self, "_train_data", None)
                                 .iloc[: min(len(getattr(self, "_train_data", [])), 1024)]
                                 .copy()
                                 if getattr(self, "_train_data", None) is not None else None,
        }
        torch.save(state, path)
        
    @classmethod
    def load(cls, path):
        # Allow custom globals (CompiledDesign etc)
        state = torch.load(path, weights_only=False)
        
        schema = state.get('schema_version')
        if schema is not None and schema != SCHEMA_VERSION:
            raise ValueError(f"Incompatible schema version: {schema} (expected {SCHEMA_VERSION})")
        
        # Instantiate
        init_params = state.get('init_params')
        if init_params is None:
            # Backward compatibility for older checkpoints
            init_params = {
                'formula': state.get('formula'),
                'outcomes': state.get('outcomes'),
            }
            config = state.get('config', {})
            init_params.update(config)
        init_params = {k: v for k, v in init_params.items() if v is not None}

        # Wrapper classes already inject family; avoid duplicate kwargs.
        if cls.__name__ in ["TabMixNNRegressor", "TabMixNNClassifier"]:
            init_params.pop('family', None)
        
        instance = cls(**init_params)
        
        instance.design = state['design_state']
        
        # Setup model
        if state.get('input_dim') is not None:
            input_dim = state['input_dim']
        else:
            w = state['model_state']['mixed_block.fixed_linear.weight']
            input_dim = w.shape[1]
        
        instance.model = TabMixNNModule(
            fixed_input_dim=input_dim,
            compiled_design=instance.design,
            embedding_dim=instance.embedding_dim,
            hidden_dims=instance.hidden_dims,
            sam_location=instance.sam_location,
            sam_dim=instance.sam_dim,  # Matches layer width if None
            cov_struct_map=instance.cov_struct,
            mixed_activation=instance.mixed_activation,
            kl_normalize=True,  # Hardcoded optimal default
            scale_random_by_terms=instance.scale_random_by_terms,
            unknown_strategy=instance.unknown_strategy,
            sam_residual=instance.sam_residual,
            sam_residual_scale=instance.sam_residual_scale,
            sam_max_offdiag=instance.sam_max_offdiag,
            hidden_dropout=instance.hidden_dropout,
            hidden_layer_norm=instance.hidden_layer_norm,
            hidden_residual=instance.hidden_residual
        )
        instance.model.load_state_dict(state['model_state'])
        instance.history = state.get('history', [])
        instance._train_data = state.get('train_data_sample', None)
        
        return instance

class TabMixNNRegressor(TabMixNN):
    def __init__(self, formula, **kwargs):
        super().__init__(formula, family=Gaussian(), **kwargs)

class TabMixNNClassifier(TabMixNN):
    def __init__(self, formula, **kwargs):
        super().__init__(formula, family=Binomial(), **kwargs)
