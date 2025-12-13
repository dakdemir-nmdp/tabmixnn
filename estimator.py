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

class TabMixNN:
    def __init__(self, 
                 formula=None, 
                 outcomes=None, # List of OutcomeSpec
                 family=None, # For single outcome short-hand
                 sam_location="output",
                 sam_dim=64, # If sam_location dictates
                 hidden_dims=[64],
                 embedding_dim=64,
                 lr=0.001,
                 weight_decay=1e-4,
                 batch_size=256,
                 epochs=20,
                 lambda_sparse=0.001,
                 lambda_kl=0.1,
                 cov_struct=None, # dict: group -> {type: "IID"|"AR1"|"K", ...}
                 device="cpu",
                 verbose=False):
        
        self.formula = formula
        self.cov_struct = cov_struct
        if outcomes is None:
            if family is None:
                raise ValueError("Must provide either 'outcomes' or 'family'.")
            self.single_outcome_family = family
            self.outcomes = [] # Will populate in fit
        else:
            self.outcomes = outcomes
            
        self.sam_location = sam_location
        self.sam_dim = sam_dim
        self.hidden_dims = hidden_dims
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.lambda_sparse = lambda_sparse
        self.lambda_kl = lambda_kl
        self.device = device
        self.verbose = verbose
        
        self.design = None
        self.model = None
        self.history = []

    def fit(self, df, y=None):
        # Cache training data for inference methods
        self._train_data = df.copy()

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
                
        # 2. Compile Design
        self.design = CompiledDesign(
            formula=self.formula, 
            outcomes=self.outcomes, 
            sam_location=self.sam_location, 
            sam_dim=self.sam_dim,
            cov_struct_map=self.cov_struct
        )
        self.design.fit(df)
        
        # 3. Transform Data
        X, Z_ind, Z_slopes, Ys = self.design.transform(df, mode='fit')
        if self.verbose:
            print(f"Db: X shape {X.shape}")
            print(f"Db: X sample:\n{X[:3]}")
            print(f"Db: Ys sample: {list(Ys.values())[0][:3] if Ys else 'None'}")
        
        # Convert to Tensors
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        Z_ind_t = [torch.tensor(z, dtype=torch.long).to(self.device) for z in Z_ind]
        Z_slopes_t = [torch.tensor(z, dtype=torch.float32).to(self.device) for z in Z_slopes]
        
        Ys_t = {}
        for k, v in Ys.items():
            Ys_t[k] = torch.tensor(v, dtype=torch.float32).to(self.device)
        if not Ys_t:
            raise ValueError("No targets found in training data for the specified outcomes.")
        # 4. Initialize Model
        # print("Debugging: Initializing TabMixNNModule...")
        input_dim = X.shape[1]
        # Torch thread safety on Mac
        torch.set_num_threads(1)
        
        self.model = TabMixNNModule(
            fixed_input_dim=input_dim,
            compiled_design=self.design,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            sam_location=self.sam_location,
            sam_dim=self.sam_dim,
            cov_struct_map=self.cov_struct
        ).to(self.device)
        # print("Debugging: Model initialized successfully.")
        
        # 5. Training Loop
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Batching
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))
        
        # print("Debugging: Starting Epochs...")
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0
            perm = torch.randperm(n_samples)
            
            pbar_iter = range(n_batches)
            if self.verbose:
                pbar_iter = tqdm(pbar_iter, desc=f"Epoch {epoch}/{self.epochs}")
                
            for i in pbar_iter:
                # if i == 0: print("Db: Inside batch loop") # Cleaned
                idx = perm[i*self.batch_size : (i+1)*self.batch_size]
                
                batch_X = X_t[idx]
                batch_X = torch.index_select(X_t, 0, idx)
                batch_Z_ind = [torch.index_select(z, 0, idx) for z in Z_ind_t]
                batch_Z_slopes = [torch.index_select(z, 0, idx) for z in Z_slopes_t]
                batch_Ys = {k: torch.index_select(v, 0, idx) for k, v in Ys_t.items()}
                
                optimizer.zero_grad()
                outputs, kl = self.model(batch_X, batch_Z_ind, batch_Z_slopes) 
                
                # NLL Loss across outcomes
                nll_total = 0.0
                nll_count = 0
                for out_spec in self.outcomes:
                    name = out_spec.name
                    if name not in batch_Ys:
                        raise ValueError(f"Missing target values for outcome '{name}' in training batch.")
                    nll_total = nll_total + out_spec.family.nll(outputs[name], batch_Ys[name])
                    nll_count += 1
                if nll_count == 0:
                    raise ValueError("No outcomes contributed to the loss; check targets and outcomes configuration.")
                nll_total = nll_total / nll_count

                # KL (scale to batch)
                kl_scaled = kl * (len(idx) / n_samples) * self.lambda_kl
                sparse_loss = self.model.get_sparsity_loss() * self.lambda_sparse
                
                loss = nll_total + kl_scaled + sparse_loss
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            self.history.append(epoch_loss / n_batches)
            if self.verbose:
                print(f"Loss: {epoch_loss / n_batches:.4f}")
                
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
            return out_spec.family.predict(outputs[outcome], mode=mode)
            
        # Return dict of all
        results = {}
        for spec in self.outcomes:
            results[spec.name] = spec.family.predict(outputs[spec.name], mode=mode)
            
        # If single outcome mode and no outcome requested, return main
        if len(self.outcomes) == 1 and outcome is None:
             return results[self.outcomes[0].name]
             
        return results

    def predict_proba(self, df, outcome=None):
        # Only valid for classification
        return self.predict(df, outcome=outcome, mode='proba')

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
        print(f"SAM Location: {self.sam_location} (dim={self.sam_dim}, sparse_lambda={self.lambda_sparse})")
        
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
                probs = torch.sigmoid(sam.alpha_logits).detach().cpu().numpy()
                n_edges = (probs > 0.5).sum()
                total = probs.size
                print(f"  Layer '{loc}': {n_edges}/{total} active edges (p>0.5). Mean prob: {probs.mean():.4f}")

    def extract_sam(self, layer="output"):
        """
        Returns (weights_B, gate_probs) for the specified SAM layer.
        """
        if not self.model: 
            raise RuntimeError("Not fitted")
            
        sam = self.model.get_sam_layer(layer)
        if sam is None:
            raise ValueError(f"No SAM layer found at '{layer}'")
            
        B = sam.get_B(deterministic=True).detach().cpu().numpy()
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

    def _evaluate_performance(self, df, outcome):
        """Evaluate model performance (for ablation studies)."""
        preds = self.predict(df, outcome=outcome)
        y_true = df[self._get_outcome_col(outcome)].values

        family_name = self.outcomes[0].family.__class__.__name__

        if family_name == 'Binomial':
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, preds)
        elif family_name in ['Gaussian', 'Poisson', 'NegBinomial']:
            from sklearn.metrics import r2_score
            return r2_score(y_true, preds)
        else:
            raise NotImplementedError(f"Metric not defined for {family_name}")

    def _predict_ablate_random(self, df, outcome):
        """Predict with all random effects set to zero."""
        # Temporarily zero out variational means
        original_params = {}
        for tid, params in self.model.mixed_block.variational_params.items():
            mu = params[0]
            original_params[tid] = mu.data.clone()
            mu.data.zero_()

        # Predict
        preds = self.predict(df, outcome=outcome)

        # Restore
        for tid, mu_orig in original_params.items():
            self.model.mixed_block.variational_params[tid][0].data = mu_orig

        return preds

    def _predict_with_term(self, df, outcome, term_id):
        """Predict with only specified term active."""
        # Zero all except term_id
        original_params = {}
        for tid, params in self.model.mixed_block.variational_params.items():
            mu = params[0]
            original_params[tid] = mu.data.clone()
            if tid != term_id:
                mu.data.zero_()

        preds = self.predict(df, outcome=outcome)

        # Restore
        for tid, mu_orig in original_params.items():
            self.model.mixed_block.variational_params[tid][0].data = mu_orig

        return preds

    def _predict_without_term(self, df, outcome, term_id):
        """Predict with specified term zeroed out."""
        # Zero only term_id
        original_params = {}
        for tid, params in self.model.mixed_block.variational_params.items():
            if tid == term_id:
                mu = params[0]
                original_params[tid] = mu.data.clone()
                mu.data.zero_()

        preds = self.predict(df, outcome=outcome)

        # Restore
        for tid, mu_orig in original_params.items():
            self.model.mixed_block.variational_params[tid][0].data = mu_orig

        return preds

    # --- Inference Methods ---

    def variance_decomposition(self, outcome=None, df=None):
        """Partition variance into fixed, random, and residual."""

        if df is None:
            df = self._train_data

        # Get predictions with all components
        y_full = self.predict(df, outcome=outcome)
        y_true = df[self._get_outcome_col(outcome)].values

        # 1. Total variance
        var_total = np.var(y_true)

        # 2. Residual variance (irreducible error)
        residuals = y_true - y_full
        var_residual = np.var(residuals)

        # 3. Fixed effects variance
        # Predict with random effects zeroed out
        y_fixed_only = self._predict_ablate_random(df, outcome=outcome)
        var_fixed = np.var(y_fixed_only)

        # 4. Random effects variance per term
        var_random = {}
        icc = {}
        variance_explained = {}

        for term in self.design.random_terms:
            tid = term.term_id

            # Predict with only this random term active
            y_with_term = self._predict_with_term(df, outcome, term_id=tid)
            y_without_term = self._predict_without_term(df, outcome, term_id=tid)

            # Variance contribution of this term
            contrib = y_with_term - y_without_term
            var_random[tid] = np.var(contrib)

            # ICC: σ²_u / (σ²_u + σ²_e)
            sigma2_u = var_random[tid]
            sigma2_e = var_residual
            icc[tid] = sigma2_u / (sigma2_u + sigma2_e)

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
        train_df = self._train_data
        n_samples = len(train_df)

        for b in tqdm(range(n_boot), desc="Bootstrap samples"):
            # Case bootstrap: resample rows with replacement
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_df = train_df.iloc[boot_idx].reset_index(drop=True)

            # Clone model architecture (reuse hyperparameters)
            boot_model = self._clone()

            # Warm start: initialize from current weights
            if warm_start:
                boot_model.model.load_state_dict(
                    self.model.state_dict(), strict=False
                )

            # Refit on bootstrap sample (fewer epochs for speed)
            boot_model.epochs = max(self.epochs // 2, 10)
            boot_model.verbose = False
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

        if method == 'gradient':
            return self._shap_gradient(df, outcome, background_size, seed)
        elif method == 'kernel':
            raise NotImplementedError("KernelSHAP not yet implemented")
        else:
            raise ValueError(f"Unknown method: {method}")

    def _shap_gradient(self, df, outcome, background_size, seed):
        """GradientSHAP: fast approximation for neural networks."""

        # Sample background data from training set
        np.random.seed(seed)
        bg_idx = np.random.choice(len(self._train_data),
                                   size=min(background_size, len(self._train_data)),
                                   replace=False)
        bg_df = self._train_data.iloc[bg_idx]

        # Get fixed design matrix (exclude random effects for SHAP)
        X_explain = self.design._encode_fixed(df)  # [n, p]
        X_bg = self.design._encode_fixed(bg_df)    # [bg_size, p]

        n_samples, n_features = X_explain.shape

        # Baseline prediction (expected value over background)
        y_bg = self.predict(bg_df, outcome=outcome)
        base_value = np.mean(y_bg)

        # Compute SHAP values: φ_j ≈ (x_j - x̄_j) × E[∂f/∂x_j]
        shap_values = np.zeros((n_samples, n_features))

        X_explain_t = torch.tensor(X_explain, dtype=torch.float32, requires_grad=True).to(self.device)

        for i in range(n_samples):
            x_i = X_explain_t[i:i+1]  # [1, p]

            # Expected gradient over background
            grad_sum = torch.zeros(n_features).to(self.device)

            for j in range(len(bg_df)):
                x_bg_j = torch.tensor(X_bg[j:j+1], dtype=torch.float32).to(self.device)

                # Interpolate: x(α) = α*x_i + (1-α)*x_bg
                # Use midpoint α=0.5 for efficiency
                x_interp = 0.5 * x_i + 0.5 * x_bg_j
                x_interp.requires_grad_(True)

                # Forward pass
                y_pred = self._forward_fixed_only(x_interp, outcome)

                # Gradient
                if x_interp.grad is not None:
                    x_interp.grad.zero_()
                y_pred.backward()
                grad_sum += x_interp.grad.squeeze()

            grad_mean = grad_sum / len(bg_df)

            # SHAP value: difference × gradient
            x_diff = X_explain[i] - X_bg.mean(axis=0)
            shap_values[i] = x_diff * grad_mean.detach().cpu().numpy()

        # Get feature names
        feature_names = self.design.fixed_cols

        return {
            'shap_values': shap_values,
            'base_value': base_value,
            'feature_names': feature_names,
            'predictions': self.predict(df, outcome=outcome)
        }

    def _forward_fixed_only(self, X_tensor, outcome):
        """Forward pass through fixed effects only (for SHAP)."""
        # Simplified forward: fixed → hidden → head
        h = self.model.mixed_block.fixed_linear(X_tensor)
        h = F.relu(h)

        for layer in self.model.hidden_layers:
            if isinstance(layer, nn.Linear):
                h = F.relu(layer(h))

        # Get outcome spec
        if outcome is None:
            outcome = self.outcomes[0].name

        logits = self.model.heads[outcome](h)

        # Return scalar prediction
        family_name = self.outcomes[0].family.__class__.__name__
        if family_name == 'Binomial':
            return torch.sigmoid(logits).squeeze()
        else:
            return logits[:, 0]  # Gaussian: first param is mean

    def plot_shap_summary(self, df, outcome=None, max_features=20):
        """Beeswarm plot of SHAP values."""
        import matplotlib.pyplot as plt

        shap_result = self.shapley_values(df, outcome=outcome)
        shap_vals = shap_result['shap_values']
        feature_names = shap_result['feature_names']

        # Get feature values for coloring
        X = self.design._encode_fixed(df)

        # Sort features by mean |SHAP|
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[::-1][:max_features]

        fig, ax = plt.subplots(figsize=(10, max_features * 0.4))

        for i, feat_idx in enumerate(top_idx):
            y_pos = max_features - i - 1
            shap_feat = shap_vals[:, feat_idx]
            x_feat = X[:, feat_idx]

            # Scatter with color = feature value
            scatter = ax.scatter(shap_feat, np.full_like(shap_feat, y_pos),
                                 c=x_feat, cmap='coolwarm', alpha=0.6, s=20)

        ax.set_yticks(range(max_features))
        ax.set_yticklabels([feature_names[i] for i in top_idx])
        ax.set_xlabel('SHAP value (impact on prediction)')
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_title(f'SHAP Feature Importance (outcome: {outcome})')

        plt.colorbar(scatter, label='Feature value')
        plt.tight_layout()

        return fig

    def save(self, path):
        # Save design and model state
        state = {
            'formula': self.formula,
            'outcomes': self.outcomes,
            'config': {
                'sam_location': self.sam_location,
                'sam_dim': self.sam_dim,
                'hidden_dims': self.hidden_dims,
                'embedding_dim': self.embedding_dim,
            },
            'design_state': self.design, # Design is picklable
            'model_state': self.model.state_dict(),
            'history': self.history
        }
        torch.save(state, path)
        
    @classmethod
    def load(cls, path):
        # Allow custom globals (CompiledDesign etc)
        state = torch.load(path, weights_only=False)
        
        # Instantiate
        # We need to recreate the object.
        # Design is already fit.
        
        # Create instance
        instance = cls(
            formula=state['formula'],
            outcomes=state['outcomes'],
            sam_location=state['config']['sam_location'],
            sam_dim=state['config']['sam_dim'],
            hidden_dims=state['config']['hidden_dims'],
            embedding_dim=state['config']['embedding_dim']
        )
        
        instance.design = state['design_state']
        
        # Setup model
        input_dim = len(instance.design.fixed_cols) + (1 if instance.design.intercept_included else 0) 
        # Actually input_dim should be retrieved from design transform output size?
        # Fixed cols list in design might not account for one-hot expansion.
        # We need to run a dummy transform or store input_dim in design.
        # Let's fix this by storing input_dim in design or inferring.
        # design.transform returns X.
        # We can store fixed_input_dim in design.
        # Let's hack: we check the weights shape of the first layer of saved model lol.
        # Or better, just rely on the stored model state dict to match new initialization.
        # But we need input_dim to init module.
        
        # Let's add input_dim to compiled design in next update.
        # For now, let's infer from the first layer weight in state_dict.
        # 'mixed_block.fixed_linear.weight' shape [emb, input_dim]
        dict_keys = state['model_state'].keys()
        w = state['model_state']['mixed_block.fixed_linear.weight']
        input_dim = w.shape[1]
        
        instance.model = TabMixNNModule(
            fixed_input_dim=input_dim,
            compiled_design=instance.design,
            embedding_dim=state['config']['embedding_dim'],
            hidden_dims=state['config']['hidden_dims'],
            sam_location=state['config']['sam_location'],
            sam_dim=state['config']['sam_dim']
        )
        instance.model.load_state_dict(state['model_state'])
        instance.history = state.get('history', [])
        
        return instance

class TabMixNNRegressor(TabMixNN):
    def __init__(self, formula, **kwargs):
        super().__init__(formula, family=Gaussian(), **kwargs)

class TabMixNNClassifier(TabMixNN):
    def __init__(self, formula, **kwargs):
        super().__init__(formula, family=Binomial(), **kwargs)
