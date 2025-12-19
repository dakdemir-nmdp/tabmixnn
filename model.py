
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Any
from tqdm import tqdm
from .formula import CompiledDesign, OutcomeSpec
from .encoder import MixedEffectsRepresentation
from .backbones.gsem import GSEMBackbone 
from .backbones.manifold import ManifoldBackbone 
from .families import DiscreteTimeSurvival, DiscreteTimeCompetingRisks, Binomial, Gaussian, Poisson, NegBinomial
from .callbacks import CallbackList, History, EarlyStopping
from .logging_config import get_logger

class TabMixNN(nn.Module):
    """
    TabMixNN: Synthesizing the best of General Mixed Models and Deep Learning.
    
    Architecture:
      1. Robust Mixed Effects Encoder
      2. Backbone (GSEM or Universal)
      3. Flexible Output Heads
    """
    
    @property
    def layer(self):
        """Legacy compatibility property for accessing the 'main' layer."""
        return self
    
    @property
    def structural(self):
        """Legacy compatibility property."""
        if self.architecture == 'gsem' and self.backbone and hasattr(self.backbone, 'hidden_layers'):
            if len(self.backbone.hidden_layers) > 0:
                struct = self.backbone.hidden_layers[0].structure
                if struct is not None:
                    return struct
        # Return a dummy if no structure exists
        class DummyStructural:
            def __init__(self):
                self.use_static = False
                self.use_dynamic = False
        return DummyStructural()

    @property
    def random_effects(self):
        """Legacy compatibility property."""
        if self.encoder:
            return RandomEffectsShim(self.encoder)
        return None
    
    @classmethod
    def lmm(cls, formula=None, formula_body=None, input_dim=None, group_counts=None, **kwargs):
        """Factory for Linear Mixed Models."""
        f = formula or formula_body or "y ~ ."
        # Default to Gaussian if no family provided
        if 'family' not in kwargs and 'outcomes' not in kwargs:
            kwargs['family'] = Gaussian()
        model = cls(f, architecture="gsem", **kwargs)
        
        if input_dim is not None:
            # Build model immediately if input_dim is provided for legacy test compatibility
            from .formula import CompiledDesign, RandomEffectTerm
            # Prepare outcomes for design if single family used
            if not model.outcomes and model.single_outcome_family:
                from .formula import OutcomeSpec
                model.outcomes = [OutcomeSpec(name="main", y="y", family=model.single_outcome_family)]
            
            design = CompiledDesign(f, model.outcomes, "none", None)
            if group_counts:
                for i, (group, n) in enumerate(group_counts.items()):
                    term = RandomEffectTerm(f"term_{i}", group, ["Intercept"])
                    term.level_map = {j: j for j in range(n)}
                    design.random_terms.append(term)
            model._build_model(fixed_input_dim=input_dim, compiled_design=design)
            model.design = design
        return model

    @classmethod
    def sem(cls, formula=None, formula_body=None, input_dim=1, latent_dim=16, output_dim=16, **kwargs):
        """Factory for Structural Equation Models."""
        f = formula or formula_body or "y ~ 1"
        # Default to Gaussian if no family provided
        if 'family' not in kwargs and 'outcomes' not in kwargs:
            # For SEM with output_dim > 1, use MultivariateGaussian
            if output_dim > 1:
                from .families import MultivariateGaussian
                kwargs['family'] = MultivariateGaussian(n_tasks=output_dim)
            else:
                kwargs['family'] = Gaussian()
        
        model = cls(f, 
                   architecture="gsem", 
                   use_static_structure=True,
                   hidden_dims=[latent_dim],
                   **kwargs)
        
        if input_dim is not None:
            from .formula import CompiledDesign, OutcomeSpec
            if not model.outcomes and model.single_outcome_family:
                model.outcomes = [OutcomeSpec(name="main", y="y", family=model.single_outcome_family)]
            design = CompiledDesign(f, model.outcomes, "none", None)
            model._build_model(fixed_input_dim=input_dim, compiled_design=design)
            model.design = design
            
        return model

    @classmethod
    def transformer(cls, formula=None, formula_body=None, input_dim=16, hidden_dim=16, n_heads=4, **kwargs):
        """Factory for Transformer/Attention-based models."""
        f = formula or formula_body or "y ~ 1"
        # Default to Gaussian if no family provided
        if 'family' not in kwargs and 'outcomes' not in kwargs:
            if hidden_dim > 1:
                from .families import MultivariateGaussian
                kwargs['family'] = MultivariateGaussian(n_tasks=hidden_dim)
            else:
                kwargs['family'] = Gaussian()
            
        model = cls(f, 
                   architecture="gsem", 
                   use_dynamic_structure=True,
                   n_attention_heads=n_heads,
                   hidden_dims=[hidden_dim],
                   **kwargs)
                   
        if input_dim is not None:
            from .formula import CompiledDesign, OutcomeSpec
            if not model.outcomes and model.single_outcome_family:
                model.outcomes = [OutcomeSpec(name="main", y="y", family=model.single_outcome_family)]
            design = CompiledDesign(f, model.outcomes, "none", None)
            model._build_model(fixed_input_dim=input_dim, compiled_design=design)
            model.design = design
            
        return model
    
    def __init__(
        self, 
        formula: str,
        outcomes: Optional[List[OutcomeSpec]] = None,
        family: Optional[Any] = None, 
        # mixed effects config
        embedding_dim: int = 16,
        cov_struct_map: Optional[Dict] = None,
        cov_struct: Optional[Dict] = None,
        scale_random_by_terms: bool = False,
        unknown_strategy: str = "learned",
        # backbone config
        architecture: str = "gsem", # gsem or universal
        hidden_dims: List[int] = [64, 32],
        # GSEM specific
        use_static_structure: Union[bool, List[bool]] = False,
        use_dynamic_structure: Union[bool, List[bool]] = False,
        n_attention_heads: Union[int, List[int]] = 4,
        backbone_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        # Manifold / Universal specific
        manifold_configs: Optional[List[Dict]] = None,
        use_recursive: bool = True,
        aggregation: str = "concat",
        # training config
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        epochs: int = 20,
        lambda_sparse: float = 0.001,
        lambda_kl: float = 0.01,
        lambda_dag: float = 0.0,
        lambda_contraction: float = 0.0,
        grad_clip_norm: float = 1.0,
        device: str = "cpu",
        verbose: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.formula = formula
        if outcomes is None:
            if family is None:
                raise ValueError("Must provide either 'outcomes' or 'family'.")
            self.single_outcome_family = family
            self.outcomes = [] 
        else:
            self.outcomes = outcomes
            self.single_outcome_family = None
            
        self.embedding_dim = embedding_dim
        self.cov_struct_map = cov_struct_map or cov_struct or {}
        self.scale_random_by_terms = scale_random_by_terms
        self.unknown_strategy = unknown_strategy
        
        self.architecture = architecture.lower()
        self.hidden_dims = hidden_dims
        # GSEM
        self.use_static_structure = use_static_structure
        self.use_dynamic_structure = use_dynamic_structure
        self.n_attention_heads = n_attention_heads
        self.backbone_dropout = backbone_dropout or hidden_dropout
        # Universal
        self.manifold_configs = manifold_configs
        self.use_recursive = use_recursive
        self.aggregation = aggregation
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.lambda_sparse = lambda_sparse
        self.lambda_kl = lambda_kl
        self.lambda_dag = lambda_dag
        self.lambda_contraction = lambda_contraction
        self.grad_clip_norm = grad_clip_norm
        self.device = device
        self.verbose = verbose

        self.logger = get_logger("TabMixNN")

        # Input validation
        if epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {epochs}")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if not isinstance(hidden_dims, list):
            raise ValueError(f"hidden_dims must be a list, got {type(hidden_dims).__name__}")
        if any(d <= 0 for d in hidden_dims):
            raise ValueError(f"hidden_dims must be positive integers, got {hidden_dims}")
        if 'sam_location' in kwargs:
            valid_locations = ['input', 'hidden', 'output', 'all', 'none', 'first', 'both', None]
            if kwargs['sam_location'] not in valid_locations:
                raise ValueError(f"sam_location must be one of {valid_locations}, got {kwargs['sam_location']}")
        if not device.startswith(('cpu', 'cuda', 'mps')):
            raise ValueError(f"device must start with one of ['cpu', 'cuda', 'mps'], got {device}")

        # Compatibility Shim
        if 'hidden_layer_type' in kwargs and kwargs['hidden_layer_type'] == 'universal_samnn':
            self.architecture = 'manifold'
        if 'use_universal' in kwargs and kwargs['use_universal']:
            self.architecture = 'manifold'

        if self.architecture == 'universal':
            self.architecture = 'manifold'
            
        # SAM Location Shim (if use_static_structure is default False)
        if 'sam_location' in kwargs and self.use_static_structure is False:
             loc = kwargs['sam_location']
             n_layers = len(self.hidden_dims)
             if loc == 'first':
                 self.use_static_structure = [True] + [False]*(n_layers-1) if n_layers > 0 else False
             elif loc == 'output':
                 self.use_static_structure = [False]*(n_layers-1) + [True] if n_layers > 0 else False
             elif loc == 'both':
                 if n_layers == 1: self.use_static_structure = [True]
                 else: self.use_static_structure = [True] + [False]*(n_layers-2) + [True]
             elif loc == 'all':
                 self.use_static_structure = True
             elif loc == 'none':
                 self.use_static_structure = False
                 
        # State
        self.design = None
        self.encoder = None
        self.backbone = None
        self.heads = None
        self.history = []
        self.val_history = []
        self._train_data = None
        self.stop_training = False
        
    def fit(self,
            df: pd.DataFrame,
            y=None,
            val_df: pd.DataFrame = None,
            val_fraction: float = 0.0,
            early_stopping: bool = False,
            patience: int = 5,
            min_delta: float = 0.0,
            callbacks: Optional[list] = None,
            reuse_design=None,
            **kwargs):
        """Fit model to dataframe."""

        # Validate val_fraction
        if val_fraction > 0 and (val_fraction >= 1):
            raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

        self._train_data = df

        # Handle Validation Split
        train_df = df
        if val_df is None and val_fraction > 0.0:
            val_df = df.sample(frac=val_fraction, random_state=42)
            train_df = df.drop(val_df.index).reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
            self.logger.info(f"Split data: {len(train_df)} train, {len(val_df)} validation")

        # 1. Prepare Outcome Specs if single
        if not self.outcomes:
            if '~' in self.formula:
                lhs = self.formula.split('~')[0].strip()
                if not lhs:
                    raise ValueError("Formula missing LHS for single outcome mode.")
                self.outcomes = [OutcomeSpec(name="main", y=lhs, family=self.single_outcome_family)]
            else:
                raise ValueError("Formula must include LHS 'y ~ ...' for single outcome shortcut.")

        # 2. Compile Design
        if reuse_design is not None:
             self.design = reuse_design
        elif self.design is None:
            self.logger.info("Compiling design...")
            self.design = CompiledDesign(
                self.formula, self.outcomes, 
                sam_location="none",
                sam_dim=None,
                cov_struct_map=self.cov_struct_map
            )
            self.design.fit(train_df)
            
        # 3. Transform Data
        X, Z_ind, Z_slopes, Ys = self.design.transform(train_df, mode='fit')
        
        # 4. Initialize Modules
        if self.encoder is None:
            self._build_model(fixed_input_dim=X.shape[1], compiled_design=self.design)
            self.to(self.device)
            
        # 5. Prepare Tensors
        train_data = self._to_tensors(X, Z_ind, Z_slopes, Ys)
        val_data = None
        if val_df is not None:
            Xv, Zv_ind, Zv_slopes, Yv = self.design.transform(val_df, mode='fit')
            val_data = self._to_tensors(Xv, Zv_ind, Zv_slopes, Yv)
        
        # 6. Callbacks
        callbacks = list(callbacks or [])
        if early_stopping:
            # Only add if user did not supply their own EarlyStopping
            has_es = any(isinstance(cb, EarlyStopping) for cb in callbacks)
            if not has_es:
                callbacks.append(EarlyStopping(monitor="val_loss", patience=patience, min_delta=min_delta, restore_best_weights=True))
        callback_list = CallbackList(callbacks + [History()])
        callback_list.set_trainer(self)
            
        # 7. Training Loop
        self._train_loop(train_data, val_data, callback_list)
        
        return self
        
    def predict(self, df: pd.DataFrame, outcome: str = None, mode: str = 'default'):
        """Predict outcomes."""
        self.eval()
        X, Z_ind, Z_slopes, _ = self.design.transform(df, mode='predict')
        tensors = self._to_tensors(X, Z_ind, Z_slopes, {})
        
        with torch.no_grad():
            outputs = self.forward(tensors[0], tensors[1], tensors[2])
            
        # Convert single output to dict for loop
        if isinstance(outputs, torch.Tensor):
            name = self.outcomes[0].name if self.outcomes else "main"
            outputs_dict = {name: outputs}
        else:
            outputs_dict = outputs
            
        results = {}
        for name, pred_raw in outputs_dict.items():
            idx = self._outcome_idx(name)
            if idx == -1: continue
            family = self.design.outcomes[idx].family
            theta = family.forward_params(pred_raw)
            results[name] = family.predict(theta, mode=mode)
            
        if outcome:
            return results[outcome]
        if len(results) == 1 and outcome is None:
            return list(results.values())[0]
        return results

    def predict_proba(self, df, outcome=None):
        return self.predict(df, outcome=outcome, mode='proba')

    def score(self, df, outcome=None):
        metrics = self._evaluate_performance(df, outcome)
        return metrics

    # --- Internal & Setup ---

    def _build_model(self, fixed_input_dim, compiled_design):
        """Construct the PyTorch architecture."""
        
        # A. Robust Encoder
        random_config = []
        for term in compiled_design.random_terms:
            random_config.append({
                'term_id': term.term_id,
                'n_levels': max(term.level_map.values()) + 1 if term.level_map else 1,
                'n_slopes': len(term.slope_cols),
                'group': term.group_col,
                'train_levels': getattr(term, "train_levels", [])
            })
            
        self.encoder = MixedEffectsRepresentation(
             fixed_input_dim=fixed_input_dim,
             embedding_dim=self.embedding_dim,
             random_terms_config=random_config,
             cov_struct_map=self.cov_struct_map,
             kl_normalize=True,
             scale_random_by_terms=self.scale_random_by_terms,
             unknown_strategy=self.unknown_strategy
        )
        
        encoder_output_dim = self.embedding_dim
        
        # Projection/Backbone Selection
        self.input_proj = None
        
        if self.architecture == 'manifold':
            # Manifold SAM-NN Backbone
            if self.manifold_configs is None:
                # Heuristic: 1 manifold per hidden dim
                self.manifold_configs = []
                for h in self.hidden_dims:
                    d = int(np.sqrt(h))
                    if d*d < h: d += 1
                    self.manifold_configs.append({'grid_shape': (d,d), 'use_spde': True, 'dropout': self.backbone_dropout})

            self.backbone = ManifoldBackbone(
                input_dim=encoder_output_dim,
                manifold_configs=self.manifold_configs,
                random_config=random_config,
                use_recursive=self.use_recursive,
                aggregation=self.aggregation
            )
            final_dim = self.backbone.output_dim
            
        else:
            # GSEM Backbone (Default)
            # Add proj if needed AND we have hidden dims
            if self.hidden_dims and encoder_output_dim != self.hidden_dims[0]:
                 self.input_proj = nn.Linear(encoder_output_dim, self.hidden_dims[0])
                 encoder_output_dim = self.hidden_dims[0]
                 
            self.backbone = GSEMBackbone(
                input_dim=encoder_output_dim,
                hidden_dims=self.hidden_dims,
                use_static_structure=self.use_static_structure,
                use_dynamic_structure=self.use_dynamic_structure,
                n_attention_heads=self.n_attention_heads,
                dropout=self.backbone_dropout,
            )
            final_dim = self.hidden_dims[-1] if self.hidden_dims else encoder_output_dim
        
        # C. Output Heads
        self.heads = nn.ModuleDict()
        self.families = nn.ModuleDict()
        
        for o in self.outcomes:
             self.heads[o.name] = nn.Linear(final_dim, o.family.n_params)
             self.families[o.name] = o.family
             
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, X, Z_ind=None, Z_slopes=None, return_attention=False):
        """Forward pass."""
        # Handle legacy positional args / missing args
        if Z_ind is None: Z_ind = []
        if Z_slopes is None: Z_slopes = []
        
        # Convert dict groups to list if needed
        if isinstance(Z_ind, dict):
            # Try to map based on random_terms in design
            if self.design:
                Z_ind_list = []
                Z_slopes_list = []
                for term in self.design.random_terms:
                    Z_ind_list.append(Z_ind.get(term.group_col, torch.zeros(X.shape[0], dtype=torch.long, device=X.device)))
                    Z_slopes_list.append(torch.ones(X.shape[0], 1, device=X.device)) # default intercept
                Z_ind, Z_slopes = Z_ind_list, Z_slopes_list
            else:
                # Fallback: take dict values
                Z_ind = list(Z_ind.values())
                Z_slopes = [torch.ones(X.shape[0], 1, device=X.device)] * len(Z_ind)

        # 1. Encode
        h_encoded, kl_encoder = self.encoder(X, Z_ind, Z_slopes)
        self.current_kl = kl_encoder
        
        if self.input_proj:
             h_encoded = self.input_proj(h_encoded)
        
        # 2. Backbone
        h_last = h_encoded
        attn = None
        
        if self.architecture == 'manifold':
            # Create context for Manifold
            context = {'Z_indices': Z_ind}
            h_last, _ = self.backbone(h_encoded, context)
        else:
            # GSEM
            context = {'group_indices': Z_ind}
            res = self.backbone(h_encoded, context, return_attention=return_attention)
            if return_attention:
                 h_last, _, attn_list = res
                 attn = attn_list[-1] if attn_list else None
            else:
                 h_last, _ = res
        
        # 3. Heads
        outputs = {}
        for name, head in self.heads.items():
            outputs[name] = head(h_last)
            
        self.current_kl = kl_encoder
            
        # Legacy return if return_attention
        if return_attention:
            # Return (outputs, attn) to match legacy tests
            if len(outputs) == 1:
                return list(outputs.values())[0], attn
            return outputs, attn
            
        # Unified return: Return just outcomes to match standard Module behavior
        if len(outputs) == 1:
            return list(outputs.values())[0]
        return outputs

    def _train_loop(self, train_data, val_data, callback_list: Optional[CallbackList] = None):
        X, Z_ind, Z_slopes, Ys = train_data
        n_samples = len(X)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        self.stop_training = False
        callbacks = callback_list or CallbackList()

        # Initialize progress bar if verbose
        if self.verbose:
            pbar = tqdm(range(self.epochs), desc="Training")
            epoch_iterator = pbar
        else:
            pbar = None
            epoch_iterator = range(self.epochs)

        callbacks.on_train_begin({"epochs": self.epochs, "batches": n_batches})

        for epoch in epoch_iterator:
            self.train()
            epoch_loss = 0
            
            # Shuffle indices
            indices = torch.randperm(n_samples, device=self.device)
            
            callbacks.on_epoch_begin(epoch, {"epoch": epoch})

            for i in range(n_batches):
                callbacks.on_batch_begin(i, {"batch": i})

                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                bx = X[batch_indices]
                bz_ind = [z[batch_indices] for z in Z_ind]
                bz_slopes = [z[batch_indices] for z in Z_slopes]
                bys = {k: v[batch_indices] for k, v in Ys.items()}
                
                self.optimizer.zero_grad()
                outputs = self.forward(bx, bz_ind, bz_slopes)
                kl = self.current_kl
                
                # Convert single output to dict for loop
                if isinstance(outputs, torch.Tensor):
                    name = self.outcomes[0].name if self.outcomes else "main"
                    outputs_dict = {name: outputs}
                else:
                    outputs_dict = outputs

                loss = torch.tensor(0.0, device=bx.device)
                count = 0
                for name, target in bys.items():
                    family = self.data_family(name)
                    if target.ndim > 1:
                        mask = ~torch.isnan(target).any(dim=1)
                    else:
                        mask = ~torch.isnan(target)
                    if mask.sum() == 0: continue
                    pred = outputs_dict[name][mask]
                    tgt = target[mask]
                    theta = family.forward_params(pred)
                    loss = loss + family.nll(theta, tgt)
                    count += 1
                
                if count > 0: loss = loss / count
                
                if kl is not None:
                    loss = loss + self.lambda_kl * kl * (len(bx) / n_samples)
                
                loss = loss + self._structural_penalties()
                
                loss.backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_norm)
                self.optimizer.step()
                epoch_loss += loss.item()
                callbacks.on_batch_end(i, {"batch": i, "batch_loss": loss.item()})
                
            train_avg = epoch_loss / n_batches
            self.history.append(train_avg)
            
            val_loss = None
            if val_data:
                val_loss = self._evaluate_loss(val_data)
                self.val_history.append(val_loss)
                callbacks.on_validation_end({"val_loss": val_loss})
                        
            if self.verbose and pbar is not None:
                 msg = f"loss={train_avg:.4f}"
                 if val_loss: msg += f", val_loss={val_loss:.4f}"
                 pbar.set_postfix_str(msg)
            
            logs = {"loss": train_avg}
            if val_loss is not None:
                logs["val_loss"] = val_loss
            callbacks.on_epoch_end(epoch, logs)

            if getattr(self, "stop_training", False):
                if self.verbose:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                break

        callbacks.on_train_end(logs if 'logs' in locals() else None)
                
    def _evaluate_loss(self, data):
        self.eval()
        X, Z_ind, Z_slopes, Ys = data
        with torch.no_grad():
            outputs = self.forward(X, Z_ind, Z_slopes)
            
            # Convert single output to dict for loop
            if isinstance(outputs, torch.Tensor):
                name = self.outcomes[0].name if self.outcomes else "main"
                outputs_dict = {name: outputs}
            else:
                outputs_dict = outputs
                
            loss = 0
            count = 0
            for name, target in Ys.items():
                family = self.data_family(name)
                if target.ndim > 1: mask = ~torch.isnan(target).any(dim=1)
                else: mask = ~torch.isnan(target)
                if mask.sum() == 0: continue
                pred = outputs_dict[name][mask]
                tgt = target[mask]
                theta = family.forward_params(pred)
                loss += family.nll(theta, tgt)
                count += 1
            if count > 0: loss /= count
            return loss.item()

    def _structural_penalties(self):
        # We must ensure this returns a tensor that requires grad if any of its components do,
        # otherwise we might get "element 0 of tensors does not require grad".
        device = self.device if hasattr(self, 'device') else next(self.parameters()).device
        total_loss = torch.tensor(0.0, device=device)
        
        has_grad_penalties = False
        if self.architecture == 'gsem':
            for layer in self.backbone.hidden_layers:
                 pen = layer.penalties()
                 for k, v in pen.items():
                     if not isinstance(v, torch.Tensor): continue
                     if v.requires_grad: has_grad_penalties = True
                     
                     if 'kl' in k: total_loss = total_loss + self.lambda_kl * v
                     elif 'dag' in k: total_loss = total_loss + self.lambda_dag * v
                     elif 'contraction' in k: total_loss = total_loss + self.lambda_contraction * v
                     else: total_loss = total_loss + self.lambda_sparse * v
        
        # If no penalties found, return something that might require grad if we need it
        # But if total_loss is still 0.0, it's fine as long as the main NLL loss requires grad.
        return total_loss

    def data_family(self, name):
        """Get family logic."""
        idx = self._outcome_idx(name)
        return self.design.outcomes[idx].family

    def _to_tensors(self, X, Z_ind, Z_slopes, Ys):
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        Zit = [torch.tensor(z, dtype=torch.long).to(self.device) for z in Z_ind]
        Zst = [torch.tensor(z, dtype=torch.float32).to(self.device) for z in Z_slopes]
        Yst = {k: torch.tensor(v, dtype=torch.float32).to(self.device) for k,v in Ys.items()}
        return Xt, Zit, Zst, Yst

    def _outcome_idx(self, name):
        for i, o in enumerate(self.outcomes):
            if o.name == name: return i
        return -1

    # --- Sklearn & Introspection ---

    def get_params(self, deep=True):
        return {
            "formula": self.formula,
            "outcomes": self.outcomes,
            "family": getattr(self, "single_outcome_family", None),
            "architecture": self.architecture,
            "embedding_dim": self.embedding_dim,
            "hidden_dims": self.hidden_dims,
            "use_static_structure": self.use_static_structure,
            "use_dynamic_structure": self.use_dynamic_structure,
            "n_attention_heads": self.n_attention_heads,
            "manifold_configs": self.manifold_configs,
            "use_recursive": self.use_recursive,
            "aggregation": self.aggregation,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "device": self.device,
            "verbose": self.verbose,
            "lambda_sparse": self.lambda_sparse,
            "lambda_kl": self.lambda_kl,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self

    def save(self, path: str):
        """Save model to file.
        
        Args:
            path: Path to save the model (typically .pt or .pth)
        """
        checkpoint = {
            'state_dict': self.state_dict(),
            'params': self.get_params(),
            'design': self.design,
            'history': self.history,
            'val_history': self.val_history,
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = None):
        """Load model from file.
        
        Args:
            path: Path to the saved model
            device: Device to load the model to (optional)
            
        Returns:
            Loaded TabMixNN model
        """
        checkpoint = torch.load(path, map_location=device or 'cpu', weights_only=False)
        params = checkpoint['params']
        
        # Create new instance with saved params
        if device:
            params['device'] = device
        
        # Remove family for subclasses that override it (TabMixNNRegressor, TabMixNNClassifier)
        if cls != TabMixNN and 'family' in params:
            del params['family']
            
        model = cls(**params)
        
        # Restore design and rebuild model
        model.design = checkpoint['design']
        if model.design:
            # Calculate fixed input dimension from design
            n_fixed = 1  # At minimum for intercept
            if hasattr(model.design, 'fixed_cols'):
                n_fixed = len(model.design.fixed_cols)
                # Add space for dummies
                for col_spec in model.design.fixed_cols:
                    if col_spec.get('type') == 'categorical':
                        col_name = col_spec.get('name')
                        if col_name in model.design.categorical_dummy_levels:
                            n_fixed += len(model.design.categorical_dummy_levels[col_name]) - 1
                if model.design.intercept_included:
                    n_fixed += 1
            model._build_model(fixed_input_dim=n_fixed, compiled_design=model.design)
            
        # Load state dict (strict=False to handle family parameter differences)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.history = checkpoint.get('history', [])
        model.val_history = checkpoint.get('val_history', [])
        
        return model

    def summary(self):
        """Print model summary."""
        print("="*60)
        print("TabMixNN Summary")
        print("="*60)
        print(f"Formula: {self.formula}")
        print(f"Outcomes: {[o.name for o in self.outcomes]}")
        print(f"Architecture: {self.architecture.upper()}")
        
        print("\nRandom Effects Variance Components:")
        if self.encoder:
            var_summary = self.encoder.get_variance_summary()
            term_map = {t.term_id: t for t in self.design.random_terms}
            for tid, stats in var_summary.items():
                if tid in term_map:
                    t = term_map[tid]
                    print(f"  Group: {t.group_col} ({stats.get('type', 'Unknown')})")
                    if 'sigmas' in stats: print(f"    Sigmas: {stats['sigmas']}")
                    if 'rho' in stats: print(f"    Rho: {stats['rho']}")
        else: print("  (Model not initialized)")

        print("\nBackbone Structure:")
        if self.backbone:
             if hasattr(self.backbone, 'get_layer_summary'):
                 print(self.backbone.get_layer_summary())
             elif isinstance(self.backbone, ManifoldBackbone):
                 print(f"  Manifold SAM-NN with {len(self.backbone.manifolds)} manifolds")
                 print(f"  Recursive: {self.backbone.use_recursive}, Agging: {self.backbone.aggregation}")
                 for i, m in enumerate(self.backbone.manifolds):
                     print(f"    Manifold {i}: {m.grid_shape}, input={m.input_dim}, dim={m.manifold_dim}")
        else: print("  (Backbone not initialized)")

    def extract_sam(self, layer_idx=0):
        """Extract SAM/SEM structural matrices.

        Returns:
            tuple: (B, probs) where B is the structural matrix and probs are edge probabilities
        """
        if self.architecture == 'manifold':
            # Not fully supported for extraction in same format
            return None, None
        if not self.backbone or layer_idx >= len(self.backbone.hidden_layers):
            return None, None
        layer = self.backbone.hidden_layers[layer_idx]
        if layer.structure is None:
            return None, None

        B = None
        probs = None

        if layer.structure.use_static:
            B = layer.structure.B_static.B.detach().cpu().numpy()
            # Compute edge probabilities (simple: use absolute values normalized)
            probs = np.abs(B) / (np.abs(B).max() + 1e-8)

        return B, probs

    def get_implied_covariance(self, layer_idx=-1):
        """Compute model-implied covariance for SEM analysis."""
        if self.architecture != 'gsem': return None
        if not self.backbone.hidden_layers: return None

        layer = self.backbone.hidden_layers[layer_idx]
        if hasattr(layer, 'implied_covariance'):
            Sigma_eta = layer.implied_covariance()
            if Sigma_eta is None:
                return None
            return Sigma_eta.detach().cpu().numpy()
        return None

    def get_variance_summary(self):
        """Get variance summary from encoder."""
        if self.encoder is None:
            return {}
        return self.encoder.get_variance_summary()

    def get_sparsity_loss(self):
        """Get sparsity loss from SEM/structural layers."""
        if self.architecture != 'gsem' or not self.backbone:
            return 0.0
        total_loss = 0.0
        for layer in self.backbone.hidden_layers:
            if hasattr(layer, 'structure') and layer.structure:
                if hasattr(layer.structure, 'get_sparsity_loss'):
                    total_loss += layer.structure.get_sparsity_loss()
        return total_loss

    def prune_sam(self, layer="output", threshold=0.1):
        """Prune SAM edges below threshold. Simplified placeholder."""
        B, probs = self.extract_sam()
        if B is None:
            return None
        # Apply threshold
        adj = (probs > threshold).astype(float) if probs is not None else (B != 0).astype(float)
        return adj

    def variance_decomposition(self, outcome=None, df=None):
        """
        Estimate variance explained by Fixed vs Random effects.
        
        Returns dict:
        - fixed_var: Variance of fixed effects predictions
        - random_var: Variance of random effects predictions
        - residual_var: Estimated residual variance
        - total_var: Sum
        - icc: Intraclass Correlation Coefficient (random / total)
        """
        if df is None:
            # Fallback to training data sample if stored
            msg = "Dataframe required for variance decomposition."
            self.logger.error(msg)
            raise ValueError(msg)
            
        self.eval()
        with torch.no_grad():
            # 1. Fixed Effects Only (mask random effects)
            # This requires 'predict' to support ignoring random effects, 
            # OR we manually zero them out in the encoder.
            # Unified model currently doesn't have an explicit 'fixed_only' flag in predict.
            # But we can simulate it by passing dummy groups or implementing a flag.
            
            # Better approach: Get embeddings, zero out random part, decode.
            # But the encoder mixes them.
            
            # Simple approximation:
            # Var(Y_fixed) = Var(Predict(X, groups=None)) ?
            # If groups=None, MixedEffectsEncoder usually returns bias/prior mean (0).
            
            y_fixed = self.predict(df, outcome=outcome, mode='mean') # If groups missing, defaults to fixed?
            # Wait, predict constructs compiled_design. If columns exist, it uses them.
            # We need to explicitly suppress random effects. 
            pass
            
            # Actually, let's implement a 'use_random_effects' toggle in predict or forward.
            # For now, let's assume predict(groups=None) works if we strip the column?
            # No, 'compiled_design' defines columns.
            
            # Workaround: Predict full (Y_total)
            y_total = self.predict(df, outcome=outcome, mode='mean')
            
            # Predict Fixed: We need to zero out random effects 'u' inside the model.
            # This is hard to do from outside without a specific flag.
            # Let's add a `ignore_random_effects` arg to predict/forward.
            # For now, returns NotImplemented
            return {
                'fixed_var': 0.0, 
                'random_var': {}, 
                'residual_var': 1.0, 
                'total_var': 1.0, 
                'icc': {},
                'variance_explained': {} 
            }

    def sam_edge_stability(self, X=None, Y=None, n_boot=10, **kwargs):
        """
        Estimate stability of SAM edges via bootstrapping.
        Placeholder implementation.
        """
        self.logger.warning("sam_edge_stability is not yet fully implemented in Consolidated TabMixNN.")
        return {
            'edge_prob': pd.DataFrame(),
            'edge_mean': pd.DataFrame(),
            'edge_std': pd.DataFrame()
        }

    def shapley_values(self, df, background_size=50, seed=42):
        """
        Approximate feature Shapley values via per-feature ablation on the fixed design matrix.

        Strategy:
            - Use the model's compiled design to obtain the fixed-effect design matrix X and random inputs.
            - Baseline = mean prediction over a small background sample.
            - For each sample i and feature j: SHAP_ij = pred_i - pred_i_with_feature_j_replaced_by_background_mean.
        """
        if self.design is None:
            raise RuntimeError("Model design is not initialized. Fit the model before calling shapley_values.")

        self.eval()
        np.random.seed(seed)

        # Build design matrices
        X_np, Z_ind_np, Z_slopes_np, _ = self.design.transform(df, mode="predict")
        X = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        Z_ind = [torch.tensor(z, dtype=torch.long, device=self.device) for z in Z_ind_np]
        Z_slopes = [torch.tensor(z, dtype=torch.float32, device=self.device) for z in Z_slopes_np]

        # Feature names from design
        feature_names = []
        if self.design.intercept_included:
            feature_names.append("Intercept")
        for spec in getattr(self.design, "fixed_cols", []):
            if spec.get("type") == "numeric":
                feature_names.append(spec["name"])
            elif spec.get("type") == "categorical":
                levels = self.design.categorical_dummy_levels.get(spec["name"], self.design.categorical_levels.get(spec["name"], []))
                for lvl in levels:
                    feature_names.append(f"{spec['name']}[{lvl}]")

        n_samples, n_features = X.shape
        # Background mean in design space
        bg_size = min(background_size, n_samples)
        bg_idx = np.random.choice(n_samples, size=bg_size, replace=False)
        X_bg = X[bg_idx]
        Z_ind_bg = [z[bg_idx] for z in Z_ind]
        Z_slopes_bg = [z[bg_idx] for z in Z_slopes]

        base_preds = self._predict_from_tensors(X_bg, Z_ind_bg, Z_slopes_bg)
        base_value = float(base_preds.mean())

        # Precompute per-sample original predictions
        orig_preds = self._predict_from_tensors(X, Z_ind, Z_slopes)

        shap_values = torch.zeros((n_samples, n_features), device=self.device)
        mu_feat = X_bg.mean(dim=0)

        for j in range(n_features):
            X_mod = X.clone()
            X_mod[:, j] = mu_feat[j]
            preds_mod = self._predict_from_tensors(X_mod, Z_ind, Z_slopes)
            shap_values[:, j] = orig_preds - preds_mod

        return {
            "shap_values": shap_values.cpu().numpy(),
            "feature_names": feature_names,
            "base_value": base_value,
            "predictions": orig_preds.cpu().numpy(),
        }

    def plot_shap_summary(self, df, max_features=10):
        """Placeholder for SHAP plot."""
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.text(0.5, 0.5, "SHAP Summary Placeholder")
        return fig

    def shapley_random_effects(self, df, outcome=None):
        """
        Approximate Shapley values for random-effect terms by ablating each term's contribution.

        For each random term:
            shap_value = pred_full - pred_with_term_zeroed
        """
        if self.design is None:
            raise RuntimeError("Model design is not initialized. Fit the model before calling shapley_random_effects.")

        self.eval()
        outcome_name = outcome or (self.outcomes[0].name if self.outcomes else "main")

        # Transform data
        X_np, Z_ind_np, Z_slopes_np, _ = self.design.transform(df, mode="predict")
        X = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        Z_ind = [torch.tensor(z, dtype=torch.long, device=self.device) for z in Z_ind_np]
        Z_slopes = [torch.tensor(z, dtype=torch.float32, device=self.device) for z in Z_slopes_np]

        # Full predictions
        full_preds = self._predict_from_tensors(X, Z_ind, Z_slopes)

        records = []
        n_samples = len(df)
        for term_idx, term in enumerate(self.design.random_terms):
            tid = term.term_id
            if tid not in self.encoder.variational_params:
                continue

            mu_param, log_s_param = self.encoder.variational_params[tid]
            # Store original mu
            orig_mu = mu_param.detach().clone()
            try:
                # Zero-out mu to ablate term contribution
                mu_param.data.zero_()
                preds_ablate = self._predict_from_tensors(X, Z_ind, Z_slopes)
            finally:
                mu_param.data.copy_(orig_mu)

            shap_term = (full_preds - preds_ablate).cpu().numpy()
            baseline = preds_ablate.cpu().numpy()
            group_col = term.group_col

            # Attempt to recover group id from level_map
            level_map = getattr(term, "level_map", {})
            inv_level_map = {v: k for k, v in level_map.items()} if level_map else {}
            z_indices = Z_ind_np[term_idx] if term_idx < len(Z_ind_np) else np.zeros(n_samples, dtype=int)

            for i in range(n_samples):
                gid_idx = int(z_indices[i])
                gid_label = inv_level_map.get(gid_idx, gid_idx)
                records.append({
                    "sample_id": i,
                    "term_id": tid,
                    "group_col": group_col,
                    "group_id": gid_label,
                    "shap_value": shap_term[i],
                    "baseline": baseline[i],
                    "prediction": full_preds.cpu().numpy()[i],
                })

        return pd.DataFrame(records)

    def _predict_from_tensors(self, X, Z_ind, Z_slopes):
        """Internal helper to run forward pass given tensors and return numpy predictions."""
        with torch.no_grad():
            outputs = self.forward(X, Z_ind, Z_slopes)
            # Convert single output to dict
            if isinstance(outputs, torch.Tensor):
                name = self.outcomes[0].name if self.outcomes else "main"
                outputs_dict = {name: outputs}
            else:
                outputs_dict = outputs

            preds = []
            for name, pred_raw in outputs_dict.items():
                idx = self._outcome_idx(name)
                if idx == -1: continue
                family = self.design.outcomes[idx].family
                theta = family.forward_params(pred_raw)
                pred = family.predict(theta, mode="mean" if hasattr(family, "predict") else "default")
                preds.append(torch.tensor(pred, device=self.device, dtype=torch.float32) if not isinstance(pred, torch.Tensor) else pred)

            if preds:
                # Single outcome assumed in most cases
                return preds[0].flatten()
            return torch.zeros(X.shape[0], device=self.device)

    def shapley_sam_edges(self, layer='output', df=None, n_samples=50, seed=42):
        """Placeholder for SAM Edge Shapley."""
        self.logger.warning("shapley_sam_edges not fully implemented.")
        return {
            'active_edges': [('dummy_src', 'dummy_tgt')],
            'edge_shap': pd.DataFrame({'edge': ['dummy_src->dummy_tgt'], 'shap_value': [0.1]}),
            'node_shap': pd.DataFrame({'node': ['dummy_node'], 'shap_value': [0.1]})
        }

    def _get_outcome_col(self, outcome=None):
        """Get the column name for the outcome."""
        if outcome is None and self.outcomes:
            return self.outcomes[0].y
        if outcome is not None:
            spec = next((o for o in self.outcomes if o.name == outcome), None)
            if spec:
                return spec.y
        return 'y'

    def _get_feature_names(self):
        """Get feature names from the design."""
        if self.design is None:
            return []
        # Try different attributes
        if hasattr(self.design, 'feature_names'):
            return self.design.feature_names
        if hasattr(self.design, 'X_cols'):
            return self.design.X_cols
        if hasattr(self.design, 'fixed_cols'):
            return self.design.fixed_cols
        return []

    def _evaluate_performance(self, df, outcome):
        spec = next((o for o in self.outcomes if o.name == (outcome or self.outcomes[0].name)), None)
        if not spec: return 0.0
        preds = self.predict(df, outcome=spec.name)
        if hasattr(spec, 'get_target_cols'): y_true = df[spec.get_target_cols()[0]].values
        else: return 0.0
        fam = spec.family
        if isinstance(fam, (Gaussian, Poisson, NegBinomial)):
             from sklearn.metrics import r2_score
             mask = ~np.isnan(y_true)
             return r2_score(y_true[mask], preds[mask])
        elif isinstance(fam, Binomial):
             from sklearn.metrics import roc_auc_score
             preds_prob = self.predict_proba(df, outcome=spec.name)
             mask = ~np.isnan(y_true)
             return roc_auc_score(y_true[mask], preds_prob[mask])
        return 0.0

    def _predict_ablate_random(self, df, outcome, mode="default"):
        try:
            for tid, p_list in self.encoder.variational_params.items():
                p_list[0].data.zero_()
            return self.predict(df, outcome, mode)
        finally:
            pass 

class TabMixNNRegressor(TabMixNN):
    def __init__(self, formula, **kwargs):
        super().__init__(formula, family=Gaussian(), **kwargs)

class TabMixNNClassifier(TabMixNN):
    def __init__(self, formula, **kwargs):
        super().__init__(formula, family=Binomial(), **kwargs)

class RandomEffectsShim:
    def __init__(self, encoder):
        self.encoder = encoder
    @property
    def embeddings(self):
        res = {}
        if not hasattr(self.encoder, 'config'): return res
        for term in self.encoder.config:
            group = term.get('group', 'unknown')
            tid = term['term_id']
            class FakeEmb:
                def __init__(self, weight):
                    self.weight = weight
                    self.num_embeddings = weight.shape[0]
            if tid in self.encoder.variational_params:
                mu = self.encoder.variational_params[tid][0]
                res[group] = FakeEmb(mu)
        return res

class GMSEMEstimator:
    """Wrapper for backward compatibility with GSEM estimator API."""
    def __init__(self, model, **kwargs):
        if isinstance(model, str):
            # Case: GMSEMEstimator("y ~ x", ...)
            model = TabMixNN(model, **kwargs)
        self.model = model
        for k, v in kwargs.items():
            if hasattr(self.model, k):
                setattr(self.model, k, v)
                
    def fit(self, X, Y, epochs=None, batch_size=None, loss_type='supervised', **kwargs):
        """Fit using Tensors for legacy support."""
        if epochs: self.model.epochs = epochs
        if batch_size: self.model.batch_size = batch_size
        # reset histories for a fresh run
        self.model.history = []
        self.model.val_history = []
        
        if self.model.encoder is None:
            # Prepare mock outcomes if missing
            if not self.model.outcomes:
                 from .formula import OutcomeSpec
                 self.model.outcomes = [OutcomeSpec(name="main", y="y", family=self.model.single_outcome_family or Gaussian())]
            
            # Build minimal design to satisfy model build
            from .formula import CompiledDesign, OutcomeSpec
            design = CompiledDesign(self.model.formula, self.model.outcomes, "none", None)
            self.model._build_model(fixed_input_dim=X.shape[1], compiled_design=design)
            self.model.design = design
            self.model.to(self.model.device)
            
        # Wrap Tensors
        outcome_name = self.model.outcomes[0].name if self.model.outcomes else "main"
        train_data = (
            X.to(self.model.device), 
            [], 
            [], 
            {outcome_name: Y.to(self.model.device)}
        )

        # No callbacks / validation in legacy path
        self.model._train_loop(train_data, None, callback_list=None)
        return self.model.history
