
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from types import SimpleNamespace
from typing import List, Dict, Union, Optional, Any
from tqdm import tqdm
from ..parsing.formula import CompiledDesign, OutcomeSpec
from .encoder import MixedEffectsRepresentation
from ..backbones.gsem import GSEMBackbone 
from ..backbones.manifold import ManifoldBackbone 
from .families import DiscreteTimeSurvival, DiscreteTimeCompetingRisks, Binomial, Gaussian, Poisson, NegBinomial, ContinuousSurvival, Multinomial
from ..utils.callbacks import CallbackList, History, EarlyStopping
from ..utils.logging_config import get_logger


class TabMixDataset(Dataset):
    """Streaming-friendly container for fixed/random design components."""
    def __init__(self, X, Z_ind, Z_slopes, Ys):
        self.X = X.clone().detach() if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        self.Z_ind = [z.clone().detach() if isinstance(z, torch.Tensor) else torch.tensor(z, dtype=torch.long) for z in Z_ind]
        self.Z_slopes = [z.clone().detach() if isinstance(z, torch.Tensor) else torch.tensor(z, dtype=torch.float32) for z in Z_slopes]
        self.Ys = {k: (v.clone().detach() if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32)) for k, v in Ys.items()}

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Each item returns per-sample views; DataLoader will collate to batches.
        return (
            self.X[idx],
            [z[idx] for z in self.Z_ind],
            [z[idx] for z in self.Z_slopes],
            {k: v[idx] for k, v in self.Ys.items()},
        )

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
            from ..parsing.formula import CompiledDesign, RandomEffectTerm
            # Prepare outcomes for design if single family used
            if not model.outcomes and model.single_outcome_family:
                from ..parsing.formula import OutcomeSpec
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
            from ..parsing.formula import CompiledDesign, OutcomeSpec
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
                   dynamic_structure_config={'feature_mixing': False},
                   **kwargs)
                   
        if input_dim is not None:
            from ..parsing.formula import CompiledDesign, OutcomeSpec
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
        dynamic_structure_config: Optional[Dict] = None,
        backbone_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        # Manifold / Universal specific
        manifold_configs: Optional[List[Dict]] = None,
        use_recursive: bool = True,
        aggregation: str = "concat",
        # Output Structure
        output_sem_mode: str = 'none',  # 'none', 'lavaan', 'learned'
        output_structure: Optional[str] = None,  # lavaan equations (if mode='lavaan')
        output_structure_config: Optional[Dict] = None,
        use_identity_head: bool = False,  # NEW: Use identity mapping for output head (for pure LMM)
        # training config
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        epochs: int = 20,
        lambda_sparse: float = 0.001,
        lambda_kl: float = 0.01,
        kl_auto_scale: bool = True,
        lambda_dag: float = 0.0,
        lambda_contraction: float = 0.0,
        grad_clip_norm: float = 1.0,
        loss_reduction: str = "outcome_mean",  # 'outcome_mean' (legacy) or 'sample_weighted'
        device: str = "cpu",
        verbose: bool = False,
        enforce_centering: bool = True, # Enforce zero-mean random effects (Default True for LMM exactness)
        kl_normalize: bool = False,
        **kwargs
    ):
        super().__init__()
        
        if isinstance(family, str):
            family = self._resolve_family(family)

        self.formula = formula
        if outcomes is None:
            if family is None:
                raise ValueError("Must provide either 'outcomes' or 'family'.")
            self.single_outcome_family = family
            self.outcomes = [] 
            self.family = family  # Backward compatibility alias
        else:
            # Normalize any string family specifications
            self.outcomes = []
            for o in outcomes:
                if isinstance(o.family, str):
                    o.family = self._resolve_family(o.family)
                if getattr(o, "y", None) in (None, ""):
                    # Only default 'y' if not using special survival columns
                    if getattr(o, "time", None) is None:
                        o.y = o.name
                self.outcomes.append(o)
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
        self.dynamic_structure_config = dynamic_structure_config
        self.backbone_dropout = backbone_dropout or hidden_dropout
        # Universal
        self.manifold_configs = manifold_configs
        self.use_recursive = use_recursive
        self.aggregation = aggregation
        self.output_sem_mode = output_sem_mode
        self.output_structure = output_structure
        self.output_structure_config = output_structure_config
        self.use_identity_head = use_identity_head

        # Validate output SEM configuration early
        if output_sem_mode == 'lavaan' and output_structure is None:
            raise ValueError("output_structure required for output_sem_mode='lavaan'")
        if output_sem_mode not in ['none', 'lavaan', 'learned']:
            raise ValueError(f"Invalid output_sem_mode: {output_sem_mode}. Must be 'none', 'lavaan', or 'learned'.")

        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.lambda_sparse = lambda_sparse
        self.lambda_kl = lambda_kl
        self.kl_auto_scale = kl_auto_scale
        self.lambda_dag = lambda_dag
        self.lambda_contraction = lambda_contraction
        self.grad_clip_norm = grad_clip_norm
        if loss_reduction not in ("outcome_mean", "sample_weighted"):
            raise ValueError("loss_reduction must be 'outcome_mean' or 'sample_weighted'")
        self.loss_reduction = loss_reduction
        self.device = device
        self.verbose = verbose
        self.enforce_centering = enforce_centering
        self.kl_normalize_encoder = kl_normalize

        # For pure linear mixed models (no hidden layers), default to identity head
        if not hidden_dims:
            self.use_identity_head = True
            # Disable aggressive grad clipping for pure linear mixed models to allow fast convergence
            if grad_clip_norm == 1.0:
                self.grad_clip_norm = 0.0

        # Reduce KL shrinkage for pure linear mixed models to better recover variance components
        # Only applies when kl_auto_scale is disabled and user hasn't changed the default
        if lambda_kl == 0.01 and hidden_dims == [] and not kl_auto_scale:
            self.lambda_kl = 0.001
            self.logger.debug("Pure LMM: lambda_kl reduced to 0.001 (use kl_auto_scale=True for data-driven scaling)")

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
            
        # Allow 'use_sem' alias to enable static structure in the backbone
        if 'use_sem' in kwargs and kwargs['use_sem']:
            self.use_static_structure = True
        if 'use_sem' in kwargs:
            kwargs.pop('use_sem', None)
        
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

        # For pure identity-head LMMs, fix initialization seed for reproducibility
        if self.use_identity_head:
            torch.manual_seed(0)

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
        if self.design is None:
            self.logger.info("Compiling design...")
            self.design = CompiledDesign(
                self.formula, self.outcomes, 
                sam_location="none",
                sam_dim=None,
                cov_struct_map=self.cov_struct_map
            )
            
        # Ensure design is fitted before transformation
        if not getattr(self.design, "is_fitted", False):
            self.design.fit(train_df)
            
        # 3. Transform Data
        X, Z_ind, Z_slopes, Ys = self.design.transform(train_df, mode='fit')
        
        # 3b. Auto-scale λ_KL = 1 / mean_group_size
        # This is an empirical β-VAE-style heuristic that damps the KL relative to
        # the NLL. It is NOT derived from the ELBO: the ELBO-correct weight is
        # λ_KL = 1 with the (B/N) mini-batch factor in the training loop handling
        # the scaling. The 1/n̄_g value was found empirically to work well across
        # mixed-effects benchmarks and is documented as a heuristic, not a theorem.
        if self.kl_auto_scale and Z_ind:
            group_sizes = []
            for z in Z_ind:
                z_arr = z if isinstance(z, np.ndarray) else np.asarray(z)
                unique, counts = np.unique(z_arr[z_arr > 0], return_counts=True)
                if len(counts) > 0:
                    group_sizes.extend(counts.tolist())
            if group_sizes:
                self.lambda_kl = 1.0 / float(np.mean(group_sizes))
                self.logger.info(f"kl_auto_scale: λ_KL set to {self.lambda_kl:.5f} (1/mean_group_size={np.mean(group_sizes):.1f})")

        # 4. Initialize Modules
        required_fixed_dim = X.shape[1]
        # Rebuild if encoder is missing or input dims don't match the design
        if self.encoder is None or getattr(self.encoder, "fixed_linear", None) is None or self.encoder.fixed_linear.in_features != required_fixed_dim:
            if self.encoder is not None and self.verbose:
                self.logger.info(f"Rebuilding model for fixed input dim {required_fixed_dim} (was {self.encoder.fixed_linear.in_features}).")
            self._build_model(fixed_input_dim=X.shape[1], compiled_design=self.design)
            self.to(self.device)
            
        # 5. Prepare Tensors
        train_data = TabMixDataset(X, Z_ind, Z_slopes, Ys)
        val_data = None
        if val_df is not None:
            Xv, Zv_ind, Zv_slopes, Yv = self.design.transform(val_df, mode='fit')
            val_data = TabMixDataset(Xv, Zv_ind, Zv_slopes, Yv)

        # Initialize Gaussian residual scale to match data variance for faster convergence
        try:
            if hasattr(self, "families") and isinstance(self.families, nn.ModuleDict):
                for name, family in self.families.items():
                    if isinstance(family, Gaussian):
                        if name in Ys and Ys[name].size > 0:
                            y_np = Ys[name]
                            if isinstance(y_np, torch.Tensor):
                                y_arr = y_np.detach().cpu().numpy()
                            else:
                                y_arr = np.asarray(y_np)
                            y_std = float(np.nanstd(y_arr))
                            y_mean = float(np.nanmean(y_arr))
                            
                            # Init sigma
                            if y_std > 0:
                                sigma_init = float(np.clip(y_std, 1e-3, 5.0))
                                family.log_sigma.data = torch.log(torch.tensor(sigma_init, device=self.device))
                                
                            # Init fixed bias to mean if available (helps identifiability)
                            if self.use_identity_head and self.encoder is not None and hasattr(self.encoder, "fixed_linear"):
                                if hasattr(self.encoder.fixed_linear, "bias") and self.encoder.fixed_linear.bias is not None:
                                    # Only if bias is close to zero (not pretrained)
                                    # Relaxed threshold to 2.0 to catch random initialization (Uniform(-1, 1) for dim 1)
                                    if torch.abs(self.encoder.fixed_linear.bias).mean() < 2.0:
                                        self.encoder.fixed_linear.bias.data.fill_(y_mean / (len(self.outcomes) if self.outcomes else 1))
                                        if self.verbose:
                                            self.logger.info(f"Initialized fixed intercept to {y_mean:.2f}")

        except Exception as e:
            if self.verbose: self.logger.warning(f"Init failed: {e}")
            pass
        
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
        
        # 8. Calculate model fit statistics (log-likelihood, AIC, BIC)
        self._calculate_fit_statistics(train_df)
        self._update_random_embeddings(train_df)
        self._rebalance_intercept_mean(train_df)
        
        return self
    
    def _resolve_family(self, fam: Any):
        """Convert common family name strings to Family instances."""
        if not isinstance(fam, str):
            return fam
        name = fam.lower()
        mapping = {
            "gaussian": Gaussian,
            "normal": Gaussian,
            "binomial": Binomial,
            "poisson": Poisson,
            "negbinomial": NegBinomial,
        }
        if name in mapping:
            return mapping[name]()
        return fam
        
    def predict(self, df: pd.DataFrame, outcome: str = None, mode: str = 'default', use_random_effects: bool = True, as_dict: Optional[bool] = None):
        """
        Predict outcomes.

        Args:
            outcome: Optional outcome name to select.
            mode: Family-specific prediction mode.
            use_random_effects: Toggle random effects contribution.
            as_dict: If True returns dict; if False returns first/selected output array;
                     if None uses legacy behavior (unwrap single outcome).
        """
        self.eval()
        X, Z_ind, Z_slopes, _ = self.design.transform(df, mode='predict')
        results = self._predict_batches(X, Z_ind, Z_slopes, mode=mode, use_random_effects=use_random_effects)

        if outcome:
            return results.get(outcome)
        unwrap = as_dict is False or (as_dict is None and len(results) == 1)
        if unwrap and len(results) == 1:
            return list(results.values())[0]
        return results

    def _predict_batches(self, X, Z_ind, Z_slopes, mode="default", use_random_effects: bool = True):
        dataset = TabMixDataset(X, Z_ind, Z_slopes, {})
        loader, _ = self._make_dataloader(dataset, shuffle=False)
        buffers: Dict[str, List[torch.Tensor]] = {}
        with torch.no_grad():
            for batch in loader:
                bx, bz_ind, bz_slopes, _ = self._move_batch_to_device(batch)
                outputs = self.forward(bx, bz_ind, bz_slopes, use_random_effects=use_random_effects)
                outputs_dict = self._ensure_dict_outputs(outputs)
                for name, pred_raw in outputs_dict.items():
                    idx = self._outcome_idx(name)
                    if idx == -1:
                        continue
                    family = self.design.outcomes[idx].family
                    theta = family.forward_params(pred_raw)
                    pred = family.predict(theta, mode=mode)
                    if not torch.is_tensor(pred):
                        pred = torch.as_tensor(pred, device=self.device, dtype=torch.float32)
                    buffers.setdefault(name, []).append(pred.detach().cpu().reshape(-1))

        results = {}
        for name, parts in buffers.items():
            if not parts:
                continue
            results[name] = torch.cat(parts, dim=0).numpy()
        return results

    def predict_proba(self, df, outcome=None, as_dict: Optional[bool] = None):
        return self.predict(df, outcome=outcome, mode='proba', as_dict=as_dict)

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
             kl_normalize=self.kl_normalize_encoder,
             scale_random_by_terms=self.scale_random_by_terms,
             unknown_strategy=self.unknown_strategy,
             use_fixed_bias=not getattr(compiled_design, "intercept_included", False),
             enforce_centering=self.enforce_centering
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
                dynamic_structure_config=self.dynamic_structure_config,
                dropout=self.backbone_dropout,
            )
            final_dim = self.hidden_dims[-1] if self.hidden_dims else encoder_output_dim
        
        # C. Output Heads
        self.heads = nn.ModuleDict()
        self.families = nn.ModuleDict()
        
        # Identity layer for pure linear mixed models
        class IdentityHead(nn.Module):
            """Identity mapping that passes through the first output_dim channels."""
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                # Dummy parameters keep optimizer construction/serialization happy
                self.weight = nn.Parameter(torch.ones(output_dim, input_dim), requires_grad=False)
                self.bias = nn.Parameter(torch.zeros(output_dim), requires_grad=False)
                
            def forward(self, x):
                # Slice the encoded representation to the required output dims
                return x[..., : self.output_dim]
        
        for o in self.outcomes:
            if self.use_identity_head and o.family.n_params == 1:
                # Use identity head for pure linear mixed models regardless of encoder width
                self.heads[o.name] = IdentityHead(final_dim, o.family.n_params)
            else:
                # Standard linear head
                self.heads[o.name] = nn.Linear(final_dim, o.family.n_params)
            self.families[o.name] = o.family
             
        # D. Output SEM (Three modes: none, lavaan, learned)
        self.output_sem = None
        if self.output_sem_mode != 'none' and len(self.outcomes) > 1:
            from ..layers.structural import StaticStructureModel
            sem_dim = len(self.outcomes)

            if self.output_sem_mode == 'lavaan':
                # Parse lavaan equations and create fixed structure
                # (validation already done in __init__)
                from ..parsing.lavaan_parser import LavanParser
                outcome_names = [o.name for o in self.outcomes]
                spec = LavanParser.parse(self.output_structure)
                init_B = LavanParser.to_adjacency(spec, outcome_names)

                # Fixed structure: B is non-learnable
                config = self.output_structure_config or {}
                self.output_sem = StaticStructureModel(
                    sem_dim,
                    init_structure=init_B,
                    learnable=False,
                    **config
                )
            elif self.output_sem_mode == 'learned':
                # Learned structure: B is fully learnable
                config = self.output_structure_config or {}
                self.output_sem = StaticStructureModel(
                    sem_dim,
                    learnable=True,
                    **config
                )
            # Note: invalid modes already caught in __init__

        # Backward compatibility: old API with output_structure_config only
        elif self.output_structure_config and len(self.outcomes) > 1:
            from ..layers.structural import StaticStructureModel
            sem_dim = len(self.outcomes)
            self.output_sem = StaticStructureModel(sem_dim, **self.output_structure_config)
            self.output_sem_mode = 'learned'  # Infer mode for backward compatibility
        
        # Separate parameter groups: variational RE params and covariance params must
        # NOT receive L2 weight decay — that would add a second conflicting prior on
        # top of the KL term and bias random effects toward zero beyond what the ELBO
        # intends. Only backbone / fixed-effect parameters get weight_decay.
        re_param_ids = set()
        if self.encoder is not None:
            for p in self.encoder.variational_params.parameters():
                re_param_ids.add(id(p))
            for p in self.encoder.cov_modules.parameters():
                re_param_ids.add(id(p))
            if hasattr(self.encoder, 'unknown_params'):
                for p in self.encoder.unknown_params.parameters():
                    re_param_ids.add(id(p))
        re_params = [p for p in self.parameters() if id(p) in re_param_ids]
        other_params = [p for p in self.parameters() if id(p) not in re_param_ids]
        self.optimizer = optim.Adam(
            [{'params': other_params, 'weight_decay': self.weight_decay},
             {'params': re_params,    'weight_decay': 0.0}],
            lr=self.lr,
        )

    def _ensure_dict_outputs(self, outputs):
        """Normalize forward outputs to a dict keyed by outcome name."""
        if isinstance(outputs, torch.Tensor):
            name = self.outcomes[0].name if self.outcomes else "main"
            return {name: outputs}
        return outputs

    def _make_dataloader(self, data, shuffle=False):
        """Create a DataLoader from a dataset/tuple while keeping tensors on CPU."""
        if isinstance(data, DataLoader):
            return data, len(data.dataset)
        if isinstance(data, TabMixDataset):
            dataset = data
        elif isinstance(data, tuple):
            dataset = TabMixDataset(*data)
        else:
            dataset = TabMixDataset(*data)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            pin_memory=self.device.startswith("cuda"),
        )
        return loader, len(dataset)

    def _move_batch_to_device(self, batch):
        """Send a collated batch to the configured device."""
        X, Z_ind, Z_slopes, Ys = batch
        X = X.to(self.device)
        Z_ind = [z.to(self.device) for z in Z_ind]
        Z_slopes = [z.to(self.device) for z in Z_slopes]
        Ys = {k: v.to(self.device) for k, v in Ys.items()}
        return X, Z_ind, Z_slopes, Ys

    def forward(self, X, Z_ind=None, Z_slopes=None, return_attention=False, use_random_effects: bool = True):
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
        h_encoded, kl_encoder = self.encoder(X, Z_ind, Z_slopes, use_random_effects=use_random_effects)
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

        # 4. Output SEM (Structure among tasks)
        # Split primary and auxiliary parameters for multi-parameter families
        if getattr(self, 'output_sem', None) is not None and len(self.heads) > 1:
            primary_scores = []  # [batch, 1] per outcome
            auxiliary_params = {}  # name -> [batch, p_k-1] tensor
            head_names = []

            for name, head in self.heads.items():
                raw_output = head(h_last)  # [batch, n_params]

                # Extract primary parameter (always first column by convention)
                primary = raw_output[:, 0:1]  # [batch, 1]
                primary_scores.append(primary)
                head_names.append(name)

                # Store auxiliary if multi-parameter
                if self.families[name].n_params > 1:
                    auxiliary_params[name] = raw_output[:, 1:]  # [batch, p_k-1]

            # Apply SEM to primary parameters: μ = (I - B)^{-1} η
            stacked_primary = torch.cat(primary_scores, dim=1)  # [batch, K]
            correlated = self.output_sem.solve(stacked_primary)  # [batch, K]

            # Reconstruct full parameter vectors
            for i, name in enumerate(head_names):
                μ_correlated = correlated[:, i:i+1]  # [batch, 1]

                if name in auxiliary_params:
                    # Multi-param: combine correlated primary + independent auxiliary
                    outputs[name] = torch.cat([μ_correlated, auxiliary_params[name]], dim=1)
                else:
                    # Scalar: just correlated primary
                    outputs[name] = μ_correlated
        else:
            # No SEM: standard pass-through
            for name, head in self.heads.items():
                outputs[name] = head(h_last)
            
        self.current_kl = kl_encoder
            
        # Legacy return if return_attention
        if return_attention:
            # Return (outputs, attn) to match legacy tests
            if len(outputs) == 1:
                return list(outputs.values())[0], attn
            return outputs, attn

        # RETURN TYPE BEHAVIOR (for backward compatibility):
        #   - Single outcome: Returns tensor (shape varies by family)
        #   - Multiple outcomes: Returns dict {outcome_name: tensor}
        #
        # MIGRATION PATH (for future major version):
        #   Consider always returning dict for consistency:
        #   - Pro: Uniform API, easier to extend, explicit outcome names
        #   - Con: Breaking change, requires updating all downstream code
        #   - Alternative: Add `always_dict=False` parameter for opt-in migration
        #
        # Current implementation maintains backward compatibility with existing tests/examples.
        if len(outputs) == 1:
            return list(outputs.values())[0]
        return outputs

    def _train_loop(self, train_data, val_data, callback_list: Optional[CallbackList] = None):
        train_loader, n_samples = self._make_dataloader(train_data, shuffle=True)
        val_loader = None
        if val_data:
            val_loader, _ = self._make_dataloader(val_data, shuffle=False)

        self.stop_training = False
        callbacks = callback_list or CallbackList()

        # Initialize progress bar if verbose
        if self.verbose:
            pbar = tqdm(range(self.epochs), desc="Training")
            epoch_iterator = pbar
        else:
            pbar = None
            epoch_iterator = range(self.epochs)

        callbacks.on_train_begin({"epochs": self.epochs, "batches": len(train_loader)})

        for epoch in epoch_iterator:
            self.train()
            epoch_loss = 0

            callbacks.on_epoch_begin(epoch, {"epoch": epoch})

            for i, batch in enumerate(train_loader):
                callbacks.on_batch_begin(i, {"batch": i})

                bx, bz_ind, bz_slopes, bys = self._move_batch_to_device(batch)

                self.optimizer.zero_grad()
                outputs = self.forward(bx, bz_ind, bz_slopes)
                outputs_dict = self._ensure_dict_outputs(outputs)
                kl = self.current_kl

                loss = torch.tensor(0.0, device=bx.device)
                total_weight = torch.tensor(0.0, device=bx.device)
                for name, target in bys.items():
                    family = self.data_family(name)
                    mask = ~torch.isnan(target).any(dim=1) if target.ndim > 1 else ~torch.isnan(target)
                    n_obs = mask.sum()
                    if n_obs == 0:
                        continue
                    pred = outputs_dict[name][mask]
                    tgt = target[mask]
                    theta = family.forward_params(pred)
                    outcome_spec = self._outcome_spec(name)
                    weight = torch.tensor(
                        getattr(outcome_spec, "loss_weight", 1.0) if outcome_spec else 1.0,
                        device=bx.device,
                        dtype=torch.float32,
                    )
                    nll_val = family.nll(theta, tgt)
                    if self.loss_reduction == "sample_weighted":
                        loss = loss + nll_val * n_obs * weight
                        total_weight = total_weight + n_obs * weight
                    else:
                        loss = loss + nll_val * weight
                        total_weight = total_weight + weight

                if total_weight.item() > 0:
                    loss = loss / total_weight

                if kl is not None:
                    loss = loss + self.lambda_kl * kl * (len(bx) / max(n_samples, 1))

                loss = loss + self._structural_penalties()

                loss.backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_norm)
                self.optimizer.step()
                epoch_loss += loss.item()
                callbacks.on_batch_end(i, {"batch": i, "batch_loss": loss.item()})

            train_avg = epoch_loss / max(len(train_loader), 1)
            self.history.append(train_avg)

            val_loss = None
            if val_loader is not None:
                val_loss = self._evaluate_loss(val_loader)
                self.val_history.append(val_loss)
                callbacks.on_validation_end({"val_loss": val_loss})

            if self.verbose and pbar is not None:
                msg = f"loss={train_avg:.4f}"
                if val_loss is not None:
                    msg += f", val_loss={val_loss:.4f}"
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
        """Evaluate average loss over a dataset/loader without moving entire data to device."""
        loader, _ = self._make_dataloader(data, shuffle=False)
        self.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in loader:
                bx, bz_ind, bz_slopes, Ys = self._move_batch_to_device(batch)
                outputs = self.forward(bx, bz_ind, bz_slopes)
                outputs_dict = self._ensure_dict_outputs(outputs)

                loss = torch.tensor(0.0, device=bx.device)
                total_weight = torch.tensor(0.0, device=bx.device)
                for name, target in Ys.items():
                    family = self.data_family(name)
                    mask = ~torch.isnan(target).any(dim=1) if target.ndim > 1 else ~torch.isnan(target)
                    n_obs = mask.sum()
                    if n_obs == 0:
                        continue
                    pred = outputs_dict[name][mask]
                    tgt = target[mask]
                    theta = family.forward_params(pred)
                    outcome_spec = self._outcome_spec(name)
                    weight = torch.tensor(
                        getattr(outcome_spec, "loss_weight", 1.0) if outcome_spec else 1.0,
                        device=bx.device,
                        dtype=torch.float32,
                    )
                    nll_val = family.nll(theta, tgt)
                    if self.loss_reduction == "sample_weighted":
                        loss = loss + nll_val * n_obs * weight
                        total_weight = total_weight + n_obs * weight
                    else:
                        loss = loss + nll_val * weight
                        total_weight = total_weight + weight
                if total_weight.item() > 0:
                    loss = loss / total_weight
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def _structural_penalties(self):
        # We must ensure this returns a tensor that requires grad if any of its components do,
        # otherwise we might get "element 0 of tensors does not require grad".
        device = self.device if hasattr(self, 'device') else next(self.parameters()).device
        total_loss = torch.tensor(0.0, device=device)
        
        if self.architecture == 'gsem':
            for layer in self.backbone.hidden_layers:
                 pen = layer.penalties()
                 for k, v in pen.items():
                     if not isinstance(v, torch.Tensor): continue
                     
                     if 'kl' in k: total_loss = total_loss + self.lambda_kl * v
                     elif 'dag' in k: total_loss = total_loss + self.lambda_dag * v
                     elif 'contraction' in k: total_loss = total_loss + self.lambda_contraction * v
                     else: total_loss = total_loss + self.lambda_sparse * v
        
        # Output SEM penalties
        if getattr(self, 'output_sem', None) is not None:
             pen = {
                 'dag': self.output_sem.dag_penalty(),
                 'sparsity': self.output_sem.sparsity_penalty()
             }
             for k, v in pen.items():
                 if 'dag' in k: total_loss = total_loss + self.lambda_dag * v
                 else: total_loss = total_loss + self.lambda_sparse * v
        
        # If no penalties found, return something that might require grad if we need it
        # But if total_loss is still 0.0, it's fine as long as the main NLL loss requires grad.
        return total_loss

    def data_family(self, name):
        """Get family logic."""
        idx = self._outcome_idx(name)
        return self.design.outcomes[idx].family

    def _to_tensors(self, X, Z_ind, Z_slopes, Ys, device=None):
        device = device or self.device
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        Zit = [torch.tensor(z, dtype=torch.long, device=device) for z in Z_ind]
        Zst = [torch.tensor(z, dtype=torch.float32, device=device) for z in Z_slopes]
        Yst = {k: torch.tensor(v, dtype=torch.float32, device=device) for k,v in Ys.items()}
        return Xt, Zit, Zst, Yst

    def _outcome_idx(self, name):
        for i, o in enumerate(self.outcomes):
            if o.name == name: return i
        return -1

    def _outcome_spec(self, name):
        for o in self.outcomes:
            if o.name == name:
                return o
        return None

    def _calculate_fit_statistics(self, df):
        """
        Calculate model fit statistics: log-likelihood, AIC, BIC, R².
        Called at the end of training to store these metrics.
        """
        try:
            import numpy as np

            self.logLik = self.AIC = self.BIC = self.R2 = self.n_params = None
            if not (self.design and getattr(self.design, "outcomes", None)):
                return

            # Prepare streaming loader
            X, Z_ind, Z_slopes, Ys = self.design.transform(df, mode='predict')
            data_loader, _ = self._make_dataloader(TabMixDataset(X, Z_ind, Z_slopes, Ys), shuffle=False)

            total_loglik = 0.0
            total_obs = 0
            gaussian_stats = {"sum_y": 0.0, "sum_y2": 0.0, "ss_res": 0.0, "n": 0}

            self.eval()
            with torch.no_grad():
                for batch in data_loader:
                    bx, bz_ind, bz_slopes, Ybatch = self._move_batch_to_device(batch)
                    outputs = self.forward(bx, bz_ind, bz_slopes)
                    outputs_dict = self._ensure_dict_outputs(outputs)

                    for name, target in Ybatch.items():
                        family = self.data_family(name)
                        if target.ndim > 1:
                            mask = ~torch.isnan(target).any(dim=1)
                        else:
                            mask = ~torch.isnan(target)
                        n_obs = int(mask.sum().item())
                        if n_obs == 0:
                            continue

                        pred = outputs_dict[name][mask]
                        tgt = target[mask]
                        theta = family.forward_params(pred)
                        nll = family.nll(theta, tgt)
                        total_loglik -= float(nll) * n_obs
                        total_obs += n_obs

                        # R² only for Gaussian outcomes
                        if isinstance(family, Gaussian):
                            mu = theta.detach().cpu().numpy() if torch.is_tensor(theta) else np.asarray(theta)
                            y_true = tgt.detach().cpu().numpy().squeeze()
                            gaussian_stats["n"] += n_obs
                            gaussian_stats["sum_y"] += float(y_true.sum())
                            gaussian_stats["sum_y2"] += float((y_true ** 2).sum())
                            gaussian_stats["ss_res"] += float(np.sum((y_true - mu) ** 2))

            if total_obs == 0:
                return

            n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            aic = -2 * total_loglik + 2 * n_params
            bic = -2 * total_loglik + n_params * np.log(max(total_obs, 1))

            self.logLik = total_loglik
            self.AIC = aic
            self.BIC = bic
            self.n_params = n_params

            if gaussian_stats["n"] > 0:
                mean_y = gaussian_stats["sum_y"] / gaussian_stats["n"]
                ss_tot = gaussian_stats["sum_y2"] - gaussian_stats["n"] * (mean_y ** 2)
                if ss_tot > 0:
                    self.R2 = 1 - (gaussian_stats["ss_res"] / ss_tot)

        except Exception:
            # If calculation fails, just skip
            pass

    def _update_random_embeddings(self, df: pd.DataFrame):
        """
        Populate legacy random_intercept_embeddings_* attributes using current variational
        parameters. This should not mutate trained parameters (read-only view).
        """
        if self.encoder is None or not hasattr(self.encoder, "variational_params"):
            return

        # Clear old random_intercept_embeddings_* attributes to prevent memory leak on multiple fit() calls
        attrs_to_remove = [attr for attr in dir(self.encoder) if attr.startswith("random_intercept_embeddings_")]
        for attr in attrs_to_remove:
            try:
                delattr(self.encoder, attr)
            except AttributeError:
                pass

        term_group_map = {}
        if self.design and getattr(self.design, "random_terms", None):
            term_group_map = {t.term_id: t.group_col for t in self.design.random_terms}

        cov_modules = dict(getattr(self.encoder, "cov_modules", {}))
        wrappers = []
        
        for term_cfg in getattr(self.encoder, "config", []):
            tid = term_cfg.get("term_id")
            if tid is None or tid not in self.encoder.variational_params:
                continue
            mu_param, _ = self.encoder.variational_params[tid]
            
            # Apply covariance structure to get u = L v
            if tid in cov_modules:
                with torch.no_grad():
                    # cov_module expects [levels, slopes, emb]
                    effect = cov_modules[tid](mu_param.unsqueeze(0) if mu_param.ndim == 2 else mu_param)
            else:
                effect = mu_param
                
            effect = effect.detach().clone()
            
            group = term_group_map.get(tid, term_cfg.get("group", "unknown"))
            wrapper = SimpleNamespace(weight=effect)
            setattr(self.encoder, f"random_intercept_embeddings_{group}", wrapper)
            wrappers.append(wrapper)
        
        if wrappers:
            # Default alias for the first term
            self.encoder.random_intercept_embeddings = wrappers[0]

        # Mean-correct bias for simple identity-head models to better match training data average
        try:
            if self.use_identity_head and self.encoder is not None and hasattr(self.encoder, "fixed_linear"):
                bias_param = getattr(self.encoder.fixed_linear, "bias", None)
                if bias_param is not None and self.design is not None and self._train_data is not None:
                    if self.embedding_dim != 1 or not self.outcomes or len(self.outcomes) != 1:
                        return
                    X, Z_ind, Z_slopes, Ys = self.design.transform(self._train_data, mode='fit')
                    Xt = torch.tensor(X, dtype=torch.float32)
                    Zit = [torch.tensor(z, dtype=torch.long) for z in Z_ind]
                    Zst = [torch.tensor(z, dtype=torch.float32) for z in Z_slopes]
                    with torch.no_grad():
                        pred = self.forward(Xt, Zit, Zst)
                        y_true = Ys.get(self.outcomes[0].name) if self.outcomes else None
                        if y_true is not None:
                            y_true_t = torch.tensor(y_true, dtype=torch.float32)
                            delta = y_true_t.mean() - pred.mean()
                            if abs(delta.item()) > 1.0:
                                self.logger.warning(
                                    f"_update_random_embeddings: intercept bias correction |delta|={abs(delta.item()):.2f} > 1.0. "
                                    "This suggests the model failed to learn the intercept during training. "
                                    "Consider increasing epochs or lr."
                                )
                            # Clamp to avoid exploding bias shifts
                            delta_clamped = torch.clamp(delta, -100.0, 100.0)
                            bias_param.data = bias_param.data + delta_clamped
        except Exception:
            pass

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
            'design_state': self.design.export_state() if self.design else None,
            'history': self.history,
            'val_history': self.val_history,
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Model saved to {path}")

    def _rebalance_intercept_mean(self, df: pd.DataFrame):
        """
        For simple identity-head models, nudge the fixed bias so that the
        predicted mean matches the empirical mean of the training outcome.
        This is a post-hoc correction that preserves fitted random effects.
        """
        try:
            if not (self.use_identity_head and getattr(self, "hidden_dims", None) == []):
                return
            if self.embedding_dim != 1 or not self.outcomes or len(self.outcomes) != 1:
                return
            if self.encoder is None or not hasattr(self.encoder, "fixed_linear"):
                return
            bias_param = getattr(self.encoder.fixed_linear, "bias", None)
            if bias_param is None:
                return
            X, Z_ind, Z_slopes, Ys = self.design.transform(df, mode='fit')
            y_true = Ys.get(self.outcomes[0].name)
            if y_true is None:
                return
            Xt = torch.tensor(X, dtype=torch.float32)
            Zit = [torch.tensor(z, dtype=torch.long) for z in Z_ind]
            Zst = [torch.tensor(z, dtype=torch.float32) for z in Z_slopes]
            with torch.no_grad():
                pred = self.forward(Xt, Zit, Zst)
                delta = torch.tensor(y_true, dtype=torch.float32).mean() - pred.mean()
                delta_clamped = torch.clamp(delta, -100.0, 100.0)
                bias_param.data = bias_param.data + delta_clamped
        except Exception:
            pass

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
        design_state = checkpoint.get('design_state')
        model.design = None
        if design_state:
            from ..parsing.formula import CompiledDesign
            model.design = CompiledDesign.restore_from_state(design_state)
        elif checkpoint.get('design'):
            # Backward compatibility for old checkpoints
            model.design = checkpoint.get('design')

        if model.design:
            if getattr(model.design, "design_info", None) is not None:
                n_fixed = len(getattr(model.design.design_info, "column_names", []))
            else:
                n_fixed = 1  # At minimum for intercept
                if hasattr(model.design, 'fixed_cols'):
                    n_fixed = len(model.design.fixed_cols)
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

        # Convergence diagnostics for HybridStructuralModel
        print("\nConvergence Diagnostics:")
        convergence_failures = self._get_convergence_failures()
        if convergence_failures:
            print("  WARNING: Fixed-point iteration convergence issues detected:")
            for layer_name, failures in convergence_failures.items():
                print(f"    {layer_name}: {failures} convergence failures")
            print("  Consider: (1) CompositeStructuralModel, (2) increase contraction penalty, (3) reduce complexity")
        else:
            print("  No convergence issues detected")

    def extract_parameters(self, data=None, sample_size: int = 256, linearize: bool = True):
        """Return interpretable parameter estimates (fixed, random, variance components, structure)."""
        from ..utils.summary import parameter_estimates
        return parameter_estimates(self, data=data, sample_size=sample_size, linearize=linearize)

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

    def _get_convergence_failures(self):
        """
        Collect convergence failure counts from HybridStructuralModel layers.

        Returns:
            dict: {layer_name: failure_count} for layers with convergence issues
        """
        failures = {}

        # Check backbone layers (GSEM architecture)
        if self.backbone and hasattr(self.backbone, 'hidden_layers'):
            for i, layer in enumerate(self.backbone.hidden_layers):
                if hasattr(layer, 'structure') and layer.structure is not None:
                    if hasattr(layer.structure, '_convergence_failures'):
                        count = layer.structure._convergence_failures
                        if count > 0:
                            failures[f"backbone.layer_{i}.structure"] = count

        # Check backbone manifolds (Manifold architecture)
        if self.backbone and hasattr(self.backbone, 'manifolds'):
            for i, manifold in enumerate(self.backbone.manifolds):
                if hasattr(manifold, 'sem_layer') and manifold.sem_layer is not None:
                    if hasattr(manifold.sem_layer, '_convergence_failures'):
                        count = manifold.sem_layer._convergence_failures
                        if count > 0:
                            failures[f"backbone.manifold_{i}.sem"] = count

        return failures

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
        if self.design is None:
            raise RuntimeError("Model design is not initialized. Fit the model before calling variance_decomposition.")

        if df is None and self._train_data is not None:
            df = self._train_data
        if df is None:
            msg = "Dataframe required for variance decomposition."
            self.logger.error(msg)
            raise ValueError(msg)

        self.eval()
        with torch.no_grad():
            outcome_col = self._get_outcome_col(outcome)
            y_true = df[outcome_col].values

            # Full prediction
            y_total = self.predict(df, outcome=outcome, mode="default", use_random_effects=True, as_dict=False)
            if isinstance(y_total, torch.Tensor):
                y_total = y_total.detach().cpu().numpy()

            # Fixed-only prediction (disable random effects cleanly)
            y_fixed = self.predict(df, outcome=outcome, mode="default", use_random_effects=False, as_dict=False)
            if isinstance(y_fixed, torch.Tensor):
                y_fixed = y_fixed.detach().cpu().numpy()

            total_var = float(np.nanvar(y_total))
            fixed_var = float(np.nanvar(y_fixed))
            random_var_total = max(total_var - fixed_var, 0.0)
            residual_var = float(np.nanvar(y_true - y_total))

            # Per-term ICC via single-term ablations
            icc = {}
            random_var = {}
            if getattr(self.design, "random_terms", None):
                for term in self.design.random_terms:
                    term_id = term.term_id
                    y_no_term = self._predict_single_term_ablation(df, outcome, term_id)
                    if isinstance(y_no_term, torch.Tensor):
                        y_no_term = y_no_term.detach().cpu().numpy()
                    term_contrib = y_total - y_no_term
                    var_term = float(np.nanvar(term_contrib))
                    random_var[term_id] = var_term
                    icc[term_id] = 0.0 if total_var == 0 else max(var_term, 0.0) / total_var

            variance_explained = {
                "fixed": 0.0 if total_var == 0 else fixed_var / total_var,
                "random": 0.0 if total_var == 0 else random_var_total / total_var,
                "residual": 0.0 if total_var == 0 else residual_var / total_var,
            }

            return {
                "fixed_var": fixed_var,
                "random_var": random_var,
                "residual_var": residual_var,
                "total_var": total_var,
                "icc": icc,
                "variance_explained": variance_explained,
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
                # Get the actual dummy levels used
                col_name = spec["name"]
                if col_name in self.design.categorical_dummy_levels:
                    levels = self.design.categorical_dummy_levels[col_name]
                elif col_name in self.design.categorical_levels:
                    levels = self.design.categorical_levels[col_name]
                else:
                    levels = []
                
                # Add dummy variable names
                for lvl in levels:
                    feature_names.append(f"{col_name}[{lvl}]")

        n_samples, n_features = X.shape
        
        # Ensure feature_names matches X dimensions
        # This handles cases where categorical encoding creates more/fewer columns than expected
        if len(feature_names) != n_features:
            if len(feature_names) < n_features:
                # Pad with generic names
                for i in range(len(feature_names), n_features):
                    feature_names.append(f"Feature_{i}")
            else:
                # Truncate (shouldn't happen, but handle it)
                feature_names = feature_names[:n_features]
        
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

    def _predict_from_tensors(self, X, Z_ind, Z_slopes, use_random_effects: bool = True):
        """Internal helper to run forward pass given tensors and return 1D predictions for SHAP.
        
        For survival outcomes, returns Expected Time Lost (ETL) = area under incidence curve.
        """
        with torch.no_grad():
            outputs = self.forward(X, Z_ind, Z_slopes, use_random_effects=use_random_effects)
            outputs_dict = self._ensure_dict_outputs(outputs)

            preds = []
            for name, pred_raw in outputs_dict.items():
                idx = self._outcome_idx(name)
                if idx == -1: continue
                family = self.design.outcomes[idx].family
                theta = family.forward_params(pred_raw)

                # Choose prediction mode appropriate to the family
                if isinstance(family, (DiscreteTimeSurvival, DiscreteTimeCompetingRisks, ContinuousSurvival)):
                    predict_mode = "survival"
                elif isinstance(family, (Binomial, Multinomial)):
                    predict_mode = "proba"
                else:
                    predict_mode = "default"

                pred = family.predict(theta, mode=predict_mode)
                
                # Force conversion to tensor
                if not torch.is_tensor(pred):
                    pred = torch.as_tensor(pred, device=self.device, dtype=torch.float32)
                
                # Check for flattened arrays (e.g. 1100 elements for 100 samples)
                if pred.ndim == 1 and pred.shape[0] > X.shape[0]:
                    # Attempt to reshape based on X.shape[0] (n_samples)
                    n_samples = X.shape[0]
                    if pred.shape[0] % n_samples == 0:
                        n_bins = pred.shape[0] // n_samples
                        pred = pred.view(n_samples, n_bins)
                        # Now it has 2 dims, fall through to ETL logic
                
                # For survival families (2D output: samples x time_bins)
                # Compute ETL = area under incidence curve = sum((1 - S(t)))
                if pred.ndim > 1:
                    if pred.shape[0] == X.shape[0]:
                        # pred is survival curve, compute incidence
                        incidence = 1.0 - pred
                        # ETL = area under incidence curve (sum across time bins)
                        etl = incidence.sum(dim=1)  # Higher ETL = worse outcome = higher risk
                        pred = etl
                    else:
                        self.logger.debug(f"Shape mismatch in SHAP predict! X:{X.shape}, pred:{pred.shape}. Using mean aggregation as fallback.")
                        # Fallback: flatten?
                        pred = pred.mean(dim=1)
                
                preds.append(pred)

            if preds:
                # Returns 1D tensor (n_samples,)
                res = preds[0]
                if res.ndim > 1:
                    res = res.flatten()
                return res
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
        if hasattr(self.design, 'fixed_feature_names') and self.design.fixed_feature_names:
            return self.design.fixed_feature_names
        if getattr(self.design, "design_info", None) is not None:
            return list(self.design.design_info.column_names)
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
        preds = self.predict(df, outcome=spec.name, as_dict=False)
        if hasattr(spec, 'get_target_cols'): y_true = df[spec.get_target_cols()[0]].values
        else: return 0.0
        fam = spec.family
        if isinstance(fam, (Gaussian, Poisson, NegBinomial)):
             from sklearn.metrics import r2_score
             mask = ~np.isnan(y_true)
             return r2_score(y_true[mask], preds[mask])
        elif isinstance(fam, Binomial):
             from sklearn.metrics import roc_auc_score
             preds_prob = self.predict_proba(df, outcome=spec.name, as_dict=False)
             mask = ~np.isnan(y_true)
             return roc_auc_score(y_true[mask], preds_prob[mask])
        return 0.0

    def _predict_ablate_random(self, df, outcome, mode="default"):
        if self.encoder is None:
            return self.predict(df, outcome, mode, as_dict=False)
        # Temporarily zero random effects but restore afterwards
        backup = {}
        try:
            for tid, p_list in self.encoder.variational_params.items():
                mu_param = p_list[0]
                backup[tid] = mu_param.data.clone()
                mu_param.data.zero_()
            return self.predict(df, outcome, mode, as_dict=False)
        finally:
            for tid, mu_val in backup.items():
                if tid in self.encoder.variational_params:
                    self.encoder.variational_params[tid][0].data.copy_(mu_val)

    def _predict_single_term_ablation(self, df, outcome, term_id, mode="default"):
        """
        Predict while ablating a single random effect term.
        """
        if self.encoder is None or term_id not in self.encoder.variational_params:
            return self.predict(df, outcome, mode, as_dict=False)
        with torch.no_grad():
            mu_param = self.encoder.variational_params[term_id][0]
            backup = mu_param.data.clone()
            try:
                mu_param.data.zero_()
                return self.predict(df, outcome, mode, as_dict=False)
            finally:
                mu_param.data.copy_(backup)

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
                 from ..parsing.formula import OutcomeSpec
                 self.model.outcomes = [OutcomeSpec(name="main", y="y", family=self.model.single_outcome_family or Gaussian())]
            
            # Build minimal design to satisfy model build
            from ..parsing.formula import CompiledDesign, OutcomeSpec
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

    def _calculate_fit_statistics(self, df):
        """
        Calculate model fit statistics: log-likelihood, AIC, BIC, R².
        Called at the end of training to store these metrics.
        """
        try:
            if hasattr(self, "model") and hasattr(self.model, "_calculate_fit_statistics"):
                self.model._calculate_fit_statistics(df)
                for attr in ["logLik", "AIC", "BIC", "R2", "n_params"]:
                    if hasattr(self.model, attr):
                        setattr(self, attr, getattr(self.model, attr))
        except Exception:
            # If calculation fails, just skip
            pass
