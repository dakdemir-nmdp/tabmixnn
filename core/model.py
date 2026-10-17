
import copy
import warnings
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
        kwargs.setdefault("hidden_dims", [])
        kwargs.setdefault("embedding_dim", 1)
        kwargs.setdefault("weight_decay", 0.0)
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
                    term.level_map = {j: j+1 for j in range(n)}
                    term.train_levels = list(range(n))
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
        unknown_strategy: str = "zero",
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
        epochs: int = 200,
        lambda_sparse: float = 0.001,
        lambda_kl: float = 1.0,
        kl_auto_scale: bool = False,
        lambda_contraction: float = 0.0,
        grad_clip_norm: float = 1.0,
        loss_reduction: str = "elbo",  # observed likelihood sum per row
        device: str = "cpu",
        verbose: bool = False,
        enforce_centering: bool = False,  # True defines a projected prior, not classical LMM
        kl_normalize: bool = False,
        random_state: Optional[int] = None,
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

        # Keep constructor family states so repeated fit() calls start afresh.
        initial_families = ({'main': self.single_outcome_family}
                            if self.single_outcome_family is not None
                            else {o.name: o.family for o in self.outcomes})
        self._initial_family_states = {
            name: copy.deepcopy(family.state_dict())
            for name, family in initial_families.items()
        }
            
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

        if "learning_rate" in kwargs:
            if lr != 1e-3:
                raise ValueError("Pass only lr, not both lr and learning_rate")
            lr = kwargs.pop("learning_rate")
            warnings.warn("learning_rate is deprecated; use lr", FutureWarning, stacklevel=2)
        self.random_state = random_state
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.lambda_sparse = lambda_sparse
        self.lambda_kl = lambda_kl
        self.kl_auto_scale = kl_auto_scale
        self.lambda_contraction = lambda_contraction
        self.grad_clip_norm = grad_clip_norm
        if loss_reduction not in ("elbo", "outcome_mean", "sample_weighted"):
            raise ValueError("loss_reduction must be elbo, outcome_mean or sample_weighted")
        if lambda_kl < 0:
            raise ValueError("lambda_kl must be nonnegative")
        if kl_auto_scale or kl_normalize or loss_reduction != "elbo" or lambda_kl != 1.0:
            warnings.warn("Nonstandard KL weighting or likelihood reduction defines a tempered/weighted objective, not the ordinary ELBO.", UserWarning, stacklevel=2)
        self.loss_reduction = loss_reduction
        self.device = device
        self.verbose = verbose
        self.enforce_centering = enforce_centering
        self.kl_normalize_encoder = kl_normalize

        # For pure linear mixed models (no hidden layers), default to identity head
        if not hidden_dims and len(self.outcomes) <= 1:
            self.use_identity_head = True
            # Disable aggressive grad clipping for pure linear mixed models to allow fast convergence
            if grad_clip_norm == 1.0:
                self.grad_clip_norm = 0.0

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
        """Fit from scratch, including preprocessing, weights, and optimizer.

        ``reuse_design`` explicitly reuses preprocessing only; it does not
        warm-start parameters. Training overrides apply to the new optimizer.
        """
        if "learning_rate" in kwargs:
            if "lr" in kwargs:
                raise ValueError("Pass only lr, not both lr and learning_rate")
            kwargs["lr"] = kwargs.pop("learning_rate")
        unknown = set(kwargs) - {"epochs", "batch_size", "lr"}
        if unknown:
            raise TypeError(f"Unsupported fit arguments: {sorted(unknown)}")
        for key, value in kwargs.items():
            if value <= 0 or (key in ("epochs", "batch_size") and int(value) != value):
                raise ValueError(f"{key} must be positive (integer for epochs/batch_size)")
            setattr(self, key, int(value) if key in ("epochs", "batch_size") else value)

        # Validate val_fraction
        if not 0 <= val_fraction < 1:
            raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Handle Validation Split
        train_df = df
        if val_df is None and val_fraction > 0.0:
            # Split by row position: dataframe index labels need not be unique.
            positions = pd.Series(np.arange(len(df)))
            val_positions = positions.sample(frac=val_fraction, random_state=self.random_state if self.random_state is not None else 42).to_numpy()
            train_mask = np.ones(len(df), dtype=bool)
            train_mask[val_positions] = False
            train_df = df.iloc[train_mask].reset_index(drop=True)
            val_df = df.iloc[val_positions]
            val_df = val_df.reset_index(drop=True)
            self.logger.info(f"Split data: {len(train_df)} train, {len(val_df)} validation")

        if len(train_df) == 0 or (val_df is not None and len(val_df) == 0):
            raise ValueError("Training and requested validation partitions must be nonempty")

        self._train_data = train_df.copy()
        if early_stopping and val_df is None:
            raise ValueError("early_stopping requires val_df or val_fraction")

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
        # λ_KL = 1 with the 1/N factor in the training loop handling
        # the scaling. The optional group-size rule is a heuristic; its benefit
        # must be validated separately on training/validation partitions.
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
        for outcome in self.outcomes:
            outcome.family.load_state_dict(self._initial_family_states[outcome.name])
        self._build_model(fixed_input_dim=X.shape[1], compiled_design=self.design)
        self.to(self.device)
        self.history = []
        self.val_history = []
            
        # 5. Prepare Tensors
        train_data = TabMixDataset(X, Z_ind, Z_slopes, Ys)
        val_data = None
        if val_df is not None:
            Xv, Zv_ind, Zv_slopes, Yv = self.design.transform(val_df, mode='fit')
            val_data = TabMixDataset(Xv, Zv_ind, Zv_slopes, Yv)

        # Initialize on observed training targets only. The explicit design
        # intercept makes the old fixed_linear.bias initialization unreachable;
        # nonlinear models need their actual output bias initialized instead.
        with torch.no_grad():
            for name, family in self.families.items():
                if not isinstance(family, Gaussian):
                    continue
                observed = np.asarray(Ys[name])
                observed = observed[np.isfinite(observed)]
                if not observed.size:
                    continue
                family.log_sigma.fill_(np.log(max(float(observed.std()), 1e-3)))
                head = self.heads[name]
                if isinstance(head, nn.Linear) and head.bias is not None and self.output_sem is None:
                    head.bias.fill_(float(observed.mean()))
        
        if (self.use_identity_head and not self.hidden_dims and len(self.outcomes) == 1
                and isinstance(self.data_family(self.outcomes[0].name), Gaussian)):
            name = self.outcomes[0].name
            valid = np.isfinite(Ys[name]).all(axis=1)
            if valid.any() and X.shape[1]:
                coef = np.linalg.lstsq(X[valid], Ys[name][valid, 0], rcond=None)[0]
                with torch.no_grad():
                    self.encoder.fixed_linear.weight[0].copy_(torch.as_tensor(coef, device=self.device))

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
        
    def predict(self, df: pd.DataFrame, outcome: str = None, mode: str = 'default', use_random_effects: bool = True, as_dict: Optional[bool] = None, **prediction_kwargs):
        """
        Predict outcomes.

        Args:
            outcome: Optional outcome name to select.
            mode: Family-specific prediction mode.
            use_random_effects: Toggle random effects contribution.
            as_dict: If True returns dict; if False returns first/selected output array;
                     if None uses legacy behavior (unwrap single outcome).
        """
        if self.design is None:
            raise RuntimeError('Fit the model before prediction')
        if outcome is not None and self._outcome_idx(outcome) == -1:
            raise ValueError(f'Unknown outcome: {outcome}')
        self.eval()
        X, Z_ind, Z_slopes, _ = self.design.transform(df, mode='predict', include_targets=False)
        results = self._predict_batches(X, Z_ind, Z_slopes, mode=mode, use_random_effects=use_random_effects, prediction_kwargs=prediction_kwargs)

        if outcome:
            return results.get(outcome)
        unwrap = as_dict is False or (as_dict is None and len(results) == 1)
        if unwrap and len(results) == 1:
            return list(results.values())[0]
        return results

    def _predict_batches(self, X, Z_ind, Z_slopes, mode="default", use_random_effects: bool = True, prediction_kwargs=None):
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
                    pred = family.predict(theta, mode=mode, **(prediction_kwargs or {}))
                    if not torch.is_tensor(pred):
                        pred = torch.as_tensor(pred, device=self.device, dtype=torch.float32)
                    buffers.setdefault(name, []).append(pred.detach().cpu().reshape((len(bx),) + tuple(pred.shape[1:])))

        results = {}
        for name, parts in buffers.items():
            if not parts:
                continue
            results[name] = torch.cat(parts, dim=0).numpy()
        return results

    def predict_proba(self, df, outcome=None, as_dict: Optional[bool] = None):
        return self.predict(df, outcome=outcome, mode='proba', as_dict=as_dict)

    def adapt_unseen_groups(self, df: pd.DataFrame, n_steps: int = 100,
                            lr: float = 0.03, query_df: pd.DataFrame = None):
        """Fit conditional MAP effects using labeled support rows only.

        ``df`` is support data. Return effects aligned to ``query_df`` (or to
        support rows for explicit fitted-value diagnostics). Only IID and UN
        priors support new unrelated groups. This is MAP adaptation, not a
        variational E-step or a source of prediction intervals.
        """
        from ..covariance import IIDCovariance, UnstructuredSlopeCovariance
        if self.design is None or self.encoder is None:
            raise RuntimeError("Fit the model before adapting groups")
        if n_steps < 1 or lr <= 0:
            raise ValueError("n_steps and lr must be positive")
        query_df = df if query_df is None else query_df
        X, zi, zs, targets = self.design.transform(df, mode="fit")
        qx, qzi, _, _ = self.design.transform(query_df, mode="predict")
        Xt, zit, zst, yt = self._to_tensors(X, zi, zs, targets)
        local, support_maps, query_maps = {}, {}, {}
        for i, term in enumerate(self.design.random_terms):
            tid = term.term_id
            unseen = zi[i] == 0
            if not unseen.any():
                continue
            cov = self.encoder.cov_modules[tid]
            if not isinstance(cov, (IIDCovariance, UnstructuredSlopeCovariance)):
                raise NotImplementedError("New-group MAP adaptation currently supports IID and UN priors only")
            labels = list(pd.unique(df.loc[unseen, term.group_col]))
            if any(pd.isna(label) for label in labels):
                raise ValueError("Support groups must have nonmissing labels")
            mapping = {label: j + 1 for j, label in enumerate(labels)}
            support_maps[tid] = torch.tensor(df[term.group_col].map(mapping).fillna(0).to_numpy(), dtype=torch.long, device=self.device)
            query_maps[tid] = torch.tensor(query_df[term.group_col].map(mapping).fillna(0).to_numpy(), dtype=torch.long, device=self.device)
            local[tid] = nn.Parameter(Xt.new_zeros(len(labels), len(term.slope_cols), self.embedding_dim))
        if not local:
            return {}
        if not targets or not any(torch.isfinite(y).all(dim=1).any() for y in yt.values()):
            raise ValueError("Adaptation requires observed outcomes in support_df")

        flags = [(p, p.requires_grad) for p in self.parameters()]
        modes = [(module, module.training) for module in self.modules()]
        def overrides(maps):
            result = {}
            for tid, v in local.items():
                # Optimize standard-normal innovations under the learned scale.
                u = self.encoder.cov_modules[tid](v)
                padded = torch.cat([u.new_zeros(1, *u.shape[1:]), u], dim=0)
                result[tid] = padded.index_select(0, maps[tid])
            return result
        try:
            self.eval()  # fixed training effects, no dropout or spatial noise
            for p, _ in flags:
                p.requires_grad_(False)
            optimizer = optim.Adam(list(local.values()), lr=lr)
            for _ in range(n_steps):
                optimizer.zero_grad()
                out = self._ensure_dict_outputs(self.forward(Xt, zit, zst, _transductive_override=overrides(support_maps)))
                likelihood_sums, _ = self._likelihood_components(out, yt)
                nll = sum(self._outcome_spec(k).loss_weight * v for k, v in likelihood_sums.items())
                prior = sum(0.5 * v.square().sum() for v in local.values())
                loss = (nll + prior) / len(df)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                return {tid: values.detach().clone() for tid, values in overrides(query_maps).items()}
        finally:
            for p, flag in flags:
                p.requires_grad_(flag)
            for module, mode in modes:
                module.training = mode

    def predict_transductive(self, df: pd.DataFrame, outcome: str = None,
                             mode: str = 'default', n_steps: int = 100,
                             lr: float = 0.03, as_dict: Optional[bool] = None,
                             *, support_df: pd.DataFrame = None):
        """Predict query rows after MAP adaptation on separate labeled support.

        Query outcomes are ignored. Support and query observations must be
        disjoint for predictive evaluation (they may share group labels).
        Calling without support_df is rejected to prevent silent label reuse.
        """
        if support_df is None:
            raise ValueError("Provide separate labeled support_df; query outcomes must not be used for adaptation")
        override = self.adapt_unseen_groups(support_df, n_steps, lr, query_df=df)
        self.eval()
        X, zi, zs, _ = self.design.transform(df, mode="predict")
        loader, _ = self._make_dataloader(TabMixDataset(X, zi, zs, {}), shuffle=False)
        buffers = {}
        offset = 0
        with torch.no_grad():
            for batch in loader:
                bx, bz_ind, bz_slopes, _ = self._move_batch_to_device(batch)
                batch_override = {tid: v[offset:offset+len(bx)] for tid, v in override.items()}
                outputs = self._ensure_dict_outputs(self.forward(bx, bz_ind, bz_slopes, _transductive_override=batch_override))
                for name, raw in outputs.items():
                    family = self.data_family(name)
                    pred = torch.as_tensor(family.predict(family.forward_params(raw), mode=mode))
                    pred = pred.reshape((len(bx),) + tuple(pred.shape[1:]))
                    buffers.setdefault(name, []).append(pred.cpu())
                offset += len(bx)
        result = {name: torch.cat(parts).numpy() for name, parts in buffers.items()}
        if outcome:
            return result[outcome]
        if len(result) == 1 and as_dict is not True:
            return next(iter(result.values()))
        return result

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
                'n_levels': len(term.level_map),
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
             use_fixed_bias=False,
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
                random_config=[],
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
                self.weight = nn.Parameter(torch.eye(output_dim, input_dim), requires_grad=False)
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
             
        if self.use_identity_head and len(self.outcomes) > 1:
            raise ValueError("Identity heads are restricted to a single outcome; use separate learned heads for multitask models")

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
        # Biases and normalization scales should not be shrunk toward zero;
        # penalizing the output bias makes training depend on response origin.
        re_param_ids.update(id(p) for p in self.parameters() if p.ndim <= 1)
        # Global observation-family scale parameters are likelihood parameters.
        re_param_ids.update(id(p) for p in self.families.parameters())
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

    def forward(self, X, Z_ind=None, Z_slopes=None, return_attention=False, use_random_effects: bool = True,
                 _transductive_override=None):
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
        h_encoded, kl_encoder = self.encoder(X, Z_ind, Z_slopes, use_random_effects=use_random_effects,
                                             _transductive_override=_transductive_override)
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

                loss = self._likelihood_loss(outputs_dict, bys, len(bx))
                if kl is not None:
                    loss = loss + self.lambda_kl * kl / max(n_samples, 1)

                loss = loss + self._structural_penalties()

                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training objective at epoch {epoch}, batch {i}")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.parameters()):
                    raise FloatingPointError(f"Nonfinite training gradient at epoch {epoch}, batch {i}")
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_norm)
                self.optimizer.step()
                epoch_loss += loss.item() * len(bx)
                callbacks.on_batch_end(i, {"batch": i, "batch_loss": loss.item()})

            train_avg = epoch_loss / max(n_samples, 1)
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

    def _likelihood_components(self, outputs, targets):
        sums, counts = {}, {}
        for name, target in targets.items():
            mask = torch.isfinite(target).all(dim=1) if target.ndim > 1 else torch.isfinite(target)
            count = int(mask.sum())
            counts[name] = count
            if count:
                family = self.data_family(name)
                sums[name] = family.nll(family.forward_params(outputs[name][mask]), target[mask]) * count
        return sums, counts

    def _reduce_likelihood(self, sums, counts, n_rows):
        loss = next(self.parameters()).new_zeros(())
        denom = 0.0
        for name, nll_sum in sums.items():
            spec = self._outcome_spec(name)
            weight = spec.loss_weight if spec else 1.0
            if self.loss_reduction == "outcome_mean":
                loss = loss + weight * nll_sum / counts[name]
                denom += weight
            else:
                loss = loss + weight * nll_sum
                denom += weight * counts[name]
        if self.loss_reduction == "elbo":
            return loss / max(n_rows, 1)
        return loss / max(denom, 1.0)

    def _likelihood_loss(self, outputs, targets, n_rows):
        sums, counts = self._likelihood_components(outputs, targets)
        if not sums:
            # A valid all-missing minibatch contributes zero likelihood; keep a
            # gradient path even in fixed-only models with no global KL.
            return sum(value.sum() * 0 for value in outputs.values())
        return self._reduce_likelihood(sums, counts, n_rows)

    def _evaluate_loss(self, data):
        """Conditional plug-in NLL, aggregated independently of batch partitions."""
        loader, n_rows = self._make_dataloader(data, shuffle=False)
        self.eval()
        sums, counts = {}, {}
        with torch.no_grad():
            for batch in loader:
                bx, zi, zs, targets = self._move_batch_to_device(batch)
                outputs = self._ensure_dict_outputs(self.forward(bx, zi, zs))
                batch_sums, batch_counts = self._likelihood_components(outputs, targets)
                for name, val in batch_sums.items():
                    sums[name] = sums.get(name, 0.0) + val
                for name, val in batch_counts.items():
                    counts[name] = counts.get(name, 0) + val
            return float(self._reduce_likelihood(sums, counts, n_rows))

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
                     elif 'contraction' in k: total_loss = total_loss + self.lambda_contraction * v
                     else: total_loss = total_loss + self.lambda_sparse * v
        
        if self.architecture == 'manifold':
            total_loss = total_loss + self.lambda_sparse * self.backbone.penalty()

        # Output SEM penalties
        if getattr(self, 'output_sem', None) is not None:
             total_loss = total_loss + self.lambda_sparse * self.output_sem.sparsity_penalty()
        
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

            self.logLik = total_loglik
            self.conditional_logLik = total_loglik
            self.AIC = None
            self.BIC = None
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
                    effect = self.encoder.random_effects_values(tid, mu_param)
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

    # --- Sklearn & Introspection ---

    def get_params(self, deep=True):
        keys = ("formula", "outcomes", "architecture", "embedding_dim", "hidden_dims",
                "use_static_structure", "use_dynamic_structure", "n_attention_heads",
                "dynamic_structure_config", "backbone_dropout", "manifold_configs", "use_recursive",
                "aggregation", "lr", "weight_decay", "batch_size", "epochs", "device", "verbose",
                "lambda_sparse", "lambda_kl", "kl_auto_scale", "lambda_contraction",
                "grad_clip_norm", "loss_reduction", "enforce_centering", "random_state",
                "cov_struct_map", "scale_random_by_terms", "unknown_strategy", "output_sem_mode",
                "output_structure", "output_structure_config", "use_identity_head")
        params = {key: getattr(self, key) for key in keys}
        params["family"] = getattr(self, "single_outcome_family", None)
        params["kl_normalize"] = self.kl_normalize_encoder
        return params

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
            'format_version': 3,
            'state_dict': {k: v for k, v in self.state_dict().items()
                           if not k.startswith(('single_outcome_family.', 'family.'))},
            'params': self.get_params(),
            'design_state': self.design.export_state() if self.design else None,
            'history': self.history,
            'val_history': self.val_history,
            'initial_family_states': self._initial_family_states,
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
        if checkpoint.get('format_version') not in (2, 3):
            raise ValueError("Legacy checkpoint uses pre-revision covariance/indexing semantics; refit with this version.")
        params = checkpoint['params']
        if checkpoint['format_version'] == 2:
            for spec in (params.get('cov_struct_map') or {}).values():
                if spec.get('type', 'IID').upper() in ('AR', 'ARMA') and spec.get('p', 1) > 1:
                    raise ValueError("Legacy AR(p>1) checkpoint uses direct AR coefficients; refit with the stable partial-correlation parameterization.")
        
        # Create new instance with saved params
        if device:
            params['device'] = device
        
        # Remove family for subclasses that override it (TabMixNNRegressor, TabMixNNClassifier)
        if cls != TabMixNN and 'family' in params:
            del params['family']
            
        model = cls(**params)
        if 'initial_family_states' in checkpoint:
            model._initial_family_states = checkpoint['initial_family_states']
        
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
            
        # Load canonical family registration strictly; aliases are not serialized.
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.to(model.device)
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

    def feature_ablation_values(self, df, background_size=50, seed=42, background_df=None, outcome=None):
        """
        Per-feature mean ablation on the fixed design matrix.

        For each feature j, replaces column j with its background mean and records
        the change in prediction.  This is a single-feature marginal ablation, not
        a Shapley value: it does not average over coalitions and need not yield an additive
        decomposition of the prediction.  Use it as a descriptive sensitivity measure only.

        Strategy:
            - Baseline = mean prediction over a small background sample.
            - For each sample i and feature j:
              ablation_ij = pred_i - pred_i_with_feature_j_replaced_by_background_mean.
        """
        if self.design is None:
            raise RuntimeError("Model design is not initialized. Fit the model before calling feature_ablation_values.")

        if background_size < 1 or int(background_size) != background_size or len(df) == 0:
            raise ValueError("Ablations require query rows and a positive integer background_size")
        background_df = df if background_df is None else background_df
        if len(background_df) == 0:
            raise ValueError("Ablation background must be nonempty")
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        rng = np.random.RandomState(seed)

        try:
            # Build design matrices
            X_np, Z_ind_np, Z_slopes_np, _ = self.design.transform(df, mode="predict", include_targets=False)
            X = torch.tensor(X_np, dtype=torch.float32, device=self.device)
            Z_ind = [torch.tensor(z, dtype=torch.long, device=self.device) for z in Z_ind_np]
            Z_slopes = [torch.tensor(z, dtype=torch.float32, device=self.device) for z in Z_slopes_np]

            feature_names = list(self.design.fixed_feature_names)
            n_samples, n_features = X.shape
            # A separate training background keeps query composition from defining
            # the reference perturbation. The historical default remains df.
            Xb, zib, zsb, _ = self.design.transform(background_df, include_targets=False)
            bg_idx = rng.choice(len(background_df), size=min(background_size, len(background_df)), replace=False)
            X_bg = torch.as_tensor(Xb[bg_idx], dtype=X.dtype, device=self.device)
            Z_ind_bg = [torch.as_tensor(z[bg_idx], dtype=torch.long, device=self.device) for z in zib]
            Z_slopes_bg = [torch.as_tensor(z[bg_idx], dtype=X.dtype, device=self.device) for z in zsb]

            base_preds = self._predict_from_tensors(X_bg, Z_ind_bg, Z_slopes_bg, outcome=outcome)
            base_value = float(base_preds.mean())

            # Precompute per-sample original predictions
            orig_preds = self._predict_from_tensors(X, Z_ind, Z_slopes, outcome=outcome)

            shap_values = torch.zeros((n_samples, n_features), device=self.device)
            mu_feat = X_bg.mean(dim=0)

            for j in range(n_features):
                X_mod = X.clone()
                X_mod[:, j] = mu_feat[j]
                preds_mod = self._predict_from_tensors(X_mod, Z_ind, Z_slopes, outcome=outcome)
                shap_values[:, j] = orig_preds - preds_mod

            return {
                "ablation_values": shap_values.cpu().numpy(),
                "feature_names": feature_names,
                "base_value": base_value,
                "predictions": orig_preds.cpu().numpy(),
            }
        finally:
            for module, training in modes:
                module.training = training

    def shapley_values(self, df, background_size=50, seed=42):
        """Deprecated: use feature_ablation_values."""
        import warnings
        warnings.warn(
            "shapley_values() is a misnomer; this method computes single-feature mean ablations, "
            "not Shapley values. Use feature_ablation_values() instead.",
            DeprecationWarning, stacklevel=2,
        )
        return self.feature_ablation_values(df, background_size=background_size, seed=seed)

    def plot_feature_ablation_summary(self, df, max_features=10):
        """Bar plot of mean absolute ablation values across features."""
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.text(0.5, 0.5, "Feature Ablation Summary Placeholder")
        return fig

    def plot_shap_summary(self, df, max_features=10):
        """Deprecated: use plot_feature_ablation_summary."""
        import warnings
        warnings.warn(
            "plot_shap_summary() is deprecated. Use plot_feature_ablation_summary() instead.",
            DeprecationWarning, stacklevel=2,
        )
        return self.plot_feature_ablation_summary(df, max_features=max_features)

    def random_effect_ablations(self, df, outcome=None):
        """
        Per-term random-effect ablations: for each random term, zeros its variational
        mean and records the change in prediction.

        This is a single-term marginal ablation, not a Shapley allocation.  Terms
        are not independent; contributions need not sum to the full prediction.
        For each random term:
            ablation = pred_full - pred_with_term_zeroed
        """
        if self.design is None:
            raise RuntimeError("Model design is not initialized. Fit the model before calling shapley_random_effects.")

        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            outcome_name = outcome

            # Transform data
            X_np, Z_ind_np, Z_slopes_np, _ = self.design.transform(df, mode="predict")
            X = torch.tensor(X_np, dtype=torch.float32, device=self.device)
            Z_ind = [torch.tensor(z, dtype=torch.long, device=self.device) for z in Z_ind_np]
            Z_slopes = [torch.tensor(z, dtype=torch.float32, device=self.device) for z in Z_slopes_np]

            # Full predictions
            full_preds = self._predict_from_tensors(X, Z_ind, Z_slopes, outcome=outcome_name)

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
                    preds_ablate = self._predict_from_tensors(X, Z_ind, Z_slopes, outcome=outcome_name)
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
                        "ablation_value": shap_term[i],
                        "baseline": baseline[i],
                        "prediction": full_preds.cpu().numpy()[i],
                    })

            return pd.DataFrame(records)
        finally:
            for module, training in modes:
                module.training = training

    def shapley_random_effects(self, df, outcome=None):
        """Deprecated: use random_effect_ablations."""
        import warnings
        warnings.warn(
            "shapley_random_effects() is a misnomer; this method computes single-term ablations, "
            "not Shapley values. Use random_effect_ablations() instead.",
            DeprecationWarning, stacklevel=2,
        )
        return self.random_effect_ablations(df, outcome=outcome)

    def _predict_from_tensors(self, X, Z_ind, Z_slopes, use_random_effects: bool = True, outcome=None):
        """Scalar response for ablations; vector reductions require an explicit estimand."""
        if outcome is None:
            if len(self.outcomes) != 1:
                raise ValueError("Select outcome explicitly for a multitask ablation")
            outcome = self.outcomes[0].name
        if self._outcome_idx(outcome) == -1:
            raise ValueError(f"Unknown outcome: {outcome}")
        family = self.data_family(outcome)
        if not isinstance(family, (Gaussian, Binomial, Poisson, NegBinomial, ContinuousSurvival)):
            raise NotImplementedError("Ablations require a scalar outcome; vector/time-curve reduction is not defined")
        with torch.no_grad():
            outputs = self._ensure_dict_outputs(self.forward(X, Z_ind, Z_slopes,
                                                use_random_effects=use_random_effects))
            theta = family.forward_params(outputs[outcome])
            mode = "proba" if isinstance(family, Binomial) else "default"
            pred = family.predict(theta, mode=mode)
            return torch.as_tensor(pred, device=self.device, dtype=torch.float32).reshape(-1)

    def shapley_sam_edges(self, layer='output', df=None, n_samples=50, seed=42):
        """Edge attribution is not implemented.

        A coalition-based edge Shapley method requires enumerating subsets of
        structural edges, which is combinatorially expensive and not validated.
        This method raises NotImplementedError unconditionally.
        """
        raise NotImplementedError(
            "Edge attribution (shapley_sam_edges) is not implemented. "
            "Use feature_ablation_values() for feature-level sensitivity."
        )

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
