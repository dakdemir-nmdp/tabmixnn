import re
import pandas as pd
import numpy as np
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from patsy import dmatrix, build_design_matrices, PatsyError
from ..core.families import DiscreteTimeSurvival, DiscreteTimeCompetingRisks

@dataclass
class OutcomeSpec:
    name: str
    family: Any # Family instance
    y: Optional[str] = None
    time: Optional[str] = None
    event: Optional[str] = None
    cause: Optional[str] = None
    loss_weight: float = 1.0
    
    def get_target_cols(self):
        if self.y: return [self.y]
        # Competing risks: time + cause only (no event column)
        if self.cause: return [self.time, self.cause]
        # Standard survival: time + event
        if self.time and self.event: return [self.time, self.event]
        return []

@dataclass
class RandomEffectTerm:
    term_id: str
    group_col: str
    slope_cols: List[str] # ["Intercept", "slope1", ...]
    level_map: Dict[Any, int] = field(default_factory=dict)
    train_levels: List[Any] = field(default_factory=list)  # specific to K-covariance support
    slope_stats: Dict[str, Any] = field(default_factory=dict)  # mean/std per non-intercept slope

class CompiledDesign:
    def __init__(self, formula, outcomes, sam_location, sam_dim, cov_struct_map=None):
        self.formula = formula
        self.outcomes = outcomes
        self.sam_location = sam_location
        self.sam_dim = sam_dim
        self.cov_struct_map = cov_struct_map or {}
        self.is_fitted = False
        
        # Fixed effects state
        self.fixed_cols = []
        self.fixed_dummy_cols = [] # If OneHot used
        self.numeric_transformer_params = {} # mean, scale per col
        self.categorical_levels = {} # col -> list of levels
        self.categorical_dummy_levels = {} # col -> levels actually encoded
        self.intercept_included = True
        self.design_info = None
        self.fixed_feature_names: List[str] = []
        self.fixed_formula = ""
        
        # Random effects state
        self.random_terms = [] # List[RandomEffectTerm]
        
    def fit(self, df):
        # 1. Parse Formula
        fixed_part, random_parts = self._parse_formula(self.formula)
        
        # 2. Process Fixed Effects
        self._fit_fixed(df, fixed_part)
        
        # 3. Process Random Effects
        self._fit_random(df, random_parts)
        
        self.is_fitted = True
        return self

    def transform(self, df, mode='predict'):
        # mode='fit' or 'predict'
        # Returns:
        # X: fixed design matrix (Tensor)
        # Z_indices: list of tensors for group indices per term
        # Z_slopes: list of tensors for slope values per term
        # Ys: dict of outcome tensors (only if keys present in df)
        
        X = self._transform_fixed(df)
        
        Z_ind, Z_slopes = self._transform_random(df)
        
        Ys = {}
        # Only try to extract Y if we are not purely predicting on new inputs
        # But if fit_transform was called or user passed targets
        for outcome in self.outcomes:
            cols = outcome.get_target_cols()
            if all(c in df.columns for c in cols):
                Ys[outcome.name] = self._extract_outcome(df, outcome)
        
        return X, Z_ind, Z_slopes, Ys

    def _parse_formula(self, formula):
        # Remove outcomes part (left of ~) if present
        if '~' in formula:
            _, rhs = formula.split('~', 1)
        else:
            rhs = formula
            
        rhs = rhs.strip()
        
        # Extract ( ... | ... )
        random_pattern = r'\((.*?)\|([^\)]+)\)'
        random_matches = re.findall(random_pattern, rhs)
        
        # Remove random parts from string to get fixed part
        fixed_str = re.sub(random_pattern, '', rhs)
        
        # Parse fixed terms
        # Split by + and cleanup
        fixed_terms = [t.strip() for t in fixed_str.split('+') if t.strip()]
        
        # Check for Intercept removal (0 or -1)
        intercept = True
        if '0' in fixed_terms:
            fixed_terms.remove('0')
            intercept = False
        if '-1' in fixed_terms:
            fixed_terms.remove('-1')
            intercept = False
        # A standalone "1" indicates an intercept and should not be treated as a column
        fixed_terms = [t for t in fixed_terms if t != '1']
        # Legacy default: handle intercept via bias, not explicit column
        self.intercept_included = intercept
            
        # Parse random terms
        random_terms_spec = []
        for i, (slopes_str, group_str) in enumerate(random_matches):
            group_col = group_str.strip()
            # Slopes
            # (1|g) -> Intercept
            # (x|g) -> Intercept, x (usually) - lme4 logic. 
            # Actually (x|g) means 1 + x.
            # (0+x|g) means x only.
            
            s_terms = [s.strip() for s in slopes_str.split('+') if s.strip()]
            r_intercept = True
            if '0' in s_terms:
                s_terms.remove('0')
                r_intercept = False
            if '1' in s_terms:
                s_terms.remove('1')
                r_intercept = True # explicit 1
            elif not s_terms and not r_intercept:
                # empty? e.g. (0|g) -> nothing
                continue
                
            cols = []
            if r_intercept:
                cols.append("Intercept")
            cols.extend(s_terms)
            
            random_terms_spec.append({
                'term_id': f"term_{i}",
                'group': group_col,
                'slopes': cols
            })
            
        return {'terms': fixed_terms, 'intercept': intercept}, random_terms_spec
    def _build_fixed_formula(self, fixed_part):
        terms = fixed_part['terms']
        if not terms:
            rhs = "1" if self.intercept_included else "-1"
        else:
            rhs = " + ".join(terms)
            if not self.intercept_included:
                rhs = rhs + " -1"
        return rhs

    def _prepare_fixed_frame(self, df: pd.DataFrame, fit: bool = False):
        """Clean and type-cast fixed effect columns for patsy."""
        df_clean = df.copy()
        for col in df_clean.columns:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                if fit or col not in self.numeric_transformer_params:
                    mu = df_clean[col].mean()
                    sigma = df_clean[col].std() if df_clean[col].std() != 0 else 1.0
                    self.numeric_transformer_params[col] = (mu, sigma)
                mu, _ = self.numeric_transformer_params.get(col, (0.0, 1.0))
                df_clean[col] = df_clean[col].fillna(mu)
            else:
                # Treat as categorical with explicit UNKNOWN level
                if fit or col not in self.categorical_levels:
                    levels = sorted(df_clean[col].dropna().astype(str).unique().tolist())
                    if "__UNKNOWN__" not in levels:
                        levels.append("__UNKNOWN__")
                    self.categorical_levels[col] = levels
                levels = self.categorical_levels.get(col, [])
                df_clean[col] = df_clean[col].astype(str).astype("category")
                if levels:
                    df_clean[col] = df_clean[col].cat.set_categories(levels)
                df_clean[col] = df_clean[col].fillna("__UNKNOWN__")
                mask_unknown = ~df_clean[col].isin(levels)
                if mask_unknown.any():
                    df_clean.loc[mask_unknown, col] = "__UNKNOWN__"
        return df_clean

    def _fit_fixed(self, df, fixed_part):
        self.intercept_included = fixed_part["intercept"]
        rhs = self._build_fixed_formula(fixed_part)
        self.fixed_formula = f"~ {rhs}"

        df_fixed = self._prepare_fixed_frame(df, fit=True)
        try:
            design_mat = dmatrix(self.fixed_formula, df_fixed, return_type="dataframe")
        except PatsyError as e:
            raise ValueError(f"Failed to parse fixed-effects formula '{self.fixed_formula}': {e}") from e

        self.design_info = design_mat.design_info
        self.fixed_feature_names = list(self.design_info.column_names)
        # Backward-compatible metadata placeholders
        self.fixed_cols = [{'name': name, 'type': 'patsy'} for name in self.fixed_feature_names]
        self.categorical_dummy_levels = {k: v for k, v in self.categorical_levels.items()}

    def _transform_fixed(self, df):
        if self.design_info is None:
            return np.zeros((len(df), 0), dtype=np.float32)

        df_fixed = self._prepare_fixed_frame(df, fit=False)
        try:
            mats = build_design_matrices([self.design_info], df_fixed, return_type="dataframe")
            X = np.asarray(mats[0], dtype=np.float32)
        except PatsyError as e:
            raise ValueError(f"Failed to build fixed design matrix with formula '{self.fixed_formula}': {e}") from e
        return X

    def export_state(self) -> Dict[str, Any]:
        """Export a picklable snapshot of the design (avoids patsy objects)."""
        if self.design_info is not None and any(
            info.state.get("transforms") for info in self.design_info.factor_infos.values()
        ):
            raise NotImplementedError("Saving stateful Patsy transforms is unsupported; preprocess explicitly using training-only parameters")
        return {
            "formula": self.formula,
            "outcomes": self.outcomes,
            "sam_location": self.sam_location,
            "sam_dim": self.sam_dim,
            "cov_struct_map": self.cov_struct_map,
            "intercept_included": self.intercept_included,
            "numeric_transformer_params": self.numeric_transformer_params,
            "categorical_levels": self.categorical_levels,
            "categorical_dummy_levels": self.categorical_dummy_levels,
            "fixed_formula": self.fixed_formula,
            "fixed_feature_names": self.fixed_feature_names,
            "random_terms": self.random_terms,
        }

    def _rebuild_design_info(self):
        """Rebuild patsy design_info from stored metadata without original data."""
        if not self.fixed_formula:
            return

        data = {}
        for col in self.numeric_transformer_params.keys():
            data[col] = pd.Series([0.0])
        for col, levels in self.categorical_levels.items():
            levels_list = list(levels)
            if not levels_list:
                levels_list = ["__UNKNOWN__"]
            cat = pd.Categorical([levels_list[0]], categories=levels_list)
            data[col] = cat

        df_dummy = pd.DataFrame(data) if data else pd.DataFrame(index=[0])
        design_mat = dmatrix(self.fixed_formula, df_dummy, return_type="dataframe")
        self.design_info = design_mat.design_info
        self.fixed_feature_names = list(self.design_info.column_names)

    @classmethod
    def restore_from_state(cls, state: Dict[str, Any]):
        design = cls(
            formula=state.get("formula"),
            outcomes=state.get("outcomes", []),
            sam_location=state.get("sam_location"),
            sam_dim=state.get("sam_dim"),
            cov_struct_map=state.get("cov_struct_map"),
        )
        design.intercept_included = state.get("intercept_included", False)
        design.numeric_transformer_params = state.get("numeric_transformer_params", {})
        design.categorical_levels = state.get("categorical_levels", {})
        design.categorical_dummy_levels = state.get("categorical_dummy_levels", {})
        design.fixed_formula = state.get("fixed_formula", "")
        design.fixed_feature_names = state.get("fixed_feature_names", [])
        design.random_terms = state.get("random_terms", [])
        design.is_fitted = True
        design._rebuild_design_info()
        return design

    def _fit_random(self, df, random_parts_spec):
        self.random_terms = []
        for spec in random_parts_spec:
            group = spec['group']
            slopes = spec['slopes']
            
            # Map groups
            # If K-structure provided with explicit levels, honor that full ordering.
            maybe_cov = self.cov_struct_map.get(group, {})
            use_k_levels = False
            if isinstance(maybe_cov, dict):
                use_k_levels = maybe_cov.get("type", "").upper() in ("K", "KRON", "GP") and "levels" in maybe_cov

            if use_k_levels:
                levels_list = list(maybe_cov["levels"])
                mapping = {val: i+1 for i, val in enumerate(levels_list)}
                if len(set(levels_list)) != len(levels_list):
                    raise ValueError(f"Duplicate covariance levels for {group}.")
                observed = set(df[group].dropna().unique())
                if not observed.issubset(set(levels_list)):
                    raise ValueError(f"Training groups missing from covariance levels for {group}.")
                train_levels = list(maybe_cov.get("train_levels", [x for x in levels_list if x in observed]))
                if len(set(train_levels)) != len(train_levels) or set(train_levels) != observed:
                    raise ValueError(f"train_levels must contain exactly the observed training groups for {group}.")
            else:
                uniques = df[group].dropna().unique()
                mapping = {val: i+1 for i, val in enumerate(uniques)}
                train_levels = list(uniques)

            slope_stats = {}
            for s in slopes:
                if s == "Intercept":
                    continue
                if s not in df.columns:
                    raise ValueError(f"Random slope column {s} not found in dataframe.")
                mu = df[s].mean()
                sigma = df[s].std()
                if sigma == 0:
                    sigma = 1.0
                slope_stats[s] = (mu, sigma)
            
            term = RandomEffectTerm(
                term_id=spec['term_id'],
                group_col=group,
                slope_cols=slopes,
                level_map=mapping,
                train_levels=train_levels,
                slope_stats=slope_stats
            )
            self.random_terms.append(term)

            
    def _transform_random(self, df):
        Z_ind = []
        Z_slopes = []
        
        for term in self.random_terms:
            # 1. Group Indices
            # IMPORTANT: Index 0 is RESERVED for unknown/unseen groups
            # - Training groups are mapped to indices 1, 2, 3, ...
            # - Groups not in level_map receive index 0 (fillna(0))
            # - The encoder uses this to apply unknown_strategy ("learned" or "zero")
            # Convert to object to avoid pandas Categorical fillna restriction
            group_vals = df[term.group_col].astype(object)
            indices = group_vals.map(term.level_map).fillna(0).astype(int).values
            Z_ind.append(indices)
            
            # 2. Slopes
            # [N, R] matrix
            slope_vals = []
            for s in term.slope_cols:
                if s == "Intercept":
                    slope_vals.append(np.ones(len(df)))
                else:
                    # Assume numeric slope for now
                    mu, sigma = term.slope_stats.get(s, (0.0, 1.0))
                    vals = df[s].fillna(mu)
                    slope_vals.append(vals.values)
            
            slope_mat = np.stack(slope_vals, axis=1).astype(np.float32)
            Z_slopes.append(slope_mat)
            
        return Z_ind, Z_slopes

    def _extract_outcome(self, df, outcome):
        # Extract targets based on family
        # Returns numpy array
        cols = outcome.get_target_cols()
        family = outcome.family

        if isinstance(family, DiscreteTimeSurvival):
            # Map continuous/discrete times to bin indices
            time_vals = df[outcome.time].values
            event = df[outcome.event].values
            time_idx = self._time_to_index(family.time_bins, time_vals)
            return np.stack([time_idx, event], axis=1).astype(np.float32)

        if isinstance(family, DiscreteTimeCompetingRisks):
            time_vals = df[outcome.time].values
            cause_vals = df[outcome.cause].values
            time_idx = self._time_to_index(family.time_bins, time_vals)
            return np.stack([time_idx, cause_vals], axis=1).astype(np.float32)

        return df[cols].values.astype(np.float32)

    def _time_to_index(self, time_bins, time_array):
        """
        Convert raw times to discrete bin indices in [0, len(time_bins)-1].
        Uses searchsorted with right-closed intervals and clips out-of-range times.
        """
        tb = np.asarray(time_bins)
        idx = np.searchsorted(tb, time_array, side="left")
        idx = np.clip(idx, 0, len(tb) - 1)
        return np.where(pd.isna(time_array), np.nan, idx).astype(np.float32)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
