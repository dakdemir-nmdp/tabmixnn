import re
import pandas as pd
import numpy as np
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .families import DiscreteTimeSurvival, DiscreteTimeCompetingRisks

@dataclass
class OutcomeSpec:
    name: str
    family: Any # Family instance
    y: Optional[str] = None
    time: Optional[str] = None
    event: Optional[str] = None
    cause: Optional[str] = None
    
    def get_target_cols(self):
        if self.y: return [self.y]
        if self.cause: return [self.time, self.event, self.cause]
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
        # We rely on bias for intercept; do not include explicit intercept column
        self.intercept_included = False
            
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

    def _fit_fixed(self, df, fixed_part):
        # Use bias for intercept; do not include explicit intercept column
        self.intercept_included = False
        
        # Identify types
        self.fixed_cols = []
        for col_name in fixed_part['terms']:
            # Naively assume direct column names for now. 
            # TODO: Handle interactions (:), functions (log(x)), etc if needed.
            # Blueprint assumes numeric columns or categorical fixed effects.
            if col_name not in df.columns:
                raise ValueError(f"Column {col_name} not found in dataframe.")
            
            if pd.api.types.is_numeric_dtype(df[col_name]):
                # Store stats (kept for compatibility), but we keep raw scale for interpretability
                mu = df[col_name].mean()
                sigma = df[col_name].std() if df[col_name].std() != 0 else 1.0
                self.numeric_transformer_params[col_name] = (mu, sigma)
                self.fixed_cols.append({'name': col_name, 'type': 'numeric'})
            else:
                # Categorical
                # For Phase A, let's just do dummy encoding (one-hot) minus one level if intercept
                # Or just simple one-hot. 
                # Blueprint: "categorical fixed levels and contrast coding scheme"
                levels = sorted(df[col_name].dropna().unique())
                self.categorical_levels[col_name] = levels
                # If intercept included, drop first level to avoid collinearity
                dummy_levels = levels[1:] if self.intercept_included else levels
                self.categorical_dummy_levels[col_name] = dummy_levels
                self.fixed_cols.append({'name': col_name, 'type': 'categorical'})

    def _transform_fixed(self, df):
        parts = []
            
        for spec in self.fixed_cols:
            col = spec['name']
            if spec['type'] == 'numeric':
                mu, sigma = self.numeric_transformer_params[col]
                vals = df[col].fillna(mu).values.reshape(-1, 1)
                parts.append(vals)
            else:
                # Categorical
                levels = self.categorical_levels[col]
                dummy_levels = self.categorical_dummy_levels.get(col, levels)
                # Map to indices
                # pd.Categorical
                cat = pd.Categorical(df[col], categories=levels)
                # One hot
                drop_first = self.intercept_included
                dummies = pd.get_dummies(cat, drop_first=drop_first)
                # Reindex to ensure stable column order and dropped base handling
                dummies = dummies.reindex(columns=dummy_levels, fill_value=0)
                # For ML, full rank isn't always strictly required given regularization, 
                # but let's stick to simple one-hot for now.
                # If intercept is included, we usually drop first.
                # Let's simple OneHot all for now to avoid collinearity headaches in simple implementation
                # or rely on regularization.
                parts.append(dummies.values)
                
        if not parts:
            # Empty fixed effects?
            return np.zeros((len(df), 0))
            
        return np.hstack(parts).astype(np.float32)

    def _fit_random(self, df, random_parts_spec):
        self.random_terms = []
        for spec in random_parts_spec:
            group = spec['group']
            slopes = spec['slopes']
            
            # Map groups
            # If K-structure provided with explicit levels, honor that full ordering.
            maybe_cov = self.cov_struct_map.get(group, {})
            use_k_levels = maybe_cov.get("type") == "K" and "levels" in maybe_cov

            if use_k_levels:
                levels_list = list(maybe_cov["levels"])
                mapping = {val: i+1 for i, val in enumerate(levels_list)}
                train_levels = maybe_cov.get("train_levels", levels_list)
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
            # 0 is reserved for unknown
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
                    vals = (vals - mu) / sigma
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
        idx = np.searchsorted(tb, time_array, side="right")
        idx = np.clip(idx, 0, len(tb) - 1)
        return idx.astype(np.int64)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
