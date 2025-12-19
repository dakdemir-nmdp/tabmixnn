"""
Domain Specific Language (DSL) for Universal SAM-NN.

⚠️ EXPERIMENTAL: This module is not yet integrated with the main TabMixNN API.
It provides a preview of a declarative model specification syntax for future releases.
Use `TabMixNN(formula="...")` for the current supported syntax.

Allows clinicians and researchers to define complex hierarchical models
in a few lines of intuitive text.

Example:
```
model_spec = '''
# Input layer
fixed: age + hla_match + kps + conditioning_intensity
random: (1 | center_id) + (age | donor_id)

# Latent manifolds
manifold_1: grid=(8,8), spde=True, alpha=2
manifold_2: grid=(4,4), spde=True, sem=True
manifold_3: grid=(8,8), spde=False, sem=True

# Output SEM (causal structure)
acute_gvhd ~ manifold_1 + donor_type
chronic_gvhd ~ acute_gvhd + manifold_2
relapse ~ manifold_1 + conditioning_intensity  
death ~ acute_gvhd + chronic_gvhd + relapse + manifold_3
'''
```
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FixedEffectsSpec:
    """Specification for fixed effects."""
    terms: List[str]
    
    def __repr__(self):
        return f"Fixed({' + '.join(self.terms)})"


@dataclass
class RandomEffectTerm:
    """Single random effect term: (slopes | group)"""
    slopes: List[str]  # ['1'] for intercept-only, ['1', 'age'] for random slopes
    group: str
    
    def __repr__(self):
        slopes_str = ' + '.join(self.slopes)
        return f"({slopes_str} | {self.group})"


@dataclass
class RandomEffectsSpec:
    """Specification for random effects."""
    terms: List[RandomEffectTerm]
    
    def __repr__(self):
        return ' + '.join(str(t) for t in self.terms)


@dataclass
class ManifoldSpec:
    """Specification for a SAM manifold layer."""
    name: str
    grid_shape: Tuple[int, int]
    use_spde: bool = True
    use_sem: bool = False
    spde_alpha: int = 2
    activation: str = "relu"
    dropout: float = 0.0
    
    def __repr__(self):
        return f"{self.name}: grid={self.grid_shape}, spde={self.use_spde}, sem={self.use_sem}"


@dataclass
class SEMEquation:
    """Single equation in the output SEM."""
    outcome: str
    predictors: List[str]
    
    def __repr__(self):
        return f"{self.outcome} ~ {' + '.join(self.predictors)}"


@dataclass
class OutputSEMSpec:
    """Specification for output SEM (causal header)."""
    equations: List[SEMEquation]
    
    def __repr__(self):
        return '\n'.join(str(eq) for eq in self.equations)


@dataclass
class UniversalSAMSpec:
    """Complete specification for Universal SAM-NN."""
    fixed: FixedEffectsSpec
    random: RandomEffectsSpec
    manifolds: List[ManifoldSpec]
    output_sem: Optional[OutputSEMSpec] = None
    
    def __repr__(self):
        lines = [
            "Universal SAM-NN Specification:",
            f"  Fixed: {self.fixed}",
            f"  Random: {self.random}",
            f"  Manifolds ({len(self.manifolds)}):"
        ]
        for m in self.manifolds:
            lines.append(f"    {m}")
        if self.output_sem:
            lines.append(f"  Output SEM:")
            for eq in self.output_sem.equations:
                lines.append(f"    {eq}")
        return '\n'.join(lines)


class SAMNNParser:
    """
    Parser for Universal SAM-NN DSL.
    
    Converts human-readable specifications into structured configs.
    """
    
    @staticmethod
    def parse(spec_text: str) -> UniversalSAMSpec:
        """
        Parse a complete model specification.
        
        Args:
            spec_text: Multi-line string with model specification
        
        Returns:
            UniversalSAMSpec object
        """
        lines = [line.strip() for line in spec_text.split('\n')]
        lines = [line for line in lines if line and not line.startswith('#')]
        
        fixed = None
        random = None
        manifolds = []
        sem_equations = []
        
        for line in lines:
            if line.startswith('fixed:'):
                fixed = SAMNNParser._parse_fixed(line)
            elif line.startswith('random:'):
                random = SAMNNParser._parse_random(line)
            elif line.startswith('manifold_'):
                manifold = SAMNNParser._parse_manifold(line)
                manifolds.append(manifold)
            elif '~' in line:
                equation = SAMNNParser._parse_sem_equation(line)
                sem_equations.append(equation)
        
        output_sem = OutputSEMSpec(sem_equations) if sem_equations else None
        
        return UniversalSAMSpec(
            fixed=fixed,
            random=random,
            manifolds=manifolds,
            output_sem=output_sem
        )
    
    @staticmethod
    def _parse_fixed(line: str) -> FixedEffectsSpec:
        """Parse fixed effects: fixed: age + hla_match + kps"""
        content = line.split(':', 1)[1].strip()
        terms = [t.strip() for t in content.split('+')]
        return FixedEffectsSpec(terms)
    
    @staticmethod
    def _parse_random(line: str) -> RandomEffectsSpec:
        """Parse random effects: random: (1 | center_id) + (age | donor_id)"""
        content = line.split(':', 1)[1].strip()
        
        # Extract all (slopes | group) patterns
        pattern = r'\(([^|]+)\|([^)]+)\)'
        matches = re.findall(pattern, content)
        
        terms = []
        for slopes_str, group in matches:
            slopes = [s.strip() for s in slopes_str.split('+')]
            terms.append(RandomEffectTerm(slopes, group.strip()))
        
        return RandomEffectsSpec(terms)
    
    @staticmethod
    def _parse_manifold(line: str) -> ManifoldSpec:
        """
        Parse manifold spec: manifold_1: grid=(8,8), spde=True, alpha=2
        """
        # Extract name and config
        name, config_str = line.split(':', 1)
        name = name.strip()

        # Special handling for grid tuple to avoid naive comma splitting
        config = {}
        grid_match = re.search(r'grid=\(([^)]+)\)', config_str)
        if grid_match:
            nums = grid_match.group(1).split(',')
            config['grid'] = tuple(int(n.strip()) for n in nums if n.strip())
            # Remove the grid portion from the string for subsequent parsing
            config_str = re.sub(r'grid=\([^)]+\)', '', config_str)

        # Parse remaining key=value pairs
        for item in [i.strip() for i in config_str.split(',') if i.strip()]:
            if '=' not in item:
                continue
            key, value = item.split('=', 1)
            key = key.strip()
            value = value.strip()

            if value.lower() in ('true', 'false'):
                config[key] = value.lower() == 'true'
            elif value.isdigit():
                config[key] = int(value)
            elif value.replace('.', '', 1).isdigit():
                config[key] = float(value)
            else:
                config[key] = value

        return ManifoldSpec(
            name=name,
            grid_shape=config.get('grid', (8, 8)),
            use_spde=config.get('spde', True),
            use_sem=config.get('sem', False),
            spde_alpha=config.get('alpha', 2),
            activation=config.get('activation', 'relu'),
            dropout=config.get('dropout', 0.0)
        )
    
    @staticmethod
    def _parse_sem_equation(line: str) -> SEMEquation:
        """Parse SEM equation: death ~ acute_gvhd + chronic_gvhd + relapse"""
        outcome, predictors_str = line.split('~', 1)
        outcome = outcome.strip()
        predictors = [p.strip() for p in predictors_str.split('+')]
        return SEMEquation(outcome, predictors)


def parse_model_spec(spec_text: str) -> UniversalSAMSpec:
    """
    Convenience function to parse model specification.
    
    Example:
    ```python
    spec = parse_model_spec('''
        fixed: age + hla_match
        random: (1 | center_id)
        manifold_1: grid=(8,8), spde=True
        death ~ manifold_1 + age
    ''')
    ```
    """
    return SAMNNParser.parse(spec_text)


# Example specifications for common use cases

HCT_MULTITASK_SPEC = """
# HCT Multitask Outcome Prediction

# Fixed effects (clinical covariates)
fixed: age + hla_match + kps + conditioning_intensity + donor_type

# Random effects (center and donor variability)
random: (1 | center_id) + (1 + age | donor_id)

# Latent manifolds (hierarchical representation)
manifold_1: grid=(8,8), spde=True, alpha=2, activation=relu
manifold_2: grid=(4,4), spde=True, sem=True, activation=gelu
manifold_3: grid=(8,8), spde=False, sem=True, activation=relu

# Output SEM (causal structure of outcomes)
acute_gvhd ~ manifold_1 + donor_type
chronic_gvhd ~ acute_gvhd + manifold_2 + conditioning_intensity
relapse ~ manifold_1 + conditioning_intensity
death ~ acute_gvhd + chronic_gvhd + relapse + manifold_3
"""

GENOMIC_PREDICTION_SPEC = """
# Genomic Prediction with Kinship

# Fixed effects
fixed: sex + age + population

# Random effects (additive genetic effects)
random: (1 | individual_id)

# Latent manifolds
manifold_1: grid=(16,16), spde=True, alpha=2
manifold_2: grid=(8,8), spde=True, sem=False

# Output
trait ~ manifold_1 + manifold_2 + sex
"""

LONGITUDINAL_SPEC = """
# Longitudinal Trajectory Modeling

# Fixed effects
fixed: time + treatment + baseline_score

# Random effects (individual trajectories)
random: (1 + time | patient_id) + (1 | site_id)

# Latent manifolds
manifold_1: grid=(10,10), spde=True, alpha=2
manifold_2: grid=(5,5), spde=True, sem=True

# Output
outcome ~ manifold_1 + time + treatment
"""


__all__ = [
    'UniversalSAMSpec',
    'FixedEffectsSpec',
    'RandomEffectsSpec',
    'ManifoldSpec',
    'OutputSEMSpec',
    'SAMNNParser',
    'parse_model_spec',
    'HCT_MULTITASK_SPEC',
    'GENOMIC_PREDICTION_SPEC',
    'LONGITUDINAL_SPEC',
]
