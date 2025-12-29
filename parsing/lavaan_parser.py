"""
Lavaan-style formula parser for output-level SEM.

Parses lavaan-style outcome equations (e.g., "y2 ~ y1") and converts them
to adjacency matrices for StaticStructureModel.

Phase 1 supports only the regression operator (~).
Future phases may add latent variables (=~) and covariances (~~).
"""

from dataclasses import dataclass
from typing import List
import torch


@dataclass
class RegressionEquation:
    """Represents: outcome ~ predictor1 + predictor2"""
    outcome: str
    predictors: List[str]


@dataclass
class LavanSpec:
    """Parsed lavaan specification."""
    regressions: List[RegressionEquation]


class LavanParser:
    """Parse lavaan-style formulas into adjacency matrices."""

    @staticmethod
    def parse(spec: str) -> LavanSpec:
        """
        Parse multi-line lavaan equations.

        Args:
            spec: Multi-line string with lavaan equations, e.g.:
                  "y2 ~ y1
                   y3 ~ y1 + y2"

        Returns:
            LavanSpec with parsed equations

        Raises:
            ValueError: If syntax is invalid or unsupported operators are used
        """
        lines = [l.strip() for l in spec.split('\n')
                 if l.strip() and not l.startswith('#')]
        regressions = []

        for line in lines:
            # Check for unsupported operators
            if '=~' in line:
                raise ValueError(
                    f"Latent variables (=~) not yet supported in Phase 1. "
                    f"Found in: {line}"
                )
            if '~~' in line:
                raise ValueError(
                    f"Covariances (~~) not yet supported in Phase 1. "
                    f"Found in: {line}"
                )

            # Parse regression operator
            if '~' in line:
                if line.count('~') > 1:
                    raise ValueError(f"Multiple ~ operators in line: {line}")

                lhs, rhs = line.split('~', 1)
                outcome = lhs.strip()
                predictors = [p.strip() for p in rhs.split('+') if p.strip()]

                if not outcome:
                    raise ValueError(f"Empty outcome in equation: {line}")
                if not predictors:
                    raise ValueError(f"No predictors in equation: {line}")

                regressions.append(RegressionEquation(outcome, predictors))
            else:
                raise ValueError(
                    f"Unsupported syntax (no recognized operator): {line}"
                )

        if not regressions:
            raise ValueError("No valid equations found in lavaan specification")

        return LavanSpec(regressions)

    @staticmethod
    def to_adjacency(spec: LavanSpec, outcome_names: List[str]) -> torch.Tensor:
        """
        Convert parsed spec to B matrix where B[i,j] ≠ 0 iff outcome_j → outcome_i.
        Validates DAG property (topological ordering).

        Args:
            spec: Parsed lavaan specification
            outcome_names: Ordered list of outcome names (defines topological order)

        Returns:
            torch.Tensor of shape [K, K] with lower-triangular structure

        Raises:
            ValueError: If DAG constraint is violated or unknown outcomes are referenced
        """
        K = len(outcome_names)
        name_to_idx = {name: i for i, name in enumerate(outcome_names)}
        B = torch.zeros(K, K)

        for eq in spec.regressions:
            if eq.outcome not in name_to_idx:
                raise ValueError(
                    f"Unknown outcome '{eq.outcome}' in equation. "
                    f"Valid outcomes: {outcome_names}"
                )
            i = name_to_idx[eq.outcome]

            for pred in eq.predictors:
                if pred not in name_to_idx:
                    # Could be a covariate (not an outcome) - skip silently
                    # This allows equations like "y2 ~ y1 + age" where age is a fixed effect
                    continue

                j = name_to_idx[pred]

                # DAG constraint: predictor must come before outcome in ordering
                if j >= i:
                    raise ValueError(
                        f"DAG violation: '{pred}' → '{eq.outcome}' requires "
                        f"'{pred}' to appear before '{eq.outcome}' in outcomes list.\n"
                        f"Current order: {outcome_names}\n"
                        f"Suggestion: Reorder outcomes so dependencies flow forward."
                    )

                # Initialize edge with small weight (will be learned if learnable=True)
                B[i, j] = 0.1

        return B
