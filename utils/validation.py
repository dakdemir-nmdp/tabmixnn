from typing import Any, Dict, Tuple
import pandas as pd


def split_train_val(df: pd.DataFrame, val_fraction: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simple utility to split a dataframe into train/validation sets."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    val_df = df.sample(frac=val_fraction, random_state=random_state)
    train_df = df.drop(val_df.index).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return train_df, val_df


class ValidationRunner:
    """Thin wrapper that delegates validation loss computation to the model."""

    def __init__(self, model: Any):
        self.model = model

    def evaluate(self, val_data: Dict[str, Any]) -> float:
        """Compute validation loss via the model's internal evaluator."""
        if not hasattr(self.model, "_evaluate_loss"):
            raise AttributeError("Model must implement _evaluate_loss")
        return self.model._evaluate_loss(val_data)
