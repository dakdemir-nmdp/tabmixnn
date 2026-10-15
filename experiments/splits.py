"""Shared split manifests for TabMixNN diagnostic benchmarks.

All benchmark scripts must import splits from here so that TabMixNN and every
baseline use identical train/test row indices.  Splits are deterministic given
the dataset length and SPLIT_SEED; they do not depend on outcome values.

Split types
-----------
obs_kfold(df, n_splits)
    Standard K-fold on observations.  Both TabMixNN and XGBoost use the same
    folds.  Preprocessing (standardization, encoding) is fitted on the training
    partition only and applied to the test partition.

group_holdout(df, group_col, holdout_frac, seed)
    Hold out a fixed fraction of groups entirely.  Correct evaluation for
    generalisation to unseen groups.  Returns (train_mask, test_mask).

These are the only two split types used in the diagnostic scripts.  The paper's
original folds are not available; no claim is made that these reproduce them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SPLIT_SEED = 20261015  # fixed; never change without bumping SPLIT_VERSION
SPLIT_VERSION = "1"


def obs_kfold(df: pd.DataFrame, n_splits: int = 5):
    """Yield (train_idx, test_idx) arrays for observation-level K-fold.

    Indices are integer positions into df (iloc-compatible).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SPLIT_SEED)
    yield from kf.split(df)


def group_holdout(
    df: pd.DataFrame,
    group_col: str,
    holdout_frac: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean (train_mask, test_mask) holding out a fraction of groups.

    The held-out groups are chosen deterministically from SPLIT_SEED.
    """
    rng = np.random.default_rng(SPLIT_SEED)
    groups = df[group_col].unique()
    n_holdout = max(1, int(len(groups) * holdout_frac))
    holdout_groups = rng.choice(groups, size=n_holdout, replace=False)
    test_mask = df[group_col].isin(holdout_groups).to_numpy()
    return ~test_mask, test_mask


def manifest_meta() -> dict:
    """Return metadata to embed in saved results for traceability."""
    return {"split_seed": SPLIT_SEED, "split_version": SPLIT_VERSION}
