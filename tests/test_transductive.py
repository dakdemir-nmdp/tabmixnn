"""
Tests for transductive RE prediction (group-holdout generalization fix).

Verifies that adapt_unseen_groups() + predict_transductive() recovers R² for
unseen groups compared to the prior-mean fallback (unknown_strategy="learned").
"""
import numpy as np
import pandas as pd
import pytest
from tabmixnn import TabMixNN


def _make_sleepstudy_like(n_groups=18, n_per_group=10, seed=42):
    """Synthetic repeated-measures dataset with strong group-level random intercepts."""
    rng = np.random.default_rng(seed)
    group_re = rng.normal(0, 30, size=n_groups)  # large RE variance
    rows = []
    for g in range(n_groups):
        days = np.arange(n_per_group, dtype=float)
        reaction = 250 + 10 * days + group_re[g] + rng.normal(0, 10, n_per_group)
        rows.append(pd.DataFrame({"Subject": g, "Days": days, "Reaction": reaction}))
    return pd.concat(rows, ignore_index=True)


def test_adapt_unseen_groups_improves_r2():
    """
    Group-holdout: unseen groups should get R² > prior-mean fallback after conditional MAP adaptation.
    Uses embedding_dim=8 + hidden=[16] so the backbone learns a useful representation
    that the conditional MAP adaptation can exploit (pure LMM with emb=1 has limited headroom).
    """
    df = _make_sleepstudy_like(n_groups=18, n_per_group=10)
    train_groups = list(range(12))
    test_groups = list(range(12, 18))  # 6 unseen groups

    train_df = df[df["Subject"].isin(train_groups)].reset_index(drop=True)
    test_df = df[df["Subject"].isin(test_groups)].reset_index(drop=True)

    model = TabMixNN(
        "Reaction ~ Days + (1|Subject)",
        family="gaussian",
        embedding_dim=8,
        hidden_dims=[16],
        epochs=500,
        lr=0.05,
        kl_auto_scale=True,
        random_state=42,
        verbose=False,
    )
    model.fit(train_df)

    support_df = test_df[test_df.Days < 5].copy()
    test_df = test_df[test_df.Days >= 5].copy()
    y_true = test_df["Reaction"].values
    ss_tot = np.var(y_true) * len(y_true)

    # Baseline: prior-mean fallback (current behavior)
    y_prior = model.predict(test_df)
    r2_prior = 1 - np.sum((y_true - y_prior) ** 2) / ss_tot

    # Transductive conditional MAP adaptation
    y_transductive = model.predict_transductive(test_df, support_df=support_df, n_steps=300, lr=0.05)
    r2_transductive = 1 - np.sum((y_true - y_transductive) ** 2) / ss_tot

    # conditional MAP adaptation must strictly improve R²
    assert r2_transductive > r2_prior, (
        f"Transductive R²={r2_transductive:.3f} should exceed prior-mean R²={r2_prior:.3f}"
    )
    # Should recover strong signal (R² > 0.7 with large RE variance and good backbone)
    assert r2_transductive > 0.7, (
        f"Transductive R²={r2_transductive:.3f} should be > 0.7 for strong-RE synthetic data"
    )


def test_adapt_unseen_groups_seen_groups_unchanged():
    """
    For seen groups, predict_transductive() must return the same result as predict().
    """
    df = _make_sleepstudy_like(n_groups=10, n_per_group=10)
    model = TabMixNN(
        "Reaction ~ Days + (1|Subject)",
        family="gaussian",
        embedding_dim=1,
        hidden_dims=[],
        epochs=100,
        lr=0.05,
        kl_auto_scale=True,
        random_state=42,
        verbose=False,
    )
    model.fit(df)

    y_standard = model.predict(df)
    y_transductive = model.predict_transductive(df, support_df=df, n_steps=10)

    # All groups are seen — transductive should not change predictions
    np.testing.assert_allclose(y_standard, y_transductive, rtol=1e-4,
                               err_msg="predict_transductive changed predictions for seen groups")


def test_adapt_unseen_groups_returns_override_dict():
    """adapt_unseen_groups() returns a dict keyed by term_id with correct shape."""
    df = _make_sleepstudy_like(n_groups=6, n_per_group=5)
    train_df = df[df["Subject"] < 4].reset_index(drop=True)
    test_df = df[df["Subject"] >= 4].reset_index(drop=True)

    model = TabMixNN(
        "Reaction ~ Days + (1|Subject)",
        family="gaussian",
        embedding_dim=4,
        hidden_dims=[],
        epochs=50,
        lr=0.05,
        kl_auto_scale=True,
        random_state=42,
        verbose=False,
    )
    model.fit(train_df)

    override = model.adapt_unseen_groups(test_df, n_steps=5)
    assert isinstance(override, dict)
    assert len(override) > 0
    for tid, tensor in override.items():
        assert tensor.shape[0] == len(test_df), f"Override shape mismatch for {tid}"
        assert tensor.shape[2] == model.embedding_dim
