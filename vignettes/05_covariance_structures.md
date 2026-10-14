# Vignette 05: Covariance Structures, KL Auto-Scale, and Approximation Diagnostics

This vignette covers three advanced topics:

1. **Covariance structures** — AR1, Kronecker (K), and Unstructured slopes
2. **`kl_auto_scale` heuristic** — how λ_KL is set automatically
3. **`approximation_error()` diagnostic** — measuring the sequential-vs-fixed-point gap

## Table of Contents

1. [Covariance Structures Overview](#covariance-structures-overview)
2. [AR1 Covariance](#ar1-covariance)
3. [Kronecker (K) Covariance](#kronecker-k-covariance)
4. [Unstructured Slope Covariance](#unstructured-slope-covariance)
5. [kl_auto_scale Heuristic](#kl_auto_scale-heuristic)
6. [approximation_error() Diagnostic](#approximation_error-diagnostic)

---

## Covariance Structures Overview

By default, TabMixNN uses IID covariance for random effects: each group's
embedding is drawn independently from N(0, σ²I). For structured data you can
specify a richer covariance via `cov_struct_map`:

```python
model = TabMixNN(
    formula="y ~ x + (1|group)",
    outcomes=outcomes,
    cov_struct_map={"group": {"type": "AR1"}},   # AR1 over group levels
)
```

Supported types: `"IID"` (default), `"AR1"`, `"K"` (known matrix),
`"Kronecker"`, `"UnstructuredSlope"`, `"CS"` (compound symmetry),
`"LowRank"`, `"Full"`, `"GP"`.

---

## AR1 Covariance

Use AR1 when group levels have a natural ordering (e.g. time points, ordered
categories) and you expect adjacent levels to be more similar.

```python
import numpy as np
import pandas as pd
from tabmixnn import TabMixNN, OutcomeSpec, Gaussian

# Simulate longitudinal data: 20 subjects × 10 time points
rng = np.random.default_rng(0)
n_subjects, n_times = 20, 10
subjects = np.repeat(np.arange(n_subjects), n_times).astype(str)
times = np.tile(np.arange(n_times), n_subjects).astype(float)

# AR1 random intercepts: correlated across time
rho_true = 0.8
u = np.zeros((n_subjects, n_times))
for t in range(1, n_times):
    u[:, t] = rho_true * u[:, t - 1] + rng.normal(0, 1, n_subjects)
u_i = u[np.repeat(np.arange(n_subjects), n_times),
        np.tile(np.arange(n_times), n_subjects)]

x = rng.standard_normal(n_subjects * n_times)
y = 3.0 + 1.5 * x + u_i + rng.normal(0, 0.5, n_subjects * n_times)

df = pd.DataFrame({"y": y, "x": x, "subject": subjects, "time": times})

outcomes = [OutcomeSpec(name="y", y="y", family=Gaussian())]

model = TabMixNN(
    formula="y ~ x + (1|subject)",
    outcomes=outcomes,
    cov_struct_map={"subject": {"type": "AR1"}},
    hidden_dims=[32],
    embedding_dim=8,
    epochs=100,
    lr=5e-3,
    kl_auto_scale=True,
    verbose=False,
)
model.fit(df)

# Variance summary reports marginal std (sigma / sqrt(1 - rho^2)), not innovation std
var_summary = model.encoder.get_variance_summary()
print("AR1 variance summary:", var_summary)
# term_vals["sigmas"][0]  → marginal std
# term_vals["innovation_sigma"][0]  → innovation std (sigma parameter)
# term_vals["rho"]  → AR1 correlation

preds = model.predict(df)
ss_res = np.sum((df["y"].values - preds) ** 2)
ss_tot = np.sum((df["y"].values - df["y"].mean()) ** 2)
print(f"R² = {1 - ss_res / ss_tot:.3f}")
```

**Key point**: `get_variance_summary()` reports the *marginal* standard
deviation σ / √(1 − ρ²), not the innovation std σ. This is the correct
quantity for interpreting variance components (e.g. ρ=0.8, σ=5 gives
marginal std ≈ 8.33, not 5.0).

---

## Kronecker (K) Covariance

Use the K (known matrix) covariance when you have a pre-computed similarity
or kinship matrix between groups (e.g. genetic relatedness, geographic
proximity).

```python
import numpy as np
import pandas as pd
from tabmixnn import TabMixNN, OutcomeSpec, Gaussian

rng = np.random.default_rng(1)
n_groups = 15
n_per_group = 20

# Simulate a kinship-like matrix (positive definite)
A = rng.standard_normal((n_groups, n_groups))
K = A @ A.T / n_groups + np.eye(n_groups) * 0.1  # PD

groups = np.repeat(np.arange(n_groups), n_per_group).astype(str)
x = rng.standard_normal(n_groups * n_per_group)

# True random effects drawn from N(0, K)
L = np.linalg.cholesky(K)
u_raw = rng.standard_normal(n_groups)
u = L @ u_raw
u_i = u[np.repeat(np.arange(n_groups), n_per_group)]
y = 2.0 + 1.0 * x + u_i + rng.normal(0, 0.5, n_groups * n_per_group)

df = pd.DataFrame({"y": y, "x": x, "group": groups})

outcomes = [OutcomeSpec(name="y", y="y", family=Gaussian())]

model = TabMixNN(
    formula="y ~ x + (1|group)",
    outcomes=outcomes,
    cov_struct_map={
        "group": {
            "type": "K",
            "matrix": K,                          # [n_groups, n_groups] array
            "full_levels": [str(i) for i in range(n_groups)],
            "train_levels": [str(i) for i in range(n_groups)],
        }
    },
    hidden_dims=[32],
    embedding_dim=8,
    epochs=100,
    lr=5e-3,
    kl_auto_scale=True,
    enforce_centering=True,
    verbose=False,
)
model.fit(df)

preds = model.predict(df)
r2 = 1 - np.sum((df["y"].values - preds) ** 2) / np.var(df["y"].values) / len(df)
print(f"K-covariance model R² = {r2:.3f}")
```

---

## Unstructured Slope Covariance

Use `UnstructuredSlope` when you have random slopes and want to learn the
full covariance between slope dimensions (intercept + slope correlations).

```python
import numpy as np
import pandas as pd
from tabmixnn import TabMixNN, OutcomeSpec, Gaussian

rng = np.random.default_rng(2)
n_groups, n_per = 20, 15
groups = np.repeat(np.arange(n_groups), n_per).astype(str)
x = rng.standard_normal(n_groups * n_per)

# True correlated random intercept + slope
u0 = rng.normal(0, 2.0, n_groups)
u1 = 0.6 * u0 + rng.normal(0, 1.0, n_groups)   # correlated with intercept
u0_i = u0[np.repeat(np.arange(n_groups), n_per)]
u1_i = u1[np.repeat(np.arange(n_groups), n_per)]
y = 3.0 + 2.0 * x + u0_i + u1_i * x + rng.normal(0, 0.5, n_groups * n_per)

df = pd.DataFrame({"y": y, "x": x, "group": groups})
outcomes = [OutcomeSpec(name="y", y="y", family=Gaussian())]

model = TabMixNN(
    formula="y ~ x + (1 + x|group)",
    outcomes=outcomes,
    cov_struct_map={"group": {"type": "UnstructuredSlope"}},
    hidden_dims=[32],
    embedding_dim=8,
    epochs=150,
    lr=5e-3,
    kl_auto_scale=True,
    enforce_centering=True,
    verbose=False,
)
model.fit(df)

preds = model.predict(df)
ss_res = np.sum((df["y"].values - preds) ** 2)
ss_tot = np.sum((df["y"].values - df["y"].mean()) ** 2)
print(f"UnstructuredSlope R² = {1 - ss_res / ss_tot:.3f}")
```

---

## kl_auto_scale Heuristic

`kl_auto_scale=True` (the default) sets λ_KL = 1 / mean_group_size after
the design is compiled. This is a β-VAE-style empirical damping heuristic —
it is **not** derived from the ELBO. The ELBO-correct weight is λ_KL = 1
with the mini-batch factor (B/N) handled in the training loop.

The heuristic works well in practice because it scales the KL penalty
relative to the NLL: with large groups the NLL dominates and the KL needs
less damping; with small groups the KL needs more damping to avoid
over-regularizing the random effects.

```python
from tabmixnn import TabMixNN, OutcomeSpec, Gaussian
import pandas as pd, numpy as np

rng = np.random.default_rng(3)
df = pd.DataFrame({
    "y": rng.standard_normal(200),
    "x": rng.standard_normal(200),
    "group": (np.arange(200) % 20).astype(str),  # 20 groups × 10 obs
})
outcomes = [OutcomeSpec(name="y", y="y", family=Gaussian())]

model = TabMixNN(
    formula="y ~ x + (1|group)",
    outcomes=outcomes,
    hidden_dims=[16],
    epochs=1,
    kl_auto_scale=True,   # default
    verbose=False,
)
model.fit(df)

# After fit(), lambda_kl ≈ 1/10 = 0.1 (mean group size = 10)
print(f"lambda_kl after fit: {model.lambda_kl:.4f}")  # → ~0.1

# To use a fixed lambda instead:
model2 = TabMixNN(
    formula="y ~ x + (1|group)",
    outcomes=outcomes,
    hidden_dims=[16],
    epochs=1,
    kl_auto_scale=False,
    lambda_kl=0.05,        # user-specified
    verbose=False,
)
model2.fit(df)
print(f"lambda_kl (fixed): {model2.lambda_kl:.4f}")   # → 0.05
```

---

## approximation_error() Diagnostic

`CompositeStructuralModel.approximation_error(xi)` measures how far the
sequential (one-pass) approximation deviates from the true fixed-point
solution. Values < 0.01 indicate the approximation is negligible.

This is the M3 diagnostic from the theory critique: the paper's Eq.22
defines a fixed-point equation, but `CompositeStructuralModel` uses a
single sequential pass for stability. Use this diagnostic to decide whether
to switch to `HybridStructuralModel` (exact fixed-point, slower).

```python
import torch
import numpy as np
import pandas as pd
from tabmixnn import TabMixNN, OutcomeSpec, Gaussian

rng = np.random.default_rng(4)
n = 200
df = pd.DataFrame({
    "y": rng.standard_normal(n),
    "x": rng.standard_normal(n),
    "group": (np.arange(n) % 10).astype(str),
})
outcomes = [OutcomeSpec(name="y", y="y", family=Gaussian())]

# Build a model with both static and dynamic structure enabled
model = TabMixNN(
    formula="y ~ x + (1|group)",
    outcomes=outcomes,
    hidden_dims=[32],
    use_static_structure=True,
    use_dynamic_structure=True,
    epochs=1,
    verbose=False,
)
model.fit(df)

# Find a CompositeStructuralModel layer in the backbone
composite = None
for layer in model.backbone.hidden_layers:
    if hasattr(layer, "structure") and layer.structure is not None:
        from tabmixnn.layers.structural import CompositeStructuralModel
        if isinstance(layer.structure, CompositeStructuralModel):
            composite = layer.structure
            break

if composite is not None:
    xi = torch.randn(16, 32)   # random batch [batch, dim]
    err = composite.approximation_error(xi)
    print(f"Approximation error: {err:.4f}")
    # < 0.01 → sequential approximation is negligible
    # > 0.05 → consider switching to HybridStructuralModel
else:
    print("No CompositeStructuralModel found (model may use IID structure).")
```

**Interpretation**:
- `err < 0.01`: Sequential approximation is negligible — use `CompositeStructuralModel`.
- `0.01 ≤ err < 0.05`: Small but non-negligible gap — acceptable for most use cases.
- `err ≥ 0.05`: Significant gap — consider `HybridStructuralModel` for exact semantics.

---

## Summary

| Feature | Parameter | Notes |
|---------|-----------|-------|
| AR1 covariance | `cov_struct_map={"g": {"type": "AR1"}}` | Ordered levels; reports marginal std |
| K covariance | `cov_struct_map={"g": {"type": "K", "matrix": K, ...}}` | Known similarity matrix |
| Unstructured slopes | `cov_struct_map={"g": {"type": "UnstructuredSlope"}}` | Learns slope correlations |
| KL auto-scale | `kl_auto_scale=True` (default) | Sets λ_KL = 1/mean_group_size |
| Approximation error | `composite.approximation_error(xi)` | M3 diagnostic; float in [0, 1] |

For more on the theory behind these choices, see the paper arXiv:2512.23787
and the theory critique notes in `AGENTS.md`.
