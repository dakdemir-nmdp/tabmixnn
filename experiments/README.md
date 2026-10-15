# TabMixNN Benchmark Experiments

Exploratory benchmark scripts. These do **not** reproduce Tables 2–4 of
arXiv:2512.23787: the dataset set and protocols differ, and original run artifacts
are unavailable. See `../docs/methods-audit-2026-10-15.md`.

For the validated methods check (train-only preprocessing, matched splits,
independent mixed-model reference, three seeds):

```bash
uv pip install --python .venv/bin/python statsmodels scipy
.venv/bin/python experiments/validate_sleepstudy.py
```

Aggregate results are in `results/methods_sleepstudy.json`. This is a focused
Sleepstudy validation, not evidence of performance across the full benchmark suite.

## Setup

```bash
# Create uv venv (avoids macOS system Python MPS segfault)
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install numpy pandas patsy scikit-learn tqdm xgboost requests
pip install -e . --no-deps
```

## Datasets

Downloaded automatically from [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/)
and cached in `experiments/data/`.

| Dataset     | Task           | n     | Groups | Formula |
|-------------|----------------|-------|--------|---------|
| Sleepstudy  | Regression     | 180   | 18 subjects | `Reaction ~ Days + (1+Days\|Subject)` |
| ChickWeight | Regression     | 578   | 50 chicks   | `weight ~ Time + Diet + (1\|Chick)` |
| Machines    | Regression     | 54    | 6 workers   | `score ~ Machine + (1\|Worker)` |
| Oxboys      | Regression     | 234   | 26 subjects | `height ~ age + (1+age\|Subject)` |
| Loblolly    | Regression     | 84    | 14 seeds    | `height ~ age + (1\|Seed)` |
| Aids2       | Classification | 2843  | states      | `status ~ age + sex + T.categ + (1\|state)` |
| Males       | Classification | 4360  | individuals | `union ~ school + exper + ethn + (1\|nr)` |
| VerbAgg     | Classification | 7584  | subjects    | `r2 ~ Anger + Gender + btype + situ + mode + (1\|id)` |
| Train       | Classification | 2929  | individuals | `choice ~ price + time + change + comfort + (1\|id)` |

## Running

```bash
cd /path/to/tabmixnn

# Regression (Tables 2-3): ~5-10 min
python experiments/run_regression.py

# Classification (Table 4): ~10-20 min
python experiments/run_classification.py
```

Results are saved to `experiments/results/`:
- `regression_results.csv` — RMSE and R² per dataset
- `classification_results.csv` — AUC and accuracy per dataset

## TabMixNN Configuration

All runs use:
- `hidden_dims=[32]`, `embedding_dim=8`
- `learning_rate=0.005` (paper ablation optimum)
- `kl_auto_scale=True` (λ_KL = 1/n̄_group, FIX-1)
- `epochs=100`, `batch_size=32` (regression) / `64` (classification)
- 5-fold CV, `random_state=42`

## Notes

- XGBoost baseline uses the same features but no random-effects structure.
- Group columns are included as plain features for XGBoost so it has access
  to the same information (but cannot model the RE covariance structure).
- SSL certificate verification is disabled for Rdatasets downloads on macOS
  (system Python 3.10 certificate issue); data is cached after first download.
