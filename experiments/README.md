# Reproducible manuscript experiments

The canonical manuscript suite is `run_manuscript_benchmarks.py`. It evaluates
all nine datasets supported by `data_utils.py` in two prediction settings:

- **A: known groups.** Hold out 25% (rounded down, minimum one) of observations
  separately within every group having at least two observations. At least one
  row remains for each tested group. Singleton groups remain training-only.
  This is random observation holdout, not future-time forecasting.
- **B: unseen groups.** Hold out 25% (rounded down, minimum one) of entire
  groups. No test outcomes or labeled support enter fitting or adaptation.

Both splits use seed 20261015 and are shared by every comparator. Source
hashes and explicit row positions are saved; group identifiers are not exported.

## Run

From the repository root, using its editable-install virtual environment:

```sh
uv pip install --python .venv/bin/python statsmodels scipy xgboost matplotlib
# R must have lme4 and jsonlite installed.
.venv/bin/python experiments/run_manuscript_benchmarks.py
.venv/bin/python experiments/run_manuscript_ablation.py
.venv/bin/python experiments/create_feature_ablation_plots.py
.venv/bin/python experiments/build_manuscript_tables.py
.venv/bin/python -m pytest
```

`--resume` skips completed dataset/regime pairs only after exact protocol and
benchmark-source-hash checks. `--datasets Sleepstudy --epochs 5 --seeds 42
--output experiments/data/smoke.json` is a smoke run, not a manuscript run.
Canonical tables require all 18 dataset/regime pairs and the separate ablations.

## Protocol

TabMixNN: one hidden layer of width 32, embedding width 8, UN covariance,
2,000 full-batch Adam updates, learning rate .01, KL multiplier one, zero
weight decay, no dropout/centering, and no early stopping. Three model seeds:
42, 123, 456. Full batches retain source order; a benchmark-only adapter caches
tensors and uses the unchanged package training loop. A gradient-equivalence
test checks this adapter against ordinary DataLoader collation.

Fixed NN removes random effects and otherwise uses the same architecture and
budget. Sleepstudy additionally has a scalar linear VI control. lme4 uses
ML for LMMs and Laplace approximate ML (`nAGQ=1`) for binomial GLMMs; known
groups use conditional fitted effects, new groups use zero effects. The latter
is a plug-in prediction, not integration over the random-effect distribution.
Convergence messages and singular fits are retained in the aggregate output.

XGBoost uses 200 trees, depth 6, learning rate .05. Random Forest uses 200 trees
and minimum leaf size 2. Both use seed 42 and one thread. All methods have
identical substantive predictors. Trees additionally receive training-fitted
one-hot group identity in A; group identity is absent from their B predictors.
Numeric standardization, dummy levels and target scaling use training rows only.
Unknown categorical levels map to the training reference coding. No model is
selected using test labels, and there is no hyperparameter search.

Regression RMSE/MAE are back-transformed to original units. Classification
reports AUROC, Brier score, log loss and accuracy. Tables summarize neural
metrics by mean and sample SD across seeds on the **same split**. This SD
measures initialization variability, not sampling uncertainty or a confidence
interval. No cross-dataset average of raw RMSE is meaningful.

Aids2 predicts recorded final vital status, not fixed-horizon survival risk.
Its varying observation times limit interpretation. Males predicts union
membership; VerbAgg predicts a dichotomized response; Train predicts choice1.
No clinical use or external validation is claimed.

## Artifacts and historical scripts

`results/manuscript_benchmarks.json` contains protocols, versions, source
hashes, split manifests, per-seed metrics and baseline diagnostics.
`results/manuscript_ablation.json` contains matched Sleepstudy component
comparisons. `results/manuscript_figures.json` records descriptive plot provenance.
Downloaded public records, intermediate R inputs and per-row predictions stay
in ignored `data/`; no source records are committed.

The earlier `run_regression.py`, `run_classification.py`,
`run_gh_validated.py`, and `validate_sleepstudy.py` are retained as historical
or targeted diagnostics. Their splits, budgets or predictors differ. They do
not generate the revised manuscript tables, and their results must not be
pooled with this suite. Unsupported historical Abalone/other benchmark rows
are excluded from the revised manuscript. Multitask and survival interfaces
are described as implemented capabilities, not empirically validated here.
