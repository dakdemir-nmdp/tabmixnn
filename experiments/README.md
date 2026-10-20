# Reproducible manuscript experiments

The canonical manuscript suite is `run_manuscript_benchmarks.py`. It evaluates
all ten datasets supported by `data_utils.py` in two prediction settings:

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
Primary tables require all 20 dataset/regime pairs and the separate ablations.

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
The binomial routine tries bobyqa, then nloptwrap and nlminbwrap after errors;
it retains the first successful fit without reference to test performance.
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

Classification outcomes are contraceptive use (Contraception), the recorded
"moderate or severe" outcome (Toenail), a dichotomized response (VerbAgg),
choice1 (Train), and normalized exam score greater than zero (Exam).
The Exam threshold is fixed, not the sample median. The dataset set was
revised after an earlier evaluation and is exploratory. Sleepstudy and
ChickWeight have 180 and 578 rows; the other eight datasets have at least 919.
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

## Deep revision: split sensitivity and mathematical checks

The primary analysis retains split seed 20261015. An exploratory extension,
specified after the primary analysis, fixes four additional seeds before
executing their fits. All five splits use the same three neural seeds, training
budget and comparator settings. No split or neural initialization is selected by performance.

```sh
for split_seed in 20261016 20261017 20261018 20261019; do
  .venv/bin/python experiments/run_manuscript_benchmarks.py \
    --split-seed "$split_seed" \
    --output "experiments/results/manuscript_split_${split_seed}.json"
done
.venv/bin/python experiments/summarize_split_sensitivity.py
.venv/bin/python experiments/verify_variational_identities.py
```

The summary script requires the exact dataset/regime inventory, matching
execution code hashes (including the R reference), a common training protocol,
and matching current source/data hashes. It independently reconstructs every metric from the ignored
local prediction files, then generates two split-sensitivity tables and one
calibration table. Neural metrics are averaged over initializations within
split before summarizing over splits. Overlapping holdouts and repeated small-
group partitions prevent interpreting split SDs as independent standard errors.
Calibration intercept/slope and bin summaries use held-out predictions only
for evaluation, not for recalibration or model selection. The symbolic script
requires SymPy; its output is an explicit finite-dimensional algebra check,
not a formal proof of all model properties. Counts are computed during
validation: the current five-split suite contains 100 dataset/regime/split
pairs, 930 model records and 3,240 metric checks. The primary split contains
186 model records. Seven primary tables and three sensitivity/calibration
tables are generated from these results and the separate ablation artifact.

Every benchmark file records hashes of the model source modules as well
as experiment scripts. The current ten-dataset payloads match the delivered
execution source. The following notes describe the earlier nine-dataset
October 16–17 runs; they do not describe an exception needed to verify the
current payloads.

### Historical nine-dataset execution notes

The earlier primary 168 model records were rerun after the API repairs;
no metric changed by more than 1e-7 from session 89.

One VerbAgg-A reference fit for split 20261018 encountered lme4's
`Downdated VtV is not positive definite` error inside the combined runner.
The identical independent R command converged on byte-identical prepared CSVs.
The transparent execution workaround is:

```sh
.venv/bin/python experiments/resume_verified_reference.py \
  --dataset VerbAgg --regime A --split-seed 20261018 \
  --output experiments/results/manuscript_split_20261018.json
.venv/bin/python experiments/run_manuscript_benchmarks.py \
  --split-seed 20261018 --output experiments/results/manuscript_split_20261018.json --resume
```

This helper independently executes the original R fit, requires convergence,
verifies regenerated input hashes and replays its predictions without changing
optimizer settings or selecting by test performance. Provenance is stored in
that model record. Use it only for an interrupted pair with its prepared CSVs
still present; completed pairs are not overwritten. The cause of the runner-
dependent numerical behavior has not been isolated.

After execution, fourteen whitespace-only blank lines in `core/model.py` were
cleaned. `results/source_equivalence.json` preserves both SHA-256 hashes and
the exact line replacements needed to reconstruct the executed source from
the delivered file. Their Python syntax trees are identical; no functional
model change followed the runs. Execution hashes are retained unchanged.
A strict `--resume` against these completed records requires the reconstructed
executed file; a fresh reproduction uses the delivered source.
