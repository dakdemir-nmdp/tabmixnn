# Package training and covariance repairs — session 86

Internal validation record, 2026-10-15. Baseline: `6673fa0` on
`revision/methods-2026-10-15`. This pass changes package behavior, not only
benchmark settings. It is not an exhaustive proof of the package's mathematics.

## Reproduced defects and changes

1. **Gaussian mean initialization.** The old initializer targeted an encoder
   bias that the formula-based constructor disables. A nonlinear model with
   training mean 300 therefore initialized its output bias near zero (-0.163
   in the regression test). Gaussian learned output heads now start at their
   own observed training-target mean. Residual scales use observed training
   standard deviations, floored at 0.001. Validation targets are excluded.
   Identity heads retain the existing linear least-squares initialization;
   coupled output SEM heads are excluded from this mean initialization because
   their intercepts are transformed jointly. Random output weights remain
   random, so initial predictions are close to, not identically, the mean.
2. **Bias regularization.** Biases and one-dimensional normalization scales
   receive zero weight decay. Matrix weights retain the requested decay;
   variational, covariance, and observation-family parameters retain their
   existing zero-decay treatment. Penalizing a large output intercept otherwise
   creates an artificial pull toward zero tied to the origin of the response.
   This does not claim complete response-scale invariance.
3. **Repeated fitting.** `fit()` now starts afresh: training-only design,
   network, constructor family state, optimizer, and histories. Previously a
   second fit could silently map all new groups to unknown, retain the old
   optimizer learning rate, and reset only selected estimates. A reused object
   now matches a fresh object at a fixed seed, including a changed learning rate.
   `reuse_design` explicitly preserves preprocessing only; it is not a warm start.
4. **Validation row selection.** Internal validation splits use row positions.
   Repeated dataframe index labels previously caused dropping every training
   row. Empty training or requested validation partitions now fail explicitly.
5. **AR(p) stability.** Componentwise `tanh(phi_raw)` does not ensure a stable
   AR polynomial. With two coefficients of 0.9, the previous 100-step impulse
   response reached 1.936e17. `phi_raw` now parameterizes partial correlations
   kappa, with the recursion

   `phi[k,j] = phi[k-1,j] - kappa[k] * phi[k-1,k-j]`, `phi[k,k] = kappa[k]`.

   Each kappa is in (-1,1), with a 1e-6 numerical margin. The stable-polynomial
   construction follows the partial-correlation approach described in
   [statsmodels' stationarity transform documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.tools.constrain_stationary_univariate.html).
   Tests check companion roots for orders 1, 2, 3, and 5, an impulse response,
   first-order and white-noise limits, and double-precision finite-difference
   gradients. These numerical checks are not a formal proof for all inputs.
   Extreme coefficients/high orders can still be poorly conditioned in finite
   precision. ARMA retains **zero pre-sample states**, not stationary initial
   covariance. MA invertibility is not imposed. Summaries expose actual AR
   coefficients and label the reported scale as an innovation scale.

## Compatibility

Checkpoint format is now 3. Version-2 checkpoints with AR/ARMA p>1 are rejected
with a refit instruction: treating old direct coefficients as partial
correlations would silently change predictions. Other version-2 checkpoints
remain loadable. Raw `state_dict` users must separately track the covariance
parameterization; that low-level API has no model checkpoint metadata.
New checkpoints preserve constructor family states for subsequent fresh fits.
Repeated `fit()` is deliberately a fresh fit; implicit training continuation
is no longer supported.

## Validation

The original suite passed **88/88** before editing. The expanded suite passed
**105/105**, including 17 new cases (`tests/test_training_repairs.py`). Existing
tempered-objective warnings remain expected. The first ten new cases were run
against the old implementation and failed, including the explicit numerical
and lifecycle reproductions; some failures were missing new introspection APIs.

Synthetic checks in `experiments/validate_training_repairs.py` passed all six
gates. Seed 19; Gaussian network [16], embedding dimension 4, LR .01, full batch,
no dropout/weight decay. Small-batch run: 16 rows, 500 steps. Label-shuffle
control: 256 training / 128 independent test rows, 300 steps, synthetic data.

| Check | Evidence |
|---|---|
| Output shape | One scalar prediction for each of 16 rows |
| Initial NLL | 2.1031, versus mean-only Gaussian reference 2.1106 |
| Fit one batch | RMSE 0.0168 |
| Gradients | Present and finite for every trainable parameter |
| Parameter updates | Every trainable tensor changed |
| Shuffled labels | Test R² 0.9993 with signal; -0.0204 after training-label shuffle |

These gates exercise the nonlinear Gaussian training path, not every family or
architecture. The broader suite covers existing mixed-effect/family behavior.

## Fixed-protocol performance checks

Exploratory diagnostics on the existing split (seed 20261015, version 1), with
model seeds 42, 123, 456. No hyperparameter search or test-driven selection.
All runs: hidden [32], embedding 8, full batch, LR .01, UN, beta=1,
kl_auto_scale=False, CPU with one Torch thread. The paired 200-epoch runs use
identical data, row indices, seeds, and settings. At full batch, epochs equal
optimizer steps. These 200-epoch runs differ from session 84's minibatch,
KL-damped protocol and must not be conflated with it.

| Dataset | Before, 200 steps | After, 200 steps | Session 85, 2000 steps | After, 2000 steps | XGB-nogroup |
|---|---:|---:|---:|---:|---:|
| Sleepstudy | 35.2454 | 30.0154 | 29.6305 | 28.1337 | 29.6441 |
| ChickWeight | 36.6311 | 40.7735 | 39.2318 | 39.2613 | 42.7524 |
| Machines | 7.3589 | 4.2238 | 5.5592 | 4.2894 | 5.0720 |
| Oxboys | 11.6899 | 11.1946 | 10.9126 | 11.2743 | 11.5112 |
| Loblolly | 1.7895 | 1.2987 | 1.7097 | 1.3070 | 1.1639 |

Numbers are mean RMSE over three model seeds; lower is better. Session 85's
values come from its preserved `gh_validated_results.json`, not a new baseline
rerun. At 200 steps, four datasets improve and ChickWeight worsens. At 2000
steps, Sleepstudy, Machines, and Loblolly improve; ChickWeight is essentially
unchanged and Oxboys worsens. The revised 2000-step model has 4/5 numerical wins
versus the existing XGB-nogroup baseline, versus 3/5 previously. This is not a
confidence interval or a generalization guarantee. The combined initialization
and decay changes were measured together; their separate causal effects were
not isolated. The ARMA change cannot explain these UN-covariance results.

After-repair per-seed RMSE at 2000 steps (42 / 123 / 456):

| Dataset | Seed 42 | Seed 123 | Seed 456 |
|---|---:|---:|---:|
| Sleepstudy | 28.1045 | 28.0606 | 28.2359 |
| ChickWeight | 39.9674 | 38.9751 | 38.8414 |
| Machines | 4.3937 | 4.1069 | 4.3676 |
| Oxboys | 11.2009 | 11.3876 | 11.2344 |
| Loblolly | 1.2164 | 1.4243 | 1.2801 |

Machines has only one held-out group. Oxboys still has negative R² and the
existing XGB comparison uses an extra `Occasion` feature. Model-seed repeats
do not quantify uncertainty over held-out groups. Prior labels of Machines
and Loblolly as proven inherent model limitations were premature. Loblolly
still loses to XGB here; its exact cause remains unresolved. No observation-CV
rerun, new split evaluation, or broad family/architecture benchmark was done.

## Reproduction and artifacts

Run from the worktree with its project environment:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python experiments/validate_training_repairs.py
.venv/bin/python experiments/diagnose_initialization.py --output experiments/results/initialization_after.json
.venv/bin/python experiments/diagnose_initialization.py --epochs 2000 --output experiments/results/initialization_after_2000.json
```

To reproduce the before run, use package code at `6673fa0` with the same
diagnostic script and a different output path. The baseline was captured before
editing package files. Local results (ignored by Git) include:

- `experiments/results/initialization_before.json`
- `experiments/results/initialization_after.json`
- `experiments/results/initialization_after_2000.json`
- `experiments/results/training_sanity.json`

Benchmark artifacts contain source/data hashes, software versions, protocol,
split row positions, per-seed predictions, loss curves, and measured run times.
The working-tree source hashes distinguish the edited runs from their parent
commit. No cloud compute was used. Remaining research proposals, not accepted
commitments: paired group-split replication, response-scale conditioning,
separate initialization/decay ablations, and stationary-initialized ARMA.
