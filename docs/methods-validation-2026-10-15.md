# Methods revision: implementation and validation

Author-review draft, 2026-10-15. Branch `revision/methods-2026-10-15`, based on
`d603397`. Work was performed in an isolated worktree; the original checkout's
uncommitted regression-script edits were preserved. Nothing was pushed.

## Delivered scope

The [fresh audit](methods-audit-2026-10-15.md) contains 28 prioritized findings,
each with baseline code locations and a proposed correction. The companion
[methods replacement](methods-revision-2026-10-15.md) specifies the revised
objective, covariance parameterization, estimands, evaluation protocols, and
claims that need revision. The original paper's LaTeX source was not available;
the replacement text has not been inserted into an arXiv submission.

All five P0 findings were addressed. This revision does not close every P1/P2
finding or establish that all published methods and benchmarks are valid.

| Audit issue | Implemented status | Remaining boundary |
|---|---|---|
| M01 objective | Observed likelihood sum per batch row + KL/N; proper missing-outcome denominators; all-missing batches; joint multilabel event NLL | Nonunit outcome weights, legacy reductions, KL damping and explicit regularizers remain alternative objectives |
| M02 adaptation | Explicit support/query API, joint new-term MAP, learned IID/UN prior, frozen parameters, restored modes, query labels ignored | Caller must ensure disjoint observations; no adaptation uncertainty; other covariance types rejected |
| M03 K/GP priors | Standard-normal innovation KL and correct covariance factor in forward; training effects preserved during extrapolation | Related unobserved levels use conditional means without their residual uncertainty |
| M04 count likelihoods | Correct NB2 logits; exact Poisson likelihood constant; singleton binomial shape | No full count-data calibration study |
| M05 SPDE | Dense inverse-Cholesky sampling; factory honors dense choice; Kronecker powers honor alpha | Separable approximation differs from dense SPDE; full spatial benchmark deferred |
| M06 level mapping | No phantom latent row; validated full/training level maps for K/GP/KRON | New unrelated groups default to zero latent effects |
| M07 centering | Optional projection of transformed samples and means using training levels; default false | Not REML and not universal neural identifiability |
| M08 KL heuristics | Default beta=1, auto-scale false; no hidden LMM override; warnings for tempered settings | Explicit auto-scale still deliberately overrides beta |
| M09 formula | Explicit requested fixed intercept, no duplicate encoder bias; raw numeric slopes | Complex random-slope formula syntax still limited |
| M10 covariance validity | CS correlation constrained to valid interval | General stationary ARMA parameterization deferred |
| M11 covariance details | Correct supplied slope Cholesky initialization, self-kernel jitter, one-slope low-rank dimensions | Broader covariance calibration deferred |
| M12 extraction | Scalar linear per-term/per-slope coefficients; nonlinear single-term ablations; correct average derivative scaling and feature names | No standard errors; nonlinear values are not BLUPs; broad multitask/SEM coefficient inference deferred |
| M13 post-fit mutation | Removed both intercept adjustments; training-only stored data; scalar linear OLS initialization before optimization | Validation may intentionally select a checkpoint when early stopping is enabled |
| M14 prediction axes | Preserve sample and event axes, including singleton batches | Family-specific continuous survival prediction semantics remain M23 |
| M15 manifold effects | Main API uses encoder as sole RE source; includes backbone penalties | Standalone blocks remain experimental; manifold theory/performance not comprehensively validated |
| M16 composite | Correct distinct-architecture description; diagnostic checks reference residual, preserves modes, rejects nonconvergence | No equivalence to Eq. 22; no exposed/configured replacement solver in this revision |
| M17 theory | Unsupported guarantees removed from replacement prose | Attention contraction/causal identification and legacy guarantee language require further revision |
| M18 output SEM | Identified and documented | Arbitrary frozen Lavaan edges/joint-model interpretation not repaired |
| M19 training | 200-epoch generic budget; explicit seed; learning-rate alias; validated fit overrides; scalar LMM factory; reproducible 2,000-update protocol | No universal convergence guarantee or completed multi-dataset tuning study; some constructor compatibility kwargs still permissive |
| M20 persistence | Versioned strict loading; full config; canonical family state; UN/K/GP/KRON and intercept-only round trips; reject stateful Patsy export | Old checkpoints must be refitted; general Patsy transforms are not serialized |
| M21 fit statistics | Conditional likelihood label; AIC/BIC unavailable | Legacy variance_decomposition still must not be interpreted as validated ICC |
| M22 interpretation | Edge-Shapley placeholder raises instead of returning fabricated values | Legacy feature/term “Shapley” labels and predictive intervals remain unsupported |
| M23 survival | Right-closed bin insertion and missing-time preservation; output axes fixed | Invalid-label clipping and continuous Weibull prediction API remain unresolved |
| M24 reproduction | Existing scripts relabeled exploratory; new matched Sleepstudy protocol and aggregate artifacts | Original Tables 2–4 not reproduced; old run_all split/coding problems remain |
| M25 paper numbers | Explicit numerical discrepancies documented | Original tables and narrative require regeneration from original runs |
| M26 examples/tests | Targeted analytic tests; support/query test repaired; centering/index tests corrected | Other examples' endpoint/leakage/metric problems remain |
| M27 legacy APIs | Documented as unresolved | GSEM legacy API, tensor regularizer and attention API issues deferred |
| M28 packaging | Explicit package mapping; isolated editable import and wheel contents checked | Tests should run from a checkout directory with a valid Python identifier |

## Automated checks

The unmodified baseline independently passed **36 tests**. That result did not
establish correctness of the ELBO, covariance models, or evaluation protocol.
The revised suite passes **70 tests**, including **34 new parameterized test
cases**, with 17 expected warnings from existing tests that explicitly request
tempered objectives. The final run reported 14.05 seconds; this is pytest's
reported test duration, not a measure of development time.

```bash
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest -q
```

New checks in [test_methods_revision.py](../tests/test_methods_revision.py):

- NB2 log PMF against an independent closed form, with mean and variance
  identities; Poisson normalizing constant and singleton binary likelihood.
- Missing-outcome likelihood/gradient invariance under unequal partitions;
  actual training-loop KL gradient of m/N in batches of sizes 5, 5, and 2;
  all-missing minibatch handling.
- Standard-normal latent KL for IID/K/GP/KRON; dense covariance factor and
  extrapolation identities with reordered training levels; finite gradients.
- Valid CS bounds, empirical dense-SPDE covariance from 50,000 draws, factory
  selection, low-rank one-slope dimensions.
- Formula intercept/slopes, training-level centering, level order, prediction
  batch invariance, survival-axis preservation and monotonicity.
- Gaussian support MAP against its analytic value; query-label invariance;
  parameter and mode restoration. Validation-label changes leave fitted
  weights exactly unchanged without early stopping or model selection.
- Fixed-covariance linear Gaussian mean gradients vanish at the independently
  solved Henderson equations. This checks the mean objective, not equivalence
  of jointly fitted covariance estimates or REML.
- Linearized effects invariant to repeated observations without destroying
  existing parameter gradients; validated training overrides.
- Strict nondefault covariance and intercept-only serialization; diagnostic
  nonconvergence and mode restoration; structural counterexample below.

The existing transductive synthetic test now evaluates separate support/query
rows. Existing parameter-recovery and training-fit tests are retained, but their
wide tolerances and in-sample metrics must not be treated as classical-estimator
equivalence or generalization evidence.

`git diff --check` passes. A wheel built successfully and contained only the
`tabmixnn` package and its distribution metadata (35 archive entries), rather
than sibling checkouts. Editable imports resolved to this revision worktree.
Validation used Python 3.11.13, PyTorch 2.5.1 on CPU, NumPy 2.4.6, pandas 3.0.6,
SciPy 1.17.1, scikit-learn 1.9.1, and statsmodels 0.15.0. GPU, other operating
systems, and other dependency versions were not tested.

## Sleepstudy comparison

Run [validate_sleepstudy.py](../experiments/validate_sleepstudy.py) with:

```bash
.venv/bin/python experiments/validate_sleepstudy.py
```

The public data are described by the
[lme4 Sleepstudy documentation](https://vincentarelbundock.github.io/Rdatasets/doc/lme4/sleepstudy.html).
The input SHA-256, source URL, dependency versions, settings, per-seed aggregate
metrics, fixed effects, and covariance estimates are saved in
[methods_sleepstudy.json](../experiments/results/methods_sleepstudy.json).
Downloaded row records are ignored by git; no subject-level records or
predictions are included in the committed results.

Both methods use a fixed intercept and Days slope plus correlated group
intercept/slope effects. Preprocessing is fitted on training rows only.
TabMixNN uses a scalar linear decoder, beta=1, no hard centering, zero decay,
full batches, learning rate 0.01, and 2,000 updates. Three seeds are 11, 22, 33.
The reference is statsmodels MixedLM **marginal ML**, not REML. Its convergence
flag was checked. Support adaptation uses 1,000 updates at 0.02 for TabMixNN and
the analytic Gaussian conditional mean for the reference.

| Evaluation target | Train / support / query rows | TabMixNN RMSE, ms (mean ± seed SD) | MixedLM ML RMSE, ms |
|---|---:|---:|---:|
| Full-data fitted values, descriptive only | 180 / 0 / 180 | 23.581 ± 0.015 | 23.504 |
| Held-out observations, known groups | 144 / 0 / 36 | 32.291 ± 0.053 | 32.190 |
| New groups, zero-shot, final five days | 120 / 0 / 30 | 32.762 ± 0.348 | 32.355 |
| Same new-group queries, first five days as support | 120 / 30 / 30 | 34.648 ± 0.532 | 35.053 |

Known-group holdout selects two rows per group with split seed 20261015. The
new-group split holds out the final six groups in the source's stable level
order. Both new-group protocols score the same final-five-day queries. These
are fixed diagnostic splits, not repeated cross-validation or a population-level
comparison of algorithms. Seed SD describes optimization variability only.

The full-fit mean fixed intercept/slope were **251.275 ms and 10.511 ms/day**,
versus **251.405 and 10.467** for marginal ML. Mean residual SD was **25.747 ms**
versus **25.592 ms**. Individual covariance estimates are in the JSON; close
point predictions do not establish covariance equality or interval calibration.

On this split, support adaptation worsened late-day RMSE relative to zero-shot
for both methods. It still produced comparable conditional predictions across
methods. This is retained as evidence against claiming adaptation always helps.
The results support the revised linear mixed-effects path on this benchmark;
they do not establish superiority over XGBoost or reproduce the paper's tables.

## Explicit counterexamples and theory limits

The composite block can have a **50% relative difference** from the coupled
reference even with zero structural operators: with B_static=0, attention=0,
and fusion alpha=1/2, the composite is xi/2 while the coupled fixed point is xi.
The new test verifies this exact case. Thus “one fixed-point iteration” is false
and negligible approximation error cannot be assumed. The revised diagnostic
will not report an error against a nonconverged reference.

Likewise, a row-stochastic matrix with both rows equal to (1,0) has spectral
norm sqrt(2), not at most one. State-dependent attention adds derivative terms
to its Jacobian. Existing attention bounds do not prove global contraction.

Mean-field variational training is not REML. Full posterior predictive
uncertainty, causal latent-edge identification, general stationary ARMA,
continuous-survival API repair, and full benchmark reproduction remain outside
the validated implementation. The replacement methods text states these limits.

## Migration and author review

These are intentional statistical/API changes: refit old models. Default KL,
centering and unknown-group semantics changed; fixed intercept coding and
random-slope units changed; checkpoints now require format version 2 and strict
state matching. `predict_transductive` requires `support_df`. Saving stateful
Patsy transforms is rejected until their fitted state can be serialized safely.
No old checkpoint migration or external publication is performed.

Before submitting a revised paper, review the replacement equations/claims and
resolve the table provenance gaps. Remaining proposed fixes in the audit are
not accepted commitments or claims of completed work.
