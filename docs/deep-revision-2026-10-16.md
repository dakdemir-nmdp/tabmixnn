# TabMixNN deep revision, session 90

Author-review draft. Starting commit: `894352d`, branch
`revision/methods-2026-10-15`. Deniz requested a substantially deeper revision
rather than accepting the earlier four-phase pass as sufficient.

## Scope and final assessment

This is a pre-submission self-review of the implementation, statistical claims,
reproducible experiments and manuscript. The central defensible claim is that
TabMixNN implements variational group effects upstream of a neural predictor
and exposes multiple covariance/likelihood choices through a formula interface.
Prediction for known groups and entirely unseen groups are different tasks.

The prior pass established reproducible benchmark tables and corrected many
method descriptions. It did not establish that every advertised family worked
through the public API, that all prediction utilities used meaningful estimands,
or that the introduction fairly represented directly relevant prior work.
A passing aggregate test count was insufficient evidence for those claims.

The major implementation and reporting defects below have been repaired, and
the manuscript now presents a narrower, supported methods/software contribution.
The revision was completed on 2026-10-17. Empirical superiority over specialized
competing methods, predictive interval calibration and general parameter
identifiability remain outside the evidence. The draft is ready for author
review; these checks do not constitute independent peer review or submission.

## Findings and repairs

| ID | Severity | Location at starting revision | Finding and action |
|---|---|---|---|
| D01 | Major | `core/families.py:ContinuousSurvival.predict`; `core/model.py:predict` | Survival requests returned the raw score; median requests without times also returned the score; public prediction could not pass times. Implemented Weibull survival, cumulative hazard, median, relative risk and explicit default log-score predictions, with validation and batched time-axis preservation. |
| D02 | Major | `parsing/formula.py:OutcomeSpec.get_target_cols` | A list of response columns became a nested list, preventing multivariate Gaussian and multilabel outcomes from fitting through the documented interface. Added ordered vector response columns and family-specific width validation. Exported Multinomial at package level. |
| D03 | Major | `CompiledDesign._time_to_index`; discrete survival likelihoods | Censoring inside an interval was rounded up, asserting survival beyond observed follow-up. Events now use their containing right-closed interval; censoring uses completed intervals only. Censoring before the first endpoint contributes zero information. Uncovered follow-up is rejected rather than silently clipped. This discretization discards partial-interval censoring information; it does not interpolate a continuous hazard. |
| D04 | Major | `Family.validate_targets`; survival and multinomial likelihoods | The validation hook was empty. Invalid labels were clipped or converted to integers; missing target columns could lead to a fit without the intended likelihood. Added training/validation schema and domain checks, allowing NaN missing outcomes but requiring observed targets for each outcome. Direct discrete-survival likelihood calls validate integer indices and labels before conversion. |
| D05 | Major | `DiscreteTimeSurvival.nll` | Probability clipping saturated tail gradients and capped losses. Replaced it with log-sigmoid calculations. An event with logit -100 now has NLL 100 and derivative -1, rather than a clipped loss with no useful gradient. |
| D06 | Major | `TabMixNN._predict_from_tensors`; ablation methods | The generic vector reduction summed `1-p` over class probabilities, making multinomial importance constant. It also treated arbitrary vectors as survival curves and ignored the selected outcome in random-effect ablations. Scalar estimands are now explicit, multitask outcomes must be selected, and undefined vector reductions raise an error. No unsupported vector attribution is silently fabricated. |
| D07 | Minor | `feature_ablation_values` | Compiled Patsy names were replaced by generic labels, NumPy's global RNG was reset, model modes were changed, and the query batch defined the reference. Added authoritative names, local RNG, mode restoration and an explicit separate `background_df`. The old default reference remains the supplied query frame for compatibility. |
| D08 | Major | `CompiledDesign._prepare_fixed_frame`; `_transform_fixed`; `TabMixNN.predict` | Prediction could process malformed query labels and mutate preprocessing metadata for extra columns. All-missing or invalid predictors could let Patsy drop rows. Ordinary prediction now excludes targets, transform metadata is not fitted on query data, and nonfinite or missing design rows fail clearly. |
| D09 | Major | `TabMixNN._train_loop` | Nonfinite objectives or gradients could enter optimizer updates without an explicit failure. Added finite-value guards with epoch/batch context. These checks diagnose numerical failure, not convergence. |
| D10 | Major | Methods 3.1.3 and 3.6.2 | A warning that latent variances were not response-scale quantities did not show the underlying identifiability problem. Added an explicit rescaling argument and executable counterexample: multiplying encoder scale by 3 and dividing its downstream projection by 3 preserves predictions and the ELBO while multiplying latent covariance by 9. |
| D11 | Major | Methods 3.1.3 and 3.4.1 | The fixed-covariance mean-equivalence claim lacked a derivation explaining why it does not imply variance-parameter equivalence. Added expected Gaussian objective, normal equations, optimal diagonal posterior variances, the covariance-dependent ELBO gap, and a two-dimensional uncertainty counterexample. |
| D12 | Major | Introduction and Background 2.2 | Directly relevant LMMNN and GPBoost work was omitted, making the gap claim too broad. Added verified primary references and distinguished implementation contributions from comparative superiority. These specialized methods are not in the executed comparator set, and that limitation is explicit. |
| D13 | Major | Background 2.3; `references.bib` | The cited deep structural-causal title belonged to Pawlowski, Coelho de Castro and Glocker (2020), not the supplied Biza et al. (2023) entry. Corrected the reference and removed the incorrect DeepSEM label. The lme4 journal article year is 2015, not its 2014 preprint year. |
| D14 | Major | Benchmarks 5.2–5.3 | Three initialization seeds on one split did not evaluate split sensitivity. The extension uses five specified split seeds in both regimes, with all three neural seeds and unchanged comparator budgets. Split SDs are descriptive because holdouts overlap and small-group partitions can repeat. |
| D15 | Minor | Benchmarks classification results | Brier score alone does not diagnose the direction or shape of miscalibration. Added held-out mean probability error, jointly fitted calibration intercept/slope, and fixed-bin summaries. These are descriptive evaluations; no model is recalibrated or selected on test data. |
| D16 | Major | `ContinuousSurvival` and Methods 3.3.2 | A free neural score intercept and free Weibull baseline scale are confounded. Added the exact invariance and restricted interpretation to conditional predictions unless a reference constraint is imposed. No uniqueness claim for baseline parameters is retained. |

The detailed diagnostic review also corrected a reporting omission: the
primary-split Train reference fits are singular in both regimes, as are Aids2.
The earlier prose mentioned only Aids2. The fitted results were retained;
the manuscript now reports both datasets and explains boundary estimates.

## Mathematical verification

The appendix assumes a full-column-rank fixed design, positive residual variance,
fixed covariance factor, scalar Gaussian response, identity decoder, no
centering and no extra penalties. In those conditions, differentiating the
expected Gaussian ELBO gives the mixed-model normal equations. With posterior
precision P, the diagonal optimum has D[j,j]=1/P[j,j], whereas the exact
posterior covariance is inverse(P). The minimum gap is
`0.5 * log(product(diag(P)) / det(P))`; its dependence on covariance parameters
precludes a general ML-equivalence inference. REML is a different objective.

`experiments/verify_variational_identities.py` uses SymPy to check derivatives
and the exact 2-by-2 counterexample, with output in
`experiments/results/variational_identities.json`. With P=[[2,1],[1,2]],
exact marginal variances are 2/3 and diagonal variational variances are 1/2.
Yet the variance of the sum is 2/3 exactly versus 1 variationally, and that of
the difference is 2 exactly versus 1 variationally. The general derivation is
written explicitly; symbolic verification covers the stated finite-dimensional
identities and example, not a Lean/formal proof of every model claim.

The package test suite independently checks the mixed-model mean gradient at
an externally solved optimum and tests the nonlinear rescaling in the actual
implementation. Survival tests use hand-calculated likelihoods, Weibull curves
and medians, probability conservation, boundary censoring, extreme logits and
finite-difference gradients. These are meaningful implementation checks, not
real-data survival validation.

## Literature checked against primary sources

- LMMNN: https://jmlr.org/papers/v24/22-0501.html
- GPBoost: https://jmlr.org/papers/v23/20-322.html
- Correct structural-causal reference: https://proceedings.neurips.cc/paper_files/paper/2020/hash/0987b8b338d6c90bbedd8631bc499221-Abstract.html
- lme4 publication metadata: https://www.jstatsoft.org/article/view/v067i01
- Mean-field covariance limitations: https://jmlr.org/papers/v19/17-670.html
- Weibull formulas: https://itl.nist.gov/div898/handbook/apr/section1/apr162.htm

The introduction now makes an implementation contribution rather than claiming
to originate mixed-effects machine learning. This was a focused verification
of load-bearing sources, not an exhaustive systematic literature review.

## Coverage and unresolved limits

The ten-family contracts now include fit, predict, singleton batches and
save/load on synthetic data. Core covariance and structural tests from earlier
revisions remain relevant, but tests do not establish global optimization,
identifiability of every parameter, posterior calibration or clinical validity.
The empirical evidence remains Gaussian/binomial prediction on public grouped
data, with fixed hyperparameters and no external validation. This is a methods
and software evaluation, not a TRIPOD+AI-complete clinical prediction study.

Survival families do not implement delayed entry, interval censoring or an
informative-censoring model. Discrete-time endpoint conventions must match the
scientific observation process. Neural latent scales and Weibull intercept/scale
components require identification constraints before parameter interpretation.
Vector ablations are intentionally unavailable without a defined scalar target.
Stateful Patsy transform persistence remains explicitly unsupported. Learned
representations and structured edges are not causal effects. LMMNN and GPBoost
comparisons and calibrated predictive intervals remain substantive evidence gaps.

## Execution record

- Full regression suite: **137 passed**, with 17 expected warnings about
  intentionally nonstandard KL weighting or likelihood reduction. The full
  run followed all functional model changes. A subsequent six-test benchmark
  integrity run also passed after the final summary-script adjustment. Use
  `.venv/bin/python -m pytest`; this local environment's standalone `pytest`
  launcher retains a stale path from an earlier directory name.
- Five splits × nine datasets × two regimes: **90 combinations and 840 model
  records**, with three neural initialization seeds and the unchanged 2,000-update
  budget. The primary 168 records reproduce every prior metric exactly.
- Every saved benchmark metric was reconstructed from the downloaded outcome
  values, recorded row positions and locally cached predictions at
  `rtol=atol=1e-10`. No aggregate was accepted solely because it existed.
- All 90 final classical reference fits report convergence; 19 are singular
  and retained. Primary-split singularities affect Aids2 and Train in both
  regimes. Singular variance estimates are distinct from optimizer failures.
- The component analysis contains **21 records** (seven configurations × three
  seeds): nine are inherited from the rerun primary comparison and twelve are
  new fits. All three full-data ablation figures were regenerated; the PNG
  values are unchanged and all six PDF/PNG checksums match their manifest.
- All **ten generated tables** reproduce byte-for-byte from the saved results.
  The ablation source-record hash, generator hash, figure hashes and all
  five benchmark source manifests were verified.
- All **72 primary-split calibration fits** (all classification comparators)
  converged without warnings. Table 10 summarizes the 24 TabMixNN fits as
  eight dataset/regime rows, averaging coefficients over the three seeds.
- `pdflatex`, `bibtex` and repeated `pdflatex` passes produce a **33-page PDF**
  with zero undefined references, missing citations or overfull boxes. One
  benign underfull paragraph remains in the survival discussion. An overview
  of all pages and detailed inspection of usage, new tables, figures and the
  mathematical appendix found no clipping or overlapping content. Benchmark
  headings were kept with explanatory text. Table 7 is on page 24, within
  the ablation subsection and before the references on pages 32–33.

### What the additional experiments establish

Across the five split summaries, known-group TabMixNN regression has lower
mean RMSE than the fixed neural control on all five datasets. It is lowest
among the included comparators on ChickWeight, Oxboys and Loblolly; the mixed
model is lowest on Sleepstudy and XGBoost on Machines. For unseen groups,
TabMixNN is lowest on ChickWeight, Machines and Oxboys, while the mixed model
is lowest on Sleepstudy and Random Forest on Loblolly. These are descriptive
rankings with no claim of significance or consistent superiority. Machines
has only two distinct held-out partitions across five seeds; Aids2 has three.

Known-group classification mean AUROC is highest for TabMixNN on VerbAgg
(0.840) and Train (0.817), while the mixed model is highest on Males. For
unseen groups, TabMixNN leads only on Aids2 (0.561); XGBoost leads on Males
and the mixed model on VerbAgg and Train. Discrimination is therefore
setting- and dataset-dependent.

The calibration finding is substantive: primary-split TabMixNN slopes range
from 0.132 to 0.731, all below one. Males regime B has observed event fraction
0.280 versus mean predicted probability 0.128 (15.2 percentage points lower).
Aids2 regime A has mean error only -0.002 but slope 0.228. Matching the overall
event fraction is not sufficient for calibrated predictions. The abstract,
results and conclusion now state this limitation. These are descriptive
primary-split diagnostics, without cluster-independent confidence intervals,
recalibration or predictive-coverage validation.

### Execution exception and source provenance

For VerbAgg-A split 20261018, lme4 failed with `Downdated VtV is not positive
definite` in the combined runner. The identical independent R command converged
on byte-identical prepared CSVs. `experiments/resume_verified_reference.py`
verifies these inputs, requires convergence and replays that result without
changing the optimizer or selecting by test performance. The record contains
input, R-script, wrapper and reference-file hashes. The remaining pairs then
completed through the canonical runner. The cause of this execution-context
numerical behavior has not been isolated; this exception is documented in
`experiments/README.md` rather than silently treated as an ordinary first-pass fit.

Fourteen whitespace-only blank lines in `core/model.py` were cleaned after
execution. `experiments/results/source_equivalence.json` records both source
hashes and exact reversible line replacements. Reconstructing the executed
file reproduces its recorded hash, and the executed/delivered Python syntax
trees are identical. Every other execution source hash matches the delivered
file directly. Historical execution hashes were not rewritten.

### Compatibility changes and remaining work

Discrete-survival default prediction now returns survival; callers needing
conditional hazards must use `mode='hazard'`. Existing fits with within-interval
censoring should be refitted under the corrected endpoint convention. Vector
ablations now raise an explicit error instead of returning an arbitrary or
constant reduction. Multitask ablations require an outcome selection. Invalid
targets and nonfinite training states now fail clearly. The README documents
these behavior changes and the separate training-background option.

All requested local revision steps are complete. Author review remains before
external use. LMMNN/GPBoost comparisons, broad hyperparameter searches,
real-data survival validation, external validation and calibrated predictive
intervals were not executed and are explicitly identified as evidence gaps.
No manuscript submission or remote push was performed.
