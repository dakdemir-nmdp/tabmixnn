# Manuscript revision, 2026-10-16 (session 89)

Draft for author review. Starting code revision: 0dee0d7 on
revision/methods-2026-10-15. User explicitly requested all four phases.

## Phase 1: code and experimental audit

- `core/families.py` implements ten concrete families: Gaussian, Binomial,
  Multinomial, Poisson, NegBinomial, MultivariateGaussian, MultiLabelBinomial,
  DiscreteTimeSurvival, DiscreteTimeCompetingRisks, ContinuousSurvival
  (Weibull proportional hazards). The incoming manuscript documented only seven.
- `core/encoder.py:MixedEffectsRepresentation` maps a formula-derived fixed
  design of width p_design through a single linear p_design-by-d projection.
  There are no separate learned categorical embedding tables in this path.
  For term t, latent and transformed effects have shape G_t-by-S_t-by-d;
  gather group rows, multiply by n-by-S_t slopes, sum over slopes and terms.
  Both fixed and random contributions are n-by-d. The fitted formula handles
  its intercept once; `core/model.py` disables the redundant fixed bias.
- All eight selectable covariance types (IID, AR1, ARMA/AR, CS, UN, KRON, K,
  GP) transform standard-normal innovations. `core/encoder.py:forward` computes
  the full standard-normal KL on those innovations. It does not compute a
  diagonal posterior KL directly against a structured effect covariance.
  `core/model.py:fit` uses mean row NLL plus KL/N for every minibatch.
- `covariance/structures.py`: IID uses positive per-slope exp(log_sigma);
  AR1 uses tanh(rho), stationary initial state and innovation sigma (marginal
  variance sigma^2/(1-rho^2), apart from numerical stabilization); ARMA uses
  bounded partial correlations, stable AR recursion, bounded MA coefficients,
  and zero presample states. CS constrains rho to (-1/(G-1),1) and uses a
  jittered Cholesky. UN learns a slope Cholesky with softplus diagonal.
  KRON applies group and slope factors. K and GP use training-level Cholesky
  factors and plug-in conditional means for supplied related unseen levels.
  GP learns per-slope lengthscale/amplitude. KL is the same in all cases.
- `layers/structural.py:StaticStructureModel.B` is strictly lower triangular;
  `acyclicity_penalty` returns zero. The manuscript's NOTEARS optimization
  description did not describe this implementation.
- `CompositeStructuralModel.forward` is u=solve(I-B_s,xi), a=Attention(u),
  h=alpha*u+(1-alpha)*a. The dynamic input is u, not an unknown h.
  `HybridStructuralModel` is a distinct optional iterative implementation;
  the default GSEM block uses CompositeStructuralModel. No convergence
  guarantee follows just from a penalty on attention weights.
- `feature_ablation_values` perturbs a fixed-design column using a background
  sampled from the supplied frame, not an automatically stored training frame.
  Group indices AND random-slope design remain unchanged. Therefore a Days
  ablation slope is not the subject's random-slope coefficient. The routine
  does not compute coalition/Shapley values; nonadditivity is the relevant
  caveat (do not assert that even the dummy-player property necessarily fails).
- The arXiv archive has three PNG files and `shap_examples.tex`, but no plot
  generator. The generator named in its comment is absent from this checkout.
  Incoming PDF pp.25–26 retained SHAP titles, axes and captions; Figure 3 plotted days
  0,4,9 whereas the caption says 0,5,9. New executable plots are required.
- Existing regression scripts cover Sleepstudy, ChickWeight, Machines,
  Oxboys, Loblolly. Existing classification script covers Aids2, Males,
  VerbAgg, Train. No reproducible script/run record was found for the other
  original table rows, Abalone multitask or Abalone tree comparison.
  Original Table 1 lists 8 datasets, Table 3 lists 15, Table 2 claims 19
  regression datasets. These sets do not match. Original Tables 5–7 cannot
  be retained as verified results without a new execution.
- Existing `run_regression.py` LMM uses fixed-only predictions even in
  observation CV; tree encodings are learned globally. `run_gh_validated.py`
  gives Oxboys an extra Occasion predictor only for XGBoost. Classification
  uses global categorical encodings and does not set the neural model seed.
  Its learning_rate alias is deprecated but accepted by the current code.
  New canonical scripts use matched covariates and train-only transforms.
- Existing results: `methods_sleepstudy.json`, `regression_raw.json`,
  `gh_validated_results.json`, and initialization/training sanity artifacts.
  `docs/methods-revision-2026-10-15.md` records session-86 aggregate results.
  These are distinct protocols, not replicas of the original tables.

## Executed common evaluation

Use all nine locally supported datasets. Regime A holds out observations
within each non-singleton group, ensuring every test group is known. Singleton
rows remain training-only. Regime B holds out entire groups. One fixed split
per regime (seed 20261015), three neural initialization seeds (42,123,456).
Report this limitation explicitly; seed dispersion is not sampling uncertainty.
Use identical substantive covariates, train-only preprocessing, and raw-scale
regression metrics. Include ML LMM / Laplace GLMM, fixed-only neural model,
XGBoost and Random Forest. Trees receive one-hot group identity only in A.
Add the scalar linear TabMixNN control for Sleepstudy. Do not tune on test
labels. Remove unsupported legacy performance tables instead of inventing
replacements. Current experiments do not establish multitask or survival
performance, and the revised text must say so.

## Sources checked for the new protocol

- lme4 prediction: https://lme4.github.io/lme4/reference/predict.merMod.html
  (conditional effects for known groups; zero effects for new groups).
- lme4 GLMM: https://lme4.github.io/lme4/reference/glmer.html
  (nAGQ=1 is Laplace approximate ML).
- R dataset manuals confirm ChickWeight grams, Oxboys centimetres and
  Loblolly feet. Aids2 is final-observation vital-status classification with
  unequal observation times, NOT a survival-risk benchmark. Its public data
  are explicitly jittered for confidentiality. Source URLs are retained in
  the executable dataset configurations and result artifacts.

Phase-2 implementation adds a full-batch adapter that caches the existing
TabMixDataset tensors and calls the unchanged package training loop. Synthetic
tests compare its objective gradients with ordinary DataLoader collation.
The new protocol uses fixed source-order full batches, no weight decay,
2,000 updates and no early stopping; these details differ from historical
session-86 runs, which therefore are not pooled with the new results.

Optimizer cross-check: `core/model.py` constructs `torch.optim.Adam`, not AdamW. The current manuscript and reproducibility README use Adam. All new fits used zero weight decay, so correcting this name changes no run settings or results. Historical methods notes called it AdamW and should not be used as the API authority.

## Phase 2: completed experiments

`experiments/results/manuscript_benchmarks.json` contains all 18 dataset/regime
pairs and 168 model records, including the three-seed linear Sleepstudy control
in both regimes. Every metric was independently reconstructed from the locally
cached per-row predictions and original outcome values. All final lme4 fits
passed their convergence checks; Aids2 has a singular random-intercept fit in
both regimes. An initial VerbAgg-A R subprocess failed; the unchanged direct
rerun succeeded, and the suite resumed with the same settings. The cause of
that initial subprocess failure was not established.

`experiments/results/manuscript_ablation.json` contains 21 records for seven
Sleepstudy configurations and three seeds. Nine records reuse the matched
main-suite fits; twelve were newly fitted. The new tables report original
outcome units throughout. Their seven generated TeX inputs reproduce
byte-for-byte from the aggregate records.

TabMixNN improves mean RMSE over the fixed neural control on all five
known-group regression tasks. Comparisons with classical mixed models and
trees depend on the dataset. For unseen groups, results are less favorable
and heterogeneous; all methods have negative test R-squared on Oxboys. The
scalar linear Sleepstudy check yields Regime-A RMSE 32.503 +/- 0.104 ms,
compared with 32.190 ms for ML LMM. This numerical proximity does not prove
covariance-estimation, posterior-uncertainty or REML equivalence.

## Phase 3: regenerated figures

`experiments/create_feature_ablation_plots.py` fits the descriptive model to
all 180 Sleepstudy observations using seed 42 and all rows as background.
An independent explicit perturbation calculation agrees with the API values.
All three plots were regenerated in PDF and PNG, visually inspected, and
recorded in `experiments/results/manuscript_figures.json`. Titles and axes use
ablation terminology, effects are in milliseconds, and Figure 3 uses the
documented days 0, 5 and 9. Anonymous subject labels replace source identifiers.
The historical `shap_` filename prefix remains solely for source compatibility.
The captions explain that fixed-design perturbations leave random slopes
unchanged and are neither causal effects nor additive prediction decompositions.

## Phase 4: manuscript and review

The main methods now document ten families, eight covariance structures,
matching encoder dimensions, the innovation-space KL, strictly triangular
static structure, and the actual sequential gated attention block. Linear
mixed-model equivalence is restricted to the appropriate fixed-covariance
special case. REML, predictive uncertainty, nonlinear variance interpretation,
and causal identification limitations are explicit. Usage examples were checked
against the implemented API, including the Adam optimizer and refitting behavior.

The benchmark section uses the same nine datasets throughout, separates the
two prediction regimes, and replaces unsupported historical performance claims.
Table 7 is in Section 5.3.6 on page 22. The empty acknowledgments heading and
dangling supplementary reference are removed from the rendered document.
All 27 pages were inspected in overview, with detailed review of the revised
methods, results tables and figures. The final pdflatex/bibtex/pdflatex/pdflatex
build produces 27 pages with zero undefined references, missing citations,
overfull boxes, underfull boxes or LaTeX warnings. PDF text checks found no
unresolved reference markers or replacement glyphs. The full final suite
passes: **110 tests, 17 expected warnings** about deliberately nonstandard KL
weighting in existing tests. Five new synthetic tests cover split semantics,
training-only preprocessing, cached-batch gradient equivalence, metric units
and the fixed-only control. All recorded benchmark/ablation/figure script
hashes and all six figure file hashes match the delivered files.

The author-review draft remains limited to one split per regime and three
initialization seeds, with fixed comparator configurations. Seed SDs are not
confidence intervals. Survival, multitask, genomic and spatial performance are
not evaluated. These are scientific limits of the completed experiments,
not unfinished runs or evidence for broader claims.

Reproduction commands are in `experiments/README.md` and
`manuscript/README.md`. Public source records, intermediate R inputs and
per-row predictions remain in ignored local caches. The committed outputs
contain aggregate metrics, row-position split manifests and provenance.
Unused original tables and the downloaded arXiv archive are retained locally
but are not inputs to the revised document or staged revision artifacts.
