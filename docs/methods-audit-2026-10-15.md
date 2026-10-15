# TabMixNN methods audit, 2026-10-15

Draft for author review. Mode A: internal self-review of the author’s paper and implementation.

Baseline: `d603397` on `review/2026-10-14`. Revision branch: `revision/methods-2026-10-15`. Original checkout’s uncommitted regression benchmark was preserved in place; this audit uses a separate worktree. Findings below cite **baseline** line numbers (GitHub links pinned to the commit). Implementation status and validation are recorded in `methods-validation-2026-10-15.md`.

## Verdict

Major revision. The architecture is modular and the formula/family separation is useful, but the implemented objective, several probability models, and evaluation semantics do not support exact LMM/REML recovery or the published benchmark claims. Correcting code requires revising the corresponding claims, not reproducing mathematically unsupported statements.

Read fresh: all tracked Python source, tests, examples and experiment scripts; paper HTML from abstract through references. Plot images and unavailable original experimental artifacts were not audited. Prior notes are background, not evidence. This is a methods audit, not an ethics or exhaustive reference audit.

Severity: P0 invalidates the affected objective/distribution/evaluation; P1 major methods or reproducibility issue; P2 secondary API/clarity issue. Findings are verified by source inspection unless a test or numerical check is explicitly reported in the validation record.

## M01. P0: Batch-size-dependent objective

Location: [core/model.py:1136](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/core/model.py#L1136).

**Problem.** Family NLLs are means, yet full KL is multiplied by B/N. At B=N the penalty is N times the per-row ELBO value. Outcome averaging and batch-specific missingness denominators also change the target.

**Concrete fix.** Use sum of observed per-outcome NLLs divided by batch rows plus beta*KL/N, beta=1 by default; test unequal batches and missing outcomes. Keep legacy reductions explicitly non-ELBO.

## M02. P0: Unseen-group evaluation can fit its own labels

Location: [core/model.py:700](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/core/model.py#L700).

**Problem.** The API adapts on df targets then predicts the same rows; test_transductive calls this held-out generalization. These are fitted conditional reconstructions, not out-of-sample scores. Frozen weights are not protection against label leakage.

**Concrete fix.** Require separate labeled support_df and query df, map effects by group label, optimize jointly with the learned covariance prior, freeze all existing parameters and dropout, restore state in finally. Test query-label invariance and analytic posterior modes.

## M03. P0: K and GP prior/forward mismatch

Location: [core/encoder.py:265](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/core/encoder.py#L265).

**Problem.** K computes KL against sigma²K on mu, then multiplies mu by sigma again in forward. GP uses a standard-normal KL but interpolates scaled independent v without a Cholesky transform; its None-indices path crashes.

**Concrete fix.** Use a single documented noncentered v~N(0,I), u=L(theta)v representation for K/GP/KRON; KL in v coordinates; conditional means at unobserved related levels; preserve fitted rows exactly. Test dense covariance identities and gradients.

## M04. P0: Wrong negative-binomial distribution

Location: [core/families.py:193](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/core/families.py#L193).

**Problem.** PyTorch counts successes before r failures, so this probability yields mean r²/mu, not mu. Predictions and fitted likelihood disagree.

**Concrete fix.** Use logits=log_mu+log_alpha with r=exp(-log_alpha). Check log PMF, mean and variance against independent NB2 formula.

## M05. P0: SPDE samples wrong covariance

Location: [layers/noise.py:269](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/layers/noise.py#L269).

**Problem.** Solving Q epsilon=z yields Q^-2 covariance; fallback samples Q^-1, so numerical failure changes the model. Factory always returns Kronecker even when disabled.

**Concrete fix.** Sample epsilon=L^-T z for Q=LLᵀ. Honor dense/separable factory selection; test empirical covariance and exact transform.

## M06. P1: Phantom level and mismatched mappings

Location: [core/model.py:776](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/core/model.py#L776).

**Problem.** Known levels are 1..G, but G+1 latent rows are created and another unknown row is padded. An unobserved row enters KL and centering. Only K honors full supplied ordering; GP/KRON can attach effects to wrong IDs.

**Concrete fix.** Allocate G genuine rows; reserve unknown only during gather. Honor and validate levels/train_levels for K/GP/KRON. Infer training levels from training data, not all supplied levels.

## M07. P1: Centering does not implement claimed LMM

Location: [core/encoder.py:262](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/core/encoder.py#L262).

**Problem.** Centering already occurs during training, contrary to background. For AR/CS it centers innovations rather than effects; samples are not centered there. K/GP/KRON center over all supplied levels, including unobserved ones. A finite sum-to-zero constraint changes the prior and does not imply REML or orthogonality to a general fixed design.

**Concrete fix.** Default to the ordinary zero-mean Gaussian prior without forced centering. If requested, project full transformed samples using training-level mean and retain KL in original latent coordinates; document induced covariance C G Cᵀ and no REML guarantee.

## M08. P1: Explicit KL settings silently overridden

Location: [core/model.py:422](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/core/model.py#L422).

**Problem.** Automatic 1/mean-group-size damping overrides user lambda, and a second pure-LMM heuristic changes it again (and references logger before creation).

**Concrete fix.** Default beta=1, auto_scale=False; remove hidden LMM rewrite; warn that explicit damping/normalization/legacy reductions define tempered objectives.

## M09. P1: Formula intercept and slope estimand

Location: [parsing/formula.py:126](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/parsing/formula.py#L126).

**Problem.** Both parsing and fitting force no explicit intercept, while the encoder always adds a bias even for ~0+x. Random slopes alone are centered and scaled, changing the intercept-slope covariance model relative to the formula. Categorical fixed effects can be overparameterized.

**Concrete fix.** Honor Patsy intercept coding with no duplicate encoder bias. Keep numeric random slopes in their original units (impute using training means); use UN for correlated intercept/slopes. Test predictions against explicit X beta+Z u.

## M10. P1: Invalid CS correlation and unstable ARMA

Location: [covariance/structures.py:112](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/covariance/structures.py#L112).

**Problem.** CS lower bound is -1/(G-1), not -0.99. Componentwise tanh AR coefficients do not ensure stationarity for p>1; zero initialization differs from stationary AR1.

**Concrete fix.** Map CS rho into valid open interval. Parameterize AR via partial autocorrelations and solve stationary covariance, or reject unsupported stationary claims; document ordering and finite-start process.

## M11. P1: Kron initialization/kernel jitter/low-rank bugs

Location: [covariance/structures.py:183](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/covariance/structures.py#L183).

**Problem.** K_slope Cholesky is subsequently passed through softplus, changing supplied covariance. Kernel self-identity is tested after creating new scaled tensors, so jitter disappears. LowRankCovariance with one slope returns 1 x rank.

**Concrete fix.** Inverse-softplus encode Cholesky diagonals; capture self-kernel identity before scaling; use D+VVᵀ for every slope count; validate matrices and labels.

## M12. P1: Parameter extraction is not BLUP recovery

Location: [utils/summary.py:141](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/utils/summary.py#L141).

**Problem.** All-RE minus fixed-only is grouped separately for every term, mixing crossed effects and collapsing slopes into Intercept. IdentityHead dummy weights are all ones although it slices one channel; linearized gradients are divided by sample count twice; Patsy names disappear.

**Concrete fix.** Extract transformed per-term coefficients for scalar linear identity models; use per-term ablations with explicit contribution labels for nonlinear models. Use actual head map and autograd gradients of summed outputs. Do not call nonlinear contributions BLUPs.

## M13. P1: Fit mutates estimates after optimization

Location: [core/model.py:496](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/core/model.py#L496).

**Problem.** Statistics are computed before two post-fit intercept shifts; one uses self._train_data containing validation rows. Thus validation targets can affect final intercept and reported metrics do not describe final weights.

**Concrete fix.** Remove post-fit shifts; store training partition only; initialize intercept before optimization and report diagnostics afterward. Test validation-label invariance.

## M14. P1: Multivariate predictions flattened

Location: [core/model.py:558](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/core/model.py#L558).

**Problem.** All output axes are flattened across batches, mixing subjects, classes, time and causes. Singleton scalar family squeezing can produce inconsistent shapes.

**Concrete fix.** Concatenate along sample axis only, preserve event dimensions and normalize scalar outputs to length B. Test singleton final batches for multinomial, multivariate and survival families.

## M15. P1: Manifold bypasses mixed-effect controls

Location: [backbones/manifold.py:47](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/backbones/manifold.py#L47).

**Problem.** Every block adds new unregularized group embeddings beyond the variational encoder, ignoring random slopes; use_random_effects=False does not disable these. Model training omits backbone penalties.

**Concrete fix.** Use encoder as sole random-effect source in the main API, or define priors for every extra effect and propagate toggles. Include SPDE/SEM penalties; distinguish independent neuron-grid noise from shared spatial random fields.

## M16. P1: Composite is a different model, not one fixed-point iteration

Location: [layers/structural.py:684](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/layers/structural.py#L684).

**Problem.** alpha*(I-Bs)^-1 xi+(1-alpha)*Attn((I-Bs)^-1 xi) is not xi+Bs xi+Attn(xi). Diagnostic compares against an unchecked possibly divergent iterate and assumes relative error <=1.

**Concrete fix.** Describe composition as a distinct architecture. Expose iterative settings through model/backbone; return diagnostics only if reference residual converges. Report the counterexample and revise Eq.22 claims.

## M17. P1: Convergence and causal claims unsupported

Location: [layers/structural.py:506](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/layers/structural.py#L506).

**Problem.** Row stochasticity bounds infinity norm, not spectral norm. Nonlinear attention Jacobian contains derivatives of attention weights; value/output norms omit these. Lower triangular DAG penalty is identically zero and latent rotations/reparameterizations prevent causal identification.

**Concrete fix.** Remove guarantee language, measure actual fixed-point residuals/Jacobians, and reserve causal claims for identified models with external assumptions. Treat B as a learned latent transformation.

## M18. P1: Lavaan fixes arbitrary coefficients

Location: [core/model.py:883](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/core/model.py#L883).

**Problem.** Parser puts 0.1 on allowed edges and output_sem freezes them. This is neither estimation under a structural mask nor user-specified coefficients. Unknown predictors silently disappear. Output SEM mixes only first logits and is not a joint outcome covariance model.

**Concrete fix.** Learn coefficients constrained by parsed edge mask, or require numeric fixed coefficients. Reject unknown outcomes and document precisely which parameters are mixed.

## M19. P1: Training configuration and reproducibility

Location: [core/model.py:197](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/core/model.py#L197).

**Problem.** Twenty epochs can be twenty updates; learning_rate used throughout examples/benchmarks is silently ignored in **kwargs. No general seed, no optimizer-step budget diagnostics, and fit overrides are ignored. Pure LMM forcibly resets global torch RNG.

**Concrete fix.** Use documented small-data preset and validation/convergence checks, honor or reject aliases, add explicit random_state, remove hidden seed reset, record optimizer steps; benchmark defaults rather than attributing all gaps to epoch count.

## M20. P1: Persistence changes the statistical model

Location: [core/model.py:1425](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/core/model.py#L1425).

**Problem.** Configuration omits covariance, centering, KL normalization, dropout, SEM and others; strict=False load can silently discard state. Patsy reconstruction from dummy zeros loses stateful-transform training statistics.

**Concrete fix.** Serialize all supported constructor choices, strict-load versioned checkpoints, test K/GP/SEM round trips; store true Patsy stateful transforms or explicitly reject unsupported persistence.

## M21. P1: Log likelihood/AIC/BIC and variance labels misleading

Location: [core/model.py:1271](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/core/model.py#L1271).

**Problem.** Conditional likelihood at variational means is not marginal likelihood; counting variational parameters is not justified AIC/BIC. variance_decomposition drops fixed-random covariance, uses predicted variance denominator and labels empirical ablations ICC.

**Concrete fix.** Label conditional plug-in log likelihood, suppress unsupported information criteria, report descriptive variances/covariances without ICC claim; offer marginal likelihood only with a justified integration method.

## M22. P1: Attributions and uncertainty claims exceed implementation

Location: [core/model.py:1781](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/core/model.py#L1781).

**Problem.** Single-feature mean ablation is not Shapley allocation, random slopes are not changed with their covariate, and edge API returns fabricated dummy values. MC latent-function quantiles omit observation noise and weight uncertainty; prediction API has no calibrated intervals.

**Concrete fix.** Rename ablations, raise NotImplementedError for dummy edges, implement a validated coalition method if SHAP is claimed; distinguish mean-function uncertainty from predictive intervals and evaluate coverage.

## M23. P1: Survival target and prediction semantics

Location: [parsing/formula.py:395](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/parsing/formula.py#L395).

**Problem.** Right-closed documentation conflicts with right insertion at bin boundaries; missing time becomes a valid last-bin index. Invalid follow-up/cause labels are clipped. ContinuousSurvival returns log-HR even for survival and requires times for median unnecessarily.

**Concrete fix.** Specify bin endpoints/censoring convention, preserve missingness, reject invalid labels/ranges, implement Weibull survival and median with explicit times passed by predict. Add boundary and censoring likelihood checks.

## M24. P1: Benchmarks do not reproduce paper tables

Location: [experiments/run_regression.py:2](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/experiments/run_regression.py#L2).

**Problem.** Directory exists, but only 5 regression and 4 classification datasets versus paper 19/15; missing original folds/raw predictions and standardization. Group-holdout TabMixNN is compared to observation-CV XGBoost. run_all has divergent Train label coding.

**Concrete fix.** Relabel these as new diagnostic scripts; shared split manifests, train-only preprocessing, matched baseline information, seeds and raw outputs. Add LMM benchmark and distinguish seen groups, unseen groups, and support-conditioned queries. Do not claim original tables reproduced.

## M25. P1: Paper numerical and unit inconsistencies

Location: [experiments/README.md:3](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/experiments/README.md#L3).

**Problem.** Paper §5.3.2 text reports Oxboys/Gasoline/Machines RMSE reductions of 93%/68%/61%, whereas Table 3 lists 41.1%/42.9%/75.9%. Table 7 says standardized targets but reports, for example, RMSE 34.95 for full random effects. Table 2 has 19 regression datasets but Table 3 lists 15 without accounting for omitted cases. See [paper §5.3 and Tables 3/7](https://arxiv.org/html/2512.23787v1).

**Concrete fix.** Regenerate every table and narrative from versioned outputs; specify units, denominator and fold aggregation; include failures and all datasets. Current evidence does not establish reported win rates.

## M26. P1: Examples/test evidence mislabel endpoints and validation

Location: [examples/06_competing_risks.py:64](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/examples/06_competing_risks.py#L64).

**Problem.** Competing-risk example is a single-endpoint survival model, not CIF estimation with competing events. Examples preprocess before split, use test as validation, and score hard binary predictions as AUC; survival test labels event AUC C-index. Existing tests often check training fit, wrong row indices or docstrings, not method identities.

**Concrete fix.** Label single-endpoint example honestly or implement true event-type data; split before transforms and tuning, use probabilities for AUC and proper concordance, add analytic and held-out regression checks.

## M27. P2: Additional public/legacy API defects

Location: [backbones/gsem.py:221](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/backbones/gsem.py#L221).

**Problem.** Legacy GSEM survival passes unsupported constructor kwargs and compute_loss refers to nonexistent hidden_layers/head methods. Tensor covariance regularizer matches covariance of averaged modes, not a tensor-normal likelihood. Attention factory/head divisibility and causal mask in feature mode are inconsistent.

**Concrete fix.** Mark unsupported legacy paths experimental or fix with focused tests; document tensor regularizer as moment matching and its scale nonidentifiability; validate attention dimensions and use one attention accessor.

## M28. P2: Packaging depends on checkout folder name

Location: [pyproject.toml:30](https://github.com/dakdemir-nmdp/tabmixnn/blob/d603397/pyproject.toml#L30).

**Problem.** Setuptools searches parent directory for tabmixnn*, so alternate checkout names can install the wrong source or extra sibling trees. This blocks trustworthy isolated tests.

**Concrete fix.** Explicitly map package tabmixnn to this repository root and list package modules; verify import path and wheel contents.

## Paper alignment and evidence

Paper reviewed: https://arxiv.org/html/2512.23787v1 (retrieved 2026-10-15). Particularly §§3.1.3–3.1.4, 3.2.1–3.2.2, 3.4–3.6 and 5. These findings are primarily original code analysis and algebraic counterexamples. The paper source is authored by the requester.

Methods revisions are supplied separately in `methods-revision-2026-10-15.md`. Evaluation should follow an ADEMP-style simulation specification and explicit prediction estimands; a full clinical TRIPOD+AI checklist is outside this software-methods audit.

Unresolved evidentiary gap: original per-fold outputs/configuration for the published tables were not present in the inspected repository. New runs cannot establish their provenance.
