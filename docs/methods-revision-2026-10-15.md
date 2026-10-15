# Proposed methods replacement — 2026-10-15

Draft for author review. This text describes the revised implementation. It is
not a claim that the current arXiv manuscript or its published tables have been
updated. No manuscript source was present in this checkout.

## Model and variational objective

For random-effect term t, let v_t have a standard multivariate normal prior and
let u_t = L_t(theta_t) v_t, where L_t is the specified covariance factor. The
encoder combines the fixed-effect design with these transformed random effects.
The downstream network maps the combined representation to the parameters of
the selected outcome distributions. Covariance parameters and network weights
are point estimates; this implementation is not fully Bayesian in those
parameters.

We approximate the posterior over training-level innovations with independent
normal coordinates,

    q(v) = product_j Normal(v_j; m_j, exp(2 a_j)).
    KL(q || p) = 1/2 sum_j [exp(2 a_j) + m_j^2 - 1 - 2 a_j].

For N training rows, the negative evidence lower bound divided by N is

    J = (1/N) sum_i E_q[sum_{k observed at i} -log p_k(y_ik | v, x_i)]
        + KL(q || p)/N.

For a uniformly sampled minibatch S of B rows, the stochastic estimator is

    J_S = (1/B) sum_{i in S} sum_{k observed at i} -log p_k(y_ik | v, x_i)
          + KL(q || p)/N,       v ~ q.

The likelihood contribution is a sum across outcomes, not a mean across
outcomes. Missing outcomes contribute zero without changing the row
denominator. Vector-valued targets currently require all components of the
outcome to be observed. This assumes the missing-data process can be ignored
for the fitted likelihood; it does not model informative missingness.

The full global KL is evaluated on each update. Multiplying it by B/N while
using a mean NLL would instead multiply its relative weight by B. Unequal final
batches do not change this formula. The stochastic optimizer path can still
depend on batch size even when the expected objective is the same. This
normalization follows the ELBO and subsampling principle, rather than an
empirical group-size correction; see [Hoffman et al., 2013](https://jmlr.org/papers/volume14/hoffman13a/hoffman13a.pdf).

The default KL multiplier is one, with parameter-count normalization and
automatic group-size damping disabled. Explicit alternative multipliers,
outcome weights, and legacy reductions define tempered or weighted objectives.
The rule beta = 1 / mean group size is an optional empirical heuristic with no
ELBO or exact-BLUP derivation. Explicit structural penalties and AdamW decay on
deterministic network weights are additional regularization, not part of the
displayed likelihood/KL objective. Variational, covariance, and likelihood
dispersion parameters are excluded from AdamW decay. The linear-model factory
uses zero decay.

## Covariance and level semantics

The same standard-normal latent prior and KL are used for IID, unstructured
slope, K, GP, and Kronecker effects. The likelihood uses the transformed
effects, not a second independently scaled version of the variational means.
For K, u_train = sigma chol(K_train + jitter I) v. For a GP the corresponding
kernel is learned through its lengthscale and amplitude. Kronecker effects
apply both the group and slope factors. Numerical jitter is part of the
implemented covariance and is retained consistently in the solves.

There is one variational row per genuine training level. Index zero is reserved
only for prediction lookup and contributes no phantom latent variable or KL.
Supplied full-level orders are checked against observed training levels. For
known but unobserved related levels, prediction uses a Gaussian conditional
mean through K_new,train K_train,train^-1 u_train. This is a plug-in prediction:
the implementation does not add the conditional residual covariance for those
levels. It must not be described as a full predictive distribution or an exact
BLUP when the fitted mean is variational and the downstream response is nonlinear.

Compound-symmetry correlations lie between -1/(G-1) and 1 for G > 1. General
ARMA coefficients still describe a finite-start recurrence; independent tanh
constraints do not establish stationary AR(p) models for p > 1. Stationary
ARMA inference is outside the validated scope of this revision.

For dense SPDE noise with precision Q = LL^T, draws are L^-T z, z ~ N(0,I),
which have covariance Q^-1. The separable Kronecker construction is a different
precision approximation, not algebraically identical to the dense graph model.

## Identifiability and relation to classical mixed models

The formula design includes its requested fixed intercept exactly once.
Numeric random slopes use the same original covariate units as the fixed
design; any external standardization must be learned from training data and
applied to all partitions. Unstructured intercept/slope covariance is available
through `cov_struct_map={group: {'type': 'UN'}}`.

A zero-mean Gaussian prior is not a hard sample sum-to-zero constraint.
Optional `enforce_centering=True` subtracts the mean of transformed effects
over training levels, for both sampled effects and posterior-mean predictions.
It therefore replaces the effect covariance G by C G C^T for the specified
projection C. The KL remains on the original latent variables, giving an
augmented latent-variable model. This is not the ordinary unconstrained LMM,
nor does unweighted centering ensure orthogonality to every fixed-effect column
in an unbalanced design. Centering is disabled by default.

When the decoder is scalar and linear with a Gaussian outcome, the mean
structure reduces to X beta + Z u. For **fixed** covariance parameters and
unrestricted Gaussian variational means, the optimum mean solves the Gaussian
mixed-model normal equations. This is the only sense in which the model
connects to classical LMM theory, and it holds only under those two
restrictions simultaneously.

This implementation does **not** perform REML. REML integrates out the fixed
effects beta analytically and maximizes a restricted likelihood that accounts
for the loss of degrees of freedom from estimating beta. Mean-field VI
optimizes the ELBO — a lower bound on the marginal likelihood — with respect
to variational parameters (m, a) and, jointly, the covariance and network
weights. These are fundamentally different algorithms: REML is a closed-form
marginal optimization; mean-field VI is a stochastic gradient procedure on a
variational bound. The mean-field restriction on q(v) can bias covariance
estimates relative to both ML and REML; nonconvex optimization can also stop
short of any stationary point. No equivalence between the two algorithms holds
in general, and none is claimed here. Compare the REML derivation in
[Bates et al., Fitting Linear Mixed-Effects Models Using lme4](https://cran.r-project.org/web/packages/lme4/vignettes/lmer.pdf).

For neural decoders, latent-channel rotations, scales, and transformations can
produce equivalent or similar fitted functions. Centering alone does not make
all embedding coordinates identifiable or give latent structural edges causal
meaning. The model is a predictive mixed-effects architecture unless additional
identification assumptions are supplied and justified.

## Training and evaluation

The generic training budget is 200 epochs, replacing the previous 20-epoch
default. A budget is not a convergence certificate. Report optimizer steps,
batch size, learning rate, stopping rule, preprocessing, seed, and validation
scheme for every experiment. The validated Sleepstudy protocol uses train-only
standardization, 2,000 full-batch AdamW updates at learning rate 0.01, a scalar
linear decoder, unstructured intercept/slope effects, beta = 1, and no centering
or weight decay. Three explicit seeds are reported. These settings were not
selected using query labels; no claim of optimal hyperparameters is made.

Keep these evaluation targets separate:

1. Training fitted values: descriptive goodness of fit only.
2. Held-out observations from known groups: fitted group information is available.
3. New unrelated groups without outcomes: zero latent effects, or the explicit
   related-level conditional mean when a relationship matrix is supplied.
4. New groups with labeled support: condition only on support outcomes and
   evaluate on disjoint query observations from those groups.

For case 4, freeze all fitted parameters and optimize standard-normal
innovations for the new groups under the learned IID or UN covariance. The
objective is the sum of support negative log likelihoods plus half the squared
innovation norm. This is conditional MAP adaptation, not a variational E-step.
For a linear Gaussian random intercept with known variances, it gives

    b_hat = sum_support(y - x beta) / (n_support + sigma_e^2 / sigma_b^2).

The revised `predict_transductive(query_df, support_df=...)` requires explicit
support data and ignores query outcomes. Users must keep support and query
observations disjoint for predictive evaluation. IID and UN are the supported
new-group adaptation priors; other structures are rejected instead of silently
approximated. Neither MAP adaptation nor zero-latent plug-in predictions provide
posterior predictive intervals, and adaptation need not improve every split.

## Estimands and reporting

In a scalar linear identity-head model without output SEM, parameter extraction
returns each transformed random coefficient separately in its formula units.
These are variational posterior-mean coefficients; use “BLUP” only when the
classical assumptions and estimation equivalence are established. For nonlinear
models, reported random-effect contributions are group-averaged, single-term
prediction ablations. They depend on the covariate distribution, need not add
up under interactions, and do not identify separate random slopes. Binomial
ablations use probability units. Linearized fixed effects are average
derivatives on the primary predictor scale, not conventional global coefficients.

Covariance summaries generally describe encoder-space scales. They are not
automatically outcome-unit variance components or ICCs in nonlinear models.
Standard errors and p values are not implemented. Conditional plug-in log
likelihood must not be substituted for marginal likelihood in classical
AIC/BIC; those criteria are now unavailable. MC variation from latent draws
alone does not include observation noise or all parameter uncertainty.

## Structural architecture

The default composite block (`CompositeStructuralModel`) is a **gated
static-dynamic block**. Its forward pass is the three-step sequential
computation:

    u = (I - B_static)^{-1} xi          # static resolve
    a = Attention(u)                     # dynamic attention over resolved state
    h = alpha * u + (1 - alpha) * a      # gated fusion, alpha in (0, 1)

This is a single-pass feedforward architecture. It is **not** a fixed-point
equation and is **not** an iterative solver. The manuscript equation that
describes a coupled fixed point (where both the static and dynamic terms depend
on the same unknown h) is a different model. The two are not equivalent and
cannot be made equivalent by choosing alpha: with B_static = 0, Attention = 0,
alpha = 0.5, the gated block returns xi/2 while the fixed-point equation
returns xi — a 50% relative error at every input. This counterexample is
verified analytically in the test suite
(`test_composite_is_not_one_iteration_even_with_zero_operators`).

The manuscript equation must be revised to describe the gated static-dynamic
block as implemented, or the coupled fixed-point solver must be separately
implemented, validated for convergence, and described on its own terms before
any claims about it are made. `HybridStructuralModel` provides the iterative
path; it is not the default and its convergence is not guaranteed for
state-dependent attention.

No contraction theorem for the gated block follows from row-normalized
attention weights alone: the spectral norm of the full map depends on the
value and output projections, not only on the attention score matrix.

## Claims to remove from the manuscript

Remove or replace with accurate descriptions: exact REML recovery, exact
nonlinear BLUPs, learned causal DAGs, implemented SHAP/edge Shapley values,
calibrated prediction intervals, and complete reproduction of the published
tables. Some public legacy methods still use these labels and are listed as
unresolved in the audit. The new Sleepstudy runs establish only the specific
validation scope in the companion report; the original Tables 2–4 require
their original datasets, splits, configurations, and per-fold outputs before
their claims can be verified.
