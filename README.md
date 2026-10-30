# TabMixNN

**TabMixNN** is a PyTorch-based deep learning library that combines variational mixed-effects encoding with neural networks for tabular data analysis.

## Features

- **Mixed Effects Modeling**: Support for random intercepts and slopes with flexible grouping structures
- **Formula-based Interface**: R-style formula syntax for model specification (similar to lme4)
- **Multiple Distributions**: Gaussian, Binomial, Poisson, Negative Binomial, Survival models, and more
- **Flexible Architectures**:
  - GSEM (Generalized Structural Equation Models) backbone
  - Manifold backbone for complex latent representations
- **Multiple Outcomes**: Simultaneously model multiple related outcomes
- **Model Summary**: Comprehensive model summaries with parameter estimates

## Installation

```bash
pip install torch  # Install PyTorch first
pip install tabmixnn

# For running examples (matplotlib, scipy, xgboost):
pip install -e ".[examples]"

# Optional: lifelines for survival analysis utilities
pip install -e ".[lifelines]"
```

Or install from source:

```bash
git clone https://github.com/dakdemir-nmdp/tabmixnn.git
cd tabmixnn
pip install -e .
```

## Quick Start

### Neural Mixed-Effects Regression

```python
import numpy as np
import pandas as pd
from tabmixnn import TabMixNN
from tabmixnn.formula import OutcomeSpec
from tabmixnn.families import Gaussian

# Create sample data
n_samples = 500
df = pd.DataFrame({
    'x1': np.random.randn(n_samples),
    'x2': np.random.randn(n_samples),
    'group': np.random.choice(['A', 'B', 'C'], n_samples),
    'y': np.random.randn(n_samples)
})

# Define the outcome
outcomes = [OutcomeSpec(name='target', family=Gaussian(), y='y')]

# Specify formula: fixed effects (x1, x2) + random intercept by group
formula = "~ x1 + x2 + (1|group)"

# Create and fit model
model = TabMixNN(formula=formula, outcomes=outcomes, epochs=10)
model.fit(df)

# Make predictions
predictions = model.predict(df)

# View model summary
model.summary()
```

### Generalized Linear Mixed Model (GLMM)

```python
from tabmixnn.families import Binomial

# Binary outcome data
df['y_binary'] = (df['y'] > 0).astype(int)

outcomes = [OutcomeSpec(name='binary', family=Binomial(), y='y_binary')]
formula = "~ x1 + x2 + (1|group)"

model = TabMixNN(formula=formula, outcomes=outcomes, epochs=10)
model.fit(df)
predictions = model.predict(df)
```

### Random Slopes Model

```python
# Random intercepts AND slopes for x1 by group
formula = "~ x1 + x2 + (1 + x1|group)"

model = TabMixNN(formula=formula, outcomes=outcomes, epochs=10)
model.fit(df)
```

## Available Families

- `Gaussian()` - Continuous outcomes with a learned common residual scale
- `Multinomial(n_classes=3)` - Integer-coded multiclass outcomes
- `Binomial()` - Binary classification
- `Poisson()` - Count data
- `NegBinomial()` - Overdispersed count data
- `MultivariateGaussian(n_tasks=2)` - Multiple continuous outcomes
- `MultiLabelBinomial(n_labels=2)` - Multi-label classification
- `DiscreteTimeSurvival(time_bins=[1, 2, 3])` - Survival analysis (discrete time)
- `DiscreteTimeCompetingRisks(time_bins=[1, 2, 3], num_causes=2)` - Competing risks
- `ContinuousSurvival()` - Survival analysis (continuous time)

## Target and prediction contracts

Use `OutcomeSpec(..., y=['y1', 'y2'])` for a vector response. Its column count
must match the family. NaN denotes missing outcomes; vector rows must be fully
observed to contribute. Each modeled outcome needs observed training and
validation targets. Count and class labels must be nonnegative integers;
binary outcomes and event indicators must be 0 or 1. Class labels are encoded
explicitly by the caller. Prediction ignores any target columns supplied.

Discrete survival grids contain increasing right endpoints. Events use the
containing interval; censoring uses only completed intervals. Follow-up beyond
the grid is rejected. These semantics apply to competing risks as well.

For a fitted `ContinuousSurvival` outcome:

```python
curves = model.predict(test_data, mode="survival", times=[0., 1., 2., 5.])
medians = model.predict(test_data, mode="median")
log_hazard_scores = model.predict(test_data)  # explicit default semantics
```

Discrete-time survival prediction defaults to survival curves; request
`mode="hazard"` for interval hazards. Older behavior returned hazards when the
model passed `mode="default"`. Refit discrete survival models trained with
non-endpoint censoring under the earlier rounding convention.

Curves are conditional predictions with shape `(n_rows, n_times)`. They are
not prediction intervals. The baseline scale and a free score intercept are
not separately identified without a reference constraint.

`feature_ablation_values(query, background_df=train)` uses a separate reference
frame, preserves NumPy RNG state and reports compiled-design feature names.
Select `outcome=...` for multitask ablations. Vector outputs require an explicit
scalar estimand and currently raise `NotImplementedError`, rather than summing
probabilities or time curves into an arbitrary score. Group and random-slope
inputs remain fixed during a fixed-design feature ablation.

## Statistical scope

`TabMixNN.lmm(...)` is a scalar linear variational special case; it is not REML.
At fixed covariance, its optimal means satisfy the mixed-model normal equations.
Mean-field covariance and jointly fitted variance parameters need not match
exact Gaussian inference. In neural models, latent rescaling can change
encoder covariance without changing predictions. Latent variance summaries and
structural edges are therefore not identified response-scale or causal effects.

The [manuscript](manuscript/tabmixnn_methods.pdf) and
[deep revision audit](docs/deep-revision-2026-10-16.md) state tested contracts,
empirical scope and unresolved limitations. See [experiments](experiments/README.md)
for reproduction; outcome support alone is not empirical validation.

## Architecture Options

### GSEM Backbone (Default)
Generalized Structural Equation Models with optional structural constraints:

```python
model = TabMixNN(
    formula=formula,
    outcomes=outcomes,
    architecture="gsem",
    hidden_dims=[64, 32],
    epochs=10
)
```

### Manifold Backbone
Universal architecture with latent manifold representations:

```python
model = TabMixNN(
    formula=formula,
    outcomes=outcomes,
    architecture="manifold",
    hidden_dims=[64, 32],
    use_recursive=True,
    aggregation="concat",
    epochs=10
)
```

## Convenience Classes

For simpler use cases, you can use the convenience classes:

```python
from tabmixnn import TabMixNNRegressor, TabMixNNClassifier

# Regression (automatically uses Gaussian family)
regressor = TabMixNNRegressor(formula="~ x1 + x2 + (1|group)")
regressor.fit(df, y='y')
predictions = regressor.predict(df)

# Classification (automatically uses Binomial family)
classifier = TabMixNNClassifier(formula="~ x1 + x2 + (1|group)")
classifier.fit(df, y='y_binary')
predictions = classifier.predict(df)
```

## Model Inspection

```python
# Comprehensive model summary
model.summary()

# Extract parameter estimates
params = model.extract_parameters()
```

## Logging Control

```python
from tabmixnn import set_log_level, disable_logging, enable_logging

set_log_level('INFO')    # Set logging level
disable_logging()         # Turn off logging
enable_logging()          # Turn on logging
```

## Documentation

- **[API Reference](API.md)** - Comprehensive API documentation with all parameters, families, and methods
- **[Examples](examples/)** - 7 practical examples with real datasets:
  - Linear mixed models, classification, random slopes
  - Multi-task learning, survival analysis, competing risks
  - Architecture comparison (GSEM vs Manifold)
- **[Vignettes](vignettes/)** - In-depth tutorials:
  - Getting started with TabMixNN
  - Mixed effects modeling in depth
  - Multi-task learning
  - Survival analysis and competing risks

## Citation

If you use TabMixNN in your research, please cite:

```bibtex
@software{tabmixnn,
  title={TabMixNN: Deep Learning for Tabular Data with Mixed Effects},
  author={Deniz Akdemir},
  year={2025},
  url={https://github.com/dakdemir-nmdp/tabmixnn}
}
```

## License

MIT License — Copyright (c) 2025 Deniz Akdemir. See the [LICENSE](LICENSE) file for full details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
