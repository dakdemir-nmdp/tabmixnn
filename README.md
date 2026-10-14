# TabMixNN

**TabMixNN** is a PyTorch-based deep learning library that combines the best of mixed effects models and neural networks for tabular data analysis.

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
```

Or install from source:

```bash
git clone https://github.com/dakdemir-nmdp/tabmixnn.git
cd tabmixnn
pip install -e .
```

## Quick Start

### Basic Linear Mixed Model

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

- `Gaussian()` - Continuous outcomes (linear regression)
- `Binomial()` - Binary classification
- `Poisson()` - Count data
- `NegBinomial()` - Overdispersed count data
- `MultivariateGaussian()` - Multiple continuous outcomes
- `MultiLabelBinomial()` - Multi-label classification
- `DiscreteTimeSurvival()` - Survival analysis (discrete time)
- `DiscreteTimeCompetingRisks()` - Competing risks
- `ContinuousSurvival()` - Survival analysis (continuous time)

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
