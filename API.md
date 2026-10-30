# TabMixNN API Reference

Complete API reference for TabMixNN - a flexible framework for modeling tabular data with mixed effects, structural equation modeling (SEM) capabilities, and modern deep learning backbones.

## Table of Contents
1. [Core Features](#core-features)
2. [Model Initialization](#model-initialization)
3. [Convenience Classes](#convenience-classes)
4. [Formula Syntax](#formula-syntax)
5. [Outcome Specification](#outcome-specification)
6. [Supported Families](#supported-families)
7. [Training](#training)
8. [Prediction](#prediction)
9. [Architecture Options](#architecture-options)
10. [Performance & Best Practices](#performance--best-practices)

---

## Core Features
- **Mixed Effects**: Supports fixed and random effects using `lme4`-style formulas (e.g., `(1|group)`).
- **Multitask Learning**: Jointly model multiple outcomes (regression, classification, survival) with shared representations.
- **Deep Backbones**: Use Generalized SEM (GSEM) or Universal Manifold backbones for complex non-linear interactions.
- **Survival Analysis**: Native support for censored data using Discrete Time and Continuous Time survival families.

---

## Model Initialization
The main class is `TabMixNN`.

```python
from tabmixnn import TabMixNN

model = TabMixNN(
    formula="~ age + (1|hospital_id)",
    outcomes=[OutcomeSpec(...)],
    architecture="gsem",
    hidden_dims=[64, 32],
    embedding_dim=16,
    epochs=100,
    lr=0.01
)
```

### Key Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `formula` | `str` | Model formula defining inputs and random effects | Required |
| `outcomes` | `List[OutcomeSpec]` | List of outcome specifications | `None` |
| `family` | `Family` | Default family (if outcomes not provided) | `Gaussian()` |
| `architecture` | `str` | Backbone architecture (`"gsem"` or `"manifold"`) | `"gsem"` |
| `embedding_dim` | `int` | Dimension for categorical embeddings | `16` |
| `hidden_dims` | `List[int]` | Hidden layer sizes for backbone | `[]` |
| `dropout` | `float` | Dropout rate | `0.0` |
| `lr` | `float` | Learning rate | `0.001` |
| `epochs` | `int` | Number of training epochs | `100` |
| `batch_size` | `int` | Batch size for training | `128` |
| `early_stopping` | `bool` | Enable early stopping | `False` |
| `patience` | `int` | Early stopping patience | `10` |
| `device` | `str` | Device (`"cpu"` or `"cuda"`) | `"cpu"` |

---

## Convenience Classes

For common use cases, TabMixNN provides specialized classes.

### TabMixNNRegressor
For regression tasks with Gaussian family.

```python
from tabmixnn import TabMixNNRegressor

model = TabMixNNRegressor(
    formula="~ x1 + x2 + (1|group)",
    epochs=50,
    lr=0.01
)
model.fit(df, y='target')
predictions = model.predict(df)
```

### TabMixNNClassifier
For binary classification with Binomial family.

```python
from tabmixnn import TabMixNNClassifier

model = TabMixNNClassifier(
    formula="~ x1 + x2 + (1|group)",
    epochs=50,
    lr=0.01
)
model.fit(df, y='label')
predictions = model.predict(df)  # Returns probabilities
```

---

## Formula Syntax
TabMixNN uses a subset of `lme4` syntax, parsed via `formulaic`.

### Basic Syntax
```python
# Fixed effects only
"y ~ x1 + x2"

# Random intercept
"y ~ x1 + (1|group)"

# Random slope
"y ~ x1 + (x1|group)"

# Random intercept and slope
"y ~ x1 + (1 + x1|group)"

# Multiple grouping factors
"y ~ x1 + (1|group1) + (1|group2)"
```

### Notes
- The left-hand side (outcome) is optional if `outcomes` list is provided explicitly
- Random effects use the pipe syntax: `(terms|grouping_variable)`
- Multiple random effects can be specified with `+`

---

## Outcome Specification
Use `OutcomeSpec` to define each target variable in multitask learning.

```python
from tabmixnn import OutcomeSpec
from tabmixnn.families import Gaussian, Binomial, DiscreteTimeSurvival

outcomes = [
    OutcomeSpec(name="height", family=Gaussian(), y="height_cm"),
    OutcomeSpec(name="disease", family=Binomial(), y="has_disease")
]
```

### OutcomeSpec Parameters

| Parameter | Description | Required For |
|-----------|-------------|--------------|
| `name` | Unique identifier for this outcome | All outcomes |
| `family` | Family distribution instance | All outcomes |
| `y` | Column name for target variable | Gaussian, Binomial, Poisson, etc. |
| `time` | Column name for time variable | All survival families |
| `event` | Column name for event indicator (0/1) | DiscreteTimeSurvival |
| `cause` | Column name for cause indicator (0,1,2,...) | DiscreteTimeCompetingRisks |
| `loss_weight` | Weight for this outcome in multitask loss | Optional (default: 1.0) |

### Survival Outcome Examples

```python
import numpy as np

# Discrete-Time Survival
time_bins = np.array([0, 6, 12, 24, 36])  # Time interval boundaries
outcomes = [
    OutcomeSpec(
        name="survival",
        time="time_column",      # Column with observed time
        event="event_column",    # Column with event indicator (1=event, 0=censored)
        family=DiscreteTimeSurvival(time_bins=time_bins)
    )
]

# Competing Risks
outcomes = [
    OutcomeSpec(
        name="competing_risks",
        time="time_column",
        cause="cause_column",    # 0=censored, 1=cause1, 2=cause2, ...
        family=DiscreteTimeCompetingRisks(time_bins=time_bins, num_causes=2)
    )
]

# Continuous Survival (Weibull-like)
outcomes = [
    OutcomeSpec(
        name="survival",
        time="time_column",
        event="event_column",
        family=ContinuousSurvival()
    )
]
```

### Multitask with Mixed Outcome Types

```python
# Combine survival with other outcome types
outcomes = [
    OutcomeSpec(
        name="survival",
        time="time",
        event="event",
        family=DiscreteTimeSurvival(time_bins=bins)
    ),
    OutcomeSpec(name="biomarker", family=Gaussian(), y="biomarker_scaled"),
    OutcomeSpec(name="response", family=Binomial(), y="response"),
]

model = TabMixNN(
    formula="~ age + treatment + (1|hospital)",
    outcomes=outcomes,
    hidden_dims=[64, 32],
    epochs=200
)
```

---

## Supported Families
Import from `tabmixnn.families`.

| Family | Task | Output | Use Case |
|--------|------|--------|----------|
| `Gaussian()` | Regression | Continuous values | Height, weight, scores |
| `Binomial()` | Binary Classification | Probability (0-1) | Disease presence, success/failure |
| `Poisson()` | Count Data | Non-negative integers | Number of events, visits |
| `NegBinomial()` | Overdispersed Counts | Non-negative integers | Rare events with high variance |
| `Multinomial()` | Multi-class | Class probabilities | Multi-category outcomes |
| `DiscreteTimeSurvival()` | Time-to-Event | Survival curves | Patient survival, time to failure |
| `DiscreteTimeCompetingRisks()` | Competing Events | Cause-specific CIF | Death vs relapse |
| `ContinuousSurvival()` | Parametric Survival | Hazard parameters | Weibull/exponential models |

### Family-Specific Parameters

```python
# Gaussian - no parameters needed
from tabmixnn.families import Gaussian
family = Gaussian()

# Discrete-Time Survival - requires time bins
from tabmixnn.families import DiscreteTimeSurvival
time_bins = np.array([0, 10, 20, 30, 40])
family = DiscreteTimeSurvival(time_bins=time_bins)

# Competing Risks - requires time bins and number of causes
from tabmixnn.families import DiscreteTimeCompetingRisks
family = DiscreteTimeCompetingRisks(time_bins=time_bins, num_causes=2)
```

---

## Training
Use the `fit` method to train the model.

Each `fit()` starts afresh, rebuilding preprocessing, network parameters,
optimizer, and training histories. `reuse_design=existing_design` explicitly
reuses preprocessing only. It does not resume training. Gaussian learned
output heads without output SEM initialize their bias from observed training
targets; validation targets do not participate. See the
[training and covariance repair notes](docs/package-improvements-2026-10-15.md)
for behavior changes and checkpoint compatibility.

```python
model.fit(
    df,                      # Training data (pandas DataFrame)
    val_df=None,            # Optional validation data
    val_fraction=0.2,       # Validation split (if val_df not provided)
    early_stopping=True,    # Enable early stopping
    patience=10,            # Early stopping patience
    batch_size=128,         # Batch size
    lr=0.001,              # Learning rate
    epochs=100             # Maximum epochs (set verbose on the constructor)
)
```

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `df` | Training DataFrame | Required |
| `val_df` | Validation DataFrame | `None` (uses val_fraction) |
| `val_fraction` | Fraction of data for validation | `0.2` |
| `early_stopping` | Stop when validation loss stops improving | `False` |
| `patience` | Epochs to wait before stopping | `10` |
| `batch_size` | Samples per batch | `128` |
| `lr` | Learning rate | `0.001` |
| `epochs` | Maximum training epochs | `100` |
| `verbose` | Print training progress | `True` |

---

## Prediction
Use `predict` with specific modes for different prediction types.

```python
# Standard point prediction (mean/probability)
preds = model.predict(df)

# Predict specific outcome in multitask model
preds = model.predict(df, outcome="target_name")

# Survival curves (returns survival probabilities over time)
surv_curves = model.predict(df, outcome="survival", mode="survival")

# Hazard probabilities
hazards = model.predict(df, outcome="survival", mode="hazard")

# Cumulative Incidence Functions (for competing risks)
cif = model.predict(df, outcome="competing_risks", mode="cif")

# Raw distribution parameters
params = model.predict(df, outcome="target", mode="raw_params")
```

### Prediction Modes

| Mode | Returns | Available For |
|------|---------|---------------|
| `"default"` | Point predictions (mean/prob) | All families |
| `"survival"` | Survival curves | DiscreteTimeSurvival, ContinuousSurvival |
| `"hazard"` | Hazard probabilities per bin | DiscreteTimeSurvival |
| `"cif"` | Cumulative incidence functions | DiscreteTimeCompetingRisks |
| `"raw_params"` | Distribution parameters (μ, σ, etc.) | All families |

---

## Architecture Options

### GSEM (Default)
Generalized Structural Equation Model backbone - robust and interpretable.

```python
model = TabMixNN(
    formula=formula,
    outcomes=outcomes,
    architecture="gsem",  # Default
    hidden_dims=[64, 32]  # Optional hidden layers
)
```

**Use GSEM when:**
- You need interpretable random effects
- Working with clear hierarchical/grouped data
- Dataset is small to medium sized
- Computational efficiency is important

### Manifold (Universal)
Learns smooth manifold representations for complex interactions.

```python
model = TabMixNN(
    formula=formula,
    outcomes=outcomes,
    architecture="manifold",
    hidden_dims=[64, 32],  # Required for manifold
    use_recursive=True,
    aggregation="concat"
)
```

**Use Manifold when:**
- Complex non-linear interactions expected
- Large datasets with rich structure
- Maximum predictive performance needed
- Exploring unknown functional forms

---

## Performance & Best Practices

### Recommended Settings by Task

#### 1. Standard Regression/Classification
```python
model = TabMixNN(
    formula="y ~ x1 + x2 + (1|group)",
    architecture="gsem",
    lr=0.01,              # Good starting point
    epochs=20-50,         # Usually sufficient
    embedding_dim=16,     # Standard for random effects
    early_stopping=True
)
```

#### 2. Multitask Learning
```python
model = TabMixNN(
    formula="~ x1 + x2 + (1|group)",
    outcomes=multiple_outcomes,
    architecture="gsem",
    lr=0.005,             # Lower for stability
    epochs=100-200,       # More epochs needed
    hidden_dims=[64, 32],
    early_stopping=True,
    patience=20
)
```

#### 3. Survival Analysis
```python
model = TabMixNN(
    formula="~ x1 + x2 + (1|group)",
    outcomes=[survival_outcome],
    architecture="gsem",
    lr=0.01,
    epochs=100-300,       # Survival needs more epochs
    batch_size=64,
    early_stopping=True,
    patience=15
)
```

### Data Preparation Best Practices

1. **Scale Continuous Targets**: Neural networks require standardization
   ```python
   df['y_scaled'] = (df['y'] - df['y'].mean()) / df['y'].std()
   ```

2. **Scale Input Features**: Especially important for manifold architecture
   ```python
   df['age_scaled'] = (df['age'] - df['age'].mean()) / df['age'].std()
   ```

3. **Categorical Variables**: Convert to strings for proper embedding
   ```python
   df['period'] = df['period'].astype(str)
   ```

4. **Binomial Data**: Expand aggregated counts to binary rows
   ```python
   # Instead of: incidence=10, size=50
   # Create: 10 rows with y=1, 40 rows with y=0
   ```

### Survival Analysis Patterns

1. **Time Bins**: Cover the full range of observed times
   ```python
   time_bins = np.percentile(df['time'], np.linspace(0, 100, 20))
   ```

2. **Competing Risks**: Use `cause` parameter (not `event`)
   ```python
   OutcomeSpec(
       name="competing",
       time="time",
       cause="cause_col",  # 0=censored, 1,2,...=causes
       family=DiscreteTimeCompetingRisks(time_bins=bins, num_causes=2)
   )
   ```

3. **Monotonic Survival Curves**: Automatically guaranteed through cumulative product

### Model Selection Guidelines

| Scenario | Architecture | Learning Rate | Epochs |
|----------|-------------|---------------|--------|
| Small dataset (N < 100) | `gsem` | 0.01 | 50-100 |
| Medium dataset (N < 1000) | `gsem` | 0.01 | 100-200 |
| Large dataset (N > 1000) | `manifold` | 0.005 | 200-500 |
| Complex non-linearity | `manifold` | 0.005 | 200-500 |
| Need interpretability | `gsem` | 0.01 | 100-200 |
| Multitask + Survival | `gsem` | 0.005 | 200-300 |

### Common Pitfalls

1. **Forgetting to scale**: Neural networks require scaled inputs/targets
2. **Too few epochs**: Survival and multitask models need 100+ epochs
3. **Learning rate too high**: Can cause instability (reduce to 0.005 or 0.001)
4. **Binomial data**: Must expand aggregated data to binary rows
5. **Time bins**: Must cover full range of observed survival times

### Performance Benchmarks

- **Abalone** (regression with groups): TabMixNN MSE 4.46 vs XGBoost 5.14
- **Sleepstudy** (longitudinal): Correctly recovers random slopes
- **CBPP** (binomial GLMM): Strong classification on grouped data
- **Rossi** (survival): C-index > 0.65 for recidivism prediction

---

## Additional Resources

- **Examples**: See [examples/](examples/) for practical, runnable code with real datasets
- **Vignettes**: See [vignettes/](vignettes/) for in-depth tutorials
- **Main README**: See [README.md](README.md) for quick start guide
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines

---

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

Copyright (c) 2025 Deniz Akdemir

See [LICENSE](LICENSE) for full details.
