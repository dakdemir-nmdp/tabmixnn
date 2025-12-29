# Getting Started with TabMixNN

This vignette provides a comprehensive introduction to TabMixNN, covering installation, basic concepts, and your first models.

## Table of Contents

1. [Installation](#installation)
2. [Core Concepts](#core-concepts)
3. [Your First Model](#your-first-model)
4. [Understanding the Formula Syntax](#understanding-the-formula-syntax)
5. [Working with Different Families](#working-with-different-families)
6. [Model Fitting and Prediction](#model-fitting-and-prediction)
7. [Model Inspection](#model-inspection)

---

## Installation

Install TabMixNN and its dependencies:

```bash
# Install PyTorch (required)
pip install torch

# Install TabMixNN
pip install tabmixnn

# Or from source
git clone https://github.com/yourusername/tabmixnn.git
cd tabmixnn
pip install -e .
```

Verify installation:

```python
import tabmixnn
print(tabmixnn.__version__)
```

---

## Core Concepts

### What is TabMixNN?

TabMixNN combines two powerful modeling paradigms:

1. **Mixed Effects Models**: Account for hierarchical/grouped data structure (e.g., patients within hospitals, students within schools)
2. **Deep Learning**: Capture complex non-linear relationships and interactions

### When to Use TabMixNN?

Use TabMixNN when you have:

- **Grouped/hierarchical data**: Observations nested within groups (centers, sites, patients)
- **Repeated measurements**: Multiple observations per subject
- **Complex relationships**: Non-linear patterns, interactions
- **Multiple outcomes**: Several related dependent variables
- **Mixed data types**: Continuous, binary, count, survival outcomes

### Key Components

1. **Formula**: R-style syntax for specifying model structure
2. **Outcomes**: Define what you're predicting and its distribution
3. **Families**: Distribution families (Gaussian, Binomial, Poisson, etc.)
4. **Architecture**: Neural network backbone (GSEM or Manifold)

---

## Your First Model

Let's build a simple linear mixed model step by step.

### Step 1: Prepare Data

```python
import numpy as np
import pandas as pd

# Simulate multi-center study data
np.random.seed(42)
n_centers = 5
n_per_center = 100

df = pd.DataFrame({
    'patient_id': range(n_centers * n_per_center),
    'age': np.random.randint(40, 80, n_centers * n_per_center),
    'treatment': np.random.choice(['A', 'B'], n_centers * n_per_center),
    'center_id': np.repeat(range(n_centers), n_per_center),
    'outcome': np.random.randn(n_centers * n_per_center)
})

print(df.head())
```

### Step 2: Create the Model

```python
from tabmixnn import TabMixNNRegressor

# Create model with formula
model = TabMixNNRegressor(
    formula="outcome ~ age + treatment + (1|center_id)",
    epochs=20,
    verbose=True
)
```

### Step 3: Fit the Model

```python
# Fit model on data
model.fit(df)
```

### Step 4: Make Predictions

```python
# Predict on same data
predictions = model.predict(df)

# Predict on new data
new_data = pd.DataFrame({
    'age': [50, 60],
    'treatment': ['A', 'B'],
    'center_id': [0, 1]
})
new_predictions = model.predict(new_data)
print(new_predictions)
```

---

## Understanding the Formula Syntax

TabMixNN uses R-style formulas (similar to lme4, statsmodels):

### Basic Syntax

```python
# General form:
"outcome ~ fixed_effects + (random_effects|grouping_variable)"

# Examples:
"y ~ x1 + x2"                    # Fixed effects only (no random effects)
"y ~ x1 + x2 + (1|group)"        # Random intercept
"y ~ x1 + x2 + (1 + x1|group)"   # Random intercept and slope
"y ~ x1 + x2 + (1|group1) + (1|group2)"  # Multiple grouping factors
```

### Formula Components

#### Left-hand side (outcome):
- **With outcomes parameter**: Use `"~ ..."` (tilde without outcome)
- **With convenience classes**: Use `"y ~ ..."` (outcome name)

#### Right-hand side:

**Fixed effects** (affect all observations the same way):
```python
"~ x1"              # Single continuous predictor
"~ x1 + x2"         # Multiple predictors
"~ x1 + x2 + x3"    # Additive effects
"~ ."               # All columns except outcome (auto-detect)
```

**Random effects** (vary by group):
```python
"~ x1 + (1|group)"           # Random intercept per group
"~ x1 + (x1|group)"          # Random slope for x1 per group
"~ x1 + (1 + x1|group)"      # Both random intercept and slope
"~ x1 + (1|group1) + (1|group2)"  # Multiple grouping factors
```

### Examples with Interpretation

```python
# Example 1: Simple random intercept
formula = "~ age + sex + (1|hospital)"
# Interpretation: age and sex have same effect across all hospitals,
# but each hospital has its own baseline level

# Example 2: Random slopes
formula = "~ age + treatment + (1 + treatment|hospital)"
# Interpretation: Treatment effect varies across hospitals

# Example 3: Multiple grouping factors
formula = "~ age + (1|hospital) + (1|physician)"
# Interpretation: Both hospital and physician have random effects
```

---

## Working with Different Families

TabMixNN supports multiple outcome distributions through the `family` parameter.

### Available Families

#### Continuous Outcomes: Gaussian

```python
from tabmixnn import TabMixNNRegressor
from tabmixnn.families import Gaussian
from tabmixnn.formula import OutcomeSpec

# Using convenience class
model = TabMixNNRegressor(formula="y ~ x1 + x2 + (1|group)")

# Or explicit family
outcomes = [OutcomeSpec(name='y', y='y', family=Gaussian())]
model = TabMixNN(formula="~ x1 + x2 + (1|group)", outcomes=outcomes)
```

#### Binary Outcomes: Binomial

```python
from tabmixnn import TabMixNNClassifier
from tabmixnn.families import Binomial

# Using convenience class
model = TabMixNNClassifier(formula="binary_outcome ~ x1 + x2 + (1|group)")

# Or explicit family
outcomes = [OutcomeSpec(name='binary', y='binary_outcome', family=Binomial())]
```

#### Count Data: Poisson and Negative Binomial

```python
from tabmixnn.families import Poisson, NegBinomial

# Poisson for count data
outcomes = [OutcomeSpec(name='count', y='visit_count', family=Poisson())]

# Negative Binomial for overdispersed counts
outcomes = [OutcomeSpec(name='count', y='visit_count', family=NegBinomial())]
```

#### Survival Outcomes

```python
from tabmixnn.families import DiscreteTimeSurvival

outcomes = [
    OutcomeSpec(
        name='survival',
        time='time_to_event',
        event='event_indicator',
        family=DiscreteTimeSurvival(max_time=10)
    )
]
```

---

## Model Fitting and Prediction

### Fitting Options

```python
model = TabMixNNRegressor(
    formula="y ~ x1 + x2 + (1|group)",
    epochs=50,              # Number of training epochs
    batch_size=64,          # Batch size for training
    learning_rate=0.01,     # Learning rate
    verbose=True,           # Print training progress
    patience=10,            # Early stopping patience
)

# Fit the model
model.fit(df)
```

### Making Predictions

```python
# Predict on training data
train_predictions = model.predict(df)

# Predict on new data (must have same columns)
new_predictions = model.predict(new_df)

# For classification, get probabilities
probabilities = model.predict(new_df)
classes = (probabilities > 0.5).astype(int)
```

### Saving and Loading Models

```python
# Save trained model
model.save('my_model.pt')

# Load model
from tabmixnn import TabMixNNRegressor
loaded_model = TabMixNNRegressor.load('my_model.pt')

# Use loaded model
predictions = loaded_model.predict(new_data)
```

---

## Model Inspection

### Model Summary

```python
# Display comprehensive summary
model.summary()
```

Output includes:
- Architecture details
- Number of parameters
- Random effects structure
- Training history

### Parameter Estimates

```python
# Extract parameter estimates
params = model.extract_parameters()
print(params)
```

### Accessing Components

```python
# Access encoder (handles mixed effects)
encoder = model.encoder

# Access backbone architecture
backbone = model.backbone

# Access outcome heads
heads = model.heads
```

---

## Next Steps

Now that you understand the basics:

1. Explore [examples](../examples/) for specific use cases
2. Read [vignette 02](02_mixed_effects_modeling.md) for deep dive into mixed effects
3. See [vignette 03](03_multitask_learning.md) for multiple outcomes
4. Check [vignette 04](04_model_selection.md) for model comparison

---

## Common Issues and Solutions

### Issue: "Column not found" error

**Solution**: Ensure all variables in the formula are present in your DataFrame.

```python
# Check columns
print(df.columns)

# Verify formula variables
formula = "y ~ age + treatment + (1|center)"
required_cols = ['y', 'age', 'treatment', 'center']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    print(f"Missing columns: {missing}")
```

### Issue: Model not converging

**Solutions**:
- Increase epochs: `epochs=100`
- Adjust learning rate: `learning_rate=0.001`
- Change batch size: `batch_size=128`
- Scale your features: `df['age'] = (df['age'] - df['age'].mean()) / df['age'].std()`

### Issue: Out of memory errors

**Solutions**:
- Reduce batch size: `batch_size=32`
- Use smaller architecture: `hidden_dims=[32, 16]`
- Process data in chunks

---

## Summary

In this vignette, you learned:

- ✓ How to install TabMixNN
- ✓ Core concepts: mixed effects + deep learning
- ✓ Formula syntax for model specification
- ✓ Different outcome families
- ✓ Fitting and predicting with models
- ✓ Inspecting trained models

Ready to go deeper? Check out the other vignettes!
