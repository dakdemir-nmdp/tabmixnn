# Multi-task Learning with TabMixNN

This vignette demonstrates how to model multiple related outcomes simultaneously using TabMixNN's multi-task learning capabilities.

## Table of Contents

1. [What is Multi-task Learning?](#what-is-multi-task-learning)
2. [Why Model Multiple Outcomes Together?](#why-model-multiple-outcomes-together)
3. [Defining Multiple Outcomes](#defining-multiple-outcomes)
4. [Heterogeneous Outcome Types](#heterogeneous-outcome-types)
5. [Structural Equation Models (SEM)](#structural-equation-models-sem)
6. [Advanced: Output-Level SEM](#advanced-output-level-sem)
7. [Best Practices](#best-practices)

---

## What is Multi-task Learning?

Multi-task learning trains a single model to predict multiple related outcomes simultaneously, sharing representations across tasks.

### Single-Task vs Multi-Task

**Single-Task** (traditional):
```
Model 1: X → y₁
Model 2: X → y₂
Model 3: X → y₃
```

**Multi-Task**:
```
        ┌→ y₁
Model: X ├→ y₂
        └→ y₃
```

Benefits:
- **Shared learning**: Information from one task helps others
- **Efficiency**: One model instead of many
- **Better generalization**: Multi-task regularization
- **Modeling relationships**: Can specify causal structure between outcomes

---

## Why Model Multiple Outcomes Together?

### Clinical Example: Transplant Outcomes

In hematopoietic cell transplantation, multiple outcomes are related:

- **Acute GVHD** → affects → **Chronic GVHD**
- **GVHD** → affects → **Relapse** (immunological)
- **GVHD + Relapse** → affects → **Survival**

Modeling these together:
1. Captures dependencies between outcomes
2. Improves predictions for all outcomes
3. Can specify causal relationships

---

## Defining Multiple Outcomes

### Basic Multi-task Model

```python
import numpy as np
import pandas as pd
from tabmixnn import TabMixNN, OutcomeSpec, Gaussian, Binomial

# Simulate data
np.random.seed(42)
n = 400

df = pd.DataFrame({
    'age': np.random.randint(18, 70, n),
    'treatment': np.random.choice(['A', 'B'], n),
    'center': np.random.choice(['C1', 'C2', 'C3'], n),
})

# Simulate related outcomes
# Quality of life (continuous)
df['qol'] = (
    60 + 0.2 * (df['age'] - 50) +
    10 * (df['treatment'] == 'B') +
    np.random.randn(n) * 5
)

# Adverse event (binary) - related to QoL
ae_prob = 1 / (1 + np.exp(-(-2 - 0.05 * df['qol'] + np.random.randn(n) * 0.5)))
df['adverse_event'] = (np.random.random(n) < ae_prob).astype(int)

# Define outcomes
outcomes = [
    OutcomeSpec(name="quality_life", y="qol", family=Gaussian()),
    OutcomeSpec(name="adverse_event", y="adverse_event", family=Binomial())
]

# Formula applies to all outcomes
formula = "~ age + treatment + (1|center)"

# Create multi-task model
model = TabMixNN(
    formula=formula,
    outcomes=outcomes,
    architecture="gsem",
    hidden_dims=[64, 32],
    epochs=20,
    verbose=True
)

model.fit(df)

# Predictions return dictionary
predictions = model.predict(df)
print("Predictions:", predictions.keys())
print("QoL predictions:", predictions['quality_life'][:5])
print("AE predictions:", predictions['adverse_event'][:5])
```

---

## Heterogeneous Outcome Types

TabMixNN supports mixing different outcome types in one model.

### Example: Comprehensive Patient Profile

```python
from tabmixnn import Gaussian, Binomial, Poisson, DiscreteTimeSurvival

np.random.seed(42)
n = 500

# Simulate comprehensive patient data
df = pd.DataFrame({
    'age': np.random.randint(30, 80, n),
    'bmi': np.random.normal(25, 5, n),
    'smoker': np.random.choice([0, 1], n, p=[0.7, 0.3]),
    'hospital': np.random.choice([f'H{i}' for i in range(5)], n),
})

# 1. Pain score (continuous, 0-10)
df['pain_score'] = np.clip(
    5 + 0.05 * df['age'] + 2 * df['smoker'] + np.random.randn(n) * 1.5,
    0, 10
)

# 2. Readmission (binary)
readmit_prob = 1 / (1 + np.exp(-(
    -3 + 0.03 * df['age'] + 0.1 * df['pain_score'] + 0.5 * df['smoker']
)))
df['readmission'] = (np.random.random(n) < readmit_prob).astype(int)

# 3. Number of complications (count)
complication_rate = np.exp(0.2 + 0.02 * df['age'] + 0.3 * df['readmission'])
df['complications'] = np.random.poisson(complication_rate)

# 4. Time to recovery (survival)
hazard = 0.15 * np.exp(0.02 * df['age'] + 0.3 * df['smoker'])
df['recovery_time'] = np.random.exponential(1/hazard)
df['recovery_event'] = (df['recovery_time'] < 5).astype(int)
df['recovery_time'] = np.minimum(df['recovery_time'], 5)

# Define all four outcomes
outcomes = [
    OutcomeSpec(
        name="pain",
        y="pain_score",
        family=Gaussian()
    ),
    OutcomeSpec(
        name="readmission",
        y="readmission",
        family=Binomial()
    ),
    OutcomeSpec(
        name="complications",
        y="complications",
        family=Poisson()
    ),
    OutcomeSpec(
        name="recovery",
        time="recovery_time",
        event="recovery_event",
        family=DiscreteTimeSurvival(time_bins=list(range(6)))  # 5 intervals: 0..5
    )
]

# Multi-task model with all four outcome types
model = TabMixNN(
    formula="~ age + bmi + smoker + (1|hospital)",
    outcomes=outcomes,
    architecture="gsem",
    hidden_dims=[64, 32],
    epochs=25,
    verbose=True
)

print("Fitting model with 4 different outcome types...")
model.fit(df)

# Get predictions for all outcomes
predictions = model.predict(df)

print("\nPredictions for all outcomes:")
for outcome_name, preds in predictions.items():
    print(f"  {outcome_name}: shape {preds.shape}")

# Evaluate each outcome
print("\nPerformance:")

# Pain (regression)
pain_r2 = 1 - np.mean((predictions['pain'] - df['pain_score'])**2) / np.var(df['pain_score'])
print(f"  Pain Score R²: {pain_r2:.4f}")

# Readmission (classification)
readmit_acc = np.mean((predictions['readmission'] > 0.5) == df['readmission'])
print(f"  Readmission Accuracy: {readmit_acc:.4f}")

# Complications (count)
comp_mae = np.mean(np.abs(predictions['complications'] - df['complications']))
print(f"  Complications MAE: {comp_mae:.4f}")

model.summary()
```

---

## Structural Equation Models (SEM)

SEM allows you to specify causal relationships between outcomes.

### Specifying Causal Structure

```python
# Example: Disease progression
# Stage 1 → Stage 2 → Stage 3 → Death

outcomes = [
    OutcomeSpec(name="stage1", y="stage1_event", family=Binomial()),
    OutcomeSpec(name="stage2", y="stage2_event", family=Binomial()),
    OutcomeSpec(name="stage3", y="stage3_event", family=Binomial()),
    OutcomeSpec(name="death", time="time_to_death", event="death_event",
                family=DiscreteTimeSurvival(max_time=10))
]

# Specify causal structure using lavaan syntax
sem_structure = """
    stage2 ~ stage1
    stage3 ~ stage1 + stage2
    death ~ stage1 + stage2 + stage3
"""

model = TabMixNN(
    formula=formula,
    outcomes=outcomes,
    architecture="gsem",
    output_sem_mode="lavaan",
    output_structure=sem_structure,
    hidden_dims=[64, 32],
    epochs=30,
    verbose=True
)
```

This creates a model where:
- Stage 2 depends on Stage 1
- Stage 3 depends on Stages 1 and 2
- Death depends on all three stages

---

## Advanced: Output-Level SEM

TabMixNN supports three SEM modes for output relationships:

### 1. Independent (no structure)

```python
model = TabMixNN(
    formula=formula,
    outcomes=outcomes,
    output_sem_mode="none",  # Default: independent outcomes
)
```

### 2. Lavaan-specified (domain knowledge)

```python
# Specify known causal structure
sem_structure = """
    chronic_gvhd ~ acute_gvhd
    relapse ~ acute_gvhd
    death ~ acute_gvhd + chronic_gvhd + relapse
"""

model = TabMixNN(
    formula=formula,
    outcomes=outcomes,
    output_sem_mode="lavaan",
    output_structure=sem_structure,
)
```

### 3. Learned (data-driven)

```python
# Let model learn causal structure
model = TabMixNN(
    formula=formula,
    outcomes=outcomes,
    output_sem_mode="learned",  # Learn structure from data
)
```

### Complete Example: HCT Outcomes

```python
from tabmixnn import ContinuousSurvival

np.random.seed(42)
n = 400

# Transplant data
df = pd.DataFrame({
    'age': np.random.randint(18, 70, n),
    'hla_match': np.random.choice(['matched', 'mismatched'], n, p=[0.7, 0.3]),
    'conditioning': np.random.choice(['MAC', 'RIC'], n, p=[0.4, 0.6]),
    'center_id': np.random.randint(1, 10, n),
})

# Causal outcomes
acute_prob = 0.2 + 0.15 * (df['hla_match'] == 'mismatched')
df['acute_gvhd'] = (np.random.random(n) < acute_prob).astype(int)

chronic_prob = 0.25 + 0.3 * df['acute_gvhd']
df['chronic_gvhd'] = (np.random.random(n) < chronic_prob).astype(int)

relapse_rate = np.exp(1.0 - 0.5 * df['acute_gvhd'])
df['relapse'] = np.random.poisson(relapse_rate)

hazard = 0.1 * np.exp(0.5 * df['acute_gvhd'] + 0.4 * df['chronic_gvhd'] + 0.3 * (df['relapse'] > 0))
df['death_time'] = np.random.exponential(1 / hazard)
df['death_event'] = (df['death_time'] < 5).astype(int)

# Define outcomes with causal structure
outcomes = [
    OutcomeSpec(name="acute_gvhd", y="acute_gvhd", family=Binomial()),
    OutcomeSpec(name="chronic_gvhd", y="chronic_gvhd", family=Binomial()),
    OutcomeSpec(name="relapse", y="relapse", family=Poisson()),
    OutcomeSpec(name="death", time="death_time", event="death_event",
                family=ContinuousSurvival())
]

# Causal structure based on clinical knowledge
sem_structure = """
    chronic_gvhd ~ acute_gvhd
    relapse ~ acute_gvhd
    death ~ acute_gvhd + chronic_gvhd + relapse
"""

model = TabMixNN(
    formula="~ age + hla_match + conditioning + (1|center_id)",
    outcomes=outcomes,
    architecture="gsem",
    output_sem_mode="lavaan",
    output_structure=sem_structure,
    hidden_dims=[64, 32],
    epochs=30,
    verbose=True
)

print("Fitting SEM model with causal structure...")
model.fit(df)
model.summary()
```

---

## Best Practices

### 1. Start Simple, Add Complexity

```python
# Step 1: Independent outcomes
model1 = TabMixNN(formula=formula, outcomes=outcomes, sem_mode="none")

# Step 2: Add structure if outcomes are related
model2 = TabMixNN(formula=formula, outcomes=outcomes,
                  sem_mode="lavaan", sem_structure=structure)

# Compare performance
```

### 2. Choose SEM Mode Wisely

| Mode | When to Use |
|------|-------------|
| `none` | Outcomes are independent or you want baseline |
| `lavaan` | You have domain knowledge of causal relationships |
| `learned` | Exploratory analysis, let data reveal structure |

### 3. Balance Model Complexity

With many outcomes, consider:
- Simpler architectures: `hidden_dims=[32, 16]`
- Fewer random effects
- More training data

### 4. Evaluate Each Outcome

Don't just look at overall loss:

```python
predictions = model.predict(df)

for outcome_name, preds in predictions.items():
    # Compute appropriate metric for each outcome
    true_values = df[outcome_y_column]
    metric = compute_metric(preds, true_values)
    print(f"{outcome_name}: {metric}")
```

### 5. Visualize Relationships

```python
import matplotlib.pyplot as plt

# Plot predicted relationships between outcomes
plt.scatter(predictions['outcome1'], predictions['outcome2'], alpha=0.5)
plt.xlabel('Predicted Outcome 1')
plt.ylabel('Predicted Outcome 2')
plt.title('Predicted Relationship')
plt.show()
```

---

## Summary

Key concepts:

- **Multi-task learning**: Model multiple outcomes simultaneously
- **Heterogeneous outcomes**: Mix continuous, binary, count, survival
- **SEM modes** (use `output_sem_mode=` and `output_structure=`):
  - `none`: Independent outcomes
  - `lavaan`: Specify causal structure
  - `learned`: Learn structure from data
- **Benefits**: Shared learning, efficiency, relationship modeling

Next: See [vignette 04](04_model_selection.md) for comparing models and selecting the best architecture.
