# Mixed Effects Modeling in Depth

This vignette provides a comprehensive guide to mixed effects modeling in TabMixNN, covering random intercepts, random slopes, and complex hierarchical structures.

## Table of Contents

1. [What are Mixed Effects Models?](#what-are-mixed-effects-models)
2. [Random Intercepts](#random-intercepts)
3. [Random Slopes](#random-slopes)
4. [Multiple Grouping Factors](#multiple-grouping-factors)
5. [Nested vs Crossed Effects](#nested-vs-crossed-effects)
6. [Interpreting Random Effects](#interpreting-random-effects)
7. [When to Use Random Effects](#when-to-use-random-effects)

---

## What are Mixed Effects Models?

Mixed effects models (also called hierarchical, multilevel, or random effects models) handle data with **grouping structure**.

### The Problem

Standard regression assumes all observations are independent. But real data often has structure:

- **Patients within hospitals**: Patients from the same hospital are more similar
- **Students within schools**: Students in the same school share characteristics
- **Repeated measures**: Multiple measurements from the same subject are correlated

### The Solution

Mixed effects models include:
- **Fixed effects**: Population-level effects (same for everyone)
- **Random effects**: Group-level deviations (different for each group)

---

## Random Intercepts

Random intercepts allow each group to have its own baseline level.

### Conceptual Model

```
y_ij = β₀ + β₁·x_ij + u_j + ε_ij

Where:
- β₀ = fixed intercept (population average)
- β₁ = fixed slope (population effect)
- u_j = random intercept for group j (group deviation)
- ε_ij = residual error
```

### Example: Multi-Center Clinical Trial

```python
import numpy as np
import pandas as pd
from tabmixnn import TabMixNNRegressor

# Simulate data: patients nested in centers
np.random.seed(42)
n_centers = 8
n_per_center = 50

# Each center has different baseline quality
center_baselines = np.random.randn(n_centers) * 5.0

df = pd.DataFrame({
    'patient_age': np.random.randint(30, 70, n_centers * n_per_center),
    'treatment_dose': np.random.uniform(0, 10, n_centers * n_per_center),
    'center': np.repeat([f'C{i}' for i in range(n_centers)], n_per_center)
})

# Outcome: baseline varies by center, but treatment effect is constant
center_ids = np.array([int(c[1:]) for c in df['center']])
df['outcome'] = (
    50 +                                      # Population baseline
    0.3 * (df['patient_age'] - 50) +        # Age effect
    2.0 * df['treatment_dose'] +            # Treatment effect (same for all)
    center_baselines[center_ids] +          # CENTER-SPECIFIC baseline
    np.random.randn(len(df)) * 3.0          # Noise
)

# Fit random intercept model
model = TabMixNNRegressor(
    formula="outcome ~ patient_age + treatment_dose + (1|center)",
    epochs=30,
    verbose=False
)
model.fit(df)

print("Random Intercept Model:")
model.summary()
```

### Interpretation

With `(1|center)`:
- Each center gets its own baseline level
- Treatment and age effects are assumed constant across centers
- Useful when groups differ in baseline but not in effects

---

## Random Slopes

Random slopes allow the effect of a predictor to vary across groups.

### Conceptual Model

```
y_ij = β₀ + β₁·x_ij + u₀j + u₁j·x_ij + ε_ij

Where:
- u₀j = random intercept for group j
- u₁j = random slope for group j (effect of x varies by group)
```

### Example: Treatment Effect Varies by Center

```python
# Simulate data where treatment effect differs by center
np.random.seed(42)

# Each center has different treatment effectiveness
center_intercepts = np.random.randn(n_centers) * 5.0
center_slopes = 2.0 + np.random.randn(n_centers) * 1.0  # Mean=2, varies

df2 = pd.DataFrame({
    'patient_age': np.random.randint(30, 70, n_centers * n_per_center),
    'treatment_dose': np.random.uniform(0, 10, n_centers * n_per_center),
    'center': np.repeat([f'C{i}' for i in range(n_centers)], n_per_center)
})

center_ids = np.array([int(c[1:]) for c in df2['center']])
df2['outcome'] = (
    50 +
    0.3 * (df2['patient_age'] - 50) +
    center_intercepts[center_ids] +                          # Random intercept
    center_slopes[center_ids] * df2['treatment_dose'] +     # Random slope!
    np.random.randn(len(df2)) * 3.0
)

# Model 1: Random intercept only (wrong model)
model_ri = TabMixNNRegressor(
    formula="outcome ~ patient_age + treatment_dose + (1|center)",
    epochs=30,
    verbose=False
)
model_ri.fit(df2)

# Model 2: Random intercept AND slope (correct model)
model_rs = TabMixNNRegressor(
    formula="outcome ~ patient_age + treatment_dose + (1 + treatment_dose|center)",
    epochs=30,
    verbose=False
)
model_rs.fit(df2)

# Compare predictions
pred_ri = model_ri.predict(df2)
pred_rs = model_rs.predict(df2)

r2_ri = 1 - np.mean((pred_ri - df2['outcome'])**2) / np.var(df2['outcome'])
r2_rs = 1 - np.mean((pred_rs - df2['outcome'])**2) / np.var(df2['outcome'])

print(f"Random Intercept only:    R² = {r2_ri:.4f}")
print(f"Random Intercept + Slope: R² = {r2_rs:.4f}")
print(f"Improvement: {r2_rs - r2_ri:.4f}")
```

### Formula Syntax for Random Slopes

```python
# Different ways to specify random slopes:

# 1. Random slope for x only (no random intercept)
formula = "y ~ x + (x|group)"
formula = "y ~ x + (0 + x|group)"  # Equivalent

# 2. Random intercept + random slope
formula = "y ~ x + (1 + x|group)"

# 3. Multiple random slopes
formula = "y ~ x1 + x2 + (1 + x1 + x2|group)"
```

---

## Multiple Grouping Factors

Real data often has multiple sources of grouping.

### Example: Patients within Physicians within Hospitals

```python
np.random.seed(42)

n_hospitals = 4
n_physicians_per_hospital = 5
n_patients_per_physician = 20
n_total = n_hospitals * n_physicians_per_hospital * n_patients_per_physician

# Create nested structure
hospitals = []
physicians = []
hospital_effects = np.random.randn(n_hospitals) * 3.0
physician_effects = np.random.randn(n_hospitals * n_physicians_per_hospital) * 2.0

for h in range(n_hospitals):
    for p in range(n_physicians_per_hospital):
        physician_id = h * n_physicians_per_hospital + p
        for _ in range(n_patients_per_physician):
            hospitals.append(f'H{h}')
            physicians.append(f'P{physician_id}')

df_nested = pd.DataFrame({
    'age': np.random.randint(30, 80, n_total),
    'severity': np.random.randint(1, 5, n_total),
    'hospital': hospitals,
    'physician': physicians
})

# Outcome affected by both hospital and physician
hospital_ids = np.array([int(h[1:]) for h in hospitals])
physician_ids = np.array([int(p[1:]) for p in physicians])

df_nested['outcome'] = (
    50 +
    0.2 * df_nested['age'] +
    5.0 * df_nested['severity'] +
    hospital_effects[hospital_ids] +    # Hospital effect
    physician_effects[physician_ids] +  # Physician effect
    np.random.randn(n_total) * 2.0
)

# Model with both grouping factors
model_multi = TabMixNNRegressor(
    formula="outcome ~ age + severity + (1|hospital) + (1|physician)",
    epochs=30,
    verbose=True
)
model_multi.fit(df_nested)

print("\nModel with multiple grouping factors:")
model_multi.summary()
```

---

## Nested vs Crossed Effects

### Nested Effects

Groups at one level are nested within groups at another level.

**Example**: Patients within physicians within hospitals
- Each physician works at only one hospital
- Patients are nested within physicians

```python
# Nested structure
formula = "outcome ~ x + (1|hospital) + (1|hospital:physician)"
# Or simply (if physician IDs are unique)
formula = "outcome ~ x + (1|hospital) + (1|physician)"
```

### Crossed Effects

Groups at different levels cross each other.

**Example**: Students taking courses from different teachers
- Each student takes multiple courses
- Each teacher teaches multiple students
- Students and teachers are crossed, not nested

```python
# Crossed structure
formula = "grade ~ ability + (1|student) + (1|teacher)"
```

---

## Interpreting Random Effects

### Variance Components

Random effects add variance at the group level:

```python
# After fitting a model
model.summary()

# Shows:
# - Fixed effects: population-level estimates
# - Random effects: variance across groups
```

### Extracting Random Effect Estimates

```python
# Get parameter estimates including random effects
params = model.extract_parameters()
print(params)
```

### Intraclass Correlation (ICC)

ICC measures how much variance is due to grouping:

```
ICC = σ²_between / (σ²_between + σ²_within)

Where:
- σ²_between = variance between groups
- σ²_within = variance within groups
```

High ICC (> 0.1) indicates strong grouping effects and justifies mixed models.

---

## When to Use Random Effects

### Use Random Intercepts When:

1. **Groups have different baselines**
   - Hospitals have different quality levels
   - Schools have different average performance

2. **You want to generalize to new groups**
   - Model should work for hospitals not in training data

3. **Sample sizes vary across groups**
   - Some centers have 10 patients, others have 100

### Use Random Slopes When:

1. **Effects vary across groups**
   - Treatment works better in some hospitals
   - Learning rates differ across schools

2. **You want group-specific predictions**
   - Predict treatment effect for a specific center

3. **Interaction with group membership**
   - Effect of age differs by gender
   - Effect of dose differs by genetic variant

### Don't Use Random Effects When:

1. **Few groups (< 5)**
   - Not enough groups to estimate variance
   - Use fixed effects instead

2. **Groups are the focus**
   - You care about specific groups, not generalizing
   - Use group as fixed effect

3. **No repeated structure**
   - Each observation is truly independent

---

## Best Practices

### 1. Start Simple

```python
# Start with random intercept
model1 = TabMixNNRegressor(formula="y ~ x + (1|group)")

# Add random slopes if needed
model2 = TabMixNNRegressor(formula="y ~ x + (1 + x|group)")

# Compare
```

### 2. Check for Overfitting

Random slopes can overfit with:
- Many slopes per group
- Few observations per group

Solution: Use regularization or simpler random effects.

### 3. Scale Predictors

Random slopes work better with scaled predictors:

```python
df['x_scaled'] = (df['x'] - df['x'].mean()) / df['x'].std()
```

### 4. Visualize Group Differences

```python
import matplotlib.pyplot as plt

# Plot predictions by group
for group in df['center'].unique():
    group_data = df[df['center'] == group]
    plt.scatter(group_data['x'], group_data['y'], label=group, alpha=0.5)

plt.legend()
plt.show()
```

---

## Summary

Key takeaways:

- **Random intercepts**: Groups have different baselines → `(1|group)`
- **Random slopes**: Effects vary by group → `(1 + x|group)`
- **Multiple grouping**: Multiple sources of clustering → `(1|group1) + (1|group2)`
- **Nested vs Crossed**: Understand your data structure
- **ICC**: Quantifies grouping strength
- **When to use**: More than 5 groups, repeated structure, want to generalize

Next: See [vignette 03](03_multitask_learning.md) for modeling multiple outcomes simultaneously.
