# Vignette 04: Survival Analysis with TabMixNN

Complete guide to time-to-event modeling using TabMixNN's survival families.

## Table of Contents
1. [Introduction](#introduction)
2. [Survival Analysis Basics](#survival-analysis-basics)
3. [DiscreteTimeSurvival](#discretetimesurvival)
4. [Competing Risks](#competing-risks)
5. [ContinuousSurvival](#continuoussurvival)
6. [Prediction Modes](#prediction-modes)
7. [Multitask Survival Models](#multitask-survival-models)
8. [Best Practices](#best-practices)

---

## Introduction

Survival analysis models the time until an event occurs, accounting for **censoring** - when the event hasn't been observed by the end of the study period. TabMixNN supports three survival families:

1. **DiscreteTimeSurvival**: Hazard-based models with discrete time intervals
2. **DiscreteTimeCompetingRisks**: Multiple mutually exclusive event types
3. **ContinuousSurvival**: Parametric models (Weibull/exponential)

This vignette focuses on discrete-time models, which are most commonly used and flexible.

---

## Survival Analysis Basics

### Key Concepts

**Time (T)**: The time when an event occurs (or study ends)

**Event indicator (δ)**:
- `δ = 1`: Event observed (death, failure, recurrence)
- `δ = 0`: Censored (lost to follow-up, study ended)

**Survival function S(t)**: Probability of surviving past time t
```
S(t) = P(T > t)
```

**Hazard function h(t)**: Instantaneous risk of event at time t
```
h(t) = lim[P(t ≤ T < t+Δt | T ≥ t) / Δt]
```

**Relationship**:
```
S(t) = exp(-∫h(u)du) = ∏(1 - h(t_i))
```

### Why Discrete-Time?

Discrete-time models:
- ✅ Handle tied event times naturally
- ✅ No proportional hazards assumption
- ✅ Easy to interpret (risk per interval)
- ✅ Extend easily to competing risks
- ✅ Compatible with neural networks

---

## DiscreteTimeSurvival

### Basic Setup

```python
import numpy as np
import pandas as pd
from tabmixnn import TabMixNN, OutcomeSpec
from tabmixnn.families import DiscreteTimeSurvival

# Define time bins (interval boundaries)
time_bins = np.array([0, 6, 12, 24, 36, 48, 60])  # Months

# Create outcome specification
outcomes = [
    OutcomeSpec(
        name="survival",
        time="time_to_event",    # Column with observed time
        event="event_occurred",  # Column with 0/1 indicator
        family=DiscreteTimeSurvival(time_bins=time_bins)
    )
]

# Build model
model = TabMixNN(
    formula="~ age + treatment + stage + (1|hospital)",
    outcomes=outcomes,
    hidden_dims=[64, 32],
    epochs=200,
    lr=0.01,
    early_stopping=True,
    patience=15
)

# Train
model.fit(train_df, val_df=test_df)
```

### How It Works

The model estimates **hazard probabilities** for each time bin:
```
h_j = P(Event in bin j | Survived to bin j)
```

Survival probability at time bin k:
```
S_k = ∏(1 - h_j) for j=1 to k
```

This guarantees monotonically decreasing survival curves.

### Choosing Time Bins

**Option 1: Equal intervals**
```python
time_bins = np.arange(0, max_time + 1, interval_size)
# Example: [0, 10, 20, 30, 40] for 10-unit intervals
```

**Option 2: Quantile-based** (recommended)
```python
time_bins = np.percentile(df['time'], np.linspace(0, 100, num_bins))
time_bins = np.unique(time_bins)  # Remove duplicates
```

**Guidelines**:
- Start with 10-20 bins
- More bins = more flexible hazard, but needs more data
- Ensure each bin has sufficient events (~10+ per bin)
- Bins don't need equal width

### Example: Patient Survival

```python
# Data format
patient_df = pd.DataFrame({
    'patient_id': [1, 2, 3, 4, 5],
    'time': [12, 8, 24, 18, 30],      # Time to death or censoring
    'event': [1, 1, 0, 1, 0],         # 1=death, 0=censored
    'age': [65, 72, 58, 63, 70],
    'treatment': ['A', 'B', 'A', 'A', 'B'],
    'hospital': [1, 1, 2, 2, 1]
})

# Scale continuous variables
patient_df['age_scaled'] = (patient_df['age'] - patient_df['age'].mean()) / patient_df['age'].std()

# Time bins based on data
time_bins = np.percentile(patient_df['time'], [0, 25, 50, 75, 100])
print(f"Time bins: {time_bins}")

# Model
outcomes = [
    OutcomeSpec(
        name="survival",
        time="time",
        event="event",
        family=DiscreteTimeSurvival(time_bins=time_bins)
    )
]

model = TabMixNN(
    formula="~ age_scaled + treatment + (1|hospital)",
    outcomes=outcomes,
    epochs=100
)

model.fit(patient_df)
```

---

## Competing Risks

When individuals can experience one of several **mutually exclusive** events.

### Example: Transplant Outcomes

Patients can experience:
- Event type 0: Censored (study ended, lost to follow-up)
- Event type 1: Relapse
- Event type 2: Transplant-related mortality (TRM)

### Setup

```python
from tabmixnn.families import DiscreteTimeCompetingRisks

# Data format
transplant_df = pd.DataFrame({
    'time': [12, 8, 24, 18],      # Time to event or censoring
    'cause': [1, 2, 0, 1],         # 0=censored, 1=relapse, 2=TRM
    'age': [45, 52, 38, 48],
    'risk_group': ['low', 'high', 'low', 'high']
})

# Time bins
time_bins = np.array([0, 6, 12, 24, 36])

# Outcome specification
outcomes = [
    OutcomeSpec(
        name="competing_risks",
        time="time",
        cause="cause",          # Use 'cause' not 'event'!
        family=DiscreteTimeCompetingRisks(
            time_bins=time_bins,
            num_causes=2        # Number of event types (excluding censoring)
        )
    )
]

model = TabMixNN(
    formula="~ age + risk_group",
    outcomes=outcomes,
    epochs=150
)

model.fit(transplant_df)
```

### Key Differences from Standard Survival

1. **Use `cause` parameter** (not `event`) in OutcomeSpec
2. **Cause column encoding**:
   - 0 = Censored
   - 1, 2, 3, ... = Specific event types
3. **Specify `num_causes`** in family (excludes censoring)

### Cumulative Incidence Functions (CIF)

For each cause, the CIF is the probability of experiencing that specific event by time t:

```
CIF_k(t) = P(Event type k by time t)
```

Unlike survival curves, CIFs:
- Can increase (accumulating events)
- Sum across causes ≤ 1 - S(t)
- Account for competing events preventing the event of interest

---

## ContinuousSurvival

Parametric survival model (Weibull/exponential distribution).

```python
from tabmixnn.families import ContinuousSurvival

outcomes = [
    OutcomeSpec(
        name="survival",
        time="time",
        event="event",
        family=ContinuousSurvival()  # No time bins needed
    )
]

model = TabMixNN(
    formula="~ age + treatment",
    outcomes=outcomes,
    epochs=100
)
```

**When to use**:
- Prefer parametric form (Weibull)
- Smoother survival curves needed
- Smaller datasets (fewer parameters)

**When to avoid**:
- Unknown hazard shape → use DiscreteTimeSurvival
- Need maximum flexibility → use DiscreteTimeSurvival

---

## Prediction Modes

TabMixNN supports multiple prediction modes for survival models.

### 1. Survival Curves (`mode="survival"`)

Returns survival probabilities S(t) at each time bin.

```python
# Predict survival curves
surv_probs = model.predict(test_df, outcome="survival", mode="survival")

# Shape: (n_samples, n_time_bins)
# surv_probs[i, j] = P(T > t_j) for sample i
```

**Interpretation**:
- `surv_probs[0, 0]` = Probability patient 0 survives past time bin 0 (usually ~1.0)
- `surv_probs[0, -1]` = Probability patient 0 survives to final time bin

**Visualization**:
```python
import matplotlib.pyplot as plt

# Plot survival curve for first patient
plt.plot(time_bins[1:], surv_probs[0], label='Patient 0')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.title('Predicted Survival Curve')
plt.legend()
plt.show()
```

### 2. Hazard Probabilities (`mode="hazard"`)

Returns hazard probabilities h(t) for each time bin.

```python
# Predict hazard probabilities
hazards = model.predict(test_df, outcome="survival", mode="hazard")

# Shape: (n_samples, n_time_bins - 1)
# hazards[i, j] = P(Event in bin j | Survived to bin j) for sample i
```

**Interpretation**:
- Higher hazard = higher risk of event in that interval
- Hazards can increase/decrease over time (non-proportional)

### 3. Cumulative Incidence Functions (`mode="cif"`)

For competing risks only. Returns cause-specific CIFs.

```python
# Predict CIFs for competing risks
cif = model.predict(test_df, outcome="competing_risks", mode="cif")

# Shape: (n_samples, n_time_bins, n_causes)
# cif[i, j, k] = P(Event type k by time j) for sample i
```

**Example**:
```python
# CIF for relapse (cause 1) for patient 0
relapse_cif = cif[0, :, 0]  # Probability of relapse by each time point

# CIF for TRM (cause 2) for patient 0
trm_cif = cif[0, :, 1]      # Probability of TRM by each time point

# Plot both
plt.plot(time_bins, relapse_cif, label='Relapse')
plt.plot(time_bins, trm_cif, label='TRM')
plt.xlabel('Time')
plt.ylabel('Cumulative Incidence')
plt.legend()
plt.show()
```

### 4. Risk Scores

For model evaluation (C-index calculation).

```python
# Convert survival probabilities to risk scores
surv_probs = model.predict(test_df, outcome="survival", mode="survival")

# Higher risk = lower survival at final time point
risk_scores = -np.log(np.clip(surv_probs[:, -1], 1e-6, 1.0))

# Calculate C-index
from sklearn.metrics import roc_auc_score
# ... (see examples for full C-index implementation)
```

---

## Multitask Survival Models

Combine survival with other outcomes for joint modeling.

### Example: Survival + Biomarkers

```python
# Model both survival and a continuous biomarker
outcomes = [
    OutcomeSpec(
        name="survival",
        time="time",
        event="event",
        family=DiscreteTimeSurvival(time_bins=time_bins)
    ),
    OutcomeSpec(
        name="biomarker",
        y="biomarker_level",
        family=Gaussian()
    )
]

model = TabMixNN(
    formula="~ age + treatment + (1|hospital)",
    outcomes=outcomes,
    hidden_dims=[64, 32],
    epochs=200
)

model.fit(train_df, val_df=test_df)

# Predict both
surv_preds = model.predict(test_df, outcome="survival", mode="survival")
bio_preds = model.predict(test_df, outcome="biomarker")
```

### Benefits of Multitask

1. **Shared representations**: Information from biomarkers improves survival prediction
2. **Regularization**: Multiple tasks prevent overfitting
3. **Efficiency**: Single model for multiple endpoints

### Example: Survival + Response

```python
outcomes = [
    OutcomeSpec(
        name="survival",
        time="time",
        event="event",
        family=DiscreteTimeSurvival(time_bins=bins)
    ),
    OutcomeSpec(
        name="treatment_response",
        y="response",
        family=Binomial()  # Binary response
    )
]
```

---

## Best Practices

### 1. Data Preparation

**Always scale continuous predictors**:
```python
df['age_scaled'] = (df['age'] - df['age'].mean()) / df['age'].std()
df['weight_scaled'] = (df['weight'] - df['weight'].mean()) / df['weight'].std()
```

**Ensure sufficient events per bin**:
```python
# Check event distribution
for i in range(len(time_bins)-1):
    mask = (df['time'] >= time_bins[i]) & (df['time'] < time_bins[i+1])
    events = df.loc[mask, 'event'].sum()
    print(f"Bin {i} ({time_bins[i]}-{time_bins[i+1]}): {events} events")
```

### 2. Model Configuration

**Survival-specific settings**:
```python
model = TabMixNN(
    formula=formula,
    outcomes=survival_outcomes,
    epochs=200,           # Survival needs more epochs than regression
    lr=0.01,             # Standard learning rate
    batch_size=64,       # Smaller batches for stability
    early_stopping=True,
    patience=15          # More patience for survival
)
```

**For competing risks**, increase epochs further:
```python
epochs=250,  # Competing risks are more complex
patience=20
```

### 3. Model Evaluation

**C-index** (concordance index) - primary metric:
- C-index > 0.5: Better than random
- C-index > 0.6: Weak discrimination
- C-index > 0.7: Good discrimination
- C-index > 0.8: Strong discrimination

**Visual inspection**:
- Plot survival curves for high-risk vs low-risk groups
- Check monotonicity (should always decrease)
- Compare with Kaplan-Meier estimates

### 4. Common Pitfalls

❌ **Using `event` instead of `cause` for competing risks**
```python
# WRONG
OutcomeSpec(time="time", event="cause", family=DiscreteTimeCompetingRisks(...))

# CORRECT
OutcomeSpec(time="time", cause="cause", family=DiscreteTimeCompetingRisks(...))
```

❌ **Time bins don't cover data range**
```python
# WRONG: max time is 60 but bins only go to 50
time_bins = np.array([0, 10, 20, 30, 40, 50])

# CORRECT: bins cover full range
max_time = df['time'].max()
time_bins = np.linspace(0, max_time * 1.1, 15)
```

❌ **Forgetting to scale predictors**
```python
# WRONG: Raw age values (0-100)
formula = "~ age + treatment"

# CORRECT: Scaled age
df['age_scaled'] = (df['age'] - df['age'].mean()) / df['age'].std()
formula = "~ age_scaled + treatment"
```

❌ **Too few epochs**
```python
# WRONG: Survival needs more epochs
epochs=50

# CORRECT: Allow full convergence
epochs=200
```

### 5. Interpretation

**Survival curves**:
- Compare curves between groups (treatment vs control)
- Identify time points where groups diverge
- Quantify survival at clinically relevant times (1-year, 5-year)

**Hazard probabilities**:
- Identify periods of highest risk
- Check for proportional hazards (parallel log-hazards)
- Detect time-varying effects

**Competing risks CIFs**:
- Quantify absolute risk for each event type
- Compare relative importance of competing events
- Identify groups at high risk for specific events

---

## Complete Example

See [examples/05_survival_analysis.py](../examples/05_survival_analysis.py) for a complete working example with the Rossi recidivism dataset.

See [examples/06_competing_risks.py](../examples/06_competing_risks.py) for competing risks with the Colon cancer dataset.

---

## Next Steps

- **Practice**: Run the survival examples with different datasets
- **Experiment**: Try different time bin configurations
- **Extend**: Combine survival with other outcome types
- **Evaluate**: Compare with Cox proportional hazards models

For more advanced topics, see the main [API documentation](../API.md).
