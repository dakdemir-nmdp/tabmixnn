"""
Example 1: Linear Mixed Model - Sleepstudy Dataset

This example demonstrates fitting a linear mixed model with random intercepts
and random slopes using the famous sleepstudy dataset from lme4.

Dataset: Reaction times (ms) vs days of sleep deprivation for 18 subjects
Reference: Belenky et al. (2003), J Sleep Res
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabmixnn import TabMixNNRegressor
from data_utils import load_dataset, get_dataset_info

# Load the sleepstudy dataset
print("="*60)
print("Example 1: Linear Mixed Model")
print("Dataset: Sleepstudy (lme4)")
print("="*60)

df = load_dataset("sleepstudy")

# Display dataset info
info = get_dataset_info("sleepstudy")
print(f"\nDataset: {info['description']}")
print(f"Observations: {info['n_obs']}")
print(f"Subjects: {info['n_groups']}")
print(f"\nFirst few rows:")
print(df.head(10))

print(f"\nData shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nSummary statistics:")
print(df.describe())

# Prepare data
# Days is the number of days of sleep deprivation (0-9)
# Reaction is the reaction time in milliseconds
# Subject is the subject ID

# Standardize reaction time for neural network stability
reaction_mean = df['Reaction'].mean()
reaction_std = df['Reaction'].std()
df['Reaction_scaled'] = (df['Reaction'] - reaction_mean) / reaction_std

print(f"\nReaction time: mean={reaction_mean:.2f} ms, std={reaction_std:.2f} ms")

# Fit Linear Mixed Model with random slopes and intercepts
# Model: Reaction ~ Days + (1 + Days|Subject)
# This allows each subject to have their own baseline and slope

print("\n" + "="*60)
print("Fitting Model: Reaction ~ Days + (1 + Days|Subject)")
print("="*60)

model = TabMixNNRegressor(
    formula="Reaction_scaled ~ Days + (1 + Days|Subject)",
    epochs=50,
    batch_size=32,
    learning_rate=0.01,
    verbose=True
)

print("\nTraining model...")
model.fit(df)

# Make predictions (scaled)
predictions_scaled = model.predict(df)

# Unscale predictions
predictions = predictions_scaled * reaction_std + reaction_mean

# Evaluate
mse = np.mean((predictions - df['Reaction'])**2)
mae = np.mean(np.abs(predictions - df['Reaction']))
r2 = 1 - mse / np.var(df['Reaction'])
rmse = np.sqrt(mse)

print(f"\n" + "="*60)
print("Model Performance")
print("="*60)
print(f"RMSE: {rmse:.4f} ms")
print(f"MAE:  {mae:.4f} ms")
print(f"R²:   {r2:.4f}")

# Display model summary
print(f"\n" + "="*60)
print("Model Summary")
print("="*60)
model.summary()

# Extract parameters
print(f"\n" + "="*60)
print("Parameter Estimates")
print("="*60)
params = model.extract_parameters()
if params:
    for key, value in params.items():
        print(f"\n{key}:")
        print(value)

# Visualize results for a sample of subjects
print(f"\n" + "="*60)
print("Visualizing Predictions for Sample Subjects")
print("="*60)

subjects = df['Subject'].unique()
sample_subjects = np.random.choice(subjects, min(4, len(subjects)), replace=False)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, subject in enumerate(sample_subjects):
    subject_data = df[df['Subject'] == subject].copy()
    subject_preds = predictions[df['Subject'] == subject]

    ax = axes[idx]
    ax.scatter(subject_data['Days'], subject_data['Reaction'],
              label='Observed', alpha=0.6, s=50)
    ax.plot(subject_data['Days'], subject_preds,
           'r-', label='Predicted', linewidth=2)
    ax.set_xlabel('Days of Sleep Deprivation')
    ax.set_ylabel('Reaction Time (ms)')
    ax.set_title(f'Subject {subject}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/01_sleepstudy_predictions.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to: output/01_sleepstudy_predictions.png")

# Summary statistics by subject
print(f"\n" + "="*60)
print("Per-Subject Statistics")
print("="*60)

subject_stats = []
for subject in subjects[:5]:  # Show first 5 subjects
    subject_data = df[df['Subject'] == subject]
    subject_preds = predictions[df['Subject'] == subject]
    subject_rmse = np.sqrt(np.mean((subject_preds - subject_data['Reaction'])**2))
    subject_stats.append({
        'Subject': subject,
        'N_obs': len(subject_data),
        'Mean_Reaction': subject_data['Reaction'].mean(),
        'RMSE': subject_rmse
    })

stats_df = pd.DataFrame(subject_stats)
print(stats_df.to_string(index=False))

print(f"\n" + "="*60)
print("Key Takeaways")
print("="*60)
print("""
1. Random effects model captures individual differences in:
   - Baseline reaction time (random intercept)
   - Rate of deterioration (random slope for Days)

2. Each subject has their own trajectory, while sharing
   information through the population-level effects.

3. This is the canonical example from lme4's sleepstudy dataset,
   demonstrating TabMixNN's ability to match traditional mixed models.
""")
