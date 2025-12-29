"""
Example 4: Multiple Outcomes - Machines Dataset

This example demonstrates multi-task learning by creating related outcomes
from the Machines dataset.

Dataset: Production scores on different machines by workers (nlme)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabmixnn import TabMixNN
from tabmixnn.parsing.formula import OutcomeSpec
from tabmixnn.core.families import Gaussian, Binomial
from data_utils import load_dataset, get_dataset_info

# Load the Machines dataset
print("="*60)
print("Example 4: Multiple Outcomes (Multi-task Learning)")
print("Dataset: Machines - Worker Productivity")
print("="*60)

df = load_dataset("machines")

print(f"\nFirst few rows:")
print(df.head(15))

print(f"\nData shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")

# Machines dataset:
# - Worker: Worker identifier (6 workers)
# - Machine: Machine type (A, B, C)
# - score: Productivity score

print(f"\nSummary by Machine:")
print(df.groupby('Machine')['score'].describe())

print(f"\nNumber of workers: {df['Worker'].nunique()}")
print(f"Number of machines: {df['Machine'].nunique()}")

# Create a second outcome: binary indicator of high performance
score_median = df['score'].median()
df['high_performance'] = (df['score'] > score_median).astype(int)

print(f"\nPerformance statistics:")
print(f"  Median score: {score_median:.2f}")
print(f"  High performance rate: {df['high_performance'].mean():.1%}")

# Standardize continuous outcome
score_mean = df['score'].mean()
score_std = df['score'].std()
df['score_scaled'] = (df['score'] - score_mean) / score_std

print(f"  Mean score: {score_mean:.2f}")
print(f"  Std score: {score_std:.2f}")

# Define two related outcomes:
# 1. Continuous productivity score (regression)
# 2. Binary high/low performance indicator (classification)

print("\n" + "="*60)
print("Fitting Multi-task Model")
print("="*60)

outcomes = [
    OutcomeSpec(
        name="productivity",
        y="score_scaled",
        family=Gaussian()
    ),
    OutcomeSpec(
        name="high_performer",
        y="high_performance",
        family=Binomial()
    )
]

# Model with random intercepts for Worker
# Predicting from Machine type
model = TabMixNN(
    formula="~ Machine + (1|Worker)",
    outcomes=outcomes,
    architecture="gsem",
    hidden_dims=[16, 8],
    epochs=50,
    batch_size=16,
    learning_rate=0.01,
    verbose=True
)

print("\nTraining multi-task model...")
print("(Sharing information between continuous and binary outcomes)")
model.fit(df)

# Get predictions for both outcomes
predictions = model.predict(df)

print(f"\n" + "="*60)
print(f"Predictions")
print(f"="*60)
print(f"Prediction keys: {list(predictions.keys())}")

# Unscale continuous predictions
pred_score_scaled = predictions['productivity']
pred_score = pred_score_scaled * score_std + score_mean
pred_binary = predictions['high_performer']

# Evaluate continuous outcome
mse = np.mean((pred_score - df['score'])**2)
r2 = 1 - mse / np.var(df['score'])
rmse = np.sqrt(mse)

print(f"\n" + "="*60)
print("Model Performance")
print("="*60)

print(f"\nContinuous Outcome (Productivity Score):")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²:   {r2:.4f}")

# Evaluate binary outcome
from sklearn.metrics import accuracy_score, roc_auc_score

pred_binary_class = (pred_binary > 0.5).astype(int)
accuracy = accuracy_score(df['high_performance'], pred_binary_class)
auc = roc_auc_score(df['high_performance'], pred_binary)

print(f"\nBinary Outcome (High/Low Performance):")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  AUC-ROC:  {auc:.4f}")

# Model summary
print(f"\n" + "="*60)
print("Model Summary")
print("="*60)
model.summary()

# Performance by machine
print(f"\n" + "="*60)
print("Performance by Machine Type")
print("="*60)

machine_stats = []
for machine in sorted(df['Machine'].unique()):
    machine_data = df[df['Machine'] == machine]
    machine_preds = pred_score[df['Machine'] == machine]

    machine_stats.append({
        'Machine': machine,
        'N_obs': len(machine_data),
        'Actual_mean': f"{machine_data['score'].mean():.2f}",
        'Predicted_mean': f"{machine_preds.mean():.2f}",
        'RMSE': f"{np.sqrt(np.mean((machine_preds - machine_data['score'])**2)):.2f}"
    })

stats_df = pd.DataFrame(machine_stats)
print(stats_df.to_string(index=False))

# Visualize predictions for both outcomes
print(f"\n" + "="*60)
print("Visualizing Multi-task Predictions")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Continuous outcome
ax1 = axes[0]
for machine in sorted(df['Machine'].unique()):
    mask = df['Machine'] == machine
    ax1.scatter(df.loc[mask, 'score'], pred_score[mask],
               label=f'Machine {machine}', alpha=0.7, s=50)

ax1.plot([df['score'].min(), df['score'].max()],
         [df['score'].min(), df['score'].max()],
         'r--', label='Perfect prediction', linewidth=2)
ax1.set_xlabel('Actual Productivity Score')
ax1.set_ylabel('Predicted Productivity Score')
ax1.set_title('Continuous Outcome (Regression)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Binary outcome
ax2 = axes[1]
for perf_class in [0, 1]:
    mask = df['high_performance'] == perf_class
    label = 'High' if perf_class == 1 else 'Low'
    ax2.scatter(np.arange(mask.sum()), pred_binary[mask],
               label=f'Actual: {label}', alpha=0.7, s=50)

ax2.axhline(y=0.5, color='r', linestyle='--', label='Decision threshold', linewidth=2)
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('Predicted Probability')
ax2.set_title('Binary Outcome (Classification)')
ax2.set_ylim(-0.1, 1.1)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/04_multitask_predictions.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to: output/04_multitask_predictions.png")

# Worker-level analysis
print(f"\n" + "="*60)
print("Worker-Level Statistics (Random Effects)")
print("="*60)

worker_stats = []
for worker in sorted(df['Worker'].unique()):
    worker_data = df[df['Worker'] == worker]
    worker_preds = pred_score[df['Worker'] == worker]

    worker_stats.append({
        'Worker': worker,
        'N_obs': len(worker_data),
        'Avg_score': f"{worker_data['score'].mean():.2f}",
        'Predicted_avg': f"{worker_preds.mean():.2f}",
        'RMSE': f"{np.sqrt(np.mean((worker_preds - worker_data['score'])**2)):.2f}"
    })

worker_df = pd.DataFrame(worker_stats)
print(worker_df.to_string(index=False))

print(f"\n" + "="*60)
print("Key Takeaways")
print("="*60)
print("""
1. Multi-task learning models multiple related outcomes simultaneously,
   sharing information through the model architecture.

2. The two outcomes (continuous score and binary performance) are related:
   - Both depend on machine type and worker skill
   - Model learns shared representations benefiting both tasks

3. Random effects for Worker capture individual skill differences,
   improving predictions for both outcomes.

4. This demonstrates how TabMixNN can handle heterogeneous outcomes
   (Gaussian + Binomial) in a single unified model.

5. Machines dataset shows clear differences between machine types,
   which the model captures in both continuous and binary predictions.
""")
