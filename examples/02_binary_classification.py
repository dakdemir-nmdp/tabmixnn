"""
Example 2: Binary Classification with Mixed Effects (GLMM)

This example demonstrates fitting a generalized linear mixed model for
binary classification using the CBPP (Contagious Bovine Pleuropneumonia) dataset.

Dataset: Cattle disease incidence across herds (lme4)
Reference: Categorical Data Analysis, Agresti (2002)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabmixnn import TabMixNN
from tabmixnn.parsing.formula import OutcomeSpec
from tabmixnn.core.families import Binomial
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from data_utils import load_dataset, get_dataset_info

# Load the CBPP dataset
print("="*60)
print("Example 2: Binary Classification (GLMM)")
print("Dataset: CBPP - Cattle Disease Incidence (lme4)")
print("="*60)

df = load_dataset("cbpp")

# Display dataset info
info = get_dataset_info("cbpp")
print(f"\nDataset: {info['description']}")
print(f"\nFirst few rows:")
print(df.head(10))

print(f"\nData shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nSummary:")
print(df.describe())

# CBPP dataset structure:
# - incidence: number of new cases in the period
# - size: number of animals at risk in the period
# - period: time period (1-4)
# - herd: herd ID

# We need to convert to binary outcomes (sick/not sick) for individual animals
# Expand the aggregated data to individual level
print("\n" + "="*60)
print("Preparing Data")
print("="*60)

expanded_data = []
for idx, row in df.iterrows():
    herd = row['herd']
    period = row['period']

    # Create individual records
    # incidence animals got sick (outcome = 1)
    for _ in range(int(row['incidence'])):
        expanded_data.append({
            'herd': herd,
            'period': period,
            'outcome': 1
        })

    # (size - incidence) animals stayed healthy (outcome = 0)
    healthy = int(row['size'] - row['incidence'])
    for _ in range(healthy):
        expanded_data.append({
            'herd': herd,
            'period': period,
            'outcome': 0
        })

df_expanded = pd.DataFrame(expanded_data)

print(f"Expanded to individual level: {len(df_expanded)} animals")
print(f"Disease incidence rate: {df_expanded['outcome'].mean():.1%}")
print(f"Number of herds: {df_expanded['herd'].nunique()}")
print(f"Observations per herd: {df_expanded.groupby('herd').size().describe()}")

# Fit Binomial GLMM
# Model: outcome ~ period + (1|herd)
# Period is treated as continuous time variable

print("\n" + "="*60)
print("Fitting Model: outcome ~ period + (1|herd)")
print("="*60)

outcomes = [
    OutcomeSpec(name="disease", y="outcome", family=Binomial())
]

model = TabMixNN(
    formula="~ period + (1|herd)",
    outcomes=outcomes,
    architecture="gsem",
    hidden_dims=[32, 16],
    epochs=30,
    batch_size=64,
    learning_rate=0.01,
    verbose=True
)

print("\nTraining model...")
model.fit(df_expanded)

# Make predictions
predictions_dict = model.predict(df_expanded)
predictions = predictions_dict['disease'] if isinstance(predictions_dict, dict) else predictions_dict

# Convert probabilities to class predictions
pred_classes = (predictions > 0.5).astype(int)

# Evaluate
accuracy = accuracy_score(df_expanded['outcome'], pred_classes)
auc = roc_auc_score(df_expanded['outcome'], predictions)

print(f"\n" + "="*60)
print("Model Performance")
print("="*60)
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC-ROC:  {auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(df_expanded['outcome'], pred_classes)
print(f"\nConfusion Matrix:")
print(f"                Predicted")
print(f"              Healthy  Sick")
print(f"Actual Healthy   {cm[0,0]:4d}   {cm[0,1]:4d}")
print(f"       Sick      {cm[1,0]:4d}   {cm[1,1]:4d}")

# Sensitivity and Specificity
sensitivity = cm[1,1] / (cm[1,0] + cm[1,1])
specificity = cm[0,0] / (cm[0,0] + cm[0,1])
print(f"\nSensitivity (True Positive Rate): {sensitivity:.4f}")
print(f"Specificity (True Negative Rate): {specificity:.4f}")

# Display model summary
print(f"\n" + "="*60)
print("Model Summary")
print("="*60)
model.summary()

# Analyze predictions by period and herd
print(f"\n" + "="*60)
print("Disease Risk by Period")
print("="*60)

period_stats = []
for period in sorted(df_expanded['period'].unique()):
    period_data = df_expanded[df_expanded['period'] == period]
    period_preds = predictions[df_expanded['period'] == period]

    actual_rate = period_data['outcome'].mean()
    predicted_rate = period_preds.mean()

    period_stats.append({
        'Period': int(period),
        'N_animals': len(period_data),
        'Actual_rate': f"{actual_rate:.1%}",
        'Predicted_rate': f"{predicted_rate:.1%}"
    })

stats_df = pd.DataFrame(period_stats)
print(stats_df.to_string(index=False))

# Visualize predictions by herd and period
print(f"\n" + "="*60)
print("Visualizing Predictions by Herd")
print("="*60)

# Calculate average predicted risk by herd and period
herd_period_risk = df_expanded.copy()
herd_period_risk['predicted_prob'] = predictions

risk_summary = herd_period_risk.groupby(['herd', 'period']).agg({
    'outcome': 'mean',
    'predicted_prob': 'mean'
}).reset_index()

# Plot for first 8 herds
herds_to_plot = sorted(df_expanded['herd'].unique())[:8]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for idx, herd in enumerate(herds_to_plot):
    herd_data = risk_summary[risk_summary['herd'] == herd]

    ax = axes[idx]
    ax.plot(herd_data['period'], herd_data['outcome'],
            'o-', label='Actual', linewidth=2, markersize=8)
    ax.plot(herd_data['period'], herd_data['predicted_prob'],
            's--', label='Predicted', linewidth=2, markersize=6)
    ax.set_xlabel('Period')
    ax.set_ylabel('Disease Risk')
    ax.set_title(f'Herd {herd}')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/02_cbpp_predictions.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to: output/02_cbpp_predictions.png")

# Herd-level statistics
print(f"\n" + "="*60)
print("Per-Herd Statistics (first 6 herds)")
print("="*60)

herd_stats = []
for herd in sorted(df_expanded['herd'].unique())[:6]:
    herd_data = df_expanded[df_expanded['herd'] == herd]
    herd_preds = predictions[df_expanded['herd'] == herd]

    herd_stats.append({
        'Herd': herd,
        'N_animals': len(herd_data),
        'Actual_incidence': f"{herd_data['outcome'].mean():.1%}",
        'Predicted_risk': f"{herd_preds.mean():.1%}",
        'AUC': f"{roc_auc_score(herd_data['outcome'], herd_preds):.3f}"
    })

herd_stats_df = pd.DataFrame(herd_stats)
print(herd_stats_df.to_string(index=False))

print(f"\n" + "="*60)
print("Key Takeaways")
print("="*60)
print("""
1. Random effects GLMM captures herd-specific disease risk,
   accounting for clustering of animals within herds.

2. The model shows how disease risk changes over time periods
   while allowing for herd-to-herd variation.

3. This is a classic example from lme4 demonstrating binomial
   mixed models for grouped binary data.
""")
