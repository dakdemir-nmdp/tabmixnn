#!/usr/bin/env python3
"""
Example 07: Architecture Comparison - GSEM vs Manifold

This example demonstrates the difference between two TabMixNN architectures:
1. GSEM (Generalized Structural Equation Model) - Standard hierarchical approach
2. Manifold - Universal backbone with latent manifold representations

Dataset: Machines (nlme) - Production scores on 3 machines by 6 workers

Key concepts:
- Architecture parameter (gsem vs manifold)
- Performance comparison on same dataset
- Visualization of predictions from both architectures
- Understanding when to use each architecture
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Add parent directory to path to import data_utils
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_dataset, get_dataset_info

try:
    from tabmixnn import TabMixNN, TabMixNNRegressor
    from tabmixnn.parsing.formula import OutcomeSpec
    from tabmixnn.core.families import Gaussian
except ImportError:
    print("ERROR: TabMixNN not installed. Install with: pip install tabmixnn")
    exit(1)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# ========== 1. Load Machines Data ==========
print("="*70)
print("Loading Machines Dataset")
print("="*70)

df = load_dataset("machines")
if df is None or len(df) == 0:
    print("Failed to load dataset. Please check your internet connection.")
    exit(1)

print(f"\nDataset shape: {df.shape}")
print(f"\nDataset info:")
print(get_dataset_info("machines"))
print(f"\nFirst few rows:")
print(df.head())

# ========== 2. Prepare Data ==========
print("\n" + "="*70)
print("Preparing Data")
print("="*70)

# Rename columns for clarity
df.columns = [col.lower() for col in df.columns]

print(f"\nColumns: {list(df.columns)}")
print(f"\nUnique workers: {df['worker'].nunique()}")
print(f"\nUnique machines: {df['machine'].nunique()}")

# Standardize score for better training
df['score_scaled'] = (df['score'] - df['score'].mean()) / df['score'].std()

# Convert Machine to categorical codes
df['machine_code'] = df['machine'].astype('category').cat.codes

# Split data (stratified by worker to ensure each worker in both sets)
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42,
                                     stratify=df['worker'])

print(f"\nTraining set: {len(train_df)} observations")
print(f"Test set: {len(test_df)} observations")
print(f"Workers in train: {train_df['worker'].nunique()}")
print(f"Workers in test: {test_df['worker'].nunique()}")

# ========== 3. Train GSEM Architecture Model ==========
print("\n" + "="*70)
print("Training Model 1: GSEM Architecture")
print("="*70)

print("\nGSEM (Generalized SEM) is the default architecture.")
print("It uses a hierarchical structure suitable for mixed effects models.")

start_time = time.time()

model_gsem = TabMixNNRegressor(
    formula="score_scaled ~ machine_code + (1|worker)",
    architecture="gsem",
    hidden_dims=[64, 32],
    embedding_dim=16,
    lr=0.01,
    batch_size=16,
    epochs=150,
    early_stopping=True,
    patience=20,
    verbose=False
)

model_gsem.fit(train_df, val_df=test_df)
gsem_time = time.time() - start_time

# Predictions
pred_gsem_train = model_gsem.predict(train_df)
pred_gsem_test = model_gsem.predict(test_df)

# Metrics
gsem_train_rmse = np.sqrt(mean_squared_error(train_df['score_scaled'], pred_gsem_train))
gsem_test_rmse = np.sqrt(mean_squared_error(test_df['score_scaled'], pred_gsem_test))
gsem_train_r2 = r2_score(train_df['score_scaled'], pred_gsem_train)
gsem_test_r2 = r2_score(test_df['score_scaled'], pred_gsem_test)
gsem_test_mae = mean_absolute_error(test_df['score_scaled'], pred_gsem_test)

print(f"\nGSEM Performance:")
print(f"  Training time: {gsem_time:.2f}s")
print(f"  Train RMSE: {gsem_train_rmse:.4f}")
print(f"  Test RMSE:  {gsem_test_rmse:.4f}")
print(f"  Train R²:   {gsem_train_r2:.4f}")
print(f"  Test R²:    {gsem_test_r2:.4f}")
print(f"  Test MAE:   {gsem_test_mae:.4f}")

# ========== 4. Train Manifold Architecture Model ==========
print("\n" + "="*70)
print("Training Model 2: Manifold Architecture")
print("="*70)

print("\nManifold (Universal) architecture uses latent manifold representations.")
print("It can capture more complex interactions through manifold learning.")

start_time = time.time()

model_manifold = TabMixNNRegressor(
    formula="score_scaled ~ machine_code + (1|worker)",
    architecture="manifold",
    hidden_dims=[64, 32],
    embedding_dim=16,
    use_recursive=True,
    aggregation="concat",
    lr=0.01,
    batch_size=16,
    epochs=150,
    early_stopping=True,
    patience=20,
    verbose=False
)

model_manifold.fit(train_df, val_df=test_df)
manifold_time = time.time() - start_time

# Predictions
pred_manifold_train = model_manifold.predict(train_df)
pred_manifold_test = model_manifold.predict(test_df)

# Metrics
manifold_train_rmse = np.sqrt(mean_squared_error(train_df['score_scaled'], pred_manifold_train))
manifold_test_rmse = np.sqrt(mean_squared_error(test_df['score_scaled'], pred_manifold_test))
manifold_train_r2 = r2_score(train_df['score_scaled'], pred_manifold_train)
manifold_test_r2 = r2_score(test_df['score_scaled'], pred_manifold_test)
manifold_test_mae = mean_absolute_error(test_df['score_scaled'], pred_manifold_test)

print(f"\nManifold Performance:")
print(f"  Training time: {manifold_time:.2f}s")
print(f"  Train RMSE: {manifold_train_rmse:.4f}")
print(f"  Test RMSE:  {manifold_test_rmse:.4f}")
print(f"  Train R²:   {manifold_train_r2:.4f}")
print(f"  Test R²:    {manifold_test_r2:.4f}")
print(f"  Test MAE:   {manifold_test_mae:.4f}")

# ========== 5. Compare Architectures ==========
print("\n" + "="*70)
print("Architecture Comparison")
print("="*70)

comparison_df = pd.DataFrame({
    'Metric': ['Test RMSE', 'Test R²', 'Test MAE', 'Training Time (s)'],
    'GSEM': [gsem_test_rmse, gsem_test_r2, gsem_test_mae, gsem_time],
    'Manifold': [manifold_test_rmse, manifold_test_r2, manifold_test_mae, manifold_time]
})

print("\n" + comparison_df.to_string(index=False))

# Determine winner
if gsem_test_rmse < manifold_test_rmse:
    print(f"\n✓ GSEM has lower test RMSE (better)")
else:
    print(f"\n✓ Manifold has lower test RMSE (better)")

if gsem_test_r2 > manifold_test_r2:
    print(f"✓ GSEM has higher test R² (better)")
else:
    print(f"✓ Manifold has higher test R² (better)")

# ========== 6. Visualize Results ==========
print("\n" + "="*70)
print("Creating Visualizations")
print("="*70)

os.makedirs('output', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Predictions vs Actual (GSEM)
axes[0, 0].scatter(test_df['score_scaled'], pred_gsem_test, alpha=0.6, s=60)
axes[0, 0].plot([test_df['score_scaled'].min(), test_df['score_scaled'].max()],
               [test_df['score_scaled'].min(), test_df['score_scaled'].max()],
               'r--', lw=2, label='Perfect prediction')
axes[0, 0].set_xlabel('Actual Score (scaled)', fontsize=11)
axes[0, 0].set_ylabel('Predicted Score', fontsize=11)
axes[0, 0].set_title(f'GSEM Architecture\n(R² = {gsem_test_r2:.3f}, RMSE = {gsem_test_rmse:.3f})',
                    fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Predictions vs Actual (Manifold)
axes[0, 1].scatter(test_df['score_scaled'], pred_manifold_test, alpha=0.6, s=60, color='orange')
axes[0, 1].plot([test_df['score_scaled'].min(), test_df['score_scaled'].max()],
               [test_df['score_scaled'].min(), test_df['score_scaled'].max()],
               'r--', lw=2, label='Perfect prediction')
axes[0, 1].set_xlabel('Actual Score (scaled)', fontsize=11)
axes[0, 1].set_ylabel('Predicted Score', fontsize=11)
axes[0, 1].set_title(f'Manifold Architecture\n(R² = {manifold_test_r2:.3f}, RMSE = {manifold_test_rmse:.3f})',
                    fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Residuals comparison
residuals_gsem = test_df['score_scaled'].values - pred_gsem_test
residuals_manifold = test_df['score_scaled'].values - pred_manifold_test

axes[1, 0].hist(residuals_gsem, bins=15, alpha=0.6, label='GSEM', color='blue', density=True)
axes[1, 0].hist(residuals_manifold, bins=15, alpha=0.6, label='Manifold', color='orange', density=True)
axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
axes[1, 0].set_xlabel('Residuals', fontsize=11)
axes[1, 0].set_ylabel('Density', fontsize=11)
axes[1, 0].set_title('Residual Distribution Comparison', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Performance metrics comparison
metrics = ['Test RMSE\n(lower better)', 'Test R²\n(higher better)', 'Test MAE\n(lower better)']
gsem_metrics = [gsem_test_rmse, gsem_test_r2, gsem_test_mae]
manifold_metrics = [manifold_test_rmse, manifold_test_r2, manifold_test_mae]

x = np.arange(len(metrics))
width = 0.35

bars1 = axes[1, 1].bar(x - width/2, gsem_metrics, width, label='GSEM', color='blue', alpha=0.7)
bars2 = axes[1, 1].bar(x + width/2, manifold_metrics, width, label='Manifold', color='orange', alpha=0.7)

axes[1, 1].set_xlabel('Metric', fontsize=11)
axes[1, 1].set_ylabel('Value', fontsize=11)
axes[1, 1].set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(metrics, fontsize=9)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
output_file = 'output/07_architecture_comparison.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_file}")

# ========== 7. Summary ==========
print("\n" + "="*70)
print("Summary")
print("="*70)
print(f"\nDataset: Machines (N={len(df)}, 6 workers, 3 machines)")
print(f"\nArchitectures compared:")
print(f"  1. GSEM (Generalized SEM) - Standard hierarchical structure")
print(f"  2. Manifold (Universal) - Latent manifold representations")
print(f"\nKey findings:")

if abs(gsem_test_r2 - manifold_test_r2) < 0.02:
    print("  - Both architectures perform similarly on this dataset")
    print("  - Choice depends on computational constraints and interpretability needs")
elif gsem_test_r2 > manifold_test_r2:
    print("  - GSEM performs better on this dataset")
    print("  - Simpler hierarchical structure is sufficient for this problem")
else:
    print("  - Manifold performs better on this dataset")
    print("  - Latent manifold representations capture additional complexity")

print(f"\nWhen to use each:")
print("  GSEM:")
print("    - Clear hierarchical/grouped structure")
print("    - Need interpretable random effects")
print("    - Smaller datasets")
print("    - Computational efficiency is important")
print("\n  Manifold:")
print("    - Complex interactions between features")
print("    - Large datasets with rich structure")
print("    - When maximum predictive performance is needed")
print("    - Multi-task learning with diverse outcomes")

print(f"\nVisualization saved: {output_file}")
print("\n" + "="*70)
