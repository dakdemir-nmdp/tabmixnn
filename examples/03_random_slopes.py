"""
Example 3: Random Slopes Model - ChickWeight Dataset

This example demonstrates fitting a mixed model with random intercepts AND
random slopes using the ChickWeight dataset.

Dataset: Chick growth over time for different diets
Reference: Handbook of Biological Statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabmixnn import TabMixNNRegressor
from data_utils import load_dataset, get_dataset_info

# Load the ChickWeight dataset
print("="*60)
print("Example 3: Random Slopes Model")
print("Dataset: ChickWeight - Growth Curves")
print("="*60)

df = load_dataset("chickweight")

# Display dataset info
info = get_dataset_info("chickweight")
print(f"\nDataset: {info['description']}")
print(f"\nFirst few rows:")
print(df.head(15))

print(f"\nData shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")

# ChickWeight dataset structure:
# - weight: body weight of chick (grams)
# - Time: number of days since birth (0-21)
# - Chick: unique chick identifier
# - Diet: diet type (1-4)

print(f"\nSummary statistics:")
print(df.groupby('Diet')['weight'].describe())

print(f"\nNumber of chicks: {df['Chick'].nunique()}")
print(f"Chicks per diet:")
print(df.groupby('Diet')['Chick'].nunique())

# Standardize weight for neural network stability
weight_mean = df['weight'].mean()
weight_std = df['weight'].std()
df['weight_scaled'] = (df['weight'] - weight_mean) / weight_std

print(f"\nWeight: mean={weight_mean:.2f}g, std={weight_std:.2f}g")

# Model 1: Random Intercepts Only
# Each chick has its own baseline weight, but same growth rate

print("\n" + "="*60)
print("Model 1: Random Intercepts Only")
print("Formula: weight ~ Time + Diet + (1|Chick)")
print("="*60)

model_ri = TabMixNNRegressor(
    formula="weight_scaled ~ Time + Diet + (1|Chick)",
    epochs=40,
    batch_size=64,
    learning_rate=0.01,
    verbose=False
)

print("\nTraining random intercepts model...")
model_ri.fit(df)

predictions_ri_scaled = model_ri.predict(df)
predictions_ri = predictions_ri_scaled * weight_std + weight_mean

mse_ri = np.mean((predictions_ri - df['weight'])**2)
r2_ri = 1 - mse_ri / np.var(df['weight'])
rmse_ri = np.sqrt(mse_ri)

print(f"\nRandom Intercepts Model Performance:")
print(f"  RMSE: {rmse_ri:.2f}g")
print(f"  R²:   {r2_ri:.4f}")

# Model 2: Random Intercepts AND Slopes
# Each chick has its own baseline AND growth rate

print("\n" + "="*60)
print("Model 2: Random Intercepts + Slopes")
print("Formula: weight ~ Time + Diet + (1 + Time|Chick)")
print("="*60)

model_rs = TabMixNNRegressor(
    formula="weight_scaled ~ Time + Diet + (1 + Time|Chick)",
    epochs=40,
    batch_size=64,
    learning_rate=0.01,
    verbose=True
)

print("\nTraining random slopes model...")
model_rs.fit(df)

predictions_rs_scaled = model_rs.predict(df)
predictions_rs = predictions_rs_scaled * weight_std + weight_mean

mse_rs = np.mean((predictions_rs - df['weight'])**2)
r2_rs = 1 - mse_rs / np.var(df['weight'])
rmse_rs = np.sqrt(mse_rs)

print(f"\nRandom Slopes Model Performance:")
print(f"  RMSE: {rmse_rs:.2f}g")
print(f"  R²:   {r2_rs:.4f}")

# Compare models
print("\n" + "="*60)
print("Model Comparison")
print("="*60)
print(f"\nRandom Intercepts Only:")
print(f"  RMSE: {rmse_ri:.2f}g")
print(f"  R²:   {r2_ri:.4f}")

print(f"\nRandom Intercepts + Slopes:")
print(f"  RMSE: {rmse_rs:.2f}g")
print(f"  R²:   {r2_rs:.4f}")

print(f"\nImprovement with random slopes:")
print(f"  ΔRMSE: {rmse_ri - rmse_rs:+.2f}g")
print(f"  ΔR²:   {r2_rs - r2_ri:+.4f}")

if r2_rs > r2_ri:
    print("\n✓ Random slopes improve fit - growth rates vary by chick!")
else:
    print("\n  Models perform similarly for this dataset.")

# Display model summary
print("\n" + "="*60)
print("Random Slopes Model Summary")
print("="*60)
model_rs.summary()

# Visualize growth curves for sample chicks
print("\n" + "="*60)
print("Visualizing Growth Curves")
print("="*60)

# Select 2 chicks from each diet
chicks_to_plot = []
for diet in sorted(df['Diet'].unique()):
    diet_chicks = df[df['Diet'] == diet]['Chick'].unique()
    chicks_to_plot.extend(list(diet_chicks[:2]))

chicks_to_plot = chicks_to_plot[:8]  # Limit to 8 for visualization

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for idx, chick in enumerate(chicks_to_plot):
    chick_data = df[df['Chick'] == chick].copy().sort_values('Time')
    chick_preds_ri = predictions_ri[df['Chick'] == chick]
    chick_preds_rs = predictions_rs[df['Chick'] == chick]

    # Get predictions in same order as sorted data
    chick_indices = df[df['Chick'] == chick].sort_values('Time').index
    chick_preds_ri = predictions_ri[chick_indices]
    chick_preds_rs = predictions_rs[chick_indices]

    diet = chick_data['Diet'].iloc[0]

    ax = axes[idx]
    ax.scatter(chick_data['Time'], chick_data['weight'],
              label='Observed', alpha=0.7, s=40, color='black')
    ax.plot(chick_data['Time'], chick_preds_ri,
           '--', label='Random Intercepts', linewidth=2, alpha=0.7)
    ax.plot(chick_data['Time'], chick_preds_rs,
           '-', label='Random Slopes', linewidth=2)
    ax.set_xlabel('Days')
    ax.set_ylabel('Weight (g)')
    ax.set_title(f'Chick {chick} (Diet {diet})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/03_chickweight_growth.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to: output/03_chickweight_growth.png")

# Diet effect analysis
print("\n" + "="*60)
print("Average Growth by Diet")
print("="*60)

diet_stats = []
for diet in sorted(df['Diet'].unique()):
    diet_data = df[df['Diet'] == diet]

    # Final weights (Time = 21)
    final_weights = diet_data[diet_data['Time'] == 21]['weight']

    diet_stats.append({
        'Diet': int(diet),
        'N_chicks': diet_data['Chick'].nunique(),
        'Avg_final_weight': f"{final_weights.mean():.1f}g",
        'Std_final_weight': f"{final_weights.std():.1f}g"
    })

diet_stats_df = pd.DataFrame(diet_stats)
print(diet_stats_df.to_string(index=False))

# Per-chick growth rates (first 8 chicks)
print("\n" + "="*60)
print("Individual Growth Rates (Sample Chicks)")
print("="*60)

chick_growth = []
for chick in df['Chick'].unique()[:8]:
    chick_data = df[df['Chick'] == chick].sort_values('Time')

    if len(chick_data) > 1:
        # Linear regression for growth rate
        time = chick_data['Time'].values
        weight = chick_data['weight'].values

        if len(time) > 1:
            growth_rate = np.polyfit(time, weight, 1)[0]

            chick_growth.append({
                'Chick': chick,
                'Diet': int(chick_data['Diet'].iloc[0]),
                'Growth_rate': f"{growth_rate:.2f} g/day",
                'Initial_weight': f"{chick_data.iloc[0]['weight']:.1f}g",
                'Final_weight': f"{chick_data.iloc[-1]['weight']:.1f}g"
            })

growth_df = pd.DataFrame(chick_growth)
print(growth_df.to_string(index=False))

print("\n" + "="*60)
print("Key Takeaways")
print("="*60)
print("""
1. Random slopes model allows each chick to have:
   - Different baseline weight (random intercept)
   - Different growth rate (random slope for Time)

2. This better captures individual variation in growth
   compared to assuming all chicks grow at the same rate.

3. The ChickWeight dataset is a classic example showing
   why random slopes are important for longitudinal data.

4. Each diet may affect both initial size and growth rate,
   which the random effects help to model.
""")
