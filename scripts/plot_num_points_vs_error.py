"""
Plot line chart: num_points (x-axis) vs mean_l2 (y-axis) for multiple models
Matching the style: white background, dashed lines with markers, data labels
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data
df = pd.read_csv('outputs/separate_calibration_evaluation_no_fixed.csv')

# Group by num_points and compute mean for each model
models = {
    'Neural Refine': 'mean_l2',
    'Original': 'original_mean_l2',
    'SimRBF': 'sim_rbf_mean_l2',
    'Similarity': 'similarity_mean_l2',
    'Polynomial': 'polynomial_mean_l2',
    'TPS': 'tps_mean_l2',
    'PWA': 'pwa_mean_l2',
    'SimTPS': 'simtps_mean_l2',
    'SimPWA': 'simpwa_mean_l2',
}

# Prepare data
grouped = df.groupby('num_points').agg({col: 'mean' for col in models.values()}).reset_index()
grouped = grouped.sort_values('num_points')

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Define colors and markers
colors = {
    'Neural Refine': '#1f77b4',  # blue
    'Original': '#d62728',        # red
    'SimRBF': '#ff7f0e',          # orange
    'Similarity': '#2ca02c',      # green
    'Polynomial': '#9467bd',      # purple
    'TPS': '#8c564b',            # brown
    'PWA': '#e377c2',             # pink
    'SimTPS': '#17becf',          # cyan
    'SimPWA': '#bcbd22',          # olive
}

markers = {
    'Neural Refine': 'o',      # circle
    'Original': 's',           # square
    'SimRBF': '^',             # triangle up
    'Similarity': 'D',         # diamond
    'Polynomial': 'v',         # triangle down
    'TPS': '<',                # triangle left
    'PWA': '>',                # triangle right
    'SimTPS': 'p',             # pentagon
    'SimPWA': '*',             # star
}

# Plot all models
for model_name, col_name in models.items():
    ax.plot(grouped['num_points'], grouped[col_name], color=colors[model_name],
            linestyle='--', marker=markers[model_name], markersize=7, linewidth=2, label=model_name)

    # Add data labels for top 3 models only
    if model_name in ['Neural Refine', 'SimRBF', 'Original']:
        for idx, row in grouped.iterrows():
            ax.text(row['num_points'], row[col_name], f"{row[col_name]:.1f}",
                    ha='center', va='bottom', fontsize=8, color=colors[model_name])

# Styling
ax.set_xlabel('Number of Calibration Points', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean L2 Error (pixels)', fontsize=12, fontweight='bold')
ax.set_title('Calibration Error vs Number of Points', fontsize=14, fontweight='bold')
ax.grid(True, axis='y', color='lightgray', linestyle='-', alpha=0.5)

# Set x-axis ticks
ax.set_xticks(grouped['num_points'])

# Set y-axis range with some padding
all_values = grouped[list(models.values())].values.flatten()
y_min = all_values.min() - 5
y_max = all_values.max() + 5
ax.set_ylim(y_min, y_max)

# Legend
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, ncol=2, fontsize=9)

# Remove top and right spines for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('outputs/num_points_vs_error_all_models.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Plot saved to outputs/num_points_vs_error_all_models.png")
print("\nData summary:")
print(grouped[['num_points'] + list(models.values())].to_string(index=False))