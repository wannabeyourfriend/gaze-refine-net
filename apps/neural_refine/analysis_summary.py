"""
Generate a formatted summary table of calibration methods performance.
"""

import pandas as pd
import numpy as np

# Load the summary CSV
summary_df = pd.read_csv('../../outputs/separate_calibration_summary.csv')

# Select columns to display
methods = ['original', 'sim_rbf', 'neural', 'similarity', 'tps', 'pwa']
metrics = ['mean_l2', 'median_l2']

print('=' * 120)
print('Multi-Baseline Neural Refine: Performance vs Calibration Point Count')
print('=' * 120)
print()

# For each calibration point count
for _, row in summary_df.iterrows():
    num_points = int(row['num_points'])
    num_datasets = int(row['num_datasets'])

    print(f'{num_points:2d} points ({num_datasets} datasets):')
    print('-' * 120)

    # Collect all methods
    method_data = []
    for method in methods:
        method_col = method if method == 'neural' else f'{method}_mean_l2'
        median_col = f'{method}_median_l2' if method != 'neural' else 'neural_median_l2'

        if method == 'neural':
            mean_val = row['neural_mean_l2']
            median_val = row['neural_median_l2']
            std_val = row['neural_std_l2']
        else:
            mean_col = f'{method}_mean_l2'
            median_col = f'{method}_median_l2'
            mean_val = row.get(mean_col)
            median_val = row.get(median_col)
            std_val = None

        if mean_val is not None and not np.isnan(mean_val):
            method_data.append({
                'name': method.replace('_', ' ').title(),
                'mean': mean_val,
                'median': median_val,
                'std': std_val
            })

    # Sort by mean error
    method_data.sort(key=lambda x: x['mean'])

    # Print table
    print(f"{'Method':<15} {'Mean L2 (px)':<15} {'Median L2 (px)':<15}")
    print('-' * 50)

    for i, m in enumerate(method_data):
        marker = '★ ' if i == 0 else '  '
        std_str = f' ±{m["std"]:.2f}' if m['std'] is not None and m['std'] > 0 else ''
        print(f'{marker}{m["name"]:<13} {m["mean"]:>6.2f}{std_str:<8} {m["median"]:>6.2f}')

    # Print improvements
    neural_mean = row['neural_mean_l2']
    orig_mean = row['original_mean_l2']
    sim_rbf_mean = row['sim_rbf_mean_l2']

    orig_imp = (orig_mean - neural_mean) / orig_mean * 100
    sim_imp = (sim_rbf_mean - neural_mean) / sim_rbf_mean * 100

    print()
    print(f'  Improvement vs Original: {orig_imp:+.1f}%')
    print(f'  Improvement vs SimRBF:   {sim_imp:+.1f}%')
    print()

print('=' * 120)

# Also create a comparison table for Mean L2 across all point counts
print()
print('Mean L2 Error Comparison Across Calibration Point Counts (px)')
print('=' * 120)
print()

# Create pivot table
point_counts = summary_df['num_points'].values
all_methods = ['original', 'sim_rbf', 'neural', 'similarity', 'tps', 'pwa']

print(f"{'Points':<8}", end='')
for method in all_methods:
    print(f"  {method.replace('_', ' ').title():<12}", end='')
print()
print('-' * 100)

for _, row in summary_df.iterrows():
    num_pts = int(row['num_points'])
    print(f'{num_pts:<8}', end='')

    for method in all_methods:
        if method == 'neural':
            val = row['neural_mean_l2']
        else:
            val = row.get(f'{method}_mean_l2')

        if val is not None and not np.isnan(val):
            print(f'  {val:>10.2f}  ', end='')
        else:
            print(f'  {"N/A":>10}  ', end='')
    print()

print()
print('=' * 120)

# Median L2 comparison
print()
print('Median L2 Error Comparison Across Calibration Point Counts (px)')
print('=' * 120)
print()

print(f"{'Points':<8}", end='')
for method in all_methods:
    print(f"  {method.replace('_', ' ').title():<12}", end='')
print()
print('-' * 100)

for _, row in summary_df.iterrows():
    num_pts = int(row['num_points'])
    print(f'{num_pts:<8}', end='')

    for method in all_methods:
        if method == 'neural':
            val = row['neural_median_l2']
        else:
            val = row.get(f'{method}_median_l2')

        if val is not None and not np.isnan(val):
            print(f'  {val:>10.2f}  ', end='')
        else:
            print(f'  {"N/A":>10}  ', end='')
    print()

print()
print('=' * 120)
