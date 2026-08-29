"""
Analyze target point distribution in JuDo-1000 dataset to avoid label leakage.
Goal: Identify unique target points and create train/val/test splits with disjoint target sets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_target_distribution(csv_path):
    """Analyze unique target points and their distribution."""
    df = pd.read_csv(csv_path)

    print("="*80)
    print("JuDo-1000 Target Point Distribution Analysis")
    print("="*80)

    # Total samples
    print(f"\nTotal samples: {len(df)}")

    # Extract unique target points (round to avoid floating point issues)
    df['target_x_rounded'] = np.round(df['target_x'], 2)
    df['target_y_rounded'] = np.round(df['target_y'], 2)
    df['target_point'] = list(zip(df['target_x_rounded'], df['target_y_rounded']))

    unique_targets = df['target_point'].unique()
    print(f"Number of unique target points: {len(unique_targets)}")

    # Count samples per target point
    target_counts = df.groupby('target_point').size().sort_values(ascending=False)

    print("\n" + "="*80)
    print("Target Point Statistics")
    print("="*80)
    print(f"Min samples per target: {target_counts.min()}")
    print(f"Max samples per target: {target_counts.max()}")
    print(f"Mean samples per target: {target_counts.mean():.1f}")
    print(f"Median samples per target: {target_counts.median():.1f}")

    print("\n" + "="*80)
    print("Samples per Target Point Distribution")
    print("="*80)
    for count in range(min(10, len(target_counts))):
        target = target_counts.index[count]
        print(f"Target ({target[0]:.2f}, {target[1]:.2f}): {target_counts.iloc[count]} samples")

    # Analyze spatial distribution
    print("\n" + "="*80)
    print("Spatial Distribution")
    print("="*80)
    print(f"X range: [{df['target_x'].min():.2f}, {df['target_x'].max():.2f}]")
    print(f"Y range: [{df['target_y'].min():.2f}, {df['target_y'].max():.2f}]")

    # Visualize target points
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: All target points
    ax = axes[0]
    scatter = ax.scatter(df['target_x'], df['target_y'],
                        c=np.arange(len(df)), cmap='viridis',
                        alpha=0.6, s=10)
    ax.set_xlabel('Target X (pixels)')
    ax.set_ylabel('Target Y (pixels)')
    ax.set_title(f'All {len(df)} Samples (colored by index)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # Plot 2: Unique target points
    ax = axes[1]
    unique_df = df.drop_duplicates(subset=['target_point'])
    # Count samples per unique target
    target_counts_map = df.groupby('target_point').size().to_dict()
    sizes = [target_counts_map[t] for t in unique_df['target_point']]

    scatter = ax.scatter(unique_df['target_x'], unique_df['target_y'],
                        c=sizes, cmap='hot', s=sizes,
                        alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Target X (pixels)')
    ax.set_ylabel('Target Y (pixels)')
    ax.set_title(f'Unique Target Points (color/size = sample count)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Number of samples')

    plt.tight_layout()

    # Save visualization
    output_dir = Path('outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'judo_1000_target_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {output_path}")

    # Suggest splitting strategy
    print("\n" + "="*80)
    print("Recommended Splitting Strategy")
    print("="*80)

    # Strategy 1: Random split by target points
    n_unique = len(unique_targets)
    n_train = int(n_unique * 0.7)
    n_val = int(n_unique * 0.15)
    n_test = n_unique - n_train - n_val

    print(f"\nStrategy 1: Random target point split")
    print(f"  Train:      {n_train} unique target points (~70%)")
    print(f"  Validation: {n_val} unique target points (~15%)")
    print(f"  Test:       {n_test} unique target points (~15%)")

    # Count approximate samples per split
    avg_samples = target_counts.mean()
    print(f"\n  Approximate samples per split (based on mean {avg_samples:.1f} samples/target):")
    print(f"  Train:      ~{int(n_train * avg_samples)} samples")
    print(f"  Validation: ~{int(n_val * avg_samples)} samples")
    print(f"  Test:       ~{int(n_test * avg_samples)} samples")

    # Strategy 2: Stratified split (if targets form a grid)
    print(f"\nStrategy 2: Grid-based stratified split (if targets form a regular grid)")
    print(f"  - Analyze grid structure and sample from different regions")
    print(f"  - Ensures spatial coverage across splits")

    # Group by rounded coordinates for easier analysis
    df['target_x_int'] = np.round(df['target_x'])
    df['target_y_int'] = np.round(df['target_y'])

    # Check if targets form a grid
    unique_x = sorted(df['target_x_int'].unique())
    unique_y = sorted(df['target_y_int'].unique())

    print(f"\n" + "="*80)
    print("Grid Structure Analysis")
    print("="*80)
    print(f"Unique X positions: {len(unique_x)}")
    print(f"Unique Y positions: {len(unique_y)}")
    print(f"If grid: {len(unique_x)} x {len(unique_y)} = {len(unique_x) * len(unique_y)} possible points")
    print(f"Actual unique points: {len(unique_targets)}")

    is_grid = len(unique_targets) >= len(unique_x) * len(unique_y) * 0.9
    if is_grid:
        print("✓ Targets appear to form a regular grid")
    else:
        print("✗ Targets do NOT form a perfect grid")

    return {
        'df': df,
        'unique_targets': unique_targets,
        'target_counts': target_counts,
        'unique_x': unique_x,
        'unique_y': unique_y,
        'is_grid': is_grid
    }


if __name__ == '__main__':
    csv_path = 'data/raw/all/filtered_trials_summary_first_tenth.csv'

    results = analyze_target_distribution(csv_path)

    print("\n" + "="*80)
    print("Next Steps")
    print("="*80)
    print("1. Create new split script that ensures disjoint target sets")
    print("2. Regenerate baseline calibrations for new splits")
    print("3. Retrain model with corrected data")
