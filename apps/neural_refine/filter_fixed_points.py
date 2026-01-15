"""
Filter out fixed calibration points from cleaned datasets.

This script removes all fixed calibration points (points that appear
multiple times in the dataset) from the cleaned datasets, keeping only
the non-fixed (test) points.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm


def filter_fixed_points(df, min_occurrences=2):
    """
    Separate fixed calibration points from non-fixed points.

    Args:
        df: Input DataFrame
        min_occurrences: Minimum number of occurrences for a point to be considered "fixed"

    Returns:
        non_fixed_df: DataFrame with only non-fixed points
        fixed_df: DataFrame with only fixed points
    """
    # Count occurrences of each (target_x, target_y) pair
    target_counts = df.groupby(['target_x', 'target_y']).size()

    # Identify which points are fixed (appear >= min_occurrences times)
    is_fixed = df.set_index(['target_x', 'target_y']).index.map(
        lambda x: target_counts[x] >= min_occurrences
    )

    fixed_df = df[is_fixed].copy()
    non_fixed_df = df[~is_fixed].copy()

    return non_fixed_df, fixed_df


def main():
    print('=' * 80)
    print('Filtering Fixed Calibration Points from Cleaned Datasets')
    print('=' * 80)

    # Paths
    data_root = Path('../../data/seperate_calibration_cleaned')
    output_root = Path('../../data/seperate_calibration_no_fixed')
    min_occurrences = 2

    # Find all calibration point configurations
    calibration_configs = sorted(data_root.iterdir(), key=lambda x: x.name)

    # Group by number of points
    from collections import defaultdict
    config_groups = defaultdict(list)
    for config_dir in calibration_configs:
        if config_dir.is_dir():
            parts = config_dir.name.split('_')
            if parts[0].replace('points', '').isdigit():
                num_points = int(parts[0].replace('points', ''))
                config_groups[num_points].append(config_dir)

    # Sort by number of points
    sorted_configs = sorted(config_groups.items())

    print(f'\nFound {len(sorted_configs)} calibration point configurations')
    for num_points, dirs in sorted_configs:
        print(f'  {num_points} points: {len(dirs)} dataset(s)')

    # Process each dataset
    print('\n' + '=' * 80)
    print('Processing Datasets')
    print('=' * 80)

    results = []

    for num_points, config_dirs in sorted_configs:
        print(f'\n--- Processing {num_points}-point calibration ---')

        for config_dir in config_dirs:
            csv_path = config_dir / 'all_trials_model_predictions_0111.csv'

            if not csv_path.exists():
                print(f'  Warning: CSV not found at {csv_path}')
                continue

            print(f'  Dataset: {config_dir.name}')

            # Load data
            df = pd.read_csv(csv_path)
            original_count = len(df)

            # Filter out fixed points
            non_fixed_df, fixed_df = filter_fixed_points(df, min_occurrences=min_occurrences)

            # Calculate some statistics
            non_fixed_count = len(non_fixed_df)
            fixed_count = len(fixed_df)
            removed_pct = fixed_count / original_count * 100

            # Calculate distance statistics
            orig_distance = np.sqrt(
                (df['origin_gaze_x'] - df['target_x'])**2 +
                (df['origin_gaze_y'] - df['target_y'])**2
            )

            non_fixed_distance = np.sqrt(
                (non_fixed_df['origin_gaze_x'] - non_fixed_df['target_x'])**2 +
                (non_fixed_df['origin_gaze_y'] - non_fixed_df['target_y'])**2
            )

            # Create output directory
            output_dir = output_root / config_dir.name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save non-fixed data (this is what we'll use for evaluation)
            output_csv = output_dir / csv_path.name
            non_fixed_df.to_csv(output_csv, index=False)

            # Also save fixed points separately for reference
            fixed_csv = output_dir / f'fixed_points_{csv_path.name}'
            fixed_df.to_csv(fixed_csv, index=False)

            # Copy other files from the original directory
            for file in config_dir.iterdir():
                if file.is_file() and file.name != csv_path.name:
                    shutil.copy2(file, output_dir / file.name)

            # Store results
            result = {
                'num_points': num_points,
                'dataset_name': config_dir.name,
                'original_count': original_count,
                'non_fixed_count': non_fixed_count,
                'fixed_count': fixed_count,
                'removed_pct': removed_pct,
                'original_mean_distance': orig_distance.mean(),
                'non_fixed_mean_distance': non_fixed_distance.mean(),
            }
            results.append(result)

            # Print summary
            print(f'    Original samples:  {original_count}')
            print(f'    Non-fixed samples: {non_fixed_count} ({non_fixed_count/original_count*100:.1f}%)')
            print(f'    Fixed samples:     {fixed_count} ({removed_pct:.1f}%)')
            print(f'    Original mean distance: {orig_distance.mean():.2f} px')
            print(f'    Non-fixed mean distance: {non_fixed_distance.mean():.2f} px')
            print(f'    Saved to: {output_csv}')

    # Save summary
    results_df = pd.DataFrame(results)
    summary_path = Path('../../outputs/fixed_points_filtering_summary.csv')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(summary_path, index=False)
    print(f'\nSummary saved to {summary_path}')

    # Print aggregated statistics
    print('\n' + '=' * 80)
    print('Aggregated Statistics by Calibration Point Count')
    print('=' * 80)

    for num_points in sorted(config_groups.keys()):
        subset = results_df[results_df['num_points'] == num_points]
        if len(subset) == 0:
            continue

        print(f'\n{num_points} points ({len(subset)} datasets):')
        print(f'  Total original:  {subset["original_count"].sum():>6}')
        print(f'  Total non-fixed: {subset["non_fixed_count"].sum():>6} ({subset["non_fixed_count"].sum()/subset["original_count"].sum()*100:.1f}%)')
        print(f'  Total fixed:     {subset["fixed_count"].sum():>6} ({subset["fixed_count"].sum()/subset["original_count"].sum()*100:.1f}%)')
        print(f'  Avg fixed %:     {subset["removed_pct"].mean():>6.1f}%')
        print(f'  Original mean dist:  {subset["original_mean_distance"].mean():>6.2f} ± {subset["original_mean_distance"].std():.2f} px')
        print(f'  Non-fixed mean dist: {subset["non_fixed_mean_distance"].mean():>6.2f} ± {subset["non_fixed_mean_distance"].std():.2f} px')

    print('\n' + '=' * 80)
    print('Fixed Points Filtering Complete!')
    print('=' * 80)
    print(f'\nDatasets without fixed points saved to: {output_root}')
    print(f'You can now run evaluation on datasets with only non-fixed (test) points.')


if __name__ == '__main__':
    main()
