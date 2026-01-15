"""
Split JuDo-1000 dataset by target points to avoid label leakage.

Key: Ensure train/val/test sets have completely disjoint target points.
This tests generalization to NEW calibration points, not just new samples.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ast
from tqdm import tqdm
import yaml


def load_judo_dataset(csv_path):
    """Load JuDo-1000 dataset and transform to expected format."""
    df = pd.read_csv(csv_path)

    # Transform to expected format
    result_df = pd.DataFrame()
    result_df['target_x'] = df['target_x']
    result_df['target_y'] = df['target_y']
    result_df['origin_gaze_x'] = df['mean_x']
    result_df['origin_gaze_y'] = df['mean_y']

    # Compute spread from raw_data
    spreads = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing spreads"):
        raw_data = ast.literal_eval(row['raw_data'])
        x_vals = [p['x_left'] for p in raw_data]
        y_vals = [p['y_left'] for p in raw_data]
        x_std = np.std(x_vals, ddof=1) if len(x_vals) > 1 else 0
        y_std = np.std(y_vals, ddof=1) if len(y_vals) > 1 else 0
        spread = np.sqrt(x_std**2 + y_std**2)
        spreads.append(spread)

    result_df['spread'] = spreads
    return result_df


def split_by_target_points(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=1047):
    """
    Split data by target points to ensure no overlap.

    Args:
        df: DataFrame with target_x, target_y columns
        train_ratio: Ratio of target points for training
        val_ratio: Ratio of target points for validation
        test_ratio: Ratio of target points for testing
        seed: Random seed for reproducibility

    Returns:
        train_df, val_df, test_df: DataFrames with disjoint target sets
    """
    np.random.seed(seed)

    # Round target coordinates to group identical points
    df['target_x_rounded'] = np.round(df['target_x'], 2)
    df['target_y_rounded'] = np.round(df['target_y'], 2)
    df['target_point'] = list(zip(df['target_x_rounded'], df['target_y_rounded']))

    # Get unique target points
    unique_targets = df['target_point'].unique()
    n_unique = len(unique_targets)

    print(f"Total unique target points: {n_unique}")

    # Shuffle targets
    np.random.shuffle(unique_targets)

    # Split target points
    n_train = int(n_unique * train_ratio)
    n_val = int(n_unique * val_ratio)
    # n_test = remaining

    train_targets = set(unique_targets[:n_train])
    val_targets = set(unique_targets[n_train:n_train + n_val])
    test_targets = set(unique_targets[n_train + n_val:])

    print(f"\nTarget point split:")
    print(f"  Train:      {len(train_targets)} points ({len(train_targets)/n_unique*100:.1f}%)")
    print(f"  Validation: {len(val_targets)} points ({len(val_targets)/n_unique*100:.1f}%)")
    print(f"  Test:       {len(test_targets)} points ({len(test_targets)/n_unique*100:.1f}%)")

    # Verify no overlap
    assert len(train_targets & val_targets) == 0, "Train and val targets overlap!"
    assert len(train_targets & test_targets) == 0, "Train and test targets overlap!"
    assert len(val_targets & test_targets) == 0, "Val and test targets overlap!"

    # Split DataFrame
    train_df = df[df['target_point'].isin(train_targets)].copy()
    val_df = df[df['target_point'].isin(val_targets)].copy()
    test_df = df[df['target_point'].isin(test_targets)].copy()

    # Drop temporary columns
    for split_df in [train_df, val_df, test_df]:
        split_df.drop(columns=['target_x_rounded', 'target_y_rounded', 'target_point'], inplace=True)

    print(f"\nSample counts:")
    print(f"  Train:      {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test:       {len(test_df)} samples")

    # Print target points for each split
    print(f"\n" + "="*80)
    print("Train Target Points:")
    print("="*80)
    train_target_points = df[df['target_point'].isin(train_targets)][['target_x', 'target_y']].drop_duplicates()
    for _, row in train_target_points.iterrows():
        print(f"  ({row['target_x']:.2f}, {row['target_y']:.2f})")

    print(f"\n" + "="*80)
    print("Validation Target Points:")
    print("="*80)
    val_target_points = df[df['target_point'].isin(val_targets)][['target_x', 'target_y']].drop_duplicates()
    for _, row in val_target_points.iterrows():
        print(f"  ({row['target_x']:.2f}, {row['target_y']:.2f})")

    print(f"\n" + "="*80)
    print("Test Target Points:")
    print("="*80)
    test_target_points = df[df['target_point'].isin(test_targets)][['target_x', 'target_y']].drop_duplicates()
    for _, row in test_target_points.iterrows():
        print(f"  ({row['target_x']:.2f}, {row['target_y']:.2f})")

    return train_df, val_df, test_df


def main():
    print("="*80)
    print("JuDo-1000 Dataset Split (By Target Points)")
    print("="*80)

    # Paths
    input_csv = 'data/raw/all/filtered_trials_summary_first_tenth.csv'
    output_dir = Path('data/prepared/judo_1000_split_no_leakage')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and transform data
    print("\nLoading JuDo-1000 dataset...")
    df = load_judo_dataset(input_csv)
    print(f"Loaded {len(df)} samples")

    # Split by target points
    train_df, val_df, test_df = split_by_target_points(df, seed=1047)

    # Save splits
    print(f"\n" + "="*80)
    print("Saving splits...")
    print("="*80)

    train_path = output_dir / 'train.csv'
    val_path = output_dir / 'val.csv'
    test_path = output_dir / 'test.csv'

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train:      {train_path}")
    print(f"Validation: {val_path}")
    print(f"Test:       {test_path}")

    # Save split metadata
    metadata = {
        'split_type': 'by_target_points',
        'total_samples': len(df),
        'unique_targets': df[['target_x', 'target_y']].drop_duplicates().shape[0],
        'train': {
            'num_target_points': train_df[['target_x', 'target_y']].drop_duplicates().shape[0],
            'num_samples': len(train_df)
        },
        'val': {
            'num_target_points': val_df[['target_x', 'target_y']].drop_duplicates().shape[0],
            'num_samples': len(val_df)
        },
        'test': {
            'num_target_points': test_df[['target_x', 'target_y']].drop_duplicates().shape[0],
            'num_samples': len(test_df)
        },
        'seed': 1047
    }

    metadata_path = output_dir / 'split_metadata.yaml'
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)

    print(f"Metadata:   {metadata_path}")

    print(f"\n" + "="*80)
    print("Summary")
    print("="*80)
    print("✓ Train and test sets have completely disjoint target points")
    print("✓ This tests generalization to NEW calibration points")
    print("✓ No label leakage")

    print("\nNext step: Generate baseline calibrations for each split")
    print("  python scripts/generate_judo_baselines_no_leakage.py")


if __name__ == '__main__':
    main()
