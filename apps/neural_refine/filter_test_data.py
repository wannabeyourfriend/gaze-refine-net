"""
Filter test datasets by removing samples that appear in training data.

This script processes the seperate_calibration datasets and removes any samples
that are present in the training set, ensuring true out-of-distribution evaluation.
"""
import pandas as pd
from pathlib import Path
import sys


def load_training_samples(train_csv_path: Path) -> set:
    """
    Load unique sample identifiers from training data.

    Returns:
        Set of tuples (subject, timestamp, target_index)
    """
    print(f"Loading training data from {train_csv_path}...")
    train_df = pd.read_csv(train_csv_path)

    # Create unique identifier from subject, timestamp, target_index
    train_samples = set(
        zip(
            train_df['subject'].values,
            train_df['timestamp'].values,
            train_df['target_index'].values
        )
    )

    print(f"  Found {len(train_samples)} unique training samples")
    return train_samples


def filter_test_data(test_csv_path: Path, train_samples: set, output_csv_path: Path) -> dict:
    """
    Filter test data by removing samples that appear in training set.

    Args:
        test_csv_path: Path to test CSV file
        train_samples: Set of (subject, timestamp, target_index) tuples to exclude
        output_csv_path: Path to save filtered CSV

    Returns:
        Dictionary with filtering statistics
    """
    print(f"\nProcessing {test_csv_path.name}...")
    test_df = pd.read_csv(test_csv_path)

    original_count = len(test_df)

    # Create unique identifiers
    test_samples = list(zip(
        test_df['subject'].values,
        test_df['timestamp'].values,
        test_df['target_index'].values
    ))

    # Find samples to keep (not in training set)
    keep_mask = [
        sample not in train_samples
        for sample in test_samples
    ]

    # Filter dataframe
    filtered_df = test_df[keep_mask].copy()

    # Count removed samples
    removed_count = original_count - len(filtered_df)
    removal_rate = removed_count / original_count * 100

    print(f"  Original: {original_count} samples")
    print(f"  Removed:  {removed_count} samples ({removal_rate:.1f}%)")
    print(f"  Remaining: {len(filtered_df)} samples")

    # Save filtered data
    filtered_df.to_csv(output_csv_path, index=False)
    print(f"  Saved to: {output_csv_path}")

    return {
        'original': original_count,
        'removed': removed_count,
        'remaining': len(filtered_df),
        'removal_rate': removal_rate
    }


def main():
    print('='*70)
    print('Filter Test Data - Remove Training Samples')
    print('='*70)

    # Paths
    base_path = Path(__file__).parent.parent.parent
    train_csv = base_path / 'data' / 'prepared' / 'all_trials_split' / 'train.csv'
    seperate_calib_dir = base_path / 'data' / 'seperate_calibration'
    output_dir = base_path / 'data' / 'seperate_calibration_filtered'

    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load training samples
    train_samples = load_training_samples(train_csv)

    # Process all subdirectories in seperate_calibration
    all_stats = {}
    test_subdirs = sorted(seperate_calib_dir.iterdir())
    test_subdirs = [d for d in test_subdirs if d.is_dir()]

    print(f"\nFound {len(test_subdirs)} test subsets to process")

    for subset_dir in test_subdirs:
        # Find CSV file in this directory
        csv_files = list(subset_dir.glob('all_trials_model_predictions*.csv'))

        if not csv_files:
            print(f"\nSkipping {subset_dir.name} (no CSV found)")
            continue

        csv_path = csv_files[0]  # Use first matching CSV

        # Create output subdirectory
        output_subset_dir = output_dir / subset_dir.name
        output_subset_dir.mkdir(exist_ok=True)

        # Filter and save
        output_csv = output_subset_dir / csv_path.name
        stats = filter_test_data(csv_path, train_samples, output_csv)

        # Copy other files from this subset
        for other_file in subset_dir.iterdir():
            if other_file.is_file() and other_file != csv_path:
                dest = output_subset_dir / other_file.name
                dest.write_bytes(other_file.read_bytes())

        all_stats[subset_dir.name] = stats

    # Print summary
    print('\n' + '='*70)
    print('Filtering Summary')
    print('='*70)

    total_original = sum(s['original'] for s in all_stats.values())
    total_removed = sum(s['removed'] for s in all_stats.values())
    total_remaining = sum(s['remaining'] for s in all_stats.values())

    print(f"\nTotal across all subsets:")
    print(f"  Original samples:  {total_original}")
    print(f"  Removed samples:   {total_removed}")
    print(f"  Remaining samples: {total_remaining}")
    print(f"  Overall removal:   {total_removed/total_original*100:.1f}%")

    # Per-subset details
    print(f"\nPer-subset details:")
    for subset_name, stats in sorted(all_stats.items()):
        print(f"  {subset_name:20s}: {stats['remaining']:5d} / {stats['original']:5d} "
              f"({stats['removal_rate']:5.1f}% removed)")

    print(f"\nFiltered data saved to: {output_dir}")


if __name__ == '__main__':
    main()
