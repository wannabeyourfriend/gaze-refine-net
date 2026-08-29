"""
Unified data processing script for cleaning, filtering, and splitting gaze calibration datasets.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def calculate_distance(df, gaze_x_col='origin_gaze_x', gaze_y_col='origin_gaze_y'):
    """Calculate Euclidean distance from gaze to target."""
    df['distance'] = np.sqrt(
        (df[gaze_x_col] - df['target_x'])**2 +
        (df[gaze_y_col] - df['target_y'])**2
    )
    return df


def detect_outliers(df, method='iqr', threshold=1.5):
    """Detect outliers using IQR or z-score method."""
    df = df.copy()

    if method == 'iqr':
        Q1 = df['distance'].quantile(0.25)
        Q3 = df['distance'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df['is_outlier'] = (df['distance'] < lower_bound) | (df['distance'] > upper_bound)

    elif method == 'zscore':
        mean = df['distance'].mean()
        std = df['distance'].std()
        df['z_score'] = (df['distance'] - mean) / std
        df['is_outlier'] = np.abs(df['z_score']) > threshold
    else:
        raise ValueError(f"Unknown method: {method}")

    return df


def clean_outliers(input_path, output_path, method='iqr', threshold=1.5, gaze_x_col='origin_gaze_x', gaze_y_col='origin_gaze_y'):
    """Remove outliers from dataset."""
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)

    print(f"Original samples: {len(df)}")
    df = calculate_distance(df, gaze_x_col, gaze_y_col)
    df = detect_outliers(df, method, threshold)

    cleaned_df = df[~df['is_outlier']].drop(columns=['distance', 'is_outlier'])
    print(f"Cleaned samples: {len(cleaned_df)}")
    print(f"Removed: {len(df) - len(cleaned_df)} outliers")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


def filter_fixed_points(input_path, output_path, min_occurrences=2):
    """Separate fixed calibration points from non-fixed points."""
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)

    target_counts = df.groupby(['target_x', 'target_y']).size()
    is_fixed = df.set_index(['target_x', 'target_y']).index.map(
        lambda x: target_counts[x] >= min_occurrences
    )

    fixed_df = df[is_fixed]
    non_fixed_df = df[~is_fixed]

    print(f"Fixed points: {len(fixed_df)} ({len(fixed_df)/len(df)*100:.1f}%)")
    print(f"Non-fixed points: {len(non_fixed_df)} ({len(non_fixed_df)/len(df)*100:.1f}%)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fixed_df.to_csv(output_path.parent / 'fixed_points.csv', index=False)
    non_fixed_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


def split_dataset(input_path, output_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42):
    """Split dataset into train/val/test sets."""
    from sklearn.model_selection import train_test_split

    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    print(f"Total samples: {len(df)}")

    train_df, temp_df = train_test_split(df, test_size=(val_ratio + test_ratio), random_state=random_seed)
    val_df, test_df = train_test_split(temp_df, test_size=test_ratio/(val_ratio + test_ratio), random_state=random_seed)

    print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)
    print(f"Saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Process gaze calibration datasets')
    subparsers = parser.add_subparsers(dest='command', help='Processing command')

    # Clean outliers command
    clean_parser = subparsers.add_parser('clean', help='Remove outliers')
    clean_parser.add_argument('--input', type=str, required=True, help='Input CSV path')
    clean_parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    clean_parser.add_argument('--method', type=str, default='iqr', choices=['iqr', 'zscore'])
    clean_parser.add_argument('--threshold', type=float, default=1.5, help='IQR multiplier or z-score threshold')
    clean_parser.add_argument('--gaze_x_col', type=str, default='origin_gaze_x')
    clean_parser.add_argument('--gaze_y_col', type=str, default='origin_gaze_y')

    # Filter fixed points command
    filter_parser = subparsers.add_parser('filter', help='Filter fixed calibration points')
    filter_parser.add_argument('--input', type=str, required=True, help='Input CSV path')
    filter_parser.add_argument('--output', type=str, required=True, help='Output CSV path for non-fixed points')
    filter_parser.add_argument('--min_occurrences', type=int, default=2, help='Minimum occurrences for fixed points')

    # Split dataset command
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val/test')
    split_parser.add_argument('--input', type=str, required=True, help='Input CSV path')
    split_parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    split_parser.add_argument('--train_ratio', type=float, default=0.6)
    split_parser.add_argument('--val_ratio', type=float, default=0.2)
    split_parser.add_argument('--test_ratio', type=float, default=0.2)
    split_parser.add_argument('--random_seed', type=int, default=42)

    args = parser.parse_args()

    if args.command == 'clean':
        clean_outliers(args.input, args.output, args.method, args.threshold, args.gaze_x_col, args.gaze_y_col)
    elif args.command == 'filter':
        filter_fixed_points(args.input, args.output, args.min_occurrences)
    elif args.command == 'split':
        split_dataset(args.input, args.output_dir, args.train_ratio, args.val_ratio, args.test_ratio, args.random_seed)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
