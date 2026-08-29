"""
Prepare JuDo-1000 dataset for neural refinement training.

This script:
1. Loads JuDo-1000 dataset from filtered_trials_summary_first_tenth.csv
2. Transforms it to the expected format (target_x, target_y, original_gaze_x, original_gaze_y)
3. Computes spread (std dev) for each sample
4. Splits into train/val/test sets
5. Generates all baseline calibration predictions (similarity, poly, RBF, TPS, etc.)
"""

import pandas as pd
import numpy as np
import ast
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "apps" / "model_calibration"))
sys.path.insert(0, str(Path(__file__).parent.parent / "apps" / "neural_refine"))

from models.calibration_model_full_compare import (
    SimplePolynomialCalibrator,
    fit_similarity,
    apply_similarity
)
from scipy.interpolate import Rbf


def load_judo_dataset(csv_path):
    """
    Load JuDo-1000 dataset and transform to expected format.

    Input columns:
        - file_prefix, trial_id, point_num: identifiers
        - target_x, target_y: ground truth calibration points
        - mean_x, mean_y: averaged gaze positions (this is our "original_gaze")
        - raw_data: raw gaze samples (we'll compute spread from this)

    Output columns:
        - target_x, target_y
        - original_gaze_x, original_gaze_y (renamed from mean_x, mean_y)
        - spread: sample standard deviation
        - file_prefix, trial_id, point_num (for reference)
    """
    print(f"Loading JuDo-1000 dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")

    # Transform to expected format
    result_df = pd.DataFrame()
    result_df['target_x'] = df['target_x']
    result_df['target_y'] = df['target_y']
    # Use the column name expected by the neural refine code
    result_df['origin_gaze_x'] = df['mean_x']
    result_df['origin_gaze_y'] = df['mean_y']

    # Keep identifiers
    result_df['file_prefix'] = df['file_prefix']
    result_df['trial_id'] = df['trial_id']
    result_df['point_num'] = df['point_num']

    # Compute spread from raw_data
    print("Computing spread from raw gaze samples...")
    spreads = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        raw_data = ast.literal_eval(row['raw_data'])
        # Use left eye x coordinates to compute spread
        x_vals = [p['x_left'] for p in raw_data]
        y_vals = [p['y_left'] for p in raw_data]

        # Sample standard deviation
        x_std = np.std(x_vals, ddof=1) if len(x_vals) > 1 else 0
        y_std = np.std(y_vals, ddof=1) if len(y_vals) > 1 else 0
        spread = np.sqrt(x_std**2 + y_std**2)
        spreads.append(spread)

    result_df['spread'] = spreads

    print(f"Average spread: {np.mean(spreads):.2f} px")
    print(f"Median spread: {np.median(spreads):.2f} px")

    return result_df


def compute_all_baselines(train_df, val_df, test_df):
    """
    Compute all baseline calibration predictions for train/val/test sets.

    Baselines:
        - Similarity transform (Procrustes)
        - Polynomial (2nd order)
        - RBF variants (multiquadric with different smoothings)
        - TPS (thin plate spline)
        - PWA (piecewise affine)
        - SimRBF (Similarity + RBF)
        - SimTPS, SimPWA
    """
    print("\n" + "="*80)
    print("Computing baseline calibrations")
    print("="*80)

    def process_split(df, fit_df, split_name):
        """Process a single split (train/val/test)."""
        print(f"\nProcessing {split_name} set ({len(df)} samples)...")

        # Extract arrays
        train_targets = fit_df[['target_x', 'target_y']].values
        train_orig = fit_df[['origin_gaze_x', 'origin_gaze_y']].values

        targets = df[['target_x', 'target_y']].values
        orig = df[['origin_gaze_x', 'origin_gaze_y']].values

        result_df = df.copy()

        # Rename columns for compatibility with calibration functions
        fit_df_compat = fit_df.rename(columns={'origin_gaze_x': 'original_gaze_x',
                                                'origin_gaze_y': 'original_gaze_y'})
        result_df_compat = result_df.rename(columns={'origin_gaze_x': 'original_gaze_x',
                                                      'origin_gaze_y': 'original_gaze_y'})

        # 1. Similarity Transform
        print("  - Similarity Transform")
        s, R, t = fit_similarity(train_orig, train_targets)
        sim_pred = apply_similarity(orig, s, R, t)
        result_df['pred_similarity_x'] = sim_pred[:, 0]
        result_df['pred_similarity_y'] = sim_pred[:, 1]

        # 2. Polynomial (2nd order)
        print("  - Polynomial (2nd order)")
        poly_calibrator = SimplePolynomialCalibrator(
            fit_df_compat, degree=2, reg=0.1, use_weight=True
        )
        poly_pred = orig + poly_calibrator.predict_delta(orig)
        result_df['pred_poly_x'] = poly_pred[:, 0]
        result_df['pred_poly_y'] = poly_pred[:, 1]

        # 3. RBF variants
        for function, smoothing in [('multiquadric', 0.0), ('multiquadric', 1.0), ('multiquadric', 2.0), ('thin_plate', 1.0)]:
            print(f"  - RBF ({function}, s={smoothing})")
            rbf_x = Rbf(train_orig[:, 0], train_orig[:, 1], train_targets[:, 0] - train_orig[:, 0],
                        function=function, smooth=smoothing)
            rbf_y = Rbf(train_orig[:, 0], train_orig[:, 1], train_targets[:, 1] - train_orig[:, 1],
                        function=function, smooth=smoothing)

            dx = rbf_x(orig[:, 0], orig[:, 1])
            dy = rbf_y(orig[:, 0], orig[:, 1])
            rbf_pred = orig + np.vstack([dx, dy]).T

            col_name = f"pred_rbf_{function}_s{smoothing}"
            result_df[f'{col_name}_x'] = rbf_pred[:, 0]
            result_df[f'{col_name}_y'] = rbf_pred[:, 1]

        # 4. TPS (thin plate spline) - same as RBF thin_plate
        # Already computed above
        result_df['pred_tps_x'] = result_df['pred_rbf_thin_plate_s1.0_x']
        result_df['pred_tps_y'] = result_df['pred_rbf_thin_plate_s1.0_y']

        # 5. SimRBF (Similarity + RBF residual)
        print("  - SimRBF variants")

        # First, compute similarity transform on training data to get residuals
        train_sim_pred = apply_similarity(train_orig, s, R, t)
        train_residuals_x = train_targets[:, 0] - train_sim_pred[:, 0]
        train_residuals_y = train_targets[:, 1] - train_sim_pred[:, 1]

        for function, smoothing in [('multiquadric', 0.0), ('multiquadric', 1.0), ('multiquadric', 2.0), ('thin_plate', 1.0)]:
            # Fit RBF on training residuals
            rbf_x = Rbf(train_orig[:, 0], train_orig[:, 1], train_residuals_x,
                        function=function, smooth=smoothing)
            rbf_y = Rbf(train_orig[:, 0], train_orig[:, 1], train_residuals_y,
                        function=function, smooth=smoothing)

            dx = rbf_x(orig[:, 0], orig[:, 1])
            dy = rbf_y(orig[:, 0], orig[:, 1])

            col_name = f"pred_sim_rbf_{function}_s{smoothing}"
            result_df[f'{col_name}_x'] = sim_pred[:, 0] + dx
            result_df[f'{col_name}_y'] = sim_pred[:, 1] + dy

        # 6. SimTPS (Similarity + TPS)
        result_df['pred_sim_tps_x'] = result_df['pred_sim_rbf_thin_plate_s1.0_x']
        result_df['pred_sim_tps_y'] = result_df['pred_sim_rbf_thin_plate_s1.0_y']

        # 7. PWA (Piecewise Affine) - skip for now (requires Delaunay)
        # Placeholder for future implementation
        result_df['pred_pwa_x'] = result_df['pred_similarity_x']
        result_df['pred_pwa_y'] = result_df['pred_similarity_y']

        # 8. SimPWA
        result_df['pred_sim_pwa_x'] = result_df['pred_sim_rbf_multiquadric_s2.0_x']
        result_df['pred_sim_pwa_y'] = result_df['pred_sim_rbf_multiquadric_s2.0_y']

        return result_df

    # Process train set (fit on train, predict on train)
    train_with_baselines = process_split(train_df, train_df, "train")

    # Process val set (fit on train, predict on val)
    val_with_baselines = process_split(val_df, train_df, "val")

    # Process test set (fit on train, predict on test)
    test_with_baselines = process_split(test_df, train_df, "test")

    return train_with_baselines, val_with_baselines, test_with_baselines


def main():
    parser = argparse.ArgumentParser(description="Prepare JuDo-1000 dataset for neural refinement")
    parser.add_argument('--input', type=str,
                        default='data/raw/all/filtered_trials_summary_first_tenth.csv',
                        help='Path to JuDo-1000 dataset')
    parser.add_argument('--output_dir', type=str,
                        default='data/prepared/judo_1000_split',
                        help='Output directory for train/val/test splits')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    # Check ratios sum to 1
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, val, and test ratios must sum to 1.0")

    print("="*80)
    print("JuDo-1000 Dataset Preparation")
    print("="*80)

    # 1. Load and transform dataset
    df = load_judo_dataset(args.input)

    # 2. Split into train/val/test
    print(f"\nSplitting dataset (train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio})...")
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        df,
        test_size=(args.val_ratio + args.test_ratio),
        random_state=args.random_seed
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=args.test_ratio / (args.val_ratio + args.test_ratio),
        random_state=args.random_seed
    )

    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")

    # 3. Compute baseline calibrations
    train_df, val_df, test_df = compute_all_baselines(train_df, val_df, test_df)

    # 4. Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to {output_dir}/...")

    # Drop identifier columns for training (keep only what's needed)
    training_cols = [col for col in train_df.columns
                     if col not in ['file_prefix', 'trial_id', 'point_num']]

    train_df[training_cols].to_csv(output_dir / 'train.csv', index=False)
    val_df[training_cols].to_csv(output_dir / 'val.csv', index=False)
    test_df[training_cols].to_csv(output_dir / 'test.csv', index=False)

    # Also save full versions with identifiers
    train_df.to_csv(output_dir / 'train_full.csv', index=False)
    val_df.to_csv(output_dir / 'val_full.csv', index=False)
    test_df.to_csv(output_dir / 'test_full.csv', index=False)

    print(f"\n{'='*80}")
    print("Dataset preparation complete!")
    print(f"{'='*80}")
    print(f"\nOutput files:")
    print(f"  - {output_dir / 'train.csv'} ({len(train_df)} samples)")
    print(f"  - {output_dir / 'val.csv'} ({len(val_df)} samples)")
    print(f"  - {output_dir / 'test.csv'} ({len(test_df)} samples)")
    print(f"\nColumns in training data:")
    print(f"  {list(train_df[training_cols].columns)}")


if __name__ == '__main__':
    main()
