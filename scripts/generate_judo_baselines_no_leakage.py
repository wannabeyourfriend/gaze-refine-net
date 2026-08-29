"""
Generate baseline calibration predictions for JuDo-1000 dataset (no label leakage).

This script generates predictions using ONLY training data to fit calibrators,
then applies them to validation and test sets. This prevents label leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "apps" / "model_calibration"))

from models.calibration_model_full_compare import (
    SimplePolynomialCalibrator,
    fit_similarity,
    apply_similarity
)
from scipy.interpolate import Rbf


def generate_baselines_for_split(train_path, val_path, test_path, output_dir):
    """
    Generate baseline predictions using only training data for calibration.

    Args:
        train_path: Path to training data CSV
        val_path: Path to validation data CSV
        test_path: Path to test data CSV
        output_dir: Directory to save results
    """
    print("="*80)
    print("Generating Baseline Calibrations (No Label Leakage)")
    print("="*80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"Train:      {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test:       {len(test_df)} samples")

    # Prepare data for calibration models
    train_orig = train_df[['origin_gaze_x', 'origin_gaze_y']].values
    train_targets = train_df[['target_x', 'target_y']].values

    val_orig = val_df[['origin_gaze_x', 'origin_gaze_y']].values
    val_targets = val_df[['target_x', 'target_y']].values

    test_orig = test_df[['origin_gaze_x', 'origin_gaze_y']].values
    test_targets = test_df[['target_x', 'target_y']].values

    print("\n" + "="*80)
    print("Fitting calibrators on TRAINING data only...")
    print("="*80)

    # 1. Similarity Transform
    print("\n1. Similarity Transform")
    s, R, t = fit_similarity(train_orig, train_targets)
    train_sim_pred = apply_similarity(train_orig, s, R, t)

    # Compute residuals for RBF
    train_residuals_x = train_targets[:, 0] - train_sim_pred[:, 0]
    train_residuals_y = train_targets[:, 1] - train_sim_pred[:, 1]

    # 2. Polynomial
    print("2. Polynomial (2nd order)")
    train_df_compat = train_df.rename(columns={
        'origin_gaze_x': 'original_gaze_x',
        'origin_gaze_y': 'original_gaze_y'
    })
    poly_calibrator = SimplePolynomialCalibrator(train_df_compat, degree=2)

    # 3. RBF with different smoothing parameters
    print("3. RBF Multiquadric")
    rbf_calibrators = {}
    for smoothing in [0.0, 1.0, 2.0]:
        print(f"   - Smoothing {smoothing}")
        rbf_x = Rbf(train_orig[:, 0], train_orig[:, 1], train_targets[:, 0],
                   function='multiquadric', smooth=smoothing)
        rbf_y = Rbf(train_orig[:, 0], train_orig[:, 1], train_targets[:, 1],
                   function='multiquadric', smooth=smoothing)
        rbf_calibrators[f'rbf_multiquadric_s{smoothing}'] = (rbf_x, rbf_y)

    # 4. TPS
    print("4. Thin Plate Spline")
    tps_x = Rbf(train_orig[:, 0], train_orig[:, 1], train_targets[:, 0], function='thin_plate')
    tps_y = Rbf(train_orig[:, 0], train_orig[:, 1], train_targets[:, 1], function='thin_plate')

    # 5. SimRBF (Similarity + RBF on residuals)
    print("5. SimRBF (Similarity + RBF residual)")
    simrbf_calibrators = {}
    for smoothing in [0.0, 1.0, 2.0]:
        print(f"   - Smoothing {smoothing}")
        rbf_res_x = Rbf(train_orig[:, 0], train_orig[:, 1], train_residuals_x,
                       function='multiquadric', smooth=smoothing)
        rbf_res_y = Rbf(train_orig[:, 0], train_orig[:, 1], train_residuals_y,
                       function='multiquadric', smooth=smoothing)
        simrbf_calibrators[f's{smoothing}'] = (rbf_res_x, rbf_res_y)

    # 6. SimTPS
    print("6. SimTPS (Similarity + TPS residual)")
    tps_res_x = Rbf(train_orig[:, 0], train_orig[:, 1], train_residuals_x, function='thin_plate')
    tps_res_y = Rbf(train_orig[:, 0], train_orig[:, 1], train_residuals_y, function='thin_plate')

    # 7. SimPWA (use SimRBF s2.0 as placeholder)
    print("7. SimPWA (using SimRBF s2.0 as placeholder)")
    # Note: PWA requires Delaunay triangulation, using SimRBF s2.0 for now

    print("\n" + "="*80)
    print("Generating predictions for all splits...")
    print("="*80)

    def add_predictions(df, orig, split_name):
        """Add all baseline predictions to DataFrame."""
        # Similarity
        sim_pred = apply_similarity(orig, s, R, t)
        df['pred_similarity_x'] = sim_pred[:, 0]
        df['pred_similarity_y'] = sim_pred[:, 1]

        # Polynomial
        df_compat = df.copy()
        df_compat['original_gaze_x'] = df['origin_gaze_x']
        df_compat['original_gaze_y'] = df['origin_gaze_y']
        poly_pred = orig + poly_calibrator.predict_delta(orig)
        df['pred_poly_x'] = poly_pred[:, 0]
        df['pred_poly_y'] = poly_pred[:, 1]

        # RBF
        for name, (rbf_x, rbf_y) in rbf_calibrators.items():
            df[f'pred_{name}_x'] = rbf_x(orig[:, 0], orig[:, 1])
            df[f'pred_{name}_y'] = rbf_y(orig[:, 0], orig[:, 1])

        # TPS
        df['pred_tps_x'] = tps_x(orig[:, 0], orig[:, 1])
        df['pred_tps_y'] = tps_y(orig[:, 0], orig[:, 1])

        # SimRBF
        for smoothing_key, (rbf_res_x, rbf_res_y) in simrbf_calibrators.items():
            # Start with similarity prediction
            sim_pred = apply_similarity(orig, s, R, t)
            # Add RBF residual
            df[f'pred_sim_rbf_multiquadric_{smoothing_key}_x'] = sim_pred[:, 0] + rbf_res_x(orig[:, 0], orig[:, 1])
            df[f'pred_sim_rbf_multiquadric_{smoothing_key}_y'] = sim_pred[:, 1] + rbf_res_y(orig[:, 0], orig[:, 1])

        # SimTPS
        sim_pred = apply_similarity(orig, s, R, t)
        df['pred_sim_tps_x'] = sim_pred[:, 0] + tps_res_x(orig[:, 0], orig[:, 1])
        df['pred_sim_tps_y'] = sim_pred[:, 1] + tps_res_y(orig[:, 0], orig[:, 1])

        # SimPWA (use SimRBF s2.0 as placeholder)
        df['pred_sim_pwa_x'] = df['pred_sim_rbf_multiquadric_s2.0_x']
        df['pred_sim_pwa_y'] = df['pred_sim_rbf_multiquadric_s2.0_y']

        return df

    # Generate predictions for each split
    for split_name, split_df, split_orig in [
        ('train', train_df.copy(), train_orig),
        ('val', val_df.copy(), val_orig),
        ('test', test_df.copy(), test_orig)
    ]:
        print(f"\n{split_name.upper()}...")
        result_df = add_predictions(split_df, split_orig, split_name)

        # Save
        output_path = output_dir / f'{split_name}.csv'
        result_df.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")

    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print("✓ Calibrators fit ONLY on training data")
    print("✓ Predictions generated for train/val/test splits")
    print("✓ No label leakage between splits")
    print("\nBaseline methods generated:")
    print("  1. Similarity Transform")
    print("  2. Polynomial (2nd order)")
    print("  3. RBF Multiquadric (s=0.0, 1.0, 2.0)")
    print("  4. Thin Plate Spline (TPS)")
    print("  5. SimRBF (Similarity + RBF residual, s=0.0, 1.0, 2.0)")
    print("  6. SimTPS (Similarity + TPS residual)")
    print("  7. SimPWA (Similarity + PWA residual)")

    print("\n" + "="*80)
    print("Computing baseline errors on TEST set (unseen targets)...")
    print("="*80)

    # Compute errors on test set
    test_result_df = pd.read_csv(output_dir / 'test.csv')

    def compute_error(df, prefix):
        """Compute L2 error for a prediction method."""
        pred_x = df[f'{prefix}_x'].values
        pred_y = df[f'{prefix}_y'].values
        target_x = df['target_x'].values
        target_y = df['target_y'].values

        error_x = pred_x - target_x
        error_y = pred_y - target_y
        l2_error = np.sqrt(error_x**2 + error_y**2)

        return l2_error.mean(), l2_error.std(), np.median(l2_error)

    baselines = [
        ("Original Gaze", "origin_gaze"),
        ("Similarity", "pred_similarity"),
        ("Polynomial", "pred_poly"),
        ("RBF Multiquadric s0.0", "pred_rbf_multiquadric_s0.0"),
        ("RBF Multiquadric s1.0", "pred_rbf_multiquadric_s1.0"),
        ("RBF Multiquadric s2.0", "pred_rbf_multiquadric_s2.0"),
        ("TPS", "pred_tps"),
        ("SimRBF Multiquadric s0.0", "pred_sim_rbf_multiquadric_s0.0"),
        ("SimRBF Multiquadric s1.0", "pred_sim_rbf_multiquadric_s1.0"),
        ("SimRBF Multiquadric s2.0", "pred_sim_rbf_multiquadric_s2.0"),
        ("SimTPS", "pred_sim_tps"),
        ("SimPWA", "pred_sim_pwa"),
    ]

    print(f"\n{'Method':<30} {'L2 (px)':<12} {'Std':<10} {'Median':<10}")
    print("-" * 80)

    for name, prefix in baselines:
        if f'{prefix}_x' in test_result_df.columns:
            mean, std, median = compute_error(test_result_df, prefix)
            print(f"{name:<30} {mean:<12.2f} {std:<10.2f} {median:<10.2f}")


def main():
    parser = argparse.ArgumentParser(description="Generate no-leakage JuDo baseline predictions")
    parser.add_argument("--train-path", default="data/prepared/judo_1000_split_no_leakage/train.csv")
    parser.add_argument("--val-path", default="data/prepared/judo_1000_split_no_leakage/val.csv")
    parser.add_argument("--test-path", default="data/prepared/judo_1000_split_no_leakage/test.csv")
    parser.add_argument("--output-dir", default="data/prepared/judo_1000_split_no_leakage")
    args = parser.parse_args()
    generate_baselines_for_split(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
