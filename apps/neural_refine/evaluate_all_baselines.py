"""
Evaluate all calibration methods on any dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def compute_metrics(df, pred_prefix):
    """Compute error metrics for a prediction method."""
    pred_x = df[f"{pred_prefix}_x"].values
    pred_y = df[f"{pred_prefix}_y"].values
    target_x = df["target_x"].values
    target_y = df["target_y"].values

    error_x = pred_x - target_x
    error_y = pred_y - target_y
    l2_error = np.sqrt(error_x**2 + error_y**2)

    return {
        "mean_l2": l2_error.mean(),
        "std_l2": l2_error.std(),
        "median_l2": np.median(l2_error),
        "min_l2": l2_error.min(),
        "max_l2": l2_error.max(),
        "mae_x": np.abs(error_x).mean(),
        "mae_y": np.abs(error_y).mean(),
        "rmse_x": np.sqrt((error_x**2).mean()),
        "rmse_y": np.sqrt((error_y**2).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate all calibration methods")
    parser.add_argument("--data", type=str,
                        default="../../data/raw/all/all_trials_model_predictions_0111_without_s1.csv",
                        help="Path to dataset CSV")
    parser.add_argument("--output", type=str,
                        default="../../outputs/baseline_comparison.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    data_path = args.data

    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Total samples: {len(df)}")

    # Rename columns for consistency
    df = df.rename(columns={"origin_gaze_x": "original_gaze_x", "origin_gaze_y": "original_gaze_y"})

    # Define all baseline methods to evaluate
    baselines = [
        ("Original Gaze", "original_gaze"),
        ("Similarity", "pred_similarity"),
        ("Polynomial", "pred_poly"),
        ("RBF Thin Plate s1.0", "pred_rbf_thin_plate_s1.0"),
        ("RBF Multiquadric s0.0", "pred_rbf_multiquadric_s0.0"),
        ("RBF Multiquadric s1.0", "pred_rbf_multiquadric_s1.0"),
        ("RBF Multiquadric s2.0", "pred_rbf_multiquadric_s2.0"),
        ("TPS", "pred_tps"),
        ("PWA", "pred_pwa"),
        ("GPR", "pred_gpr"),
        ("SimRBF Thin Plate s1.0", "pred_sim_rbf_thin_plate_s1.0"),
        ("SimRBF Multiquadric s0.0", "pred_sim_rbf_multiquadric_s0.0"),
        ("SimRBF Multiquadric s1.0", "pred_sim_rbf_multiquadric_s1.0"),
        ("SimRBF Multiquadric s2.0", "pred_sim_rbf_multiquadric_s2.0"),
        ("SimTPS", "pred_sim_tps"),
        ("SimPWA", "pred_sim_pwa"),
        ("SimGPR", "pred_sim_gpr"),
    ]

    print("\n" + "="*80)
    print("Evaluating All Calibration Methods")
    print("="*80)

    results = []
    for name, prefix in baselines:
        if f"{prefix}_x" in df.columns and f"{prefix}_y" in df.columns:
            metrics = compute_metrics(df, prefix)
            results.append({"Method": name, **metrics})
            print(f"\n{name}:")
            print(f"  L2:  {metrics['mean_l2']:.2f} px (std: {metrics['std_l2']:.2f}, median: {metrics['median_l2']:.2f})")
            print(f"  MAE: ({metrics['mae_x']:.2f}, {metrics['mae_y']:.2f}) px")
            print(f"  RMSE: ({metrics['rmse_x']:.2f}, {metrics['rmse_y']:.2f}) px")
        else:
            print(f"\n{name}: SKIPPED (columns not found)")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Sort by mean L2 error
    results_df = results_df.sort_values("mean_l2")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to {output_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("Summary Table (Sorted by L2 Error)")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Method':<30} {'L2 (px)':<12} {'vs Original':<12}")
    print(f"{'-'*80}")
    original_l2 = results_df[results_df["Method"] == "Original Gaze"]["mean_l2"].values[0]
    for idx, row in results_df.iterrows():
        rank = results_df.index.get_loc(idx) + 1
        improvement = (1 - row["mean_l2"]/original_l2) * 100
        print(f"{rank:<5} {row['Method']:<30} {row['mean_l2']:<12.2f} {improvement:>+10.1f}%")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
