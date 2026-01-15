"""
Evaluate neural refine model on the large all_trials dataset.
"""
import sys
from pathlib import Path

import torch
import yaml
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GazeDataset, build_model


def evaluate_on_all_trials(config_path, checkpoint_path, data_path, output_csv):
    """Load model and run inference on all_trials dataset."""

    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load all_trials dataset
    print(f"Loading data from {data_path}")
    df_all = pd.read_csv(data_path)
    print(f"Total samples: {len(df_all)}")

    # Rename columns to match expected format
    df_all = df_all.rename(columns={"origin_gaze_x": "original_gaze_x", "origin_gaze_y": "original_gaze_y"})

    # Add sim_rbf_gaze column if not exists
    # Try different baselines in order of preference
    if "sim_rbf_gaze_x" not in df_all.columns:
        if "pred_sim_rbf_multiquadric_s1.0_x" in df_all.columns:
            df_all["sim_rbf_gaze_x"] = df_all["pred_sim_rbf_multiquadric_s1.0_x"]
            df_all["sim_rbf_gaze_y"] = df_all["pred_sim_rbf_multiquadric_s1.0_y"]
        elif "pred_sim_rbf_multiquadric_s2.0_x" in df_all.columns:
            df_all["sim_rbf_gaze_x"] = df_all["pred_sim_rbf_multiquadric_s2.0_x"]
            df_all["sim_rbf_gaze_y"] = df_all["pred_sim_rbf_multiquadric_s2.0_y"]
        else:
            raise ValueError("Cannot find sim_rbf_gaze columns or suitable replacement")

    # Create temporary CSV file in the expected format
    temp_csv = Path("/tmp/all_trials_temp.csv")
    required_cols = ["target_x", "target_y", "original_gaze_x", "original_gaze_y", "sim_rbf_gaze_x", "sim_rbf_gaze_y"]

    # Add all other baseline predictions
    for baseline in cfg["data"]["multi_baseline_features"]["baselines"]:
        if f"{baseline}_x" in df_all.columns:
            required_cols.extend([f"{baseline}_x", f"{baseline}_y"])

    df_temp = df_all[required_cols]
    df_temp.to_csv(temp_csv, index=False)

    # Create dataset
    test_ds = GazeDataset(
        str(temp_csv),
        coordinate_scale=cfg["model"]["coordinate_scale"],
        normalize=cfg["data"]["normalize"],
        model_type=cfg["model"]["type"],
        augmentation=None,
        sim_rbf_perturbation=None,
        high_freq_features=cfg["data"].get("high_freq_features"),
        multi_baseline_features=cfg["data"].get("multi_baseline_features"),
        is_training=False,
    )

    # Build model and load checkpoint
    mb_feat_dim = test_ds.mb_features.shape[1] if test_ds.mb_features is not None else 0
    hf_feat_dim = test_ds.hf_features.shape[1] if test_ds.hf_features is not None else 0
    model = build_model(cfg["model"], hf_feature_dim=mb_feat_dim + hf_feat_dim)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Run inference
    predictions = []
    scale = cfg["model"]["coordinate_scale"] if cfg["data"]["normalize"] else 1.0

    with torch.no_grad():
        for i in range(len(test_ds)):
            inputs, _ = test_ds[i]
            pred_residual = model(inputs.unsqueeze(0)).squeeze(0)
            pred_residual_px = pred_residual * scale

            # Get original gaze
            orig_px = test_ds.orig_inputs_px[i]
            target_px = test_ds.targets_px[i]

            # Final prediction: original + predicted_residual
            pred_gaze_px = orig_px + pred_residual_px

            # Calculate error
            error_px = pred_gaze_px - target_px
            error_distance = torch.linalg.norm(error_px).item()

            predictions.append({
                "index": i,
                "target_x": float(target_px[0]),
                "target_y": float(target_px[1]),
                "original_gaze_x": float(orig_px[0]),
                "original_gaze_y": float(orig_px[1]),
                "pred_residual_x": float(pred_residual_px[0]),
                "pred_residual_y": float(pred_residual_px[1]),
                "pred_gaze_x": float(pred_gaze_px[0]),
                "pred_gaze_y": float(pred_gaze_px[1]),
                "error_x": float(error_px[0]),
                "error_y": float(error_px[1]),
                "error_distance": error_distance,
            })

    # Save predictions
    df_pred = pd.DataFrame(predictions)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(output_path, index=False)

    # Compute summary statistics
    mean_error = df_pred["error_distance"].mean()
    std_error = df_pred["error_distance"].std()
    median_error = df_pred["error_distance"].median()
    min_error = df_pred["error_distance"].min()
    max_error = df_pred["error_distance"].max()

    # MAE per dimension
    mae_x = df_pred["error_x"].abs().mean()
    mae_y = df_pred["error_y"].abs().mean()

    # RMSE per dimension
    rmse_x = np.sqrt((df_pred["error_x"]**2).mean())
    rmse_y = np.sqrt((df_pred["error_y"]**2).mean())

    print(f"\n{'='*80}")
    print(f"Neural Refine - All Trials Evaluation Results")
    print(f"{'='*80}")
    print(f"Total samples: {len(df_pred)}")
    print(f"\nOverall Error:")
    print(f"  Mean L2:   {mean_error:.4f} px")
    print(f"  Std L2:    {std_error:.4f} px")
    print(f"  Median L2: {median_error:.4f} px")
    print(f"  Min L2:    {min_error:.4f} px")
    print(f"  Max L2:    {max_error:.4f} px")
    print(f"\nPer-Dimension MAE:")
    print(f"  X: {mae_x:.4f} px")
    print(f"  Y: {mae_y:.4f} px")
    print(f"\nPer-Dimension RMSE:")
    print(f"  X: {rmse_x:.4f} px")
    print(f"  Y: {rmse_y:.4f} px")

    # Compare with baselines
    orig_error_x = (df_all["original_gaze_x"].values - df_all["target_x"].values)
    orig_error_y = (df_all["original_gaze_y"].values - df_all["target_y"].values)
    orig_l2 = np.sqrt(orig_error_x**2 + orig_error_y**2)
    orig_mae_x = np.abs(orig_error_x).mean()
    orig_mae_y = np.abs(orig_error_y).mean()
    orig_rmse_x = np.sqrt((orig_error_x**2).mean())
    orig_rmse_y = np.sqrt((orig_error_y**2).mean())

    print(f"\n{'='*80}")
    print("Baseline Comparison:")
    print(f"{'='*80}")
    print(f"\nOriginal Gaze:")
    print(f"  L2:  {orig_l2.mean():.2f} px (std: {orig_l2.std():.2f})")
    print(f"  MAE: ({orig_mae_x:.2f}, {orig_mae_y:.2f}) px")
    print(f"  RMSE: ({orig_rmse_x:.2f}, {orig_rmse_y:.2f}) px")

    # SimRBF s1.0 (best baseline) - try different baselines
    if "pred_sim_rbf_multiquadric_s1.0_x" in df_all.columns:
        sim_rbf_col = "pred_sim_rbf_multiquadric_s1.0"
        sim_rbf_name = "SimRBF Multiquadric s1.0"
    elif "pred_sim_rbf_multiquadric_s2.0_x" in df_all.columns:
        sim_rbf_col = "pred_sim_rbf_multiquadric_s2.0"
        sim_rbf_name = "SimRBF Multiquadric s2.0"
    else:
        sim_rbf_col = "sim_rbf_gaze"
        sim_rbf_name = "SimRBF Gaze"

    sim_rbf_error_x = (df_all[f"{sim_rbf_col}_x"].values - df_all["target_x"].values)
    sim_rbf_error_y = (df_all[f"{sim_rbf_col}_y"].values - df_all["target_y"].values)
    sim_rbf_l2 = np.sqrt(sim_rbf_error_x**2 + sim_rbf_error_y**2)
    sim_rbf_mae_x = np.abs(sim_rbf_error_x).mean()
    sim_rbf_mae_y = np.abs(sim_rbf_error_y).mean()
    sim_rbf_rmse_x = np.sqrt((sim_rbf_error_x**2).mean())
    sim_rbf_rmse_y = np.sqrt((sim_rbf_error_y**2).mean())

    print(f"\n{sim_rbf_name} (Best Baseline):")
    print(f"  L2:  {sim_rbf_l2.mean():.2f} px (std: {sim_rbf_l2.std():.2f})")
    print(f"  MAE: ({sim_rbf_mae_x:.2f}, {sim_rbf_mae_y:.2f}) px")
    print(f"  RMSE: ({sim_rbf_rmse_x:.2f}, {sim_rbf_rmse_y:.2f}) px")

    print(f"\nNeural Refine (Small Network):")
    print(f"  L2:  {mean_error:.2f} px (std: {std_error:.2f})")
    print(f"  MAE: ({mae_x:.2f}, {mae_y:.2f}) px")
    print(f"  RMSE: ({rmse_x:.2f}, {rmse_y:.2f}) px")

    print(f"\n{'='*80}")
    print("Improvement vs Baselines:")
    print(f"{'='*80}")
    print(f"  vs Original Gaze:  {(1 - mean_error/orig_l2.mean())*100:+.1f}%")
    print(f"  vs {sim_rbf_name}:    {(1 - mean_error/sim_rbf_l2.mean())*100:+.1f}%")

    print(f"\nPredictions saved to: {output_path}")
    print(f"{'='*80}\n")

    return df_pred


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate neural refine on all_trials dataset")
    parser.add_argument("--config", type=str, default="config/multi_baseline_s1_small.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="../../checkpoints/multi_baseline_s1_small/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--data", type=str, default="../../data/raw/all/all_trials_model_predictions_0111_without_s1.csv", help="Path to all_trials data")
    parser.add_argument("--output", type=str, default="../../outputs/all_trials_neural_refine_predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    evaluate_on_all_trials(args.config, args.checkpoint, args.data, args.output)
