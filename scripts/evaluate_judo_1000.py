"""
Evaluate trained multi-baseline neural refine model on JuDo-1000 test set.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "apps" / "neural_refine"))

from src.model import GazeDataset, build_model
from torch.utils.data import DataLoader


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


def evaluate_neural_model(model, dataset, coordinate_scale=100.0, device='cpu'):
    """
    Evaluate trained neural model on dataset.

    Returns:
        dict: Contains mean_l2, std_l2, median_l2, mae_x, mae_y, rmse_x, rmse_y
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_l2_distances = []
    all_mae_x = []
    all_mae_y = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            residuals_pred = model(inputs)

            # Convert to pixel space
            scale = coordinate_scale
            residuals_pred_px = residuals_pred * scale
            targets_px = targets * scale

            # For multi-baseline: prediction = origin_gaze + residual
            start_idx = batch_idx * loader.batch_size
            end_idx = start_idx + inputs.size(0)
            orig_gaze = dataset.orig_inputs_px[start_idx:end_idx]

            # Final prediction
            pred_gaze = orig_gaze + residuals_pred_px

            # True targets
            true_targets = dataset.targets_px[start_idx:end_idx]

            # Compute errors
            l2_dist = torch.sqrt(((pred_gaze - true_targets) ** 2).sum(dim=1))
            mae_x = torch.abs(pred_gaze[:, 0] - true_targets[:, 0])
            mae_y = torch.abs(pred_gaze[:, 1] - true_targets[:, 1])

            all_l2_distances.extend(l2_dist.cpu().numpy())
            all_mae_x.extend(mae_x.cpu().numpy())
            all_mae_y.extend(mae_y.cpu().numpy())

    all_l2_distances = np.array(all_l2_distances)
    all_mae_x = np.array(all_mae_x)
    all_mae_y = np.array(all_mae_y)

    return {
        'mean_l2': all_l2_distances.mean(),
        'std_l2': all_l2_distances.std(),
        'median_l2': np.median(all_l2_distances),
        'mae_x': all_mae_x.mean(),
        'mae_y': all_mae_y.mean(),
        'rmse_x': np.sqrt((all_mae_x ** 2).mean()),
        'rmse_y': np.sqrt((all_mae_y ** 2).mean()),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate multi-baseline model on JuDo-1000')
    parser.add_argument('--config', type=str,
                        default='apps/neural_refine/config/judo_1000.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/judo_1000/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str,
                        default='data/prepared/judo_1000_split/test.csv',
                        help='Path to test data')
    args = parser.parse_args()

    print('='*80)
    print('JuDo-1000 Multi-Baseline Neural Refine Evaluation')
    print('='*80)

    # Load config
    config_path = Path(args.config)
    print(f'\nLoading config from {config_path}')
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Check checkpoint exists
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f'Error: Checkpoint not found at {ckpt_path}')
        print('Please train the model first using: python apps/neural_refine/main.py --config apps/neural_refine/config/judo_1000.yaml')
        return

    # Load test data
    test_path = Path(args.data)
    print(f'\nLoading test data from {test_path}')
    test_df = pd.read_csv(test_path)
    print(f'Test samples: {len(test_df)}')

    # Build model
    print('\nBuilding model...')
    coordinate_scale = cfg['model']['coordinate_scale']

    # Load test dataset to get feature dimensions
    test_dataset = GazeDataset(
        test_path,
        coordinate_scale=coordinate_scale,
        normalize=cfg['data']['normalize'],
        model_type='multi_baseline',
        multi_baseline_features=cfg['data']['multi_baseline_features'],
        is_training=False
    )
    mb_feature_dim = test_dataset.mb_features.shape[1] if test_dataset.mb_features is not None else 0

    model = build_model(cfg['model'], hf_feature_dim=mb_feature_dim)

    # Load checkpoint
    print(f'Loading checkpoint from {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded model from epoch {checkpoint["epoch"]}')
    if 'best_val_l2' in checkpoint:
        print(f'Best validation L2: {checkpoint["best_val_l2"]:.2f} px')

    device = torch.device('cpu')
    model.to(device)

    # Evaluate neural model
    print('\n' + '='*80)
    print('Neural Model Evaluation')
    print('='*80)
    neural_metrics = evaluate_neural_model(model, test_dataset, coordinate_scale, device)
    print(f"\nNeural Model Results:")
    print(f"  L2:  {neural_metrics['mean_l2']:.2f} px (std: {neural_metrics['std_l2']:.2f}, median: {neural_metrics['median_l2']:.2f})")
    print(f"  MAE: ({neural_metrics['mae_x']:.2f}, {neural_metrics['mae_y']:.2f}) px")
    print(f"  RMSE: ({neural_metrics['rmse_x']:.2f}, {neural_metrics['rmse_y']:.2f}) px")

    # Evaluate baseline methods
    print('\n' + '='*80)
    print('Baseline Methods Evaluation')
    print('='*80)

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

    results = []
    for name, prefix in baselines:
        if f"{prefix}_x" in test_df.columns and f"{prefix}_y" in test_df.columns:
            metrics = compute_metrics(test_df, prefix)
            results.append({"Method": name, **metrics})
            print(f"\n{name}:")
            print(f"  L2:  {metrics['mean_l2']:.2f} px (std: {metrics['std_l2']:.2f}, median: {metrics['median_l2']:.2f})")
            print(f"  MAE: ({metrics['mae_x']:.2f}, {metrics['mae_y']:.2f}) px")
            print(f"  RMSE: ({metrics['rmse_x']:.2f}, {metrics['rmse_y']:.2f}) px")

    # Add neural model results
    results.append({"Method": "Neural Multi-Baseline", **neural_metrics})

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Sort by mean L2 error
    results_df = results_df.sort_values("mean_l2")

    # Save results
    output_path = Path('outputs/judo_1000_evaluation.csv')
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


if __name__ == '__main__':
    main()
