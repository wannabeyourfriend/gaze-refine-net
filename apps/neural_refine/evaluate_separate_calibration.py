"""
Evaluate multi-baseline neural refine model on separate calibration datasets.

This script evaluates the model trained on 12-point calibration data
across all calibration point configurations (4, 6, 8, 10, 12, 14, 16, 18 points)
to assess performance with varying calibration data sizes.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from src.model import GazeDataset, build_model
from torch.utils.data import DataLoader


def evaluate_on_dataset(model, dataset, coordinate_scale=100.0, device='cpu'):
    """
    Evaluate model on a dataset.

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

            # For multi-baseline: prediction = original_gaze + residual
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


def get_baseline_errors(csv_path):
    """Compute baseline errors from original gaze and SimRBF."""
    df = pd.read_csv(csv_path)

    target_x = df['target_x'].values
    target_y = df['target_y'].values

    results = {'original_mean_l2': None, 'original_median_l2': None,
               'sim_rbf_mean_l2': None, 'sim_rbf_median_l2': None}

    # Original gaze error
    orig_gaze_x = df['origin_gaze_x'].values
    orig_gaze_y = df['origin_gaze_y'].values
    orig_l2 = np.sqrt((orig_gaze_x - target_x)**2 + (orig_gaze_y - target_y)**2)
    results['original_mean_l2'] = orig_l2.mean()
    results['original_median_l2'] = np.median(orig_l2)

    # SimRBF baseline error (s2.0 is the standard)
    sim_rbf_x = df['pred_sim_rbf_multiquadric_s2.0_x'].values
    sim_rbf_y = df['pred_sim_rbf_multiquadric_s2.0_y'].values
    sim_rbf_l2 = np.sqrt((sim_rbf_x - target_x)**2 + (sim_rbf_y - target_y)**2)
    results['sim_rbf_mean_l2'] = sim_rbf_l2.mean()
    results['sim_rbf_median_l2'] = np.median(sim_rbf_l2)

    # Other baselines available in the dataset
    baseline_methods = [
        ('pred_similarity', 'Similarity'),
        ('pred_poly', 'Polynomial'),
        ('pred_tps', 'TPS'),
        ('pred_pwa', 'PWA'),
        ('pred_sim_tps', 'SimTPS'),
        ('pred_sim_pwa', 'SimPWA'),
    ]

    for col_prefix, name in baseline_methods:
        col_x = f'{col_prefix}_x'
        col_y = f'{col_prefix}_y'
        if col_x in df.columns and col_y in df.columns:
            pred_x = df[col_x].values
            pred_y = df[col_y].values
            l2 = np.sqrt((pred_x - target_x)**2 + (pred_y - target_y)**2)
            results[f'{name.lower()}_mean_l2'] = l2.mean()
            results[f'{name.lower()}_median_l2'] = np.median(l2)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate multi-baseline model on separate calibration datasets')
    parser.add_argument('--data_dir', type=str, default='../../data/seperate_calibration_filtered',
                        help='Path to calibration datasets directory')
    parser.add_argument('--output_suffix', type=str, default='',
                        help='Suffix for output files (e.g., "_cleaned" for cleaned data)')
    args = parser.parse_args()

    print('='*80)
    print('Multi-Baseline Neural Refine Evaluation on Separate Calibration Datasets')
    print('='*80)

    # Config paths
    config_path = Path('config/multi_baseline_s1_small_12.yaml')
    ckpt_path = Path('../../checkpoints/multi_baseline_s1_small_12/best_model.pt')
    data_root = Path(args.data_dir)
    coordinate_scale = 100.0

    # Load config
    print(f'\nLoading config from {config_path}')
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Check checkpoint exists
    if not ckpt_path.exists():
        print(f'Error: Checkpoint not found at {ckpt_path}')
        return

    # Build model
    print('\nBuilding model...')
    # Load any dataset to get feature dimensions
    temp_dataset = GazeDataset(
        cfg['data']['train_data_path'],
        coordinate_scale=coordinate_scale,
        normalize=cfg['data']['normalize'],
        model_type='multi_baseline',
        multi_baseline_features=cfg['data']['multi_baseline_features'],
        is_training=False
    )
    mb_feature_dim = temp_dataset.mb_features.shape[1] if temp_dataset.mb_features is not None else 0

    model = build_model(cfg['model'], hf_feature_dim=mb_feature_dim)

    # Load checkpoint
    print(f'Loading checkpoint from {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded model from epoch {checkpoint["epoch"]}')

    device = torch.device('cpu')
    model.to(device)

    # Find all calibration point configurations
    calibration_configs = sorted(data_root.iterdir(), key=lambda x: x.name)

    # Group by number of points
    config_groups = defaultdict(list)
    for config_dir in calibration_configs:
        if config_dir.is_dir():
            # Extract number of points from directory name (e.g., "12points_1")
            parts = config_dir.name.split('_')
            if parts[0].replace('points', '').isdigit():
                num_points = int(parts[0].replace('points', ''))
                config_groups[num_points].append(config_dir)

    # Sort by number of points
    sorted_configs = sorted(config_groups.items())

    print(f'\nFound {len(sorted_configs)} calibration point configurations')
    for num_points, dirs in sorted_configs:
        print(f'  {num_points} points: {len(dirs)} dataset(s)')

    # Evaluate on each configuration
    results = []

    print('\n' + '='*80)
    print('Running Evaluations')
    print('='*80)

    for num_points, config_dirs in sorted_configs:
        print(f'\n--- Evaluating {num_points}-point calibration ---')

        for config_dir in config_dirs:
            csv_path = config_dir / 'all_trials_model_predictions_0111.csv'

            if not csv_path.exists():
                print(f'  Warning: CSV not found at {csv_path}')
                continue

            print(f'  Dataset: {config_dir.name}')

            # Load dataset
            try:
                dataset = GazeDataset(
                    csv_path,
                    coordinate_scale=coordinate_scale,
                    normalize=cfg['data']['normalize'],
                    model_type='multi_baseline',
                    multi_baseline_features=cfg['data']['multi_baseline_features'],
                    is_training=False
                )
            except Exception as e:
                print(f'    Error loading dataset: {e}')
                continue

            # Evaluate model
            neural_metrics = evaluate_on_dataset(model, dataset, coordinate_scale, device)

            # Get baseline errors
            baseline_metrics = get_baseline_errors(csv_path)

            # Compute improvements
            orig_improvement = (baseline_metrics['original_mean_l2'] - neural_metrics['mean_l2']) / baseline_metrics['original_mean_l2'] * 100
            sim_improvement = (baseline_metrics['sim_rbf_mean_l2'] - neural_metrics['mean_l2']) / baseline_metrics['sim_rbf_mean_l2'] * 100

            # Store results
            result = {
                'num_points': num_points,
                'dataset_name': config_dir.name,
                'num_samples': len(dataset),
                **neural_metrics,
                **baseline_metrics,
                'orig_improvement_pct': orig_improvement,
                'sim_improvement_pct': sim_improvement,
            }
            results.append(result)

            # Print summary
            print(f'    Samples: {len(dataset)}')
            print(f'    Neural Mean L2:   {neural_metrics["mean_l2"]:.2f} px (±{neural_metrics["std_l2"]:.2f})')
            print(f'    Original Mean L2: {baseline_metrics["original_mean_l2"]:.2f} px')
            print(f'    SimRBF Mean L2:   {baseline_metrics["sim_rbf_mean_l2"]:.2f} px')
            print(f'    Improvement vs Original: {orig_improvement:+.1f}%')
            print(f'    Improvement vs SimRBF:   {sim_improvement:+.1f}%')

    # Save results to CSV
    results_df = pd.DataFrame(results)
    output_path = Path(f'../../outputs/separate_calibration_evaluation{args.output_suffix}.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f'\nResults saved to {output_path}')

    # Compute aggregated statistics by number of points
    print('\n' + '='*80)
    print('Aggregated Results by Calibration Point Count')
    print('='*80)

    summary_rows = []
    for num_points in sorted(config_groups.keys()):
        subset = results_df[results_df['num_points'] == num_points]
        if len(subset) == 0:
            continue

        print(f'\n{num_points} points ({len(subset)} datasets):')
        print(f'  Neural Mean L2:     {subset["mean_l2"].mean():.2f} ± {subset["mean_l2"].std():.2f} px')
        print(f'  Neural Median L2:   {subset["median_l2"].mean():.2f} ± {subset["median_l2"].std():.2f} px')
        print(f'  Original Mean L2:   {subset["original_mean_l2"].mean():.2f} ± {subset["original_mean_l2"].std() if len(subset) > 1 else 0:.2f} px')
        print(f'  Original Median L2: {subset["original_median_l2"].mean():.2f} ± {subset["original_median_l2"].std() if len(subset) > 1 else 0:.2f} px')
        print(f'  SimRBF Mean L2:     {subset["sim_rbf_mean_l2"].mean():.2f} ± {subset["sim_rbf_mean_l2"].std() if len(subset) > 1 else 0:.2f} px')
        print(f'  SimRBF Median L2:   {subset["sim_rbf_median_l2"].mean():.2f} ± {subset["sim_rbf_median_l2"].std() if len(subset) > 1 else 0:.2f} px')

        # Add other baselines if available
        baseline_methods = ['similarity', 'poly', 'tps', 'pwa', 'sim_tps', 'sim_pwa']
        for method in baseline_methods:
            mean_col = f'{method}_mean_l2'
            median_col = f'{method}_median_l2'
            if mean_col in subset.columns and subset[mean_col].notna().any():
                print(f'  {method.capitalize()} Mean L2:   {subset[mean_col].mean():.2f} ± {subset[mean_col].std() if len(subset) > 1 else 0:.2f} px')
                print(f'  {method.capitalize()} Median L2: {subset[median_col].mean():.2f} ± {subset[median_col].std() if len(subset) > 1 else 0:.2f} px')

        print(f'  Improvement vs Original: {subset["orig_improvement_pct"].mean():+.1f}%')
        print(f'  Improvement vs SimRBF:   {subset["sim_improvement_pct"].mean():+.1f}%')

        summary_row = {
            'num_points': num_points,
            'num_datasets': len(subset),
            'neural_mean_l2': subset['mean_l2'].mean(),
            'neural_std_l2': subset['mean_l2'].std() if len(subset) > 1 else 0,
            'neural_median_l2': subset['median_l2'].mean(),
            'neural_median_std': subset['median_l2'].std() if len(subset) > 1 else 0,
            'original_mean_l2': subset['original_mean_l2'].mean(),
            'original_median_l2': subset['original_median_l2'].mean(),
            'sim_rbf_mean_l2': subset['sim_rbf_mean_l2'].mean(),
            'sim_rbf_median_l2': subset['sim_rbf_median_l2'].mean(),
            'improvement_vs_original': subset['orig_improvement_pct'].mean(),
            'improvement_vs_simrbf': subset['sim_improvement_pct'].mean(),
        }

        # Add other baselines to summary
        for method in baseline_methods:
            mean_col = f'{method}_mean_l2'
            median_col = f'{method}_median_l2'
            if mean_col in subset.columns and subset[mean_col].notna().any():
                summary_row[f'{method}_mean_l2'] = subset[mean_col].mean()
                summary_row[f'{method}_median_l2'] = subset[median_col].mean()

        summary_rows.append(summary_row)

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path(f'../../outputs/separate_calibration_summary{args.output_suffix}.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f'\nSummary saved to {summary_path}')

    print('\n' + '='*80)
    print('Evaluation Complete!')
    print('='*80)


if __name__ == '__main__':
    main()
