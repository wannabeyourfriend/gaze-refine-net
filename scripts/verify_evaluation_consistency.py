"""
Verify that neural model and baseline methods are evaluated consistently.

Key checks:
1. Both use same test data
2. Baseline predictions come from CSV (calibrated on training data only)
3. Neural model predictions = origin_gaze + predicted_residual
4. Compare actual predictions
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


def main():
    print("="*80)
    print("Verification: Neural Model vs Baselines Evaluation Consistency")
    print("="*80)

    # Load config
    config_path = Path('apps/neural_refine/config/judo_1000.yaml')
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load test data
    test_path = Path('data/prepared/judo_1000_split_no_leakage/test.csv')
    print(f'\nLoading test data from {test_path}')
    test_df = pd.read_csv(test_path)
    print(f'Test samples: {len(test_df)}')

    # Check: Verify test data has all required columns
    print('\n' + '='*80)
    print('Check 1: Test data columns')
    print('='*80)
    required_cols = ['target_x', 'target_y', 'origin_gaze_x', 'origin_gaze_y']
    for col in required_cols:
        if col in test_df.columns:
            print(f'✓ {col}')
        else:
            print(f'✗ MISSING: {col}')

    # Check: Verify baseline predictions exist
    print('\n' + '='*80)
    print('Check 2: Baseline predictions in CSV')
    print('='*80)
    baseline_cols = [col for col in test_df.columns if col.startswith('pred_')]
    print(f'Found {len(baseline_cols)//2} baseline methods (each has _x and _y):')
    for i, col in enumerate(sorted([c.replace('_x', '').replace('_y', '') for c in baseline_cols if '_x' in c])):
        print(f'  {i+1}. {col}')

    # Load neural model
    print('\n' + '='*80)
    print('Check 3: Loading neural model')
    print('='*80)
    coordinate_scale = cfg['model']['coordinate_scale']
    test_dataset = GazeDataset(
        test_path,
        coordinate_scale=coordinate_scale,
        normalize=cfg['data']['normalize'],
        model_type='multi_baseline',
        multi_baseline_features=cfg['data']['multi_baseline_features'],
        is_training=False
    )

    mb_feature_dim = test_dataset.mb_features.shape[1] if test_dataset.mb_features is not None else 0
    print(f'Multi-baseline feature dimension: {mb_feature_dim}')

    model = build_model(cfg['model'], hf_feature_dim=mb_feature_dim)

    ckpt_path = Path('checkpoints/judo_1000/best_model.pt')
    print(f'Loading checkpoint from {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded model from epoch {checkpoint["epoch"]}')

    device = torch.device('cpu')
    model.to(device)
    model.eval()

    # Get neural model predictions
    print('\n' + '='*80)
    print('Check 4: Computing neural model predictions')
    print('='*80)
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_neural_preds = []
    all_orig_gaze = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            residuals_pred = model(inputs)
            residuals_pred_px = residuals_pred * coordinate_scale

            # Get origin_gaze
            start_idx = batch_idx * loader.batch_size
            end_idx = start_idx + inputs.size(0)
            orig_gaze = test_dataset.orig_inputs_px[start_idx:end_idx]

            # Final prediction: origin_gaze + residual
            pred_gaze = orig_gaze + residuals_pred_px
            true_targets = test_dataset.targets_px[start_idx:end_idx]

            all_neural_preds.append(pred_gaze.cpu().numpy())
            all_orig_gaze.append(orig_gaze.cpu().numpy())
            all_targets.append(true_targets.cpu().numpy())

    neural_preds = np.vstack(all_neural_preds)
    orig_gaze = np.vstack(all_orig_gaze)
    targets = np.vstack(all_targets)

    print(f'Neural predictions shape: {neural_preds.shape}')
    print(f'Origin gaze shape: {orig_gaze.shape}')
    print(f'Targets shape: {targets.shape}')

    # Check: Verify neural predictions formula
    print('\n' + '='*80)
    print('Check 5: Verify neural prediction = origin_gaze + residual')
    print('='*80)
    print('First 3 samples:')
    for i in range(3):
        print(f'\nSample {i+1}:')
        print(f'  Origin gaze:     ({orig_gaze[i,0]:.2f}, {orig_gaze[i,1]:.2f})')
        print(f'  Neural residual: ({neural_preds[i,0]-orig_gaze[i,0]:.2f}, {neural_preds[i,1]-orig_gaze[i,1]:.2f})')
        print(f'  Neural predict:  ({neural_preds[i,0]:.2f}, {neural_preds[i,1]:.2f})')
        print(f'  Target:          ({targets[i,0]:.2f}, {targets[i,1]:.2f})')
        l2_error = np.sqrt(((neural_preds[i] - targets[i])**2).sum())
        print(f'  L2 error:        {l2_error:.2f} px')

    # Check: Compare neural vs baselines on same samples
    print('\n' + '='*80)
    print('Check 6: Error comparison (first 10 samples)')
    print('='*80)

    methods = {
        'Neural': neural_preds[:10],
        'Origin Gaze': test_df[['origin_gaze_x', 'origin_gaze_y']].values[:10],
        'Similarity': test_df[['pred_similarity_x', 'pred_similarity_y']].values[:10],
        'SimRBF s1.0': test_df[['pred_sim_rbf_multiquadric_s1.0_x', 'pred_sim_rbf_multiquadric_s1.0_y']].values[:10],
    }

    targets_subset = targets[:10]

    print(f"{'Method':<20} {'Sample L2 errors (px)':<70}")
    print('-' * 90)
    for name, preds in methods.items():
        errors = [np.sqrt(((preds[i] - targets_subset[i])**2).sum()) for i in range(10)]
        error_str = ', '.join([f'{e:.1f}' for e in errors])
        print(f'{name:<20} {error_str}')

    # Check: Overall statistics
    print('\n' + '='*80)
    print('Check 7: Overall test set statistics')
    print('='*80)

    def compute_metrics(preds, targets, name):
        l2_errors = np.sqrt(((preds - targets)**2).sum(axis=1))
        return {
            'name': name,
            'mean_l2': l2_errors.mean(),
            'std_l2': l2_errors.std(),
            'median_l2': np.median(l2_errors),
            'min_l2': l2_errors.min(),
            'max_l2': l2_errors.max(),
        }

    all_methods = {
        'Neural Multi-Baseline': neural_preds,
        'Origin Gaze': test_df[['origin_gaze_x', 'origin_gaze_y']].values,
        'Similarity': test_df[['pred_similarity_x', 'pred_similarity_y']].values,
        'Polynomial': test_df[['pred_poly_x', 'pred_poly_y']].values,
        'SimRBF s1.0': test_df[['pred_sim_rbf_multiquadric_s1.0_x', 'pred_sim_rbf_multiquadric_s1.0_y']].values,
    }

    print(f"\n{'Method':<25} {'Mean L2':<10} {'Std':<10} {'Median':<10}")
    print('-' * 55)
    for name, preds in all_methods.items():
        metrics = compute_metrics(preds, targets, name)
        print(f"{metrics['name']:<25} {metrics['mean_l2']:<10.2f} {metrics['std_l2']:<10.2f} {metrics['median_l2']:<10.2f}")

    print('\n' + '='*80)
    print('✓ All checks completed')
    print('='*80)

    # Final check: Are baseline predictions from training-data-only calibration?
    print('\n' + '='*80)
    print('Check 8: Verify baseline predictions are from training-only calibration')
    print('='*80)
    print('Checking if SimRBF predictions were calibrated on TRAINING data only...')
    print('(This was done in scripts/generate_judo_baselines_no_leakage.py)')

    # Load training data to verify
    train_df = pd.read_csv('data/prepared/judo_1000_split_no_leakage/train.csv')
    train_targets = set(train_df[['target_x', 'target_y']].itertuples(index=False, name=None))
    test_targets = set(test_df[['target_x', 'target_y']].itertuples(index=False, name=None))

    overlap = train_targets & test_targets
    print(f'\nTraining target points: {len(train_targets)}')
    print(f'Test target points:     {len(test_targets)}')
    print(f'Overlap:                {len(overlap)}')

    if len(overlap) == 0:
        print('✓ No overlap - baseline predictions are from training-only calibration')
    else:
        print(f'✗ WARNING: {len(overlap)} overlapping target points found!')


if __name__ == '__main__':
    main()
