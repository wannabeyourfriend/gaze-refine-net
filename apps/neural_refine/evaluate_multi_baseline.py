"""
Evaluate multi-baseline model on test set
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.model import GazeDataset, build_model
from torch.utils.data import DataLoader

def main():
    print('='*60)
    print('Multi-Baseline Model Evaluation')
    print('='*60)

    # Config
    config_path = Path('config/multi_baseline_s1_small_12.yaml')
    ckpt_path = Path('../../checkpoints/multi_baseline_s1_small_12/best_model.pt')
    coordinate_scale = 100.0

    # Load config
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Build dataloaders manually
    print('\nLoading data...')

    train_dataset = GazeDataset(
        cfg['data']['train_data_path'],
        coordinate_scale=coordinate_scale,
        normalize=cfg['data']['normalize'],
        model_type='multi_baseline',
        multi_baseline_features=cfg['data']['multi_baseline_features'],
        is_training=False
    )

    val_dataset = GazeDataset(
        cfg['data']['val_data_path'],
        coordinate_scale=coordinate_scale,
        normalize=cfg['data']['normalize'],
        model_type='multi_baseline',
        multi_baseline_features=cfg['data']['multi_baseline_features'],
        is_training=False
    )

    test_dataset = GazeDataset(
        cfg['data']['test_data_path'],
        coordinate_scale=coordinate_scale,
        normalize=cfg['data']['normalize'],
        model_type='multi_baseline',
        multi_baseline_features=cfg['data']['multi_baseline_features'],
        is_training=False
    )

    # Get feature dimensions
    hf_feature_dim = 0
    mb_feature_dim = train_dataset.mb_features.shape[1] if train_dataset.mb_features is not None else 0

    print(f'  Train: {len(train_dataset)} samples')
    print(f'  Val:   {len(val_dataset)} samples')
    print(f'  Test:  {len(test_dataset)} samples')
    print(f'  Multi-baseline feature dim: {mb_feature_dim}')

    # Create dataloaders
    batch_size = cfg['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Build model
    print('\nBuilding model...')
    model = build_model(cfg['model'], hf_feature_dim=mb_feature_dim)
    model.eval()

    # Load checkpoint
    if not ckpt_path.exists():
        print(f'Error: Checkpoint not found at {ckpt_path}')
        return

    print(f'Loading checkpoint from {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded model from epoch {checkpoint["epoch"]}')

    # Evaluate on test set
    print('\n' + '='*60)
    print('Test Set Evaluation')
    print('='*60)

    device = torch.device('cpu')
    model.to(device)

    # Manual evaluation
    model.eval()
    all_l2_distances = []
    all_mae_x = []
    all_mae_y = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            residuals_pred = model(inputs)

            # Convert to pixel space
            scale = coordinate_scale
            residuals_pred_px = residuals_pred * scale
            targets_px = targets * scale

            # For multi-baseline: prediction = original_gaze + residual
            # Get original gaze from test dataset
            start_idx = batch_idx * test_loader.batch_size
            end_idx = start_idx + inputs.size(0)
            orig_gaze = test_dataset.orig_inputs_px[start_idx:end_idx]

            # Final prediction
            pred_gaze = orig_gaze + residuals_pred_px

            # True targets
            true_targets = test_dataset.targets_px[start_idx:end_idx]

            # Compute errors
            l2_dist = torch.sqrt(((pred_gaze - true_targets) ** 2).sum(dim=1))
            mae_x = torch.abs(pred_gaze[:, 0] - true_targets[:, 0])
            mae_y = torch.abs(pred_gaze[:, 1] - true_targets[:, 1])

            all_l2_distances.extend(l2_dist.cpu().numpy())
            all_mae_x.extend(mae_x.cpu().numpy())
            all_mae_y.extend(mae_y.cpu().numpy())

    # Convert to numpy arrays
    all_l2_distances = np.array(all_l2_distances)
    all_mae_x = np.array(all_mae_x)
    all_mae_y = np.array(all_mae_y)

    # Print results
    print('\n' + '='*60)
    print('Final Results on Test Set')
    print('='*60)
    print(f"L2 Distance (px):")
    print(f"  Mean:   {all_l2_distances.mean():.2f}")
    print(f"  Std:    {all_l2_distances.std():.2f}")
    print(f"  Median: {np.median(all_l2_distances):.2f}")
    print(f"\nMAE (px):")
    print(f"  X:      {all_mae_x.mean():.2f}")
    print(f"  Y:      {all_mae_y.mean():.2f}")
    print(f"\nRMSE (px):")
    print(f"  X:      {np.sqrt((all_mae_x ** 2).mean()):.2f}")
    print(f"  Y:      {np.sqrt((all_mae_y ** 2).mean()):.2f}")

    # Load test dataset for baseline comparison
    print('\n' + '='*60)
    print('Baseline Comparison')
    print('='*60)

    test_df = pd.read_csv(cfg['data']['test_data_path'])

    # Original gaze error
    orig_gaze_x = test_df['origin_gaze_x'].values
    orig_gaze_y = test_df['origin_gaze_y'].values
    target_x = test_df['target_x'].values
    target_y = test_df['target_y'].values

    orig_l2 = np.sqrt((orig_gaze_x - target_x)**2 + (orig_gaze_y - target_y)**2)
    print(f"\nOriginal Gaze:")
    print(f"  Mean L2:   {orig_l2.mean():.2f} px")
    print(f"  Median L2: {np.median(orig_l2):.2f} px")

    # SimRBF baseline error
    sim_rbf_x = test_df['pred_sim_rbf_multiquadric_s2.0_x'].values
    sim_rbf_y = test_df['pred_sim_rbf_multiquadric_s2.0_y'].values

    sim_rbf_l2 = np.sqrt((sim_rbf_x - target_x)**2 + (sim_rbf_y - target_y)**2)
    print(f"\nSimRBF Baseline:")
    print(f"  Mean L2:   {sim_rbf_l2.mean():.2f} px")
    print(f"  Median L2: {np.median(sim_rbf_l2):.2f} px")

    # Improvement
    improvement = (orig_l2.mean() - all_l2_distances.mean()) / orig_l2.mean() * 100
    sim_improvement = (sim_rbf_l2.mean() - all_l2_distances.mean()) / sim_rbf_l2.mean() * 100

    print('\n' + '='*60)
    print('Improvement Summary')
    print('='*60)
    print(f'vs Original Gaze:  {improvement:+.1f}%')
    print(f'vs SimRBF Baseline: {sim_improvement:+.1f}%')

if __name__ == '__main__':
    main()