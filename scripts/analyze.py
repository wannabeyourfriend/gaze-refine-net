"""
Unified dataset analysis and evaluation script.
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def analyze_dataset(train_path, val_path, test_path):
    """Analyze dataset statistics."""
    print('='*60)
    print('Dataset Statistics')
    print('='*60)

    train_df = pd.read_csv(train_path) if train_path else None
    val_df = pd.read_csv(val_path) if val_path else None
    test_df = pd.read_csv(test_path) if test_path else None

    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        if df is not None:
            print(f'{name}: {len(df)} samples')

    if train_df and val_df and test_df:
        all_df = pd.concat([train_df, val_df, test_df])
        print(f'Total: {len(all_df)} samples')

        unique_targets = all_df.groupby(['target_x', 'target_y']).size()
        print(f'\nUnique target points: {len(unique_targets)}')
        print(f'Total subjects: {all_df["subject"].nunique() if "subject" in all_df.columns else "N/A"}')


def evaluate_baselines(data_path, output_path=None):
    """Evaluate all baseline calibration methods."""
    print(f"Evaluating baselines on {data_path}")
    df = pd.read_csv(data_path)

    methods = [
        ('Original', 'origin_gaze'),
        ('Similarity', 'pred_similarity'),
        ('Polynomial', 'pred_poly'),
        ('SimRBF(s2.0)', 'pred_sim_rbf_multiquadric_s2.0'),
    ]

    results = []
    for name, prefix in methods:
        pred_x = df[f'{prefix}_x'].values
        pred_y = df[f'{prefix}_y'].values
        target_x = df['target_x'].values
        target_y = df['target_y'].values

        l2_error = np.sqrt((pred_x - target_x)**2 + (pred_y - target_y)**2)
        results.append({
            'Method': name,
            'Mean': l2_error.mean(),
            'Std': l2_error.std(),
            'Median': np.median(l2_error),
        })

    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))

    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Dataset analysis and evaluation')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze dataset statistics')
    analyze_parser.add_argument('--train', type=str, help='Train CSV path')
    analyze_parser.add_argument('--val', type=str, help='Validation CSV path')
    analyze_parser.add_argument('--test', type=str, help='Test CSV path')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate baseline methods')
    eval_parser.add_argument('--data', type=str, required=True, help='Dataset CSV path')
    eval_parser.add_argument('--output', type=str, help='Output CSV path')

    args = parser.parse_args()

    if args.command == 'analyze':
        analyze_dataset(args.train, args.val, args.test)
    elif args.command == 'evaluate':
        evaluate_baselines(args.data, args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
