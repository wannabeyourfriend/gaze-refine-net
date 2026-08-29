"""
Generate calibration comparison tables (ICML-style, compact, or detailed)
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def calculate_errors(test_df, prefix):
    """Calculate L2, MAE X/Y for a given prediction prefix."""
    pred_x = test_df[f'{prefix}_x'].values
    pred_y = test_df[f'{prefix}_y'].values
    target_x = test_df['target_x'].values
    target_y = test_df['target_y'].values

    l2 = np.sqrt((pred_x - target_x)**2 + (pred_y - target_y)**2)
    mae_x = np.abs(pred_x - target_x).mean()
    mae_y = np.abs(pred_y - target_y).mean()

    return l2.mean(), l2.std(), np.median(l2), mae_x, mae_y


def generate_icml_table(test_df, include_x_y=False):
    """Generate ICML-style comparison table."""
    methods = [
        ('Original', 'origin_gaze'),
        ('Similarity', 'pred_similarity'),
        ('Polynomial', 'pred_poly'),
        ('RBF-MQ(s2.0)', 'pred_rbf_multiquadric_s2.0'),
        ('TPS', 'pred_tps'),
        ('GPR', 'pred_gpr'),
        ('Sim+RBF-MQ(s2.0)', 'pred_sim_rbf_multiquadric_s2.0'),
        ('Sim+TPS', 'pred_sim_tps'),
        ('Sim+GPR', 'pred_sim_gpr'),
    ]

    results = []
    for name, prefix in methods:
        mean, std, median, mae_x, mae_y = calculate_errors(test_df, prefix)
        if include_x_y:
            results.append([name, f"{mean:.2f}", f"{std:.2f}", f"{median:.2f}", f"{mae_x:.2f}", f"{mae_y:.2f}"])
        else:
            results.append([name, f"{mean:.2f}", f"{std:.2f}", f"{median:.2f}"])

    columns = ['Method', 'Mean', 'Std', 'Median']
    if include_x_y:
        columns.extend(['MAE X', 'MAE Y'])

    df = pd.DataFrame(results, columns=columns)
    return df


def generate_compact_table(test_df):
    """Generate compact table with best methods from each category."""
    methods = [
        ('Original', 'origin_gaze'),
        ('Similarity', 'pred_similarity'),
        ('Polynomial', 'pred_poly'),
        ('RBF-MQ(s2.0)', 'pred_rbf_multiquadric_s2.0'),
        ('TPS', 'pred_tps'),
        ('GPR', 'pred_gpr'),
        ('Sim+RBF-MQ(s2.0)', 'pred_sim_rbf_multiquadric_s2.0'),
    ]

    results = []
    for name, prefix in methods:
        mean, std, median, _, _ = calculate_errors(test_df, prefix)
        results.append([name, f"{mean:.2f}±{std:.2f}", f"{median:.2f}"])

    df = pd.DataFrame(results, columns=['Method', 'Mean±Std', 'Median'])
    return df


def main():
    parser = argparse.ArgumentParser(description='Generate calibration comparison tables')
    parser.add_argument('--data', type=str, default='data/prepared/all_trials_split/test.csv',
                        help='Path to test CSV file')
    parser.add_argument('--format', type=str, choices=['icml', 'compact', 'detailed'],
                        default='icml', help='Table format')
    parser.add_argument('--output', type=str, default=None, help='Output CSV path')
    args = parser.parse_args()

    test_df = pd.read_csv(args.data)

    if args.format == 'icml':
        df = generate_icml_table(test_df, include_x_y=False)
    elif args.format == 'compact':
        df = generate_compact_table(test_df)
    elif args.format == 'detailed':
        df = generate_icml_table(test_df, include_x_y=True)

    print(df.to_string(index=False))

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
