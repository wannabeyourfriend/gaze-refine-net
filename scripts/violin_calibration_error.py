import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def filter_outliers_iqr(errors_df: pd.DataFrame, k: float = 3) -> pd.DataFrame:
    """
    Filter outliers using IQR method (per method).

    Points beyond Q3 + k*IQR are removed (only upper bound, since errors >= 0).
    k=1.5 is the standard Tukey fence for outliers.
    """
    filtered_dfs = []
    for method in errors_df['Method'].unique():
        method_data = errors_df[errors_df['Method'] == method].copy()
        q1 = method_data['Error (pixels)'].quantile(0.25)
        q3 = method_data['Error (pixels)'].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + k * iqr

        filtered = method_data[method_data['Error (pixels)'] <= upper_bound]
        filtered_dfs.append(filtered)

    return pd.concat(filtered_dfs, ignore_index=True)


def calculate_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Euclidean calibration errors for all methods.

    Returns a long-form DataFrame with columns: Method, Error (pixels)
    """
    # Define all calibration methods and their column prefixes
    methods = {
        'Origin': ('origin_gaze_x', 'origin_gaze_y'),
        'Similarity': ('pred_similarity_x', 'pred_similarity_y'),
        'Poly': ('pred_poly_x', 'pred_poly_y'),
        'RBF-TP': ('pred_rbf_thin_plate_s1.0_x', 'pred_rbf_thin_plate_s1.0_y'),
        'RBF-MQ-s0': ('pred_rbf_multiquadric_s0.0_x', 'pred_rbf_multiquadric_s0.0_y'),
        'RBF-MQ-s1': ('pred_rbf_multiquadric_s1.0_x', 'pred_rbf_multiquadric_s1.0_y'),
        'RBF-MQ-s2': ('pred_rbf_multiquadric_s2.0_x', 'pred_rbf_multiquadric_s2.0_y'),
        'TPS': ('pred_tps_x', 'pred_tps_y'),
        'PWA': ('pred_pwa_x', 'pred_pwa_y'),
        'GPR': ('pred_gpr_x', 'pred_gpr_y'),
        'Sim+RBF-TP': ('pred_sim_rbf_thin_plate_s1.0_x', 'pred_sim_rbf_thin_plate_s1.0_y'),
        'Sim+RBF-MQ-s0': ('pred_sim_rbf_multiquadric_s0.0_x', 'pred_sim_rbf_multiquadric_s0.0_y'),
        'Sim+RBF-MQ-s1': ('pred_sim_rbf_multiquadric_s1.0_x', 'pred_sim_rbf_multiquadric_s1.0_y'),
        'Sim+RBF-MQ-s2': ('pred_sim_rbf_multiquadric_s2.0_x', 'pred_sim_rbf_multiquadric_s2.0_y'),
        'Sim+TPS': ('pred_sim_tps_x', 'pred_sim_tps_y'),
        'Sim+PWA': ('pred_sim_pwa_x', 'pred_sim_pwa_y'),
        'Sim+GPR': ('pred_sim_gpr_x', 'pred_sim_gpr_y'),
    }

    errors_data = []
    for method_name, (x_col, y_col) in methods.items():
        if x_col not in df.columns or y_col not in df.columns:
            print(f"Warning: Columns for {method_name} not found, skipping...")
            continue

        error = np.sqrt(
            (df[x_col] - df['target_x'])**2 +
            (df[y_col] - df['target_y'])**2
        )
        for e in error.dropna():
            errors_data.append({'Method': method_name, 'Error (pixels)': e})

    return pd.DataFrame(errors_data)


def create_violin_plot(errors_df: pd.DataFrame, output_path: Path, k_value: float = None, ax=None) -> None:
    """
    Create and save violin plot ordered by median error.

    If ax is provided, plot on that axis (for subplots).
    """
    # Calculate median error for each method and sort
    median_errors = errors_df.groupby('Method')['Error (pixels)'].median().sort_values()
    method_order = median_errors.index.tolist()

    # Create figure if no axis provided
    standalone = ax is None
    if standalone:
        plt.figure(figsize=(16, 8))
        ax = plt.gca()

    # Create violin plot
    sns.violinplot(
        data=errors_df,
        x='Method',
        y='Error (pixels)',
        order=method_order,
        palette='Blues_r',
        inner='box',
        linewidth=0.8,
        cut=0,  # Clip KDE at data range (no extension below 0)
        ax=ax,
    )

    # Styling
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Calibration Method', fontsize=10)
    ax.set_ylabel('Calibration Error (pixels)', fontsize=10)
    title = 'Calibration Error Distribution by Method'
    if k_value is not None:
        title += f'\n(IQR filter k={k_value})'
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add median values as text
    for i, method in enumerate(method_order):
        median_val = median_errors[method]
        ax.text(i, median_val, f'{median_val:.1f}',
                ha='center', va='bottom', fontsize=7, color='darkblue')

    if standalone:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Violin plot saved to: {output_path}")


def create_violin_plot_series(errors_df: pd.DataFrame, output_path: Path, k_values: list) -> None:
    """
    Create a series of violin plots with different k values for IQR filtering.
    """
    n_plots = len(k_values)
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 6 * n_plots))

    if n_plots == 1:
        axes = [axes]

    for ax, k in zip(axes, k_values):
        # Filter with current k value
        filtered_df = filter_outliers_iqr(errors_df, k=k)
        removed_pct = 100 * (len(errors_df) - len(filtered_df)) / len(errors_df)
        print(f"k={k}: Removed {len(errors_df) - len(filtered_df)} outliers ({removed_pct:.1f}%)")

        # Create violin plot on this axis
        create_violin_plot(filtered_df, output_path, k_value=k, ax=ax)

        # Add removal info to title
        current_title = ax.get_title()
        ax.set_title(f"{current_title}\nRemoved {removed_pct:.1f}% outliers", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nViolin plot series saved to: {output_path}")


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'raw' / 'all' / 'all_trials_model_predictions_0111.csv'
    output_path = project_root / 'outputs' / 'violin_calibration_errors_series.png'

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")

    # Calculate errors
    errors_df = calculate_errors(df)
    print(f"Calculated errors for {errors_df['Method'].nunique()} methods")
    print(f"Total data points: {len(errors_df)}")

    # Create series of violin plots with different k values
    k_values = [4, 5, 6]
    print(f"\nGenerating violin plots for k values: {k_values}")
    create_violin_plot_series(errors_df, output_path, k_values)


if __name__ == '__main__':
    main()
