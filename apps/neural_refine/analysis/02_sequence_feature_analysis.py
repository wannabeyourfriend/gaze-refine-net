"""
Sequence Feature Analysis

This script analyzes the raw gaze sequence data to extract features and
evaluate their correlation with calibration error.

Key findings:
1. Sequence features (spread, velocity) correlate with error MAGNITUDE (~0.25)
2. But they do NOT correlate with error DIRECTION (residual x/y) (~0)
3. Different aggregation methods (mean, median, trimmed) perform similarly
4. Temporal drift within fixations is minimal

Usage:
    python apps/neural_refine/analysis/02_sequence_feature_analysis.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_PATH = PROJECT_ROOT / "data/raw/all/all_trials_model_predictions_0111.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract statistical features from raw gaze sequences."""
    features = []

    for idx, row in df.iterrows():
        n = int(row.n_samples)

        # Get sequence points
        seq_x = [row[f'test_x_{i}'] for i in range(min(n, 148))
                 if pd.notna(row.get(f'test_x_{i}'))]
        seq_y = [row[f'test_y_{i}'] for i in range(min(n, 148))
                 if pd.notna(row.get(f'test_y_{i}'))]

        if len(seq_x) < 5:
            features.append({
                'idx': idx,
                'std_x': np.nan, 'std_y': np.nan,
                'range_x': np.nan, 'range_y': np.nan,
                'skew_x': np.nan, 'skew_y': np.nan,
                'drift_x': np.nan, 'drift_y': np.nan,
                'velocity': np.nan,
            })
            continue

        seq_x = np.array(seq_x)
        seq_y = np.array(seq_y)
        t = np.arange(len(seq_x))

        # Statistical features
        std_x = seq_x.std()
        std_y = seq_y.std()
        range_x = seq_x.max() - seq_x.min()
        range_y = seq_y.max() - seq_y.min()
        skew_x = stats.skew(seq_x)
        skew_y = stats.skew(seq_y)

        # Drift (linear regression slope)
        drift_x = np.polyfit(t, seq_x, 1)[0]
        drift_y = np.polyfit(t, seq_y, 1)[0]

        # Velocity (mean distance between consecutive points)
        if len(seq_x) > 1:
            dx = np.diff(seq_x)
            dy = np.diff(seq_y)
            velocity = np.sqrt(dx**2 + dy**2).mean()
        else:
            velocity = 0

        features.append({
            'idx': idx,
            'std_x': std_x, 'std_y': std_y,
            'range_x': range_x, 'range_y': range_y,
            'skew_x': skew_x, 'skew_y': skew_y,
            'drift_x': drift_x, 'drift_y': drift_y,
            'velocity': velocity,
        })

    return pd.DataFrame(features).set_index('idx')


def analyze_temporal_drift(df: pd.DataFrame):
    """Analyze temporal drift patterns within fixations."""
    drift_x_corrs = []
    drift_y_corrs = []
    early_errors = []
    late_errors = []

    for idx, row in df.iterrows():
        n = int(row.n_samples)
        if n < 20:
            continue

        seq_x = [row[f'test_x_{i}'] for i in range(min(n, 148))
                 if pd.notna(row.get(f'test_x_{i}'))]
        seq_y = [row[f'test_y_{i}'] for i in range(min(n, 148))
                 if pd.notna(row.get(f'test_y_{i}'))]

        if len(seq_x) < 20:
            continue

        seq_x = np.array(seq_x)
        seq_y = np.array(seq_y)
        t = np.arange(len(seq_x))

        # Time correlation
        corr_x = np.corrcoef(t, seq_x)[0, 1]
        corr_y = np.corrcoef(t, seq_y)[0, 1]
        drift_x_corrs.append(corr_x)
        drift_y_corrs.append(corr_y)

        # Early vs late error comparison
        half = len(seq_x) // 2
        target_x, target_y = row.target_x, row.target_y

        early_err = np.sqrt((seq_x[:half] - target_x)**2 + (seq_y[:half] - target_y)**2).mean()
        late_err = np.sqrt((seq_x[half:] - target_x)**2 + (seq_y[half:] - target_y)**2).mean()
        early_errors.append(early_err)
        late_errors.append(late_err)

    return {
        'drift_x_corrs': drift_x_corrs,
        'drift_y_corrs': drift_y_corrs,
        'early_errors': early_errors,
        'late_errors': late_errors,
    }


def test_aggregation_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Test different aggregation methods for sequence points."""
    results = []

    for idx, row in df.iterrows():
        n = int(row.n_samples)
        if n < 10:
            continue

        seq_x = np.array([row[f'test_x_{i}'] for i in range(min(n, 148))
                         if pd.notna(row.get(f'test_x_{i}'))])
        seq_y = np.array([row[f'test_y_{i}'] for i in range(min(n, 148))
                         if pd.notna(row.get(f'test_y_{i}'))])

        if len(seq_x) < 10:
            continue

        target_x, target_y = row.target_x, row.target_y

        # 1. Simple mean (current method)
        mean_x, mean_y = seq_x.mean(), seq_y.mean()
        err_mean = np.sqrt((mean_x - target_x)**2 + (mean_y - target_y)**2)

        # 2. Median
        median_x, median_y = np.median(seq_x), np.median(seq_y)
        err_median = np.sqrt((median_x - target_x)**2 + (median_y - target_y)**2)

        # 3. Trimmed mean (remove top/bottom 10%)
        trim_pct = 10
        trim_x = np.percentile(seq_x, [trim_pct, 100 - trim_pct])
        trim_y = np.percentile(seq_y, [trim_pct, 100 - trim_pct])
        mask_x = (seq_x >= trim_x[0]) & (seq_x <= trim_x[1])
        mask_y = (seq_y >= trim_y[0]) & (seq_y <= trim_y[1])
        trimmed_mean_x = seq_x[mask_x].mean() if mask_x.sum() > 0 else mean_x
        trimmed_mean_y = seq_y[mask_y].mean() if mask_y.sum() > 0 else mean_y
        err_trimmed = np.sqrt((trimmed_mean_x - target_x)**2 + (trimmed_mean_y - target_y)**2)

        # 4. Late half mean (assuming fixation stabilizes)
        half = len(seq_x) // 2
        late_mean_x, late_mean_y = seq_x[half:].mean(), seq_y[half:].mean()
        err_late = np.sqrt((late_mean_x - target_x)**2 + (late_mean_y - target_y)**2)

        # 5. Weighted mean (later points have higher weight)
        weights = np.linspace(0.5, 1.5, len(seq_x))
        weighted_x = np.average(seq_x, weights=weights)
        weighted_y = np.average(seq_y, weights=weights)
        err_weighted = np.sqrt((weighted_x - target_x)**2 + (weighted_y - target_y)**2)

        results.append({
            'err_mean': err_mean,
            'err_median': err_median,
            'err_trimmed': err_trimmed,
            'err_late': err_late,
            'err_weighted': err_weighted,
        })

    return pd.DataFrame(results)


def main():
    print("Loading raw sequence data...")
    df = pd.read_csv(RAW_DATA_PATH)

    print("=" * 60)
    print("SEQUENCE FEATURE ANALYSIS")
    print("=" * 60)

    print(f"\n=== Dataset Overview ===")
    print(f"Total samples: {len(df)}")
    print(f"Subjects: {df.subject.unique()}")
    print(f"Sessions: {df.timestamp.nunique()}")
    print(f"Sequence length (n_samples): mean={df.n_samples.mean():.1f}, "
          f"min={df.n_samples.min()}, max={df.n_samples.max()}")

    # Extract features
    print("\nExtracting sequence features...")
    feat_df = extract_sequence_features(df)
    df = df.join(feat_df)

    # Calculate SimRBF error
    df['sim_rbf_err'] = np.sqrt(
        (df['target_x'] - df['pred_sim_rbf_multiquadric_s1.0_x'])**2 +
        (df['target_y'] - df['pred_sim_rbf_multiquadric_s1.0_y'])**2
    )
    df['res_x'] = df['target_x'] - df['pred_sim_rbf_multiquadric_s1.0_x']
    df['res_y'] = df['target_y'] - df['pred_sim_rbf_multiquadric_s1.0_y']

    # Feature correlation analysis
    print("\n=== Feature vs Error MAGNITUDE Correlation ===")
    feat_cols = ['spread', 'n_samples', 'std_x', 'std_y', 'range_x', 'range_y',
                 'skew_x', 'skew_y', 'drift_x', 'drift_y', 'velocity']

    for col in feat_cols:
        valid = df[[col, 'sim_rbf_err']].dropna()
        if len(valid) > 10:
            r = np.corrcoef(valid[col], valid['sim_rbf_err'])[0, 1]
            print(f"{col:15s}: r = {r:+.3f}")

    print("\n=== Feature vs Residual X Correlation ===")
    for col in feat_cols:
        valid = df[[col, 'res_x']].dropna()
        if len(valid) > 10:
            r = np.corrcoef(valid[col], valid['res_x'])[0, 1]
            print(f"{col:15s}: r = {r:+.3f}")

    print("\n=== Feature vs Residual Y Correlation ===")
    for col in feat_cols:
        valid = df[[col, 'res_y']].dropna()
        if len(valid) > 10:
            r = np.corrcoef(valid[col], valid['res_y'])[0, 1]
            print(f"{col:15s}: r = {r:+.3f}")

    # Temporal drift analysis
    print("\n=== Temporal Drift Analysis ===")
    drift_results = analyze_temporal_drift(df)
    print(f"Samples analyzed: {len(drift_results['drift_x_corrs'])}")
    print(f"Time correlation X: mean={np.mean(drift_results['drift_x_corrs']):.3f}, "
          f"std={np.std(drift_results['drift_x_corrs']):.3f}")
    print(f"Time correlation Y: mean={np.mean(drift_results['drift_y_corrs']):.3f}, "
          f"std={np.std(drift_results['drift_y_corrs']):.3f}")
    print(f"Early half error: {np.mean(drift_results['early_errors']):.1f} px")
    print(f"Late half error: {np.mean(drift_results['late_errors']):.1f} px")

    # Aggregation method comparison
    print("\n=== Aggregation Method Comparison ===")
    agg_results = test_aggregation_methods(df)
    for col in agg_results.columns:
        print(f"{col:15s}: {agg_results[col].mean():.2f} px")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Time correlation distribution
    axes[0, 0].hist(drift_results['drift_x_corrs'], bins=50, alpha=0.7, label='X')
    axes[0, 0].hist(drift_results['drift_y_corrs'], bins=50, alpha=0.7, label='Y')
    axes[0, 0].axvline(0, color='k', linestyle='--')
    axes[0, 0].set_xlabel('Time correlation')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Gaze drift over time within fixation')
    axes[0, 0].legend()

    # 2. Early vs late error
    axes[0, 1].scatter(drift_results['early_errors'], drift_results['late_errors'],
                       alpha=0.3, s=10)
    axes[0, 1].plot([0, 200], [0, 200], 'r--')
    axes[0, 1].set_xlabel('Early half error (px)')
    axes[0, 1].set_ylabel('Late half error (px)')
    axes[0, 1].set_title('Early vs Late fixation error')

    # 3. Spread vs error
    df_valid = df[['spread', 'sim_rbf_err']].dropna()
    corr = np.corrcoef(df_valid['spread'], df_valid['sim_rbf_err'])[0, 1]
    axes[1, 0].scatter(df_valid['spread'], df_valid['sim_rbf_err'], alpha=0.3, s=10)
    axes[1, 0].set_xlabel('Spread (within-fixation std)')
    axes[1, 0].set_ylabel('SimRBF error (px)')
    axes[1, 0].set_title(f'Spread vs Error (r={corr:.3f})')

    # 4. n_samples distribution
    axes[1, 1].hist(df['n_samples'], bins=50)
    axes[1, 1].set_xlabel('n_samples (fixation length)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Fixation length distribution')

    plt.tight_layout()
    output_path = OUTPUT_DIR / "sequence_analysis.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nFigure saved to: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
1. Sequence features (spread, velocity, std) correlate with error MAGNITUDE
   (r ~ 0.25), useful for uncertainty estimation / sample weighting.

2. BUT these features do NOT correlate with error DIRECTION (residual x/y),
   so they cannot help predict the correction needed.

3. Different aggregation methods perform similarly - mean is already optimal.

4. Temporal drift within fixations is minimal (correlation ~ 0).
""")


if __name__ == "__main__":
    main()
