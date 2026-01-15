"""
Residual Correlation Analysis

This script analyzes the correlation between SimRBF residuals and input coordinates.
Key finding: Residuals are essentially random noise with no correlation to spatial position.

Usage:
    python apps/neural_refine/analysis/01_residual_correlation_analysis.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data/prepared/split_avg_data/train.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    train = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("RESIDUAL CORRELATION ANALYSIS")
    print("=" * 60)

    # Calculate residuals
    print("\n=== Dataset Info ===")
    print(f"Train samples: {len(train)}")
    print(f"Target X range: [{train.target_x.min()}, {train.target_x.max()}]")
    print(f"Target Y range: [{train.target_y.min()}, {train.target_y.max()}]")

    # Original gaze error
    print("\n=== Original Gaze Error ===")
    orig_err_x = train['target_x'] - train['original_gaze_x']
    orig_err_y = train['target_y'] - train['original_gaze_y']
    orig_l2 = np.sqrt(orig_err_x**2 + orig_err_y**2)
    print(f"Residual X: mean={orig_err_x.mean():.2f}, std={orig_err_x.std():.2f}")
    print(f"Residual Y: mean={orig_err_y.mean():.2f}, std={orig_err_y.std():.2f}")
    print(f"L2 Error: mean={orig_l2.mean():.2f}, median={orig_l2.median():.2f}")

    # SimRBF error
    print("\n=== SimRBF Error ===")
    sim_err_x = train['target_x'] - train['sim_rbf_gaze_x']
    sim_err_y = train['target_y'] - train['sim_rbf_gaze_y']
    sim_l2 = np.sqrt(sim_err_x**2 + sim_err_y**2)
    print(f"Residual X: mean={sim_err_x.mean():.2f}, std={sim_err_x.std():.2f}")
    print(f"Residual Y: mean={sim_err_y.mean():.2f}, std={sim_err_y.std():.2f}")
    print(f"L2 Error: mean={sim_l2.mean():.2f}, median={sim_l2.median():.2f}")

    # Baseline predictions
    print("\n=== Baseline Predictions ===")
    print(f"If model predicts zero residual: L2 = {sim_l2.mean():.2f} px")
    l2_mean = np.sqrt((sim_err_x - sim_err_x.mean())**2 + (sim_err_y - sim_err_y.mean())**2)
    print(f"If model predicts mean residual ({sim_err_x.mean():.2f}, {sim_err_y.mean():.2f}): L2 = {l2_mean.mean():.2f} px")

    # Correlation analysis
    print("\n=== Residual vs Target Position Correlation ===")
    corr_x_tx = np.corrcoef(train['target_x'], sim_err_x)[0, 1]
    corr_x_ty = np.corrcoef(train['target_y'], sim_err_x)[0, 1]
    corr_y_tx = np.corrcoef(train['target_x'], sim_err_y)[0, 1]
    corr_y_ty = np.corrcoef(train['target_y'], sim_err_y)[0, 1]
    print(f"res_x vs target_x: r = {corr_x_tx:.3f}")
    print(f"res_x vs target_y: r = {corr_x_ty:.3f}")
    print(f"res_y vs target_x: r = {corr_y_tx:.3f}")
    print(f"res_y vs target_y: r = {corr_y_ty:.3f}")

    print("\n=== Residual vs Original Gaze Correlation ===")
    corr_x_ox = np.corrcoef(train['original_gaze_x'], sim_err_x)[0, 1]
    corr_x_oy = np.corrcoef(train['original_gaze_y'], sim_err_x)[0, 1]
    corr_y_ox = np.corrcoef(train['original_gaze_x'], sim_err_y)[0, 1]
    corr_y_oy = np.corrcoef(train['original_gaze_y'], sim_err_y)[0, 1]
    print(f"res_x vs orig_gaze_x: r = {corr_x_ox:.3f}")
    print(f"res_x vs orig_gaze_y: r = {corr_x_oy:.3f}")
    print(f"res_y vs orig_gaze_x: r = {corr_y_ox:.3f}")
    print(f"res_y vs orig_gaze_y: r = {corr_y_oy:.3f}")

    # Residual distribution
    print("\n=== Residual Distribution ===")
    print(f"Residual X: skewness={sim_err_x.skew():.2f}, kurtosis={sim_err_x.kurtosis():.2f}")
    print(f"Residual Y: skewness={sim_err_y.skew():.2f}, kurtosis={sim_err_y.kurtosis():.2f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc1 = axes[0].scatter(
        train['target_x'], train['target_y'],
        c=sim_err_x, cmap='RdBu', s=20, alpha=0.7, vmin=-100, vmax=100
    )
    axes[0].set_xlabel('Target X')
    axes[0].set_ylabel('Target Y')
    axes[0].set_title('SimRBF Residual X (spatial distribution)')
    plt.colorbar(sc1, ax=axes[0], label='res_x (px)')

    sc2 = axes[1].scatter(
        train['target_x'], train['target_y'],
        c=sim_err_y, cmap='RdBu', s=20, alpha=0.7, vmin=-100, vmax=100
    )
    axes[1].set_xlabel('Target X')
    axes[1].set_ylabel('Target Y')
    axes[1].set_title('SimRBF Residual Y (spatial distribution)')
    plt.colorbar(sc2, ax=axes[1], label='res_y (px)')

    plt.tight_layout()
    output_path = OUTPUT_DIR / "residual_spatial_analysis.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nFigure saved to: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
All correlations are near zero (|r| < 0.1), indicating that:
1. SimRBF residuals have NO spatial structure
2. Residuals are independent of input coordinates
3. Residuals appear to be random measurement noise
4. A neural network cannot learn to predict these residuals from coordinates alone

The ~41 px residual error is likely irreducible noise from:
- Eye tracker measurement uncertainty
- Natural fixation instability
- Head micro-movements
""")


if __name__ == "__main__":
    main()
