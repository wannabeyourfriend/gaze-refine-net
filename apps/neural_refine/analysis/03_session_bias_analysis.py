"""
Session Bias Analysis

This script analyzes session-level and subject-level bias patterns in gaze data.

Key findings:
1. Different subjects have different systematic biases
2. Different sessions (even for the same subject) have different biases
3. Per-session bias correction can reduce error by ~12% (theoretical maximum)
4. BUT session bias cannot be predicted from other sessions (cross-session generalization fails)

Usage:
    python apps/neural_refine/analysis/03_session_bias_analysis.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_PATH = PROJECT_ROOT / "data/raw/all/all_trials_model_predictions_0111.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading data...")
    df = pd.read_csv(RAW_DATA_PATH)

    # Prepare data
    df['sim_rbf_x'] = df['pred_sim_rbf_multiquadric_s1.0_x']
    df['sim_rbf_y'] = df['pred_sim_rbf_multiquadric_s1.0_y']
    df['res_x'] = df['target_x'] - df['sim_rbf_x']
    df['res_y'] = df['target_y'] - df['sim_rbf_y']
    df['session'] = df['subject'] + '_' + df['timestamp']
    df['err_orig'] = np.sqrt(
        (df['target_x'] - df['sim_rbf_x'])**2 +
        (df['target_y'] - df['sim_rbf_y'])**2
    )

    print("=" * 60)
    print("SESSION BIAS ANALYSIS")
    print("=" * 60)

    print(f"\n=== Dataset Overview ===")
    print(f"Total samples: {len(df)}")
    print(f"Subjects: {df.subject.nunique()}")
    print(f"Sessions: {df.session.nunique()}")
    print(f"Samples per session: {df.groupby('session').size().describe()}")

    # Subject-level bias analysis
    print("\n=== Subject-Level Bias ===")
    subject_stats = []
    for subject in df['subject'].unique():
        sub_df = df[df['subject'] == subject]
        mean_err = sub_df['err_orig'].mean()
        bias_x = sub_df['res_x'].mean()
        bias_y = sub_df['res_y'].mean()
        subject_stats.append({
            'subject': subject,
            'n_samples': len(sub_df),
            'n_sessions': sub_df['session'].nunique(),
            'mean_err': mean_err,
            'bias_x': bias_x,
            'bias_y': bias_y,
        })
        print(f"{subject:15s}: err={mean_err:.1f}px, bias=({bias_x:+.1f}, {bias_y:+.1f}), "
              f"n={len(sub_df)}, sessions={sub_df['session'].nunique()}")

    subject_df = pd.DataFrame(subject_stats)

    # Session-level bias analysis
    print("\n=== Session-Level Bias Variance ===")
    session_bias = df.groupby('session').agg({
        'res_x': 'mean',
        'res_y': 'mean',
        'err_orig': 'mean'
    }).reset_index()
    print(f"Session bias X: mean={session_bias.res_x.mean():.2f}, std={session_bias.res_x.std():.2f}")
    print(f"Session bias Y: mean={session_bias.res_y.mean():.2f}, std={session_bias.res_y.std():.2f}")

    # Method 1: Per-subject bias correction
    print("\n=== Method 1: Per-Subject Bias Correction ===")
    for subject in df['subject'].unique():
        mask = df['subject'] == subject
        bias_x = df.loc[mask, 'res_x'].mean()
        bias_y = df.loc[mask, 'res_y'].mean()
        df.loc[mask, 'subject_corrected_x'] = df.loc[mask, 'sim_rbf_x'] + bias_x
        df.loc[mask, 'subject_corrected_y'] = df.loc[mask, 'sim_rbf_y'] + bias_y

    df['err_subject'] = np.sqrt(
        (df['target_x'] - df['subject_corrected_x'])**2 +
        (df['target_y'] - df['subject_corrected_y'])**2
    )
    print(f"Original SimRBF error: {df.err_orig.mean():.2f} px")
    print(f"After subject bias correction: {df.err_subject.mean():.2f} px")
    print(f"Improvement: {df.err_orig.mean() - df.err_subject.mean():.2f} px "
          f"({100 * (df.err_orig.mean() - df.err_subject.mean()) / df.err_orig.mean():.1f}%)")

    # Method 2: Per-session bias correction (oracle - uses test data)
    print("\n=== Method 2: Per-Session Bias Correction (Oracle) ===")
    for session in df['session'].unique():
        mask = df['session'] == session
        bias_x = df.loc[mask, 'res_x'].mean()
        bias_y = df.loc[mask, 'res_y'].mean()
        df.loc[mask, 'session_corrected_x'] = df.loc[mask, 'sim_rbf_x'] + bias_x
        df.loc[mask, 'session_corrected_y'] = df.loc[mask, 'sim_rbf_y'] + bias_y

    df['err_session'] = np.sqrt(
        (df['target_x'] - df['session_corrected_x'])**2 +
        (df['target_y'] - df['session_corrected_y'])**2
    )
    print(f"After session bias correction: {df.err_session.mean():.2f} px")
    print(f"Improvement: {df.err_orig.mean() - df.err_session.mean():.2f} px "
          f"({100 * (df.err_orig.mean() - df.err_session.mean()) / df.err_orig.mean():.1f}%)")

    # Method 3: Cross-validation (predict session bias from other sessions)
    print("\n=== Method 3: LOO-CV Session Bias (No Data Leakage) ===")
    sessions = df['session'].unique()
    predictions_x = np.zeros(len(df))
    predictions_y = np.zeros(len(df))

    for session in sessions:
        mask = df['session'] == session
        subject = df[mask]['subject'].iloc[0]

        # Use other sessions from same subject to estimate bias
        other_sessions = df[(df['subject'] == subject) & (df['session'] != session)]

        if len(other_sessions) > 0:
            bias_x = other_sessions['res_x'].mean()
            bias_y = other_sessions['res_y'].mean()
        else:
            # No other sessions, use global bias
            bias_x = df[~mask]['res_x'].mean()
            bias_y = df[~mask]['res_y'].mean()

        predictions_x[mask] = df.loc[mask, 'sim_rbf_x'] + bias_x
        predictions_y[mask] = df.loc[mask, 'sim_rbf_y'] + bias_y

    df['pred_cv_x'] = predictions_x
    df['pred_cv_y'] = predictions_y
    df['err_cv'] = np.sqrt(
        (df['target_x'] - df['pred_cv_x'])**2 +
        (df['target_y'] - df['pred_cv_y'])**2
    )
    print(f"CV session bias error: {df.err_cv.mean():.2f} px")
    print(f"Improvement: {df.err_orig.mean() - df.err_cv.mean():.2f} px "
          f"({100 * (df.err_orig.mean() - df.err_cv.mean()) / df.err_orig.mean():.1f}%)")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Subject bias scatter
    axes[0, 0].scatter(subject_df['bias_x'], subject_df['bias_y'], s=100, alpha=0.7)
    for _, row in subject_df.iterrows():
        axes[0, 0].annotate(row['subject'][:8], (row['bias_x'], row['bias_y']),
                            fontsize=8, alpha=0.7)
    axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].axvline(0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].set_xlabel('Bias X (px)')
    axes[0, 0].set_ylabel('Bias Y (px)')
    axes[0, 0].set_title('Per-Subject Systematic Bias')

    # 2. Session bias distribution
    axes[0, 1].hist(session_bias['res_x'], bins=30, alpha=0.7, label='Bias X')
    axes[0, 1].hist(session_bias['res_y'], bins=30, alpha=0.7, label='Bias Y')
    axes[0, 1].axvline(0, color='k', linestyle='--')
    axes[0, 1].set_xlabel('Session Bias (px)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Session Bias Distribution')
    axes[0, 1].legend()

    # 3. Error comparison bar chart
    methods = ['Original', 'Subject\nBias', 'Session\nBias (Oracle)', 'CV\nBias']
    errors = [df.err_orig.mean(), df.err_subject.mean(),
              df.err_session.mean(), df.err_cv.mean()]
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728']
    axes[1, 0].bar(methods, errors, color=colors)
    axes[1, 0].set_ylabel('Mean L2 Error (px)')
    axes[1, 0].set_title('Error by Correction Method')
    for i, v in enumerate(errors):
        axes[1, 0].text(i, v + 1, f'{v:.1f}', ha='center', fontsize=10)

    # 4. Residual after perfect session bias
    df['final_res_x'] = df['res_x'] - df.groupby('session')['res_x'].transform('mean')
    df['final_res_y'] = df['res_y'] - df.groupby('session')['res_y'].transform('mean')
    axes[1, 1].hist(df['final_res_x'], bins=50, alpha=0.7, label='Res X')
    axes[1, 1].hist(df['final_res_y'], bins=50, alpha=0.7, label='Res Y')
    axes[1, 1].set_xlabel('Residual after session bias (px)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Residual Distribution (after perfect session bias)')
    axes[1, 1].legend()

    plt.tight_layout()
    output_path = OUTPUT_DIR / "session_bias_analysis.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nFigure saved to: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print(f"""
Summary of Results:
-------------------
Original SimRBF error:        {df.err_orig.mean():.2f} px
Per-subject bias correction:  {df.err_subject.mean():.2f} px ({100 * (df.err_orig.mean() - df.err_subject.mean()) / df.err_orig.mean():+.1f}%)
Per-session bias (oracle):    {df.err_session.mean():.2f} px ({100 * (df.err_orig.mean() - df.err_session.mean()) / df.err_orig.mean():+.1f}%)
CV session bias:              {df.err_cv.mean():.2f} px ({100 * (df.err_orig.mean() - df.err_cv.mean()) / df.err_orig.mean():+.1f}%)

Key Insight:
------------
- Per-session bias correction achieves ~12% improvement (theoretical max)
- BUT cross-session generalization FAILS (CV actually makes it worse)
- Session bias is NOT predictable from other sessions
- The only way to leverage session bias is ONLINE ADAPTATION
  (use calibration points from the current session)
""")


if __name__ == "__main__":
    main()
