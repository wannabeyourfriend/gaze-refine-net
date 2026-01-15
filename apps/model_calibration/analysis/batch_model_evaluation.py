# ============================================
# batch_model_evaluation.py
# Batch analysis of model calibration results across participants and sessions
# Uses the similarity model as the baseline for normalization
# ============================================

import os
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path
from model_calibration.models.calibration_model_full_compare import run_one_session_vertical, run_one_session_horizontal, run_one_session   # The source script must provide these functions
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

def check_valid_session(sess_dir):
    """
    Validate that a timestamped folder has the expected structure:
      1. Contains origin and test subfolders
      2. Each subfolder has grid_gaze_log.csv
    """
    origin_path = os.path.join(sess_dir, "origin", "grid_gaze_log.csv")
    test_path   = os.path.join(sess_dir, "test",   "grid_gaze_log.csv")
    if not (os.path.isdir(os.path.join(sess_dir, "origin")) and
            os.path.isdir(os.path.join(sess_dir, "test"))):
        return False, None, None
    if not (os.path.exists(origin_path) and os.path.exists(test_path)):
        return False, None, None
    return True, origin_path, test_path

def evaluate_batch_vertical(root_dir, save_csv=True):
    """
    Batch evaluation entry point (without similarity normalization).
    """
    results = []

    for person in sorted(os.listdir(root_dir)):
        person_dir = os.path.join(root_dir, person)
        if not os.path.isdir(person_dir):
            continue

        print(f"\n👤 Processing participant: {person}")
        sessions = sorted(os.listdir(person_dir))

        for sess in sessions:
            sess_dir = os.path.join(person_dir, sess)
            if not os.path.isdir(sess_dir):
                continue

            valid, origin_path, test_path = check_valid_session(sess_dir)
            if not valid:
                print(f"⚠️ Skipping {person}/{sess} — incomplete structure or missing CSV")
                continue

            print(f"✅ Processing {person}/{sess}")
            try:
                stats_df = run_one_session(origin_path, test_path)
                stats_df["person"] = person
                stats_df["session"] = sess
                results.append(stats_df)
            except Exception as e:
                print(f"❌ Failed to process {person}/{sess}: {e}")

    if not results:
        print("❌ No valid experiment data found.")
        return None, None

    # Aggregate results
    print("results length:", len(results))
    all_df = pd.concat(results, ignore_index=True)

    # === Model-wise averages (no normalization) ===
    summary = (
        all_df.groupby("model")[["mean", "median", "p95"]]
        .agg(["mean", "std"])
        .sort_values(("mean", "mean"))   # Sort by average mean error
    )

    print("\n=== Overall Summary (absolute error, no normalization) ===")
    print(summary)
    
    import matplotlib.pyplot as plt
    import numpy as np

    # Extract data
    means  = summary[("mean", "mean")]
    stds   = summary[("mean", "std")]
    labels = summary.index.astype(str)
    N = len(summary)

    x = np.arange(N)

    # same base color, alpha increases from 0.3 → 1.0
    base_color = "steelblue"   # Can be changed to "black", "gray", "navy", etc.
    alphas = np.linspace(0.3, 1.0, N)

    plt.figure(figsize=(10, 6))

    for i in range(N):
        plt.bar(
            x[i],
            means[i],
            yerr=stds[i],
            capsize=5,
            color=base_color,
            alpha=alphas[i]
        )

    plt.title(f"Model Mean Error (mean ± std)\nN = {len(results)}")
    plt.xlabel("Model")
    plt.ylabel("Mean Error")
    plt.xticks(x, labels, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


    if save_csv:
        os.makedirs("batch_results", exist_ok=True)
        all_df.to_csv("batch_results/results_all_sessions_no_norm.csv", index=False)
        summary.to_csv("batch_results/summary_overall_no_norm.csv")
        print("\n✅ Results saved to ./batch_results/")
    if save_csv:
        os.makedirs("batch_results", exist_ok=True)
        all_df.to_csv("batch_results/results_all_sessions_no_norm.csv", index=False)
        summary.to_csv("batch_results/summary_overall_no_norm.csv")
        print("\n✅ Results saved to ./batch_results/")

    return all_df, summary

def evaluate_batch_horizontal(root_dir, save_csv=True):
    """
    Batch evaluation entry point (without similarity normalization).
    """
    results = []

    for person in sorted(os.listdir(root_dir)):
        person_dir = os.path.join(root_dir, person)
        if not os.path.isdir(person_dir):
            continue

        print(f"\n👤 Processing participant: {person}")
        sessions = sorted(os.listdir(person_dir))

        for sess in sessions:
            sess_dir = os.path.join(person_dir, sess)
            if not os.path.isdir(sess_dir):
                continue

            valid, origin_path, test_path = check_valid_session(sess_dir)
            if not valid:
                print(f"⚠️ Skipping {person}/{sess} — incomplete structure or missing CSV")
                continue

            print(f"✅ Processing {person}/{sess}")
            try:
                stats_df = run_one_session_horizontal(origin_path, test_path)
                stats_df["person"] = person
                stats_df["session"] = sess
                results.append(stats_df)
            except Exception as e:
                print(f"❌ Failed to process {person}/{sess}: {e}")

    if not results:
        print("❌ No valid experiment data found.")
        return None, None

    # Aggregate results
    print("results length:", len(results))
    all_df = pd.concat(results, ignore_index=True)

    # === Model-wise averages (no normalization) ===
    summary = (
        all_df.groupby("model")[["mean", "median", "p95"]]
        .agg(["mean", "std"])
        .sort_values(("mean", "mean"))   # Sort by average mean error
    )

    print("\n=== Overall Summary (absolute error, no normalization) ===")
    print(summary)


    return all_df, summary


def get_model_errors(all_df, model_name, metric="mean"):
    """
    Return error values for a model across all sessions, paired after sorting by (person, session).
    """
    df = all_df[all_df["model"] == model_name].copy()
    df = df.sort_values(["person", "session"])
    return df[metric].values


def paired_t_test(all_df_horizontal, all_df_vertical, model_A, model_B, metric="mean"):
    """
    Paired t-test: evaluate whether model_A is significantly better than model_B (smaller error).
    """
    errors_A = get_model_errors(all_df_horizontal, model_A, metric)
    errors_B = get_model_errors(all_df_vertical, model_B, metric)

    stat, p = ttest_rel(errors_A, errors_B, alternative='less')

    print("==============================================")
    print(f"Paired t-test: {all_df_horizontal} < {all_df_vertical} ?")
    print(f"t = {stat:.4f}, p = {p:.6f}")
    print(p)
    if p < 0.05:
        print("✔ Significant: horizontal error is significantly smaller")
    else:
        print("✘ Not significant: cannot reject equal error hypothesis")
    print("==============================================")

if __name__ == "__main__":
    root = Path.home() / "Desktop" / "systematic_recalibration"
    all_df_vertical, summary = evaluate_batch_vertical(root)
    # all_df_horizontal, summary = evaluate_batch_horizontal(root)
    # paired_t_test(all_df_horizontal, all_df_vertical, "sim+pwa-X", "sim+pwa-Y")
