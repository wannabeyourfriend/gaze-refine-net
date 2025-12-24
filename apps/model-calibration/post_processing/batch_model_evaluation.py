# ============================================
# batch_model_evaluation.py
# 批量分析多被试、多会话的模型校正效果
# 以 similarity 模型为基准进行误差归一化
# ============================================

import os
import pandas as pd
import numpy as np
from glob import glob
from calibration_model_full_compare import run_one_session_vertical, run_one_session_horizontal, run_one_session   # 你原脚本中需提供该函数
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

def check_valid_session(sess_dir):
    """
    检查一个时间文件夹是否结构正确：
      1. 含 origin 和 test 两个文件夹
      2. 各自含 grid_gaze_log.csv
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
    批量分析入口（不做 similarity 归一化）。
    """
    results = []

    for person in sorted(os.listdir(root_dir)):
        person_dir = os.path.join(root_dir, person)
        if not os.path.isdir(person_dir):
            continue

        print(f"\n👤 检测被试: {person}")
        sessions = sorted(os.listdir(person_dir))

        for sess in sessions:
            sess_dir = os.path.join(person_dir, sess)
            if not os.path.isdir(sess_dir):
                continue

            valid, origin_path, test_path = check_valid_session(sess_dir)
            if not valid:
                print(f"⚠️ 跳过 {person}/{sess} —— 结构不完整或缺少CSV")
                continue

            print(f"✅ 正在处理 {person}/{sess}")
            try:
                stats_df = run_one_session(origin_path, test_path)
                stats_df["person"] = person
                stats_df["session"] = sess
                results.append(stats_df)
            except Exception as e:
                print(f"❌ 处理 {person}/{sess} 失败: {e}")

    if not results:
        print("❌ 未找到任何有效实验数据。")
        return None, None

    # 汇总所有结果
    print("results length:", len(results))
    all_df = pd.concat(results, ignore_index=True)

    # === 按模型统计平均表现（不做归一化） ===
    summary = (
        all_df.groupby("model")[["mean", "median", "p95"]]
        .agg(["mean", "std"])
        .sort_values(("mean", "mean"))   # 按 mean 的平均值排序
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
    base_color = "steelblue"   # 可换成 "black", "gray", "navy", etc.
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
        print("\n✅ 结果已保存到 ./batch_results/")
    if save_csv:
        os.makedirs("batch_results", exist_ok=True)
        all_df.to_csv("batch_results/results_all_sessions_no_norm.csv", index=False)
        summary.to_csv("batch_results/summary_overall_no_norm.csv")
        print("\n✅ 结果已保存到 ./batch_results/")

    return all_df, summary

def evaluate_batch_horizontal(root_dir, save_csv=True):
    """
    批量分析入口（不做 similarity 归一化）。
    """
    results = []

    for person in sorted(os.listdir(root_dir)):
        person_dir = os.path.join(root_dir, person)
        if not os.path.isdir(person_dir):
            continue

        print(f"\n👤 检测被试: {person}")
        sessions = sorted(os.listdir(person_dir))

        for sess in sessions:
            sess_dir = os.path.join(person_dir, sess)
            if not os.path.isdir(sess_dir):
                continue

            valid, origin_path, test_path = check_valid_session(sess_dir)
            if not valid:
                print(f"⚠️ 跳过 {person}/{sess} —— 结构不完整或缺少CSV")
                continue

            print(f"✅ 正在处理 {person}/{sess}")
            try:
                stats_df = run_one_session_horizontal(origin_path, test_path)
                stats_df["person"] = person
                stats_df["session"] = sess
                results.append(stats_df)
            except Exception as e:
                print(f"❌ 处理 {person}/{sess} 失败: {e}")

    if not results:
        print("❌ 未找到任何有效实验数据。")
        return None, None

    # 汇总所有结果
    print("results length:", len(results))
    all_df = pd.concat(results, ignore_index=True)

    # === 按模型统计平均表现（不做归一化） ===
    summary = (
        all_df.groupby("model")[["mean", "median", "p95"]]
        .agg(["mean", "std"])
        .sort_values(("mean", "mean"))   # 按 mean 的平均值排序
    )

    print("\n=== Overall Summary (absolute error, no normalization) ===")
    print(summary)


    return all_df, summary


def get_model_errors(all_df, model_name, metric="mean"):
    """
    返回某个模型在所有 session 上的误差值，按 (person, session) 排序后配对。
    """
    df = all_df[all_df["model"] == model_name].copy()
    df = df.sort_values(["person", "session"])
    return df[metric].values


def paired_t_test(all_df_horizontal, all_df_vertical, model_A, model_B, metric="mean"):
    """
    配对 t 检验：检验 model_A 是否显著优于 model_B（误差更小）
    """
    errors_A = get_model_errors(all_df_horizontal, model_A, metric)
    errors_B = get_model_errors(all_df_vertical, model_B, metric)

    stat, p = ttest_rel(errors_A, errors_B, alternative='less')

    print("==============================================")
    print(f"Paired t-test: {all_df_horizontal} < {all_df_vertical} ?")
    print(f"t = {stat:.4f}, p = {p:.6f}")
    print(p)
    if p < 0.05:
        print("✔ 结果显著： 水平误差显著更小")
    else:
        print("✘ 不显著：不能拒绝两种方法误差相同的假设")
    print("==============================================")

if __name__ == "__main__":
    root = r"C:\\Users\\SCCN\\Desktop\\New folder (2)"
    all_df_vertical, summary = evaluate_batch_vertical(root)
    # all_df_horizontal, summary = evaluate_batch_horizontal(root)
    # paired_t_test(all_df_horizontal, all_df_vertical, "sim+pwa-X", "sim+pwa-Y")

