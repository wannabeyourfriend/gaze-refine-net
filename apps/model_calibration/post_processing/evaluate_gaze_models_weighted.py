import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(df, pred_x, pred_y, label="Model"):
    """
    计算模型的加权 MAE 和 RMSE
    """
    # 实际 gaze 和预测 gaze 的误差距离
    dist = np.sqrt((df[pred_x] - df["target_x"]) ** 2 +
                   (df[pred_y] - df["target_y"]) ** 2)

    # spread 越小 → 权重越大
    eps = 1e-6
    weights = 1.0 / (df["spread"].values + eps)
    weights = weights / np.max(weights)  # 归一化

    # 或者更直观的：直接对距离 dist 做 MAE / RMSE
    mae_dist = np.sum(weights * np.abs(dist)) / np.sum(weights)
    rmse_dist = np.sqrt(np.sum(weights * (dist ** 2)) / np.sum(weights))

    print(f"\n📊 {label}")
    print(f"  Weighted MAE  : {mae_dist:.2f} px")
    print(f"  Weighted RMSE : {rmse_dist:.2f} px")

    return {
        "label": label,
        "Weighted_MAE": mae_dist,
        "Weighted_RMSE": rmse_dist
    }

def evaluate_gaze_log(csv_path):
    """
    输入 gaze_log.csv 文件，输出 RBF 和 Polynomial 的加权 MAE/RMSE 对比
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV file is empty.")

    print(f"✅ Loaded {len(df)} gaze samples from {csv_path}")

    results = []
    results.append(evaluate_model(df, "original_gaze_x", "original_gaze_y", "Original"))
    results.append(evaluate_model(df, "rbf_gaze_x", "rbf_gaze_y", "RBF"))
    results.append(evaluate_model(df, "poly_gaze_x", "poly_gaze_y", "Polynomial"))

    # 汇总结果表
    res_df = pd.DataFrame(results)
    res_df.to_csv("model_weighted_summary.csv", index=False)
    print("\n✅ Saved results to model_weighted_summary.csv")

    # ===== 绘图部分 =====
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = res_df["label"]
    mae_vals = res_df["Weighted_MAE"]
    rmse_vals = res_df["Weighted_RMSE"]

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, mae_vals, width, label="Weighted MAE")
    ax.bar(x + width / 2, rmse_vals, width, label="Weighted RMSE")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pixels")
    ax.set_title("Weighted MAE / RMSE Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig("weighted_model_comparison_bar.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    evaluate_gaze_log("C:\\Users\\SCCN\\Desktop\\systematic_recalibration\\Liu Jiaqi\\2025-10-17_20-56-54\\test2\\grid_gaze_log.csv")
