import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from gaze_correction_polynomial import PolynomialDriftCalibrator  # 你上面定义的类

def weighted_metrics(df, pred_x, pred_y):
    """计算加权 MAE / RMSE（基于 target_x/y）"""
    dist = np.sqrt((df[pred_x] - df["target_x"])**2 +
                   (df[pred_y] - df["target_y"])**2)
    w = 1.0 / (df["spread"].values + 1e-6)**2
    w /= np.max(w)
    mae = np.sum(w * np.abs(dist)) / np.sum(w)
    rmse = np.sqrt(np.sum(w * dist**2) / np.sum(w))
    return mae, rmse


def evaluate_parameter_grid(train_csv, val_csv):
    # 超参数范围
    degrees = [1, 2]
    alphas = [0.0, 1e-4, 1e-3, 1e-2]
    weight_flags = [True, False]

    val_df = pd.read_csv(val_csv)
    results = []

    for degree, alpha, weighted in product(degrees, alphas, weight_flags):
        print(f"\n=== Training degree={degree}, alpha={alpha}, weighted={weighted} ===")
        model = PolynomialDriftCalibrator(train_csv, degree=degree, regularization=alpha, use_weight=weighted)

        # 对验证集反解 gaze
        pred_xs, pred_ys = [], []
        for _, r in val_df.iterrows():
            gx, gy = r["original_gaze_x"], r["original_gaze_y"]
            tx, ty = model.correct_gaze(gx, gy)
            pred_xs.append(tx)
            pred_ys.append(ty)

        val_df["poly_pred_x"] = pred_xs
        val_df["poly_pred_y"] = pred_ys

        mae, rmse = weighted_metrics(val_df, "poly_pred_x", "poly_pred_y")
        results.append({"degree": degree, "alpha": alpha, "weighted": weighted,
                        "Weighted_MAE": mae, "Weighted_RMSE": rmse})
        print(f"→ Weighted MAE={mae:.2f}, RMSE={rmse:.2f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv("poly_param_sweep_results.csv", index=False)
    print("\n✅ Results saved to poly_param_sweep_results.csv")

    # 绘图：每个参数组合的RMSE条形图
    fig, ax = plt.subplots(figsize=(10,5))
    labels = [f"d={r['degree']}, α={r['alpha']}, w={r['weighted']}" for _, r in res_df.iterrows()]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, res_df["Weighted_MAE"], width, label="Weighted MAE")
    ax.bar(x + width/2, res_df["Weighted_RMSE"], width, label="Weighted RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Pixels")
    ax.legend()
    plt.title("Polynomial Model Parameter Sweep (Weighted MAE/RMSE)")
    plt.tight_layout()
    plt.savefig("poly_param_sweep_bar.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    evaluate_parameter_grid(r"C:\\Users\\SCCN\\Desktop\\systematic_recalibration\\Liu Jiaqi\\2025-10-20_16-26-10\\origin\\grid_gaze_log.csv",
        r"C:\\Users\\SCCN\\Desktop\\systematic_recalibration\\Liu Jiaqi\\2025-10-20_16-26-10\\test\\grid_gaze_log.csv")
