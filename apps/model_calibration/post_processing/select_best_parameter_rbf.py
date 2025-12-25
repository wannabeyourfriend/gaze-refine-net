import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import product
from scipy.interpolate import Rbf

# ========== 1️⃣ 基础 RBF 模型类 ==========
class WeightedRBFModel:
    def __init__(self, csv_path, function='thin_plate', smooth=0.5, epsilon=1.0, use_weight=True):
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError("CSV is empty.")

        self.function = function
        self.smooth = smooth
        self.epsilon = epsilon
        self.use_weight = use_weight

        xi = df["target_x"].values
        yi = df["target_y"].values

        dx = df["original_dx"].values
        dy = df["original_dy"].values

        # === 权重 ===
        if use_weight and "spread" in df.columns:
            eps = 1e-6
            self.wgt = 1.0 / (df["spread"].values + eps)**2
            self.wgt /= np.max(self.wgt)
        else:
            self.wgt = np.ones_like(dx)

        # === 构建RBF模型 ===
        print(f"✅ Training RBF: f={function}, smooth={smooth}, eps={epsilon}, weight={use_weight}")
        self.rbf_dx = Rbf(xi, yi, dx, function=function, smooth=smooth, epsilon=epsilon, weights=self.wgt)
        self.rbf_dy = Rbf(xi, yi, dy, function=function, smooth=smooth, epsilon=epsilon, weights=self.wgt)

    def _numeric_jacobian(self, x, y, h=1.0):
        """
        使用中心差分对 rbf_dx/rbf_dy 关于 (x,y) 求数值雅可比矩阵。
        返回 2x2 矩阵 J_delta where:
          J_delta[0,0] = d f_x / dx
          J_delta[0,1] = d f_x / dy
          J_delta[1,0] = d f_y / dx
          J_delta[1,1] = d f_y / dy
        h: 差分步长（像素），默认 1.0；可在需要更精细时减小为 0.5。
        """
        # central differences
        xph = x + h
        xmh = x - h
        yph = y + h
        ymh = y - h

        fx_xph = self.rbf_dx(xph, y)
        fx_xmh = self.rbf_dx(xmh, y)
        dfx_dx = (fx_xph - fx_xmh) / (2.0 * h)

        fx_yph = self.rbf_dx(x, yph)
        fx_ymh = self.rbf_dx(x, ymh)
        dfx_dy = (fx_yph - fx_ymh) / (2.0 * h)

        fy_xph = self.rbf_dy(xph, y)
        fy_xmh = self.rbf_dy(xmh, y)
        dfy_dx = (fy_xph - fy_xmh) / (2.0 * h)

        fy_yph = self.rbf_dy(x, yph)
        fy_ymh = self.rbf_dy(x, ymh)
        dfy_dy = (fy_yph - fy_ymh) / (2.0 * h)

        J = np.array([[dfx_dx, dfx_dy],
                      [dfy_dx, dfy_dy]], dtype=float)
        return J

    def correct_gaze_iterative(self, gx, gy,
                               method='newton',
                               max_iter=15,
                               tol=0.5,
                               fd_eps=1.0,
                               damping=0.7,
                               verbose=False):
        """
        更精确地求解 T 使得 G = T + Delta(T).
        输入:
          gx, gy: 系统认定的 gaze 坐标 G
          method: 'newton' 或 'fixed' (fixed-point)
          max_iter: 最大迭代次数
          tol: 收敛阈值，单位像素（当 |T_{n+1}-T_n| < tol 则认为收敛）
          fd_eps: 数值差分步长 h（像素），用于雅可比估计
          damping: 牛顿步长缩放因子（0<damping<=1），防止发散
        返回:
          (x_corr, y_corr, info)
          info: dict 包含 keys: converged(bool), iterations(int), last_step_norm(float)
        说明:
          - 如果牛顿步失败（矩阵奇异或步长异常），会回退到 fixed-point 更新一次再继续。
          - fixed-point 迭代是 T <- G - Delta(T).
        """
        if self.rbf_dx is None or self.rbf_dy is None:
            raise RuntimeError("RBF model not initialized.")

        G = np.array([gx, gy], dtype=float)

        # 初始猜测：一次简单修正 (one-shot)
        dx0 = float(self.rbf_dx(gx, gy))
        dy0 = float(self.rbf_dy(gx, gy))
        T = G - np.array([dx0, dy0], dtype=float)  # 初始估计

        last_step_norm = np.inf
        converged = False

        for i in range(max_iter):
            # 计算 F(T) = T + Delta(T) - G
            fx = float(self.rbf_dx(T[0], T[1]))
            fy = float(self.rbf_dy(T[0], T[1]))
            Delta = np.array([fx, fy], dtype=float)
            F = T + Delta - G   # 2-vector

            if method == 'fixed':
                # 固定点：T_next = G - Delta(T)
                T_next = G - Delta
                step = T_next - T
                last_step_norm = np.linalg.norm(step)
                T = T_next
                if verbose:
                    print(f"[fixed] iter {i}: ||step||={last_step_norm:.4f}")
                if last_step_norm < tol:
                    converged = True
                    break
                continue

            # method == 'newton'
            try:
                Jd = self._numeric_jacobian(T[0], T[1], h=fd_eps)  # J_delta
                JF = np.eye(2) + Jd  # J_F = I + J_delta
                # Solve JF * s = F  for s
                # Newton step: T_next = T - s
                s = np.linalg.solve(JF, F)
                # apply damping to step
                T_next = T - damping * s
                step = T_next - T
                last_step_norm = np.linalg.norm(step)

                if verbose:
                    print(f"[newton] iter {i}: ||s||={np.linalg.norm(s):.4f}, ||step||={last_step_norm:.4f}")

                T = T_next
                if last_step_norm < tol:
                    converged = True
                    break

            except np.linalg.LinAlgError:
                # 奇异或不可逆 -> 回退一次 fixed-point 更新并继续
                if verbose:
                    print(f"[newton] iter {i}: Jacobian singular, fallback to fixed-point once")
                T_next = G - Delta
                step = T_next - T
                last_step_norm = np.linalg.norm(step)
                T = T_next
                if last_step_norm < tol:
                    converged = True
                    break
                continue

        info = {"converged": bool(converged),
                "iterations": i + 1,
                "last_step_norm": float(last_step_norm)}
        x_corr, y_corr = float(T[0]), float(T[1])
        return x_corr, y_corr, info
    


# ========== 2️⃣ 加权指标 ==========
def weighted_metrics(df, pred_x, pred_y):
    dist = np.sqrt((df[pred_x] - df["target_x"])**2 +
                   (df[pred_y] - df["target_y"])**2)
    w = 1.0 / (df["spread"].values + 1e-6)
    w /= np.max(w)
    mae = np.sum(w * np.abs(dist)) / np.sum(w)
    rmse = np.sqrt(np.sum(w * dist**2) / np.sum(w))
    return mae, rmse


# ========== 3️⃣ 实验主函数 ==========
def evaluate_rbf_parameter_grid(train_csv, val_csv):
    functions = ['thin_plate']
    smooths = [0.0, 0.1, 1, 10, 100]
    epsilons = [0.1, 1, 5, 10, 50]
    weight_flags = [True]

    val_df = pd.read_csv(val_csv)
    results = []

    for f, s, e, wflag in product(functions, smooths, epsilons, weight_flags):
        print(f"\n=== {f}, smooth={s}, eps={e}, weight={wflag} ===")
        try:
            model = WeightedRBFModel(train_csv, function=f, smooth=s, epsilon=e, use_weight=wflag)
        except Exception as ex:
            print(f"⚠️ Skip ({f},{s},{e}): {ex}")
            continue

        pred_xs, pred_ys = [], []
        for _, r in val_df.iterrows():
            gx, gy = r["original_gaze_x"], r["original_gaze_y"]
            x_corr, y_corr, _ = model.correct_gaze_iterative(gx, gy)
            pred_xs.append(x_corr)
            pred_ys.append(y_corr)

        val_df["rbf_pred_x"] = pred_xs
        val_df["rbf_pred_y"] = pred_ys
        print(pred_xs)

        mae, rmse = weighted_metrics(val_df, "rbf_pred_x", "rbf_pred_y")
        results.append({
            "function": f,
            "smooth": s,
            "epsilon": e,
            "weighted": wflag,
            "Weighted_MAE": mae,
            "Weighted_RMSE": rmse
        })
        print(f"→ MAE={mae:.2f}, RMSE={rmse:.2f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv("rbf_param_sweep_results.csv", index=False)
    print("\n✅ Results saved to rbf_param_sweep_results.csv")

    # 可视化
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [f"{r['function']}\ns={r['smooth']},e={r['epsilon']},w={r['weighted']}" for _, r in res_df.iterrows()]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, res_df["Weighted_MAE"], width=0.4, label="Weighted MAE")
    ax.bar(x + 0.2, res_df["Weighted_RMSE"], width=0.4, label="Weighted RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylabel("Error (px)")
    plt.title("RBF Model Parameter Sweep")
    plt.tight_layout()
    plt.savefig("rbf_param_sweep_bar.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # 替换为你的训练和验证文件路径
    evaluate_rbf_parameter_grid(
        r"C:\\Users\\SCCN\\Desktop\\systematic_recalibration\\Liu Jiaqi\\2025-10-19_17-03-19\\origin\\grid_gaze_log.csv",
        r"C:\\Users\\SCCN\\Desktop\\systematic_recalibration\\Liu Jiaqi\\2025-10-19_17-03-19\\test2\\grid_gaze_log.csv"
    )
