"""
gaze_correction_rbf.py
----------------------
功能：
1. 从 grid_gaze_log.csv 读取数据
2. 使用径向基函数（RBF）插值建立漂移校正模型
3. 提供 gaze 坐标校正函数

依赖：
- numpy
- pandas
- scipy.interpolate.Rbf
"""

import numpy as np
import pandas as pd
from scipy.interpolate import Rbf

class GazeRBFCalibrator:
    def __init__(self, csv_path, smooth=1.0, function='thin_plate'):
        """
        初始化并训练RBF漂移模型
        :param csv_path: grid_gaze_log.csv 文件路径
        :param smooth: 平滑参数 (越大越平滑)
        :param function: RBF核函数 ('thin_plate', 'linear', 'multiquadric' 等)
        """
        self.csv_path = csv_path
        self.smooth = smooth
        self.function = function
        self.rbf_dx = None
        self.rbf_dy = None

        self._train_model()

    def _train_model(self):
        """读取CSV并训练RBF模型"""
        df = pd.read_csv(self.csv_path)
        if df.empty:
            raise ValueError("❌ CSV is empty.")

        # 提取输入与漂移值
        xi = np.array(df["target_x"])
        yi = np.array(df["target_y"])
        dx = np.array(df["original_dx"])
        dy = np.array(df["original_dy"])

        # ==== ✅ 计算权重 ====
        if "spread" in df.columns:
            # spread 越大表示越分散 → 权重越小
            eps = 1e-6
            w = 1.0 / (df["spread"].values + eps)
            # 可选：归一化权重，使最大权重为 1
            w = w / np.max(w)
            print(f"✅ Using weighted RBF training based on spread (max weight={w.max():.2f})")
        else:
            w = np.ones_like(dx)
            print("⚠️ spread not found in CSV, using uniform weights")

        # ==== ✅ 使用权重加权数据 ====
        # RBF 本身不直接接受 sample_weight，但我们可以通过将加权漂移缩放来等效实现
        dx_weighted = dx * w
        dy_weighted = dy * w

        # 构建 RBF 模型
        print(f"✅ building RBF model (function={self.function}, smooth={self.smooth})...")
        local_smooth = self.smooth / (w + 1e-6)
        self.rbf_dx = Rbf(xi, yi, dx_weighted, function=self.function, smooth=local_smooth)
        self.rbf_dy = Rbf(xi, yi, dy_weighted, function=self.function, smooth=local_smooth)
        print("✅ Weighted RBF model built!")

    def correct_gaze(self, x, y):
        """
        输入 gaze 点 (x, y)，输出校正后的 (x', y')
        :param x: gaze 原始 X 坐标
        :param y: gaze 原始 Y 坐标
        :return: (x_corr, y_corr)
        """
        if self.rbf_dx is None or self.rbf_dy is None:
            raise RuntimeError("⚠️ RBF hasn't been built, please first initialize GazeRBFCalibrator.")

        dx_corr = self.rbf_dx(x, y)
        dy_corr = self.rbf_dy(x, y)
        return x - dx_corr, y - dy_corr

    def visualize_field(self, width, height, step=80, scale=800, save_path=None):
        import matplotlib.pyplot as plt
        X, Y = np.meshgrid(np.arange(0, width, step),
                        np.arange(0, height, step))
        Zx = self.rbf_dx(X, Y)
        Zy = self.rbf_dy(X, Y)
        mag = np.sqrt(Zx**2 + Zy**2)

        fig, ax = plt.subplots(figsize=(10, 6))
        q = ax.quiver(X, Y, Zx, Zy, mag, scale=scale, cmap='coolwarm')
        ax.invert_yaxis()
        ax.set_title("RBF Gaze Drift Field")
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        plt.colorbar(q, label="Drift magnitude (px)")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()

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
    