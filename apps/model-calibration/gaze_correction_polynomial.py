import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge  # 使用Ridge代替LinearRegression

class PolynomialDriftCalibrator:
    """
    使用二维多项式回归建模 gaze 漂移场（支持正则化和聚集权重）。
    """
    def __init__(self, csv_path, degree=2, regularization=0.0, use_weight=True):
        df = pd.read_csv(csv_path)
        self.degree = degree
        self.regularization = regularization
        self.use_weight = use_weight

        X = df[["target_x", "target_y"]].values
        y_dx = df["original_dx"].values
        y_dy = df["original_dy"].values

        # 构建多项式特征
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = self.poly.fit_transform(X)

        # === 加权 ===
        if use_weight and "spread" in df.columns:
            eps = 1e-6
            w = 1.0 / (df["spread"].values + eps)**2
            w = w / np.max(w)
            print(f"✅ Using weighted training (max weight={w.max():.2f})")
        else:
            w = np.ones_like(y_dx)

        # === 带正则化的模型 ===
        self.model_dx = Ridge(alpha=regularization).fit(X_poly, y_dx, sample_weight=w)
        self.model_dy = Ridge(alpha=regularization).fit(X_poly, y_dy, sample_weight=w)

        print(f"✅ Polynomial model trained: degree={degree}, alpha={regularization}, weighted={use_weight}")

    def predict(self, x, y):
        X_poly = self.poly.transform(np.array([[x, y]]))
        dx = float(self.model_dx.predict(X_poly))
        dy = float(self.model_dy.predict(X_poly))
        return dx, dy

    def correct_gaze(self, gx, gy, max_iter=10, tol=0.5, damping=0.7):
        """反解 G = T + Δ(T)"""
        T = np.array([gx, gy], dtype=float)
        G = np.array([gx, gy], dtype=float)
        for _ in range(max_iter):
            dx, dy = self.predict(T[0], T[1])
            F = T + np.array([dx, dy]) - G
            step = -damping * F
            T += step
            if np.linalg.norm(step) < tol:
                break
        return float(T[0]), float(T[1])



    def visualize_field(self, width, height, step=80, scale=800, save_path=None):
        """
        可视化整个屏幕的漂移场
        """
        import matplotlib.pyplot as plt
        X, Y = np.meshgrid(np.arange(0, width, step),
                           np.arange(0, height, step))
        coords = np.column_stack([X.ravel(), Y.ravel()])
        X_poly = self.poly.transform(coords)
        Zx = self.model_dx.predict(X_poly).reshape(X.shape)
        Zy = self.model_dy.predict(X_poly).reshape(Y.shape)
        mag = np.sqrt(Zx**2 + Zy**2)

        fig, ax = plt.subplots(figsize=(10, 6))
        q = ax.quiver(X, Y, Zx, Zy, mag, scale=scale, cmap='coolwarm')
        ax.invert_yaxis()
        ax.set_title(f"2D Polynomial Drift Field (degree={self.degree})")
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        plt.colorbar(q, label="Drift magnitude (px)")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"✅ Polynomial drift field saved to: {save_path}")
        else:
            plt.show()
