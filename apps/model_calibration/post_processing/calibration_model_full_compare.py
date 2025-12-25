#!/usr/bin/env python3
"""
calibration_model_full_compare.py

一键比较多种校正模型：
- Similarity (Procrustes)
- Polynomial (use provided PolynomialDriftCalibrator)
- RBF residual (scipy.interpolate.Rbf)
- TPS residual (thin-plate spline, numpy implementation)
- Piecewise Affine (Delaunay + per-triangle affine)
- GPR residual (sklearn GaussianProcessRegressor)

依赖: numpy, scipy, sklearn, pandas, matplotlib
不需要额外安装第三方库。
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy.spatial import Delaunay
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as SK_RBF, WhiteKernel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# ========== 用户可改配置 ==========
ORIGIN_PATH = "C:\\Users\\Liu Jiaqi\\Desktop\\systematic_recalibration\\Unnamed\\2025-12-14_17-53-40\\origin\\grid_gaze_log.csv"   # 若不存在，会尝试 "data.csv" 并自动切分
TEST_PATH = "C:\\Users\\Liu Jiaqi\\Desktop\\systematic_recalibration\\Unnamed\\2025-12-14_17-53-40\\test\\grid_gaze_log.csv"
RBF_PARAM_GRID = [("thin_plate", 1.0), ("multiquadric", 0.0), ("multiquadric", 1.0), ("multiquadric", 2.0)]
POLY_DEGREE = 2
POLY_ALPHA = 0.5
# =================================

# -----------------------------
# utility: load/split data
# -----------------------------
def load_or_split(origin_path=ORIGIN_PATH, test_path=TEST_PATH):
    origin_df = pd.read_csv(origin_path)
    test_df = pd.read_csv(test_path)
    return origin_df, test_df

# -----------------------------
# Similarity (Procrustes)
# -----------------------------
def fit_similarity(obs, tgt):
    obs_mean = obs.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    obs_c = obs - obs_mean
    tgt_c = tgt - tgt_mean

    U, S, Vt = np.linalg.svd(tgt_c.T @ obs_c)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:,-1] *= -1
        R = U @ Vt

    num = np.trace(tgt_c @ R @ obs_c.T)
    den = np.trace(obs_c @ obs_c.T)
    s = num / den
    t = tgt_mean - s * R @ obs_mean
    return s, R, t

def apply_similarity(obs, s, R, t):
    return (s * (R @ obs.T)).T + t

# -----------------------------
# Polynomial calibrator (inlined from user class, simplified)
# -----------------------------
class SimplePolynomialCalibrator:
    def __init__(self, df, degree=2, reg=0.1, use_weight=True):
        X = df[['target_x','target_y']].values
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        Xp = self.poly.fit_transform(X)
        dx = df['original_gaze_x'].values - df['target_x'].values
        dy = df['original_gaze_y'].values - df['target_y'].values
        if use_weight and 'spread' in df.columns:
            eps = 1e-6
            w = 1.0 / (df['spread'].values + eps)**2
            w = w / np.max(w)
        else:
            w = None
        self.model_dx = Ridge(alpha=reg).fit(Xp, dx, sample_weight=w)
        self.model_dy = Ridge(alpha=reg).fit(Xp, dy, sample_weight=w)

    def predict_delta(self, x_arr):
        # x_arr: Nx2
        Xp = self.poly.transform(x_arr)
        dx = self.model_dx.predict(Xp)
        dy = self.model_dy.predict(Xp)
        return np.vstack([dx, dy]).T

    def correct_batch(self, gaze_xy):
        out=[]
        for gx,gy in gaze_xy:
            T = np.array([gx,gy], dtype=float)
            G = np.array([gx,gy], dtype=float)
            for _ in range(20):
                d = self.predict_delta(T.reshape(1,2))[0]
                F = T + d - G
                step = -0.7 * F
                T += step
                if np.linalg.norm(step) < 0.5:
                    break
            out.append(T.copy())
        return np.array(out)

# -----------------------------
# RBF residual helper
# -----------------------------
def fit_rbf_residual(obs, corrected, tgt, kernel='Gaussian', smooth=0.0):
    resid = tgt - corrected
    rbf_x = Rbf(obs[:,0], obs[:,1], resid[:,0], function=kernel, smooth=smooth)
    rbf_y = Rbf(obs[:,0], obs[:,1], resid[:,1], function=kernel, smooth=smooth)
    return rbf_x, rbf_y

def apply_rbf_to_points(obs, base_pred, rbf_x, rbf_y):
    res = np.stack([rbf_x(obs[:,0], obs[:,1]), rbf_y(obs[:,0], obs[:,1])], axis=1)
    return base_pred + res

# -----------------------------
# TPS (Thin-Plate Spline) implementation (2D -> 2D)
# pure numpy implementation, solve weights for control points.
# -----------------------------
def _tps_kernel(r):
    # r: array
    # U(r) = r^2 * log(r^2) or r^2 * log(r). We use r^2 * log(r) with safe handling.
    with np.errstate(divide='ignore', invalid='ignore'):
        out = r**2 * np.log(r + (r==0))
    out[np.isnan(out)] = 0.0
    return out

class ThinPlateSpline2D:
    def __init__(self, src_pts, tgt_pts, reg=1e-3):
        """
        src_pts: (N,2) control source points (e.g., corrected obs)
        tgt_pts: (N,2) control target displacements or absolute targets
        reg: regularization for K matrix
        """
        self.src = np.asarray(src_pts)
        self.N = self.src.shape[0]
        self.reg = reg
        self.tgt = np.asarray(tgt_pts)

        # Build K (N,N)
        dists = np.linalg.norm(self.src[:,None,:] - self.src[None,:,:], axis=2)
        K = _tps_kernel(dists)
        # P matrix (N,3)
        P = np.concatenate([np.ones((self.N,1)), self.src], axis=1)  # [1 x y]
        # Assemble L matrix
        top = np.concatenate([K + reg*np.eye(self.N), P], axis=1)
        bottom = np.concatenate([P.T, np.zeros((3,3))], axis=1)
        L = np.concatenate([top, bottom], axis=0)  # (N+3, N+3)

        # rhs: target coords (for dx or for absolute)
        # We will solve for mapping to absolute coordinates (tgt_pts)
        Vx = np.concatenate([self.tgt[:,0], np.zeros(3)], axis=0)
        Vy = np.concatenate([self.tgt[:,1], np.zeros(3)], axis=0)

        # Solve weights
        solx = np.linalg.solve(L, Vx)
        soly = np.linalg.solve(L, Vy)
        # store components
        self.wx = solx[:self.N]; self.ax = solx[self.N:]  # ax: [a0, a1, a2]
        self.wy = soly[:self.N]; self.ay = soly[self.N:]

    def transform(self, pts):
        pts = np.asarray(pts)
        d = np.linalg.norm(pts[:,None,:] - self.src[None,:,:], axis=2)  # (M,N)
        U = _tps_kernel(d)  # (M,N)
        x = U @ self.wx + self.ax[0] + self.ax[1]*pts[:,0] + self.ax[2]*pts[:,1]
        y = U @ self.wy + self.ay[0] + self.ay[1]*pts[:,0] + self.ay[2]*pts[:,1]
        return np.vstack([x,y]).T

# -----------------------------
# Piecewise Affine (Delaunay triangles)
# -----------------------------
class PiecewiseAffine:
    def __init__(self, src_pts, tgt_pts):
        # src_pts, tgt_pts: (N,2)
        self.src = np.asarray(src_pts)
        self.tgt = np.asarray(tgt_pts)
        self.tri = Delaunay(self.src)
        # precompute affine transform per simplex (triangle)
        self.transforms = []
        for simp in self.tri.simplices:
            A_src = np.vstack([np.ones(3), self.src[simp].T])  # 3x3
            A_tgt_x = self.tgt[simp,0]
            A_tgt_y = self.tgt[simp,1]
            # solve A_src.T * coeff = tgt -> coeff shape (3,)
            coeff_x = np.linalg.solve(A_src.T, A_tgt_x)
            coeff_y = np.linalg.solve(A_src.T, A_tgt_y)
            # store coeffs where mapping: [1 x y] @ coeff = mapped coord
            self.transforms.append((simp, coeff_x, coeff_y))

    def transform(self, pts):
        pts = np.asarray(pts)
        out = np.zeros_like(pts)
        simplex_idx = self.tri.find_simplex(pts)
        for i,p in enumerate(pts):
            si = simplex_idx[i]
            if si == -1:
                # outside triangulation -> fallback to nearest control point mapping (translation)
                nn = np.argmin(np.linalg.norm(self.src - p, axis=1))
                out[i] = self.tgt[nn]
            else:
                simp, coeff_x, coeff_y = self.transforms[si]
                vec = np.array([1.0, p[0], p[1]])
                out[i,0] = vec @ coeff_x
                out[i,1] = vec @ coeff_y
        return out

# -----------------------------
# GPR residual (fit residuals as function of obs coords)
# -----------------------------
def fit_gpr_residual(obs, corrected, tgt):
    resid = tgt - corrected
    # kernel: RBF (length_scale ~ 200 px) + white noise
    kernel = SK_RBF(length_scale=200.0) + WhiteKernel(noise_level=2.0)
    gpr_x = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, normalize_y=True).fit(obs, resid[:,0])
    gpr_y = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, normalize_y=True).fit(obs, resid[:,1])
    return gpr_x, gpr_y

def apply_gpr(obs, base_pred, gpr_x, gpr_y):
    rx, sx = gpr_x.predict(obs, return_std=True)
    ry, sy = gpr_y.predict(obs, return_std=True)
    res = np.vstack([rx, ry]).T
    return base_pred + res, np.vstack([sx, sy]).T

# -----------------------------
# evaluation / plotting
# -----------------------------
def eval_stats(pred, tgt):
    errs = np.linalg.norm(pred - tgt, axis=1)
    return {'mean': errs.mean(), 'median': np.median(errs), 'p95': np.percentile(errs,95), 'all': errs}

def plot_arrow_field(df_targets, corrected, title, fname=None):
    tgt = df_targets[['target_x','target_y']].values
    plt.figure(figsize=(7,6))
    plt.scatter(tgt[:,0], tgt[:,1], c='k', s=14)
    for (x0,y0), (x1,y1) in zip(tgt, corrected):
        plt.arrow(x0, y0, x1-x0, y1-y0, color='0.2', alpha=0.7, head_width=6, length_includes_head=True)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    if fname: plt.savefig(fname, dpi=200)
    plt.show()

# def plot_error_cdf(stats_dict, fname=None):
#     plt.figure(figsize=(6,4))
#     for name, v in stats_dict.items():
#         errs = v['all']
#         srt = np.sort(errs)
#         p = np.linspace(0,1,len(srt))
#         plt.plot(srt, p, label=f"{name} (mean={v['mean']:.1f})", linewidth=1.5)
#     plt.xlabel("Error (px)")
#     plt.ylabel("CDF")
#     plt.grid(False)
#     plt.legend()
#     if fname: plt.savefig(fname, dpi=200)
#     plt.show()

def run_one_session(origin_path, test_path):
    """
    运行一次 origin/test 的模型比较（不产生图像）。
    返回 DataFrame:
        columns = ["model", "mean", "median", "p95"]
    """
    # === 1. 读取数据 ===
    origin_df = pd.read_csv(origin_path)
    test_df   = pd.read_csv(test_path)

    origin_obs = origin_df[['original_gaze_x','original_gaze_y']].values
    origin_tgt = origin_df[['target_x','target_y']].values
    test_obs   = test_df[['original_gaze_x','original_gaze_y']].values
    test_tgt   = test_df[['target_x','target_y']].values

    results = []

    # ======================================
    # 2. Similarity baseline
    # ======================================
    s, R, t = fit_similarity(origin_obs, origin_tgt)
    pred_sim = apply_similarity(test_obs, s, R, t)
    stats_sim = eval_stats(pred_sim, test_tgt)
    results.append(["similarity",
                    stats_sim["mean"], stats_sim["median"], stats_sim["p95"]])

    # ======================================
    # 3. Polynomial
    # ======================================
    poly = SimplePolynomialCalibrator(origin_df, degree=POLY_DEGREE, reg=POLY_ALPHA)
    poly_pred = poly.correct_batch(test_obs)
    stats_poly = eval_stats(poly_pred, test_tgt)
    results.append([f"poly(deg={POLY_DEGREE})",
                    stats_poly["mean"], stats_poly["median"], stats_poly["p95"]])


    # ======================================
    # 4. Similarity + RBF (多组参数)
    # ======================================
    origin_sim = apply_similarity(origin_obs, s, R, t)
    for kernel, smooth in RBF_PARAM_GRID:
        rbf_x, rbf_y = fit_rbf_residual(origin_obs, origin_sim, origin_tgt,
                                        kernel=kernel, smooth=smooth)
        rbf_pred = apply_rbf_to_points(test_obs, pred_sim, rbf_x, rbf_y)
        stats_rbf = eval_stats(rbf_pred, test_tgt)
        results.append([f"sim+rbf[{kernel},s={smooth}]",
                        stats_rbf["mean"], stats_rbf["median"], stats_rbf["p95"]])

    # ======================================
    # 5. TPS residual
    # ======================================
    tps = ThinPlateSpline2D(origin_sim, origin_tgt, reg=1e-3)
    tps_pred = tps.transform(pred_sim)
    stats_tps = eval_stats(tps_pred, test_tgt)
    results.append(["sim+tps",
                    stats_tps["mean"], stats_tps["median"], stats_tps["p95"]])

    # ======================================
    # 6. Piecewise Affine
    # ======================================
    pwa = PiecewiseAffine(origin_sim, origin_tgt)
    pwa_pred = pwa.transform(pred_sim)
    stats_pwa = eval_stats(pwa_pred, test_tgt)
    results.append(["sim+pwa",
                    stats_pwa["mean"], stats_pwa["median"], stats_pwa["p95"]])

    # ======================================
    # 7. GPR residual
    # ======================================
    gpr_x, gpr_y = fit_gpr_residual(origin_obs, origin_sim, origin_tgt)
    gpr_pred, _ = apply_gpr(test_obs, pred_sim, gpr_x, gpr_y)
    stats_gpr = eval_stats(gpr_pred, test_tgt)
    results.append(["sim+gpr",
                    stats_gpr["mean"], stats_gpr["median"], stats_gpr["p95"]])

    # === 最终输出 DataFrame ===
    df = pd.DataFrame(results, columns=["model", "mean", "median", "p95"])
    return df

# def run_one_session_horizontal(origin_path, test_path):
#     """
#     只分析水平误差（x方向）。
#     输出 DataFrame:
#         columns = ["model", "mean", "median", "p95"]
#     """
#     # === 1. 读取数据 ===
#     origin_df = pd.read_csv(origin_path)
#     test_df   = pd.read_csv(test_path)

#     origin_obs = origin_df[['original_gaze_x','original_gaze_y']].values
#     origin_tgt = origin_df[['target_x','target_y']].values
#     test_obs   = test_df[['original_gaze_x','original_gaze_y']].values
#     test_tgt   = test_df[['target_x','target_y']].values

#     results = []

#     # ======================================
#     # (A) 定义仅计算水平误差的 eval
#     # ======================================
#     def eval_stats_x(pred_xy, tgt_xy):
#         err = np.abs(pred_xy[:,0] - tgt_xy[:,0])   # horizontal error only
#         return {
#             "mean":   float(np.mean(err)),
#             "median": float(np.median(err)),
#             "p95":    float(np.percentile(err,95))
#         }

#     # ======================================
#     # 2. Similarity baseline  (2D变换，但只用 x误差)
#     # ======================================
#     s, R, t = fit_similarity(origin_obs, origin_tgt)
#     pred_sim = apply_similarity(test_obs, s, R, t)
#     stats_sim = eval_stats_x(pred_sim, test_tgt)
#     results.append(["similarity-X",
#                     stats_sim["mean"], stats_sim["median"], stats_sim["p95"]])

#     # ======================================
#     # 3. Polynomial （只拟合 original_dx，水平误差模型）
#     # ======================================
#     if {'original_dx','original_dy'}.issubset(origin_df.columns):
#         # 构造只训练水平模型的简单版本
#         class Poly1D_X:
#             def __init__(self, df, deg=POLY_DEGREE, alpha=POLY_ALPHA):
#                 X = df[['target_x','target_y']].values
#                 dx = df['original_dx'].values  # only use horizontal residual
#                 self.poly = PolynomialFeatures(degree=deg, include_bias=False)
#                 Xp = self.poly.fit_transform(X)
#                 self.model_dx = Ridge(alpha=alpha).fit(Xp, dx)

#             def predict_delta_x(self, pts):
#                 Xp = self.poly.transform(pts)
#                 return self.model_dx.predict(Xp)

#             def correct_batch(self, gaze_xy):
#                 out=[]
#                 for (gx,gy) in gaze_xy:
#                     T=np.array([gx,gy],dtype=float)
#                     Gx=gx
#                     for _ in range(20):
#                         dx = self.predict_delta_x(T.reshape(1,2))[0]
#                         F = (T[0] + dx) - Gx
#                         T[0] -= 0.7*F
#                         if abs(F)<0.5:
#                             break
#                     out.append([T[0], gy])  # only modify x
#                 return np.array(out)

#         poly = Poly1D_X(origin_df)
#         poly_pred = poly.correct_batch(test_obs)
#         stats_poly = eval_stats_x(poly_pred, test_tgt)
#         results.append([f"poly-X(deg={POLY_DEGREE})",
#                         stats_poly["mean"], stats_poly["median"], stats_poly["p95"]])
#     else:
#         poly = None

#     # ======================================
#     # 4. Similarity + RBF (只拟合水平残差 rbf_x)
#     # ======================================
#     origin_sim = apply_similarity(origin_obs, s, R, t)

#     for kernel, smooth in RBF_PARAM_GRID:
#         resid_x = origin_tgt[:,0] - origin_sim[:,0]
#         rbf_x = Rbf(origin_obs[:,0], origin_obs[:,1], resid_x,
#                     function=kernel, smooth=smooth)

#         # apply
#         res_x = rbf_x(test_obs[:,0], test_obs[:,1])
#         rbf_pred = pred_sim.copy()
#         rbf_pred[:,0] += res_x  # only modify x

#         stats_rbf = eval_stats_x(rbf_pred, test_tgt)
#         results.append([f"sim+rbf-X[{kernel},s={smooth}]",
#                         stats_rbf["mean"], stats_rbf["median"], stats_rbf["p95"]])

#     # ======================================
#     # 5. TPS residual (仍是2D，但只使用 x输出)
#     # ======================================
#     tps = ThinPlateSpline2D(origin_sim, origin_tgt, reg=1e-3)
#     tps_pred = tps.transform(pred_sim)
#     stats_tps = eval_stats_x(tps_pred, test_tgt)
#     results.append(["sim+tps-X",
#                     stats_tps["mean"], stats_tps["median"], stats_tps["p95"]])

#     # ======================================
#     # 6. Piecewise Affine (仍是2D，但只使用 x输出)
#     # ======================================
#     pwa = PiecewiseAffine(origin_sim, origin_tgt)
#     pwa_pred = pwa.transform(pred_sim)
#     stats_pwa = eval_stats_x(pwa_pred, test_tgt)
#     results.append(["sim+pwa-X",
#                     stats_pwa["mean"], stats_pwa["median"], stats_pwa["p95"]])

#     # ======================================
#     # 7. GPR residual (只训练水平 GPR)
#     # ======================================
#     resid_x = origin_tgt[:,0] - origin_sim[:,0]
#     gpr_kernel = SK_RBF(length_scale=200.0) + WhiteKernel(noise_level=1.0)
#     gpr_x = GaussianProcessRegressor(
#         kernel=gpr_kernel, alpha=1e-2, normalize_y=True
#     ).fit(origin_obs, resid_x)

#     gpr_res_x = gpr_x.predict(test_obs)
#     gpr_pred = pred_sim.copy()
#     gpr_pred[:,0] += gpr_res_x

#     stats_gpr = eval_stats_x(gpr_pred, test_tgt)
#     results.append(["sim+gpr-X",
#                     stats_gpr["mean"], stats_gpr["median"], stats_gpr["p95"]])

#     # === 最终输出 DataFrame ===
#     df = pd.DataFrame(results, columns=["model", "mean", "median", "p95"])
#     return df

# def run_one_session_vertical(origin_path, test_path):
#     """
#     只分析垂直误差（y方向）。
#     输出 DataFrame:
#         columns = ["model", "mean", "median", "p95"]
#     """
#     # === 1. load data ===
#     origin_df = pd.read_csv(origin_path)
#     test_df   = pd.read_csv(test_path)

#     origin_obs = origin_df[['original_gaze_x','original_gaze_y']].values
#     origin_tgt = origin_df[['target_x','target_y']].values
#     test_obs   = test_df[['original_gaze_x','original_gaze_y']].values
#     test_tgt   = test_df[['target_x','target_y']].values

#     results = []

#     # ======================================
#     # eval: only vertical (y) error
#     # ======================================
#     def eval_stats_y(pred_xy, tgt_xy):
#         err = np.abs(pred_xy[:,1] - tgt_xy[:,1])   # vertical error only
#         return {
#             "mean":   float(np.mean(err)),
#             "median": float(np.median(err)),
#             "p95":    float(np.percentile(err,95))
#         }

#     # ======================================
#     # 2. Similarity baseline (still 2D transform)
#     # ======================================
#     s, R, t = fit_similarity(origin_obs, origin_tgt)
#     pred_sim = apply_similarity(test_obs, s, R, t)
#     stats_sim = eval_stats_y(pred_sim, test_tgt)
#     results.append(["similarity-Y",
#                     stats_sim["mean"], stats_sim["median"], stats_sim["p95"]])

#     # ======================================
#     # 3. Polynomial (only dy)
#     # ======================================
#     if {'original_dx','original_dy'}.issubset(origin_df.columns):
#         class Poly1D_Y:
#             def __init__(self, df, deg=POLY_DEGREE, alpha=POLY_ALPHA):
#                 X = df[['target_x','target_y']].values
#                 dy = df['original_dy'].values  # only vertical residual
#                 self.poly = PolynomialFeatures(degree=deg, include_bias=False)
#                 Xp = self.poly.fit_transform(X)
#                 self.model_dy = Ridge(alpha=alpha).fit(Xp, dy)

#             def predict_delta_y(self, pts):
#                 Xp = self.poly.transform(pts)
#                 return self.model_dy.predict(Xp)

#             def correct_batch(self, gaze_xy):
#                 out=[]
#                 for (gx,gy) in gaze_xy:
#                     T=np.array([gx,gy],dtype=float)
#                     Gy=gy
#                     for _ in range(20):
#                         dy = self.predict_delta_y(T.reshape(1,2))[0]
#                         F = (T[1] + dy) - Gy
#                         T[1] -= 0.7*F
#                         if abs(F)<0.5:
#                             break
#                     out.append([gx, T[1]])  # only modify y
#                 return np.array(out)

#         poly = Poly1D_Y(origin_df)
#         poly_pred = poly.correct_batch(test_obs)
#         stats_poly = eval_stats_y(poly_pred, test_tgt)
#         results.append([f"poly-Y(deg={POLY_DEGREE})",
#                         stats_poly["mean"], stats_poly["median"], stats_poly["p95"]])
#     else:
#         poly = None

#     # ======================================
#     # 4. Similarity + RBF (only vertical residual)
#     # ======================================
#     origin_sim = apply_similarity(origin_obs, s, R, t)

#     for kernel, smooth in RBF_PARAM_GRID:
#         resid_y = origin_tgt[:,1] - origin_sim[:,1]
#         rbf_y = Rbf(origin_obs[:,0], origin_obs[:,1], resid_y,
#                     function=kernel, smooth=smooth)

#         res_y = rbf_y(test_obs[:,0], test_obs[:,1])
#         rbf_pred = pred_sim.copy()
#         rbf_pred[:,1] += res_y  # only modify y

#         stats_rbf = eval_stats_y(rbf_pred, test_tgt)
#         results.append([f"sim+rbf-Y[{kernel},s={smooth}]",
#                         stats_rbf["mean"], stats_rbf["median"], stats_rbf["p95"]])

#     # ======================================
#     # 5. TPS residual (2D, but only Y used)
#     # ======================================
#     tps = ThinPlateSpline2D(origin_sim, origin_tgt, reg=1e-3)
#     tps_pred = tps.transform(pred_sim)
#     stats_tps = eval_stats_y(tps_pred, test_tgt)
#     results.append(["sim+tps-Y",
#                     stats_tps["mean"], stats_tps["median"], stats_tps["p95"]])

#     # ======================================
#     # 6. Piecewise Affine (2D, but only Y used)
#     # ======================================
#     pwa = PiecewiseAffine(origin_sim, origin_tgt)
#     pwa_pred = pwa.transform(pred_sim)
#     stats_pwa = eval_stats_y(pwa_pred, test_tgt)
#     results.append(["sim+pwa-Y",
#                     stats_pwa["mean"], stats_pwa["median"], stats_pwa["p95"]])

#     # ======================================
#     # 7. GPR residual (only vertical GPR)
#     # ======================================
#     resid_y = origin_tgt[:,1] - origin_sim[:,1]
#     gpr_kernel = SK_RBF(length_scale=200.0) + WhiteKernel(noise_level=1.0)
#     gpr_y = GaussianProcessRegressor(
#         kernel=gpr_kernel, alpha=1e-2, normalize_y=True
#     ).fit(origin_obs, resid_y)

#     gpr_res_y = gpr_y.predict(test_obs)
#     gpr_pred = pred_sim.copy()
#     gpr_pred[:,1] += gpr_res_y

#     stats_gpr = eval_stats_y(gpr_pred, test_tgt)
#     results.append(["sim+gpr-Y",
#                     stats_gpr["mean"], stats_gpr["median"], stats_gpr["p95"]])

#     # === Final dataframe ===
#     df = pd.DataFrame(results, columns=["model", "mean", "median", "p95"])
#     return df

# -----------------------------
# main pipeline
# -----------------------------
def main():
    origin_df, test_df = load_or_split()
    origin_obs = origin_df[['original_gaze_x','original_gaze_y']].values
    origin_tgt = origin_df[['target_x','target_y']].values
    test_obs = test_df[['original_gaze_x','original_gaze_y']].values
    test_tgt = test_df[['target_x','target_y']].values
    

    # Similarity baseline (train on origin)
    s, R, t = fit_similarity(origin_obs, origin_tgt)
    sim_test_pred = apply_similarity(test_obs, s, R, t)
    stats = {}
    stats['similarity'] = eval_stats(sim_test_pred, test_tgt)
    print("Similarity:", stats['similarity'])

    # Polynomial (train on origin)
    # Expect origin_df to have original_dx, original_dy columns (residuals)
    poly = SimplePolynomialCalibrator(origin_df, degree=POLY_DEGREE, reg=POLY_ALPHA)
    poly_pred = poly.correct_batch(test_obs)
    stats['polynomial'] = eval_stats(poly_pred, test_tgt)
    print("Polynomial:", stats['polynomial'])

    # Similarity + RBF (grid)
    origin_sim = apply_similarity(origin_obs, s, R, t)
    for kernel, smooth in RBF_PARAM_GRID:
        print(f"\n--- RBF kernel={kernel} smooth={smooth} ---")
        rbf_x, rbf_y = fit_rbf_residual(origin_obs, origin_sim, origin_tgt, kernel=kernel, smooth=smooth)
        rbf_test_pred = apply_rbf_to_points(test_obs, sim_test_pred, rbf_x, rbf_y)
        name = f"sim+rbf[{kernel},s={smooth}]"
        stats[name] = eval_stats(rbf_test_pred, test_tgt)
        print(name, stats[name])

    # Similarity + TPS (fit tps mapping from sim(origin_obs) -> origin_tgt)
    print("\n--- TPS residual ---")
    # Here we model mapping: mapped_point = TPS(origin_sim) (absolute mapping to tgt)
    tps = ThinPlateSpline2D(origin_sim, origin_tgt, reg=1e-3)
    tps_test_pred = tps.transform(sim_test_pred)
    stats['sim+tps'] = eval_stats(tps_test_pred, test_tgt)
    print("sim+tps", stats['sim+tps'])

    # Similarity + Piecewise Affine
    print("\n--- Piecewise Affine ---")
    pwa = PiecewiseAffine(origin_sim, origin_tgt)
    pwa_test_pred = pwa.transform(sim_test_pred)
    stats['sim+pwa'] = eval_stats(pwa_test_pred, test_tgt)
    print("sim+pwa", stats['sim+pwa'])

    # Similarity + GPR residual
    print("\n--- GPR residual ---")
    gpr_x, gpr_y = fit_gpr_residual(origin_obs, origin_sim, origin_tgt)
    gpr_pred, gpr_std = apply_gpr(test_obs, sim_test_pred, gpr_x, gpr_y)
    stats['sim+gpr'] = eval_stats(gpr_pred, test_tgt)
    print("sim+gpr", stats['sim+gpr'])

    # Plotting: arrow fields for chosen models
    plot_arrow_field(test_df, sim_test_pred, "Similarity baseline")
    plot_arrow_field(test_df, poly_pred, f"Polynomial degree={POLY_DEGREE}")
    for k in list(stats.keys()):
        if k.startswith("sim+rbf"):
            plot_arrow_field(test_df, apply_rbf_to_points(test_obs, sim_test_pred, rbf_x, rbf_y),
                             f"{k}")
    plot_arrow_field(test_df, tps_test_pred, "Similarity + TPS")
    plot_arrow_field(test_df, pwa_test_pred, "Similarity + PiecewiseAffine")
    plot_arrow_field(test_df, gpr_pred, "Similarity + GPR")

    # # CDF plot
    # plot_error_cdf(stats)

    # print summary table
    print("\n=== Summary table (mean/median/p95 px) ===")
    for k,v in stats.items():
        print(f"{k:20s}  mean={v['mean']:.2f}  median={v['median']:.2f}  p95={v['p95']:.2f}")

if __name__ == "__main__":
    main()
