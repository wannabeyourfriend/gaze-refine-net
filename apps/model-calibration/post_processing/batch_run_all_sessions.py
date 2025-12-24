import os
import numpy as np
import pandas as pd

# ===== 导入你现有的模型实现 =====
from calibration_model_full_compare import (
    fit_similarity, apply_similarity,
    SimplePolynomialCalibrator,
    ThinPlateSpline2D,
    PiecewiseAffine,
    fit_rbf_residual, apply_rbf_to_points,
    fit_gpr_residual, apply_gpr
)

ROOT_DIR = r"C:\Users\Liu Jiaqi\Desktop\systematic_recalibration"

# ==========================
# 单试次处理
# ==========================
def process_one_trial(subject, timestamp, origin_path, test_path):
    origin_df = pd.read_csv(origin_path)
    test_df   = pd.read_csv(test_path)

    origin_obs = origin_df[['original_gaze_x','original_gaze_y']].values
    origin_tgt = origin_df[['target_x','target_y']].values
    test_obs   = test_df[['original_gaze_x','original_gaze_y']].values
    test_tgt   = test_df[['target_x','target_y']].values

    out = pd.DataFrame({
        "subject": subject,
        "timestamp": timestamp,
        "target_index": test_df.get("target_index", np.nan),
        "origin_gaze_x": test_obs[:,0],
        "origin_gaze_y": test_obs[:,1],
        "target_x": test_tgt[:,0],
        "target_y": test_tgt[:,1],
        "spread": test_df.get("spread", np.nan),
        "n_samples": test_df.get("n_samples", np.nan)
    })

    # ===== Similarity =====
    s, R, t = fit_similarity(origin_obs, origin_tgt)
    sim_pred = apply_similarity(test_obs, s, R, t)
    out["pred_similarity_x"] = sim_pred[:,0]
    out["pred_similarity_y"] = sim_pred[:,1]

    # ===== Polynomial =====
    poly = SimplePolynomialCalibrator(origin_df)
    poly_pred = poly.correct_batch(test_obs)
    out["pred_poly_x"] = poly_pred[:,0]
    out["pred_poly_y"] = poly_pred[:,1]

    # ===== Similarity + RBF (multiple params) =====
    from calibration_model_full_compare import RBF_PARAM_GRID

    origin_sim = apply_similarity(origin_obs, s, R, t)

    for kernel, smooth in RBF_PARAM_GRID:
        rbf_x, rbf_y = fit_rbf_residual(
            origin_obs, origin_sim, origin_tgt,
            kernel=kernel, smooth=smooth
        )

        rbf_pred = apply_rbf_to_points(
            test_obs, sim_pred, rbf_x, rbf_y
        )

        col_prefix = f"pred_sim_rbf_{kernel}_s{smooth}"
        out[f"{col_prefix}_x"] = rbf_pred[:, 0]
        out[f"{col_prefix}_y"] = rbf_pred[:, 1]

    # ===== Similarity + TPS =====
    origin_sim = apply_similarity(origin_obs, s, R, t)
    tps = ThinPlateSpline2D(origin_sim, origin_tgt)
    tps_pred = tps.transform(sim_pred)
    out["pred_sim_tps_x"] = tps_pred[:,0]
    out["pred_sim_tps_y"] = tps_pred[:,1]

    # ===== Similarity + Piecewise Affine =====
    pwa = PiecewiseAffine(origin_sim, origin_tgt)
    pwa_pred = pwa.transform(sim_pred)
    out["pred_sim_pwa_x"] = pwa_pred[:,0]
    out["pred_sim_pwa_y"] = pwa_pred[:,1]

    # ===== Similarity + GPR =====
    gpr_x, gpr_y = fit_gpr_residual(origin_obs, origin_sim, origin_tgt)
    gpr_pred, _ = apply_gpr(test_obs, sim_pred, gpr_x, gpr_y)
    out["pred_sim_gpr_x"] = gpr_pred[:,0]
    out["pred_sim_gpr_y"] = gpr_pred[:,1]

    return out


# ==========================
# 批量扫描全部试次
# ==========================
all_rows = []

for subject in os.listdir(ROOT_DIR):
    subject_path = os.path.join(ROOT_DIR, subject)
    if not os.path.isdir(subject_path):
        continue

    for timestamp in os.listdir(subject_path):
        ts_path = os.path.join(subject_path, timestamp)
        if not os.path.isdir(ts_path):
            continue

        origin_path = os.path.join(ts_path, "origin", "grid_gaze_log.csv")
        test_path   = os.path.join(ts_path, "test",   "grid_gaze_log.csv")

        if not (os.path.exists(origin_path) and os.path.exists(test_path)):
            continue

        print(f"Processing {subject} | {timestamp}")
        df_trial = process_one_trial(
            subject, timestamp,
            origin_path, test_path
        )
        all_rows.append(df_trial)

# ==========================
# 汇总输出 CSV
# ==========================
final_df = pd.concat(all_rows, ignore_index=True)
out_csv = os.path.join(ROOT_DIR, "all_trials_model_predictions.csv")
final_df.to_csv(out_csv, index=False)

# ==========================
# 终端输出误差统计
# ==========================
print("\n===== Overall model error statistics =====")

models = {
    "similarity": ("pred_similarity_x", "pred_similarity_y"),
    "poly": ("pred_poly_x", "pred_poly_y"),

    "sim+rbf[thin_plate,s=1.0]":
        ("pred_sim_rbf_thin_plate_s1.0_x", "pred_sim_rbf_thin_plate_s1.0_y"),
    "sim+rbf[multiquadric,s=0.0]":
        ("pred_sim_rbf_multiquadric_s0.0_x", "pred_sim_rbf_multiquadric_s0.0_y"),
    "sim+rbf[multiquadric,s=1.0]":
        ("pred_sim_rbf_multiquadric_s1.0_x", "pred_sim_rbf_multiquadric_s1.0_y"),
    "sim+rbf[multiquadric,s=2.0]":
        ("pred_sim_rbf_multiquadric_s2.0_x", "pred_sim_rbf_multiquadric_s2.0_y"),

    "sim+tps": ("pred_sim_tps_x", "pred_sim_tps_y"),
    "sim+pwa": ("pred_sim_pwa_x", "pred_sim_pwa_y"),
    "sim+gpr": ("pred_sim_gpr_x", "pred_sim_gpr_y"),
}


for name, (px, py) in models.items():
    pred = final_df[[px, py]].values
    tgt  = final_df[["target_x","target_y"]].values
    err = np.linalg.norm(pred - tgt, axis=1)
    print(
        f"{name:10s} | mean = {err.mean():.2f} px | var = {err.var():.2f}"
    )

# ==========================
# 各被试分别的终端输出
# ==========================
print("\n===== Per-subject model error statistics =====")

for subject, g in final_df.groupby("subject"):
    print(f"\n--- Subject: {subject} ---")

    tgt = g[["target_x", "target_y"]].values

    for name, (px, py) in models.items():
        pred = g[[px, py]].values
        err = np.linalg.norm(pred - tgt, axis=1)

        print(
            f"{name:10s} | mean = {err.mean():.2f} px | var = {err.var():.2f}"
        )

print("\n✅ 批量试次处理完成")
