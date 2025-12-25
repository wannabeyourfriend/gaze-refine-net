# import os
# import pandas as pd
# from pathlib import Path

# from calibration_model_full_compare import (
#     fit_similarity, apply_similarity,
#     fit_rbf_residual, apply_rbf_to_points,
#     ThinPlateSpline2D, PiecewiseAffine,
#     fit_gpr_residual, apply_gpr
# )


# def process_one_session(origin_path, test_path):
#     """对单个 subject 进行 origin→test 模型拟合 + 校正"""
#     origin = pd.read_csv(origin_path)
#     test = pd.read_csv(test_path)

#     origin_obs = origin[['original_gaze_x', 'original_gaze_y']].values
#     origin_tgt = origin[['target_x', 'target_y']].values
#     test_obs   = test[['original_gaze_x', 'original_gaze_y']].values

#     # ---- 1) similarity baseline ----
#     s, R, t = fit_similarity(origin_obs, origin_tgt)
#     pred_sim_test = apply_similarity(test_obs, s, R, t)
#     pred_sim_origin = apply_similarity(origin_obs, s, R, t)

#     # ---- 2) sim + RBF ----
#     rbf_x, rbf_y = fit_rbf_residual(origin_obs, pred_sim_origin, origin_tgt, kernel="multiquadric", smooth=1.0)
#     sim_rbf_pred = apply_rbf_to_points(test_obs, pred_sim_test, rbf_x, rbf_y)

#     # ---- 3) sim + PWA ----
#     pwa = PiecewiseAffine(pred_sim_origin, origin_tgt)
#     sim_pwa_pred = pwa.transform(pred_sim_test)

#     # ---- 4) sim + GPR ----
#     gpr_x, gpr_y = fit_gpr_residual(origin_obs, pred_sim_origin, origin_tgt)
#     sim_gpr_pred, _ = apply_gpr(test_obs, pred_sim_test, gpr_x, gpr_y)

#     # ---- 添加到 test df ----
#     test['sim_rbf_gaze_x'] = sim_rbf_pred[:, 0]
#     test['sim_rbf_gaze_y'] = sim_rbf_pred[:, 1]

#     test['sim_pwa_gaze_x'] = sim_pwa_pred[:, 0]
#     test['sim_pwa_gaze_y'] = sim_pwa_pred[:, 1]

#     test['sim_gpr_gaze_x'] = sim_gpr_pred[:, 0]
#     test['sim_gpr_gaze_y'] = sim_gpr_pred[:, 1]

#     return test


# def find_sessions(root_dir):
#     """寻找所有 subject 的 origin / test"""
#     root = Path(root_dir)
#     sessions = []

#     for subject in root.iterdir():
#         if not subject.is_dir():
#             continue

#         # 搜索二级子目录
#         for sub in subject.rglob("*"):
#             origin = sub / "origin" / "grid_gaze_log.csv"
#             test   = sub / "test"   / "grid_gaze_log.csv"

#             if origin.exists() and test.exists():
#                 sessions.append((subject.name, origin, test))

#     return sessions


# def main(root_dir):
#     # 直接保留所有数据
#     all_data = []

#     sessions = find_sessions(root_dir)
#     print(f"Found {len(sessions)} sessions")

#     for subject_name, origin_path, test_path in sessions:
#         print(f"Processing {subject_name} ...")
#         df = process_one_session(origin_path, test_path)
#         df['subject_name'] = subject_name
#         all_data.append(df)

#     final = pd.concat(all_data, ignore_index=True)

#     out_path = Path(root_dir) / "final_all_subjects.csv"
#     final.to_csv(out_path, index=False)
#     print("Saved:", out_path)

#     return final

# # def main(root_dir):
# #     # 过滤拟合数据
# #     all_data = []

# #     sessions = find_sessions(root_dir)
# #     print(f"Found {len(sessions)} sessions")

# #     for subject_name, origin_path, test_path in sessions:
# #         print(f"Processing {subject_name} ...")

# #         # ---- 读取 origin & test ----
# #         origin_df = pd.read_csv(origin_path)
# #         test_df = pd.read_csv(test_path)

# #         # ---- 去除 test 中 target_x,target_y 在 origin 中已有的点 ----
# #         origin_targets = set(map(tuple, origin_df[['target_x', 'target_y']].values))
# #         mask = test_df[['target_x', 'target_y']].apply(tuple, axis=1).isin(origin_targets)
# #         test_df_filtered = test_df[~mask].reset_index(drop=True)

# #         # ---- 将过滤后的 test 保存为临时文件并继续流程 ----
# #         tmp_path = test_path.with_name("tmp_filtered_test.csv")
# #         test_df_filtered.to_csv(tmp_path, index=False)

# #         # ---- 主处理 ----
# #         df = process_one_session(origin_path, tmp_path)
# #         df['subject_name'] = subject_name
# #         all_data.append(df)

# #         # 删除临时文件
# #         os.remove(tmp_path)

# #     final = pd.concat(all_data, ignore_index=True)

# #     out_path = Path(root_dir) / "final_all_subjects.csv"
# #     final.to_csv(out_path, index=False)
# #     print("Saved:", out_path)
# #     return final

# if __name__ == "__main__":
#     main("C:\\Users\\SCCN\\Desktop\\systematic_recalibration")


import pandas as pd
import numpy as np

# 读取你已经上传的原文件
df = pd.read_csv("C:\\Users\\SCCN\\Desktop\\systematic_recalibration\\final_all_subjects.csv")

# ============================
# 筛选数据：距离 > 300 px 去掉
# ============================
dist = np.sqrt((df["original_gaze_x"] - df["target_x"])**2 +
               (df["original_gaze_y"] - df["target_y"])**2)

df = df[dist <= 300].reset_index(drop=True)

# ============================
# 添加 poly / rbf / pwa / gpr 残差（4个模型 × 2维 = 8维输出）
# ============================

df["poly_dx"] = df["poly_gaze_x"] - df["target_x"]
df["poly_dy"] = df["poly_gaze_y"] - df["target_y"]

df["sim_rbf_dx"] = df["sim_rbf_gaze_x"] - df["target_x"]
df["sim_rbf_dy"] = df["sim_rbf_gaze_y"] - df["target_y"]

df["sim_pwa_dx"] = df["sim_pwa_gaze_x"] - df["target_x"]
df["sim_pwa_dy"] = df["sim_pwa_gaze_y"] - df["target_y"]

df["sim_gpr_dx"] = df["sim_gpr_gaze_x"] - df["target_x"]
df["sim_gpr_dy"] = df["sim_gpr_gaze_y"] - df["target_y"]

# ============================
# 保存新文件
# ============================
df.to_csv("C:\\Users\\SCCN\\learn-res\\data\\all_data_without_filtered.csv", index=False)

print("已生成文件：final_all_subjects_with_poly_residual.csv")
print("数据量:", len(df))