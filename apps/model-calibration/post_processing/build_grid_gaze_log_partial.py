import os
import re
import numpy as np
import pandas as pd

ROOT_DIR = r"C:\Users\Liu Jiaqi\Desktop\systematic_recalibration"

sample_pattern = re.compile(r"samples_target_(\d+)\.csv")

RATIOS = {
    "1s": 0.25,
    "2s": 0.50,
    "3s": 0.75
}

# =======================
# 工具函数
# =======================
def compute_gaze_stats(xs, ys):
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    spread = np.sqrt(np.var(xs) + np.var(ys))
    return mean_x, mean_y, spread


# =======================
# 扫描三层目录
# =======================
for subject in os.listdir(ROOT_DIR):
    subject_path = os.path.join(ROOT_DIR, subject)
    if not os.path.isdir(subject_path):
        continue

    for timestamp in os.listdir(subject_path):
        ts_path = os.path.join(subject_path, timestamp)
        if not os.path.isdir(ts_path):
            continue

        for data_type in os.listdir(ts_path):
            data_path = os.path.join(ts_path, data_type)
            if not os.path.isdir(data_path):
                continue

            grid_path = os.path.join(data_path, "grid_gaze_log.csv")
            if not os.path.exists(grid_path):
                continue

            sample_files = [
                f for f in os.listdir(data_path)
                if sample_pattern.match(f)
            ]
            if not sample_files:
                continue

            print(f"Processing: {data_path}")

            # 只读取需要迁移的 4 列
            grid_df = pd.read_csv(
                grid_path,
                usecols=["timestamp", "target_index", "target_x", "target_y"]
            )

            # 建立 samples 映射
            sample_map = {}
            for fname in sample_files:
                idx = int(sample_pattern.match(fname).group(1))
                sample_map[idx] = pd.read_csv(
                    os.path.join(data_path, fname)
                )

            # 针对每个比例生成新 grid
            for tag, ratio in RATIOS.items():
                rows = []

                for _, row in grid_df.iterrows():
                    target_idx = row["target_index"]

                    base_row = {
                        "timestamp": row["timestamp"],
                        "target_index": target_idx,
                        "target_x": row["target_x"],
                        "target_y": row["target_y"],
                    }

                    if target_idx in sample_map:
                        df_s = sample_map[target_idx]
                        N = len(df_s)
                        use_n = max(1, int(N * ratio))

                        xs = df_s["x"].values[:use_n]
                        ys = df_s["y"].values[:use_n]

                        mean_x, mean_y, spread = compute_gaze_stats(xs, ys)

                        base_row.update({
                            "original_gaze_x": mean_x,
                            "original_gaze_y": mean_y,
                            "spread": spread,
                            "n_samples": use_n
                        })
                    else:
                        base_row.update({
                            "original_gaze_x": np.nan,
                            "original_gaze_y": np.nan,
                            "spread": np.nan,
                            "n_samples": 0
                        })

                    rows.append(base_row)

                new_df = pd.DataFrame(rows)

                out_path = os.path.join(
                    data_path, f"grid_gaze_log_{tag}.csv"
                )
                new_df.to_csv(out_path, index=False)

print("✅ 精简字段后的 grid_gaze_log_1s / 2s / 3s 已生成")
