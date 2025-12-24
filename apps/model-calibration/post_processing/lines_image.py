import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取你的文件
df = pd.read_csv("C:\\Users\\SCCN\\Desktop\\systematic_recalibration\\Liu Jiaqi\\2025-11-18_18-43-23\\vertical\\reading_alignment.csv")

# 按时间排序（很重要，保证轨迹连续）
df = df.sort_values("timestamp")

# --- 过滤：移除 dot-gaze 距离超过 500px 的点 ---
dist = np.sqrt((df["dot_x"] - df["gaze_x"])**2 + (df["dot_y"] - df["gaze_y"])**2)
df = df[dist <= 1500].copy()

print(f"过滤后的点数: {len(df)}")

# 提取数据
dot_x = df["dot_x"]
dot_y = df["dot_y"]
gaze_x = df["gaze_x"]
gaze_y = df["gaze_y"]

plt.figure(figsize=(12, 6))

# 画真实红点轨迹
plt.plot(dot_x, dot_y, '-o', color='blue', markersize=2, label='Dot (Ground Truth)')

# 画眼动仪 gaze 轨迹
plt.plot(gaze_x, gaze_y, '-o', color='red', markersize=2, label='Gaze (Eye Tracker)')

plt.gca().invert_yaxis()  # 屏幕坐标系 y 轴向下，需要反转
plt.xlabel("X (px)")
plt.ylabel("Y (px)")
plt.title("Dot Trajectory vs Gaze Trajectory")
plt.legend()
plt.grid(True)

plt.show()
# ===========================
# 1. 过滤：移除 dot-gaze 距离 > 500 px
# ===========================
dist = np.sqrt((df["dot_x"] - df["gaze_x"])**2 +
               (df["dot_y"] - df["gaze_y"])**2)
df = df[dist <= 500].copy()

# ===========================
# 2. 过滤：去除每行 dot_x 两端的 100 px 区域
# ===========================
filtered_rows = []

for line_id, sub in df.groupby("line_idx"):
    min_x = sub["dot_x"].min()
    max_x = sub["dot_x"].max()

    sub_filt = sub[
        (sub["dot_x"] >= min_x + 100) &
        (sub["dot_x"] <= max_x - 100)
    ].copy()

    filtered_rows.append(sub_filt)

df = pd.concat(filtered_rows, ignore_index=True)

print(f"过滤后剩余点数: {len(df)}")

# ===========================
# 3. 逐行作图（不连接不同的行）
# ===========================
plt.figure(figsize=(12, 6))

for line_id, sub in df.groupby("line_idx"):
    plt.plot(sub["dot_x"], sub["dot_y"], '-o',
             markersize=2, color='blue', label='Dot' if line_id==0 else "")
    plt.plot(sub["gaze_x"], sub["gaze_y"], '-o',
             markersize=2, color='red',  label='Gaze' if line_id==0 else "")

plt.gca().invert_yaxis()
plt.xlabel("X (px)")
plt.ylabel("Y (px)")
plt.title("Dot vs Gaze (Filtered, Per-Line)")
plt.legend()
plt.grid(True)

plt.show()