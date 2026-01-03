import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

ROOT_DIR = Path.home() / "Desktop" / "systematic_recalibration"
OUTPUT_DIR = os.path.join(ROOT_DIR, "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

pattern = re.compile(r"samples_target_(\d+)_before\.csv")

# =======================
# Helper functions
# =======================
def find_first_all_inside(xs, ys, cx, cy, r):
    dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    inside = dists <= r
    for k in range(len(inside)):
        if inside[k:].all():
            return k
    return np.nan


def plot_sample(xs, ys, cx, cy, r, k, title, save_path):
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, "-o", alpha=0.6, label="Samples")

    circle = plt.Circle((cx, cy), r, fill=False, color="r", label="3σ region")
    plt.gca().add_patch(circle)

    plt.scatter(cx, cy, c="red", marker="+", s=120, label="Original gaze")

    if not np.isnan(k):
        k = int(k)
        plt.scatter(xs[k], ys[k], c="green", s=100, label=f"k={k}")

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


# =======================
# Batch scanning
# =======================
records = []

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
                f for f in os.listdir(data_path) if pattern.match(f)
            ]
            if not sample_files:
                continue

            grid_df = pd.read_csv(grid_path)
            grid_df = grid_df.set_index("target_index")

            for fname in sample_files:
                target_idx = int(pattern.match(fname).group(1))
                sample_df = pd.read_csv(os.path.join(data_path, fname))

                row = grid_df.loc[target_idx]
                cx = row["original_gaze_x"]
                cy = row["original_gaze_y"]
                r = 3 * row["spread"]   # 🔥 3σ

                k = find_first_all_inside(
                    sample_df["x"].values,
                    sample_df["y"].values,
                    cx,
                    cy,
                    r,
                )

                records.append(
                    {
                        "subject": subject,
                        "timestamp": timestamp,
                        "data_type": data_type,
                        "target_index": target_idx,
                        "first_valid_index": k,
                        "data_path": data_path,
                        "sample_file": fname,
                    }
                )

# =======================
# Save summary table
# =======================
df = pd.DataFrame(records)
summary_path = os.path.join(OUTPUT_DIR, "first_valid_index_summary.csv")
df.to_csv(summary_path, index=False)

# =======================
# Histogram (overall)
# =======================
total_n = len(df)
nan_n = df["first_valid_index"].isna().sum()
nan_ratio = nan_n / total_n * 100

plt.figure()
df["first_valid_index"].dropna().astype(int).hist(bins=30)
plt.xlabel("first_valid_index")
plt.ylabel("count")
plt.title("Overall first_valid_index distribution")

plt.text(
    0.95,
    0.95,
    f"Not in 3σ: {nan_n} / {total_n}\n({nan_ratio:.1f}%)",
    transform=plt.gca().transAxes,
    ha="right",
    va="top",
    fontsize=11,
    bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
)

plt.savefig(os.path.join(OUTPUT_DIR, "hist_overall.png"))
plt.close()

# =======================
# Histograms per participant
# =======================
for subject, g in df.groupby("subject"):
    total_n = len(g)
    nan_n = g["first_valid_index"].isna().sum()
    nan_ratio = nan_n / total_n * 100

    plt.figure()
    g["first_valid_index"].dropna().astype(int).hist(bins=30)
    plt.xlabel("first_valid_index")
    plt.ylabel("count")
    plt.title(f"{subject} first_valid_index distribution")

    plt.text(
        0.95,
        0.95,
        f"Not in 3σ: {nan_n} / {total_n}\n({nan_ratio:.1f}%)",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
    )

    plt.savefig(os.path.join(OUTPUT_DIR, f"hist_{subject}.png"))
    plt.close()

# =======================
# Quantile example plots
# =======================
valid_df = df.dropna(subset=["first_valid_index"])
quantiles = valid_df["first_valid_index"].quantile([0.25, 0.504, 0.752])

for q, val in quantiles.items():
    row = valid_df.iloc[(valid_df["first_valid_index"] - val).abs().argmin()]

    sample_df = pd.read_csv(
        os.path.join(row["data_path"], row["sample_file"])
    )
    grid_df = pd.read_csv(
        os.path.join(row["data_path"], "grid_gaze_log.csv")
    ).set_index("target_index")

    grid_row = grid_df.loc[row["target_index"]]
    cx, cy = grid_row["original_gaze_x"], grid_row["original_gaze_y"]
    r = 3 * grid_row["spread"]

    plot_sample(
        sample_df["x"].values,
        sample_df["y"].values,
        cx,
        cy,
        r,
        row["first_valid_index"],
        f"{int(q * 100)}% quantile example",
        os.path.join(OUTPUT_DIR, f"quantile_{int(q * 100)}.png"),
    )

print("✅ Batch processing complete")
