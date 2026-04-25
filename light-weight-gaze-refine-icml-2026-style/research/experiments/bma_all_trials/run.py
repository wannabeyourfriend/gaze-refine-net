"""BMA evaluation on all_trials test split."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bma_pipeline import evaluate_bma

DATA = ROOT.parent.parent / "data" / "prepared" / "all_trials_split"
test = pd.read_csv(DATA / "test.csv")

baselines = [
    "pred_sim_rbf_multiquadric_s2.0",
    "pred_similarity",
    "pred_poly",
    "pred_gpr",
    "pred_tps",
    "pred_sim_pwa",
]

# Reference: each baseline on its own
print("=== Single baseline test L2 ===")
ref = {}
for b in baselines:
    err = np.linalg.norm(
        test[[f"{b}_x", f"{b}_y"]].values - test[["target_x", "target_y"]].values, axis=1
    )
    ref[b] = float(err.mean())
    print(f"  {b:42s} {err.mean():.2f}")

# Naive average of baselines
print("\n=== Uniform average of all baselines ===")
bs_arr = np.stack([test[[f"{b}_x", f"{b}_y"]].values for b in baselines], axis=1)
avg = bs_arr.mean(axis=1)
err = np.linalg.norm(avg - test[["target_x", "target_y"]].values, axis=1)
print(f"  uniform_avg: {err.mean():.2f}")

# BMA at multiple temperatures
print("\n=== BMA (per-trial leave-one-out softmin weights) ===")
rows = []
for T in [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]:
    res = evaluate_bma(test, baselines, temperature=T)
    print(f"  T={T:6.1f}  L2={res['l2_mean']:.2f}  med={res['l2_median']:.2f}  p95={res['l2_p95']:.2f}  weights={[f'{w:.2f}' for w in res['weights_mean']]}")
    rows.append({"temperature": T, **res})

out = Path(__file__).parent / "results.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(rows, f, indent=2)
print(f"\nSaved to {out}")
