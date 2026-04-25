"""James-Stein-style shrinkage on the per-trial context bias.

For small K, the empirical mean residual is noisy. Shrinking it toward
zero by a factor lambda can reduce MSE. Sweep lambda for each K.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "all_trials_combined.csv"

ANCHOR = "pred_sim_rbf_multiquadric_s2.0"

df = pd.read_csv(DATA)
df["trial_id"] = df["subject"] + "|" + df["timestamp"]


def shrink_bias_eval(df: pd.DataFrame, k: int, lam: float) -> dict:
    l2s = []
    for trial_id, g in df.groupby("trial_id"):
        if len(g) <= k:
            continue
        rng = np.random.default_rng(hash(trial_id) & 0xFFFFFFFF)
        idx = rng.permutation(len(g))
        ctx_idx, q_idx = idx[:k], idx[k:]
        ctx, q = g.iloc[ctx_idx], g.iloc[q_idx]
        if k > 0:
            bias = (ctx[["target_x", "target_y"]].values - ctx[[f"{ANCHOR}_x", f"{ANCHOR}_y"]].values).mean(axis=0)
        else:
            bias = np.zeros(2)
        bias = lam * bias
        pred = q[[f"{ANCHOR}_x", f"{ANCHOR}_y"]].values + bias
        l2 = np.linalg.norm(pred - q[["target_x", "target_y"]].values, axis=1)
        l2s.append(l2)
    if not l2s:
        return {"k": k, "lam": lam, "l2_mean": float("nan"), "n": 0}
    l2 = np.concatenate(l2s)
    return {"k": k, "lam": lam, "l2_mean": float(l2.mean()), "l2_median": float(np.median(l2)),
            "l2_p95": float(np.percentile(l2, 95)), "n": int(len(l2))}


# Sweep
results = []
for k in [1, 2, 3, 5, 8, 12, 18]:
    print(f"\n=== K={k} ===")
    for lam in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        r = shrink_bias_eval(df, k, lam)
        results.append(r)
        print(f"  lam={lam:.1f}  L2={r['l2_mean']:6.2f}  med={r['l2_median']:6.2f}  p95={r['l2_p95']:6.2f}  (n={r['n']})")

out = Path(__file__).parent / "results.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(results, f, indent=2)

# Best lambda per K
print("\n=== Best lambda per K ===")
df_r = pd.DataFrame(results)
for k in [1, 2, 3, 5, 8, 12, 18]:
    sub = df_r[df_r["k"] == k]
    best = sub.loc[sub["l2_mean"].idxmin()]
    print(f"K={k:2d}: best_lam={best['lam']:.1f}  L2={best['l2_mean']:.2f}")
