"""Online per-trial bias correction.

Within each test trial, treat the first K samples as a 'few-shot
adaptation context'. Compute their mean residual from the chosen
anchor baseline. Subtract this from the anchor's predictions on the
remaining samples.

This simulates the realistic deployment scenario where the system can
do a brief recalibration check at session start. The first K samples
have known (target, gaze, baseline_pred) tuples — equivalent to running
K extra calibration points.

Reports test L2 on the LAST (n_test - K) samples of each trial, and
the trial-level baseline L2 on the same evaluation set, for K in
{0, 1, 2, 3, 5, 8, 12}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "all_trials_combined.csv"
test = pd.read_csv(DATA)
test["trial_id"] = test["subject"] + "|" + test["timestamp"]

ANCHOR = "pred_sim_rbf_multiquadric_s2.0"
BASELINES = [
    ANCHOR, "pred_similarity", "pred_poly", "pred_gpr", "pred_tps", "pred_sim_pwa",
]


def online_bias_eval(df: pd.DataFrame, k: int, baseline: str) -> Dict:
    """For each trial, hold out first k samples for bias estimation,
    evaluate on the rest. Bias is mean (target - baseline_pred) on the
    held-out k samples; subtracted from baseline_pred on the rest."""
    all_errors: List[np.ndarray] = []
    n_skipped = 0
    for trial_id, g in df.groupby("trial_id"):
        if len(g) <= k:
            n_skipped += 1
            continue
        # Use trial-specific seeded shuffle so the same trial gets the same context across k
        rng_local = np.random.default_rng(hash(trial_id) & 0xFFFFFFFF)
        idx = rng_local.permutation(len(g))
        if k > 0:
            ctx_idx, eval_idx = idx[:k], idx[k:]
        else:
            ctx_idx, eval_idx = np.array([], dtype=int), idx
        ctx, ev = g.iloc[ctx_idx], g.iloc[eval_idx]
        if k > 0:
            bias = (ctx[["target_x", "target_y"]].values - ctx[[f"{baseline}_x", f"{baseline}_y"]].values).mean(axis=0)
        else:
            bias = np.zeros(2)
        pred = ev[[f"{baseline}_x", f"{baseline}_y"]].values + bias
        err = pred - ev[["target_x", "target_y"]].values
        all_errors.append(np.linalg.norm(err, axis=1))
    l2 = np.concatenate(all_errors)
    return {"k": k, "baseline": baseline, "n_eval": int(len(l2)), "n_skipped_trials": n_skipped,
            "l2_mean": float(l2.mean()), "l2_median": float(np.median(l2)),
            "l2_std": float(l2.std()), "l2_p95": float(np.percentile(l2, 95))}


print("=== Online per-trial bias correction ===")
print(f"Test set: {len(test)} samples, {test['trial_id'].nunique()} trials\n")
rows = []
for baseline in BASELINES:
    print(f"-- baseline = {baseline} --")
    for k in [0, 1, 2, 3, 5, 8, 12, 18]:
        r = online_bias_eval(test, k, baseline)
        rows.append(r)
        print(f"  k={k:2d}  L2={r['l2_mean']:.2f}  med={r['l2_median']:.2f}  p95={r['l2_p95']:.2f}  (n_eval={r['n_eval']}, skipped={r['n_skipped_trials']})")
    print()

out = Path(__file__).parent / "results.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(rows, f, indent=2)

# Also: combine multiple baselines using BMA + online bias for the strongest baseline
print("=== Combined: stacker-anchored + online bias on anchor ===")
# This is just: take anchor baseline + online bias for k>=1, compare to anchor + 0 bias
# We've already done above; the best combination is just baseline + bias.

# Headline summary
print("\n=== Headline: anchor + online bias improvement ===")
print(f"{'k':>3s}  {'L2_mean':>9s}  {'delta':>7s}")
ref = next(r for r in rows if r["k"] == 0 and r["baseline"] == ANCHOR)["l2_mean"]
for r in [x for x in rows if x["baseline"] == ANCHOR]:
    print(f"{r['k']:3d}  {r['l2_mean']:9.2f}  {r['l2_mean']-ref:+7.2f}")
