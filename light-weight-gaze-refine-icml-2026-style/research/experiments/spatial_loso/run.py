"""LOSO evaluation of v3 spatial shrinkage on the self-collected dataset.

For each held-out subject:
  - Train two models: scalar shrinkage (v1, no spatial term) and spatial
    shrinkage (v3, position-dependent).
  - Train on all other subjects' trials, validate on a 10% trial split.
  - Evaluate at K in {1, 2, 3, 5, 8, 12, 18}.

Also evaluate the parameter-free baselines:
  - anchor (no correction)
  - raw mean bias (lambda = 1)
  - fixed-lambda sweep (best per K)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from spatial_shrinkage import (
    SpatialShrinkageConfig,
    train_spatial_shrinkage,
    evaluate_spatial_shrinkage,
    load_trials_from_csv,
)

DATA = ROOT / "data" / "all_trials_combined.csv"
ANCHOR = "pred_sim_rbf_multiquadric_s2.0"


def fixed_lambda_eval(test_trials, K: int, lam: float) -> float:
    errs = []
    for tr in test_trials:
        T = len(tr["anchor"])
        if T <= K: continue
        rng = np.random.default_rng(hash(tr["trial_id"]) & 0xFFFFFFFF)
        idx = rng.permutation(T)
        ctx_idx, q_idx = idx[:K], idx[K:]
        bar_b = (tr["target"][ctx_idx] - tr["anchor"][ctx_idx]).mean(axis=0)
        pred = tr["anchor"][q_idx] + lam * bar_b
        e = pred - tr["target"][q_idx]
        errs.append(np.linalg.norm(e, axis=1))
    return float(np.concatenate(errs).mean()) if errs else float("nan")


def best_fixed_lambda(test_trials, K: int) -> tuple:
    """Return (best_lam, best_l2)."""
    grid = np.linspace(0.0, 1.0, 21)
    best = (0.0, float("inf"))
    for lam in grid:
        l2 = fixed_lambda_eval(test_trials, K, lam)
        if not np.isnan(l2) and l2 < best[1]:
            best = (float(lam), l2)
    return best


def main():
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_trials = load_trials_from_csv(str(DATA), ANCHOR)
    subjects = sorted({t["subject"] for t in all_trials})
    print(f"Total trials: {len(all_trials)}, subjects: {len(subjects)}")
    for s in subjects:
        n = sum(1 for t in all_trials if t["subject"] == s)
        print(f"  {s:25s}  {n:3d} trials")

    K_list = [1, 2, 3, 5, 8, 12, 18]
    results = []

    for held in subjects:
        sub_trials = [t for t in all_trials if t["subject"] == held]
        rest = [t for t in all_trials if t["subject"] != held]
        # 10% trial-disjoint val
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(rest))
        n_val = max(1, len(rest) // 10)
        val_trials = [rest[i] for i in idx[:n_val]]
        train_trials = [rest[i] for i in idx[n_val:]]

        # ---- baselines (no training) ----
        per_k = {}
        for K in K_list:
            per_k[f"baseline_anchor_K{K}"] = fixed_lambda_eval(sub_trials, K, 0.0)
            per_k[f"baseline_raw_K{K}"]    = fixed_lambda_eval(sub_trials, K, 1.0)
            best_lam, best_l2 = best_fixed_lambda(sub_trials, K)
            per_k[f"baseline_fixed_K{K}_best_lam"] = best_lam
            per_k[f"baseline_fixed_K{K}"]  = best_l2

        # ---- v1: scalar shrinkage (non-spatial) ----
        cfg_v1 = SpatialShrinkageConfig(
            spatial=False, epochs=80,
            ctx_hidden=64, ctx_summary_dim=32, head_hidden=(64, 32),
            screen_w=1920.0, screen_h=1080.0, seed=1047,
        )
        m_v1, _ = train_spatial_shrinkage(train_trials, val_trials, cfg_v1, device=device, verbose=False)
        for K in K_list:
            r = evaluate_spatial_shrinkage(m_v1, sub_trials, K, device=device)
            per_k[f"v1_scalar_K{K}"] = r["l2_mean"]
            per_k[f"v1_scalar_K{K}_n"] = r.get("n_eval", 0)

        # ---- v3: spatial shrinkage ----
        cfg_v3 = SpatialShrinkageConfig(
            spatial=True, epochs=120,
            ctx_hidden=128, ctx_summary_dim=64, head_hidden=(128, 64),
            screen_w=1920.0, screen_h=1080.0, fourier_bands=6, seed=1047,
        )
        m_v3, _ = train_spatial_shrinkage(train_trials, val_trials, cfg_v3, device=device, verbose=False)
        for K in K_list:
            r = evaluate_spatial_shrinkage(m_v3, sub_trials, K, device=device)
            per_k[f"v3_spatial_K{K}"] = r["l2_mean"]

        row = {"held": held, "n_test_trials": len(sub_trials), **per_k}
        results.append(row)
        sb = lambda k: per_k[k] if not np.isnan(per_k[k]) else float("nan")
        line = f"  {held:18s} | "
        for K in [1, 5, 12, 18]:
            line += f"K{K:2d}: anchor={sb(f'baseline_anchor_K{K}'):5.1f}  v3={sb(f'v3_spatial_K{K}'):5.1f}  "
        print(line)

    out_csv = out_dir / "results.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    with open(out_csv.with_suffix(".json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_csv}")

    # Aggregate
    df = pd.DataFrame(results)
    print("\n=== Subject-mean aggregate ===")
    print(f"{'K':>3s}  {'anchor':>8s}  {'raw':>8s}  {'fixed':>8s}  {'v1':>8s}  {'v3':>8s}")
    for K in K_list:
        cols = [f"baseline_anchor_K{K}", f"baseline_raw_K{K}", f"baseline_fixed_K{K}", f"v1_scalar_K{K}", f"v3_spatial_K{K}"]
        means = [df[c].mean() if c in df else float("nan") for c in cols]
        print(f"{K:>3d}  {means[0]:>8.2f}  {means[1]:>8.2f}  {means[2]:>8.2f}  {means[3]:>8.2f}  {means[4]:>8.2f}")


if __name__ == "__main__":
    main()
