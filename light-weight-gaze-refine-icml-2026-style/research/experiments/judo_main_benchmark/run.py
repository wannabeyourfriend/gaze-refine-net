"""Main JuDo1000 benchmark.

Compares all methods on the canonical 33/7/8 target-disjoint split:

  - origin_gaze (raw)
  - 6 classical baselines (similarity, poly, RBF*, TPS, sim+RBF, sim+PWA, sim+TPS)
  - Multi-Baseline Residual learning (MBR): an MLP that maps
    [orig, K classical baseline preds] -> residual on top of anchor
  - Anchored shrinkage (ours v1): scalar shrinkage on per-trial bias
  - Anchored spatial shrinkage (ours v3): position-dependent shrinkage

Trial structure: since the JuDo splits don't carry per-participant ids,
we use **per-target-point grouping** as the "trial" unit (each unique
(target_x, target_y) corresponds to one set of fixation samples
contributed by many participants — the per-target bias captures
systematic position-dependent drift in the same way as per-trial bias).

This is the right unit anyway for the spatial-shrinkage method, since
it's the per-target context that gives us the empirical bias used by
the shrinkage estimator.
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

from baselines import MBRConfig, train_mbr, evaluate_mbr, per_trial_fixed_lambda_eval
from spatial_shrinkage import (
    SpatialShrinkageConfig,
    train_spatial_shrinkage,
    evaluate_spatial_shrinkage,
)

DATA = ROOT.parent.parent / "data" / "prepared" / "judo_1000_split_no_leakage"
ANCHOR = "pred_similarity"  # similarity is the strongest classical on JuDo

BASELINE_COLS = [
    "pred_similarity",
    "pred_poly",
    "pred_rbf_multiquadric_s2.0",
    "pred_tps",
    "pred_sim_rbf_multiquadric_s2.0",
    "pred_sim_tps",
    "pred_sim_pwa",
]


def add_trial_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trial_id"] = df["target_x"].round(0).astype(int).astype(str) + "_" + df["target_y"].round(0).astype(int).astype(str)
    df["subject"] = "judo"  # required by the pipeline; treated as one big subject
    df["timestamp"] = df["trial_id"]
    return df


def baseline_test_metrics(df: pd.DataFrame, col: str) -> Dict[str, float]:
    pred = df[[f"{col}_x", f"{col}_y"]].to_numpy(dtype=np.float32)
    target = df[["target_x", "target_y"]].to_numpy(dtype=np.float32)
    err = pred - target
    l2 = np.linalg.norm(err, axis=1)
    return {
        "l2_mean": float(l2.mean()), "l2_median": float(np.median(l2)),
        "l2_std": float(l2.std()), "l2_p95": float(np.percentile(l2, 95)),
        "mae_x": float(np.mean(np.abs(err[:,0]))), "mae_y": float(np.mean(np.abs(err[:,1]))),
        "rmse_x": float(np.sqrt((err[:,0]**2).mean())), "rmse_y": float(np.sqrt((err[:,1]**2).mean())),
        "n": int(len(l2)),
    }


def origin_test_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if "origin_gaze_x" in df.columns:
        ox, oy = "origin_gaze_x", "origin_gaze_y"
    else:
        ox, oy = "original_gaze_x", "original_gaze_y"
    pred = df[[ox, oy]].to_numpy(dtype=np.float32)
    target = df[["target_x", "target_y"]].to_numpy(dtype=np.float32)
    err = pred - target
    l2 = np.linalg.norm(err, axis=1)
    return {
        "l2_mean": float(l2.mean()), "l2_median": float(np.median(l2)),
        "l2_std": float(l2.std()), "l2_p95": float(np.percentile(l2, 95)),
        "mae_x": float(np.mean(np.abs(err[:,0]))), "mae_y": float(np.mean(np.abs(err[:,1]))),
        "rmse_x": float(np.sqrt((err[:,0]**2).mean())), "rmse_y": float(np.sqrt((err[:,1]**2).mean())),
        "n": int(len(l2)),
    }


def main():
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_df = add_trial_id(pd.read_csv(DATA / "train.csv"))
    val_df = add_trial_id(pd.read_csv(DATA / "val.csv"))
    test_df = add_trial_id(pd.read_csv(DATA / "test.csv"))
    print(f"Train: {len(train_df)} samples, val: {len(val_df)}, test: {len(test_df)}")
    print(f"Test target points: {test_df['trial_id'].nunique()}")

    rows: List[Dict] = []

    # 1. Origin
    m = origin_test_metrics(test_df)
    rows.append({"method": "origin_gaze", **m})
    print(f"  origin_gaze:                {m['l2_mean']:.2f}")

    # 2. Classical baselines
    for col in BASELINE_COLS:
        m = baseline_test_metrics(test_df, col)
        rows.append({"method": f"classical_{col}", **m})
        print(f"  {col:38s} {m['l2_mean']:.2f}")

    # 3. Multi-Baseline Residual learning (MBR)
    print("\nTraining Multi-Baseline Residual (MBR)...")
    mbr_cfg = MBRConfig(
        baseline_cols=BASELINE_COLS,
        hidden_dims=(128, 64),
        dropout=0.1,
        epochs=200,
        lr=1e-3,
        weight_decay=1e-3,
        batch_size=64,
        seed=1047,
    )
    mbr_model, mbr_info = train_mbr(train_df, val_df, mbr_cfg, ANCHOR, device=device, verbose=True)
    m = evaluate_mbr(mbr_model, test_df, mbr_cfg, ANCHOR, mbr_info["bs_mean"], mbr_info["bs_std"], device=device)
    rows.append({"method": "MBR_anchored", **m})
    print(f"  MBR (anchored):             {m['l2_mean']:.2f}")

    # 4. Per-trial fixed-lambda baselines at multiple K
    for K in [1, 2, 3, 5, 8, 12, 18]:
        for lam in [0.0, 0.5, 1.0]:
            mm = per_trial_fixed_lambda_eval(test_df, ANCHOR, K, lam)
            rows.append({"method": f"fixed_lambda{lam}_K{K}", **mm})
        # Best lambda search
        best = (0.0, float("inf"))
        for lam in np.linspace(0, 1, 21):
            mm = per_trial_fixed_lambda_eval(test_df, ANCHOR, K, float(lam))
            if not np.isnan(mm["l2_mean"]) and mm["l2_mean"] < best[1]:
                best = (float(lam), mm["l2_mean"])
        rows.append({"method": f"best_fixed_K{K}", "best_lambda": best[0], "l2_mean": best[1]})
        print(f"  K={K:2d}: best_fixed lambda={best[0]:.2f} L2={best[1]:.2f}")

    # 5. Anchored shrinkage v1 (scalar) and v3 (spatial)
    print("\nTraining v1 scalar shrinkage on JuDo...")
    train_trials = []
    val_trials = []
    test_trials = []
    from spatial_shrinkage import _trial_arrays
    train_trials = _trial_arrays(train_df, ANCHOR)
    val_trials = _trial_arrays(val_df, ANCHOR)
    test_trials = _trial_arrays(test_df, ANCHOR)
    print(f"  train trials={len(train_trials)} val={len(val_trials)} test={len(test_trials)}")

    # v1
    cfg_v1 = SpatialShrinkageConfig(
        spatial=False, epochs=80, K_train_min=1, K_train_max=18,
        ctx_hidden=64, ctx_summary_dim=32, head_hidden=(64, 32),
        screen_w=1024.0, screen_h=768.0, seed=1047,
    )
    m_v1, _ = train_spatial_shrinkage(train_trials, val_trials, cfg_v1, device=device, verbose=False)
    for K in [1, 2, 3, 5, 8, 12, 18]:
        rr = evaluate_spatial_shrinkage(m_v1, test_trials, K, device=device)
        rows.append({"method": f"v1_scalar_K{K}", **rr})
        print(f"  v1 scalar K={K:2d}: {rr.get('l2_mean', float('nan')):.2f}")

    # v3
    print("\nTraining v3 spatial shrinkage on JuDo...")
    cfg_v3 = SpatialShrinkageConfig(
        spatial=True, epochs=120, K_train_min=1, K_train_max=18,
        ctx_hidden=128, ctx_summary_dim=64, head_hidden=(128, 64),
        screen_w=1024.0, screen_h=768.0, fourier_bands=6, seed=1047,
    )
    m_v3, _ = train_spatial_shrinkage(train_trials, val_trials, cfg_v3, device=device, verbose=False)
    for K in [1, 2, 3, 5, 8, 12, 18]:
        rr = evaluate_spatial_shrinkage(m_v3, test_trials, K, device=device)
        rows.append({"method": f"v3_spatial_K{K}", **rr})
        print(f"  v3 spatial K={K:2d}: {rr.get('l2_mean', float('nan')):.2f}")

    # Save
    pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)
    with open(out_dir / "results.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)

    print("\nDone.")


if __name__ == "__main__":
    main()
