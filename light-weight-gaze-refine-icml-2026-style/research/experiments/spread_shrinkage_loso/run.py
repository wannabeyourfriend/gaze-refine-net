"""LOSO evaluation of spread-aware shrinkage on JuDo1000 (full).

For each held-out subject (sampled set of 30 of 150):
  - Train pool: all other subjects' trials.
  - Validate on a 10% trial-disjoint subset of the pool.
  - Train and evaluate v1 (scalar), v3 (spatial), v4a (spread-feat-only),
    v4b (inv-var-mean only), v4c (full spread-aware).
  - Eval at K in {1, 2, 3, 5, 8, 12, 18} on the held subject's trials.

Anchor: pred_tps (strongest classical on JuDo at ~58 px).
"""
from __future__ import annotations

import json
import sys
import time
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
    _trial_arrays,
)
from spread_shrinkage import (
    SpreadShrinkageConfig,
    train_spread_shrinkage,
    evaluate_spread_shrinkage,
    _trial_arrays_with_spread,
)


# Default to remote path; allow override via CLI.
DEFAULT_DATA = "/home/2025user/zhou/klab-workspace/gaze-refine-net/data/prepared/judo1000_full/all_with_baselines.csv"

ANCHOR = "pred_tps"
K_LIST = [1, 2, 3, 5, 8, 12, 18]
N_HELD_SUBJECTS = 30
SEED = 1047


def fixed_lambda_eval(test_trials, K: int, lam: float) -> float:
    errs = []
    for tr in test_trials:
        T = len(tr["anchor"])
        if T <= K:
            continue
        rng = np.random.default_rng(hash(tr["trial_id"]) & 0xFFFFFFFF)
        idx = rng.permutation(T)
        ctx_idx, q_idx = idx[:K], idx[K:]
        bar_b = (tr["target"][ctx_idx] - tr["anchor"][ctx_idx]).mean(axis=0)
        pred = tr["anchor"][q_idx] + lam * bar_b
        e = pred - tr["target"][q_idx]
        errs.append(np.linalg.norm(e, axis=1))
    return float(np.concatenate(errs).mean()) if errs else float("nan")


def main():
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
    print(f"Loading data: {data_path}")
    t0 = time.time()
    df = pd.read_csv(data_path)
    print(f"  loaded {len(df)} rows in {time.time() - t0:.1f}s")

    # subjects
    subjects = sorted(df["subject"].unique().tolist())
    print(f"Total subjects: {len(subjects)}")
    rng_pick = np.random.default_rng(SEED)
    held_subjects = sorted(rng_pick.choice(subjects, size=min(N_HELD_SUBJECTS, len(subjects)), replace=False).tolist())
    print(f"Held subjects ({len(held_subjects)}): {held_subjects[:10]}{'...' if len(held_subjects) > 10 else ''}")

    # Build per-trial arrays once for both engines (they share the trial format,
    # plus the spread engine adds a 'spread' field).
    print("Grouping into trials...")
    t0 = time.time()
    all_trials_v1v3 = _trial_arrays(df, ANCHOR)  # for v1 & v3
    all_trials_v4 = _trial_arrays_with_spread(df, ANCHOR)  # for v4*
    print(f"  v1/v3 trials: {len(all_trials_v1v3)}, v4 trials: {len(all_trials_v4)} in {time.time() - t0:.1f}s")

    # Sanity check: consistent ordering
    assert len(all_trials_v1v3) == len(all_trials_v4)
    for a, b in zip(all_trials_v1v3, all_trials_v4):
        assert a["trial_id"] == b["trial_id"], (a["trial_id"], b["trial_id"])

    rows: List[Dict] = []

    for fold_idx, held in enumerate(held_subjects):
        t_fold = time.time()
        sub_trials_v1v3 = [t for t in all_trials_v1v3 if t["subject"] == held]
        sub_trials_v4 = [t for t in all_trials_v4 if t["subject"] == held]
        rest_v1v3 = [t for t in all_trials_v1v3 if t["subject"] != held]
        rest_v4 = [t for t in all_trials_v4 if t["subject"] != held]

        rng = np.random.default_rng(SEED)
        idx = rng.permutation(len(rest_v1v3))
        n_val = max(1, len(rest_v1v3) // 10)
        val_idx = set(idx[:n_val].tolist())
        train_v1v3 = [rest_v1v3[i] for i in range(len(rest_v1v3)) if i not in val_idx]
        val_v1v3 = [rest_v1v3[i] for i in val_idx]
        train_v4 = [rest_v4[i] for i in range(len(rest_v4)) if i not in val_idx]
        val_v4 = [rest_v4[i] for i in val_idx]

        per_k: Dict[str, float] = {}

        # ---- baselines (no training) ----
        for K in K_LIST:
            per_k[f"baseline_anchor_K{K}"] = fixed_lambda_eval(sub_trials_v1v3, K, 0.0)
            per_k[f"baseline_raw_K{K}"] = fixed_lambda_eval(sub_trials_v1v3, K, 1.0)

        # ---- v1: scalar shrinkage ----
        cfg_v1 = SpatialShrinkageConfig(
            spatial=False, epochs=80, K_train_min=1, K_train_max=18,
            ctx_hidden=64, ctx_summary_dim=32, head_hidden=(64, 32),
            screen_w=1024.0, screen_h=768.0, seed=SEED,
        )
        m_v1, _ = train_spatial_shrinkage(train_v1v3, val_v1v3, cfg_v1, device=device, verbose=False)
        for K in K_LIST:
            r = evaluate_spatial_shrinkage(m_v1, sub_trials_v1v3, K, device=device)
            per_k[f"v1_scalar_K{K}"] = r["l2_mean"]

        # ---- v3: spatial shrinkage ----
        cfg_v3 = SpatialShrinkageConfig(
            spatial=True, epochs=120, K_train_min=1, K_train_max=18,
            ctx_hidden=128, ctx_summary_dim=64, head_hidden=(128, 64),
            screen_w=1024.0, screen_h=768.0, fourier_bands=6, seed=SEED,
        )
        m_v3, _ = train_spatial_shrinkage(train_v1v3, val_v1v3, cfg_v3, device=device, verbose=False)
        for K in K_LIST:
            r = evaluate_spatial_shrinkage(m_v3, sub_trials_v1v3, K, device=device)
            per_k[f"v3_spatial_K{K}"] = r["l2_mean"]

        # ---- v4a: spread-feat only (no inv-var mean) ----
        cfg_v4a = SpreadShrinkageConfig(
            spatial=True, epochs=120, K_train_min=1, K_train_max=18,
            ctx_hidden=128, ctx_summary_dim=64, head_hidden=(128, 64),
            screen_w=1024.0, screen_h=768.0, fourier_bands=6, seed=SEED,
            use_spread_item_feat=True, use_spread_global_feat=True,
            inverse_variance_mean=False,
        )
        m_v4a, _ = train_spread_shrinkage(train_v4, val_v4, cfg_v4a, device=device, verbose=False)
        for K in K_LIST:
            r = evaluate_spread_shrinkage(m_v4a, sub_trials_v4, K, device=device)
            per_k[f"v4a_spreadfeat_K{K}"] = r["l2_mean"]

        # ---- v4b: inv-variance mean only (no spread features) ----
        cfg_v4b = SpreadShrinkageConfig(
            spatial=True, epochs=120, K_train_min=1, K_train_max=18,
            ctx_hidden=128, ctx_summary_dim=64, head_hidden=(128, 64),
            screen_w=1024.0, screen_h=768.0, fourier_bands=6, seed=SEED,
            use_spread_item_feat=False, use_spread_global_feat=False,
            inverse_variance_mean=True,
        )
        m_v4b, _ = train_spread_shrinkage(train_v4, val_v4, cfg_v4b, device=device, verbose=False)
        for K in K_LIST:
            r = evaluate_spread_shrinkage(m_v4b, sub_trials_v4, K, device=device)
            per_k[f"v4b_invvar_K{K}"] = r["l2_mean"]

        # ---- v4c: full spread-aware ----
        cfg_v4c = SpreadShrinkageConfig(
            spatial=True, epochs=120, K_train_min=1, K_train_max=18,
            ctx_hidden=128, ctx_summary_dim=64, head_hidden=(128, 64),
            screen_w=1024.0, screen_h=768.0, fourier_bands=6, seed=SEED,
            use_spread_item_feat=True, use_spread_global_feat=True,
            inverse_variance_mean=True,
        )
        m_v4c, _ = train_spread_shrinkage(train_v4, val_v4, cfg_v4c, device=device, verbose=False)
        for K in K_LIST:
            r = evaluate_spread_shrinkage(m_v4c, sub_trials_v4, K, device=device)
            per_k[f"v4c_full_K{K}"] = r["l2_mean"]

        row = {"held": held, "n_test_trials": len(sub_trials_v1v3), **per_k}
        rows.append(row)

        # Print summary line
        sb = lambda k: per_k.get(k, float("nan"))
        line = f"[{fold_idx + 1:2d}/{len(held_subjects)}] {held:10s} ({time.time() - t_fold:5.1f}s) | "
        for K in [1, 5, 12, 18]:
            line += (
                f"K{K:2d}: anc={sb(f'baseline_anchor_K{K}'):4.1f} v1={sb(f'v1_scalar_K{K}'):4.1f} "
                f"v3={sb(f'v3_spatial_K{K}'):4.1f} v4c={sb(f'v4c_full_K{K}'):4.1f}  "
            )
        print(line, flush=True)

        # Persist intermediate results after every fold (in case of crash)
        out_csv = out_dir / "results.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        with open(out_dir / "results.json", "w") as f:
            json.dump(rows, f, indent=2)

    # Aggregate
    print("\n=== Subject-mean aggregate ===")
    df_res = pd.DataFrame(rows)
    cols_per_method = {
        "anchor": "baseline_anchor_K{K}",
        "raw":    "baseline_raw_K{K}",
        "v1":     "v1_scalar_K{K}",
        "v3":     "v3_spatial_K{K}",
        "v4a":    "v4a_spreadfeat_K{K}",
        "v4b":    "v4b_invvar_K{K}",
        "v4c":    "v4c_full_K{K}",
    }
    headers = list(cols_per_method.keys())
    print(f"{'K':>3s}  " + "  ".join(f"{h:>7s}" for h in headers))
    summary: List[Dict] = []
    for K in K_LIST:
        means = []
        for h in headers:
            col = cols_per_method[h].format(K=K)
            means.append(df_res[col].mean() if col in df_res else float("nan"))
        print(f"{K:>3d}  " + "  ".join(f"{m:>7.2f}" for m in means))
        summary.append({"K": K, **{h: m for h, m in zip(headers, means)}})
    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)

    # Final delta table vs anchor
    print("\n=== Delta vs anchor (positive = improvement) ===")
    print(f"{'K':>3s}  " + "  ".join(f"{h:>7s}" for h in headers if h != "anchor"))
    for K in K_LIST:
        anc = df_res[f"baseline_anchor_K{K}"].mean()
        deltas = []
        for h in headers:
            if h == "anchor":
                continue
            col = cols_per_method[h].format(K=K)
            v = df_res[col].mean() if col in df_res else float("nan")
            deltas.append(anc - v)
        print(f"{K:>3d}  " + "  ".join(f"{d:>+7.2f}" for d in deltas))

    print("\nDone.")


if __name__ == "__main__":
    main()
