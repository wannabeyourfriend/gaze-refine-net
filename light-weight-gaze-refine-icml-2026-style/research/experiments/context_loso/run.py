"""Context-conditioned refinement, LOSO across 12 subjects.

Compares the trained context-aware network against the parameter-free
mean-bias baseline at multiple context sizes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from context_pipeline import CtxCfg, train_context

DATA = ROOT / "data" / "all_trials_combined.csv"

baselines = (
    "pred_sim_rbf_multiquadric_s2.0",
    "pred_similarity",
    "pred_poly",
    "pred_gpr",
    "pred_tps",
    "pred_sim_pwa",
)
anchor = "pred_sim_rbf_multiquadric_s2.0"

df_all = pd.read_csv(DATA)
df_all["trial_id"] = df_all["subject"] + "|" + df_all["timestamp"]
print(f"Combined: {len(df_all)} rows, {df_all['subject'].nunique()} subjects, {df_all['trial_id'].nunique()} trials")


def mean_bias_baseline(test_df: pd.DataFrame, k: int, anchor_col: str) -> dict:
    """Replicate the parameter-free mean-bias correction baseline."""
    test_df = test_df.copy()
    test_df["trial_id"] = test_df["subject"] + "|" + test_df["timestamp"]
    l2s = []
    for trial_id, g in test_df.groupby("trial_id"):
        if len(g) <= k:
            continue
        rng = np.random.default_rng(hash(trial_id) & 0xFFFFFFFF)
        idx = rng.permutation(len(g))
        ctx_idx, q_idx = idx[:k], idx[k:]
        ctx, q = g.iloc[ctx_idx], g.iloc[q_idx]
        if k > 0:
            bias = (ctx[["target_x", "target_y"]].values - ctx[[f"{anchor_col}_x", f"{anchor_col}_y"]].values).mean(axis=0)
        else:
            bias = np.zeros(2)
        pred = q[[f"{anchor_col}_x", f"{anchor_col}_y"]].values + bias
        l2 = np.linalg.norm(pred - q[["target_x", "target_y"]].values, axis=1)
        l2s.append(l2)
    if not l2s:
        return {"k": k, "l2_mean": float("nan")}
    l2 = np.concatenate(l2s)
    return {
        "k": k,
        "l2_mean": float(l2.mean()),
        "l2_median": float(np.median(l2)),
        "l2_std": float(l2.std()),
        "l2_p95": float(np.percentile(l2, 95)),
        "n": int(len(l2)),
    }


cfg_template = CtxCfg(
    baselines=baselines,
    anchor=anchor,
    screen_w=1920.0,
    screen_h=1080.0,
    use_fourier=True,
    fourier_bands=4,
    context_size=12,
    encoder_hidden=64,
    encoder_layers=2,
    summary_dim=32,
    predictor_hidden=(64, 32),
    epochs=80,
    batch_size_trials=8,
    queries_per_trial=4,
    lr=1e-3,
    weight_decay=1e-3,
    seed=1047,
)

# A subset of LOSO subjects for now to keep runtime manageable.
# Use the subjects with at least 5 trials so context evaluation is meaningful.
trial_counts = df_all.groupby("subject")["trial_id"].nunique().sort_values(ascending=False)
print("\nTrials per subject:")
print(trial_counts.to_string())

candidate_subjects = trial_counts[trial_counts >= 1].index.tolist()
print(f"\nLOSO over {len(candidate_subjects)} subjects.")

results = []
for held in candidate_subjects:
    sub_df = df_all[df_all["subject"] == held]
    rest_df = df_all[df_all["subject"] != held].copy()
    if len(sub_df) < 5 or len(rest_df) < 100:
        continue
    rest_df["trial_id"] = rest_df["subject"] + "|" + rest_df["timestamp"]
    trials = sorted(rest_df["trial_id"].unique())
    np.random.seed(cfg_template.seed)
    np.random.shuffle(trials)
    n_val = max(1, int(len(trials) * 0.10))
    val_trials = set(trials[:n_val])
    val_df = rest_df[rest_df["trial_id"].isin(val_trials)].drop(columns=["trial_id"])
    train_df = rest_df[~rest_df["trial_id"].isin(val_trials)].drop(columns=["trial_id"])

    print(f"\n=== held subject = {held} (n_test={len(sub_df)}, n_train_trials={len(trials)-n_val}) ===")
    out = train_context(train_df, val_df, sub_df, cfg_template, verbose=False)

    # Mean-bias baseline at the same K values
    base_per_k = {k: mean_bias_baseline(sub_df, k, anchor) for k in [3, 5, 8, 12, 18]}
    raw_l2 = mean_bias_baseline(sub_df, 0, anchor)["l2_mean"]
    print(f"  raw_anchor (k=0) L2 = {raw_l2:.2f}")
    for k in [3, 5, 8, 12, 18]:
        ours = out[f"test_K{k}"]["l2_mean"]
        base = base_per_k[k]["l2_mean"]
        print(f"  K={k:2d}  mean_bias={base:6.2f}  ours={ours:6.2f}  delta_vs_baseline={ours - base:+6.2f}")

    results.append({
        "held": held,
        "n_test": len(sub_df),
        "raw_anchor_l2": raw_l2,
        **{f"baseline_K{k}_{kk}": v for k, d in base_per_k.items() for kk, v in d.items() if kk != "k"},
        **{f"ours_K{k}_{kk}": v for k in [3,5,8,12,18] for kk, v in out[f"test_K{k}"].items()},
        "best_val_l2": out["best_val_l2"],
        "best_epoch": out["best_epoch"],
    })

out_csv = Path(__file__).parent / "results.csv"
pd.DataFrame(results).to_csv(out_csv, index=False)
with open(out_csv.with_suffix(".json"), "w") as f:
    json.dump(results, f, indent=2)

# Aggregate
df_r = pd.DataFrame(results)
print("\n\n=== Aggregated (mean over LOSO folds) ===")
print(f"{'K':>4s}  {'baseline_L2':>13s}  {'ours_L2':>10s}  {'delta':>7s}")
for k in [3, 5, 8, 12, 18]:
    base = df_r[f"baseline_K{k}_l2_mean"].mean()
    ours = df_r[f"ours_K{k}_l2_mean"].mean()
    print(f"{k:4d}  {base:13.2f}  {ours:10.2f}  {ours-base:+7.2f}")
