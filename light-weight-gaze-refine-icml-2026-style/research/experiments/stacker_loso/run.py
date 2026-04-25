"""Stacker LOSO evaluation. Anchor on best classical baseline,
predict residual from (orig, all baselines) features."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stacking_pipeline import StackCfg, train_stacker

DATA = ROOT / "data" / "all_trials_combined.csv"
df_all = pd.read_csv(DATA)
print(f"Combined: {len(df_all)} rows, {df_all['subject'].nunique()} subjects")

baselines = (
    "pred_sim_rbf_multiquadric_s2.0",
    "pred_similarity",
    "pred_poly",
    "pred_gpr",
    "pred_tps",
    "pred_sim_pwa",
)
anchor = "pred_sim_rbf_multiquadric_s2.0"

cfg = StackCfg(
    baselines=baselines,
    anchor=anchor,
    screen_w=1920.0,
    screen_h=1080.0,
    use_fourier=True,
    fourier_bands=4,
    hidden_dims=(64, 32),
    noise_sigma_max=10.0,
    noise_prob=0.5,
    epochs=200,
    seed=1047,
)

results = []
subjects = sorted(df_all["subject"].unique())
for held in subjects:
    sub = df_all[df_all["subject"] == held]
    rest = df_all[df_all["subject"] != held].copy()
    if len(sub) < 5:
        continue
    rest["trial_id"] = rest["subject"] + "|" + rest["timestamp"]
    trials = sorted(rest["trial_id"].unique())
    np.random.seed(cfg.seed)
    np.random.shuffle(trials)
    n_val = max(1, int(len(trials) * 0.10))
    val_trials = set(trials[:n_val])
    val_df = rest[rest["trial_id"].isin(val_trials)].drop(columns=["trial_id"])
    train_df = rest[~rest["trial_id"].isin(val_trials)].drop(columns=["trial_id"])

    err = np.linalg.norm(sub[[f"{anchor}_x", f"{anchor}_y"]].values - sub[["target_x", "target_y"]].values, axis=1)
    base_l2 = float(err.mean())

    out = train_stacker(train_df, val_df, sub, cfg, verbose=False)
    test = out["test"]
    delta = test["l2_mean"] - base_l2
    print(f"  {held:25s}  n_test={len(sub):4d}  base={base_l2:6.2f}  ours={test['l2_mean']:6.2f}  "
          f"delta={delta:+6.2f}  med={test['l2_median']:.1f}  p95={test['l2_p95']:.1f}")
    results.append({
        "held": held, "n_test": len(sub),
        "base_l2": base_l2,
        "best_val_l2": out["best_val_l2"], "best_epoch": out["best_epoch"],
        **{f"test_{k}": v for k, v in test.items()},
        "delta": delta,
    })

out_csv = Path(__file__).parent / "results.csv"
pd.DataFrame(results).to_csv(out_csv, index=False)
with open(out_csv.with_suffix(".json"), "w") as f:
    json.dump(results, f, indent=2)

df_r = pd.DataFrame(results)
print("\n=== Aggregate (mean over LOSO folds) ===")
print(f"  Baseline L2: {df_r['base_l2'].mean():.2f}")
print(f"  Stacker L2:  {df_r['test_l2_mean'].mean():.2f}")
print(f"  Mean delta:  {df_r['delta'].mean():+.2f}")
print(f"  Median delta:{df_r['delta'].median():+.2f}")
print(f"  Subjects improved: {(df_r['delta']<0).sum()}/{len(df_r)}")
