"""Run v2 architectures on the all_trials_split (12-subject, per-trial
baselines, ~1500 train / 500 test). This dataset has higher noise and
real per-trial structure — much more room for neural refinement to help.

NOTE: subjects overlap across splits in this version. We document this
explicitly. The point is to test whether the method has any value when
the data has real signal to learn from.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from refine_v2 import V2Config, train_v2

DATA = ROOT.parent.parent / "data" / "prepared" / "all_trials_split"

BASE = dict(
    train_csv=str(DATA / "train.csv"),
    val_csv=str(DATA / "val.csv"),
    test_csv=str(DATA / "test.csv"),
    # Self-collected dataset uses 1920x1080 screen
    screen_w=1920.0,
    screen_h=1080.0,
    epochs=300,
    seed=1047,
    lr=1e-3,
    weight_decay=1e-3,
    batch_size=64,
    anchor_baseline="pred_sim_rbf_multiquadric_s2.0",
    baselines=("pred_sim_rbf_multiquadric_s2.0", "pred_similarity", "pred_poly", "pred_gpr", "pred_tps", "pred_sim_pwa"),
)

variants = [
    ("v2_mlp_residual_default",
        dict(arch="mlp_residual", hidden_dims=(64, 32), use_fourier=True, fourier_bands=4, noise_sigma_max=10.0)),
    ("v2_mlp_residual_no_noise",
        dict(arch="mlp_residual", hidden_dims=(64, 32), use_fourier=True, fourier_bands=4, noise_sigma_max=0.0)),
    ("v2_mlp_residual_strong_noise",
        dict(arch="mlp_residual", hidden_dims=(64, 32), use_fourier=True, fourier_bands=4, noise_sigma_max=25.0)),
    ("v2_mlp_residual_no_fourier",
        dict(arch="mlp_residual", hidden_dims=(64, 32), use_fourier=False, noise_sigma_max=10.0)),
    ("v2_mlp_residual_bigger",
        dict(arch="mlp_residual", hidden_dims=(128, 64), use_fourier=True, fourier_bands=4, noise_sigma_max=10.0, dropout=0.1)),
    ("v2_mlp_residual_with_resreg",
        dict(arch="mlp_residual", hidden_dims=(64, 32), use_fourier=True, fourier_bands=4, noise_sigma_max=10.0, residual_l2=1e-3)),
    ("v2_softmax_blend_default",
        dict(arch="softmax_blend", hidden_dims=(64, 32), use_fourier=True, fourier_bands=4, noise_sigma_max=10.0)),
    ("v2_softmax_blend_strong_noise",
        dict(arch="softmax_blend", hidden_dims=(64, 32), use_fourier=True, fourier_bands=4, noise_sigma_max=25.0)),
    ("v2_mlp_residual_anchor_gpr",
        dict(arch="mlp_residual", hidden_dims=(64, 32), use_fourier=True, fourier_bands=4, noise_sigma_max=10.0, anchor_baseline="pred_gpr")),
    ("v2_mlp_residual_anchor_poly",
        dict(arch="mlp_residual", hidden_dims=(64, 32), use_fourier=True, fourier_bands=4, noise_sigma_max=10.0, anchor_baseline="pred_poly")),
]

print("Reference: best classical = pred_sim_rbf_multiquadric_s2.0 (45.53 px on test)\n")

rows = []
for label, override in variants:
    print(f"=== {label} ===")
    cfg = V2Config(**{**BASE, **override})
    t0 = time.time()
    r = train_v2(cfg, verbose=False)
    elapsed = time.time() - t0
    test = r["test"]
    row = {
        "label": label,
        "arch": cfg.arch,
        "anchor": cfg.anchor_baseline,
        "noise_sigma_max": cfg.noise_sigma_max,
        "best_val_l2": r["best_val_l2"],
        "best_epoch": r["best_epoch"],
        "test_l2_mean": test["l2_mean"],
        "test_l2_median": test["l2_median"],
        "test_l2_std": test["l2_std"],
        "test_l2_p95": test["l2_p95"],
        "wall_s": elapsed,
    }
    rows.append(row)
    print(f"  val_L2={r['best_val_l2']:.2f}  test_L2_mean={test['l2_mean']:.2f}  "
          f"med={test['l2_median']:.2f}  p95={test['l2_p95']:.2f}  ({elapsed:.0f}s)")

out = Path(__file__).parent / "results.csv"
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
with open(out.with_suffix(".json"), "w") as f:
    json.dump(rows, f, indent=2)

print("\n=== Summary ===")
print(f"{'label':38s} {'val':>6s} {'test':>6s} {'med':>6s} {'p95':>6s}  vs baseline")
ref = 45.53
for row in sorted(rows, key=lambda x: x["test_l2_mean"]):
    delta = row["test_l2_mean"] - ref
    print(f"{row['label']:38s} {row['best_val_l2']:>6.2f} {row['test_l2_mean']:>6.2f} {row['test_l2_median']:>6.1f} {row['test_l2_p95']:>6.1f}  {delta:+.2f}")
