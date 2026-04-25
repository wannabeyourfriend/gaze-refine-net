"""Quick search over the v2 architecture knobs to find a config that
beats the strongest classical baseline (similarity, 21.66 px on JuDo)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from refine_v2 import V2Config, train_v2

JUDO = ROOT.parent.parent / "data" / "prepared" / "judo_1000_split_no_leakage"

BASE = dict(
    train_csv=str(JUDO / "train.csv"),
    val_csv=str(JUDO / "val.csv"),
    test_csv=str(JUDO / "test.csv"),
    # JuDo screen extent (tight around target region: x in [320, 1000], y in [240, 770])
    screen_w=1024.0,
    screen_h=768.0,
    epochs=300,
    seed=1047,
    lr=1e-3,
    weight_decay=1e-3,
    batch_size=64,
)

variants = [
    # --- mlp_residual head, varying inputs ---
    ("v2_mlp_residual_default",
        dict(arch="mlp_residual", hidden_dims=(64, 32), use_fourier=True, fourier_bands=4,
             noise_sigma_max=8.0, anchor_baseline="pred_similarity")),
    ("v2_mlp_residual_no_fourier",
        dict(arch="mlp_residual", hidden_dims=(64, 32), use_fourier=False,
             noise_sigma_max=8.0, anchor_baseline="pred_similarity")),
    ("v2_mlp_residual_smaller",
        dict(arch="mlp_residual", hidden_dims=(32, 16), use_fourier=True, fourier_bands=4,
             noise_sigma_max=8.0, anchor_baseline="pred_similarity")),
    ("v2_mlp_residual_no_noise",
        dict(arch="mlp_residual", hidden_dims=(64, 32), use_fourier=True, fourier_bands=4,
             noise_sigma_max=0.0, anchor_baseline="pred_similarity")),
    ("v2_mlp_residual_residual_l2",
        dict(arch="mlp_residual", hidden_dims=(64, 32), use_fourier=True, fourier_bands=4,
             noise_sigma_max=8.0, anchor_baseline="pred_similarity", residual_l2=1e-2)),

    # --- softmax_blend head ---
    ("v2_softmax_blend_default",
        dict(arch="softmax_blend", hidden_dims=(64, 32), use_fourier=True, fourier_bands=4,
             noise_sigma_max=8.0, anchor_baseline="pred_similarity")),
    ("v2_softmax_blend_no_fourier",
        dict(arch="softmax_blend", hidden_dims=(64, 32), use_fourier=False,
             noise_sigma_max=8.0, anchor_baseline="pred_similarity")),
    ("v2_softmax_blend_no_noise",
        dict(arch="softmax_blend", hidden_dims=(64, 32), use_fourier=True, fourier_bands=4,
             noise_sigma_max=0.0, anchor_baseline="pred_similarity")),

    # --- different anchor baselines ---
    ("v2_mlp_residual_anchor_poly",
        dict(arch="mlp_residual", hidden_dims=(64, 32), use_fourier=True, fourier_bands=4,
             noise_sigma_max=8.0, anchor_baseline="pred_poly")),

    # --- expanded baseline pool ---
    ("v2_mlp_residual_3baselines",
        dict(arch="mlp_residual", hidden_dims=(64, 32), use_fourier=True, fourier_bands=4,
             noise_sigma_max=8.0, anchor_baseline="pred_similarity",
             baselines=("pred_similarity", "pred_poly", "pred_sim_rbf_multiquadric_s2.0"))),
    ("v2_softmax_blend_3baselines",
        dict(arch="softmax_blend", hidden_dims=(64, 32), use_fourier=True, fourier_bands=4,
             noise_sigma_max=8.0, anchor_baseline="pred_similarity",
             baselines=("pred_similarity", "pred_poly", "pred_sim_rbf_multiquadric_s2.0"))),
]

rows = []
for label, override in variants:
    print(f"\n=== {label} ===")
    cfg = V2Config(**{**BASE, **override})
    t0 = time.time()
    r = train_v2(cfg, verbose=False)
    elapsed = time.time() - t0
    test = r["test"]
    row = {
        "label": label,
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
pd.DataFrame(rows).to_csv(out, index=False)
with open(out.with_suffix(".json"), "w") as f:
    json.dump(rows, f, indent=2)

print("\n\n=== Summary (sorted by test_l2_mean) ===")
print(f"{'label':40s} {'val':>7s} {'test':>7s} {'med':>6s} {'p95':>6s}")
for row in sorted(rows, key=lambda x: x["test_l2_mean"]):
    print(f"{row['label']:40s} {row['best_val_l2']:>7.2f} {row['test_l2_mean']:>7.2f} {row['test_l2_median']:>6.1f} {row['test_l2_p95']:>6.1f}")
