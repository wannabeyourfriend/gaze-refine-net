"""Run the paper-faithful JuDo headline + diagnostic comparisons.

Selection ranking is computed on the held-out validation targets, which is
the proper interpretation of "calibration risk on calibration points" in a
target-disjoint split — a baseline that interpolates training targets
exactly is not informative.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from honest_pipeline import (
    DEFAULT_BASELINES,
    PipelineConfig,
    baseline_only_metrics,
    result_to_dict,
    save_results,
    train_one,
)

JUDO = ROOT.parent.parent / "data" / "prepared" / "judo_1000_split_no_leakage"

print(f"Reading from {JUDO}")
test_df = pd.read_csv(JUDO / "test.csv")
print(f"  Test rows: {len(test_df)}, unique targets: {test_df[['target_x','target_y']].drop_duplicates().shape[0]}")

# Baseline-only reference table
print("\n=== Baseline-only test metrics ===")
b_metrics = baseline_only_metrics(test_df, list(DEFAULT_BASELINES))
for k, v in b_metrics.items():
    print(f"  {k:38s}  L2={v['l2_mean']:6.2f}  med={v['l2_median']:6.2f}  std={v['l2_std']:6.2f}  p95={v['l2_p95']:6.2f}")

with open(Path(__file__).parent / "baseline_metrics.json", "w") as f:
    json.dump(b_metrics, f, indent=2)

base_kw = dict(
    train_csv=str(JUDO / "train.csv"),
    val_csv=str(JUDO / "val.csv"),
    test_csv=str(JUDO / "test.csv"),
    baselines=DEFAULT_BASELINES,
    hidden_dims=(64, 32, 16),
    use_batchnorm=True,
    epochs=200,
    seed=1047,
    lr=3e-4,
    weight_decay=0.01,
    batch_size=64,
    selection_source="val",
)

variants = [
    # Honest, paper-faithful (top-M=4, noise sigma=20 px)
    ("honest_topM4_noise20",  dict(selection="topM", top_m=4, noise_sigma_max=20.0, leaky_features=False)),
    # Honest no noise
    ("honest_topM4_noise0",   dict(selection="topM", top_m=4, noise_sigma_max=0.0,  leaky_features=False)),
    # Honest all baselines, no noise
    ("honest_all_noise0",     dict(selection="all",            noise_sigma_max=0.0, leaky_features=False)),
    # Honest all baselines, with noise
    ("honest_all_noise20",    dict(selection="all",            noise_sigma_max=20.0,leaky_features=False)),
    # Oracle selection (rank baselines on test set — upper bound)
    ("oracle_topM4_noise0",   dict(selection="topM_oracle", top_m=4, noise_sigma_max=0.0, leaky_features=False)),
    # Diagnostic LEAKY runs to confirm the mechanism
    ("LEAKY_topM4_noise20",   dict(selection="topM", top_m=4, noise_sigma_max=20.0, leaky_features=True)),
    ("LEAKY_all_noise0",      dict(selection="all",            noise_sigma_max=0.0, leaky_features=True)),
]

rows = []
for label, override in variants:
    cfg_dict = {**base_kw, **override}
    cfg = PipelineConfig(**cfg_dict)
    print(f"\n=== {label} ===")
    r = train_one(cfg, verbose=False)
    print(f"  selected: {r.selected_baselines}")
    print(f"  best_val_l2: {r.best_val_l2:.3f} px @ epoch {r.best_epoch}")
    print(f"  test L2 mean={r.test_metrics['l2_mean']:.3f}  median={r.test_metrics['l2_median']:.3f}  "
          f"std={r.test_metrics['l2_std']:.3f}  p95={r.test_metrics['l2_p95']:.3f}")
    d = result_to_dict(r)
    d["label"] = label
    rows.append(d)

out_csv = Path(__file__).parent / "results.csv"
save_results(rows, out_csv)

print("\n\n=== Summary table ===")
print(f"{'label':28s} {'sel':>14s} {'noise':>6s} {'val_L2':>8s} {'test_L2':>8s} {'med':>6s} {'std':>6s} {'p95':>6s}")
print("-" * 100)
for row in rows:
    sel = f"{row['selection']}-{row['top_m']}"
    print(f"{row['label']:28s} {sel:>14s} {row['noise_sigma_max']:>6.0f} {row['best_val_l2']:>8.2f} {row['test_l2_mean']:>8.2f} {row['test_l2_median']:>6.1f} {row['test_l2_std']:>6.1f} {row['test_l2_p95']:>6.1f}")
print(f"\nResults saved to {out_csv}")
