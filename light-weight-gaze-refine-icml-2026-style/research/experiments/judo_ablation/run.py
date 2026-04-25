"""JuDo systematic ablation (honest pipeline only).

Each subcategory varies one axis while holding others at paper-faithful
defaults. All runs use the same data, seed, and budget.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from honest_pipeline import (
    DEFAULT_BASELINES,
    PipelineConfig,
    baseline_only_metrics,
    result_to_dict,
    save_results,
    train_one,
)

JUDO = ROOT.parent.parent / "data" / "prepared" / "judo_1000_split_no_leakage"

BASE_KW = dict(
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
    selection="topM",
    top_m=4,
    selection_source="val",
    noise_sigma_max=20.0,
    leaky_features=False,
)


def run_one(label: str, axis: str, override: Dict[str, Any]) -> Dict[str, Any]:
    cfg_dict = {**BASE_KW, **override}
    cfg = PipelineConfig(**cfg_dict)
    t0 = time.time()
    r = train_one(cfg, verbose=False)
    elapsed = time.time() - t0
    d = result_to_dict(r)
    d["label"] = label
    d["axis"] = axis
    d["wall_min"] = elapsed / 60.0
    print(f"  {label:38s}  test_L2={d['test_l2_mean']:6.2f}  med={d['test_l2_median']:5.1f}  "
          f"p95={d['test_l2_p95']:5.1f}  ({elapsed:.0f}s)")
    return d


variants: List[Dict[str, Any]] = []

# ---------------- Axis 1: selection strategy ----------------
print("=== Axis 1: selection strategy ===")
selection_axis = [
    ("sel_all",            dict(selection="all")),
    ("sel_topM_1",         dict(selection="topM", top_m=1)),
    ("sel_topM_2",         dict(selection="topM", top_m=2)),
    ("sel_topM_3",         dict(selection="topM", top_m=3)),
    ("sel_topM_4",         dict(selection="topM", top_m=4)),
    ("sel_topM_6",         dict(selection="topM", top_m=6)),
    ("sel_topM_8",         dict(selection="topM", top_m=8)),
    ("sel_oracle_topM_4",  dict(selection="topM_oracle", top_m=4)),
    ("sel_oracle_topM_2",  dict(selection="topM_oracle", top_m=2)),
]
for label, ovr in selection_axis:
    variants.append(("selection", label, ovr))

# ---------------- Axis 2: noise sigma_max ----------------
print("\n=== Axis 2: noise injection ===")
for sigma in [0.0, 5.0, 10.0, 20.0, 40.0, 80.0]:
    variants.append(("noise", f"noise_sigma_{int(sigma):02d}", dict(noise_sigma_max=float(sigma))))

# ---------------- Axis 3: network capacity ----------------
print("\n=== Axis 3: network capacity ===")
capacities = [
    ("cap_16",         (16,)),
    ("cap_64",         (64,)),
    ("cap_64_32",      (64, 32)),
    ("cap_64_32_16",   (64, 32, 16)),
    ("cap_256_128",    (256, 128)),
    ("cap_1024_512",   (1024, 512)),
]
for label, h in capacities:
    variants.append(("capacity", label, dict(hidden_dims=h)))

# ---------------- Axis 4: BatchNorm on/off ----------------
print("\n=== Axis 4: BatchNorm ===")
variants.append(("batchnorm", "bn_on",  dict(use_batchnorm=True)))
variants.append(("batchnorm", "bn_off", dict(use_batchnorm=False)))

# ---------------- Axis 5: baseline pool ablation ----------------
print("\n=== Axis 5: drop-one baseline family ===")
families = {
    "similarity": ["pred_similarity"],
    "poly":       ["pred_poly"],
    "rbf":        ["pred_rbf_multiquadric_s0.0", "pred_rbf_multiquadric_s1.0", "pred_rbf_multiquadric_s2.0"],
    "tps":        ["pred_tps"],
    "sim_rbf":    ["pred_sim_rbf_multiquadric_s0.0", "pred_sim_rbf_multiquadric_s1.0", "pred_sim_rbf_multiquadric_s2.0"],
    "sim_tps":    ["pred_sim_tps"],
    "sim_pwa":    ["pred_sim_pwa"],
}
for fam, drop in families.items():
    keep = tuple(b for b in DEFAULT_BASELINES if b not in drop)
    variants.append(("baseline_pool", f"drop_{fam}", dict(baselines=keep, top_m=min(4, len(keep)))))

# ---------------- Axis 6: seed sensitivity ----------------
print("\n=== Axis 6: seed sensitivity (paper-faithful default) ===")
for s in [1047, 7, 42, 123, 2024]:
    variants.append(("seed", f"seed_{s}", dict(seed=s)))


# ---------------- Run all ----------------
results: List[Dict[str, Any]] = []
print(f"\nRunning {len(variants)} configurations...\n")
for axis, label, override in variants:
    d = run_one(label, axis, override)
    results.append(d)
    out_csv = Path(__file__).parent / "results.csv"
    save_results(results, out_csv)

# ---------------- Diagnostic LEAKY runs (a few for the report) ----------------
print("\n=== Diagnostic LEAKY runs (for leakage gap quantification) ===")
leaky_variants = [
    ("LEAKY_sel_all_noise0",       dict(selection="all",            noise_sigma_max=0.0,  leaky_features=True)),
    ("LEAKY_sel_topM4_noise0",     dict(selection="topM", top_m=4,  noise_sigma_max=0.0,  leaky_features=True)),
    ("LEAKY_sel_topM4_noise20",    dict(selection="topM", top_m=4,  noise_sigma_max=20.0, leaky_features=True)),
    ("LEAKY_cap_1024_512_topM4",   dict(hidden_dims=(1024, 512), selection="topM", top_m=4, noise_sigma_max=0.0, leaky_features=True)),
]
for label, ovr in leaky_variants:
    d = run_one(label, "leaky_diagnostic", ovr)
    results.append(d)

out_csv = Path(__file__).parent / "results.csv"
save_results(results, out_csv)
print(f"\nDone. Results saved to {out_csv}")
