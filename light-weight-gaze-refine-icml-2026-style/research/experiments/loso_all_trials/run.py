"""LOSO evaluation on the all_trials data with v2 honest pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from loso_pipeline import loso_evaluate
from refine_v2 import V2Config

DATA = ROOT.parent.parent / "data" / "prepared" / "all_trials_split"

# Combine train+val+test into a single CSV so we can re-split LOSO.
train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "val.csv")
test = pd.read_csv(DATA / "test.csv")
combined = pd.concat([train, val, test], ignore_index=True)
combined_csv = ROOT / "data" / "all_trials_combined.csv"
combined_csv.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(combined_csv, index=False)
print(f"Combined: {len(combined)} rows, {combined['subject'].nunique()} subjects")

cfg_template = V2Config(
    train_csv="",
    val_csv="",
    test_csv="",
    screen_w=1920.0,
    screen_h=1080.0,
    epochs=200,
    seed=1047,
    lr=1e-3,
    weight_decay=1e-3,
    batch_size=64,
    arch="mlp_residual",
    hidden_dims=(64, 32),
    use_fourier=True,
    fourier_bands=4,
    noise_sigma_max=10.0,
    anchor_baseline="pred_sim_rbf_multiquadric_s2.0",
    baselines=("pred_sim_rbf_multiquadric_s2.0", "pred_similarity", "pred_poly", "pred_gpr", "pred_tps", "pred_sim_pwa"),
    use_layernorm=True,
)

results = loso_evaluate(str(combined_csv), cfg_template)

# Save
out_csv = Path(__file__).parent / "results.csv"
pd.DataFrame(results).to_csv(out_csv, index=False)
with open(out_csv.with_suffix(".json"), "w") as f:
    json.dump(results, f, indent=2)

# Aggregate
print("\n\n=== LOSO Summary ===")
df = pd.DataFrame(results)
print(f"Mean baseline L2: {df['baseline_l2'].mean():.2f} px")
print(f"Mean ours L2:     {df['test_l2_mean'].mean():.2f} px")
print(f"Mean delta:       {df['delta_vs_baseline'].mean():+.2f} px")
print(f"Median delta:     {df['delta_vs_baseline'].median():+.2f} px")
print(f"# subjects improved: {(df['delta_vs_baseline'] < 0).sum()} / {len(df)}")
