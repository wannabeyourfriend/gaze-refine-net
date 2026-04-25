"""Synthetic benchmark for the Bayes-optimal shrinkage theory.

Setting: many synthetic 'trials'. Each trial t has a per-trial bias
    b_t ~ N(0, sigma2_bias I_2)
and per-fixation noise
    eps_{t,j} ~ N(0, sigma2_noise I_2).
Observations are r_{t,j} = b_t + eps_{t,j}.

We split each trial into K context samples and Q query samples; the
context yields an empirical mean bar_b. The "method" outputs a
shrinkage lambda to apply to bar_b; the prediction is then
    hat r_{t,j} = lambda * bar_b
and the loss on each query is ||hat r - r_query||_2^2 -- but since
r_query = b + eps, the irreducible noise floor is sigma2_noise.

Goals:
  1. Show that the closed-form Bayes-optimal lambda
     (sigma2_bias / (sigma2_bias + sigma2_noise/K))
     achieves the lowest L2.
  2. Show that the learned scalar shrinkage MLP recovers it.
  3. Add a heteroscedastic ablation where per-trial noise variance
     varies, and show that the learned shrinkage strictly outperforms
     the (mismatched) closed-form.
"""
from __future__ import annotations

import json
import math
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
    SpatialShrinkageNet,
    bayes_optimal_scalar,
    train_spatial_shrinkage,
    evaluate_spatial_shrinkage,
)


def make_trials(
    n_trials: int,
    samples_per_trial: int,
    sigma_bias: float,
    sigma_noise_dist: tuple,  # ('homo', sigma) or ('hetero', sigma_lo, sigma_hi)
    rng: np.random.Generator,
) -> List[Dict]:
    out = []
    for t in range(n_trials):
        b = rng.normal(0, sigma_bias, size=2)
        if sigma_noise_dist[0] == "homo":
            sigma = sigma_noise_dist[1]
        else:
            lo, hi = sigma_noise_dist[1], sigma_noise_dist[2]
            sigma = rng.uniform(lo, hi)
        anchor = np.zeros((samples_per_trial, 2), dtype=np.float32)  # arbitrary anchor (let's keep at 0)
        target = b + rng.normal(0, sigma, size=(samples_per_trial, 2))
        target = target.astype(np.float32)
        out.append({
            "trial_id": f"synth|t{t}",
            "subject": "synth",
            "anchor": anchor,
            "target": target,
            "orig":   anchor,  # not used here
            "b": b,
            "sigma": sigma,
        })
    return out


def closed_form_eval(trials: List[Dict], K: int, sigma_bias: float, sigma_noise: float) -> float:
    """Apply the closed-form lambda."""
    lam = bayes_optimal_scalar(sigma_bias ** 2, sigma_noise ** 2, K)
    errs = []
    for tr in trials:
        T = len(tr["target"])
        if T <= K: continue
        rng = np.random.default_rng(hash(tr["trial_id"]) & 0xFFFFFFFF)
        idx = rng.permutation(T)
        ctx_idx, q_idx = idx[:K], idx[K:]
        bar_b = tr["target"][ctx_idx].mean(axis=0)
        pred = lam * bar_b
        true_res = tr["target"][q_idx]
        e = pred - true_res  # since anchor==0, residual==target
        errs.append(np.linalg.norm(e, axis=1))
    return float(np.concatenate(errs).mean()) if errs else float("nan")


def raw_mean_eval(trials: List[Dict], K: int) -> float:
    return closed_form_eval_with_lambda(trials, K, 1.0)


def closed_form_eval_with_lambda(trials: List[Dict], K: int, lam: float) -> float:
    errs = []
    for tr in trials:
        T = len(tr["target"])
        if T <= K: continue
        rng = np.random.default_rng(hash(tr["trial_id"]) & 0xFFFFFFFF)
        idx = rng.permutation(T)
        ctx_idx, q_idx = idx[:K], idx[K:]
        bar_b = tr["target"][ctx_idx].mean(axis=0)
        pred = lam * bar_b
        true_res = tr["target"][q_idx]
        e = pred - true_res
        errs.append(np.linalg.norm(e, axis=1))
    return float(np.concatenate(errs).mean()) if errs else float("nan")


def main():
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rng = np.random.default_rng(0)

    SIGMA_BIAS = 30.0
    SIGMA_NOISE = 20.0  # homoscedastic
    SAMPLES_PER_TRIAL = 30

    # Generate train/val/test
    train_trials = make_trials(2000, SAMPLES_PER_TRIAL, SIGMA_BIAS, ("homo", SIGMA_NOISE), rng)
    val_trials   = make_trials(300,  SAMPLES_PER_TRIAL, SIGMA_BIAS, ("homo", SIGMA_NOISE), rng)
    test_trials  = make_trials(500,  SAMPLES_PER_TRIAL, SIGMA_BIAS, ("homo", SIGMA_NOISE), rng)

    # Train scalar (non-spatial) shrinkage model
    cfg = SpatialShrinkageConfig(
        spatial=False,
        epochs=80,
        K_train_min=1,
        K_train_max=18,
        ctx_hidden=64, ctx_summary_dim=32, head_hidden=(64, 32),
        screen_w=400.0, screen_h=400.0,  # synthetic 'screen' just for normalization
        seed=1047,
    )
    print("Training scalar shrinkage on synthetic homoscedastic data...")
    model_scalar, info = train_spatial_shrinkage(train_trials, val_trials, cfg, device=device, verbose=True)

    rows = []
    for K in [1, 2, 3, 5, 8, 12, 18]:
        lam_star = bayes_optimal_scalar(SIGMA_BIAS ** 2, SIGMA_NOISE ** 2, K)
        # 1) Bayes-optimal closed form
        cf = closed_form_eval(test_trials, K, SIGMA_BIAS, SIGMA_NOISE)
        # 2) Raw empirical mean (lambda = 1)
        raw = closed_form_eval_with_lambda(test_trials, K, 1.0)
        # 3) Learned scalar shrinkage
        learned = evaluate_spatial_shrinkage(model_scalar, test_trials, K, device=device)
        # 4) Anchor only (lambda = 0)
        anchor = closed_form_eval_with_lambda(test_trials, K, 0.0)
        # Theoretical floor: just the noise after applying lam_star
        # E ||lam* bar_b - b - eps||^2 = (1-lam*)^2 * sigma_b^2 + (lam*^2 / K) * sigma_eps^2 + sigma_eps^2
        var_term = (1 - lam_star) ** 2 * SIGMA_BIAS ** 2 + (lam_star ** 2 / K) * SIGMA_NOISE ** 2
        # Per-coord MSE; total expected ||.||^2 = 2*(var_term + sigma_eps^2)
        # but we report mean L2 norm, not L2^2.
        # Approx: mean of chi(2) with variance v is sqrt(pi*v/2)
        v_total_per_coord = var_term + SIGMA_NOISE ** 2
        theory_mean_l2 = math.sqrt(math.pi * v_total_per_coord)
        rows.append({
            "K": K, "lam_star": lam_star,
            "anchor_only": anchor, "raw_mean": raw,
            "closed_form": cf, "learned_scalar": learned["l2_mean"],
            "theory_mean_l2": theory_mean_l2,
        })
        print(f"K={K:2d}  lam*={lam_star:.3f}  anchor={anchor:.2f}  raw={raw:.2f}  "
              f"closed_form={cf:.2f}  learned={learned['l2_mean']:.2f}  theory={theory_mean_l2:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results_homo.csv", index=False)
    with open(out_dir / "results_homo.json", "w") as f:
        json.dump(rows, f, indent=2)

    # ---- heteroscedastic: same setup but per-trial sigma in [10, 30] ----
    print("\n=== Heteroscedastic experiment ===")
    rng2 = np.random.default_rng(1)
    train_h = make_trials(2000, SAMPLES_PER_TRIAL, SIGMA_BIAS, ("hetero", 10.0, 35.0), rng2)
    val_h   = make_trials(300,  SAMPLES_PER_TRIAL, SIGMA_BIAS, ("hetero", 10.0, 35.0), rng2)
    test_h  = make_trials(500,  SAMPLES_PER_TRIAL, SIGMA_BIAS, ("hetero", 10.0, 35.0), rng2)

    cfg_h = SpatialShrinkageConfig(
        spatial=False, epochs=80, K_train_min=1, K_train_max=18,
        ctx_hidden=64, ctx_summary_dim=32, head_hidden=(64, 32),
        screen_w=400.0, screen_h=400.0, seed=1047,
    )
    print("Training scalar shrinkage on synthetic heteroscedastic data...")
    model_h, _ = train_spatial_shrinkage(train_h, val_h, cfg_h, device=device, verbose=True)

    # closed-form is misspecified: it assumes a single sigma. Use the population-mean sigma.
    SIGMA_NOISE_FAKE = math.sqrt(((10.0 ** 2 + 35.0 ** 2) / 2))  # avg variance
    rows_h = []
    for K in [1, 2, 3, 5, 8, 12, 18]:
        lam_star = bayes_optimal_scalar(SIGMA_BIAS ** 2, SIGMA_NOISE_FAKE ** 2, K)
        cf = closed_form_eval_with_lambda(test_h, K, lam_star)
        raw = closed_form_eval_with_lambda(test_h, K, 1.0)
        learned = evaluate_spatial_shrinkage(model_h, test_h, K, device=device)
        anchor = closed_form_eval_with_lambda(test_h, K, 0.0)
        rows_h.append({
            "K": K, "lam_star_misspec": lam_star,
            "anchor_only": anchor, "raw_mean": raw,
            "closed_form_misspec": cf, "learned_scalar": learned["l2_mean"],
        })
        print(f"K={K:2d}  lam*~={lam_star:.3f}  anchor={anchor:.2f}  raw={raw:.2f}  "
              f"closed_form_misspec={cf:.2f}  learned={learned['l2_mean']:.2f}")

    pd.DataFrame(rows_h).to_csv(out_dir / "results_hetero.csv", index=False)
    with open(out_dir / "results_hetero.json", "w") as f:
        json.dump(rows_h, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
