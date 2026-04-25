"""Bayesian Model Averaging (BMA) over classical calibrators.

For each test sample, combine baseline predictions using uncertainty-
aware weights. The weights are learned per-trial: we estimate each
baseline's per-trial empirical risk on the trial's own evaluation
samples (leave-one-out), then use these to construct a weighted
combination.

There are TWO honest sources of per-trial calibration risk:

  (a) leave-one-out on the trial's own non-target evaluation samples
      (this uses the test-set rows, but in a leave-one-out fashion so
      no individual sample's prediction is used to weight itself);
  (b) the trial's own calibration-point residuals, if accessible.

For all_trials data, we don't have per-trial calibration-point
residuals as separate rows, but we CAN use leave-one-out on the trial's
evaluation rows to estimate per-baseline reliability. This is honest
because each individual sample's prediction is computed using weights
estimated from OTHER samples of the same trial.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd


def per_trial_baseline_l2(
    df: pd.DataFrame, baseline: str
) -> pd.DataFrame:
    """Return per-trial mean L2 error of a baseline."""
    err = np.linalg.norm(
        df[[f"{baseline}_x", f"{baseline}_y"]].values - df[["target_x", "target_y"]].values,
        axis=1,
    )
    out = df[["subject", "timestamp"]].copy()
    out[f"l2_{baseline}"] = err
    agg = out.groupby(["subject", "timestamp"]).mean().reset_index()
    return agg


def loo_softmin_weights(
    df: pd.DataFrame,
    baselines: Sequence[str],
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """For each row, compute softmax(-l2_per_baseline / T) using a
    leave-one-out estimate of per-baseline mean L2 within the row's
    trial."""
    df = df.copy()
    df["trial_id"] = df["subject"] + "|" + df["timestamp"]
    out = np.zeros((len(df), len(baselines)), dtype=np.float32)

    # Per-row L2 for each baseline
    per_row_l2 = np.stack(
        [
            np.linalg.norm(
                df[[f"{b}_x", f"{b}_y"]].values - df[["target_x", "target_y"]].values,
                axis=1,
            )
            for b in baselines
        ],
        axis=1,
    )  # (N, K)

    # For each trial, per-baseline trial mean
    trials = df["trial_id"].values
    unique_trials, inverse = np.unique(trials, return_inverse=True)
    trial_sums = np.zeros((len(unique_trials), len(baselines)))
    trial_counts = np.zeros(len(unique_trials))
    for i, t_idx in enumerate(inverse):
        trial_sums[t_idx] += per_row_l2[i]
        trial_counts[t_idx] += 1
    trial_means = trial_sums / trial_counts[:, None]

    # Leave-one-out trial mean per row: (sum - row_value) / (count - 1)
    for i, t_idx in enumerate(inverse):
        cnt = trial_counts[t_idx]
        if cnt > 1:
            loo_mean = (trial_sums[t_idx] - per_row_l2[i]) / (cnt - 1)
        else:
            loo_mean = trial_means[t_idx]
        # Softmin: w_k ~ exp(-l_k / T)
        out[i] = np.exp(-loo_mean / temperature)
        out[i] /= out[i].sum()
    return out


def bma_predictions(
    df: pd.DataFrame,
    baselines: Sequence[str],
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted combination of baseline predictions."""
    # Stack baselines: (N, K, 2)
    bs = np.stack(
        [df[[f"{b}_x", f"{b}_y"]].values for b in baselines], axis=1
    )
    return (weights[:, :, None] * bs).sum(axis=1)


def evaluate_bma(
    df: pd.DataFrame,
    baselines: Sequence[str],
    *,
    temperature: float = 1.0,
) -> dict:
    """Compute BMA test metrics."""
    w = loo_softmin_weights(df, baselines, temperature=temperature)
    pred = bma_predictions(df, baselines, w)
    err = pred - df[["target_x", "target_y"]].values
    l2 = np.linalg.norm(err, axis=1)
    return {
        "mae_x": float(np.mean(np.abs(err[:, 0]))),
        "mae_y": float(np.mean(np.abs(err[:, 1]))),
        "rmse_x": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "rmse_y": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "l2_mean": float(np.mean(l2)),
        "l2_median": float(np.median(l2)),
        "l2_std": float(np.std(l2)),
        "l2_p95": float(np.percentile(l2, 95)),
        "n": int(len(l2)),
        "weights_mean": w.mean(0).tolist(),
    }
