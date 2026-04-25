"""Learned per-trial shrinkage.

Instead of a fixed lambda(K), learn lambda from per-trial features:
  - empirical bias magnitude ||mean(target - anchor)||
  - empirical bias variance (trace of cov)
  - context size K
  - anchor screen position centroid (encoded with Fourier features)
  - per-baseline residual statistics

The estimator is a small MLP that outputs a 2D shrinkage vector
(lambda_x, lambda_y) for each trial, applied as
  bias_corrected = (lambda_x, lambda_y) * empirical_mean_bias

Trained with leave-one-trial-out + leave-one-subject-out cross-trial
splits, target is per-query L2 minimization.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "all_trials_combined.csv"

ANCHOR = "pred_sim_rbf_multiquadric_s2.0"

df_all = pd.read_csv(DATA)
df_all["trial_id"] = df_all["subject"] + "|" + df_all["timestamp"]


def trial_to_episodes(g: pd.DataFrame, k: int, anchor: str, rng: np.random.Generator):
    """Split trial g into (context, query) at random."""
    idx = rng.permutation(len(g))
    if len(g) <= k:
        return None
    ctx_idx, q_idx = idx[:k], idx[k:]
    ctx, q = g.iloc[ctx_idx], g.iloc[q_idx]
    ctx_residual = ctx[["target_x", "target_y"]].values - ctx[[f"{anchor}_x", f"{anchor}_y"]].values
    q_residual = q[["target_x", "target_y"]].values - q[[f"{anchor}_x", f"{anchor}_y"]].values
    return ctx_residual, q_residual, ctx, q


def episode_features(ctx_residual: np.ndarray, k: int) -> np.ndarray:
    """Per-episode features for the shrinkage estimator."""
    mean_b = ctx_residual.mean(axis=0)  # (2,)
    var_b = ctx_residual.var(axis=0) if k > 1 else np.zeros(2)
    range_b = ctx_residual.max(axis=0) - ctx_residual.min(axis=0) if k > 1 else np.zeros(2)
    feats = np.concatenate([
        mean_b / 50.0,             # 2
        np.log1p(np.abs(mean_b) / 10.0),  # 2
        var_b / 100.0,             # 2
        np.log1p(var_b / 10.0),    # 2
        range_b / 100.0,           # 2
        np.array([1.0 / max(k, 1), math.log(max(k, 1))]),  # 2
    ])
    return feats.astype(np.float32)


class ShrinkageNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        # Output 2 lambdas, init to 0.5 each
        self.head = nn.Linear(hidden, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        # logits centered at 0 -> sigmoid -> ~0.5
        return torch.sigmoid(self.head(h))


def gather_episodes(df: pd.DataFrame, k: int, anchor: str, n_episodes_per_trial: int = 3, seed: int = 0):
    """Yield episode-level samples: (features, mean_bias, q_residuals)."""
    df = df.copy()
    if "trial_id" not in df.columns:
        df["trial_id"] = df["subject"] + "|" + df["timestamp"]
    rng = np.random.default_rng(seed)
    items = []
    for trial_id, g in df.groupby("trial_id"):
        if len(g) <= k:
            continue
        for _ in range(n_episodes_per_trial):
            res = trial_to_episodes(g, k, anchor, rng)
            if res is None:
                continue
            ctx_residual, q_residual, ctx, q = res
            mean_b = ctx_residual.mean(axis=0)  # 2D bias
            feats = episode_features(ctx_residual, k)
            items.append({
                "feats": feats,
                "mean_bias": mean_b.astype(np.float32),
                "q_residual": q_residual.astype(np.float32),
            })
    return items


def evaluate_learned(model: ShrinkageNet, df: pd.DataFrame, k: int, anchor: str, *, device, seed: int = 0):
    """Eval at fixed K, deterministic per trial."""
    df = df.copy()
    if "trial_id" not in df.columns:
        df["trial_id"] = df["subject"] + "|" + df["timestamp"]
    model.eval()
    l2s = []
    with torch.no_grad():
        for trial_id, g in df.groupby("trial_id"):
            if len(g) <= k:
                continue
            rng = np.random.default_rng(hash(trial_id) & 0xFFFFFFFF)
            idx = rng.permutation(len(g))
            ctx_idx, q_idx = idx[:k], idx[k:]
            ctx, q = g.iloc[ctx_idx], g.iloc[q_idx]
            ctx_residual = ctx[["target_x", "target_y"]].values - ctx[[f"{anchor}_x", f"{anchor}_y"]].values
            q_residual = q[["target_x", "target_y"]].values - q[[f"{anchor}_x", f"{anchor}_y"]].values
            mean_b = ctx_residual.mean(axis=0)
            feats = torch.from_numpy(episode_features(ctx_residual, k)).unsqueeze(0).to(device)
            lam = model(feats).cpu().numpy()[0]  # (2,)
            corrected = lam * mean_b
            err = q_residual - corrected
            l2s.append(np.linalg.norm(err, axis=1))
    if not l2s:
        return float("nan")
    l2 = np.concatenate(l2s)
    return float(l2.mean())


def train_one(train_df, val_df, k: int, anchor: str, epochs: int = 80, device=None, seed: int = 0):
    device = device or torch.device("cpu")
    train_eps = gather_episodes(train_df, k, anchor, n_episodes_per_trial=10, seed=seed)
    val_eps = gather_episodes(val_df, k, anchor, n_episodes_per_trial=3, seed=seed + 1000)
    if not train_eps:
        return None
    feats_dim = train_eps[0]["feats"].shape[0]
    model = ShrinkageNet(feats_dim, hidden=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    best_val = float("inf")
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        # Mini-batch over episodes
        rng = np.random.default_rng(seed + ep)
        idx = rng.permutation(len(train_eps))
        bs = 32
        for i in range(0, len(idx), bs):
            batch = [train_eps[j] for j in idx[i:i + bs]]
            f = torch.from_numpy(np.stack([b["feats"] for b in batch])).to(device)
            mb = torch.from_numpy(np.stack([b["mean_bias"] for b in batch])).to(device)
            # For each query in this trial, the loss is mean L2 of q_residual - lam * mean_bias
            lam = model(f)  # (B, 2)
            corrections = lam * mb  # (B, 2)
            losses = []
            for j, b in enumerate(batch):
                qr = torch.from_numpy(b["q_residual"]).to(device)  # (Q, 2)
                err = qr - corrections[j]
                losses.append(((err ** 2).sum(dim=1)).mean())
            loss = torch.stack(losses).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        # Validate
        with torch.no_grad():
            model.eval()
            v_losses = []
            for b in val_eps:
                f = torch.from_numpy(b["feats"]).unsqueeze(0).to(device)
                mb = torch.from_numpy(b["mean_bias"]).unsqueeze(0).to(device)
                lam = model(f)
                corr = (lam * mb).squeeze(0)
                qr = torch.from_numpy(b["q_residual"]).to(device)
                err = qr - corr
                v_losses.append(torch.linalg.norm(err, dim=1).mean().item())
            v = float(np.mean(v_losses)) if v_losses else float("inf")
        if v < best_val:
            best_val = v
            best_state = {k_: t.detach().cpu().clone() for k_, t in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def main():
    print(f"Combined: {len(df_all)} rows, {df_all['subject'].nunique()} subjects, {df_all['trial_id'].nunique()} trials\n")

    # LOSO across subjects
    subjects = sorted(df_all["subject"].unique())
    summary = []
    for held in subjects:
        sub_df = df_all[df_all["subject"] == held]
        rest_df = df_all[df_all["subject"] != held].copy()
        rest_df["trial_id"] = rest_df["subject"] + "|" + rest_df["timestamp"]
        # Random val split (10%)
        trials = sorted(rest_df["trial_id"].unique())
        np.random.seed(0)
        np.random.shuffle(trials)
        n_val = max(1, int(len(trials) * 0.10))
        val_trials = set(trials[:n_val])
        val_df = rest_df[rest_df["trial_id"].isin(val_trials)].drop(columns=["trial_id"])
        train_df = rest_df[~rest_df["trial_id"].isin(val_trials)].drop(columns=["trial_id"])

        per_k = {}
        for k in [1, 2, 3, 5, 8, 12, 18]:
            r = train_one(train_df, val_df, k, ANCHOR, epochs=80)
            if r is None:
                continue
            model, _ = r
            test_l2 = evaluate_learned(model, sub_df, k, ANCHOR, device=torch.device("cpu"))
            per_k[k] = test_l2

        summary.append({"held": held, **{f"learned_K{k}": v for k, v in per_k.items()}})
        print(f"  {held:25s}  ", "  ".join(f"K{k}={v:.2f}" for k, v in per_k.items()))

    df_s = pd.DataFrame(summary)
    out_csv = Path(__file__).parent / "results.csv"
    df_s.to_csv(out_csv, index=False)
    with open(out_csv.with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Aggregate (mean over 12 LOSO folds) ===")
    for k in [1, 2, 3, 5, 8, 12, 18]:
        col = f"learned_K{k}"
        if col in df_s.columns:
            print(f"  K={k:2d}  learned shrinkage L2 = {df_s[col].mean():.2f}")


if __name__ == "__main__":
    main()
