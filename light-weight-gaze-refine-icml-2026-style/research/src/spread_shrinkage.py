"""Spread-aware spatial shrinkage (variant 3 + 5).

Extends `spatial_shrinkage.SpatialShrinkageNet` with two related ideas:

  (3) **Spread-aware features**: include per-fixation `spread` as a per-context-item
      feature. The Bayes-optimal scalar is
          lambda* = sigma_b^2 / (sigma_b^2 + sigma_eps^2 / K)
      where sigma_eps is the per-fixation noise std. Spread is a direct estimate
      of sigma_eps, so the network has the relevant statistic to choose a
      data-adaptive lambda.

  (5) **Soft inverse-variance weighted mean**: instead of an unweighted empirical
      mean of context residuals, compute a heteroscedastic mean estimator
          bar_b = sum_j w_j r_j / sum_j w_j  with  w_j = 1 / (spread_j^2 + eps).
      This down-weights noisy context fixations.

Both can be toggled independently. They share the same context-encoder /
shrinkage-head architecture as v3.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from spatial_shrinkage import (
    fourier,
    screen_norm,
    ContextEncoder,
    SpatialShrinkageHead,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SpreadShrinkageConfig:
    # Spatial encoding
    screen_w: float = 1920.0
    screen_h: float = 1080.0
    use_fourier: bool = True
    fourier_bands: int = 6

    # Context encoder
    ctx_hidden: int = 128
    ctx_layers: int = 2
    ctx_summary_dim: int = 64

    # Shrinkage head
    head_hidden: Sequence[int] = (128, 64)
    dropout: float = 0.0
    use_layernorm: bool = True

    # Output structure
    spatial: bool = True
    lambda_min: float = 0.0
    lambda_max: float = 1.0

    # Spread features
    use_spread_item_feat: bool = True   # add spread as per-item feature
    use_spread_global_feat: bool = True  # add per-trial spread aggregates to head input
    inverse_variance_mean: bool = True   # use inverse-variance-weighted bar_b

    # Multi-anchor (variant 1, optional)
    use_second_anchor: bool = False
    # When True, the model receives a second anchor's residual as additional context
    # and outputs a single shrinkage; the corrected prediction uses the primary anchor.
    # Multi-anchor blending is handled at the data-loading layer.

    # Training
    epochs: int = 120
    lr: float = 3e-3
    weight_decay: float = 1e-3
    batch_size_trials: int = 16
    queries_per_trial: int = 8
    seed: int = 1047
    K_train_min: int = 1
    K_train_max: int = 18


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SpreadShrinkageNet(nn.Module):
    """Spread-aware spatial shrinkage."""

    def __init__(self, cfg: SpreadShrinkageConfig) -> None:
        super().__init__()
        self.cfg = cfg
        anchor_n = 2
        if cfg.use_fourier:
            anchor_n = 2 * (1 + 2 * cfg.fourier_bands)
        residual_n = 2
        K_feat = 2  # 1/K, log K
        spread_item_feat = 2 if cfg.use_spread_item_feat else 0  # spread/50, log(spread)
        item_dim = anchor_n + residual_n + K_feat + spread_item_feat
        self.encoder = ContextEncoder(item_dim, cfg.ctx_hidden, cfg.ctx_layers, cfg.ctx_summary_dim)

        # Per-trial spread summary fed to the head: mean(spread)/50, min(spread)/50,
        # max(spread)/50, std(spread)/50.
        self.spread_global_dim = 4 if cfg.use_spread_global_feat else 0
        pos_n = anchor_n + self.spread_global_dim
        self.head = SpatialShrinkageHead(
            ctx_dim=cfg.ctx_summary_dim,
            pos_dim=pos_n,
            hidden=cfg.head_hidden,
            dropout=cfg.dropout,
            use_layernorm=cfg.use_layernorm,
            lambda_min=cfg.lambda_min,
            lambda_max=cfg.lambda_max,
        )

    def _featurize_position(self, p: torch.Tensor) -> torch.Tensor:
        n = screen_norm(p, self.cfg.screen_w, self.cfg.screen_h)
        if self.cfg.use_fourier:
            n = fourier(n, self.cfg.fourier_bands)
        return n

    def _featurize_context(
        self,
        anchor: torch.Tensor,    # (B, K, 2)
        residual: torch.Tensor,  # (B, K, 2)
        spread: torch.Tensor,    # (B, K)
        K_actual: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        anchor_feat = self._featurize_position(anchor)
        res_feat = residual / 50.0
        invK = (1.0 / K_actual.clamp(min=1).float()).view(-1, 1, 1).expand(-1, anchor.shape[1], 1)
        logK = torch.log(K_actual.clamp(min=1).float()).view(-1, 1, 1).expand(-1, anchor.shape[1], 1)
        feats = [anchor_feat, res_feat, invK, logK]
        if self.cfg.use_spread_item_feat:
            sp = spread.unsqueeze(-1)  # (B, K, 1)
            feats.append(sp / 50.0)
            feats.append(torch.log1p(sp.clamp(min=0)) / 5.0)
        return torch.cat(feats, dim=-1)

    def _spread_global_feats(self, spread: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute per-trial spread aggregates: (mean, min, max, std)/50."""
        m = mask.float()
        m_sum = m.sum(dim=1).clamp(min=1)
        sp = spread * m
        mean = sp.sum(dim=1) / m_sum
        very_large = torch.full_like(spread, 1e6)
        very_small = torch.full_like(spread, -1e6)
        sp_for_min = torch.where(mask, spread, very_large)
        sp_for_max = torch.where(mask, spread, very_small)
        smin = sp_for_min.min(dim=1).values
        smax = sp_for_max.max(dim=1).values
        var = ((spread - mean.unsqueeze(1)) ** 2 * m).sum(dim=1) / m_sum.clamp(min=1)
        std = var.sqrt()
        return torch.stack([mean, smin, smax, std], dim=-1) / 50.0

    def forward(
        self,
        ctx_anchor: torch.Tensor,    # (B, K, 2)
        ctx_residual: torch.Tensor,  # (B, K, 2)
        ctx_mask: torch.Tensor,      # (B, K)
        ctx_spread: torch.Tensor,    # (B, K)
        query_pos: torch.Tensor,     # (B, Q, 2)
        K_actual: torch.Tensor,      # (B,)
    ) -> torch.Tensor:
        ctx_items = self._featurize_context(ctx_anchor, ctx_residual, ctx_spread, K_actual)
        h = self.encoder(ctx_items, ctx_mask)
        # Per-query lambda
        if self.cfg.spatial:
            q_feat = self._featurize_position(query_pos)
            if self.cfg.use_spread_global_feat:
                gs = self._spread_global_feats(ctx_spread, ctx_mask)
                gs_b = gs.unsqueeze(1).expand(-1, q_feat.shape[1], -1)
                q_feat = torch.cat([q_feat, gs_b], dim=-1)
            h_b = h.unsqueeze(1).expand(-1, q_feat.shape[1], -1)
            return self.head(h_b, q_feat)
        # Non-spatial: lambda is per-trial; broadcast across queries
        zero_pos = torch.zeros_like(self._featurize_position(query_pos[:, :1]))[:, 0]
        if self.cfg.use_spread_global_feat:
            zero_pos = torch.cat([zero_pos, self._spread_global_feats(ctx_spread, ctx_mask)], dim=-1)
        lam_per_trial = self.head(h, zero_pos)
        return lam_per_trial.unsqueeze(1).expand(-1, query_pos.shape[1], -1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _trial_arrays_with_spread(df: pd.DataFrame, anchor_col: str) -> List[Dict[str, np.ndarray]]:
    """Group a DataFrame into per-trial dicts, including per-fixation spread."""
    if "origin_gaze_x" in df.columns:
        ox, oy = "origin_gaze_x", "origin_gaze_y"
    else:
        ox, oy = "original_gaze_x", "original_gaze_y"
    df = df.copy()
    df["trial_id"] = df["subject"] + "|" + df["timestamp"]
    has_spread = "spread" in df.columns
    out: List[Dict[str, np.ndarray]] = []
    for trial_id, g in df.groupby("trial_id"):
        if len(g) < 4:
            continue
        anchor = g[[f"{anchor_col}_x", f"{anchor_col}_y"]].to_numpy(dtype=np.float32)
        target = g[["target_x", "target_y"]].to_numpy(dtype=np.float32)
        orig = g[[ox, oy]].to_numpy(dtype=np.float32)
        spread = g["spread"].to_numpy(dtype=np.float32) if has_spread else np.full(len(g), 30.0, dtype=np.float32)
        out.append({
            "trial_id": trial_id,
            "subject": g["subject"].iloc[0],
            "anchor": anchor,
            "target": target,
            "orig": orig,
            "spread": spread,
        })
    return out


def load_trials_with_spread(csv_path: str, anchor_col: str) -> List[Dict[str, np.ndarray]]:
    return _trial_arrays_with_spread(pd.read_csv(csv_path), anchor_col)


def _sample_episode_v4(trial, K, Q, rng, K_max):
    T = len(trial["anchor"])
    if T <= K:
        return None
    idx = rng.permutation(T)
    ctx_idx, q_idx = idx[:K], idx[K:K + Q]
    if len(q_idx) == 0:
        q_idx = idx[K:K + 1]
    n = K
    a = np.zeros((K_max, 2), dtype=np.float32)
    r = np.zeros((K_max, 2), dtype=np.float32)
    sp = np.zeros(K_max, dtype=np.float32)
    a[:n] = trial["anchor"][ctx_idx]
    r[:n] = trial["target"][ctx_idx] - trial["anchor"][ctx_idx]
    sp[:n] = trial["spread"][ctx_idx]
    mask = np.zeros(K_max, dtype=np.bool_)
    mask[:n] = True
    return {
        "ctx_anchor": a, "ctx_residual": r, "ctx_spread": sp, "ctx_mask": mask, "K_actual": np.int64(n),
        "q_pos":   trial["anchor"][q_idx],
        "q_target": trial["target"][q_idx],
    }


def _empirical_mean(residual: torch.Tensor, mask: torch.Tensor, spread: torch.Tensor,
                    inverse_variance: bool, eps: float = 25.0) -> torch.Tensor:
    """Empirical mean bar_b from context residuals. spread in pixels.

    If inverse_variance: weights = 1 / (spread^2 + eps^2).
    """
    if not inverse_variance:
        m = mask.unsqueeze(-1).float()
        return (residual * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
    w = (1.0 / (spread.float() ** 2 + eps ** 2)) * mask.float()  # (B, K)
    w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return (residual * w.unsqueeze(-1)).sum(dim=1) / w_sum


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------

def train_spread_shrinkage(
    train_trials: List[Dict[str, np.ndarray]],
    val_trials: List[Dict[str, np.ndarray]],
    cfg: SpreadShrinkageConfig,
    *,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Tuple[SpreadShrinkageNet, Dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model = SpreadShrinkageNet(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

    K_max = cfg.K_train_max
    best_val = float("inf"); best_state = None; best_epoch = 0

    for ep in range(1, cfg.epochs + 1):
        model.train()
        rng.shuffle(train_trials)
        for start in range(0, len(train_trials), cfg.batch_size_trials):
            batch_trials = train_trials[start:start + cfg.batch_size_trials]
            episodes = []
            for tr in batch_trials:
                K = int(rng.integers(cfg.K_train_min, cfg.K_train_max + 1))
                e = _sample_episode_v4(tr, K, cfg.queries_per_trial, rng, K_max)
                if e is not None:
                    episodes.append(e)
            if not episodes:
                continue
            max_q = max(e["q_pos"].shape[0] for e in episodes)
            B = len(episodes)
            ctx_anchor = torch.from_numpy(np.stack([e["ctx_anchor"] for e in episodes])).to(device)
            ctx_residual = torch.from_numpy(np.stack([e["ctx_residual"] for e in episodes])).to(device)
            ctx_spread = torch.from_numpy(np.stack([e["ctx_spread"] for e in episodes])).to(device)
            ctx_mask = torch.from_numpy(np.stack([e["ctx_mask"] for e in episodes])).to(device)
            K_actual = torch.from_numpy(np.array([e["K_actual"] for e in episodes])).to(device)
            q_pos = torch.zeros((B, max_q, 2), device=device)
            q_tgt = torch.zeros((B, max_q, 2), device=device)
            q_mask = torch.zeros((B, max_q), dtype=torch.bool, device=device)
            for i, e in enumerate(episodes):
                qn = e["q_pos"].shape[0]
                q_pos[i, :qn] = torch.from_numpy(e["q_pos"]).to(device)
                q_tgt[i, :qn] = torch.from_numpy(e["q_target"]).to(device)
                q_mask[i, :qn] = True
            bar_b = _empirical_mean(ctx_residual, ctx_mask, ctx_spread, cfg.inverse_variance_mean)
            lam = model(ctx_anchor, ctx_residual, ctx_mask, ctx_spread, q_pos, K_actual)
            corr = lam * bar_b.unsqueeze(1)
            true_res = q_tgt - q_pos
            err = (corr - true_res) * q_mask.unsqueeze(-1).float()
            loss = (err.pow(2).sum(dim=-1) * q_mask.float()).sum() / q_mask.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        with torch.no_grad():
            model.eval()
            v_loss = 0.0; n = 0
            K_eval = max(1, K_max // 2)
            for tr in val_trials:
                ep_data = _sample_episode_v4(
                    tr, K_eval, len(tr["anchor"]),
                    np.random.default_rng(hash(tr["trial_id"]) & 0xFFFFFFFF),
                    K_max,
                )
                if ep_data is None:
                    continue
                ctx_anchor = torch.from_numpy(ep_data["ctx_anchor"]).unsqueeze(0).to(device)
                ctx_residual = torch.from_numpy(ep_data["ctx_residual"]).unsqueeze(0).to(device)
                ctx_spread = torch.from_numpy(ep_data["ctx_spread"]).unsqueeze(0).to(device)
                ctx_mask = torch.from_numpy(ep_data["ctx_mask"]).unsqueeze(0).to(device)
                K_actual = torch.tensor([ep_data["K_actual"]]).to(device)
                q_pos = torch.from_numpy(ep_data["q_pos"]).unsqueeze(0).to(device)
                q_tgt = torch.from_numpy(ep_data["q_target"]).unsqueeze(0).to(device)
                bar_b = _empirical_mean(ctx_residual, ctx_mask, ctx_spread, cfg.inverse_variance_mean)
                lam = model(ctx_anchor, ctx_residual, ctx_mask, ctx_spread, q_pos, K_actual)
                corr = lam * bar_b.unsqueeze(1)
                true_res = q_tgt - q_pos
                err = (corr - true_res).pow(2).sum(dim=-1).sqrt().mean()
                v_loss += float(err) * q_pos.shape[1]
                n += q_pos.shape[1]
            val = v_loss / max(n, 1)
        if val < best_val:
            best_val = val; best_epoch = ep
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if verbose and (ep % 20 == 0 or ep == cfg.epochs):
            print(f"  ep {ep:3d}  val_L2 = {val:.2f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_l2": best_val, "best_epoch": best_epoch}


@torch.no_grad()
def evaluate_spread_shrinkage(
    model: SpreadShrinkageNet,
    test_trials: List[Dict[str, np.ndarray]],
    K: int,
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    device = device or next(model.parameters()).device
    model.eval()
    cfg = model.cfg
    K_max = cfg.K_train_max
    all_l2: List[np.ndarray] = []
    n_skipped = 0
    for tr in test_trials:
        T = len(tr["anchor"])
        if T <= K:
            n_skipped += 1; continue
        rng = np.random.default_rng(hash(tr["trial_id"]) & 0xFFFFFFFF)
        idx = rng.permutation(T)
        K_use = min(K, K_max)
        ctx_idx, q_idx = idx[:K_use], idx[K_use:]
        a = np.zeros((K_max, 2), dtype=np.float32)
        r = np.zeros((K_max, 2), dtype=np.float32)
        sp = np.zeros(K_max, dtype=np.float32)
        a[:K_use] = tr["anchor"][ctx_idx]
        r[:K_use] = tr["target"][ctx_idx] - tr["anchor"][ctx_idx]
        sp[:K_use] = tr["spread"][ctx_idx]
        mask = np.zeros(K_max, dtype=np.bool_); mask[:K_use] = True
        ctx_anchor = torch.from_numpy(a).unsqueeze(0).to(device)
        ctx_residual = torch.from_numpy(r).unsqueeze(0).to(device)
        ctx_spread = torch.from_numpy(sp).unsqueeze(0).to(device)
        ctx_mask = torch.from_numpy(mask).unsqueeze(0).to(device)
        K_actual = torch.tensor([K_use]).to(device)
        q_pos = torch.from_numpy(tr["anchor"][q_idx]).unsqueeze(0).to(device)
        q_tgt = torch.from_numpy(tr["target"][q_idx]).unsqueeze(0).to(device)
        bar_b = _empirical_mean(ctx_residual, ctx_mask, ctx_spread, cfg.inverse_variance_mean)
        lam = model(ctx_anchor, ctx_residual, ctx_mask, ctx_spread, q_pos, K_actual)
        pred = q_pos + lam * bar_b.unsqueeze(1)
        err = (pred - q_tgt).cpu().numpy()[0]
        all_l2.append(np.linalg.norm(err, axis=1))
    if not all_l2:
        return {"l2_mean": float("nan"), "n_eval": 0, "n_skipped_trials": n_skipped}
    l2 = np.concatenate(all_l2)
    return {
        "l2_mean": float(l2.mean()),
        "l2_median": float(np.median(l2)),
        "l2_std": float(l2.std()),
        "n_eval": int(len(l2)),
        "n_skipped_trials": int(n_skipped),
    }
