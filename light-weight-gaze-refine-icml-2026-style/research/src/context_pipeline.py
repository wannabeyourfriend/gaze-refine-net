"""Context-conditioned refinement.

At inference time, the model receives:
  - the test sample's (orig_gaze, baseline_predictions, anchor)
  - K *context* samples from the same trial, each with their
    (orig_gaze, baseline_predictions, anchor, target).

It uses the context to estimate per-trial bias and OTHER signals
(e.g., baseline-specific per-trial reliability, spatial structure of
residuals), then refines the test sample's prediction.

Architecture:
  - Context encoder: small Set-Transformer / DeepSets over context
    items (each item: anchor_relative orig, baselines, residual).
    Pools to a per-trial summary vector h_trial.
  - Predictor: MLP on (anchor_pos_features, query_orig_relative,
    baselines_relative, h_trial) -> 2-d residual added to anchor.

Training: per-trial mini-batches. For each trial, sample K context
points and treat the rest as queries. The network must predict the
query's target from anchor + residual head.

Strict no-leakage rule: at training time, the residual head sees the
TRUE residual on context samples (target - anchor) but never on query
samples.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass
class CtxCfg:
    baselines: Sequence[str]
    anchor: str
    screen_w: float = 1920.0
    screen_h: float = 1080.0
    use_fourier: bool = True
    fourier_bands: int = 4

    context_size: int = 12         # K samples
    context_dropout_p: float = 0.0  # randomly drop ctx samples in training

    encoder_hidden: int = 64
    encoder_layers: int = 2
    summary_dim: int = 32

    predictor_hidden: Sequence[int] = (64, 32)
    dropout: float = 0.0

    epochs: int = 200
    batch_size_trials: int = 8
    queries_per_trial: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-3
    seed: int = 1047


def fourier(x: torch.Tensor, n: int) -> torch.Tensor:
    if n <= 0:
        return x
    bands = 2.0 ** torch.arange(n, dtype=x.dtype, device=x.device)
    xb = x.unsqueeze(-1) * bands
    return torch.cat([x, torch.sin(math.pi * xb).flatten(-2), torch.cos(math.pi * xb).flatten(-2)], -1)


def screen_norm(xy: torch.Tensor, w: float, h: float) -> torch.Tensor:
    out = xy.clone()
    out[..., 0] = (out[..., 0] / w) * 2 - 1
    out[..., 1] = (out[..., 1] / h) * 2 - 1
    return out


def _trial_iter(df: pd.DataFrame, cfg: CtxCfg, anchor: str, baselines: Sequence[str]):
    """Yield per-trial dicts of arrays needed for context-aware training.

    Returns dict with shapes (n_trial, ...):
      orig: (T, 2), baselines: (T, K, 2), anchor: (T, 2), targets: (T, 2)
    """
    if "origin_gaze_x" in df.columns:
        ox, oy = "origin_gaze_x", "origin_gaze_y"
    else:
        ox, oy = "original_gaze_x", "original_gaze_y"
    df = df.copy()
    df["trial_id"] = df["subject"] + "|" + df["timestamp"]
    anchor_idx = list(baselines).index(anchor)
    for trial_id, g in df.groupby("trial_id"):
        if len(g) < 4:
            continue
        orig = g[[ox, oy]].to_numpy(dtype=np.float32)
        targets = g[["target_x", "target_y"]].to_numpy(dtype=np.float32)
        bs = np.stack([g[[f"{b}_x", f"{b}_y"]].to_numpy(dtype=np.float32) for b in baselines], axis=1)
        anchor_pred = bs[:, anchor_idx, :]
        yield trial_id, orig, bs, anchor_pred, targets


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ContextEncoder(nn.Module):
    """DeepSets-style encoder that summarizes context items into h_trial."""

    def __init__(self, item_dim: int, hidden: int, layers: int, summary_dim: int) -> None:
        super().__init__()
        mods: List[nn.Module] = []
        prev = item_dim
        for _ in range(layers):
            mods += [nn.Linear(prev, hidden), nn.LayerNorm(hidden), nn.GELU()]
            prev = hidden
        self.body = nn.Sequential(*mods)
        self.proj = nn.Linear(prev, summary_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, K, item_dim), mask: (B, K) bool, True=valid
        h = self.body(x)  # (B, K, H)
        h = self.proj(h)  # (B, K, S)
        m = mask.unsqueeze(-1).float()
        s = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        return s


class ContextRefiner(nn.Module):
    def __init__(self, cfg: CtxCfg, ctx_item_dim: int, query_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = ContextEncoder(ctx_item_dim, cfg.encoder_hidden, cfg.encoder_layers, cfg.summary_dim)
        layers: List[nn.Module] = []
        prev = query_dim + cfg.summary_dim
        for h in cfg.predictor_hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU()]
            if cfg.dropout > 0:
                layers += [nn.Dropout(cfg.dropout)]
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, ctx_items: torch.Tensor, ctx_mask: torch.Tensor, query_feat: torch.Tensor) -> torch.Tensor:
        h = self.encoder(ctx_items, ctx_mask)  # (B, S)
        x = torch.cat([query_feat, h], dim=-1)
        return self.head(self.body(x))


def build_ctx_features(orig_q, bs_q, anchor_q, cfg: CtxCfg, *, residual_tgt: Optional[torch.Tensor] = None):
    """Build (query_feat, ctx_item_template_dim).

    query_feat encodes only QUERY information (no target).
    ctx_item_template_dim is computed from a single context sample with
    (orig, baselines, anchor, target_residual).
    """
    px = 50.0
    anchor_n = screen_norm(anchor_q, cfg.screen_w, cfg.screen_h)
    if cfg.use_fourier:
        anchor_n = fourier(anchor_n, cfg.fourier_bands)
    orig_rel = (orig_q - anchor_q) / px
    bs_rel = (bs_q - anchor_q.unsqueeze(-2)) / px
    bs_flat = bs_rel.reshape(*bs_rel.shape[:-2], -1)
    return torch.cat([anchor_n, orig_rel, bs_flat], dim=-1)


def build_ctx_item_features(orig_c, bs_c, anchor_c, residual_c, cfg: CtxCfg):
    """Per-context-item: same as query feat, plus (target - anchor) / px."""
    px = 50.0
    anchor_n = screen_norm(anchor_c, cfg.screen_w, cfg.screen_h)
    if cfg.use_fourier:
        anchor_n = fourier(anchor_n, cfg.fourier_bands)
    orig_rel = (orig_c - anchor_c) / px
    bs_rel = (bs_c - anchor_c.unsqueeze(-2)) / px
    bs_flat = bs_rel.reshape(*bs_rel.shape[:-2], -1)
    res_rel = residual_c / px  # ground truth residual on context (target - anchor)
    return torch.cat([anchor_n, orig_rel, bs_flat, res_rel], dim=-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrialBatcher:
    def __init__(self, df: pd.DataFrame, cfg: CtxCfg, baselines: Sequence[str], anchor: str, *, is_training: bool):
        self.cfg = cfg
        self.is_training = is_training
        self.baselines = list(baselines)
        self.anchor = anchor
        self.trials: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = list(_trial_iter(df, cfg, anchor, baselines))

    def __len__(self) -> int:
        return len(self.trials)

    def sample_batch(self, rng: np.random.Generator, n_trials: int):
        chosen = rng.choice(len(self.trials), size=min(n_trials, len(self.trials)), replace=False)
        cfg = self.cfg
        K = cfg.context_size
        Q = cfg.queries_per_trial
        all_ctx: List[np.ndarray] = []
        all_ctx_mask: List[np.ndarray] = []
        all_q_orig: List[np.ndarray] = []
        all_q_bs: List[np.ndarray] = []
        all_q_anchor: List[np.ndarray] = []
        all_q_target: List[np.ndarray] = []
        for ti in chosen:
            _, orig, bs, anchor, targets = self.trials[ti]
            T = len(orig)
            idx = rng.permutation(T)
            n_ctx = min(K, T - 1)
            ctx_idx = idx[:n_ctx]
            q_idx = idx[n_ctx:n_ctx + Q]
            if len(q_idx) == 0:
                q_idx = idx[:1]
            ctx_orig = orig[ctx_idx]
            ctx_bs = bs[ctx_idx]
            ctx_anchor = anchor[ctx_idx]
            ctx_residual = targets[ctx_idx] - ctx_anchor
            all_ctx_mask.append(np.concatenate([np.ones(n_ctx, dtype=np.bool_), np.zeros(K - n_ctx, dtype=np.bool_)]))
            ctx_pad_orig = np.concatenate([ctx_orig, np.zeros((K - n_ctx, 2), dtype=np.float32)], axis=0)
            ctx_pad_bs = np.concatenate([ctx_bs, np.zeros((K - n_ctx, len(self.baselines), 2), dtype=np.float32)], axis=0)
            ctx_pad_anchor = np.concatenate([ctx_anchor, np.zeros((K - n_ctx, 2), dtype=np.float32)], axis=0)
            ctx_pad_residual = np.concatenate([ctx_residual, np.zeros((K - n_ctx, 2), dtype=np.float32)], axis=0)
            all_ctx.append((ctx_pad_orig, ctx_pad_bs, ctx_pad_anchor, ctx_pad_residual))
            all_q_orig.append(orig[q_idx])
            all_q_bs.append(bs[q_idx])
            all_q_anchor.append(anchor[q_idx])
            all_q_target.append(targets[q_idx])
        return all_ctx, all_ctx_mask, all_q_orig, all_q_bs, all_q_anchor, all_q_target


def evaluate_context(
    model: ContextRefiner,
    df: pd.DataFrame,
    cfg: CtxCfg,
    baselines: Sequence[str],
    anchor: str,
    *,
    device: torch.device,
    fixed_K: Optional[int] = None,
) -> Dict[str, float]:
    """Test-time eval: for each trial, hold out first fixed_K samples
    as context (deterministic per trial), predict on the rest. Returns
    aggregated metrics over query samples."""
    K = fixed_K if fixed_K is not None else cfg.context_size
    model.eval()
    all_l2: List[np.ndarray] = []
    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    n_skipped = 0
    with torch.no_grad():
        for trial_id, orig, bs, anchor_pred, targets in _trial_iter(df, cfg, anchor, baselines):
            T = len(orig)
            if T <= K:
                n_skipped += 1
                continue
            rng_local = np.random.default_rng(hash(trial_id) & 0xFFFFFFFF)
            idx = rng_local.permutation(T)
            # Cap K at the trained context size to avoid overflowing the padded buffer.
            effective_K = min(K, cfg.context_size, T - 1)
            ctx_idx, q_idx = idx[:effective_K], idx[effective_K:]
            # Pad context to cfg.context_size
            n_ctx = effective_K
            target_K = cfg.context_size
            ctx_orig = np.zeros((target_K, 2), dtype=np.float32)
            ctx_bs = np.zeros((target_K, len(baselines), 2), dtype=np.float32)
            ctx_anchor = np.zeros((target_K, 2), dtype=np.float32)
            ctx_resid = np.zeros((target_K, 2), dtype=np.float32)
            ctx_orig[:n_ctx] = orig[ctx_idx]
            ctx_bs[:n_ctx] = bs[ctx_idx]
            ctx_anchor[:n_ctx] = anchor_pred[ctx_idx]
            ctx_resid[:n_ctx] = targets[ctx_idx] - anchor_pred[ctx_idx]
            mask = np.zeros(target_K, dtype=np.bool_)
            mask[:n_ctx] = True
            qb = bs[q_idx]
            qo = orig[q_idx]
            qa = anchor_pred[q_idx]
            qt = targets[q_idx]

            ctx_orig_t = torch.from_numpy(ctx_orig).unsqueeze(0).to(device)
            ctx_bs_t = torch.from_numpy(ctx_bs).unsqueeze(0).to(device)
            ctx_anchor_t = torch.from_numpy(ctx_anchor).unsqueeze(0).to(device)
            ctx_resid_t = torch.from_numpy(ctx_resid).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

            ctx_items = build_ctx_item_features(ctx_orig_t, ctx_bs_t, ctx_anchor_t, ctx_resid_t, cfg)
            q_orig_t = torch.from_numpy(qo).to(device)
            q_bs_t = torch.from_numpy(qb).to(device)
            q_anchor_t = torch.from_numpy(qa).to(device)
            q_feat = build_ctx_features(q_orig_t, q_bs_t, q_anchor_t, cfg)

            # broadcast ctx to (Q, K, item_dim)
            n_q = q_feat.shape[0]
            ctx_items_b = ctx_items.expand(n_q, -1, -1)
            mask_b = mask_t.expand(n_q, -1)
            resid = model(ctx_items_b, mask_b, q_feat)
            pred = q_anchor_t + resid
            err = (pred - torch.from_numpy(qt).to(device)).cpu().numpy()
            all_x.append(err[:, 0]); all_y.append(err[:, 1])
            all_l2.append(np.sqrt((err ** 2).sum(axis=1)))
    if not all_l2:
        return {"l2_mean": float("nan"), "n_eval": 0, "n_skipped_trials": n_skipped}
    l2 = np.concatenate(all_l2); ex = np.concatenate(all_x); ey = np.concatenate(all_y)
    return {
        "l2_mean": float(l2.mean()), "l2_median": float(np.median(l2)),
        "l2_std": float(l2.std()), "l2_p95": float(np.percentile(l2, 95)),
        "mae_x": float(np.mean(np.abs(ex))), "mae_y": float(np.mean(np.abs(ey))),
        "rmse_x": float(np.sqrt((ex ** 2).mean())), "rmse_y": float(np.sqrt((ey ** 2).mean())),
        "n_eval": int(len(l2)), "n_skipped_trials": int(n_skipped),
    }


def train_context(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: CtxCfg,
    *,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    train_batcher = TrialBatcher(train_df, cfg, cfg.baselines, cfg.anchor, is_training=True)
    n_baselines = len(cfg.baselines)

    # Inspect dims
    sample_q = build_ctx_features(
        torch.zeros(1, 2), torch.zeros(1, n_baselines, 2), torch.zeros(1, 2), cfg
    )
    sample_c = build_ctx_item_features(
        torch.zeros(1, 1, 2), torch.zeros(1, 1, n_baselines, 2), torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), cfg
    )
    query_dim = sample_q.shape[-1]
    ctx_item_dim = sample_c.shape[-1]
    model = ContextRefiner(cfg, ctx_item_dim=ctx_item_dim, query_dim=query_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    iters_per_epoch = max(1, len(train_batcher) // cfg.batch_size_trials)

    for ep in range(1, cfg.epochs + 1):
        model.train()
        for _ in range(iters_per_epoch):
            ctx_list, mask_list, qo_list, qb_list, qa_list, qt_list = train_batcher.sample_batch(rng, cfg.batch_size_trials)
            # Build per-trial query batches and run
            losses = []
            for (ctx_orig_np, ctx_bs_np, ctx_anchor_np, ctx_resid_np), mask_np, qo, qb, qa, qt in zip(ctx_list, mask_list, qo_list, qb_list, qa_list, qt_list):
                ctx_orig = torch.from_numpy(ctx_orig_np).unsqueeze(0).to(device)
                ctx_bs = torch.from_numpy(ctx_bs_np).unsqueeze(0).to(device)
                ctx_anchor = torch.from_numpy(ctx_anchor_np).unsqueeze(0).to(device)
                ctx_resid = torch.from_numpy(ctx_resid_np).unsqueeze(0).to(device)
                mask = torch.from_numpy(mask_np).unsqueeze(0).to(device)
                ctx_items = build_ctx_item_features(ctx_orig, ctx_bs, ctx_anchor, ctx_resid, cfg)
                q_o = torch.from_numpy(qo).to(device)
                q_bs = torch.from_numpy(qb).to(device)
                q_a = torch.from_numpy(qa).to(device)
                q_t = torch.from_numpy(qt).to(device)
                q_feat = build_ctx_features(q_o, q_bs, q_a, cfg)
                n_q = q_feat.shape[0]
                resid = model(ctx_items.expand(n_q, -1, -1), mask.expand(n_q, -1), q_feat)
                pred = q_a + resid
                losses.append(((pred - q_t) ** 2).mean())
            loss = torch.stack(losses).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        v = evaluate_context(model, val_df, cfg, cfg.baselines, cfg.anchor, device=device)
        if v["l2_mean"] < best_val:
            best_val = v["l2_mean"]; best_epoch = ep
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if verbose and ep % 20 == 0:
            print(f"  ep {ep:3d}: val_L2={v['l2_mean']:.2f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    out = {"best_val_l2": best_val, "best_epoch": best_epoch}
    # Eval at multiple K values
    for k in [3, 5, 8, 12, 18]:
        out[f"test_K{k}"] = evaluate_context(model, test_df, cfg, cfg.baselines, cfg.anchor, device=device, fixed_K=k)
    return out
