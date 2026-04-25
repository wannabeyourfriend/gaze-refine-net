"""Comparison baselines for the NeurIPS resubmission.

We implement and evaluate several methods uniformly:

  1. classical_anchor: pred_sim_rbf_multiquadric_s2.0 (best classical)
  2. raw_mean_bias_K: subtract empirical mean residual from K context
  3. fixed_lambda_K: shrunk version of (2) with global lambda
  4. multi_baseline_residual: an MLP that maps (orig, [baselines])
     to a 2-D residual added to the anchor. This is the
     residual-learning style of the previous literature.
  5. anchored_offset_v3: a variant that predicts the residual on top
     of the anchor only, conditioned on per-trial bias features.
  6. learned_shrinkage_scalar (ours v1): learned scalar lambda from
     per-trial statistics.
  7. learned_shrinkage_spatial (ours v3): position-dependent lambda(p).

All models are trained leave-one-(subject|trial-group)-out and
evaluated with the same protocol so the table is apples-to-apples.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Multi-baseline residual learning (the comparison baseline)
# ---------------------------------------------------------------------------

@dataclass
class MBRConfig:
    """Multi-baseline residual learning."""
    baseline_cols: Sequence[str]
    hidden_dims: Sequence[int] = (128, 64)
    dropout: float = 0.1
    use_layernorm: bool = True
    coordinate_scale: float = 100.0
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 64
    seed: int = 1047
    noise_sigma_max: float = 10.0


class MBRNet(nn.Module):
    """Predict the residual on top of a chosen anchor baseline,
    given the orig gaze and a stack of K classical baseline predictions
    as raw coordinates (NOT as residuals against the target -- those
    inputs would leak the label)."""

    def __init__(self, in_dim: int, hidden_dims: Sequence[int], dropout: float, use_layernorm: bool) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h)]
            if use_layernorm:
                layers += [nn.LayerNorm(h)]
            layers += [nn.GELU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        self.body = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(prev, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def build_mbr_features(
    df: pd.DataFrame,
    baseline_cols: Sequence[str],
    coord_scale: float,
    bs_mean: Optional[np.ndarray] = None,
    bs_std: Optional[np.ndarray] = None,
):
    """Returns input array (N, in_dim), and bs_mean, bs_std.

    Features: [orig_x/coord_scale, orig_y/coord_scale,
               (b1_x - mean)/std, (b1_y - mean)/std, ..., bK_y_norm]
    bs_mean/bs_std default to TRAIN statistics; pass them when
    featurising val/test.
    """
    if "origin_gaze_x" in df.columns:
        ox, oy = "origin_gaze_x", "origin_gaze_y"
    else:
        ox, oy = "original_gaze_x", "original_gaze_y"
    orig = df[[ox, oy]].to_numpy(dtype=np.float32) / coord_scale
    bs = np.concatenate(
        [df[[f"{c}_x", f"{c}_y"]].to_numpy(dtype=np.float32) for c in baseline_cols],
        axis=1,
    )
    if bs_mean is None:
        bs_mean = bs.mean(axis=0)
        bs_std = bs.std(axis=0) + 1e-6
    bs_norm = (bs - bs_mean) / bs_std
    inp = np.concatenate([orig, bs_norm], axis=1).astype(np.float32)
    return inp, bs_mean, bs_std


def train_mbr(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: MBRConfig,
    anchor_col: str,
    *,
    device: Optional[torch.device] = None,
    verbose: bool = False,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    X_train, mean, std = build_mbr_features(train_df, cfg.baseline_cols, cfg.coordinate_scale)
    X_val, _, _ = build_mbr_features(val_df, cfg.baseline_cols, cfg.coordinate_scale, mean, std)

    # Target = (target - anchor) / coord_scale, network outputs residual on top of anchor
    y_train_anchor = train_df[[f"{anchor_col}_x", f"{anchor_col}_y"]].to_numpy(dtype=np.float32)
    y_train_target = train_df[["target_x", "target_y"]].to_numpy(dtype=np.float32)
    res_train = (y_train_target - y_train_anchor) / cfg.coordinate_scale
    y_val_anchor = val_df[[f"{anchor_col}_x", f"{anchor_col}_y"]].to_numpy(dtype=np.float32)
    y_val_target = val_df[["target_x", "target_y"]].to_numpy(dtype=np.float32)
    res_val = (y_val_target - y_val_anchor) / cfg.coordinate_scale

    X_train_t = torch.from_numpy(X_train).to(device)
    res_train_t = torch.from_numpy(res_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    res_val_t = torch.from_numpy(res_val).to(device)

    model = MBRNet(X_train.shape[1], cfg.hidden_dims, cfg.dropout, cfg.use_layernorm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

    n = X_train.shape[0]
    best_val = float("inf"); best_state = None; best_epoch = 0
    rng = np.random.default_rng(cfg.seed)
    for ep in range(1, cfg.epochs + 1):
        model.train()
        idx = rng.permutation(n)
        for s in range(0, n, cfg.batch_size):
            b = idx[s:s + cfg.batch_size]
            xb = X_train_t[b]
            yb = res_train_t[b]
            if cfg.noise_sigma_max > 0:
                noise = (torch.rand(xb.shape[0], device=device) * cfg.noise_sigma_max).unsqueeze(1) * torch.randn_like(xb)
                xb = xb + noise * 0.01
            pred = model(xb)
            loss = ((pred - yb) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v_err = ((model(X_val_t) - res_val_t) * cfg.coordinate_scale).pow(2).sum(-1).sqrt().mean().item()
        if v_err < best_val:
            best_val = v_err; best_epoch = ep
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if verbose and (ep % 25 == 0 or ep == cfg.epochs):
            print(f"  ep {ep:3d}  val_L2 = {v_err:.2f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_l2": best_val, "best_epoch": best_epoch, "bs_mean": mean, "bs_std": std}


@torch.no_grad()
def evaluate_mbr(
    model: MBRNet,
    df: pd.DataFrame,
    cfg: MBRConfig,
    anchor_col: str,
    bs_mean: np.ndarray,
    bs_std: np.ndarray,
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    device = device or next(model.parameters()).device
    X, _, _ = build_mbr_features(df, cfg.baseline_cols, cfg.coordinate_scale, bs_mean, bs_std)
    anchor = df[[f"{anchor_col}_x", f"{anchor_col}_y"]].to_numpy(dtype=np.float32)
    target = df[["target_x", "target_y"]].to_numpy(dtype=np.float32)
    pred_res = model(torch.from_numpy(X).to(device)).cpu().numpy() * cfg.coordinate_scale
    pred = anchor + pred_res
    err = pred - target
    l2 = np.linalg.norm(err, axis=1)
    return {
        "l2_mean": float(l2.mean()), "l2_median": float(np.median(l2)),
        "l2_std": float(l2.std()), "l2_p90": float(np.percentile(l2, 90)),
        "l2_p95": float(np.percentile(l2, 95)),
        "mae_x": float(np.mean(np.abs(err[:,0]))), "mae_y": float(np.mean(np.abs(err[:,1]))),
        "rmse_x": float(np.sqrt((err[:,0]**2).mean())), "rmse_y": float(np.sqrt((err[:,1]**2).mean())),
        "n": int(len(l2)),
    }


# ---------------------------------------------------------------------------
# Meta-learning baseline: MAML-style few-shot adaptation
# ---------------------------------------------------------------------------

@dataclass
class MAMLConfig:
    hidden_dims: Sequence[int] = (64, 32)
    inner_lr: float = 1e-2
    meta_lr: float = 1e-3
    inner_steps: int = 5
    coord_scale: float = 100.0
    epochs: int = 80
    batch_tasks: int = 16
    queries_per_task: int = 8
    seed: int = 1047


class MAMLNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def maml_inner_step(
    model: MAMLNet,
    x: torch.Tensor,
    y: torch.Tensor,
    inner_lr: float,
    create_graph: bool,
) -> List[torch.Tensor]:
    """Returns adapted parameters (list of tensors)."""
    pred = model(x)
    loss = ((pred - y) ** 2).mean()
    grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=create_graph)
    adapted = [p - inner_lr * g for p, g in zip(model.parameters(), grads)]
    return adapted


def functional_forward(model: MAMLNet, params: List[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Manual forward using `params` instead of model.parameters()."""
    h = x
    p_iter = iter(params)
    for layer in model.body:
        if isinstance(layer, nn.Linear):
            w = next(p_iter); b = next(p_iter)
            h = torch.nn.functional.linear(h, w, b)
        elif isinstance(layer, nn.GELU):
            h = torch.nn.functional.gelu(h)
    w = next(p_iter); b = next(p_iter)
    return torch.nn.functional.linear(h, w, b)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fixed_lambda_eval_df(df: pd.DataFrame, anchor_col: str, lam: float) -> dict:
    """No context: just apply (1-lam) shrinkage trivially. Reduces to anchor."""
    a = df[[f"{anchor_col}_x", f"{anchor_col}_y"]].to_numpy(dtype=np.float32)
    t = df[["target_x", "target_y"]].to_numpy(dtype=np.float32)
    pred = a  # no context
    err = pred - t
    l2 = np.linalg.norm(err, axis=1)
    return {"l2_mean": float(l2.mean()), "l2_median": float(np.median(l2))}


def per_trial_fixed_lambda_eval(
    df_with_trial: pd.DataFrame,
    anchor_col: str,
    K: int,
    lam: float,
) -> dict:
    errs = []
    for trial_id, g in df_with_trial.groupby("trial_id"):
        if len(g) <= K: continue
        rng = np.random.default_rng(hash(str(trial_id)) & 0xFFFFFFFF)
        idx = rng.permutation(len(g))
        ctx_idx, q_idx = idx[:K], idx[K:]
        ctx, q = g.iloc[ctx_idx], g.iloc[q_idx]
        bias = (ctx[["target_x", "target_y"]].values - ctx[[f"{anchor_col}_x", f"{anchor_col}_y"]].values).mean(axis=0)
        pred = q[[f"{anchor_col}_x", f"{anchor_col}_y"]].values + lam * bias
        l2 = np.linalg.norm(pred - q[["target_x", "target_y"]].values, axis=1)
        errs.append(l2)
    if not errs:
        return {"l2_mean": float("nan")}
    arr = np.concatenate(errs)
    return {"l2_mean": float(arr.mean()), "l2_median": float(np.median(arr)),
            "l2_std": float(arr.std()), "l2_p95": float(np.percentile(arr, 95)),
            "n": int(len(arr))}
