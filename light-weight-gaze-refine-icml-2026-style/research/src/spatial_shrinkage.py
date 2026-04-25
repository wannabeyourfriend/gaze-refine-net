"""Spatial shrinkage for per-trial calibration refinement.

The classical anchor calibrator g_a returns predictions $\hat a_j$ at
each calibration sample. The per-trial bias is

    b_j = p_j^star - hat a_j ,

where $p_j^star$ is the ground-truth target. Given K context samples
collected after the standard calibration phase, we form the empirical
mean bias $\bar b_K$, and we apply an anchored correction at any new
query position p:

    hat p_refined(p) = a(p) + Lambda(p ; ctx) odot (p_q^star - a(p_q^star))

In the simplest case Lambda(p ; ctx) is a per-trial scalar matrix
diag(lambda_x, lambda_y) (the original v1 estimator). In v3 we let
Lambda depend on the query position p through Fourier features:

    Lambda(p; ctx) = sigmoid( head( h_ctx, fourier(p_norm) ) )

where h_ctx is a DeepSets summary of the context residuals and
fourier(.) is sin/cos at log-spaced frequencies.

This file provides:
  - SpatialShrinkageConfig: hyperparameters
  - SpatialShrinkageNet: the model
  - train_spatial_shrinkage: LOSO training loop on a DataFrame of trials
  - evaluate_spatial_shrinkage: deterministic per-trial K-context eval
  - bayes_optimal_scalar: closed-form optimum for the homoscedastic
    Gaussian-bias model (used by the synthetic benchmark)
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
# Closed-form Bayes-optimal shrinkage (used by the theory section)
# ---------------------------------------------------------------------------

def bayes_optimal_scalar(
    sigma2_bias: float,
    sigma2_noise: float,
    K: int,
) -> float:
    """In the model
        b ~ N(0, sigma2_bias I_2),
        eps_j ~ N(0, sigma2_noise I_2),
        observe r_j = b + eps_j  for j = 1..K,
        empirical mean b_bar = (1/K) sum r_j,
    the Bayes-optimal scalar lambda that minimizes
        E ||lambda * b_bar - b||^2
    is given by
        lambda* = sigma2_bias / (sigma2_bias + sigma2_noise / K).
    Returns lambda* in [0, 1].
    """
    if K <= 0:
        return 0.0
    return float(sigma2_bias / (sigma2_bias + sigma2_noise / max(K, 1)))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SpatialShrinkageConfig:
    # Spatial encoding
    screen_w: float = 1920.0
    screen_h: float = 1080.0
    use_fourier: bool = True
    fourier_bands: int = 6

    # Context encoder (DeepSets over residual samples)
    ctx_hidden: int = 64
    ctx_layers: int = 2
    ctx_summary_dim: int = 32

    # Shrinkage head
    head_hidden: Sequence[int] = (64, 32)
    dropout: float = 0.0
    use_layernorm: bool = True

    # Output structure
    spatial: bool = True       # if False, output a per-trial scalar lambda (= v1)
    lambda_min: float = 0.0
    lambda_max: float = 1.0

    # Training
    epochs: int = 80
    lr: float = 3e-3
    weight_decay: float = 1e-3
    batch_size_trials: int = 16
    queries_per_trial: int = 8
    seed: int = 1047
    K_train_min: int = 1
    K_train_max: int = 18  # randomize K in this range to make the model K-agnostic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fourier(x: torch.Tensor, n: int) -> torch.Tensor:
    """Sin/cos at log-spaced frequencies. x in roughly [-1, 1]."""
    if n <= 0:
        return x
    bands = 2.0 ** torch.arange(n, dtype=x.dtype, device=x.device)
    xb = x.unsqueeze(-1) * bands
    feats = torch.cat([torch.sin(math.pi * xb), torch.cos(math.pi * xb)], dim=-1)
    return torch.cat([x, feats.flatten(start_dim=-2)], dim=-1)


def screen_norm(xy: torch.Tensor, w: float, h: float) -> torch.Tensor:
    out = xy.clone()
    out[..., 0] = (out[..., 0] / w) * 2 - 1
    out[..., 1] = (out[..., 1] / h) * 2 - 1
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ContextEncoder(nn.Module):
    """DeepSets encoder over context items.

    Each item is (anchor_pos_norm, residual_normalized, K_features) ->
    a vector pooled by mean to produce h_ctx in R^{summary_dim}.
    """

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
        # x: (B, K, item_dim), mask: (B, K) bool, True = valid
        h = self.proj(self.body(x))  # (B, K, S)
        m = mask.unsqueeze(-1).float()
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


class SpatialShrinkageHead(nn.Module):
    """Outputs Lambda(p) in (lambda_min, lambda_max)^2 from
    [h_ctx, fourier(p_norm), K_features]."""

    def __init__(
        self,
        ctx_dim: int,
        pos_dim: int,
        hidden: Sequence[int],
        dropout: float,
        use_layernorm: bool,
        lambda_min: float,
        lambda_max: float,
    ) -> None:
        super().__init__()
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        layers: List[nn.Module] = []
        in_dim = ctx_dim + pos_dim
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h)]
            if use_layernorm:
                layers += [nn.LayerNorm(h)]
            layers += [nn.GELU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 2)
        # zero init -> sigmoid(0) = 0.5 (mid-shrinkage)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, ctx: torch.Tensor, pos_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([ctx, pos_feat], dim=-1)
        logits = self.head(self.body(x))
        return self.lambda_min + (self.lambda_max - self.lambda_min) * torch.sigmoid(logits)


class SpatialShrinkageNet(nn.Module):
    """End-to-end model:
       inputs:  context (orig, anchor, target) tuples of size K, query position
       output:  Lambda in [0,1]^2 to be applied to bar_b.
    """

    def __init__(self, cfg: SpatialShrinkageConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Each context item: (anchor_xy_norm + (1 + 2*Fb)*2 fourier, residual/50 (2 dim))
        # Plus 2 features per item: 1/K and log K (constant within a trial but useful to give the encoder)
        anchor_n = 2
        if cfg.use_fourier:
            anchor_n = 2 * (1 + 2 * cfg.fourier_bands)
        residual_n = 2
        K_feat = 2
        item_dim = anchor_n + residual_n + K_feat
        self.encoder = ContextEncoder(item_dim, cfg.ctx_hidden, cfg.ctx_layers, cfg.ctx_summary_dim)

        # query position features
        pos_n = anchor_n
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
        anchor: torch.Tensor,    # (B, K, 2) anchor predictions
        residual: torch.Tensor,  # (B, K, 2) target - anchor
        K_actual: torch.Tensor,  # (B,) ints
    ) -> torch.Tensor:
        anchor_feat = self._featurize_position(anchor)  # (B, K, anchor_n)
        res_feat = residual / 50.0
        # broadcast K-features
        invK = (1.0 / K_actual.clamp(min=1).float()).view(-1, 1, 1).expand(-1, anchor.shape[1], 1)
        logK = torch.log(K_actual.clamp(min=1).float()).view(-1, 1, 1).expand(-1, anchor.shape[1], 1)
        return torch.cat([anchor_feat, res_feat, invK, logK], dim=-1)

    def forward(
        self,
        ctx_anchor: torch.Tensor,    # (B, K, 2)
        ctx_residual: torch.Tensor,  # (B, K, 2)
        ctx_mask: torch.Tensor,      # (B, K) bool
        query_pos: torch.Tensor,     # (B, Q, 2)
        K_actual: torch.Tensor,      # (B,)
    ) -> torch.Tensor:
        ctx_items = self._featurize_context(ctx_anchor, ctx_residual, K_actual)
        h = self.encoder(ctx_items, ctx_mask)            # (B, S)
        # Per-query lambda
        if self.cfg.spatial:
            q_feat = self._featurize_position(query_pos)  # (B, Q, pos_dim)
            h_b = h.unsqueeze(1).expand(-1, q_feat.shape[1], -1)
            return self.head(h_b, q_feat)                # (B, Q, 2)
        # Non-spatial: lambda is per-trial; broadcast across queries
        zero_pos = torch.zeros_like(self._featurize_position(query_pos[:, :1]))[:, 0]  # (B, pos_dim)
        lam_per_trial = self.head(h, zero_pos)           # (B, 2)
        return lam_per_trial.unsqueeze(1).expand(-1, query_pos.shape[1], -1)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def _trial_arrays(df: pd.DataFrame, anchor_col: str) -> List[Dict[str, np.ndarray]]:
    """Group a DataFrame into per-trial dicts of arrays."""
    if "origin_gaze_x" in df.columns:
        ox, oy = "origin_gaze_x", "origin_gaze_y"
    else:
        ox, oy = "original_gaze_x", "original_gaze_y"
    df = df.copy()
    df["trial_id"] = df["subject"] + "|" + df["timestamp"]
    out: List[Dict[str, np.ndarray]] = []
    for trial_id, g in df.groupby("trial_id"):
        if len(g) < 4:
            continue
        anchor = g[[f"{anchor_col}_x", f"{anchor_col}_y"]].to_numpy(dtype=np.float32)
        target = g[["target_x", "target_y"]].to_numpy(dtype=np.float32)
        orig = g[[ox, oy]].to_numpy(dtype=np.float32)
        out.append({
            "trial_id": trial_id,
            "subject": g["subject"].iloc[0],
            "anchor": anchor,
            "target": target,
            "orig": orig,
        })
    return out


def _sample_episode(
    trial: Dict[str, np.ndarray],
    K: int,
    Q: int,
    rng: np.random.Generator,
    K_max: int,
):
    T = len(trial["anchor"])
    if T <= K:
        return None
    idx = rng.permutation(T)
    ctx_idx, q_idx = idx[:K], idx[K:K + Q]
    if len(q_idx) == 0:
        q_idx = idx[K:K + 1]
    # pad context to K_max
    n = K
    a = np.zeros((K_max, 2), dtype=np.float32)
    r = np.zeros((K_max, 2), dtype=np.float32)
    a[:n] = trial["anchor"][ctx_idx]
    r[:n] = trial["target"][ctx_idx] - trial["anchor"][ctx_idx]
    mask = np.zeros(K_max, dtype=np.bool_)
    mask[:n] = True
    return {
        "ctx_anchor": a, "ctx_residual": r, "ctx_mask": mask, "K_actual": np.int64(n),
        "q_pos":   trial["anchor"][q_idx],
        "q_target": trial["target"][q_idx],
    }


def train_spatial_shrinkage(
    train_trials: List[Dict[str, np.ndarray]],
    val_trials: List[Dict[str, np.ndarray]],
    cfg: SpatialShrinkageConfig,
    *,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Tuple[SpatialShrinkageNet, Dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model = SpatialShrinkageNet(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

    K_max = cfg.K_train_max
    best_val = float("inf")
    best_state = None
    best_epoch = 0

    for ep in range(1, cfg.epochs + 1):
        model.train()
        # Form a batch: random sample of trials, random K per trial in [Kmin..Kmax]
        rng.shuffle(train_trials)
        for start in range(0, len(train_trials), cfg.batch_size_trials):
            batch_trials = train_trials[start:start + cfg.batch_size_trials]
            episodes = []
            for tr in batch_trials:
                K = int(rng.integers(cfg.K_train_min, cfg.K_train_max + 1))
                ep_data = _sample_episode(tr, K, cfg.queries_per_trial, rng, K_max)
                if ep_data is not None:
                    episodes.append(ep_data)
            if not episodes:
                continue
            # We let queries-per-trial vary; pad to max
            max_q = max(e["q_pos"].shape[0] for e in episodes)
            B = len(episodes)
            ctx_anchor = torch.from_numpy(np.stack([e["ctx_anchor"] for e in episodes])).to(device)
            ctx_residual = torch.from_numpy(np.stack([e["ctx_residual"] for e in episodes])).to(device)
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
            # Empirical bias bar_b from context (B, 2)
            res = ctx_residual * ctx_mask.unsqueeze(-1).float()
            bar_b = res.sum(dim=1) / ctx_mask.sum(dim=1).clamp(min=1).unsqueeze(-1).float()
            lam = model(ctx_anchor, ctx_residual, ctx_mask, q_pos, K_actual)  # (B, Q, 2)
            corr = lam * bar_b.unsqueeze(1)  # (B, Q, 2)
            # Predicted residual at query position is lam * bar_b; loss vs true residual
            true_res = q_tgt - q_pos
            err = (corr - true_res) * q_mask.unsqueeze(-1).float()
            loss = (err.pow(2).sum(dim=-1) * q_mask.float()).sum() / q_mask.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # Validate (single deterministic episode at K = floor(K_max/2) per trial)
        with torch.no_grad():
            model.eval()
            v_loss = 0.0; n = 0
            K_eval = max(1, K_max // 2)
            for tr in val_trials:
                ep_data = _sample_episode(
                    tr, K_eval, len(tr["anchor"]),
                    np.random.default_rng(hash(tr["trial_id"]) & 0xFFFFFFFF),
                    K_max,
                )
                if ep_data is None:
                    continue
                ctx_anchor = torch.from_numpy(ep_data["ctx_anchor"]).unsqueeze(0).to(device)
                ctx_residual = torch.from_numpy(ep_data["ctx_residual"]).unsqueeze(0).to(device)
                ctx_mask = torch.from_numpy(ep_data["ctx_mask"]).unsqueeze(0).to(device)
                K_actual = torch.tensor([ep_data["K_actual"]]).to(device)
                q_pos = torch.from_numpy(ep_data["q_pos"]).unsqueeze(0).to(device)
                q_tgt = torch.from_numpy(ep_data["q_target"]).unsqueeze(0).to(device)
                bar_b = (ctx_residual * ctx_mask.unsqueeze(-1).float()).sum(dim=1) / ctx_mask.sum(dim=1).clamp(min=1).unsqueeze(-1).float()
                lam = model(ctx_anchor, ctx_residual, ctx_mask, q_pos, K_actual)
                corr = lam * bar_b.unsqueeze(1)
                true_res = q_tgt - q_pos
                err = (corr - true_res).pow(2).sum(dim=-1).sqrt().mean()
                v_loss += float(err) * q_pos.shape[1]
                n += q_pos.shape[1]
            val = v_loss / max(n, 1)
        if val < best_val:
            best_val = val
            best_epoch = ep
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if verbose and (ep % 10 == 0 or ep == cfg.epochs):
            print(f"  ep {ep:3d}  val_L2 = {val:.2f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_l2": best_val, "best_epoch": best_epoch}


@torch.no_grad()
def evaluate_spatial_shrinkage(
    model: SpatialShrinkageNet,
    test_trials: List[Dict[str, np.ndarray]],
    K: int,
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Deterministic per-trial K-context evaluation."""
    device = device or next(model.parameters()).device
    model.eval()
    cfg = model.cfg
    K_max = cfg.K_train_max
    all_l2: List[np.ndarray] = []
    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    n_skipped = 0
    for tr in test_trials:
        T = len(tr["anchor"])
        if T <= K:
            n_skipped += 1
            continue
        rng = np.random.default_rng(hash(tr["trial_id"]) & 0xFFFFFFFF)
        idx = rng.permutation(T)
        K_use = min(K, K_max)
        ctx_idx, q_idx = idx[:K_use], idx[K_use:]
        # pad
        a = np.zeros((K_max, 2), dtype=np.float32)
        r = np.zeros((K_max, 2), dtype=np.float32)
        a[:K_use] = tr["anchor"][ctx_idx]
        r[:K_use] = tr["target"][ctx_idx] - tr["anchor"][ctx_idx]
        mask = np.zeros(K_max, dtype=np.bool_)
        mask[:K_use] = True
        ctx_anchor = torch.from_numpy(a).unsqueeze(0).to(device)
        ctx_residual = torch.from_numpy(r).unsqueeze(0).to(device)
        ctx_mask = torch.from_numpy(mask).unsqueeze(0).to(device)
        K_actual = torch.tensor([K_use]).to(device)
        q_pos = torch.from_numpy(tr["anchor"][q_idx]).unsqueeze(0).to(device)
        q_tgt = torch.from_numpy(tr["target"][q_idx]).unsqueeze(0).to(device)
        bar_b = (ctx_residual * ctx_mask.unsqueeze(-1).float()).sum(dim=1) / ctx_mask.sum(dim=1).clamp(min=1).unsqueeze(-1).float()
        lam = model(ctx_anchor, ctx_residual, ctx_mask, q_pos, K_actual)
        pred = q_pos + lam * bar_b.unsqueeze(1)
        err = (pred - q_tgt).cpu().numpy()[0]
        all_x.append(err[:, 0]); all_y.append(err[:, 1])
        all_l2.append(np.linalg.norm(err, axis=1))
    if not all_l2:
        return {"l2_mean": float("nan"), "n_eval": 0, "n_skipped_trials": n_skipped}
    l2 = np.concatenate(all_l2)
    ex = np.concatenate(all_x); ey = np.concatenate(all_y)
    return {
        "l2_mean": float(l2.mean()),
        "l2_median": float(np.median(l2)),
        "l2_std": float(l2.std()),
        "l2_p90": float(np.percentile(l2, 90)),
        "l2_p95": float(np.percentile(l2, 95)),
        "mae_x": float(np.mean(np.abs(ex))),
        "mae_y": float(np.mean(np.abs(ey))),
        "rmse_x": float(np.sqrt((ex ** 2).mean())),
        "rmse_y": float(np.sqrt((ey ** 2).mean())),
        "n_eval": int(len(l2)),
        "n_skipped_trials": int(n_skipped),
    }


# Convenience: load combined CSV and group into trials
def load_trials_from_csv(csv_path: str, anchor_col: str) -> List[Dict[str, np.ndarray]]:
    return _trial_arrays(pd.read_csv(csv_path), anchor_col)
