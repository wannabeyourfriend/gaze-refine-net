"""Improved refinement architecture (v2) for the NeurIPS resubmission.

Key changes from the paper-faithful pipeline:

  1. The network predicts the residual of the BEST classical baseline,
     not the absolute orig→target offset. The strongest baseline
     (similarity for JuDo) already explains most of the variance —
     the network only needs to learn the small position-dependent error
     it leaves behind.

  2. The input includes the raw gaze coordinate normalized to a
     screen-relative range, so the network has access to *where* on the
     screen the correction is being applied. Multiple baseline
     predictions (also in coordinate space, scaled) are concatenated.

  3. Optional: weighted blend over baselines as a learned softmax,
     rather than a free MLP. This is a structured contribution: the
     network learns reliability weights w_k(p) over a curated set of
     classical calibrators, plus a small additive residual.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_BASELINES: Tuple[str, ...] = (
    "pred_similarity",
    "pred_poly",
    "pred_rbf_multiquadric_s2.0",
    "pred_sim_rbf_multiquadric_s2.0",
    "pred_sim_pwa",
)


@dataclass
class V2Config:
    train_csv: str
    val_csv: str
    test_csv: str

    baselines: Tuple[str, ...] = DEFAULT_BASELINES
    anchor_baseline: str = "pred_similarity"  # network predicts residual on top of this

    # Architecture
    arch: str = "mlp_residual"  # "mlp_residual" | "softmax_blend"
    hidden_dims: Tuple[int, ...] = (64, 32)
    dropout: float = 0.0
    use_layernorm: bool = True

    # Spatial encoding
    screen_w: float = 1280.0  # for JuDo
    screen_h: float = 1024.0
    use_fourier: bool = True
    fourier_bands: int = 4   # number of sinusoid frequencies per axis

    # Noise injection on baseline coordinates during training
    noise_sigma_max: float = 8.0
    noise_prob: float = 0.5

    # Training
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 64
    seed: int = 1047

    # Regularization on residual magnitude (anchor the network near zero)
    residual_l2: float = 0.0

    def __post_init__(self) -> None:
        self.baselines = tuple(self.baselines)
        self.hidden_dims = tuple(self.hidden_dims)
        if self.anchor_baseline not in self.baselines:
            object.__setattr__(self, "baselines", (self.anchor_baseline, *self.baselines))


def fourier_features(x: torch.Tensor, n_bands: int) -> torch.Tensor:
    """Apply sin/cos at log-spaced frequencies. Input x in roughly [-1, 1]."""
    if n_bands <= 0:
        return x
    bands = 2.0 ** torch.arange(n_bands, dtype=x.dtype, device=x.device)
    xb = x.unsqueeze(-1) * bands  # (..., D, B)
    feats = torch.cat([torch.sin(math.pi * xb), torch.cos(math.pi * xb)], dim=-1)
    return torch.cat([x, feats.flatten(start_dim=-2)], dim=-1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class V2Dataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: V2Config,
        is_training: bool,
    ) -> None:
        self.cfg = cfg
        self.is_training = is_training
        # Support both column conventions: origin_gaze_* (JuDo, all_trials)
        # and original_gaze_* (s1).
        if "origin_gaze_x" in df.columns:
            ox, oy = "origin_gaze_x", "origin_gaze_y"
        else:
            ox, oy = "original_gaze_x", "original_gaze_y"
        orig = df[[ox, oy]].to_numpy(dtype=np.float32)
        targets = df[["target_x", "target_y"]].to_numpy(dtype=np.float32)
        baseline_arrays = []
        for b in cfg.baselines:
            baseline_arrays.append(df[[f"{b}_x", f"{b}_y"]].to_numpy(dtype=np.float32))
        # (N, K, 2)
        baselines = np.stack(baseline_arrays, axis=1)
        anchor_idx = list(cfg.baselines).index(cfg.anchor_baseline)
        anchor_pred = baselines[:, anchor_idx, :]

        self.orig = torch.from_numpy(orig)
        self.targets = torch.from_numpy(targets)
        self.baselines = torch.from_numpy(baselines)
        self.anchor_pred = torch.from_numpy(anchor_pred)
        self.anchor_idx = anchor_idx
        self.K = len(cfg.baselines)

    def __len__(self) -> int:
        return self.orig.shape[0]

    def _maybe_noise(self, baselines: torch.Tensor) -> torch.Tensor:
        if not self.is_training or self.cfg.noise_sigma_max <= 0:
            return baselines
        if torch.rand(1).item() > self.cfg.noise_prob:
            return baselines
        sigmas = torch.rand(baselines.shape) * self.cfg.noise_sigma_max
        return baselines + torch.randn_like(baselines) * sigmas

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "orig": self.orig[idx],
            "baselines": self._maybe_noise(self.baselines[idx]),
            "anchor": self.anchor_pred[idx],
            "target": self.targets[idx],
        }


def to_screen_norm(xy: torch.Tensor, w: float, h: float) -> torch.Tensor:
    """Normalize pixel coords to [-1, 1]."""
    out = xy.clone()
    out[..., 0] = (out[..., 0] / w) * 2.0 - 1.0
    out[..., 1] = (out[..., 1] / h) * 2.0 - 1.0
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLPResidualHead(nn.Module):
    """Predict a residual (dx, dy) on top of the anchor baseline, from
    (raw_gaze, all_baseline_predictions) all in screen-normalized coords."""

    def __init__(self, in_dim: int, hidden_dims: Sequence[int], dropout: float, use_layernorm: bool) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 2)
        # Initialize head near-zero so we start near the anchor baseline.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


class SoftmaxBlendHead(nn.Module):
    """Predict logits over baselines, then blend their predictions.
    Produces a refined gaze coordinate, plus a small additive residual."""

    def __init__(self, in_dim: int, n_baselines: int, hidden_dims: Sequence[int], dropout: float, use_layernorm: bool) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.body = nn.Sequential(*layers)
        self.weight_head = nn.Linear(prev, n_baselines)
        self.residual_head = nn.Linear(prev, 2)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        # Bias the weights so anchor (idx 0) starts dominant.
        with torch.no_grad():
            nn.init.zeros_(self.weight_head.weight)
            self.weight_head.bias.fill_(0.0)
            self.weight_head.bias[0] = 4.0

    def forward(self, feats: torch.Tensor, baselines: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        # feats: (B, F), baselines: (B, K, 2) in pixels
        h = self.body(feats)
        w = torch.softmax(self.weight_head(h), dim=-1)  # (B, K)
        blended = (w.unsqueeze(-1) * baselines).sum(dim=1)  # (B, 2) in pixels
        residual = self.residual_head(h)  # in pixels
        return blended + residual, w


class GazeRefineV2(nn.Module):
    def __init__(self, cfg: V2Config) -> None:
        super().__init__()
        self.cfg = cfg
        # Compute input dim
        screen = self._screen_dim(2)  # 2 raw coords
        bs_dim = self._screen_dim(2 * len(cfg.baselines))  # K baselines x 2
        in_dim = screen + bs_dim
        self.in_dim = in_dim
        if cfg.arch == "softmax_blend":
            self.head = SoftmaxBlendHead(in_dim, len(cfg.baselines), cfg.hidden_dims, cfg.dropout, cfg.use_layernorm)
        else:
            self.head = MLPResidualHead(in_dim, cfg.hidden_dims, cfg.dropout, cfg.use_layernorm)

    def _screen_dim(self, base: int) -> int:
        if self.cfg.use_fourier:
            return base * (1 + 2 * self.cfg.fourier_bands)
        return base

    def _featurize(self, orig: torch.Tensor, baselines: torch.Tensor) -> torch.Tensor:
        # orig (B, 2), baselines (B, K, 2) in pixel coords
        cfg = self.cfg
        orig_n = to_screen_norm(orig, cfg.screen_w, cfg.screen_h)
        bs_n = to_screen_norm(baselines, cfg.screen_w, cfg.screen_h).flatten(start_dim=1)
        if cfg.use_fourier:
            orig_n = fourier_features(orig_n, cfg.fourier_bands)
            bs_n = fourier_features(bs_n, cfg.fourier_bands)
        return torch.cat([orig_n, bs_n], dim=-1)

    def forward(
        self,
        orig: torch.Tensor,
        baselines: torch.Tensor,
        anchor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        feats = self._featurize(orig, baselines)
        if self.cfg.arch == "softmax_blend":
            pred_pixels, w = self.head(feats, baselines, anchor)
            return pred_pixels, w
        residual = self.head(feats)  # in pixels (since head outputs unnormalized)
        return anchor + residual, None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def evaluate_v2(model: GazeRefineV2, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    errs_x: List[np.ndarray] = []
    errs_y: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            orig = batch["orig"].to(device)
            baselines = batch["baselines"].to(device)
            anchor = batch["anchor"].to(device)
            target = batch["target"].to(device)
            pred, _ = model(orig, baselines, anchor)
            err = (pred - target).cpu().numpy()
            errs_x.append(err[:, 0])
            errs_y.append(err[:, 1])
    ex = np.concatenate(errs_x)
    ey = np.concatenate(errs_y)
    l2 = np.sqrt(ex ** 2 + ey ** 2)
    return {
        "mae_x": float(np.mean(np.abs(ex))),
        "mae_y": float(np.mean(np.abs(ey))),
        "rmse_x": float(np.sqrt(np.mean(ex ** 2))),
        "rmse_y": float(np.sqrt(np.mean(ey ** 2))),
        "l2_mean": float(np.mean(l2)),
        "l2_median": float(np.median(l2)),
        "l2_std": float(np.std(l2)),
        "l2_p90": float(np.percentile(l2, 90)),
        "l2_p95": float(np.percentile(l2, 95)),
        "n": int(len(l2)),
    }


def train_v2(cfg: V2Config, *, verbose: bool = False, device: Optional[torch.device] = None) -> Dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_df = pd.read_csv(cfg.train_csv)
    val_df = pd.read_csv(cfg.val_csv)
    test_df = pd.read_csv(cfg.test_csv)

    train_ds = V2Dataset(train_df, cfg, is_training=True)
    val_ds = V2Dataset(val_df, cfg, is_training=False)
    test_ds = V2Dataset(test_df, cfg, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = GazeRefineV2(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

    best_val = math.inf
    best_state = None
    best_epoch = 0
    curve = []
    for ep in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for batch in train_loader:
            orig = batch["orig"].to(device)
            baselines = batch["baselines"].to(device)
            anchor = batch["anchor"].to(device)
            target = batch["target"].to(device)
            opt.zero_grad(set_to_none=True)
            pred, w = model(orig, baselines, anchor)
            loss = ((pred - target) ** 2).mean()
            if cfg.residual_l2 > 0:
                resid = pred - anchor
                loss = loss + cfg.residual_l2 * (resid ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * orig.size(0)
            n += orig.size(0)
        scheduler.step()
        train_loss = running / max(1, n)
        v = evaluate_v2(model, val_loader, device)
        curve.append({"epoch": ep, "train_loss": train_loss, "val_l2": v["l2_mean"]})
        if v["l2_mean"] < best_val:
            best_val = v["l2_mean"]
            best_epoch = ep
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if verbose and (ep % 25 == 0 or ep == cfg.epochs):
            print(f"  ep {ep:3d}: train_loss={train_loss:.4f}  val_l2={v['l2_mean']:.2f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate_v2(model, test_loader, device)
    return {
        "cfg": cfg.__dict__,
        "best_epoch": best_epoch,
        "best_val_l2": best_val,
        "test": test,
        "curve": curve,
    }
