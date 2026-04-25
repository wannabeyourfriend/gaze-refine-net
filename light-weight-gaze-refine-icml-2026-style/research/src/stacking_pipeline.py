"""Trial-stratified stacking ensemble.

Train a small MLP that, given (raw_gaze, baseline_predictions) for a
sample, outputs a refined gaze coordinate. Training uses leave-one-
TRIAL-out cross-validation: for each (subject, trial) we train on all
OTHER trials and produce out-of-fold predictions for that trial. This
gives an honest stacking signal that doesn't leak target information.

Then the final model is trained on the union of all out-of-fold
training data and evaluated on a held-out subject (LOSO).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class StackCfg:
    baselines: Sequence[str]
    anchor: str
    screen_w: float = 1920.0
    screen_h: float = 1080.0
    use_fourier: bool = True
    fourier_bands: int = 4
    hidden_dims: Sequence[int] = (64, 32)
    dropout: float = 0.0
    noise_sigma_max: float = 10.0
    noise_prob: float = 0.5
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-3
    seed: int = 1047


def fourier(x: torch.Tensor, n: int) -> torch.Tensor:
    if n <= 0:
        return x
    bands = 2.0 ** torch.arange(n, dtype=x.dtype, device=x.device)
    xb = x.unsqueeze(-1) * bands
    return torch.cat([x, torch.sin(math.pi * xb).flatten(-2), torch.cos(math.pi * xb).flatten(-2)], -1)


class StackerNet(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def featurize(orig: torch.Tensor, baselines: torch.Tensor, anchor: torch.Tensor,
              screen_w: float, screen_h: float, use_fourier: bool, fourier_bands: int) -> torch.Tensor:
    """orig (B,2), baselines (B,K,2), anchor (B,2). All in pixels.

    Features: anchor-relative offsets (orig - anchor, baseline - anchor) normalized
    by a small px scale, plus the anchor's normalized screen position with
    Fourier features for spatial conditioning.
    """
    px_scale = 50.0
    # Anchor screen position (normalized to [-1,1])
    anchor_n = anchor.clone()
    anchor_n[:, 0] = (anchor_n[:, 0] / screen_w) * 2 - 1
    anchor_n[:, 1] = (anchor_n[:, 1] / screen_h) * 2 - 1
    if use_fourier:
        anchor_n = fourier(anchor_n, fourier_bands)
    # Relative offsets (small, capped)
    orig_rel = (orig - anchor) / px_scale
    bs_rel = (baselines - anchor.unsqueeze(1)) / px_scale  # (B,K,2)
    bs_rel = bs_rel.flatten(start_dim=1)
    return torch.cat([anchor_n, orig_rel, bs_rel], dim=-1)


class StackDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: StackCfg, is_training: bool) -> None:
        if "origin_gaze_x" in df.columns:
            ox, oy = "origin_gaze_x", "origin_gaze_y"
        else:
            ox, oy = "original_gaze_x", "original_gaze_y"
        self.cfg = cfg
        self.is_training = is_training
        self.orig = torch.from_numpy(df[[ox, oy]].to_numpy(dtype=np.float32))
        self.targets = torch.from_numpy(df[["target_x", "target_y"]].to_numpy(dtype=np.float32))
        bs = np.stack([df[[f"{b}_x", f"{b}_y"]].to_numpy(dtype=np.float32) for b in cfg.baselines], axis=1)
        self.baselines = torch.from_numpy(bs)
        anchor_idx = list(cfg.baselines).index(cfg.anchor)
        self.anchor = self.baselines[:, anchor_idx, :].clone()

    def __len__(self) -> int:
        return self.orig.shape[0]

    def __getitem__(self, idx: int):
        orig = self.orig[idx]
        bs = self.baselines[idx]
        anchor = self.anchor[idx]
        if self.is_training and self.cfg.noise_sigma_max > 0 and torch.rand(1).item() < self.cfg.noise_prob:
            sigmas = torch.rand(bs.shape) * self.cfg.noise_sigma_max
            bs = bs + torch.randn_like(bs) * sigmas
            anchor = anchor + torch.randn_like(anchor) * (torch.rand(2) * self.cfg.noise_sigma_max)
        return {"orig": orig, "baselines": bs, "anchor": anchor, "target": self.targets[idx]}


def train_stacker(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: StackCfg,
    *,
    device: Optional[torch.device] = None,
    verbose: bool = False,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_ds = StackDataset(train_df, cfg, is_training=True)
    val_ds = StackDataset(val_df, cfg, is_training=False)
    test_ds = StackDataset(test_df, cfg, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # Determine input dim
    sample = train_ds[0]
    feats = featurize(sample["orig"].unsqueeze(0), sample["baselines"].unsqueeze(0), sample["anchor"].unsqueeze(0),
                      cfg.screen_w, cfg.screen_h, cfg.use_fourier, cfg.fourier_bands)
    in_dim = feats.shape[-1]
    model = StackerNet(in_dim, cfg.hidden_dims, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

    def eval_loader(loader: DataLoader):
        model.eval()
        l2s, exs, eys = [], [], []
        with torch.no_grad():
            for b in loader:
                orig = b["orig"].to(device); bs = b["baselines"].to(device); anchor = b["anchor"].to(device); tgt = b["target"].to(device)
                feats = featurize(orig, bs, anchor, cfg.screen_w, cfg.screen_h, cfg.use_fourier, cfg.fourier_bands)
                resid = model(feats)  # in pixels (small)
                pred = anchor + resid
                err = (pred - tgt).cpu().numpy()
                exs.append(err[:, 0]); eys.append(err[:, 1])
                l2s.append(np.sqrt((err ** 2).sum(axis=1)))
        l2 = np.concatenate(l2s); ex = np.concatenate(exs); ey = np.concatenate(eys)
        return {"l2_mean": float(l2.mean()), "l2_median": float(np.median(l2)), "l2_std": float(l2.std()),
                "l2_p95": float(np.percentile(l2, 95)),
                "mae_x": float(np.mean(np.abs(ex))), "mae_y": float(np.mean(np.abs(ey))),
                "rmse_x": float(np.sqrt((ex**2).mean())), "rmse_y": float(np.sqrt((ey**2).mean())),
                "n": int(len(l2))}

    best_val = math.inf
    best_state = None
    best_epoch = 0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        for b in train_loader:
            orig = b["orig"].to(device); bs = b["baselines"].to(device); anchor = b["anchor"].to(device); tgt = b["target"].to(device)
            feats = featurize(orig, bs, anchor, cfg.screen_w, cfg.screen_h, cfg.use_fourier, cfg.fourier_bands)
            resid = model(feats)
            pred = anchor + resid
            loss = ((pred - tgt) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        v = eval_loader(val_loader)
        if v["l2_mean"] < best_val:
            best_val = v["l2_mean"]; best_epoch = ep
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if verbose and ep % 25 == 0:
            print(f"  ep {ep:3d}: val_L2={v['l2_mean']:.2f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_l2": best_val, "best_epoch": best_epoch, "test": eval_loader(test_loader)}
