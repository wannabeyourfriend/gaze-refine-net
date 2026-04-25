"""
Honest implementation of the JuDo multi-baseline neural refinement pipeline.

This faithfully implements what the ICML 2026 paper §3 claims:

  Stage I  (existing) — diverse classical calibrators (similarity, poly,
                       RBF, TPS, SimRBF, SimTPS, SimPWA), fit on train
                       calibration points only.
  Stage II            — per-trial top-M selection by calibration risk
                       E_k = (1/N) sum ||delta_k(p_i) - delta(p_i)||
                       computed on a held-out calibration subset.
  Stage III           — lightweight MLP refiner with [64, 32, 16] hidden,
                       BatchNorm + ReLU; input is concatenation of
                       (raw_gaze, top_M baseline predictions). Trained
                       with Gaussian noise injected on baseline offsets,
                       sigma ~ U[0, sigma_max].

Critically, the network INPUT does NOT contain (baseline_pred - target).
mb_features are raw baseline predictions only. Normalization uses
training-set statistics applied unchanged to val/test.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

DEFAULT_BASELINES: Tuple[str, ...] = (
    "pred_similarity",
    "pred_poly",
    "pred_rbf_multiquadric_s0.0",
    "pred_rbf_multiquadric_s1.0",
    "pred_rbf_multiquadric_s2.0",
    "pred_tps",
    "pred_sim_rbf_multiquadric_s0.0",
    "pred_sim_rbf_multiquadric_s1.0",
    "pred_sim_rbf_multiquadric_s2.0",
    "pred_sim_tps",
    "pred_sim_pwa",
)


@dataclass
class PipelineConfig:
    """All knobs needed to instantiate a single training run."""

    train_csv: str
    val_csv: str
    test_csv: str

    baselines: Tuple[str, ...] = DEFAULT_BASELINES
    selection: str = "all"            # "all" | "topM" | "topM_oracle"
    top_m: int = 4                    # used iff selection startswith "topM"
    selection_source: str = "val"     # "val" | "train_holdout" | "test_oracle"
    selection_pool_frac: float = 0.5  # used if selection_source == "train_holdout"

    hidden_dims: Tuple[int, ...] = (64, 32, 16)
    dropout: float = 0.0
    use_batchnorm: bool = True

    coordinate_scale: float = 100.0
    noise_sigma_max: float = 0.0      # if > 0, sigma ~ U[0, sigma_max] in pixels
    noise_prob: float = 1.0

    batch_size: int = 64
    epochs: int = 200
    lr: float = 3e-4
    weight_decay: float = 0.01
    seed: int = 1047

    leaky_features: bool = False      # diagnostic only — recreates paper bug

    def __post_init__(self) -> None:
        self.baselines = tuple(self.baselines)
        self.hidden_dims = tuple(self.hidden_dims)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def _baseline_array(df: pd.DataFrame, baselines: Sequence[str]) -> np.ndarray:
    """Return (N, 2K) array of stacked baseline predictions."""
    chunks = []
    for b in baselines:
        chunks.append(df[[f"{b}_x", f"{b}_y"]].to_numpy(dtype=np.float32))
    return np.concatenate(chunks, axis=1)


def compute_selection_ranking(
    df_pool: pd.DataFrame, baselines: Sequence[str]
) -> List[int]:
    """Return baseline indices ordered by ascending mean L2 risk on a pool."""
    targets = df_pool[["target_x", "target_y"]].to_numpy(dtype=np.float32)
    risks = []
    for b in baselines:
        pred = df_pool[[f"{b}_x", f"{b}_y"]].to_numpy(dtype=np.float32)
        l2 = np.linalg.norm(pred - targets, axis=1).mean()
        risks.append(float(l2))
    order = np.argsort(risks).tolist()
    return order


def load_split(
    csv_path: str | Path,
    baselines: Sequence[str],
    *,
    coordinate_scale: float,
    selected_idx: Optional[Sequence[int]] = None,
    leaky: bool = False,
) -> Dict[str, np.ndarray]:
    df = pd.read_csv(csv_path)
    orig = df[["origin_gaze_x", "origin_gaze_y"]].to_numpy(dtype=np.float32)
    targets = df[["target_x", "target_y"]].to_numpy(dtype=np.float32)
    bs = _baseline_array(df, baselines).reshape(len(df), len(baselines), 2)

    if selected_idx is not None:
        bs = bs[:, list(selected_idx), :]

    bs_flat = bs.reshape(len(df), -1)

    if leaky:
        # Reproduce the exact bug from src/model.py:254 — features are
        # baseline_pred - target, then per-split z-score.
        residual = bs_flat - np.tile(targets, (1, bs.shape[1]))
        mean, std = residual.mean(0), residual.std(0) + 1e-6
        mb_norm = (residual - mean) / std
        return dict(
            orig=orig,
            targets=targets,
            mb=mb_norm.astype(np.float32),
            mb_mean=mean,
            mb_std=std,
            bs_raw=bs_flat,
            scale=coordinate_scale,
        )

    return dict(
        orig=orig,
        targets=targets,
        bs_raw=bs_flat,
        scale=coordinate_scale,
    )


def normalize_with(
    bs: np.ndarray, orig: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    bs_norm = (bs - mean) / std
    return bs_norm.astype(np.float32), orig.astype(np.float32)


class GazeArrayDataset(Dataset):
    """Lightweight dataset over precomputed numpy arrays.

    Inputs:  orig (N,2) and baselines (N, 2M) — stacked at __getitem__
    Targets: residual = (target - orig) / scale   (i.e. predict orig→target)

    If sigma_max > 0 and is_training, baseline coordinates are perturbed
    independently per (sample, baseline) with sigma ~ U[0, sigma_max]
    pixels.
    """

    def __init__(
        self,
        orig: np.ndarray,
        baselines_norm: np.ndarray,
        baselines_raw: np.ndarray,
        targets: np.ndarray,
        *,
        scale: float,
        bs_mean: np.ndarray,
        bs_std: np.ndarray,
        is_training: bool,
        sigma_max: float = 0.0,
        noise_prob: float = 1.0,
    ) -> None:
        self.orig = torch.from_numpy(orig.astype(np.float32))
        self.bs_norm = torch.from_numpy(baselines_norm.astype(np.float32))
        self.bs_raw = torch.from_numpy(baselines_raw.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))
        self.scale = scale
        self.bs_mean = torch.from_numpy(bs_mean.astype(np.float32))
        self.bs_std = torch.from_numpy(bs_std.astype(np.float32))
        self.is_training = is_training
        self.sigma_max = sigma_max
        self.noise_prob = noise_prob

    def __len__(self) -> int:
        return self.orig.shape[0]

    def _maybe_noise(self, bs_raw: torch.Tensor) -> torch.Tensor:
        if not self.is_training or self.sigma_max <= 0.0:
            return bs_raw
        if torch.rand(1).item() > self.noise_prob:
            return bs_raw
        # Sample one sigma per coordinate, U[0, sigma_max]
        sigmas = torch.rand(bs_raw.shape) * self.sigma_max
        return bs_raw + torch.randn_like(bs_raw) * sigmas

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        orig = self.orig[idx]
        bs_raw = self.bs_raw[idx]
        target = self.targets[idx]

        bs_raw_pert = self._maybe_noise(bs_raw)
        bs_norm = (bs_raw_pert - self.bs_mean) / self.bs_std

        # Network input: [orig/scale, bs_norm]
        inp = torch.cat([orig / self.scale, bs_norm], dim=0)
        residual = (target - orig) / self.scale
        return inp, residual


class LeakyDataset(Dataset):
    """Diagnostic-only dataset reproducing the original buggy pipeline."""

    def __init__(
        self,
        orig: np.ndarray,
        mb: np.ndarray,
        targets: np.ndarray,
        scale: float,
    ) -> None:
        self.orig = torch.from_numpy(orig.astype(np.float32))
        self.mb = torch.from_numpy(mb.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))
        self.scale = scale

    def __len__(self) -> int:
        return self.orig.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([self.orig[idx] / self.scale, self.mb[idx]], dim=0)
        residual = (self.targets[idx] - self.orig[idx]) / self.scale
        return inp, residual


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

class GazeRefineMLP(nn.Module):
    """[input_dim] -> hidden_dims -> 2 with optional BatchNorm/Dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        *,
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(prev, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ----------------------------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------------------------

@dataclass
class RunResult:
    cfg: PipelineConfig
    selected_baselines: List[str]
    epochs_run: int
    best_val_l2: float
    best_epoch: int
    test_metrics: Dict[str, float]
    train_curve: List[Dict[str, float]] = field(default_factory=list)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    scale: float,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    errs_x: List[float] = []
    errs_y: List[float] = []
    l2s: List[float] = []
    with torch.no_grad():
        for inp, residual in loader:
            inp = inp.to(device)
            residual = residual.to(device)
            pred = model(inp)
            err_px = (pred - residual) * scale
            errs_x.append(err_px[:, 0].cpu().numpy())
            errs_y.append(err_px[:, 1].cpu().numpy())
            l2s.append(torch.linalg.norm(err_px, dim=1).cpu().numpy())
    ex = np.concatenate(errs_x)
    ey = np.concatenate(errs_y)
    l2 = np.concatenate(l2s)
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


def baseline_only_metrics(
    df: pd.DataFrame, baselines: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    targets = df[["target_x", "target_y"]].to_numpy(dtype=np.float32)
    out: Dict[str, Dict[str, float]] = {}
    for b in ["origin_gaze", *baselines]:
        col_x = f"{b}_x"
        col_y = f"{b}_y"
        if col_x not in df.columns or col_y not in df.columns:
            continue
        pred = df[[col_x, col_y]].to_numpy(dtype=np.float32)
        err = pred - targets
        l2 = np.linalg.norm(err, axis=1)
        out[b] = {
            "mae_x": float(np.mean(np.abs(err[:, 0]))),
            "mae_y": float(np.mean(np.abs(err[:, 1]))),
            "rmse_x": float(np.sqrt(np.mean(err[:, 0] ** 2))),
            "rmse_y": float(np.sqrt(np.mean(err[:, 1] ** 2))),
            "l2_mean": float(np.mean(l2)),
            "l2_median": float(np.median(l2)),
            "l2_std": float(np.std(l2)),
            "l2_p90": float(np.percentile(l2, 90)),
            "l2_p95": float(np.percentile(l2, 95)),
            "n": int(len(l2)),
        }
    return out


def make_loaders(
    cfg: PipelineConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], int, pd.DataFrame, np.ndarray, np.ndarray]:
    """Materialize datasets, applying selection and normalization correctly."""
    train_df = pd.read_csv(cfg.train_csv)
    val_df = pd.read_csv(cfg.val_csv)
    test_df = pd.read_csv(cfg.test_csv)

    baselines = list(cfg.baselines)
    rng = np.random.default_rng(cfg.seed)

    if cfg.selection in {"topM", "topM_oracle"}:
        if cfg.selection == "topM_oracle":
            # Diagnostic: rank using test-set risk (cheating, gives upper bound).
            pool_df = test_df
        elif cfg.selection_source == "val":
            # Honest: rank using held-out validation targets.
            pool_df = val_df
        elif cfg.selection_source == "train_holdout":
            perm = rng.permutation(len(train_df))
            n_pool = max(1, int(len(train_df) * cfg.selection_pool_frac))
            pool_df = train_df.iloc[perm[:n_pool]]
        else:
            raise ValueError(f"Unknown selection_source={cfg.selection_source}")

        order = compute_selection_ranking(pool_df, baselines)
        keep = order[: cfg.top_m]
        selected = [baselines[i] for i in keep]
    else:
        keep = list(range(len(baselines)))
        selected = list(baselines)

    # All training rows are used for the network (no train holdout for selection
    # in the val-based protocol).
    train_idx = np.arange(len(train_df))

    if cfg.leaky_features:
        # Diagnostic-only path matching the original code bug.
        train_pack = load_split(
            cfg.train_csv, baselines, coordinate_scale=cfg.coordinate_scale,
            selected_idx=keep, leaky=True,
        )
        val_pack = load_split(
            cfg.val_csv, baselines, coordinate_scale=cfg.coordinate_scale,
            selected_idx=keep, leaky=True,
        )
        test_pack = load_split(
            cfg.test_csv, baselines, coordinate_scale=cfg.coordinate_scale,
            selected_idx=keep, leaky=True,
        )
        train_ds = LeakyDataset(train_pack["orig"], train_pack["mb"], train_pack["targets"], cfg.coordinate_scale)
        val_ds = LeakyDataset(val_pack["orig"], val_pack["mb"], val_pack["targets"], cfg.coordinate_scale)
        test_ds = LeakyDataset(test_pack["orig"], test_pack["mb"], test_pack["targets"], cfg.coordinate_scale)
        input_dim = 2 + 2 * len(keep)
    else:
        # Honest path. Normalize baselines using train statistics ONLY.
        train_pack = load_split(
            cfg.train_csv, baselines, coordinate_scale=cfg.coordinate_scale,
            selected_idx=keep, leaky=False,
        )
        train_pack["orig"] = train_pack["orig"][train_idx]
        train_pack["targets"] = train_pack["targets"][train_idx]
        train_pack["bs_raw"] = train_pack["bs_raw"][train_idx]
        bs_mean = train_pack["bs_raw"].mean(0)
        bs_std = train_pack["bs_raw"].std(0) + 1e-6

        val_pack = load_split(
            cfg.val_csv, baselines, coordinate_scale=cfg.coordinate_scale,
            selected_idx=keep, leaky=False,
        )
        test_pack = load_split(
            cfg.test_csv, baselines, coordinate_scale=cfg.coordinate_scale,
            selected_idx=keep, leaky=False,
        )

        def make_ds(pack: Dict[str, np.ndarray], training: bool) -> GazeArrayDataset:
            bs_norm = ((pack["bs_raw"] - bs_mean) / bs_std).astype(np.float32)
            return GazeArrayDataset(
                orig=pack["orig"],
                baselines_norm=bs_norm,
                baselines_raw=pack["bs_raw"],
                targets=pack["targets"],
                scale=cfg.coordinate_scale,
                bs_mean=bs_mean,
                bs_std=bs_std,
                is_training=training,
                sigma_max=cfg.noise_sigma_max if training else 0.0,
                noise_prob=cfg.noise_prob,
            )

        train_ds = make_ds(train_pack, training=True)
        val_ds = make_ds(val_pack, training=False)
        test_ds = make_ds(test_pack, training=False)
        input_dim = 2 + 2 * len(keep)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, selected, input_dim, test_df, np.array([]), np.array([])


def train_one(
    cfg: PipelineConfig,
    *,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> RunResult:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_loader, val_loader, test_loader, selected, input_dim, _, _, _ = make_loaders(cfg)

    model = GazeRefineMLP(
        input_dim,
        cfg.hidden_dims,
        dropout=cfg.dropout,
        use_batchnorm=cfg.use_batchnorm,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = math.inf
    best_state = None
    best_epoch = 0
    curve: List[Dict[str, float]] = []

    for ep in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for inp, residual in train_loader:
            inp = inp.to(device)
            residual = residual.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(inp)
            loss = loss_fn(pred, residual)
            loss.backward()
            opt.step()
            running += float(loss.item()) * inp.size(0)
            n += inp.size(0)
        train_loss = running / max(1, n)
        val_metrics = evaluate_model(model, val_loader, scale=cfg.coordinate_scale, device=device)
        curve.append({"epoch": ep, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})

        if val_metrics["l2_mean"] < best_val:
            best_val = val_metrics["l2_mean"]
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (ep % 20 == 0 or ep == cfg.epochs):
            print(f"  ep {ep:3d}: train_loss={train_loss:.4f}  val_l2={val_metrics['l2_mean']:.2f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model(model, test_loader, scale=cfg.coordinate_scale, device=device)
    return RunResult(
        cfg=cfg,
        selected_baselines=selected,
        epochs_run=cfg.epochs,
        best_val_l2=best_val,
        best_epoch=best_epoch,
        test_metrics=test_metrics,
        train_curve=curve,
    )


def result_to_dict(r: RunResult) -> Dict[str, Any]:
    cfg = r.cfg
    return {
        "selection": cfg.selection,
        "top_m": cfg.top_m if cfg.selection == "topM" else len(cfg.baselines),
        "noise_sigma_max": cfg.noise_sigma_max,
        "hidden_dims": list(cfg.hidden_dims),
        "use_batchnorm": cfg.use_batchnorm,
        "dropout": cfg.dropout,
        "leaky_features": cfg.leaky_features,
        "selected_baselines": r.selected_baselines,
        "best_epoch": r.best_epoch,
        "best_val_l2": r.best_val_l2,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "seed": cfg.seed,
        **{f"test_{k}": v for k, v in r.test_metrics.items()},
    }


def save_results(rows: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(rows, f, indent=2)
