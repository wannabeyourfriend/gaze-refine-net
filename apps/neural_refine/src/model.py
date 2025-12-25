"""
Lightweight components for the neural-refine gaze correction model.

The goal is to learn 2D residuals (dx, dy) that transform a noisy gaze
estimate into a calibrated gaze target. Inputs are pixel coordinates in
the range of roughly [0, 2000] x [0, 1000]; targets are residuals in the
range of roughly [-50, 50] per dimension.

Key ideas implemented here:
- A compact MLP backbone with an optional residual block to keep the
  network stable while allowing a bit of depth.
- Dataset utilities that normalize coordinates to a smaller numeric
  range for easier optimization and return residual targets.
- Simple helper functions to build the model from a config dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


@dataclass
class ModelConfig:
    """Typed view over model-related configuration."""

    type: str
    input_dim: int
    hidden_dims: Sequence[int]
    dropout: float = 0.0
    coordinate_scale: float = 100.0


class GazeDataset(Dataset):
    """
    PyTorch Dataset for gaze calibration examples stored in CSV form.

    Modes:
        - end_to_end: inputs are (original_gaze_x, original_gaze_y);
          targets are residuals target - original.
        - cascade: inputs are (original_gaze_x, original_gaze_y,
          sim_rbf_gaze_x, sim_rbf_gaze_y); targets are residuals
          target - sim_rbf.

    All coordinates can optionally be normalized by ``coordinate_scale`` to
    keep values small during optimization.
    """

    required_columns_base = [
        "target_x",
        "target_y",
        "original_gaze_x",
        "original_gaze_y",
    ]

    def __init__(
        self,
        csv_path: str | Path,
        coordinate_scale: float = 100.0,
        normalize: bool = True,
        model_type: str = "end_to_end",
    ) -> None:
        super().__init__()
        if model_type not in {"end_to_end", "cascade"}:
            raise ValueError(f"Unsupported model_type: {model_type}")
        self.model_type = model_type
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV path does not exist: {path}")

        df = pd.read_csv(path)
        required_columns = list(self.required_columns_base)
        if model_type == "cascade":
            required_columns.extend(["sim_rbf_gaze_x", "sim_rbf_gaze_y"])
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {missing}")

        targets_px = df[["target_x", "target_y"]].to_numpy(dtype=np.float32)
        orig_px = df[["original_gaze_x", "original_gaze_y"]].to_numpy(
            dtype=np.float32
        )

        if model_type == "cascade":
            sim_px = df[["sim_rbf_gaze_x", "sim_rbf_gaze_y"]].to_numpy(
                dtype=np.float32
            )
            inputs_px = np.concatenate([orig_px, sim_px], axis=1)
            residuals_px = targets_px - sim_px
            self.sim_inputs_px = torch.from_numpy(sim_px)
        else:
            inputs_px = orig_px
            residuals_px = targets_px - orig_px

        # Store pixel-space copies for later exports / metrics.
        self.orig_inputs_px = torch.from_numpy(orig_px)
        self.inputs_px = torch.from_numpy(inputs_px)
        self.targets_px = torch.from_numpy(targets_px)
        self.residuals_px = torch.from_numpy(residuals_px)

        if normalize:
            inputs_px = inputs_px / coordinate_scale
            residuals_px = residuals_px / coordinate_scale

        self.inputs = torch.from_numpy(inputs_px)
        self.targets = torch.from_numpy(residuals_px)
        self.coordinate_scale = coordinate_scale
        self.normalize = normalize

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


class ResidualBlock(nn.Module):
    """A small residual MLP block used to stabilize deeper stacks."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = dropout
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.fc1(x))
        if self.dropout > 0:
            out = nn.functional.dropout(out, p=self.dropout, training=self.training)
        out = self.fc2(out)
        return self.act(x + out)


class GazeRefineNet(nn.Module):
    """
    A minimal MLP that predicts gaze residuals from 2D coordinates.

    Residual connections are included at the tail of the network to keep
    gradients healthy without complicating the architecture.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Iterable[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims or [512])

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
        self.backbone = nn.Sequential(*layers)

        self.tail_residual = ResidualBlock(prev_dim, dropout) if prev_dim else None
        self.head = nn.Linear(prev_dim or input_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x) if len(self.backbone) > 0 else x
        if self.tail_residual is not None:
            h = self.tail_residual(h)
        return self.head(h)


def build_model(cfg: dict) -> GazeRefineNet:
    """
    Convenience helper to build a model instance from a config mapping.

    Parameters
    ----------
    cfg: dict
        Config dictionary containing keys: input_dim, hidden_dims,
        dropout.
    """

    model_cfg = ModelConfig(
        type=cfg.get("type", "end_to_end"),
        input_dim=cfg.get("input_dim", 2),
        hidden_dims=cfg.get("hidden_dims", [512]),
        dropout=cfg.get("dropout", 0.0),
        coordinate_scale=cfg.get("coordinate_scale", 100.0),
    )

    supported_types = {"end_to_end", "cascade"}
    if model_cfg.type not in supported_types:
        raise ValueError(f"Unsupported model type: {model_cfg.type}")

    expected_input_dim = 4 if model_cfg.type == "cascade" else 2
    if model_cfg.input_dim != expected_input_dim:
        raise ValueError(
            f"{model_cfg.type} expects input_dim={expected_input_dim}, "
            f"got {model_cfg.input_dim}"
        )

    return GazeRefineNet(
        input_dim=model_cfg.input_dim,
        hidden_dims=model_cfg.hidden_dims,
        dropout=model_cfg.dropout,
    )
