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
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from src.feature_extractor import CalibrationFeatureExtractor, FeatureConfig


@dataclass
class ModelConfig:
    """Typed view over model-related configuration."""

    type: str
    input_dim: int
    hidden_dims: Sequence[int]
    dropout: float = 0.0
    coordinate_scale: float = 100.0
    use_layer_norm: bool = False
    use_batch_norm: bool = False
    activation: str = "relu"


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""

    enabled: bool = False
    noise_std: float = 2.0
    translate_range: float = 5.0
    scale_range: tuple = (0.98, 1.02)


@dataclass
class SimRBFPerturbationConfig:
    """
    Configuration for SimRBF perturbation augmentation.

    This augmentation simulates varying calibration quality by perturbing
    the sim_rbf_gaze predictions. The key insight is that few-shot (18-point)
    SimRBF calibration has high variance, and we want the neural network to
    learn to correct for poor calibrations.

    The network learns:
    - When (original, sim_rbf) shows large consistent correction → good calibration → small residual
    - When (original, sim_rbf) shows inconsistent correction → poor calibration → large residual
    """

    enabled: bool = False
    noise_std: float = 15.0
    bias_range: float = 30.0
    prob: float = 0.5


@dataclass
class HighFrequencyFeaturesConfig:
    """
    Configuration for high-frequency calibration point features.

    When enabled, extracts fixed-dimensional features from test_x_0, test_y_0, ...
    columns that characterize the calibration cluster (statistics, spatial distribution,
    quality indicators, relative positions).
    """

    enabled: bool = False
    calibration_point_prefix: str = "test"
    include_statistical: bool = True
    include_spatial: bool = True
    include_quality: bool = True
    include_relative: bool = True
    normalize_features: bool = True


@dataclass
class MultiBaselineFeaturesConfig:
    """
    Configuration for multiple baseline calibration features.

    When enabled, includes predictions from multiple calibration methods as input features.
    The network learns to fuse these baselines to predict original -> target offset.

    Key difference from cascade mode:
    - Cascade: predicts sim_rbf -> target residual
    - Multi-baseline: predicts original -> target offset using multiple baselines as features
    """

    enabled: bool = False
    baselines: list = None  # List of baseline names (e.g., ["sim_rbf_gaze", "sim_tps", "poly"])
    normalize_features: bool = True

    def __post_init__(self):
        if self.baselines is None:
            self.baselines = []


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

    SimRBF Perturbation Augmentation (cascade mode only):
        When enabled, perturbs sim_rbf_gaze to simulate varying calibration
        quality. This teaches the network to:
        - Output small residuals when calibration is good
        - Output large residuals when calibration is poor
        The target is recomputed as: target_point - perturbed_sim_rbf
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
        augmentation: AugmentationConfig | dict | None = None,
        sim_rbf_perturbation: SimRBFPerturbationConfig | dict | None = None,
        high_freq_features: HighFrequencyFeaturesConfig | dict | None = None,
        multi_baseline_features: MultiBaselineFeaturesConfig | dict | None = None,
        is_training: bool = True,
    ) -> None:
        super().__init__()
        if model_type not in {"end_to_end", "cascade", "multi_baseline"}:
            raise ValueError(f"Unsupported model_type: {model_type}")
        self.model_type = model_type
        self.is_training = is_training
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV path does not exist: {path}")

        self.aug_cfg, self.perturb_cfg, self.hf_cfg, self.mb_cfg = (
            self._parse_config(augmentation, AugmentationConfig),
            self._parse_config(sim_rbf_perturbation, SimRBFPerturbationConfig),
            self._parse_config(high_freq_features, HighFrequencyFeaturesConfig),
            self._parse_config(multi_baseline_features, MultiBaselineFeaturesConfig),
        )

        df = pd.read_csv(path)

        column_mapping = {
            "original_gaze_x": "origin_gaze_x",
            "original_gaze_y": "origin_gaze_y",
            "sim_rbf_gaze_x": "pred_sim_rbf_multiquadric_s2.0_x",
            "sim_rbf_gaze_y": "pred_sim_rbf_multiquadric_s2.0_y",
        }

        required_columns = list(self.required_columns_base)
        if model_type == "cascade":
            required_columns.extend(["sim_rbf_gaze_x", "sim_rbf_gaze_y"])

        actual_required_columns = [
            column_mapping.get(col, col) for col in required_columns
        ]
        missing = [
            req_col
            for req_col, actual_col in zip(required_columns, actual_required_columns)
            if actual_col not in df.columns
        ]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {missing}")

        targets_px = df[["target_x", "target_y"]].to_numpy(dtype=np.float32)
        orig_px = df[[column_mapping["original_gaze_x"], column_mapping["original_gaze_y"]]].to_numpy(dtype=np.float32)

        self.spread = torch.from_numpy(df["spread"].to_numpy(dtype=np.float32)) if "spread" in df.columns else None

        if model_type == "cascade":
            sim_px = df[[column_mapping["sim_rbf_gaze_x"], column_mapping["sim_rbf_gaze_y"]]].to_numpy(dtype=np.float32)
            inputs_px = np.concatenate([orig_px, sim_px], axis=1)
            residuals_px = targets_px - sim_px
            self.sim_inputs_px = torch.from_numpy(sim_px)
        else:
            inputs_px, residuals_px = orig_px, targets_px - orig_px

        self.orig_inputs_px = torch.from_numpy(orig_px)
        self.inputs_px = torch.from_numpy(inputs_px)
        self.targets_px = torch.from_numpy(targets_px)
        self.residuals_px = torch.from_numpy(residuals_px)

        self.hf_features = None
        self.hf_feature_extractor = None
        if self.hf_cfg.enabled:
            print(f"Extracting high-frequency calibration features...")
            feature_config = FeatureConfig(
                enabled=True,
                calibration_point_prefix=self.hf_cfg.calibration_point_prefix,
                include_statistical=self.hf_cfg.include_statistical,
                include_spatial=self.hf_cfg.include_spatial,
                include_quality=self.hf_cfg.include_quality,
                include_relative=self.hf_cfg.include_relative,
                normalize_features=self.hf_cfg.normalize_features,
            )
            self.hf_feature_extractor = CalibrationFeatureExtractor(feature_config)

            hf_features_array = self.hf_feature_extractor.fit_transform(df, gaze_data=orig_px)
            print(f"  Extracted {hf_features_array.shape[1]} features from {len(df)} samples")
            print(f"  Feature names: {self.hf_feature_extractor.get_feature_names()[:5]}...")
            self.hf_features = torch.from_numpy(hf_features_array)

        self.mb_features = None
        self.mb_feature_names = []
        if self.mb_cfg.enabled and len(self.mb_cfg.baselines) > 0:
            print(f"Extracting multi-baseline features...")

            baseline_column_mapping = {
                "sim_rbf_gaze": "pred_sim_rbf_multiquadric_s2.0",
            }

            baseline_features_list = []
            for baseline_name in self.mb_cfg.baselines:
                actual_baseline = baseline_column_mapping.get(baseline_name, baseline_name)

                col_x = f"{actual_baseline}_x"
                col_y = f"{actual_baseline}_y"
                if col_x in df.columns and col_y in df.columns:
                    baseline_data = df[[col_x, col_y]].to_numpy(dtype=np.float32)
                    baseline_residual = baseline_data - targets_px
                    baseline_features_list.append(baseline_residual)
                    self.mb_feature_names.extend([f"{baseline_name}_x_residual", f"{baseline_name}_y_residual"])
                    print(f"  Added {baseline_name} baseline (as residual) from columns {col_x}, {col_y}")
                else:
                    print(f"  Warning: {baseline_name} columns ({col_x}, {col_y}) not found in data")

            if baseline_features_list:
                self.mb_features = np.concatenate(baseline_features_list, axis=1)
                print(f"  Total multi-baseline residual features: {self.mb_features.shape[1]} ({len(self.mb_cfg.baselines)} baselines)")

                if self.mb_cfg.normalize_features:
                    self.mb_mean = self.mb_features.mean(axis=0)
                    self.mb_std = self.mb_features.std(axis=0) + 1e-6
                    self.mb_features = (self.mb_features - self.mb_mean) / self.mb_std
                    print(f"  Normalized multi-baseline residual features")

                self.mb_features = torch.from_numpy(self.mb_features)

        if normalize:
            inputs_px, residuals_px = inputs_px / coordinate_scale, residuals_px / coordinate_scale

        self.inputs = torch.from_numpy(inputs_px)
        self.targets = torch.from_numpy(residuals_px)
        self.coordinate_scale = coordinate_scale
        self.normalize = normalize

        if self.mb_features is not None:
            self.inputs = torch.cat([self.inputs, self.mb_features], dim=1)
            print(f"  Input dimension with multi-baseline features: {self.inputs.shape[1]}")

        if self.hf_features is not None:
            self.inputs = torch.cat([self.inputs, self.hf_features], dim=1)
            print(f"  Input dimension with HF features: {self.inputs.shape[1]}")

    def _parse_config(self, config, config_class):
        """Parse config from dict or return existing config object."""
        if config is None:
            return config_class()
        elif isinstance(config, dict):
            return config_class(**{k: v for k, v in config.items() if k in config_class.__dataclass_fields__})
        else:
            return config

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def _apply_sim_rbf_perturbation(
        self, inputs: torch.Tensor, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply SimRBF perturbation augmentation (cascade mode only).

        Simulates varying calibration quality by perturbing sim_rbf_gaze.
        The network learns to recognize calibration quality from (original, sim_rbf)
        relationship and output appropriate residual corrections.
        """
        if not self.perturb_cfg.enabled or not self.is_training:
            return inputs, self.targets[idx]

        if self.model_type != "cascade":
            return inputs, self.targets[idx]

        if torch.rand(1).item() > self.perturb_cfg.prob:
            return inputs, self.targets[idx]

        scale = self.coordinate_scale if self.normalize else 1.0
        inputs_px = inputs.clone() * scale

        orig_gaze = inputs_px[:2]
        sim_rbf_gaze = inputs_px[2:4]
        hf_features = inputs_px[4:]

        target_px = self.targets_px[idx].clone()

        perturbation = torch.zeros(2)
        if self.perturb_cfg.noise_std > 0:
            perturbation += torch.randn(2) * self.perturb_cfg.noise_std
        if self.perturb_cfg.bias_range > 0:
            bias = (torch.rand(2) * 2 - 1) * self.perturb_cfg.bias_range
            perturbation += bias

        perturbed_sim_rbf = sim_rbf_gaze + perturbation
        new_target_px = target_px - perturbed_sim_rbf
        new_inputs_px = torch.cat([orig_gaze, perturbed_sim_rbf, hf_features])

        new_inputs = new_inputs_px / scale
        new_target = new_target_px / scale

        return new_inputs, new_target

    def _apply_augmentation(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply general data augmentation to a single sample."""
        if not self.aug_cfg.enabled or not self.is_training:
            return inputs, targets

        scale = self.coordinate_scale if self.normalize else 1.0
        inputs_px = inputs.clone() * scale
        targets_px = targets.clone() * scale

        if self.aug_cfg.noise_std > 0:
            noise = torch.randn_like(inputs_px) * self.aug_cfg.noise_std
            inputs_px = inputs_px + noise

        if self.aug_cfg.translate_range > 0:
            translate = (torch.rand(2) * 2 - 1) * self.aug_cfg.translate_range
            if self.model_type == "cascade":
                inputs_px[:2] = inputs_px[:2] + translate
                inputs_px[2:] = inputs_px[2:] + translate
            else:
                inputs_px = inputs_px + translate

        if self.aug_cfg.scale_range[0] < self.aug_cfg.scale_range[1]:
            scale_factor = (
                torch.rand(1).item()
                * (self.aug_cfg.scale_range[1] - self.aug_cfg.scale_range[0])
                + self.aug_cfg.scale_range[0]
            )
            if self.model_type == "cascade":
                inputs_px[:2] = inputs_px[:2] * scale_factor
                inputs_px[2:] = inputs_px[2:] * scale_factor
            else:
                inputs_px = inputs_px * scale_factor
            targets_px = targets_px * scale_factor

        inputs_aug = inputs_px / scale
        targets_aug = targets_px / scale

        return inputs_aug, targets_aug

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.inputs[idx]
        targets = self.targets[idx]

        if self.is_training and self.perturb_cfg.enabled:
            inputs, targets = self._apply_sim_rbf_perturbation(inputs.clone(), idx)

        if self.is_training and self.aug_cfg.enabled:
            inputs, targets = self._apply_augmentation(inputs.clone(), targets.clone())

        return inputs, targets

    def get_sample_weights(self, scale: float = 1.0) -> torch.Tensor | None:
        """Get sample weights based on spread (inverse uncertainty weighting)."""
        if self.spread is None:
            return None

        epsilon = 1e-6
        weights = 1.0 / (self.spread + epsilon)
        weights = weights / weights.mean()
        weights = weights ** scale
        return weights


class ResidualBlock(nn.Module):
    """A small residual MLP block used to stabilize deeper stacks."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1, self.fc2, self.dropout, self.act = nn.Linear(dim, dim), nn.Linear(dim, dim), dropout, nn.ReLU()

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


def build_model(cfg: dict, hf_feature_dim: int = 0) -> GazeRefineNet:
    """Build a model instance from a config dictionary."""
    model_cfg = ModelConfig(
        type=cfg.get("type", "end_to_end"),
        input_dim=cfg.get("input_dim", 2),
        hidden_dims=cfg.get("hidden_dims", [512]),
        dropout=cfg.get("dropout", 0.0),
        coordinate_scale=cfg.get("coordinate_scale", 100.0),
    )

    supported_types = {"end_to_end", "cascade", "cascade_hf", "end_to_end_hf", "multi_baseline"}
    if model_cfg.type not in supported_types:
        raise ValueError(f"Unsupported model type: {model_cfg.type}")

    base_input_dim = 4 if model_cfg.type in ["cascade", "cascade_hf"] else 2
    total_input_dim = base_input_dim + hf_feature_dim

    if model_cfg.input_dim != base_input_dim:
        raise ValueError(
            f"{model_cfg.type} expects input_dim={base_input_dim}, "
            f"got {model_cfg.input_dim}"
        )

    return GazeRefineNet(
        input_dim=total_input_dim,
        hidden_dims=model_cfg.hidden_dims,
        dropout=model_cfg.dropout,
    )
