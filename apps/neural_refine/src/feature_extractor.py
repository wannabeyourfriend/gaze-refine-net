"""
Feature extraction module for high-frequency calibration points.

This module processes raw calibration point data (test_x_0 to test_y_N) and
extracts fixed-dimensional features that characterize:
1. Statistical properties of the calibration cluster
2. Spatial relationships
3. Quality indicators of the calibration session
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class FeatureConfig:
    """Configuration for feature extraction from calibration points."""

    enabled: bool = False
    calibration_point_prefix: str = "test"  # Prefix for calibration columns (test_x_0, test_y_0)

    # Feature selection
    include_statistical: bool = True  # Mean, std, min, max, etc.
    include_spatial: bool = True  # Distances, bounding box, etc.
    include_quality: bool = True  # Spread ratios, density, etc.
    include_relative: bool = True  # Position relative to calibration cluster

    # Normalization
    normalize_features: bool = True  # Normalize extracted features

    # Calibration point columns (auto-detected if None)
    calibration_x_cols: Optional[List[str]] = None
    calibration_y_cols: Optional[List[str]] = None


class CalibrationFeatureExtractor:
    """
    Extract fixed-dimensional features from high-frequency calibration points.

    Input: DataFrame with columns test_x_0, test_y_0, ..., test_x_N, test_y_N
    Output: Fixed-dimensional feature vector (default ~50-100 features)

    Features extracted:
    1. Statistical: mean, std, min, max, median, q1, q3, skew, kurtosis (per dimension)
    2. Spatial: bounding box, coverage area, centroid, pairwise distances
    3. Quality: spread ratios, density metrics, consistency measures
    4. Relative: position of gaze points relative to calibration cluster
    """

    def __init__(self, config: FeatureConfig | dict | None = None):
        if config is None:
            self.config = FeatureConfig()
        elif isinstance(config, dict):
            self.config = FeatureConfig(**config)
        else:
            self.config = config

        self.feature_names: List[str] = []
        self.fitted_stats: Dict[str, float] = {}

    def _detect_calibration_columns(
        self, df: pd.DataFrame
    ) -> tuple[List[str], List[str]]:
        """Auto-detect calibration point columns."""
        prefix = self.config.calibration_point_prefix

        x_cols = sorted([col for col in df.columns if col.startswith(f"{prefix}_x_")])
        y_cols = sorted([col for col in df.columns if col.startswith(f"{prefix}_y_")])

        if not x_cols or not y_cols:
            raise ValueError(
                f"No calibration columns found with prefix '{prefix}_x_' and '{prefix}_y_'"
            )

        return x_cols, y_cols

    def _extract_statistical_features(
        self, points: np.ndarray, prefix: str = ""
    ) -> Dict[str, float]:
        """
        Extract statistical features from calibration points.

        Args:
            points: Array of shape (N, 2) containing calibration point coordinates
            prefix: Prefix for feature names

        Returns:
            Dictionary of feature names to values
        """
        features = {}

        if len(points) == 0:
            return features

        x_coords = points[:, 0]
        y_coords = points[:, 1]

        # Per-dimension statistics
        for i, (coords, dim) in enumerate([(x_coords, "x"), (y_coords, "y")]):
            # Basic statistics
            features[f"{prefix}calib_{dim}_mean"] = np.mean(coords)
            features[f"{prefix}calib_{dim}_std"] = np.std(coords)
            features[f"{prefix}calib_{dim}_min"] = np.min(coords)
            features[f"{prefix}calib_{dim}_max"] = np.max(coords)
            features[f"{prefix}calib_{dim}_median"] = np.median(coords)
            features[f"{prefix}calib_{dim}_q1"] = np.percentile(coords, 25)
            features[f"{prefix}calib_{dim}_q3"] = np.percentile(coords, 75)
            features[f"{prefix}calib_{dim}_range"] = np.max(coords) - np.min(coords)
            features[f"{prefix}calib_{dim}_iqr"] = (
                np.percentile(coords, 75) - np.percentile(coords, 25)
            )

            # Distribution shape (if enough points)
            if len(coords) >= 3:
                features[f"{prefix}calib_{dim}_skew"] = stats.skew(coords)
                features[f"{prefix}calib_{dim}_kurtosis"] = stats.kurtosis(coords)

        # 2D statistics
        centroid = np.mean(points, axis=0)
        features[f"{prefix}calib_centroid_x"] = centroid[0]
        features[f"{prefix}calib_centroid_y"] = centroid[1]

        # Spread from centroid
        distances_from_centroid = np.linalg.norm(points - centroid, axis=1)
        features[f"{prefix}calib_spread_mean"] = np.mean(distances_from_centroid)
        features[f"{prefix}calib_spread_std"] = np.std(distances_from_centroid)
        features[f"{prefix}calib_spread_max"] = np.max(distances_from_centroid)

        return features

    def _extract_spatial_features(
        self, points: np.ndarray, prefix: str = ""
    ) -> Dict[str, float]:
        """
        Extract spatial features from calibration points.

        Args:
            points: Array of shape (N, 2) containing calibration point coordinates
            prefix: Prefix for feature names

        Returns:
            Dictionary of feature names to values
        """
        features = {}

        if len(points) < 2:
            return features

        # Bounding box
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        features[f"{prefix}calib_bbox_area"] = (max_x - min_x) * (max_y - min_y)
        features[f"{prefix}calib_bbox_width"] = max_x - min_x
        features[f"{prefix}calib_bbox_height"] = max_y - min_y
        features[f"{prefix}calib_bbox_aspect_ratio"] = (max_x - min_x) / (
            max_y - min_y + 1e-6
        )

        # Pairwise distances
        from scipy.spatial.distance import pdist

        pairwise_dists = pdist(points, metric="euclidean")
        features[f"{prefix}calib_pairwise_dist_mean"] = np.mean(pairwise_dists)
        features[f"{prefix}calib_pairwise_dist_std"] = np.std(pairwise_dists)
        features[f"{prefix}calib_pairwise_dist_min"] = np.min(pairwise_dists)
        features[f"{prefix}calib_pairwise_dist_max"] = np.max(pairwise_dists)

        # Coverage (convex hull area if enough points)
        if len(points) >= 3:
            from scipy.spatial import ConvexHull

            try:
                hull = ConvexHull(points)
                features[f"{prefix}calib_hull_area"] = hull.volume  # In 2D, volume = area
                features[f"{prefix}calib_hull_perimeter"] = hull.area
            except:
                # Points are collinear or other degenerate case
                features[f"{prefix}calib_hull_area"] = 0.0
                features[f"{prefix}calib_hull_perimeter"] = 0.0

        return features

    def _extract_quality_features(
        self, points: np.ndarray, prefix: str = ""
    ) -> Dict[str, float]:
        """
        Extract quality indicators from calibration points.

        Args:
            points: Array of shape (N, 2) containing calibration point coordinates
            prefix: Prefix for feature names

        Returns:
            Dictionary of feature names to values
        """
        features = {}

        if len(points) == 0:
            return features

        # Density metrics
        if len(points) >= 2:
            from scipy.spatial.distance import pdist

            pairwise_dists = pdist(points, metric="euclidean")
            # Average nearest neighbor distance
            features[f"{prefix}calib_nnd_mean"] = np.mean(pairwise_dists)

            # Density: points per unit area (using bounding box)
            min_x, min_y = np.min(points, axis=0)
            max_x, max_y = np.max(points, axis=0)
            area = (max_x - min_x) * (max_y - min_y) + 1e-6
            features[f"{prefix}calib_density"] = len(points) / area

        # Spread consistency (std / mean)
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        features[f"{prefix}calib_spread_cv"] = (
            std_dist / (mean_dist + 1e-6)
        )  # Coefficient of variation

        # Symmetry metrics (how balanced the points are around centroid)
        for i, dim in enumerate(["x", "y"]):
            dim_values = points[:, i]
            centroid_dim = centroid[i]
            # Fraction of points on each side of centroid
            left_ratio = np.sum(dim_values < centroid_dim) / len(dim_values)
            features[f"{prefix}calib_{dim}_symmetry"] = abs(
                left_ratio - 0.5
            ) * 2  # 0 = perfectly balanced

        return features

    def _extract_relative_features(
        self,
        points: np.ndarray,
        gaze_points: np.ndarray,
        prefix: str = "",
    ) -> Dict[str, float]:
        """
        Extract features relating gaze points to calibration cluster.

        Args:
            points: Array of shape (N, 2) containing calibration point coordinates
            gaze_points: Array of shape (M, 2) containing gaze coordinates
            prefix: Prefix for feature names

        Returns:
            Dictionary of feature names to values
        """
        features = {}

        if len(points) == 0 or len(gaze_points) == 0:
            return features

        centroid = np.mean(points, axis=0)

        # For each gaze point, compute relative position
        # If multiple gaze points (e.g., orig and sim_rbf), compute for all
        for i, gaze in enumerate(gaze_points):
            suffix = f"_g{i}" if len(gaze_points) > 1 else ""

            # Distance from calibration centroid
            dist_to_centroid = np.linalg.norm(gaze - centroid)
            features[f"{prefix}dist_to_calib_centroid{suffix}"] = dist_to_centroid

            # Relative position (normalized by bounding box)
            if len(points) >= 2:
                min_pt = np.min(points, axis=0)
                max_pt = np.max(points, axis=0)
                bbox_size = max_pt - min_pt + 1e-6
                relative_pos = (gaze - centroid) / bbox_size
                features[f"{prefix}relative_x{suffix}"] = relative_pos[0]
                features[f"{prefix}relative_y{suffix}"] = relative_pos[1]

            # Distance to nearest calibration point
            distances_to_calib = np.linalg.norm(points - gaze, axis=1)
            features[f"{prefix}dist_to_nearest_calib{suffix}"] = np.min(
                distances_to_calib
            )
            features[f"{prefix}dist_to_farthest_calib{suffix}"] = np.max(
                distances_to_calib
            )

        return features

    def fit(
        self,
        df: pd.DataFrame,
        gaze_columns: Optional[List[str]] = None,
    ) -> "CalibrationFeatureExtractor":
        """
        Fit the feature extractor on a dataset.

        Computes normalization statistics for feature scaling.

        Args:
            df: DataFrame containing calibration point columns
            gaze_columns: Optional list of gaze column names to use for relative features

        Returns:
            Self (fitted extractor)
        """
        if not self.config.enabled:
            return self

        # Detect calibration columns if not specified
        if self.config.calibration_x_cols is None:
            x_cols, y_cols = self._detect_calibration_columns(df)
        else:
            x_cols = self.config.calibration_x_cols
            y_cols = self.config.calibration_y_cols

        # Extract features from all samples to compute normalization stats
        all_features = []
        for _, row in df.iterrows():
            features = self._extract_features_from_row(
                row, x_cols, y_cols, gaze_columns
            )
            all_features.append(features)

        if not all_features:
            return self

        # Convert to array for normalization
        feature_df = pd.DataFrame(all_features)

        # Compute mean and std for each feature
        for col in feature_df.columns:
            self.fitted_stats[f"{col}_mean"] = feature_df[col].mean()
            self.fitted_stats[f"{col}_std"] = feature_df[col].std()

        return self

    def _extract_features_from_row(
        self,
        row: pd.Series,
        x_cols: List[str],
        y_cols: List[str],
        gaze_columns: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Extract features from a single row of data."""
        features = {}

        # Extract calibration points
        calib_points = []
        for x_col, y_col in zip(x_cols, y_cols):
            x_val = row.get(x_col)
            y_val = row.get(y_col)
            if pd.notna(x_val) and pd.notna(y_val):
                calib_points.append([x_val, y_val])

        if len(calib_points) == 0:
            return features

        calib_points = np.array(calib_points, dtype=np.float32)

        # Extract different feature groups
        if self.config.include_statistical:
            features.update(
                self._extract_statistical_features(calib_points, prefix="")
            )

        if self.config.include_spatial:
            features.update(self._extract_spatial_features(calib_points, prefix=""))

        if self.config.include_quality:
            features.update(self._extract_quality_features(calib_points, prefix=""))

        # Relative features (need gaze points)
        if self.config.include_relative and gaze_columns is not None:
            gaze_points = []
            for gaze_col in gaze_columns:
                if gaze_col in row and pd.notna(row[gaze_col]):
                    # Extract x and y from column name
                    # Assuming columns are named like "original_gaze_x", "sim_rbf_gaze_x"
                    # Need both x and y for each gaze type
                    pass  # Handled in transform method

        return features

    def transform(
        self,
        df: pd.DataFrame,
        gaze_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Transform a dataset to extract features.

        Args:
            df: DataFrame containing calibration point columns
            gaze_data: Optional array of shape (N, 2) or (N, 4) containing gaze coordinates
                      For cascade mode: [orig_x, orig_y, sim_x, sim_y]
                      For end_to_end mode: [orig_x, orig_y]

        Returns:
            Array of shape (N, D) where D is the number of extracted features
        """
        if not self.config.enabled:
            return np.zeros((len(df), 0))

        # Detect calibration columns if not specified
        if self.config.calibration_x_cols is None:
            x_cols, y_cols = self._detect_calibration_columns(df)
        else:
            x_cols = self.config.calibration_x_cols
            y_cols = self.config.calibration_y_cols

        feature_list = []
        self.feature_names = []

        for idx, row in df.iterrows():
            # Extract calibration points
            calib_points = []
            for x_col, y_col in zip(x_cols, y_cols):
                x_val = row.get(x_col)
                y_val = row.get(y_col)
                if pd.notna(x_val) and pd.notna(y_val):
                    calib_points.append([x_val, y_val])

            if len(calib_points) == 0:
                # No calibration points for this sample
                feature_list.append(np.zeros(len(self.fitted_stats) // 2))
                continue

            calib_points = np.array(calib_points, dtype=np.float32)

            # Extract features
            features = {}

            if self.config.include_statistical:
                features.update(self._extract_statistical_features(calib_points))

            if self.config.include_spatial:
                features.update(self._extract_spatial_features(calib_points))

            if self.config.include_quality:
                features.update(self._extract_quality_features(calib_points))

            # Relative features (if gaze data provided)
            if self.config.include_relative and gaze_data is not None:
                gaze_point = gaze_data[idx]
                # Reshape if needed
                if gaze_point.ndim == 0:
                    continue
                elif gaze_point.ndim == 1 and len(gaze_point) >= 2:
                    # Extract gaze points based on mode
                    if len(gaze_point) == 2:
                        # end_to_end: only original gaze
                        gaze_points = np.array([gaze_point[:2]])
                    elif len(gaze_point) == 4:
                        # cascade: original and sim_rbf
                        gaze_points = np.array([gaze_point[:2], gaze_point[2:]])
                    else:
                        gaze_points = np.array([gaze_point[:2]])

                    features.update(
                        self._extract_relative_features(
                            calib_points, gaze_points, prefix=""
                        )
                    )

            # Store feature names on first iteration
            if idx == 0:
                self.feature_names = list(features.keys())

            # Convert to array
            feature_vector = np.array(
                [features.get(name, 0.0) for name in self.feature_names],
                dtype=np.float32,
            )
            feature_list.append(feature_vector)

        feature_array = np.array(feature_list, dtype=np.float32)

        # Normalize if configured and stats are available
        if self.config.normalize_features and self.fitted_stats:
            for i, name in enumerate(self.feature_names):
                mean_key = f"{name}_mean"
                std_key = f"{name}_std"
                if mean_key in self.fitted_stats and std_key in self.fitted_stats:
                    mean = self.fitted_stats[mean_key]
                    std = self.fitted_stats[std_key] + 1e-6
                    feature_array[:, i] = (feature_array[:, i] - mean) / std

        return feature_array

    def fit_transform(
        self,
        df: pd.DataFrame,
        gaze_data: Optional[np.ndarray] = None,
        gaze_columns: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(df, gaze_columns)
        return self.transform(df, gaze_data)

    def get_feature_dim(self) -> int:
        """Get the output dimension of extracted features."""
        return len(self.feature_names)

    def get_feature_names(self) -> List[str]:
        """Get the names of extracted features."""
        return self.feature_names
