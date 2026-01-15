"""
Dataset for user identification from gaze patterns.

Each sample is a session (one timestamp from one user) containing multiple gaze points.
The goal is to identify which user produced the gaze pattern in a given session.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


@dataclass
class SessionData:
    """Data for a single session."""

    user_id: int
    user_name: str
    timestamp: str
    features: np.ndarray  # Shape: (num_points, num_features)
    num_points: int


class GazeIdentityDataset(Dataset):
    """
    Dataset for user identification from gaze sessions.

    Each sample is a session containing multiple gaze points with features.
    The target is the user identity (classification).
    """

    # Feature column groups
    ORIGINAL_GAZE_COLS = ["origin_gaze_x", "origin_gaze_y"]
    TARGET_COLS = ["target_x", "target_y"]
    SIM_RBF_COLS = ["pred_sim_rbf_multiquadric_s1.0_x", "pred_sim_rbf_multiquadric_s1.0_y"]
    SPREAD_COL = ["spread"]

    def __init__(
        self,
        sessions: List[SessionData],
        max_points: int = 24,
        normalize: bool = True,
        feature_stats: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """
        Args:
            sessions: List of SessionData objects
            max_points: Maximum number of points per session (for padding/truncation)
            normalize: Whether to normalize features
            feature_stats: Dict with 'mean' and 'std' for normalization (computed if None)
        """
        self.sessions = sessions
        self.max_points = max_points
        self.normalize = normalize

        # Compute or use provided stats
        if normalize:
            if feature_stats is None:
                self.feature_stats = self._compute_feature_stats()
            else:
                self.feature_stats = feature_stats
        else:
            self.feature_stats = None

    def _compute_feature_stats(self) -> Dict[str, np.ndarray]:
        """Compute mean and std across all features."""
        all_features = np.concatenate([s.features for s in self.sessions], axis=0)
        return {
            "mean": np.mean(all_features, axis=0),
            "std": np.std(all_features, axis=0) + 1e-8,
        }

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            features: (max_points, num_features) - padded/truncated point features
            mask: (max_points,) - 1 for valid points, 0 for padding
            label: scalar - user ID
        """
        session = self.sessions[idx]
        features = session.features.copy()

        # Normalize if needed
        if self.normalize and self.feature_stats is not None:
            features = (features - self.feature_stats["mean"]) / self.feature_stats["std"]

        num_points = min(session.num_points, self.max_points)

        # Create padded features and mask
        padded_features = np.zeros((self.max_points, features.shape[1]), dtype=np.float32)
        mask = np.zeros(self.max_points, dtype=np.float32)

        padded_features[:num_points] = features[:num_points]
        mask[:num_points] = 1.0

        return (
            torch.from_numpy(padded_features),
            torch.from_numpy(mask),
            torch.tensor(session.user_id, dtype=torch.long),
        )


def load_sessions_from_csv(
    csv_path: str | Path,
    min_points_per_session: int = 10,
    use_original_gaze: bool = True,
    use_target: bool = True,
    use_sim_rbf: bool = True,
    use_spread: bool = True,
    use_residuals: bool = True,
) -> Tuple[List[SessionData], Dict[str, int], int]:
    """
    Load session data from CSV file.

    Args:
        csv_path: Path to the CSV file
        min_points_per_session: Minimum points required for a session to be included
        use_*: Feature flags

    Returns:
        sessions: List of SessionData objects
        user_to_id: Mapping from username to numeric ID
        num_features: Number of features per point
    """
    df = pd.read_csv(csv_path)

    # Build feature columns list
    feature_cols = []
    if use_original_gaze:
        feature_cols.extend(GazeIdentityDataset.ORIGINAL_GAZE_COLS)
    if use_target:
        feature_cols.extend(GazeIdentityDataset.TARGET_COLS)
    if use_sim_rbf:
        feature_cols.extend(GazeIdentityDataset.SIM_RBF_COLS)
    if use_spread:
        feature_cols.extend(GazeIdentityDataset.SPREAD_COL)

    # Check columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Compute residuals if requested
    if use_residuals:
        # Error: target - original_gaze
        df["residual_orig_x"] = df["target_x"] - df["origin_gaze_x"]
        df["residual_orig_y"] = df["target_y"] - df["origin_gaze_y"]
        feature_cols.extend(["residual_orig_x", "residual_orig_y"])

        # Error: target - sim_rbf (if sim_rbf is used)
        if use_sim_rbf:
            df["residual_sim_x"] = df["target_x"] - df["pred_sim_rbf_multiquadric_s1.0_x"]
            df["residual_sim_y"] = df["target_y"] - df["pred_sim_rbf_multiquadric_s1.0_y"]
            feature_cols.extend(["residual_sim_x", "residual_sim_y"])

    # Create user ID mapping
    unique_users = sorted(df["subject"].unique())
    user_to_id = {name: idx for idx, name in enumerate(unique_users)}

    # Group by session (subject + timestamp)
    sessions = []
    grouped = df.groupby(["subject", "timestamp"])

    for (subject, timestamp), group in grouped:
        if len(group) < min_points_per_session:
            continue

        # Extract features
        features = group[feature_cols].values.astype(np.float32)

        # Handle any NaN values
        if np.any(np.isnan(features)):
            # Fill NaN with column mean
            col_means = np.nanmean(features, axis=0)
            nan_mask = np.isnan(features)
            features[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        sessions.append(
            SessionData(
                user_id=user_to_id[subject],
                user_name=subject,
                timestamp=timestamp,
                features=features,
                num_points=len(group),
            )
        )

    num_features = len(feature_cols)
    return sessions, user_to_id, num_features


def split_sessions(
    sessions: List[SessionData],
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    min_sessions_per_user: int = 2,
    seed: int = 42,
) -> Tuple[List[SessionData], List[SessionData], List[SessionData], Dict[str, int]]:
    """
    Split sessions into train/val/test sets, ensuring each user has sessions in each split.

    Args:
        sessions: All sessions
        test_ratio: Fraction for test set
        val_ratio: Fraction for validation set
        min_sessions_per_user: Minimum sessions per user to be included
        seed: Random seed

    Returns:
        train_sessions, val_sessions, test_sessions, filtered_user_to_id
    """
    np.random.seed(seed)

    # Group sessions by user
    user_sessions: Dict[int, List[SessionData]] = {}
    for session in sessions:
        if session.user_id not in user_sessions:
            user_sessions[session.user_id] = []
        user_sessions[session.user_id].append(session)

    # Filter users with enough sessions
    valid_users = {
        uid: sess
        for uid, sess in user_sessions.items()
        if len(sess) >= min_sessions_per_user
    }

    if len(valid_users) == 0:
        raise ValueError(
            f"No users have >= {min_sessions_per_user} sessions. "
            f"Session counts: {[(s.user_name, len(user_sessions.get(s.user_id, []))) for s in sessions[:5]]}"
        )

    # Create new user ID mapping for filtered users
    old_to_new_id = {}
    new_user_to_id = {}
    for new_id, (old_id, sess_list) in enumerate(sorted(valid_users.items())):
        old_to_new_id[old_id] = new_id
        new_user_to_id[sess_list[0].user_name] = new_id

    # Split each user's sessions
    train_sessions = []
    val_sessions = []
    test_sessions = []

    for old_uid, sess_list in valid_users.items():
        new_uid = old_to_new_id[old_uid]

        # Update user IDs
        for s in sess_list:
            s.user_id = new_uid

        n = len(sess_list)
        if n < 3:
            # Not enough for 3-way split, put all in train
            train_sessions.extend(sess_list)
            continue

        # Shuffle sessions
        indices = np.random.permutation(n)
        shuffled = [sess_list[i] for i in indices]

        # Calculate split sizes
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_test - n_val

        if n_train < 1:
            n_train = 1
            n_val = max(0, n - n_train - n_test)

        train_sessions.extend(shuffled[:n_train])
        val_sessions.extend(shuffled[n_train : n_train + n_val])
        test_sessions.extend(shuffled[n_train + n_val :])

    return train_sessions, val_sessions, test_sessions, new_user_to_id


def create_datasets(
    csv_path: str | Path,
    min_sessions_per_user: int = 2,
    min_points_per_session: int = 10,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    max_points: int = 24,
    seed: int = 42,
    **feature_flags,
) -> Tuple[GazeIdentityDataset, GazeIdentityDataset, GazeIdentityDataset, Dict[str, int], int]:
    """
    Create train/val/test datasets from CSV file.

    Returns:
        train_dataset, val_dataset, test_dataset, user_to_id, num_features
    """
    # Load all sessions
    sessions, user_to_id, num_features = load_sessions_from_csv(
        csv_path,
        min_points_per_session=min_points_per_session,
        **feature_flags,
    )

    print(f"Loaded {len(sessions)} sessions from {len(user_to_id)} users")

    # Split sessions
    train_sessions, val_sessions, test_sessions, filtered_user_to_id = split_sessions(
        sessions,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        min_sessions_per_user=min_sessions_per_user,
        seed=seed,
    )

    print(f"After filtering: {len(filtered_user_to_id)} users")
    print(f"Split: {len(train_sessions)} train, {len(val_sessions)} val, {len(test_sessions)} test")

    # Create datasets with shared normalization stats
    train_dataset = GazeIdentityDataset(
        train_sessions, max_points=max_points, normalize=True
    )

    val_dataset = GazeIdentityDataset(
        val_sessions,
        max_points=max_points,
        normalize=True,
        feature_stats=train_dataset.feature_stats,
    )

    test_dataset = GazeIdentityDataset(
        test_sessions,
        max_points=max_points,
        normalize=True,
        feature_stats=train_dataset.feature_stats,
    )

    return train_dataset, val_dataset, test_dataset, filtered_user_to_id, num_features
