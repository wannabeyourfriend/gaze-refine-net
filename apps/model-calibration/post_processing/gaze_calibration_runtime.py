import numpy as np
import pandas as pd
import sys
from pathlib import Path

# ====== 直接复用你文件里的代码 ======
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from post_processing.calibration_model_full_compare import (
    fit_similarity,
    apply_similarity,
    fit_rbf_residual,
)

class SimRBFCalibrator:
    """
    Similarity + RBF (multiquadric, smooth=1.0)
    用于实时 gaze 校正
    """

    def __init__(self, origin_csv_path, rbf_kernel="multiquadric", smooth=1.0):
        df = pd.read_csv(origin_csv_path)

        self.origin_obs = df[['original_gaze_x','original_gaze_y']].values
        self.origin_tgt = df[['target_x','target_y']].values

        # --- similarity ---
        self.s, self.R, self.t = fit_similarity(
            self.origin_obs,
            self.origin_tgt
        )

        origin_sim = apply_similarity(
            self.origin_obs,
            self.s, self.R, self.t
        )

        # --- RBF residual ---
        self.rbf_x, self.rbf_y = fit_rbf_residual(
            self.origin_obs,
            origin_sim,
            self.origin_tgt,
            kernel=rbf_kernel,
            smooth=smooth
        )

    def correct(self, x, y):
        """校正单个 gaze 点"""
        obs = np.array([[x, y]], dtype=float)

        sim_xy = apply_similarity(obs, self.s, self.R, self.t)
        rx = self.rbf_x(obs[:,0], obs[:,1])
        ry = self.rbf_y(obs[:,0], obs[:,1])

        out = sim_xy + np.stack([rx, ry], axis=1)
        return float(out[0,0]), float(out[0,1])
