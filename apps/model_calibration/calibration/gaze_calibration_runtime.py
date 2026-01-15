import numpy as np
import pandas as pd
import sys
from pathlib import Path
import torch
import yaml
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
# ====== Reuse code directly from existing modules ======
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
NEURAL_REFINE_ROOT = REPO_ROOT / "apps" / "neural_refine"

for p in [PROJECT_ROOT, REPO_ROOT, NEURAL_REFINE_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from model_calibration.models.calibration_model_full_compare import (
    fit_similarity,
    apply_similarity,
    fit_rbf_residual,
)
from neural_refine.src.model import build_model

class SimRBFCalibrator:
    """
    Similarity + RBF (multiquadric, smooth=1.0)
    Used for real-time gaze correction.
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
        """Correct a single gaze point."""
        obs = np.array([[x, y]], dtype=float)

        sim_xy = apply_similarity(obs, self.s, self.R, self.t)
        rx = self.rbf_x(obs[:,0], obs[:,1])
        ry = self.rbf_y(obs[:,0], obs[:,1])

        out = sim_xy + np.stack([rx, ry], axis=1)
        return float(out[0,0]), float(out[0,1])


class CascadeNeuralRefiner:
    """
    Use neural_refine (cascade mode) to refine RBF outputs.
    Predicts residuals as target - sim_rbf_gaze.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.config_path = (
            Path(config_path).resolve()
            if config_path is not None
            else REPO_ROOT / "apps" / "neural_refine" / "config" / "cascade.yaml"
        )

        with self.config_path.open("r") as f:
            cfg = yaml.safe_load(f)

        self.coordinate_scale = cfg["model"].get("coordinate_scale", 100.0)
        self.model = build_model(cfg["model"]).to(self.device)
        state = torch.load(self.checkpoint_path, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def refine(self, orig_x: float, orig_y: float, sim_x: float, sim_y: float):
        scale = self.coordinate_scale
        inp = torch.tensor(
            [[orig_x / scale, orig_y / scale, sim_x / scale, sim_y / scale]],
            dtype=torch.float32,
            device=self.device,
        )
        pred_res = self.model(inp)[0].cpu().numpy()
        pred_res_px = pred_res * scale
        refined_x = sim_x + float(pred_res_px[0])
        refined_y = sim_y + float(pred_res_px[1])
        return refined_x, refined_y, float(pred_res_px[0]), float(pred_res_px[1])


class SimRBFWithNeuralCascadeCalibrator(SimRBFCalibrator):
    """
    Apply similarity + RBF, then refine residuals with neural_refine (cascade).
    """

    def __init__(
        self,
        origin_csv_path,
        checkpoint_path: Path,
        config_path: Path | None = None,
        device: str = "cpu",
        rbf_kernel="multiquadric",
        smooth=1.0,
    ):
        super().__init__(origin_csv_path, rbf_kernel=rbf_kernel, smooth=smooth)
        self.refiner = CascadeNeuralRefiner(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device,
        )

    def correct(self, x, y):
        sim_x, sim_y = super().correct(x, y)
        refined_x, refined_y, _, _ = self.refiner.refine(x, y, sim_x, sim_y)
        return refined_x, refined_y

class PolynomialCalibrator:
    """
    Quadratic polynomial calibrator (based on SimplePolynomialCalibrator).
    """

    def __init__(self, origin_csv_path, degree=2, reg=0.5, use_weight=True):
        """
        Args:
        - origin_csv_path: path to calibration CSV data
        - degree: polynomial degree
        - reg: regularization strength
        - use_weight: whether to use the spread column as weights
        """
        df = pd.read_csv(origin_csv_path)
        
        # Extract training data
        X = df[['target_x', 'target_y']].values
        self.degree = degree
        
        # Polynomial feature transform
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        Xp = self.poly.fit_transform(X)
        
        # Compute residuals
        dx = df['original_gaze_x'].values - df['target_x'].values
        dy = df['original_gaze_y'].values - df['target_y'].values
        
        # Handle weights
        if use_weight and 'spread' in df.columns:
            eps = 1e-6
            w = 1.0 / (df['spread'].values + eps)**2
            w = w / np.max(w)
        else:
            w = None
        
        # Train models
        self.model_dx = Ridge(alpha=reg).fit(Xp, dx, sample_weight=w)
        self.model_dy = Ridge(alpha=reg).fit(Xp, dy, sample_weight=w)
        
        # Keep original data available for debugging
        self.df = df

    def predict_delta(self, x_arr):
        """Predict residual offsets."""
        Xp = self.poly.transform(x_arr)
        dx = self.model_dx.predict(Xp)
        dy = self.model_dy.predict(Xp)
        return np.vstack([dx, dy]).T

    def correct(self, x, y):
        """Correct a single gaze point (iterative)."""
        T = np.array([x, y], dtype=float)
        G = np.array([x, y], dtype=float)
        
        # Iterative refinement
        for _ in range(20):
            d = self.predict_delta(T.reshape(1, 2))[0]
            F = T + d - G
            step = -0.7 * F
            T += step
            if np.linalg.norm(step) < 0.5:
                break
                
        return float(T[0]), float(T[1])
    
    def correct_batch(self, gaze_points):
        """Correct a batch of gaze points."""
        out = []
        for gx, gy in gaze_points:
            corrected = self.correct(gx, gy)
            out.append(corrected)
        return np.array(out)
