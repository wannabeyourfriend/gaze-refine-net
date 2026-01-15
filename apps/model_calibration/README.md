# Model-Based Gaze Calibration Module

Traditional model-based gaze calibration methods and data collection tools for Pupil Labs eye trackers.

## Directory Structure

```
model_calibration/
├── models/                      # Calibration model implementations
│   └── calibration_model_full_compare.py  # All calibration algorithms
├── calibration/                 # Runtime calibration tools
│   └── gaze_calibration_runtime.py        # Real-time SimRBF calibration
├── analysis/                    # Batch evaluation and analysis
│   ├── batch_run_all_sessions.py          # Process multiple sessions
│   └── batch_model_evaluation.py          # Compare model performance
├── scripts/                     # Utility scripts
│   ├── reaction_time_analysis.py          # 2-second reaction analysis
│   └── build_grid_gaze_log_partial.py     # Partial grid log builder
├── apriltags/                   # AprilTag detection assets
├── systematic_drift_calibration.py        # Main data collection script
├── environment.yaml             # Conda environment specifications
└── pyproject.toml              # Project metadata
```

## Installation

```bash
# Create conda environment
conda env create -f environment.yaml
conda activate gazetoword
```

## Usage

### Data Collection

Collect calibration data using a 24-point grid:

```bash
python systematic_drift_calibration.py
```

This creates a structured output:
```
session_dir/
├── origin/
│   ├── grid_gaze_log.csv          # Calibration data
│   └── samples_target_*.csv        # Raw samples per target
└── test/
    ├── grid_gaze_log.csv          # Test data
    └── samples_target_*.csv        # Test samples
```

### Runtime Calibration

Apply SimRBF calibration in real-time:

```python
from model_calibration.calibration.gaze_calibration_runtime import SimRBFCalibrator

calibrator = SimRBFCalibrator("origin/grid_gaze_log.csv")
corrected_gaze = calibrator.correct(raw_gaze_x, raw_gaze_y)
```

### Batch Analysis

Process multiple sessions:

```bash
# Generate predictions for all sessions
python analysis/batch_run_all_sessions.py

# Evaluate model performance
python analysis/batch_model_evaluation.py
```

## Calibration Models

The module implements a cascade of calibration methods:

1. **Similarity Transform**: Global Procrustes alignment (scale + rotation + translation)
2. **Polynomial**: 2nd-order polynomial surface fitting
3. **SimRBF**: Similarity + RBF residual interpolation
   - Kernel: multiquadric
   - Smooth: 1.0 (default)
4. **Advanced variants**:
   - SimTPS: Similarity + Thin Plate Spline
   - SimGPR: Similarity + Gaussian Process Regression
   - SimPWA: Similarity + Piecewise Affine

### Model Performance

Typical error (pixels) on 24-point calibration:

| Method | Mean Error | Description |
|--------|-----------|-------------|
| Original | ~100 px | Raw eye tracker output |
| Similarity | ~60 px | Global alignment |
| Polynomial | ~45 px | Surface fitting |
| SimRBF | ~35 px | Recommended baseline |

## File Formats

### grid_gaze_log.csv

Columns:
- `original_gaze_x`, `original_gaze_y`: Raw gaze coordinates
- `target_x`, `target_y`: Ground truth calibration points
- `timestamp`: Sample timestamp
- `confidence`: Pupil Labs detection confidence

### Output CSV (batch_run_all_sessions.py)

Contains predictions from all calibration methods:
- `origin_gaze_x`, `origin_gaze_y`: Original gaze
- `pred_similarity_x`, `pred_similarity_y`: Similarity transform
- `pred_poly_x`, `pred_poly_y`: Polynomial calibration
- `pred_sim_rbf_*_x`, `pred_sim_rbf_*_y`: SimRBF variants
- `target_x`, `target_y`: Ground truth

## Requirements

- Python 3.10
- PyQt6 (GUI)
- numpy, pandas, scikit-learn
- Pupil Labs eye tracker hardware

## Development Notes

- All models follow scikit-learn API conventions (`fit`, `predict`, `transform`)
- RBF smooth parameter controls regularization (higher = smoother)
- Default RBF kernel is multiquadric for gaze calibration
- AprilTag images are for 36h11 tag detection family
