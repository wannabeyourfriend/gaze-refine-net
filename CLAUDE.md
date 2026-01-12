# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NRMBC (Neural Refined Model-Based Gaze Point Calibration) is a Python eye-tracking calibration system that combines traditional model-based approaches (polynomial fitting, RBF) with deep learning refinement. Designed for Pupil Labs eye-tracking hardware.

## Build & Run Commands

### Installation

```bash
# neural_refine module (uv)
cd apps/neural_refine
uv sync
source .venv/bin/activate

# model_calibration module (Conda)
conda env create -f apps/model_calibration/environment.yaml
conda activate gazetoword
```

### Training

```bash
cd apps/neural_refine

# End-to-end training (original gaze -> residual)
python main.py --config config/end_to_end.yaml

# Cascade training (simRBF baseline -> neural refinement)
python main.py --config config/cascade.yaml
```

### Evaluation

```bash
# Evaluate original gaze error
python scripts/eval.py original data/prepared/split_avg_data/test.csv

# Evaluate simRBF calibration
python scripts/eval.py sim_rbf data/prepared/split_avg_data/test.csv

# Evaluate neural-refined predictions
python scripts/eval.py pred_gaze outputs/predictions_test.csv
```

### Data Collection & Demo

```bash
# Calibration data collection
python apps/model_calibration/systematic_drift_calibration.py

# Demo game
python apps/demo_game/run_demo_game.py
```

## Architecture

### Calibration Pipeline

1. **Data Collection**: Pupil Labs tracker → ZMQ/msgpack → `systematic_drift_calibration.py` → grid_gaze_log.csv (24-point calibration)
2. **Preprocessing**: Raw data → `data_preprocess/scripts/split.py` → train/val/test.csv
3. **Training**: Split data → `neural_refine/main.py` → checkpoints/*.pt

### Calibration Models (cascading accuracy)

- **Similarity Transform**: Global Procrustes alignment
- **Polynomial Calibrator**: 2nd-order polynomial surface fitting
- **SimRBF**: Similarity + RBF residual interpolation (multiquadric kernel)
- **SimRBF + Neural Cascade**: RBF baseline + ResNet residual correction (highest accuracy)

### Neural Model

`GazeRefineNet` in `apps/neural_refine/src/model.py`:
- MLP with optional residual blocks
- Input: 2D (end_to_end) or 4D (cascade: original + simRBF coordinates)
- Output: 2D residual (dx, dy)
- Coordinates normalized by `coordinate_scale=100.0`

### Key Modules

| Directory | Purpose |
|-----------|---------|
| `apps/neural_refine/` | Neural network training/inference (PyTorch, uv) |
| `apps/model_calibration/` | Calibration data collection & traditional models (PyQt6, Conda) |
| `apps/demo_game/` | Interactive gaze validation game |
| `apps/data_preprocess/` | Data cleaning and train/val/test splitting |
| `checkpoints/` | Model weights (`cascade_sim_rbf/best_model.pt`) |

## Data Formats

**Calibration CSV columns:**
- `target_x`, `target_y`: Ground truth coordinates (pixels)
- `original_gaze_x`, `original_gaze_y`: Raw eye tracker gaze
- `sim_rbf_gaze_x`, `sim_rbf_gaze_y`: SimRBF-corrected gaze (cascade mode)
- `spread`: Sample standard deviation (for weighting)

## Development Notes

- **Residual learning**: Neural models predict residuals (target - baseline), not absolute coordinates
- **Cross-module imports**: Files add parent directories to `sys.path`
- **ZMQ fallback**: Eye tracker communication falls back to mouse input if unavailable
- **Python versions**: neural_refine requires 3.11+, model_calibration uses 3.10
