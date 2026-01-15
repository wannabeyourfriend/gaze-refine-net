# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NRMBC (Neural Refined Model-Based Gaze Point Calibration) is a Python eye-tracking calibration system that combines traditional model-based approaches (polynomial fitting, RBF) with deep learning refinement. Designed for Pupil Labs eye-tracking hardware.

## Build & Run Commands

### Installation

```bash
# neural_refine module (uv, Python 3.11+)
cd apps/neural_refine
uv sync
source .venv/bin/activate

# model_calibration module (Conda, Python 3.10)
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
# Calibration data collection (24-point grid)
python apps/model_calibration/systematic_drift_calibration.py

# Demo game for validation
python apps/demo_game/run_demo_game.py
```

### Analysis

```bash
# Run all analysis scripts
python apps/neural_refine/analysis/run_all.py

# Individual analyses in apps/neural_refine/analysis/:
# 01_residual_correlation_analysis.py - Spatial correlation
# 02_sequence_feature_analysis.py - Temporal features
# 03_session_bias_analysis.py - Session/subject bias
# 04_online_adaptation_analysis.py - Calibration point experiments
```

## Architecture

### Calibration Pipeline

1. **Data Collection**: Pupil Labs tracker → ZMQ/msgpack → `systematic_drift_calibration.py` → grid_gaze_log.csv
2. **Preprocessing**: Raw data → `data_preprocess/scripts/split.py` → train/val/test.csv
3. **Training**: Split data → `neural_refine/main.py` → checkpoints/*.pt

### Calibration Models (cascading accuracy)

- **Similarity Transform**: Global Procrustes alignment
- **Polynomial Calibrator**: 2nd-order polynomial surface fitting
- **SimRBF**: Similarity + RBF residual interpolation (multiquadric kernel)
- **SimRBF + Neural Cascade**: RBF baseline + ResNet residual correction

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
| `apps/data_process/` | Data cleaning and train/val/test splitting |
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
- **No unit tests**: Validation is done through evaluation scripts and demo game

---

## Research Approach: Calibration Quality-Aware Neural Refinement

### Problem Context

Few-shot (18-point) SimRBF calibration has high variance across sessions. Analysis showed:
- SimRBF residuals are spatially uncorrelated (r ≈ 0) - no learnable spatial pattern
- Session-specific bias is the main error source (~20px std)
- Cross-session bias prediction fails

### Solution: SimRBF Perturbation Augmentation

Instead of learning spatial patterns, we train the network to **sense calibration quality** and output appropriate corrections.

**Key insight**: The relationship between `(original_gaze, sim_rbf_gaze)` contains information about calibration quality:
- Good calibration → large, consistent correction from original to sim_rbf
- Poor calibration → small or inconsistent correction

**Training approach**:
```python
# Perturb sim_rbf to simulate varying calibration quality
perturbed_sim_rbf = sim_rbf_gaze + noise + bias

# Network learns to predict residual for the perturbed calibration
Input:  (original_gaze_x, original_gaze_y, perturbed_sim_rbf_x, perturbed_sim_rbf_y)
Output: residual (dx, dy)
Target: target_point - perturbed_sim_rbf
```

**Configuration** (`config/cascade.yaml`):
```yaml
sim_rbf_perturbation:
  enabled: true
  noise_std: 20.0    # Gaussian noise (SimRBF variance ~20px)
  bias_range: 30.0   # Uniform bias (session bias ~20-30px)
  prob: 0.5          # Apply to 50% of samples
```

### Alternative: Online Bias Correction

For simpler deployments, use 8-12 calibration points at session start:

```python
bias_x = mean(target_x - sim_rbf_x)
bias_y = mean(target_y - sim_rbf_y)
corrected = sim_rbf + bias
```

| Calibration Points | Error | Improvement |
|-------------------|-------|-------------|
| 0 (baseline) | 48.49 px | 0% |
| 8 | 45.09 px | ~5% |
| 12 | 43.71 px | ~7% |

**Full analysis**: `outputs/analysis/ANALYSIS_REPORT.md`


