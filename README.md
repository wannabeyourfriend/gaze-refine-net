# Drift Calibration in Geometric Based Eye Tracking Systems

<p align="center">
<img src="assets/figures/pipeline.png" width="800"/>
</p>

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hardware](https://img.shields.io/badge/Hardware-Pupil_Labs-blue)](https://pupil-labs.com/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)

Head-mounted eye trackers are calibrated per session, and a classical calibrator suit ( similarity/affine, quadratic polynomial, RBF, thin-plate spline, piecewise-affine, GPR ) — is fitted to that session's calibration fixations. This repository adds a learned layer on top of that geometry, and the tooling to collect the data it trains on.

Four stages, each usable on its own:

- **collect** — an 18-point calibration grid and a scattered-target protocol for a Pupil Labs Core tracker (`apps/model_calibration/`)
- **fit** — 16 classical per-session calibrators, fitted from raw gaze→target pairs (`scripts/process_data.py`)
- **refine** — a small MLP that takes several calibrators' predictions as hypotheses and outputs a residual correction, with top-M hypothesis selection and Gaussian noise augmentation (`apps/neural_refine/`)
- **validate** — a gaze-controlled rhythm game that measures whether calibration accuracy translates into task performance (`apps/demo_game/`)

## Setup

```bash
# refinement training (Python 3.11+)
cd apps/neural_refine && uv sync && source .venv/bin/activate

# data collection and the demo game (Conda, Python 3.10, PyQt6 + Pupil Labs)
conda env create -f apps/model_calibration/environment.yaml
conda activate gazetoword
```

Optional, for experiment tracking and dataset upload — copy `.env.example` to `.env` and fill in what you use:

```
WANDB_API_KEY=      # experiment logging; set WANDB_MODE=offline to skip
HF_TOKEN=           # only needed to push/pull datasets or checkpoints
```

**Config paths are relative to `apps/neural_refine/`.** Every config refers to data as `../../data/...` and checkpoints as `../../checkpoints/...`, so training must be launched from that directory, not from the repo root.

## run

### 1. Train the refiner on the shipped self-collected data

```bash
cd apps/neural_refine
python main.py --config config/multi_baseline_s1.yaml
```

### 2. Collect new calibration data

```bash
conda activate gazetoword
python apps/model_calibration/systematic_drift_calibration.py
python scripts/process_data.py        # raw logs -> per-session baselines
```

### 3. Run the validation game

```bash
python apps/demo_game/run_demo_game.py
```

## Dataset

Self-collected: 12 participants on a Pupil Labs Core (monocular, right eye), a 24-inch 1920×1080 display at 70 cm. Distributed in `data/`, pseudonymised.

## Hardware

- Pupil Labs Core eye tracker (monocular)
- 24-inch display, 1920×1080, fixed at 70 cm

## License

MIT — see [LICENSE](LICENSE).
