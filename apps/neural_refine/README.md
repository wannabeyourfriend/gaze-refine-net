# Neural Refine

Deep learning module for refining gaze point predictions from traditional calibration models.

## Overview

This module implements a neural network approach to further improve gaze estimation accuracy beyond traditional calibration methods. It takes the output of model-based calibration (polynomial or RBF) and applies learned corrections to reduce systematic errors.

## Architecture

The neural refinement model uses a ResNet-based architecture that:
- Takes as input: original gaze coordinates, target coordinates, and model-based predictions
- Learns residual corrections to the model-based predictions
- Outputs refined gaze coordinates with improved accuracy

## Project Structure

```
neural_refine/
├── config/               # Configuration files
│   ├── default.yaml     # Base configuration
│   └── cascade.yaml     # Cascade training configuration
├── src/                 # Source code
│   └── model.py        # Model definitions and dataset classes
├── main.py             # Training and evaluation script
├── pyproject.toml      # Project dependencies
└── uv.lock            # Locked dependencies
```

## Installation

```bash
uv sync
source .venv/bin/activate
```

## Usage

### Training

Train from original gaze data:
```bash
python main.py --checkpoint ../../checkpoints/epoch_0100.pt --config config/default.yaml
```

Train from model based calibration sim_rbf baseline:
```bash
python main.py --config config/cascade.yaml
```



### Configuration

Configuration files are located in the `config/` directory. Key parameters include:
- **model**: Model architecture settings (type, hidden dimensions, etc.)
- **data**: Dataset paths and preprocessing parameters
- **training**: Learning rate, batch size, epochs, optimizer settings
- **coordinate_scale**: Normalization factor for gaze coordinates

## Output

Training produces:
- Model checkpoints in `../../checkpoints/`
- Prediction files in `../../outputs/`
- Training logs and metrics (if using tensorboard/wandb)
