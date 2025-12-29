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
cd apps/neural_refine
pip install -e .
```

Or using uv:
```bash
uv sync
```

## Usage

### Training

Train from sim_rbf baseline:
```bash
python main.py --config config/cascade.yaml
```

Train from original gaze data:
```bash
python main.py --checkpoint ../../checkpoints/epoch_0100.pt --config config/default.yaml
```

### Configuration

Configuration files are located in the `config/` directory. Key parameters include:
- **model**: Model architecture settings (type, hidden dimensions, etc.)
- **data**: Dataset paths and preprocessing parameters
- **training**: Learning rate, batch size, epochs, optimizer settings
- **coordinate_scale**: Normalization factor for gaze coordinates

### Evaluation

After training, evaluate the model using the root-level evaluation script:

```bash
cd ../..  # Return to project root

# Evaluate initial gaze error
python scripts/eval.py original data/prepared/split_avg_data/test.csv

# Evaluate sim RBF baseline
python scripts/eval.py sim_rbf data/prepared/split_avg_data/test.csv

# Evaluate refined predictions
python scripts/eval.py pred_gaze outputs/predictions_test.csv
```

## Model Types

The system supports multiple model architectures:
- **resnet**: ResNet-based refinement network (recommended)
- **mlp**: Multi-layer perceptron baseline
- **cascade**: Multi-stage refinement with intermediate outputs

## Training Pipeline

1. **Data Loading**: Loads preprocessed CSV files with gaze coordinates
2. **Model Initialization**: Builds the neural network based on configuration
3. **Training Loop**: 
   - Forward pass through the network
   - Loss computation (typically MSE between predicted and target coordinates)
   - Backpropagation and optimization
   - Validation on held-out data
4. **Checkpointing**: Saves model weights periodically
5. **Prediction**: Generates refined gaze predictions on test data

## Output

Training produces:
- Model checkpoints in `../../checkpoints/`
- Prediction files in `../../outputs/`
- Training logs and metrics (if using tensorboard/wandb)

## Dependencies

Key dependencies (see `pyproject.toml` for complete list):
- **torch** >= 2.9.1: Deep learning framework
- **pandas** >= 2.3.3: Data manipulation
- **numpy** >= 2.3.5: Numerical computations
- **pyyaml** >= 6.0.3: Configuration file parsing
- **matplotlib** / **seaborn**: Visualization
- **tensorboard** / **wandb**: Experiment tracking (optional)

## Performance

The neural refinement approach typically achieves:
- **Baseline (original)**: Higher gaze estimation error
- **Model-based (RBF/polynomial)**: Moderate improvement
- **Neural refined**: Best performance with learned corrections

Exact improvements depend on the quality of calibration data and target accuracy requirements.

## Tips

- Start with the cascade configuration for best results
- Use coordinate_scale parameter to normalize gaze coordinates to [-1, 1] range
- Monitor both training and validation loss to detect overfitting
- The 4-second aggregation window for calibration data has been found optimal
- Experiment with different model architectures if the default doesn't meet your needs

## Troubleshooting

**Out of memory errors**: Reduce batch size in configuration file

**Poor convergence**: Adjust learning rate or increase training epochs

**Overfitting**: Add dropout layers or reduce model complexity

**Data format issues**: Ensure CSV files have the expected columns and coordinate ranges
