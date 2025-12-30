# Model Calibration

This module provides comprehensive calibration infrastructure for improving eye tracking accuracy. It implements systematic data collection, multiple calibration models, and post-processing utilities for gaze point refinement.

## Overview

The model calibration system performs 18-point calibration data collection and 32-point testing to evaluate and improve gaze estimation accuracy. It supports multiple calibration approaches including polynomial fitting, Radial Basis Functions (RBF), and neural network refinement.

## Project Structure

```
model_calibration/
├── apriltags/                            # AprilTag markers for screen positioning
├── systematic_drift_calibration.py       # Main calibration data collection script
├── post_processing/                      # Data analysis and model utilities
│   ├── 2s_reaction.py                    # Reaction time analysis
│   ├── batch_model_evaluation.py         # Statistical model comparison
│   ├── batch_run_all_sessions.py         # Batch processing pipeline
│   ├── build_grid_gaze_log_partial.py    # Partial data extraction
│   ├── calibration_model_full_compare.py # Model comparison framework
│   └── gaze_calibration_runtime.py       # Runtime calibration for demo
└── environment.yaml                      # Conda environment specification
```

## Components

### apriltags/
- **Purpose**: Contains AprilTag markers for screen coordinate system calibration
- **Usage**: Place AprilTag markers at known screen positions to establish accurate coordinate mapping
- **Features**: 
  - Enables precise screen position detection
  - Supports automatic coordinate system alignment
  - Required for accurate spatial calibration

### systematic_drift_calibration.py
- **Purpose**: Main script for systematic calibration data collection
- **Features**:
  - **18-point calibration**: Collects calibration data at 18 grid positions
  - **32-point testing**: Validates calibration with 32 test positions
  - **6×4 grid layout**: Systematic coverage of screen space
  - **2-second wait + 4-second sampling**: Each point lights up, waits 2s for eye movement, then samples 4s of gaze data
  - **Outlier filtering**: Removes samples with >250px distance from target
  - **Two-pass collection**: 
    - First pass: Collect calibration data and save to `samples_target_*.csv` and `grid_gaze_log.csv`
    - Creates "origin" subfolder with first-pass results
    - Second pass: User-triggered collection for validation
- **Output**: 
  - CSV files with timestamps, gaze coordinates, and target positions
  - Session-organized directory structure

### post_processing/

#### 2s_reaction.py
- **Purpose**: Analyzes reaction time and convergence speed
- **Features**:
  - Measures time for gaze to stabilize after target appears
  - Calculates 3σ (three standard deviations) convergence threshold
  - Statistics on time to reach final predicted position vicinity
  - Useful for understanding visual attention latency

#### batch_model_evaluation.py
- **Purpose**: Statistical evaluation and comparison of calibration models
- **Features**:
  - Paired t-test implementation for model comparison
  - Batch processing across multiple sessions
  - Statistical significance testing
  - Performance metric aggregation

#### batch_run_all_sessions.py
- **Purpose**: Batch processing pipeline for model fitting and data preparation
- **Features**:
  - Calls `calibration_model_full_compare` for batch model fitting
  - Processes multiple calibration sessions automatically
  - Generates `all_trials_model_predictions` files
  - Stacks model-corrected predictions with test data for deep learning training
  - Validates optimal sampling duration (found 4s to be best)
  - Tests multiple time windows (1s, 2s, 3s, 4s) to demonstrate model generalizability
  - Modifies input from `grid_gaze_log` for different time windows

#### build_grid_gaze_log_partial.py
- **Purpose**: Extracts partial time-window data for temporal analysis
- **Features**:
  - Creates new `grid_gaze_log_ns` files with first 1s, 2s, 3s of data
  - Tests whether 4-second sampling produces best results
  - Enables comparison of different sampling durations
  - Processes data per trial from same session folder

#### calibration_model_full_compare.py
- **Purpose**: Central model comparison framework
- **Features**:
  - Implements multiple calibration models:
    - **Polynomial calibration**: 2nd-order polynomial fitting
    - **simRBF**: Simplified Radial Basis Function
    - **simRBF + Neural refinement**: RBF with ResNet correction
  - Processes individual trials
  - Provides unified interface for model evaluation
  - Compares model performance on same data

#### gaze_calibration_runtime.py
- **Purpose**: Real-time calibration service for demo game application
- **Features**:
  - Provides runtime calibration for `demo_game`
  - Calls `calibration_model_full_compare` to load models
  - Serves three model types:
    - **PolynomialCalibrator**: Polynomial fitting model
    - **SimRBFCalibrator**: RBF-based calibration
    - **SimRBFWithNeuralCascadeCalibrator**: Neural-refined model
  - Low-latency prediction for interactive applications

## Usage

### Collecting Calibration Data

Run the systematic calibration procedure:
```bash
cd apps/model_calibration
python systematic_drift_calibration.py
```

**Calibration Procedure:**
1. First pass: System displays 6×4 grid of targets
2. For each target:
   - Target lights up
   - Wait 2 seconds for eye movement
   - Sample gaze data for 4 seconds
   - Remove outliers (>250px from target)
3. Data saved to session directory
4. Original data copied to "origin" subfolder
5. Press Enter to start second pass for validation

### Processing Collected Data

#### Batch Process All Sessions
```bash
cd apps/model_calibration/post_processing
python batch_run_all_sessions.py
```

This will:
- Fit all calibration models to each session
- Generate comparison statistics
- Create training data for neural refinement
- Validate 4-second sampling optimality

#### Compare Models on Single Trial
```bash
python calibration_model_full_compare.py --session <session_path>
```

#### Analyze Reaction Times
```bash
python 2s_reaction.py --data <data_path>
```

#### Statistical Evaluation
```bash
python batch_model_evaluation.py --sessions <sessions_dir>
```

### Using Calibration at Runtime

For integrating calibration into applications (like demo_game):
```python
from post_processing.gaze_calibration_runtime import (
    PolynomialCalibrator,
    SimRBFCalibrator,
    SimRBFWithNeuralCascadeCalibrator
)

# Initialize calibrator
calibrator = SimRBFWithNeuralCascadeCalibrator(calibration_data_path)

# Calibrate gaze point
corrected_x, corrected_y = calibrator.calibrate(raw_x, raw_y)
```

## Calibration Models

### 1. Polynomial Calibration
- **Type**: Parametric model
- **Method**: 2nd-order polynomial surface fitting
- **Pros**: Fast, simple, generalizes well
- **Cons**: Limited accuracy for non-linear distortions
- **Best for**: Uniform distortion patterns

### 2. simRBF (Simplified Radial Basis Function)
- **Type**: Non-parametric interpolation
- **Method**: RBF kernel-based warping
- **Pros**: Better handles local variations, higher accuracy than polynomial
- **Cons**: Requires more calibration points, slower than polynomial
- **Best for**: Complex, spatially-varying distortions

### 3. simRBF + Neural Refinement
- **Type**: Hybrid model (RBF + deep learning)
- **Method**: RBF baseline with ResNet residual correction
- **Pros**: Highest accuracy, learns systematic errors
- **Cons**: Requires training data, computational overhead
- **Best for**: Applications requiring maximum accuracy

## Data Format

Calibration data files (`grid_gaze_log.csv`, `samples_target_*.csv`) contain:
- **Timestamp**: Millisecond-precision timing
- **target_x, target_y**: Ground truth target coordinates (pixels)
- **gaze_x, gaze_y**: Raw gaze coordinates from eye tracker (pixels)
- **Session ID**: Identifier for calibration session
- **Trial number**: Sequential trial index

## Configuration

Key parameters in `systematic_drift_calibration.py`:
- **GRID_SIZE**: 6×4 grid (24 total points for systematic coverage)
- **WAIT_TIME**: 2 seconds (allows eye movement to target)
- **SAMPLE_TIME**: 4 seconds (optimal for model fitting, validated empirically)
- **OUTLIER_THRESHOLD**: 250 pixels (removes poor fixations)
- **SCREEN_RESOLUTION**: Configurable for different displays

## Output Files

After calibration:
```
session_YYYYMMDD_HHMMSS/
├── origin/                              # First-pass data backup
│   ├── grid_gaze_log.csv               # All calibration data
│   └── samples_target_*.csv            # Per-target samples
├── grid_gaze_log.csv                   # Latest calibration data
├── samples_target_*.csv                # Per-target samples
└── all_trials_model_predictions.csv    # Model comparison results
```
