# NRMBC: Neural Refined Model-Based Gaze Point Calibration

A eye gaze point calibration system that combines traditional model-based approaches with deep learning refinement for improved eye-tracking accuracy.

## Overview

This project implements a hybrid approach to gaze point calibration that:
1. Uses traditional calibration models (polynomial fitting, RBF) for initial correction
2. Applies neural network refinement to further reduce gaze estimation errors
3. Provides real-time gaze tracking and calibration capabilities
4. Includes demo applications for validation and testing

## Project Structure

```
├── apps/                     # Application modules
│   ├── data_preprocess/      # Data preprocessing scripts
│   ├── demo_game/            # Demo game for validation
│   ├── model_calibration/    # Calibration model implementations
│   └── neural_refine/        # Neural network refinement
├── assets/                   # Static assets
├── checkpoints/              # Model checkpoints
├── data/                     # Data storage
├── outputs/                  # Output files
└── scripts/                  # Evaluation scripts
```

See individual README files in each `apps/` subdirectory for detailed information.

## Requirements

- Python 3.11+
- PyTorch 2.9+
- Eye tracker hardware (Pupil Labs recommended)

## Evaluation

```bash
# Evaluate initial gaze error
python scripts/eval.py original data/prepared/split_avg_data/test.csv

# Evaluate sim RBF
python scripts/eval.py sim_rbf data/prepared/split_avg_data/test.csv

# Evaluate nueral-refined predictions
python scripts/eval.py pred_gaze outputs/predictions_test.csv
```

## Running

See [apps/model_calibration/README.md](apps/model_calibration/README.md) for calibration procedures.

See [apps/neural_refine/README.md](apps/neural_refine/README.md) for neural refine procedures.

See [apps/demo_game/README.md](apps/demo_game/README.md) for demo game instructions.


## Features

- **18-point calibration**: Systematic collection of calibration data
- **Model-based Multiple calibration**: Polynomial fitting, RBF, and neural refinement
- **Real-time processing**: Stream eye-tracking data with low latency
- **Demo validation**: Interactive game to test calibration quality

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```
@software{nrmbc2025,
  author = {Jiaqi Liu, Zixuan Wang, and ...},
  title = {NRMBC: Neural Refined Model-Based Gaze Point Calibration},
  year = {2025},
  url = {https://github.com/wannabeyourfriend/NRMBC-Neural-Refined-Model-Based-Gazing-Point-Calibration}
}
```

## Acknowledgments

- Pupil Labs for eye-tracking hardware and software support
- Contributors and researchers in the eye-tracking community