# Demo Game

Interactive gaze tracking game for validating and demonstrating the quality of different calibration approaches. The demo uses music visualization where circles appear on screen following musical notes, requiring accurate gaze tracking to interact successfully.

## Overview

This module provides a music-based demo game that tests eye tracking accuracy in a real-world interactive scenario. Users can compare performance across different calibration models including baseline tracking, polynomial calibration, simRBF calibration, and neural-refined calibration.

## Usage

### Prerequisites
- Pupil Labs eye tracker hardware
- Python 3.11+
- PyQt6 for UI
- Audio files prepared with `audio_processor_split.py`

### Running the Baseline Demo
```bash
cd apps/demo_game
python single_music_ui.py
```

### Running the Calibrated Demo Game
After completing 18-point calibration (see [model_calibration](../model_calibration/README.md)):
```bash
python run_demo_game.py
```

### Comparing Multiple Calibration Models
```bash
python multiple_music_ui.py
```

## Game Mechanics

- **Circles appear** on screen at positions corresponding to musical notes
- **Gaze at circles** using your eyes to interact
- **Track accuracy** is measured by how precisely your gaze aligns with targets
- **Visual feedback** indicates current gaze position and target locations
- **Performance metrics** are collected for evaluation

## Configuration

Key parameters can be adjusted in the script files:
- **CIRCLE_RADIUS**: Size of target circles (default: 50 pixels)
- **WINDOW_W, WINDOW_H**: Display resolution (default: 1920×1080)
- **BASE_NOTES**: Musical note range for note mapping
- **APRILTAG_DIR**: Path to AprilTag markers for screen positioning

## Output

The demo game generates:
- Real-time visual feedback during gameplay
- Performance logs for accuracy analysis
- Timing data for latency evaluation
- Comparison metrics across calibration models