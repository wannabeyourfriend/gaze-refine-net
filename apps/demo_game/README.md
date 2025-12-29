# Demo Game

Interactive gaze tracking game for validating and demonstrating the quality of different calibration approaches. The demo uses music visualization where circles appear on screen following musical notes, requiring accurate gaze tracking to interact successfully.

## Overview

This module provides a music-based demo game that tests eye tracking accuracy in a real-world interactive scenario. Users can compare performance across different calibration models including baseline tracking, polynomial calibration, simRBF calibration, and neural-refined calibration.

## Components

### audio_processor_split.py
- **Purpose**: Processes audio files (e.g., "Dance_Of_the_Golden_Snake") into structured note sequences
- **Features**:
  - Transcribes WAV audio files
  - Segments audio into time intervals with corresponding pitches
  - Outputs note sequence to text file for game synchronization

### eye_tracker_stream.py
- **Purpose**: Real-time data streaming from Pupil Labs eye tracker
- **Features**:
  - Establishes live connection to Pupil Labs eye tracking hardware
  - Streams gaze data in real-time
  - Provides low-latency data for interactive applications

### single_music_ui.py
- **Purpose**: Tests baseline eye tracking performance without additional calibration
- **Features**:
  - Uses only the default Pupil Labs device calibration
  - Provides baseline performance metrics
  - Single music track demo
  - Useful for comparing improvement gained from additional calibration

### run_demo_game.py
- **Purpose**: Main demo game interface for 18-point calibrated systems
- **Features**:
  - Second-stage UI after 18-point calibration
  - Interactive music-based game
  - Real-time gaze tracking visualization
  - Performance evaluation

### multiple_music_ui.py
- **Purpose**: Comprehensive comparison interface for testing multiple calibration models
- **Features**:
  - Tests three calibration approaches:
    1. **Polynomial calibration**: 2nd-order polynomial fitting with 18 calibration points
    2. **simRBF calibration**: Radial Basis Function calibration
    3. **simRBF + ResNet refinement**: Neural network-enhanced calibration
  - Side-by-side model comparison
  - Performance metrics for each approach
  - Multiple music track support

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

## Tips

- Ensure proper eye tracker calibration before running demos
- Adjust circle size if targets are too small/large for your setup
- Use `single_music_ui.py` first to establish baseline performance
- Compare results with `multiple_music_ui.py` to validate calibration improvements
- Good lighting conditions improve tracking accuracy

## Dependencies

- **PyQt6**: UI framework
- **PyQt6.QtMultimedia**: Audio playback
- **librosa**: Audio analysis
- **numpy/scipy**: Numerical processing
- Eye tracker hardware and drivers (Pupil Labs)