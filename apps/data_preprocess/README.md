# Data Preprocessing

This module contains scripts for preprocessing raw gaze tracking data into formats suitable for model training and evaluation.

## Overview

The data preprocessing pipeline handles cleaning, filtering, and splitting of eye-tracking data collected during calibration sessions. It prepares the data for use in both traditional calibration models and neural network refinement.

## Scripts

### clean_outlier.py
- **Purpose**: Removes outlier data points from raw gaze tracking datasets
- **Features**:
  - Statistical outlier detection
  - Data quality validation
  - Preserves data integrity while removing noise

### fixed_filter.py
- **Purpose**: Applies filtering algorithms to smooth gaze data
- **Features**:
  - Temporal filtering for reducing jitter
  - Fixed-window smoothing

### select_ana_jiaqi.py
- **Purpose**: Analyzes and selects specific subsets of data for experiments
- **Features**:
  - Custom data selection criteria
  - Experiment-specific data extraction

### split.py
- **Purpose**: Splits data into training, validation, and test sets
- **Features**:
  - Stratified splitting to maintain data distribution
  - Configurable train/val/test ratios
  - Session-aware splitting to prevent data leakage

## Usage

### Data Cleaning
```bash
cd apps/data_preprocess/scripts
python clean_outlier.py --input <path_to_raw_data> --output <path_to_cleaned_data>
```

### Data Filtering
```bash
python fixed_filter.py --input <path_to_data> --output <path_to_filtered_data>
```

### Data Splitting
```bash
python split.py --input <path_to_data> --output_dir <path_to_output>
```

## Data Format

The preprocessing scripts expect and produce CSV files with the following columns:
- Timestamp information
- Original gaze coordinates (x, y)
- Target coordinates (x, y)
- Session/trial identifiers
- Additional metadata

## Output

Processed data is typically saved to `../../data/prepared/` with the following structure:
```
data/prepared/
├── split_avg_data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── [other processed datasets]
```

## Dependencies

- Python 3.11+
- pandas
- numpy
- scipy (for outlier detection and filtering)

## Notes

- Always backup raw data before running preprocessing scripts
- Outlier removal parameters may need adjustment based on your specific eye tracker and setup
- The 4-second aggregation window has been found to produce optimal results for model fitting
