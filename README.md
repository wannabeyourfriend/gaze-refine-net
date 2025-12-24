
## Jiaqi Work Notes
Notes:
- The 7th version is the latest. CSV outputs for origin+test now contain only origin and target gaze data; poly and RBF data are removed and handled in later processing.
- `calibration_model_full_compare` performs first-stage per-file post-processing: fit a model using a session's origin data and evaluate with test data for comparison.

`batch_run_all_sessions` goals:
- Call `calibration_model_full_compare` to batch model fitting and comparisons.
- Stack test CSVs with model-corrected data to produce datasets for deep learning.
- Produce `all_trials_model_predictions`; 4s error fitting remains best. Try multiple durations to demonstrate model robustness.

Small experiments to add:
- Check whether horizontal vs vertical errors differ significantly.
- Test if origin points in the test set perform better than other points to avoid overfitting.
- Validate whether 2s intervals are appropriate and whether 4s aggregation is optimal.
- Add a script to process all sessions with a `before` file to measure fixed time to enter a spread circle (resolved in `post_processing/2s_reaction`).
- Add a script to create new `grid_gaze_log_ns` files using the first 1/2/3s samples in each folder (resolved in `build_grid_gaze_log_partial`).
- Deep learning alone performs poorly; run ablations to justify the need for prior model fitting.

Build a demo to validate performance (typing? multi-dimensional representation of fit quality).

`judgement_application` components:
- `audio_process`: convert downloaded MP3s into usable WAVs.
- `eye_tracker_stream`: handle real-time data stream only.
- `audio_processor_split`: split audio files into TXT segmentation outputs.
- `gaze_music_ui`: main "Golden Snake Dance" UI.
- `gaze_music_ui_2th`: variant used by `run_full_experiment` with small API tweaks (RBF model integration).
- `run_full_experiment`: full 18-point + UI pipeline; already running.

Status checklist:
- Real-time data stream ingestion √
- Fixation scoring and other quantitative metrics √
- Fusion with calibration to threshold decisions on processed data √
- Manual TXT alignment fixes √



## Zixuan Work notes

```bash
# initial gaze error
python scripts/eval.py original data/prepared/split_avg_data/test.csv
# sim rbf
python scipts/eval.py sim_rbf data/prepared/split_avg_data/test.csv
# refine
python scripts/eval.py pred_gaze outputs/predictions_test.csv
```