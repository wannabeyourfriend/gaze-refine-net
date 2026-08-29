# scripts/

Two kinds of thing live here: pipeline steps you run in order to turn raw gaze
logs into trainable splits, and one-off analyses that answer a specific
question. They are not interchangeable.

## pipeline — run in this order

| script | does |
|---|---|
| `process_data.py` | Cleans, filters and splits the self-collected gaze logs; fits the classical per-session calibrators and writes their predictions as `pred_*` columns into `data/prepared/`. |
| `anonymize_data.py` | Replaces real subject names with `subject_NNN`. One global mapping across all files, so a person keeps the same ID everywhere. Leaves JuDo1000's own `judo_N` IDs alone. Idempotent; `--dry-run` first. |
| `prepare_judo_data.py` | Builds JuDo1000 train/val/test from the raw download. **Splits rows at random** (`--train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15`) — see the protocol warning below. |
| `generate_judo_baselines_no_leakage.py` | Refits the JuDo1000 baselines using training data only, so calibrators never see evaluation targets. |

## splits — pick one deliberately

| script | question it answers |
|---|---|
| `split_judo_by_targets.py` | How does the model do on *screen positions* it never saw? Train/val/test get disjoint target points. |
| `split_judo_balanced_targets.py` | Same, but spatially balanced — the older `judo_1000_split_no_leakage` shuffled the 48 targets without controlling for where they fell. |
| `analyze_judo_targets.py` | Which target points exist, and how should they be partitioned to avoid leakage? Run before either splitter. |

**The split you choose changes the conclusion.** A random row split lets a
per-session calibrator interpolate between train and test fixations from the
same session at the same screen positions. Target-disjoint and subject-disjoint
splits give substantially different — and much less flattering — numbers.
Whatever you run, report which one it was.

## evaluation

| script | question it answers |
|---|---|
| `eval.py` | What is the error of a trained refiner on a given split? |
| `evaluate_judo_1000.py` | How does a trained multi-baseline refiner do on the JuDo1000 test set? |
| `verify_evaluation_consistency.py` | Are the neural model and the classical baselines being scored on the same rows, the same way? Run this whenever a comparison looks surprising. |
| `analyze.py` | Unified dataset summary and evaluation across splits. |
| `print_calibration_summary.py` | What baselines were fitted, and how did each do? |

## figures and tables

| script | produces |
|---|---|
| `generate_table.py` | Calibration comparison tables (compact or detailed). |
| `plot_num_points_vs_error.py` | Error vs. number of calibration points, across models. |
| `violin_calibration_error.py` | Violin plot of per-method calibration error. |
| `analyze_filtered_trials.py` | Parses the filtered-trials summary markdown into per-trial statistics. |

## notes

- Most scripts assume the repo root as the working directory and resolve paths
  relative to their own location — run them as `python scripts/<name>.py`.
- JuDo1000 is not redistributed with this repository. The `judo_*` scripts need
  `data/raw/judo1000_source/JuDo1000/` populated first.
- `eval.py` and `violin_calibration_error.py` have no module docstring; read the
  argparse block at the bottom for their options.
