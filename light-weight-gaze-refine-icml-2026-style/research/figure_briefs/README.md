# Figure briefs for the NeurIPS 2026 resubmission

This directory contains one markdown file per intended figure. Each brief is
written so that an image-generation system (e.g. GPT-image, mid-journey,
or a human illustrator) can produce the figure without further context.

## Figure roster (main body, 7 pages allowed for figures + text)

| # | File | Replaces (ICML version) | Purpose |
|---|------|-------------------------|---------|
| 1 | `fig01_pipeline.md` | old Fig 1 | Three-tier overview: end-to-end vs vector-field calibration vs our shrinkage-refined pipeline |
| 2 | `fig02_leakage_diagnosis.md` | NEW | Bar chart contrasting honest vs leaky pipeline on JuDo and self-collected — the headline of our critique |
| 3 | `fig03_dataset_visualization.md` | old Fig 2 | EDA panel of the self-collected dataset: target grid, raw scatter, per-subject error, drift vector field |
| 4 | `fig04_shrinkage_main.md` | NEW | Main positive result: shrinkage L2 vs K context size; per-trial bias decomposition |
| 5 | `fig05_calibration_density.md` | old Fig 4 | Calibration density study (4-18 points) on the honest pipeline |
| 6 | `fig06_game_validation.md` | old Figs 5+6 (combined) | Game score and angular-error correlation under four conditions |
| 7 | `fig07_loso_per_subject.md` | NEW | Per-subject leave-one-out delta vs anchor baseline |

## Appendix figures

| # | File | Replaces | Purpose |
|---|------|----------|---------|
| A1 | `figA1_calibration_interface.md` | old appendix realdisplay/UI | Photographs of the data-collection setup and 18-point/32-point/game UIs |
| A2 | `figA2_fixation_ccdf.md` | old fig_appendix4 | CCDF justifying the 2-second fixation onset window |
| A3 | `figA3_gaze_trajectories.md` | old fig_appendix5 | 25/50/75 percentile gaze-trajectory visualizations |
| A4 | `figA4_judo_split.md` | old fig_appendix2 | JuDo target-disjoint split visualization |
| A5 | `figA5_subject_per_baseline_error.md` | NEW | Detailed per-subject × per-baseline heatmap |
| A6 | `figA6_lesspoints_full.md` | old fig_appendix1+3 | Full per-configuration results across 4-18 calibration points and structural variants |

## Style conventions

All figures should follow these conventions for consistency with the NeurIPS
template (US-letter, 5.5-inch text width):

- Font: Times / serif on axis labels, sans-serif (Inter / Helvetica) inside
  pipeline diagrams.
- Color palette: scheme = `["#3454D1","#D1495B","#2E933C","#F4A261","#7A3293","#264653"]`.
  Anchor baseline (sim+RBF) always purple. Our method always green. Raw gaze
  always grey-dashed.
- Line weights: 1.6pt for main lines, 0.8pt for grid (alpha 0.3).
- Titles: 11pt, axis labels 10pt, ticks 8pt.
- DPI: render at 300 DPI for pixel figures; vector PDF for diagrams.
- All figures must remain legible if printed in greyscale (use markers + line
  styles, not just color).
- Maximum width: `\linewidth` (single column) or `\textwidth` (two-column).
