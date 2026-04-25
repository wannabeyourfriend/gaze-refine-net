# Figure 3 — Self-collected dataset visualization (NEW, replaces old Fig 2)

## Why this figure
Reviewer Nmnp asked "how does this perform under real-world disturbances?"
and reviewer ELq4 noted the dataset's small scale. This figure
characterizes the data so readers understand both the geometry of the
calibration task and the per-subject variability that any honest method
must contend with.

## Layout
2 × 2 grid (\linewidth wide, ~4.8 in tall). Subplots labelled (a)–(d).

## Panel (a) — Calibration grid + raw scatter (one example trial)

- Background: a 1920 × 1080 white rectangle representing the screen.
- Plot the 18 calibration target points (small red squares) and 32 test
  target points (small grey squares) for a representative trial.
- Overlay each test target's raw gaze sample cloud: ~30 small light-blue
  dots per target representing the raw eye-tracker samples; a darker blue
  filled dot for the mean (the "raw predicted" point).
- Black arrow from each raw mean to its target.
- Title: "(a) Trial layout: 18 fitting + 32 testing points"

## Panel (b) — Per-subject error distribution

Box plot (one box per subject):
- X-axis: subject id (12 subjects, sorted by median error ascending).
- Y-axis: per-fixation L2 error (px) of the strongest classical baseline
  (sim_rbf_multiquadric_s2.0). Range 0–200 px.
- Show median, IQR, whiskers (5th/95th percentile), outliers as light dots.
- Highlight Liu Jiaqi (87 trials) and ZAnna (35 trials) with thicker
  borders to indicate they dominate the dataset by sample count.
- Title: "(b) Per-subject classical-baseline error"

## Panel (c) — Per-trial bias structure

Scatter of 143 per-trial bias vectors (target − sim_rbf_pred):
- X-axis: bias_x (px), range −80 to 80.
- Y-axis: bias_y (px), range −80 to 80.
- Each point is one trial; size proportional to log(num samples), color
  encodes subject identity.
- Dashed circles at radii 25, 50, 75 px to give scale.
- An inset histogram on the top edge shows the marginal of bias magnitude
  ‖bias‖.
- Title: "(c) Per-trial bias = target − anchor mean"

## Panel (d) — Drift vector field (single trial example)

- Take the same trial as in (a).
- Show the 32 test target locations as black dots and overlay the
  classical baseline's drift vector at each location (anchor pred − target)
  as a coloured arrow whose color encodes magnitude (yellow=small,
  red=large).
- Underlay light grey lines connecting calibration fitting points to give
  spatial context.
- Title: "(d) Residual vector field after classical calibration"

## Color & style
- Discrete subject palette: 12 colors from `tab20` or a custom hue ramp.
- For panel (b): `seaborn` boxplot style, no whisker-flier outliers shown
  beyond 99.5%.
- For (c): use viridis for log(n_samples) size scaling.
- For (d): use a continuous error colormap (e.g. `magma_r`).

## Caption text
> Self-collected dataset structure. (a) One representative trial: 18
> fitting points (red) and 32 test points (grey) on a 1920 × 1080 screen,
> with raw eye-tracker samples (light blue) clustered around each test
> mean (dark blue). Black arrows show the residual the calibrator must
> correct. (b) Per-subject distribution of classical-baseline error
> (sim+RBF), 12 subjects sorted by median; bold box outlines mark the two
> subjects (Liu Jiaqi, ZAnna) that contribute > 90 % of trials. (c)
> Per-trial bias vectors after classical calibration: most trials retain
> a coherent bias of 25–60 px that is roughly direction-stable per
> session. (d) Residual vector field for a single trial; the spatial
> structure suggests a small-magnitude additive correction is sufficient
> when estimated from a few extra fixations.
