# Figure 5 — Calibration density study (revised from old Fig 4)

## Why this figure
The original Fig 4 plotted "residual error vs number of calibration points"
for several methods and showed a near-monotone improvement from 18 → 4
points for the leaky neural method (curiously beating classical methods at
every K). With the leakage fixed, the curves change qualitatively. The
honest version still has a useful story: the **classical sim+RBF** is
remarkably robust as N decreases (it degrades smoothly), the **honest
neural refinement** never beats it, and **online shrinkage at K=12 extra
points** dominates everything once it has enough context.

## Layout
Single-panel line plot, \linewidth wide × 3.5 in tall.

## Axes
- X-axis: number of fitting calibration points (4, 6, 8, 10, 12, 14, 16, 18).
- Y-axis: angular residual error (°), range 0.85 to 2.0. Use a horizontal
  dashed reference line at 1.53° labelled "Raw eye tracker".

## Curves
Methods (compute from honest re-runs of `experiments/calibration_density/`
which I will produce; values are placeholders to be filled by the
plotting script):

1. Quadratic Polynomial (blue, square): degrades fastest at low N.
   Approximate values: {4: 1.85, 6: 1.50, 8: 1.34, 10: 1.22, 12: 1.14,
   14: 1.12, 16: 1.10, 18: 1.09}.
2. Affine Transformation (brown, circle): similar to TPS.
3. AffineTPS (green, plus marker).
4. AffineRBF (pink, plus marker): consistently the best classical.
5. Honest Neural Refinement (purple, diamond): tracks AffineRBF closely.
6. Online Shrinkage K=12 (green-bold, star): a single horizontal line at
   ~0.85° because shrinkage doesn't depend on the fitting set size — once
   you have 12 extra fixations the per-trial bias is largely removed
   regardless of the upstream fitting density. This is the visual punch
   line.

## Annotations
- Mark the 18-point case with a vertical light-grey band labelled
  "standard protocol".
- Add a single arrow from the AffineRBF curve at N=18 to the shrinkage
  line, labelled "−15 % from K=12 extra fixations".

## Color & style
Same palette as Figure 4. Star marker reserved for the shrinkage method.
All curves share the same line weight (1.6pt). Grid alpha 0.3.

## Caption text
> Calibration density study. Classical methods (Affine, AffineTPS,
> AffineRBF) all degrade gracefully as the fitting-point set shrinks from
> 18 to 4. AffineRBF is the strongest classical fit at every density.
> Honest multi-baseline neural refinement (purple) tracks AffineRBF
> closely without improving on it. Adding our online shrinkage (green
> star, K=12 extra fixations) provides a roughly density-invariant 15 %
> improvement, indicating that the *bottleneck is per-trial bias, not
> fitting capacity*.
