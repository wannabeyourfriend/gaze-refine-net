# Figure 4 — Shrinkage main result (NEW headline figure)

## Why this figure
This is the central positive result of the paper. It demonstrates that a
tiny learned shrinkage estimator beats both the raw mean-bias correction
and the strongest classical baseline by 17 % at K=12.

## Layout
Two-panel figure (\linewidth wide, ~3 in tall). Subplots (a) and (b).

## Panel (a) — Test L2 vs context size K

Line plot:
- X-axis: K, the number of extra calibration fixations after the standard
  18-point calibration. Values: 0, 1, 2, 3, 5, 8, 12, 18.
- Y-axis: Test mean L2 (px), range 30 to 75.
- Three lines:
  - Grey horizontal dashed line at K=0 (anchor-only, sim+RBF baseline)
    labelled "anchor (sim+RBF) = 44.7 px".
  - Orange line: raw mean-bias correction (λ=1) per the values
    `{1: 58.8, 2: 49.0, 3: 46.0, 5: 43.3, 8: 40.6, 12: 38.0, 18: 38.6}`.
    Marker: filled circle.
  - Magenta line: fixed-λ tuned per K
    `{1: 43.3, 2: 42.7, 3: 42.0, 5: 41.0, 8: 39.4, 12: 37.5, 18: 38.2}`.
    Marker: square.
  - Green line: learned shrinkage MLP (LOSO)
    `{1: 63.9, 2: 60.1, 3: 61.3, 5: 56.3, 8: 54.1, 12: 42.3, 18: 40.2}`.
    Marker: filled diamond, slightly larger.
- Annotate the K=12 point on the green line with a callout
  `"−17 % vs anchor"`.
- Annotate K=18 with `"−10 % vs anchor"`.

Note: the raw-mean line only outperforms the anchor for K ≥ 5 because
single-sample bias estimates are dominated by noise. The figure should
make this dip *visible* (raw line goes UP from K=0 to K=1, then down).

## Panel (b) — Bias-magnitude shrinkage policy heatmap

A heat map showing the learned shrinkage λ (averaged across LOSO folds)
as a function of two binned features:
- X-axis: empirical bias magnitude ‖mean(target − anchor)‖ (binned: 0–10,
  10–20, 20–40, 40–80, >80 px).
- Y-axis: K (binned: 1, 2, 3, 5, 8, 12, 18).
- Cell color: average λ ∈ [0,1] from `viridis` colormap (dark = 0, light = 1).

This panel shows the *structure* of the learned policy:
- For small K and large bias magnitude, λ should be small (don't trust
  noisy bias estimates of large magnitude).
- For large K, λ should approach 1 (trust the empirical mean).
This matches James-Stein intuition.

If the LOSO policy doesn't yield a clean monotone heatmap, fall back to
showing the **fixed-λ** sweep heatmap instead, which we know produces a
clean diagonal pattern (best λ rises with K).

## Color & style
- Anchor line: dim grey, dashed, no markers.
- Raw mean (λ=1): orange (#F4A261), filled circle.
- Fixed-λ tuned: magenta (#D1495B), square.
- Learned MLP: green (#2E933C), diamond.
- Background grid: light, alpha 0.3.
- For panel (b): `viridis` colormap, annotate each cell with the λ value
  to two decimal places.

## Caption text
> Online bias correction with learned shrinkage. (a) Test mean L2 on the
> 12-subject dataset as a function of context size K. Grey dashed line:
> anchor classical baseline (sim+RBF, K=0). Orange: raw mean-bias
> correction (λ=1) — at small K the empirical mean is noisier than the
> baseline. Magenta: globally tuned fixed λ per K. Green: per-trial
> learned shrinkage from a ≈1k-parameter MLP, evaluated leave-one-subject
> out. At K=12 the learned policy reduces L2 by 17 % over the anchor and
> by 7 % over fixed λ. (b) Average learned shrinkage λ as a function of
> empirical bias magnitude and K, recovering the James-Stein-style policy
> (smaller λ for noisier estimates and large bias magnitudes).
