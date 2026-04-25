# Figure A6 — Full per-configuration density results (revised from old appendix1+3)

## Why this figure
The old figure presented every configuration of fitting points (4–18, A–F
permutations) for all baseline methods plus the leaky neural method. We
need to redo this honestly: drop the leaky line and add a comparison
showing the shrinkage method's density invariance.

## Layout
Two-panel figure (\linewidth × 4.5 in tall), stacked vertically.

## Panel (a) — Per-configuration line plot

X-axis: configuration label (4points_A, 6points_A...C, 8points_A...D,
10points_A...D, 12points_A...F, 14points_A...E, 16points_A...C, 18points_A).

Y-axis: residual error (°), 0.85–1.50.

Lines (one per method, no leaky neural):
- Raw Predicted (dashed grey)
- PWA (purple)
- TPS (cyan)
- AffineRBF (pink)
- AffineTPS (green)
- AffineGPR (yellow)
- Honest Multi-Baseline Neural Refinement (purple, diamond)
- **Online Shrinkage K=12 (green-bold, star)** — flat horizontal line at
  ~0.85°.

## Panel (b) — Density-invariance summary

A small bar chart showing the *median improvement* of shrinkage over the
strongest classical baseline at each fitting density. X-axis: 4, 6, 8,
10, 12, 14, 16, 18 points. Y-axis: improvement (px or °).
Shrinkage at K=12 should give a roughly constant ~0.18° / 5 px improvement
across all fitting densities.

## Color & style
Panel (a) uses the same palette as Fig 5; emphasize the shrinkage line
with bold weight. Panel (b) uses solid green bars with the value
annotated above each bar.

## Caption text
> Per-configuration honest evaluation of all classical baselines, the
> honest multi-baseline neural refinement, and our online shrinkage at
> K=12, across fitting-set sizes 4–18 with multiple structural
> arrangements per size. Top: AffineRBF dominates classical methods at
> every density; the honest neural refinement tracks AffineRBF without
> improving on it. Online shrinkage at K=12 maintains a flat ~0.85° error
> regardless of fitting density. Bottom: median improvement of shrinkage
> over the strongest classical baseline as a function of fitting density;
> the absolute gain is roughly constant in N, demonstrating that the
> per-trial bias removed by shrinkage is *orthogonal* to the fitting
> capacity.
