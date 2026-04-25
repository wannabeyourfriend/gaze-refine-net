# Figure A5 — Subject × baseline error heatmap (NEW)

## Why this figure
Reviewer Nmnp asked about robustness across diverse conditions. A 12 × 7
heatmap showing per-subject performance of every classical baseline +
our shrinkage method makes the variability visible at a glance.

## Layout
Single-panel heatmap.

## Axes
- Rows (12): subjects sorted by trial count descending (Liu Jiaqi at top).
- Columns (8): origin_gaze, similarity, polynomial, gpr, tps, sim_pwa,
  sim_rbf_s2.0, **shrinkage K=12**.
- Cell value: per-subject mean L2 (px). Use a divergent colormap centered
  at the per-subject anchor (sim_rbf_s2.0); cells better than anchor
  become greens, cells worse become reds.
- Annotate each cell with the numeric L2 to one decimal place.

## Color & style
- Diverging colormap `RdYlGn_r` truncated to ±20 px around the
  per-subject anchor.
- Black grid between cells.

## Caption text
> Per-subject × per-method test L2 (px). Each row is normalized so colors
> reflect *relative* improvement over that subject's classical anchor
> (sim+RBF). Greens denote methods better than anchor for that subject;
> reds denote worse. The shrinkage column shows consistently green or
> neutral cells, even for subjects with very high baseline error.
