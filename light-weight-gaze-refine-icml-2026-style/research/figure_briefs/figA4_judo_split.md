# Figure A4 — JuDo target-disjoint split (kept from old appendix2)

## Why this figure
Documents that our JuDo evaluation uses a 33/7/8 target-point disjoint
split, addressing reviewer concerns about subject-level / setting-level
splits.

## Layout
Single panel, square aspect.

## Axes
- X-axis: target X (px), 250–1000.
- Y-axis: target Y (px), 250–800.

## Markers
- 33 training target points: blue circles.
- 7 validation target points: green squares.
- 8 test target points: red triangles.

## Annotations
A legend in the upper-right corner. A text box in the upper-left:
"Baseline calibration: 33 training points → fit ALL baselines → tested on 8
unseen points."

## Color & style
Same as the figure shown previously — keep it pristine. White background,
light grid.

## Caption text
> JuDo1000 target-disjoint split used in our experiments. All classical
> calibrators are fit using only the 33 training targets pooled across
> participants; baselines are then queried at the 7 validation and 8 test
> targets they have never seen. This guards against the trivial
> interpolation that ranks low-smoothing RBF as the best baseline on its
> own training set.
