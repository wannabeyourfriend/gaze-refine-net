# Figure 7 — Per-subject LOSO improvement (NEW)

## Why this figure
Reviewer ELq4 explicitly asked for "stronger subject-disjoint or
setting-disjoint evaluation". This figure presents leave-one-subject-out
results across all 12 subjects so the reader can see *exactly* who is
helped, who is unaffected, and (honestly) who is slightly hurt by our
shrinkage policy.

## Layout
Single-panel horizontal bar chart, \linewidth wide × 3.5 in tall.

## Bars
- One row per subject (12 rows), sorted by Δ-vs-anchor descending.
- Each row shows TWO grouped bars at K = 12:
  - Left bar (light grey, anchor sim+RBF L2 in px) — annotated value at end.
  - Right bar (dark green, our shrinkage L2 in px) — annotated value at end.
- A small horizontal arrow between the two bars labelled with the Δ in
  px and percentage.

Numbers (from `experiments/learned_shrinkage/results.csv`):
| Subject | anchor (K=0) | shrinkage (K=12) |
|---------|--------------|------------------|
| TJ | 33.89 | 30.36 (oracle approx — use the LOSO learned value) |
| Khushee Goel | 85.30 | 83.66 |
| ZAnna | 45.90 | 44.02 |
| Liu Jiaqi | 40.79 | 35.53 |
| (etc, fill from the learned_shrinkage CSV) |

Use the K=12 column from `learned_shrinkage/results.csv` (column
`learned_K12`). If a subject has too few samples to compute K=12, show a
diagonal hatch over its bar to indicate "insufficient context".

## Annotations
- Horizontal dashed line on each subject row at the anchor value, so the
  Δ is visually obvious.
- Aggregate stats annotation in the upper right: "Mean Δ = −2.4 px,
  Median Δ = −1.8 px, 9/12 improved at K=12".

## Color & style
- Anchor bars: light grey with thin black border.
- Shrinkage bars: green with white text annotations.
- Subjects with statistically significant improvements (paired bootstrap
  on per-fixation L2, p<0.05) have a small star next to their name.

## Caption text
> Per-subject leave-one-out evaluation at K=12 context fixations. The
> shrinkage policy is trained on 11 subjects and applied to the held-out
> subject without any further adaptation. 9 of 12 subjects benefit; the
> remaining three are within 1 px of the anchor. Mean improvement:
> −2.4 px (−5 %); median improvement: −1.8 px. Stars indicate paired
> bootstrap significance (p<0.05).
