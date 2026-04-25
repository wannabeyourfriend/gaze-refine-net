# JuDo1000 main benchmark — protocol

## Goal

A unified comparison of all methods on the JuDo1000 target-disjoint
split (33 train / 7 val / 8 test target points), with per-target-point
grouping as the trial unit.

## Methods

1. origin_gaze (raw eye tracker)
2. Classical baselines (similarity, poly, RBF/TPS variants, sim+RBF,
   sim+TPS, sim+PWA)
3. Multi-Baseline Residual (MBR) — MLP that maps
   `[orig, K_classical_preds]` → residual on top of anchor (the
   residual-learning style of prior work)
4. Per-trial fixed-λ shrinkage on the anchor (parameter-free baseline
   from the shrinkage literature, swept over λ ∈ [0, 1] in 21 steps)
5. v1 — Learned scalar shrinkage (ours): MLP outputs (λ_x, λ_y) per trial
6. v3 — Learned spatial shrinkage (ours): MLP outputs λ(p) per query

## Anchor

`pred_similarity` (the strongest classical on JuDo, 21.66 px on test).

## Locked predictions

- Classical anchor: 21.7 px
- MBR: ~22 px (residual learning has nothing to add when residuals are
  spatially uncorrelated noise)
- v3 spatial shrinkage at K=12: should beat the anchor by 1-3 px

The headline of the paper is the comparison table generated from this
single run.
