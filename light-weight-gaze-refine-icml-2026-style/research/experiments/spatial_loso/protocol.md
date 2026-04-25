# Spatial shrinkage LOSO — protocol

## Hypothesis

H1. Spatial shrinkage λ(p) (v3) outperforms scalar shrinkage (v1) on the
12-subject self-collected dataset under leave-one-subject-out, primarily
because per-trial residual reliability has a position-dependent
component.

H2. Both v1 and v3 outperform the parameter-free fixed-λ sweep at
K ≥ 8.

## Configuration

- Anchor: pred_sim_rbf_multiquadric_s2.0
- LOSO: train on 11 subjects (10% trial-disjoint val), test on held-out 12th.
- K_train range: [1, 18] randomized per episode; eval at K ∈ {1, 2, 3, 5, 8, 12, 18}.
- v1: SpatialShrinkageConfig(spatial=False, ctx_hidden=64).
- v3: SpatialShrinkageConfig(spatial=True, ctx_hidden=128, fourier_bands=6).
- Same seed across folds.

## Locked predictions

- v1 should match the previous (learned-shrinkage) results (-17% at K=12).
- v3 should improve by an additional 1-3% at K=12-18.

## Outcome to record

`results.csv` with one row per held subject and one column per
(method, K) pair.
