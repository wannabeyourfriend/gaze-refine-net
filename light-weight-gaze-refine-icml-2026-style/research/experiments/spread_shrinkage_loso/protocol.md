# Spread-aware shrinkage LOSO on JuDo1000 — protocol

## Hypothesis

H1. **Spread-aware shrinkage** outperforms v3 spatial shrinkage on JuDo1000
under leave-one-subject-out (LOSO). Per-fixation `spread` is a direct estimate
of the per-fixation eye-tracker noise std (sigma_eps). The Bayes-optimal
homoscedastic shrinkage is

    lambda* = sigma_b^2 / (sigma_b^2 + sigma_eps^2 / K)

so providing per-fixation spread to the shrinkage MLP should let it
choose a more reliable lambda than v3, which sees only positions and
residuals.

H2. The **inverse-variance weighted mean** estimator
`bar_b = sum_j w_j r_j / sum_j w_j` with `w_j = 1 / (spread_j^2 + eps^2)`
should further reduce L2 vs the simple unweighted mean, because it is
the optimum heteroscedastic mean estimator under independent Gaussian noise.

## Sub-variants compared

| label | use_spread_item_feat | use_spread_global_feat | inverse_variance_mean |
|-------|----------------------|------------------------|-----------------------|
| anchor (TPS only)        | -- | -- | -- |
| v1 scalar (existing)     | no | no | no |
| v3 spatial (existing)    | no | no | no |
| v4a spread-feat-only     | yes | yes | no  |
| v4b inv-var only         | no  | no  | yes |
| v4c full spread-aware    | yes | yes | yes |

## Anchor

`pred_similarity` (strongest classical on the held-out is_fit=False split:
mean 81 px L2; median 41 px). The original protocol called for `pred_tps`,
citing "strongest classical at 58 px"; that 58 px figure includes is_fit=True
rows (TPS interpolating its own training points to ~zero error). On the
held-out split TPS extrapolates badly (mean 169 px, dominated by 4 subjects
with anchor mean L2 > 200 px). Switching to `pred_similarity` gives a
realistic comparison of shrinkage gain on a sane anchor.

## Data filtering

We drop is_fit=True rows before any analysis: they were used to fit the
classical baselines and would leak the calibration set into the LOSO
context/query partitions.

## Configuration

- Data: `/home/2025user/zhou/klab-workspace/gaze-refine-net/data/prepared/judo1000_full/all_with_baselines.csv`
- 150 subjects total; we sample **30** subjects uniformly at random (seed 1047)
  as the held-out LOSO fold set, to keep wall-clock under one hour.
- For each held subject:
  - Train pool = all other 149 subjects.
  - 10% trial-disjoint split of the train pool used as validation.
  - Train v1, v3, v4a, v4b, v4c with the same K-randomization
    (K_train_min=1, K_train_max=18) and the same number of epochs (80 for v1, 120 for v3 + v4*).
  - Evaluate at K in {1, 2, 3, 5, 8, 12, 18} on the held subject's trials
    (4 sessions × ~540 fixations = ~2160 query fixations available).
- Screen size for normalization: 1024 x 768.
- Anchor column: `pred_tps`.
- Random seed: 1047 throughout.

## Locked predictions

- v3 spatial improves ~3-5% over v1 scalar at K=12-18 (consistent with
  the self-collected dataset finding).
- v4c full spread-aware should improve further by 1-3% at K in [3, 12]
  where heteroscedasticity matters most. At K=18 the gain shrinks since
  the unweighted mean is already a strong estimator.
- If v4 fails to beat v3 by more than 0.3 px mean L2, we conclude the
  spread feature is uninformative for shrinkage on JuDo (likely because
  spread mostly tracks fixation duration, not the systematic bias
  variance the shrinkage estimator targets).

## Outcome to record

`results.csv` with one row per (held_subject, method, K). Aggregated
subject-mean printed at end, plus a JSON dump.
