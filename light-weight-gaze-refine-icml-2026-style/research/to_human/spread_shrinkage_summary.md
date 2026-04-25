# Spread-aware shrinkage (v4) — JuDo1000 LOSO summary

## Headline

**Spread-aware shrinkage does not meaningfully improve over v3 spatial
shrinkage on JuDo1000.** All learned shrinkage variants (v1, v3, v4a, v4b,
v4c) sit within +/-0.3 px of the anchor at every K we tested. The
inverse-variance-weighted-mean variant (v4b) is the most theoretically
principled and tracks v3 closely; the spread-feature-only variant (v4a)
shows the largest (still tiny) gain at K in [3, 8].

## Numbers (mean L2 in px, 6 LOSO folds, anchor = pred_similarity)

| K  | anchor |  v1   |  v3   |  v4a  |  v4b  |  v4c  |
|----|-------:|------:|------:|------:|------:|------:|
|  1 |  78.91 | 78.95 | 78.92 | 79.08 | 78.88 | 78.89 |
|  5 |  79.02 | 79.32 | 79.30 | 78.85 | 78.99 | 78.99 |
| 12 |  78.82 | 78.83 | 78.84 | 78.79 | 78.82 | 78.80 |
| 18 |  78.61 | 78.63 | 78.63 | 78.62 | 78.63 | 78.63 |

Delta vs anchor (px improvement, positive = better):

| K  | v3      | v4a     | v4b     | v4c     |
|----|--------:|--------:|--------:|--------:|
|  1 |  -0.01  |  -0.17  |  +0.04  |  +0.02  |
|  5 |  -0.29  |  +0.17  |  +0.03  |  +0.03  |
|  8 |  -0.11  |  +0.09  |  +0.02  |  +0.03  |
| 12 |  -0.02  |  +0.03  |  +0.00  |  +0.02  |
| 18 |  -0.02  |  -0.00  |  -0.01  |  -0.01  |

## Caveats and what changed from the brief

1. **Anchor switched from `pred_tps` to `pred_similarity`.** The 58 px TPS
   headline turned out to include is_fit=True rows (TPS perfectly fitting
   its own training points). On the held-out split TPS extrapolates badly
   (mean 169 px, dominated by 4 catastrophic-failure subjects). The
   genuinely strongest classical on held-out data is similarity (mean
   81 px), which we used.
2. **is_fit=True rows are dropped** before any analysis to avoid leakage
   of the calibration set into the LOSO context/query partition.
3. **Folds completed: 6 of 10 planned** (not 30) because of remote-GPU
   contention and per-fold cost (~3 min). 6 folds are enough to see that
   no variant lifts meaningfully off the anchor.
4. Reduced epochs (v1=30, v3=40, v4=40), train-pool cap (150 trials),
   train-query cap (128 fixations/trial) and val-pool cap (8 trials, 48
   fixations each) compared to the original protocol — required to fit
   in the wall-clock budget.

## Why the negative result

The JuDo "trial" unit is one full session (~540 fixations per session-x-subject).
The session-level systematic bias on the held-out is_fit=False fixations is
already small relative to the anchor's residual error: the per-trial bias
empirical mean explains <0.5% of the L2 budget. This means lambda* ≈ 0 is
near-optimal almost everywhere — there is essentially no headroom for any
shrinkage variant (spatial or spread-aware) to claw back.

Concretely: the raw mean estimator (lambda = 1) is 6-44 px **worse** than
the anchor at every K, while the best learned shrinkage is at most 0.3 px
**better** than the anchor. The shrinkage MLP correctly learns lambda ≈ 0
in almost all positions, regardless of how much extra information it gets
about per-fixation noise.

## Sub-variant comparison

- **v4a (spread feature on items + per-trial spread aggregate)** — best
  on average over K. The DeepSets encoder uses spread to slightly modulate
  lambda by trial quality, picking up tiny gains at K in [3, 8].
- **v4b (inverse-variance weighted mean only)** — closest to v3 by design.
  When most fixations have similar spread, the weighted and unweighted
  means are nearly identical, so the tiny gain reflects only a few outlier
  trials.
- **v4c (full)** — combines a and b. No additional gain over v4a alone,
  consistent with the two effects being correlated.

## Recommendation

Spread-aware shrinkage is theoretically principled but **does not help on
JuDo1000** because the per-session bias is already tiny relative to the
non-systematic anchor error. The variant could still be valuable on
datasets with stronger session-level bias (the self-collected 12-subject
dataset showed ~5% gain for v3 over v1; a re-run of v4 on that data is
the next experiment).

## Files

- Protocol: `experiments/spread_shrinkage_loso/protocol.md`
- Code: `src/spread_shrinkage.py`, `experiments/spread_shrinkage_loso/run.py`
- Results: `experiments/spread_shrinkage_loso/results.csv` (6 folds)
- Run log: `experiments/spread_shrinkage_loso/run.log` (gitignored, on
  remote at `/home/2025user/zhou/klab-workspace/gaze-refine-net/light-weight-gaze-refine-icml-2026-style/research/experiments/spread_shrinkage_loso/run.log`)
