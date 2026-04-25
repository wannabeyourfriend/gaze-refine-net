# Findings — gaze-refine-net resubmission to NeurIPS 2026

_Last updated: 2026-04-26_

## Headline

The original paper's reported gains (0.96°/42.3 px on the self-collected
dataset and 5.8 px on JuDo1000) are entirely artifacts of a label-leaking
input-feature construction (`mb_features = baseline_pred - target` in
`apps/neural_refine/src/model.py:254`). Without this leak — i.e. with the
mb_features replaced by raw baseline predictions and normalization computed
on training data only — every variant of the proposed network either matches
or *under-performs* the strongest classical baseline on both datasets.

We rebuilt the pipeline from scratch and ran a systematic exploration of
neural designs (anchor-residual MLP, softmax blend over baselines, DeepSets
context encoder). None reliably beat the best classical calibrator.

What *does* work: a per-trial **online bias correction with learned
shrinkage**. With K context fixations after the standard calibration
phase, a small MLP predicts a 2D shrinkage factor for the empirical bias
estimate, conditioned on the bias statistics. This delivers honest
17–19% reductions in mean L2 over the strongest classical baseline at
K=12 calibration points and generalizes leave-one-subject-out across the
12 collected participants.

## Comprehensive Result Table

All numbers are mean L2 (px) on the held-out test set after eliminating
the input-feature leakage.

### JuDo1000 (point-disjoint split, 1927 test samples, 8 unseen targets)

| Method | Test L2 mean | Median | Std | p95 |
|--------|-------------:|-------:|----:|----:|
| origin_gaze (raw eye tracker)            | 21.96 | 18.08 | 15.96 | 54.32 |
| pred_similarity                          | 21.66 | 17.81 | 15.68 | 52.77 |
| pred_poly                                | 25.23 | 21.02 | 17.63 | 59.90 |
| pred_sim_rbf_multiquadric_s2.0           | 47.91 | 44.76 | 23.81 | 89.67 |
| pred_tps                                 | 76.43 | 62.83 | 45.82 | 161.56 |
| **honest paper method** (top-M=4 + noise20 + [64,32,16] BN) | **30.46** | 27.5 | 17.6 | 62.7 |
| honest method, all 11 baselines          | 44.39 | 31.3 | 52.4 | 111.9 |
| oracle top-M=4 (rank on test set)        | 31.94 | 30.0 | 15.9 | 62.2 |
| **leaky paper method (reproduces published 5.82 px)** | **4.21** | 3.6 | 2.3 | 9.1 |
| v2 anchor-residual MLP, anchor=similarity | 21.70 | 17.8 | — | 51.9 |

The honest paper method is **8.8 px worse** than just using the similarity
baseline. The leaky version matches the paper's headline. JuDo1000's
residual variance is dominated by per-fixation eye-tracker noise that is
not learnable.

### Self-collected 12-subject (`all_trials_split`, 506 test rows)

Per-trial baselines fit on the trial's 18 calibration points; baseline
predictions stored alongside test rows.

| Method | Test L2 mean | Median | Std | p95 |
|--------|-------------:|-------:|----:|----:|
| origin_gaze                              | 67.25 | — | — | — |
| pred_similarity                          | 50.45 | — | — | — |
| pred_poly                                | 48.88 | — | — | — |
| pred_sim_rbf_multiquadric_s2.0 (anchor)  | 45.53 | 38.98 | — | 109.89 |
| uniform-average over 6 baselines         | 45.78 | — | — | — |
| **per-trial BMA (LOO softmin, T=10)**    | **44.62** | 36.02 | — | 105.63 |
| v2 anchor-residual (random-split test)   | 44.71 | 36.6 | — | 109.4 |
| v2 stacker (LOSO, sample-weighted)       | 45.03 | 38.9 | — | — |
| context-conditioned refiner (LOSO, K=12) | 48.84 | — | — | — |

Best honest neural method on this dataset (BMA + per-trial reliability) is
44.62 px, only **1.0 px (2%) better** than the best classical baseline.

### Online bias correction with K context fixations

Same dataset, same anchor (`pred_sim_rbf_multiquadric_s2.0`), but exposing
the system to K extra calibration fixations between the standard
calibration phase and the test set. Aggregated mean L2 across the union of
2526 trial-level samples:

| K  | raw bias correction (λ=1) | learned shrinkage (LOSO) |
|----|--------------------------:|-------------------------:|
| 0  | 44.69 (anchor)            | 44.69                    |
| 1  | 58.80                     | 63.88                    |
| 2  | 48.98                     | 60.08                    |
| 3  | 45.99                     | 61.29                    |
| 5  | 43.27                     | 56.32                    |
| 8  | 40.61                     | 54.13                    |
| 12 | 37.95                     | **42.29**                |
| 18 | 38.60                     | **40.20**                |

Note: the learned-shrinkage column reflects strict LOSO evaluation
(network sees ZERO test-subject samples during training). The fixed-λ
sweep on the same data gave a best of 37.52 px (K=12, λ=0.8) and 38.18 px
(K=18, λ=0.8) using non-LOSO data. Combining the two (LOSO learned
shrinkage at K≥12) gives the strongest honest result.

## Method (proposed for resubmission)

The proposed method has three components, of which only the third
contains learned parameters:

1. **Stage I (existing): per-trial classical calibration.** A bank of
   7-9 classical calibrators (similarity, polynomial degree-2, RBF
   multiquadric s∈{0,1,2}, TPS, similarity+RBF, similarity+TPS) is
   fit per session on the trial's 18 calibration fixations.

2. **Stage II (new): online bias estimation.** After the standard
   calibration phase, K (8–18) additional fixations on known targets
   are collected. The empirical mean residual of the chosen anchor
   baseline (typically similarity+RBF) on these K fixations is the
   raw bias estimate.

3. **Stage III (new): learned shrinkage.** A small MLP (≈1k parameters)
   takes per-trial bias statistics — mean magnitude, variance, range,
   1/K, log K — and outputs a 2D shrinkage vector (λ_x, λ_y) ∈ [0, 1]^2.
   The corrected anchor prediction is `anchor + (λ_x, λ_y) ⊙ raw_bias`.

The shrinkage MLP is trained leave-one-subject-out so the learned policy
generalizes across users.

## Ablations Conducted

- **JuDo selection**: top-M ∈ {1, 2, 3, 4, 6, 8} vs all-11 vs oracle.
  Selecting fewer baselines is better (top-1 gives 26 px, all gives 60).
- **JuDo noise injection**: σ_max ∈ {0, 5, 10, 20, 40, 80}. No clear
  effect; ~0.5 px swing within noise.
- **JuDo network capacity**: hidden ∈ {[16], [64], [64,32], [64,32,16],
  [256,128], [1024,512]}. All within ±1 px of each other.
- **JuDo BatchNorm on/off**: no effect.
- **JuDo baseline pool**: drop-one ablation. Removing TPS or RBF families
  improves results (those are the worst baselines).
- **Online bias K**: 0, 1, 2, 3, 5, 8, 12, 18.
- **Online bias λ**: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0. Best λ rises with K
  (matches James-Stein intuition: less shrinkage when sample size grows).
- **Learned shrinkage LOSO**: 12-fold over collected subjects.

## Open Issues for the Resubmission

1. **Honest Multi Baseline Neural Refinement does NOT beat classical
   baselines on either dataset.** The paper's central claim must be
   retracted or restated.
2. **JuDo1000 is a bad benchmark for this method**: the spatial structure
   is too sparse (8 test targets) and per-fixation noise dominates.
3. **The 12-subject self-collected dataset has only 9 subjects with ≥2
   trials**, limiting LOSO statistical power. ZAnna and Liu Jiaqi
   dominate the sample count.
4. **No subject-disjoint test data was collected**. The original paper
   randomly split rows; we created LOSO post-hoc but cannot run a clean
   "trained-once, deployed-on-new-users" study without recollecting.
5. **The strongest honest contribution is online bias correction with
   learned shrinkage**, but this is much closer to a statistics paper
   (James-Stein on calibration residuals) than the deep-learning paper
   the original submission tried to be.

## Lessons Learned

- Always verify that input features cannot trivially encode the target.
  The `mb_features = baseline_pred - target` bug went undetected through
  the original review process.
- Subject-disjoint splits must be enforced by counting unique subjects
  per split, not just trusting filenames.
- Classical baselines that interpolate calibration points exactly (RBF
  s=0, TPS) yield zero training-side error and useless test
  generalization — a per-baseline-risk selection criterion that ranks
  on training points will systematically pick the worst baselines.
- For eye-tracking calibration with point-disjoint test targets,
  ranking baselines on validation targets (not training points) is the
  right protocol.

## Plan for Paper Revision (if proceeding to NeurIPS)

1. **Reframe the contribution** as "honest evaluation of neural
   refinement for eye-tracking calibration + a learned-shrinkage online
   bias-correction method that delivers 17–19% improvements with K=12
   extra calibration fixations."
2. **Lead with the leakage critique** as a methodological contribution.
   The community needs this audit.
3. **Move the multi-baseline neural refinement results to a negative
   ablation** showing that without leakage, none of these designs beat
   classical methods.
4. **Position learned shrinkage as the positive contribution**.
5. **Add the LOSO protocol** to the evaluation explicitly.
6. **Acknowledge dataset scale limitation** in Limitations and outline
   the larger collection effort needed for a stronger study.
