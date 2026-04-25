# Findings — gaze-refine-net ICML 2026 Resubmission

_Last updated: 2026-04-25_

## Current Understanding

The paper's headline numbers are **not reproducible from the code as a clean ML pipeline**. The reported 0.96°/42.3 px on the "self-collected dataset" and 5.8 px on JuDo1000 arise from a combination of (i) a label-leaking input feature, (ii) a non-subject-disjoint split, and (iii) a single-subject training/test set, none of which match the methodological description in the manuscript.

## Critical Mismatches Identified

### M1 — Headline result uses single-subject data (not 12-subject subject-disjoint splits)

- **Paper claim**: "12 participants, 6/3/3 subject-level train-validation-test split, refinement network never observes any test-subject data."
- **Reality**: The config that produced the headline (`multi_baseline_s1.yaml`) points at `data/prepared/s1_for_training/`. All three splits in that directory contain only one subject (`Liu Jiaqi`).
- The directory `data/prepared/all_trials_split/` contains 12 subjects but is split by trial (every subject appears in train AND test); it is NOT subject-disjoint either.
- A truly subject-disjoint split (e.g. 6/3/3) does not appear to exist on disk.

### M2 — Label leakage in `multi_baseline` features

- `src/model.py:254`: `baseline_residual = baseline_data - targets_px` — i.e. each baseline's contribution to the network input is computed using the target the network is supposed to predict.
- The features are then z-score normalized **within each split** (`src/model.py:265–268`), which superficially conceals magnitude but preserves enough signal that the model overfits on the leakage.
- Empirically: with this exact pipeline, train L2 collapses to ~12 px while test L2 sits at ~42 px (matches paper's 42.3 px). Without the leaky features (using the published baselines themselves), the best classical baseline (Affine+RBF) is 1.03°/45.5 px — so the supposed neural improvement (42.3 px) is mostly an artifact of leakage-driven training-time advantage, not a real generalization gain.

### M3 — Noise-aware training is described but disabled

- Paper §3.3 makes additive Gaussian noise on baseline offsets a central design pillar; the related ablation table (`tab:ablation_training`) presents "w/ Data Augmentation" as the headline configuration.
- The only such mechanism in code (`SimRBFPerturbationConfig` / `_apply_sim_rbf_perturbation`) is gated to `model_type == 'cascade'` (model.py:314) and is never invoked from the multi-baseline path.
- Every `multi_baseline_s1*.yaml` config has `augmentation.enabled: false` and `sim_rbf_perturbation.enabled: false`.
- Consequence: the headline number was produced with NO noise injection of any kind.

### M4 — Top-M selection is described but not implemented

- Paper §3.2: per-baseline calibration error E_k computed on calibration points, top-M=4 baselines selected for refinement.
- Code: all baselines listed in the YAML are concatenated unconditionally; there is no per-trial sorting, no rank-slot mapping, no E_k calculation.
- The rebuttal-promised "fixed-rank-slot semantics" experiment cannot be performed because nothing is being ranked.

### M5 — Architecture mismatch

- Paper §3.3: hidden_dims = [64, 32, 16], BatchNorm, ReLU.
- Code (`multi_baseline_s1.yaml`): hidden_dims = [1024, 512], dropout 0.15, ResidualBlock tail. No BatchNorm anywhere in `GazeRefineNet`.

### M6 — Baseline set inconsistency

- Paper text: "K=7: affine, RBF, affine-RBF, polynomials of orders 2/3/4" (only 6 listed).
- Code config: 7 baselines, none of which are the bare affine or RBF described in the paper. They are all sim+X variants (sim_rbf, sim_tps, sim_pwa, similarity, poly, tps, sim_rbf_s0.0).
- Tables 1 and 2 in the paper show "Affine" and "RBF" as standalone baselines — these baselines are NOT in the trained input set.

### M7 — JuDo evaluation: baseline-level fix exists, but `mb_features` leakage still applies

- `data/prepared/judo_1000_split_no_leakage/` correctly splits JuDo by **calibration target points** (33 train / 7 val / 8 test, all spatially disjoint). Baselines are fit on training points only — this is correct.
- The wandb log of run `20260129_023313` shows the headline 5.82 px JuDo result was indeed produced from this no-leakage split (verified). Good.
- **But** the multi-baseline INPUT FEATURE construction in `model.py:254` still computes `mb_features = baseline_pred - target` for every sample, including test samples. So even with point-disjoint splits, the network's input on test contains the test target subtracted out.
- Empirical proof (replicated headline pipeline):
  - Leaky path (matches code, produces headline): **best test L2 = 2.61 px** (matches paper's 5.82 from best epoch)
  - Honest path (mb_features = raw baseline preds, shared normalization): **best test L2 = 22.06 px**
  - The honest number ≈ the Original Gaze baseline (22.0 px in Table 2).
  - **The entire 73.6% JuDo improvement is leakage.** With honest features, the network does not beat raw gaze.

So both datasets are affected by the same `mb_features = baseline - target` bug. The s1 dataset compounds it with non-subject-disjoint splits and a single training subject.

## Patterns and Insights

- The code has been iterated heavily (multiple `multi_baseline_s1_*.yaml` variants, multiple checkpoint dirs, JuDo-specific re-derivations). The team appears to have caught some leakage problems but not unified the fix across datasets.
- The headline 42.3 px on the self-collected dataset is suspiciously close to the best classical baseline's test-set error (~40 px for Affine+RBF), suggesting the network is essentially regressing the baselines themselves once leakage is partially defeated by per-split normalization.
- The rebuttal claims about subject-disjoint evaluation and noise-aware training are not supported by the code.

## Lessons and Constraints

- Never trust input features that depend on the target. Per-split z-scoring does not safely conceal them — the local model can still exploit residual structure.
- Subject-level splits must be verified explicitly by counting unique subjects per split.
- Reviewers' concerns about "novelty being too thin" are now compounded by reproducibility concerns. Resubmission must rebuild from the ground up.

## Open Questions

1. Is there a private 12-subject subject-disjoint split somewhere not committed?
2. Was the 5.8 px JuDo headline computed with or without leakage?
3. Can we collect more subjects to address dataset scale concerns?
4. What is a defensible methodological framing that the reviewers might accept (Bayesian model averaging? mixture of experts over geometric calibrators? something else)?

## Plan Forward

1. Build a clean, reproducible pipeline:
   - Subject-disjoint splits (real ones) on the 12-subject data.
   - Baselines computed from training calibration points only (no leakage).
   - Inputs: only `(orig_x, orig_y, baseline_pred_x, baseline_pred_y)` — never `(baseline_pred - target)`.
   - Implement top-M selection and noise injection as described.
2. Re-evaluate ALL headline numbers honestly.
3. Decide which paper claims survive and which need to change.
4. Improve the pipeline (e.g., reliability-weighted softmax over baselines, calibration-confidence-aware loss).
5. Rewrite the manuscript.
