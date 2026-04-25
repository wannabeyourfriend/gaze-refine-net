# Research Log

## 2026-04-25 — Bootstrap

### Context
- Paper "Lightweight Neural Refinement for Drift Calibration in Eye Tracking Systems" rejected from ICML 2026 (scores 2/3/3/3).
- User asked to first reconcile codebase/paper mismatch, verify results, then improve pipeline and rewrite.

### P1 Audit — initial findings (codebase vs paper)

**MISMATCH 1 — Noise-aware training claimed but disabled in production config.**
- Paper §3.3 ("Stage III"): describes additive Gaussian noise on baseline offsets, sigma ~ U[0, sigma_max], as central to robustness.
- Paper Table `tab:ablation_training`: "w/ Data Augmentation" is the headline run.
- Code: every `multi_baseline_s1*.yaml` config has BOTH
    - `augmentation: enabled: false`
    - `sim_rbf_perturbation: enabled: false`
  → headline 0.96°/42.3 px was produced without noise-aware training.
- Furthermore, `sim_rbf_perturbation` (the only baseline-offset noise mechanism that exists in code, model.py:301-343) only operates in `cascade` mode, not `multi_baseline`. There is no implementation of "additive Gaussian on each baseline offset, independently sampled sigma" as claimed in the paper.

**MISMATCH 2 — Accuracy-conditioned selection (top-M) is not implemented.**
- Paper §3.2: per-baseline error E_k computed on training calibration points, top-M=4 selected.
- Code (`multi_baseline_s1.yaml`): all 7 baselines are concatenated as input features unconditionally; there is no per-trial sorting by E_k, no slot mapping, no "rank-slot" semantics described in rebuttal.
- The model_type "multi_baseline" in `src/model.py` simply concatenates all listed baseline residuals (mb_features) into the input; selection is config-time, not trial-time.

**MISMATCH 3 — Architecture differs.**
- Paper §3.3: "[64, 32, 16] hidden, ReLU, BatchNorm".
- Code (`multi_baseline_s1.yaml`): `hidden_dims: [1024, 512]`, `dropout: 0.15`. No BatchNorm in `GazeRefineNet`. Has a `ResidualBlock` tail not mentioned in paper.

**MISMATCH 4 — Number of baselines.**
- Paper §3.2.3: K=7 baselines (affine, RBF, affine-RBF, poly-2,3,4). That's only 6 listed, not 7. Likely intent: similarity, RBF, sim+RBF, poly-2, poly-3, poly-4, plus possibly TPS — paper text inconsistent.
- Code (`multi_baseline_s1.yaml`): 7 baselines listed but they are: sim_rbf_gaze, pred_sim_rbf_multiquadric_s0.0, pred_sim_tps, pred_sim_pwa, pred_similarity, pred_poly, pred_tps. None are RBF-only or affine-only as described in the paper; all are "sim_X" combinations or polynomial.

**MISMATCH 5 — "Multi-baseline" features are stored as residuals (baseline - target), which leaks the target at training time.**
- `model.py` line 254: `baseline_residual = baseline_data - targets_px`. The per-trial baseline error vector includes `target_px`, the ground truth.
- This is potential label leakage if interpretation is naive. We need to confirm: are these used for the network in a way that hides the target at test? Looking again — they ARE leaked. Test set targets are visible in `mb_features`. This needs urgent verification.

### Action items
- Verify whether mb_features label leakage actually contaminates evaluation. If yes, the JuDo 5.8 px headline number is invalid.
- Run multi_baseline_s1.yaml with current code to reproduce 42.3 px and confirm/refute.
- Build a clean implementation of paper's claimed pipeline (top-M selection, noise injection on baseline offsets, [64,32,16] arch).
- Re-evaluate.
