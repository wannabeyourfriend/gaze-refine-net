# JuDo systematic ablation

## Hypothesis

Under the honest pipeline (no `mb_features = baseline - target` leak), the
paper's individual design components — top-M selection, noise injection,
[64,32,16] architecture — should each show measurable test-L2 improvement
if they encode a real ML contribution. We expect at least one component
to push the network below the best classical baseline (~22 px).

## Ablation axes (each holds others fixed at honest paper-faithful defaults)

Defaults: `topM=4`, `noise_sigma_max=20`, `hidden=[64,32,16]`, BN=on, no dropout.

1. **Selection strategy**: {all, topM-1, topM-2, topM-3, topM-4, topM-6, topM-8, oracle-topM-4}
2. **Noise sigma_max (px)**: {0, 5, 10, 20, 40, 80}
3. **Network capacity**: hidden_dims in {[16], [64], [64,32], [64,32,16] (paper), [256,128], [1024,512] (original code)}
4. **BatchNorm**: {on, off}
5. **Baseline pool**: drop one family (similarity, poly, RBF*, TPS, SimRBF*, SimTPS, SimPWA) at a time
6. **Seed sensitivity**: 5 seeds for the headline configuration

## Outputs

- `results.csv` and `results.json` — one row per run with all metrics.
- `summary.md` — markdown table per ablation axis.
