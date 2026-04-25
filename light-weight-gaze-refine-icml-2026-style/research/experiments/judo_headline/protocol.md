# JuDo headline — paper-faithful execution

## Hypothesis (locked before run)

The paper's claimed pipeline (Stage I baselines + Stage II top-M=4 selection by
calibration risk + Stage III [64,32,16] MLP with BatchNorm and noise-aware
training) will produce:

- **Test L2 mean** somewhere between the best classical baseline (~22 px) and
  the leakage-driven 5.82 px headline. We expect **15–22 px** if the method
  truly contributes signal beyond classical baselines on JuDo, and **22 px or
  worse** if it does not (since the honest features without leakage barely
  beat raw gaze in the linear-probe diagnostic).

## Configuration

- Data: `data/prepared/judo_1000_split_no_leakage/{train,val,test}.csv`
- Baselines pool (11): similarity, poly, rbf_multiquadric_s{0,1,2},
  tps, sim_rbf_multiquadric_s{0,1,2}, sim_tps, sim_pwa.
- Selection: top-M=4 by calibration risk on a held-out 50% subset of training
  rows.
- Architecture: hidden_dims=[64,32,16], BatchNorm, ReLU, no dropout.
- Noise injection: sigma_max=20 px, prob=1.0 (matches paper's sigma_max=0.5°
  given JuDo's pixel-per-degree).
- Optim: AdamW, lr=3e-4, wd=0.01, batch=64, epochs=200, seed=1047.
- Coord scale: 100.

## Diagnostic comparison run

We also run, with identical seeds and code path:

- **Leaky** (reproduce paper bug): `mb_features = baseline - target`, per-split
  z-score. Same architecture and optimizer. Expected: test L2 ≈ 5–6 px.

The gap between honest and leaky tells us how much of the paper's reported
improvement is leakage-driven.

## Predicted outcome

- Honest_topM4_noise: test_l2_mean ≈ 18–25 px
- Leaky_topM4_noise: test_l2_mean ≈ 5–8 px
- Gap >> 10 px confirms the leakage hypothesis is correct.
