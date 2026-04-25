# Figure 2 — Leakage diagnosis (NEW)

## Why this figure
Reviewers and readers should immediately see why a previously published
"5.8 px on JuDo / 42.3 px on the self-collected dataset" cannot be
reproduced under a clean evaluation. This figure is the visual core of our
methodological audit.

## Layout
Two side-by-side bar groups, both inside one figure (\linewidth wide,
~3 in tall). Subplot titles: "(a) JuDo1000" and "(b) Self-collected".

## Panel (a) — JuDo1000

X-axis (categorical, in this order):
1. `origin_gaze` (raw eye tracker, no calibration)
2. `pred_similarity` (best classical baseline)
3. `Honest paper method` (top-M=4 + noise20, [64,32,16] BN MLP)
4. `Leaky paper method` (the bug: mb_features = baseline − target)

Y-axis: Test mean L2 error (px), 0 to 35 px.

Bars (numerical values, all from `experiments/judo_headline/`):
- origin_gaze ........ 21.96
- pred_similarity .... 21.66
- Honest paper method 30.46
- Leaky paper method . 4.21  ← annotate "matches published 5.82"

Color: bars 1-3 in greys / muted blues. Bar 4 in red with a hatched fill
to visually mark it as flawed. Add a horizontal dashed line at the
similarity baseline (21.66 px) labelled "best classical".

Above bar 4, add a callout text box with an arrow:
> "Bug: `mb_features = baseline − target`
>  reproduces the published 5.8 px headline."

## Panel (b) — Self-collected

X-axis (same ordering):
1. `origin_gaze` ........ 67.25 px
2. `pred_sim_rbf_s2.0` (best classical) .... 45.53 px
3. `Honest paper method` .... ~46 px (we will use 45.06 from v2)
4. `Leaky paper method` .... 12 (training) and 42 (test) — choose to show 12 with hatched bar plus a small inset showing test = 42

For panel (b) we include the same callout: "Same bug, milder appearance
because per-split z-score normalization mostly conceals targets at test
time, but corrupts training."

## Annotations
- Caption: lower case, lower-cased except first word.
- Both panels share a single legend on the right: "raw" / "classical" /
  "honest neural" / "leaky (paper)".

## Color & style
- raw: dark grey
- classical: navy
- honest neural: teal
- leaky: red, hatched
- error bars: ±1 std across seeds (we have one seed, so for honest we use
  ±std across the 5-seed sensitivity row from the JuDo ablation, and for
  leaky we also report a single seed but with hatch indicating
  reproducibility caveat).

## Caption text
> Honest evaluation eliminates the published headline. We replace
> `mb_features = baseline_pred − target` (the implementation in
> `apps/neural_refine/src/model.py:254`, which subtracts the regression
> target from the network's input feature) with raw baseline predictions
> normalized by training-set statistics. (a) On JuDo1000, the honest
> pipeline yields 30.5 px — *worse* than the 21.7 px similarity baseline.
> The leaky pipeline reproduces the published 5.8 px. (b) On our 12-subject
> dataset the same bug inflates training-time signal; the honest neural
> method matches the strongest classical baseline rather than improving
> over it.
