# Synthetic shrinkage benchmark — protocol

## Hypothesis

H1 (theory matches practice). In the homoscedastic Gaussian-bias model
the closed-form Bayes-optimal scalar shrinkage λ* = σ_b² / (σ_b² + σ_ε²/K)
achieves the lowest expected L2 across context sizes K. The learned
scalar shrinkage MLP recovers this minimum to within Monte-Carlo noise.

H2 (heteroscedasticity). When per-trial noise variance varies, a single
closed-form λ misspecified to the population-mean σ_ε² is suboptimal,
and the learned shrinkage MLP — which conditions on per-trial sample
statistics — strictly improves on it.

## Setup

- 2 000 train / 300 val / 500 test trials.
- 30 samples per trial.
- Per-trial bias b ~ N(0, σ_b² I_2) with σ_b = 30 px.
- Homoscedastic experiment: σ_ε = 20 px.
- Heteroscedastic experiment: per-trial σ_ε ~ U[10, 35].

## Sweeps

K ∈ {1, 2, 3, 5, 8, 12, 18}.

## Expected outcome

| K | closed-form L2 | learned L2 | gap |
|---|----------------|------------|-----|
| 1 | high           | ≈ closed   | ≈ 0 |
| 18 | low           | ≈ closed   | ≈ 0 |

For heteroscedasticity, learned should beat misspecified closed form
by a clear margin at K ≥ 5.

## Pre-registration

Locking the protocol before running the experiment.
