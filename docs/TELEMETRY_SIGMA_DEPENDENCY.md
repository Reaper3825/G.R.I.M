# Telemetry Lattice: σ (sigma) as the Common Variable

## Problem

Across all lattice-derived metrics — volatility, rate of change, smoothed delta bar, level, momentum, and even the adaptive outlier threshold — the same variable appears as a critical dependency: **σ (sigma)**, specifically through `sigma_prev`.

## Affected Metrics & Their σ Dependency

Every non-trivial lattice output passes through `sigma` or `sigma_prev` as either a denominator or a gate:

| Metric | Formula | σ Dependency |
|--------|---------|-------------|
| `delta_hat` (normalized slope) | `(x_t - μ_prev) / (σ_prev + ε)` | **denominator** |
| `delta_bar` (smoothed delta) | `β_δ · δ̄ + (1-β_δ) · delta_hat` | inherits from `delta_hat` |
| `p` (momentum/direction) | `β_δ · p + (1-β_δ) · sign(delta_hat)` | inherits from `delta_hat` |
| `v_sigma` (volatility-of-vol) | `β_v · v_σ + (1-β_v) · (σ - σ_prev)²` | **direct difference** |
| `k_out` (adaptive threshold) | `k₀ · (1 + α_v · √v_σ)` | inherits from `v_sigma` |
| `c_out` (outlier cutoff) | `μ + k_out · σ` | **direct multiplier** |
| `r_out` (outlier frequency) | `sigmoid((x_t - c_out) / (σ + ε))` | **denominator + via c_out** |
| `ℓ_out` (persistence/level) | EMA of `w_out` | inherits from `r_out` gate |
| `μ_ex` (excess magnitude) | `EMA(max(0, x_t - c_out))` | inherits from `c_out` |
| `σ̃` (normalized volatility) | `σ / (|μ| + ε)` | **numerator** |

## The Dependency Graph

```
raw_observation (x_t)
    │
    ├──→ μ  (EMA mean)
    │      │
    │      ├──→ σ = √(m2 - μ²)  ◄── THE COMMON VARIABLE
    │      │      │
    │      │      ├──→ σ_prev  (stored from previous step)
    │      │      │      │
    │      │      │      ├──→ delta_hat = (x_t - μ_prev) / (σ_prev + ε)
    │      │      │      │      ├──→ delta_bar  (smoothed delta)
    │      │      │      │      └──→ p  (momentum)
    │      │      │      │
    │      │      │      └──→ v_sigma = EMA((σ - σ_prev)²)
    │      │      │             └──→ k_out = k₀(1 + α_v · √v_σ)
    │      │      │                    └──→ c_out = μ + k_out · σ
    │      │      │                           ├──→ r_out (outlier freq)
    │      │      │                           ├──→ ℓ_out (persistence)
    │      │      │                           └──→ μ_ex (excess)
    │      │      │
    │      │      └──→ σ̃ = σ / (|μ| + ε)
    │      │
    │      └──→ μ_a, σ_a  (slow anchor — also σ-dependent)
    │             ├──→ δμ = μ - μ_a
    │             └──→ δσ = σ - σ_a
    │
    └──→ (only raw grad_norm escapes σ normalization)
```

**σ and σ_prev are the single shared variable that every derived lattice metric flows through.**

## Why This Matters for Rho Variance

The rho diagnostic showed ~21% CV with near-zero autocorrelation (lag-1 = 0.13). The lattice metrics computed from rho all exhibited the same pattern of high variance, specifically because:

1. σ is computed from an EMA (β_mu = 0.95), so it has a ~20-step memory
2. When the raw observation (rho) jumps 0.15–0.26 step-to-step, σ tracks that noise
3. Every downstream metric divides by or multiplies with this noisy σ estimate
4. The noise in σ_prev amplifies through `delta_hat = Δx / σ_prev` — when σ_prev happens to be small (after a quiet period), the same raw delta produces a large normalized spike

## Initialization Transient

At step 1: `sigma_prev = 0`, `sigma = 0` (single observation → zero variance)
At step 2: `delta_hat = (x_2 - μ_1) / (0 + ε)` → divides by epsilon → massive spike

This explains the violent transients visible in the first ~2500 steps of all lattice plots.
