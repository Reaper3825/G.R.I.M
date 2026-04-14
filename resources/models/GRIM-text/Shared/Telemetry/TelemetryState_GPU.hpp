#pragma once
/**
 * @file TelemetryState_GPU.hpp
 * @brief GPU-resident telemetry state for hierarchical tracking
 * 
 * DESIGN: Pure GPU state, no CPU round-trips
 * Math follows the exact specification provided by user
 * 
 * STATE VARIABLES (persistent on GPU):
 *   μ, m2, σ, σ̃          - Fast magnitude statistics
 *   μ_a, σ_a, δμ, δσ    - Slow anchor and drift
 *   v_σ                 - Volatility-of-volatility
 *   Δ̄, p                - Normalized slope and directional bias
 *   r_out, ℓ_out, μ_ex  - Outlier frequency, persistence, excess
 * 
 * INPUT: Single scalar x_t per timestep
 * OUTPUT: 10-element telemetry vector T_t
 */

#include <cuda_runtime.h>
#include <cstdint>

namespace GRIM::Telemetry {

//=============================================================================
// HYPERPARAMETERS - Tunable via config
//=============================================================================

struct TelemetryHyperParams {
    // EMA decay rates
    float beta_mu = 0.95f;      // Fast mean/variance
    float beta_a = 0.995f;      // Slow anchor (higher = slower)
    float beta_delta = 0.90f;   // Slope EMA
    float beta_r = 0.85f;       // Outlier frequency
    float beta_run = 0.80f;     // Persistence EMA
    float beta_v = 0.90f;       // Volatility-of-volatility
    
    // Outlier detection
    float k_out0 = 2.5f;        // Base threshold (sigmas)
    float alpha_v = 1.5f;       // Volatility inflation factor
    
    // Numerical stability
    float epsilon = 1e-7f;      // Floor for divisions
    
    // Validation
    bool strict_mode = true;    // Fail loud on NaN/Inf
};

//=============================================================================
// STATE (persistent GPU memory, one per metric stream)
//=============================================================================

struct TelemetryState {
    // Fast magnitude
    float mu = 0.0f;            // EMA mean
    float m2 = 0.0f;            // EMA second moment
    float sigma = 0.0f;         // √(m2 - μ²)
    float sigma_tilde = 0.0f;   // σ / (|μ| + ε) - normalized volatility
    
    // Slow anchor
    float mu_a = 0.0f;          // Anchor mean
    float sigma_a = 0.0f;       // Anchor std
    float delta_mu = 0.0f;      // μ - μ_a (drift)
    float delta_sigma = 0.0f;   // σ - σ_a (volatility drift)
    
    // Meta-volatility
    float v_sigma = 0.0f;       // EMA of (σ - σ_prev)²
    float sigma_prev = 0.0f;    // Previous σ for derivative
    
    // Trend
    float delta_bar = 0.0f;     // EMA of normalized slope
    float p = 0.0f;             // Directional bias in [-1, +1]
    float mu_prev = 0.0f;       // Previous μ for slope
    
    // Outliers
    float r_out = 0.0f;         // Frequency EMA
    float ell_out = 0.0f;       // Persistence EMA
    float mu_ex = 0.0f;         // Excess magnitude EMA
    
    // Adaptive threshold
    float k_out = 2.5f;         // Current threshold (adapts with v_σ)
    float c_out = 0.0f;         // Current cutoff = μ + k_out*σ
    
    // Bias correction (Adam-style: corrected = raw / (1 - beta^t))
    float mu_raw = 0.0f;        // Uncorrected EMA of x_t
    float m2_raw = 0.0f;        // Uncorrected EMA of x_t²
    float beta_mu_power = 1.0f; // beta_mu^t — decays each step
    
    // Metadata
    uint32_t step_count = 0;    // Total updates
    uint32_t initialized = 0;   // 1 after first update
};

//=============================================================================
// OUTPUT TELEMETRY VECTOR (10 elements)
//=============================================================================

struct TelemetryVector {
    float mu;                   // Current magnitude baseline
    float sigma_tilde;          // Scale-normalized volatility
    float v_sigma;              // Meta-volatility
    float delta_bar;            // Directional trend strength
    float p;                    // Directional bias (up/down)
    float r_out;                // Outlier frequency
    float ell_out;              // Outlier persistence
    float mu_ex;                // Excess severity
    float delta_mu;             // Mean drift vs anchor
    float delta_sigma;          // Volatility drift vs anchor
};

//=============================================================================
// MULTI-SCALE LATTICE LEVEL
//=============================================================================

struct LatticeLevelState {
    TelemetryState state;       // Current state for this level
    uint32_t stride;            // Update every n_k steps (2^k)
    uint32_t last_update;       // Last global step this level updated
};

//=============================================================================
// ERROR CODES
//=============================================================================

enum class TelemetryError : int32_t {
    OK = 0,
    ERR_NAN_IN_INPUT = 1,
    ERR_INF_IN_INPUT = 2,
    ERR_NAN_IN_STATE = 3,
    ERR_KERNEL_LAUNCH = 4,
    ERR_NULL_POINTER = 5,
    ERR_INVALID_PARAMS = 6
};

const char* getTelemetryErrorMessage(TelemetryError err);

} // namespace GRIM::Telemetry
