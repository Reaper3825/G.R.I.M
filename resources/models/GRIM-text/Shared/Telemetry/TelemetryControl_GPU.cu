/**
 * @file TelemetryControl_GPU.cu
 * @brief Production-grade telemetry-driven adaptive control (GPU-native)
 * 
 * ARCHITECTURE:
 *   - Single fused kernel computes entire control decision on GPU
 *   - Only ONE 48-byte D2H transfer per evaluate() call
 *   - State persists GPU-resident between calls
 *   - NO per-call allocations (Rule 22 compliant)
 */

#include "TelemetryControl_GPU.hpp"
#include "TelemetryLattice_Internal.hpp"
#include "../HyperParameters/HyperParameters_GPU.hpp"
#include "../StreamController/StreamController_GPU.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <iostream>

namespace GRIM::Telemetry {

//=============================================================================
// STRING HELPERS (CPU-side)
//=============================================================================

const char* getSpikeSeverityName(SpikeSeverity severity) {
    switch (severity) {
        case SpikeSeverity::None:     return "none";
        case SpikeSeverity::Mild:     return "mild";
        case SpikeSeverity::Moderate: return "moderate";
        case SpikeSeverity::Severe:   return "severe";
        default: return "unknown";
    }
}

const char* getControlActionName(ControlAction action) {
    switch (action) {
        case ControlAction::Continue:           return "continue";
        case ControlAction::ScaleGradients:     return "scale_gradients";
        case ControlAction::ExtendCooldown:     return "extend_cooldown";
        case ControlAction::SkipStep:           return "skip_step";
        case ControlAction::TriggerSoftRestart: return "soft_restart";
        case ControlAction::InjectPlateauNoise: return "inject_plateau_noise";
        case ControlAction::FatalError:         return "fatal_error";
        default: return "unknown";
    }
}

//=============================================================================
// DEVICE HELPER FUNCTIONS
//=============================================================================

__device__ __forceinline__ float d_computeNormalizedGrad(
    float raw_grad_norm, 
    int valid_tokens, 
    float reference_tokens
) {
    // CRITICAL: backward() scales each gradient by 1/valid_tokens, but the L2 norm
    // (sum of squared gradients) is still proportional to sqrt(valid_tokens) because
    // fewer tokens = less gradient energy overall.
    // 
    // To fairly compare gradients across batches with different token counts, we must
    // normalize by sqrt(reference_tokens/valid_tokens). This makes grad_norm independent
    // of batch size, so TelemetryControl can detect true pathologies vs token variance.
    //
    // Example: 93 tokens with grad_norm=48.5 → normalized = 48.5 * sqrt(720/93) ≈ 135
    //          3114 tokens with grad_norm=1558 → normalized = 1558 * sqrt(720/3114) ≈ 747
    // Both are now comparable on same scale.
    
    if (valid_tokens <= 0 || reference_tokens <= 0.0f) return raw_grad_norm;
    const float scale = sqrtf(reference_tokens / static_cast<float>(valid_tokens));
    return raw_grad_norm * scale;
}

__device__ __forceinline__ SpikeSeverity d_computeSpikeSeverity(
    float current_grad, 
    float baseline_grad,
    const TelemetryControlConfig* cfg
) {
    if (baseline_grad < 1e-8f) return SpikeSeverity::None;
    
    const float ratio = current_grad / baseline_grad;
    
    if (ratio >= cfg->spike_severe_threshold) return SpikeSeverity::Severe;
    if (ratio >= cfg->spike_moderate_threshold) return SpikeSeverity::Moderate;
    if (ratio >= cfg->spike_mild_threshold) return SpikeSeverity::Mild;
    return SpikeSeverity::None;
}

__device__ __forceinline__ float d_computeVolatilityDamping(
    float v_sigma,
    const TelemetryControlConfig* cfg
) {
    if (v_sigma <= cfg->volatility_damping_threshold) return 1.0f;
    const float excess = v_sigma - cfg->volatility_damping_threshold;
    float damping = 1.0f / (1.0f + excess);
    return fmaxf(damping, cfg->max_volatility_damping);
}

__device__ __forceinline__ float d_computeDecayBoost(
    float current_mu, 
    float baseline_mu,
    const TelemetryControlConfig* cfg
) {
    if (baseline_mu < 1e-8f || current_mu <= 0.0f) return 1.0f;
    const float health = current_mu / baseline_mu;
    if (health >= cfg->gradient_decay_threshold) return 1.0f;
    float boost = 1.0f / health;
    return fminf(boost, cfg->max_decay_boost);
}

__device__ __forceinline__ float d_computeProgressBoost(
    float delta_mu,
    float sigma_tilde,
    const TelemetryControlConfig* cfg
) {
    // Boost gradients when loss is dropping rapidly (negative delta_mu)
    // This accelerates learning during productive phases
    if (sigma_tilde < 1e-8f || delta_mu >= 0.0f) return 1.0f;
    const float progress_magnitude = -delta_mu / sigma_tilde;  // Negative = improvement
    if (progress_magnitude < cfg->progress_boost_threshold) return 1.0f;
    // Linear boost: more progress = more boost, capped at max_progress_boost
    const float boost = 1.0f + (progress_magnitude - cfg->progress_boost_threshold) * 0.1f;
    return fminf(boost, cfg->max_progress_boost);
}

__device__ __forceinline__ bool d_checkAccumulationBug(
    float grad_norm, 
    float loss,
    const TelemetryControlConfig* cfg
) {
    return (loss > cfg->loss_threshold_for_grad_check && 
            grad_norm < cfg->min_grad_for_nonzero_loss);
}

__device__ __forceinline__ bool d_detectOutlierRegime(
    float r_out, 
    float ell_out,
    const TelemetryControlConfig* cfg
) {
    return (r_out > cfg->outlier_frequency_trigger &&
            ell_out > cfg->outlier_persistence_trigger);
}

__device__ __forceinline__ bool d_detectAnchorDrift(
    float delta_mu, 
    float sigma_tilde,
    const TelemetryControlConfig* cfg
) {
    // CRITICAL: Only trigger on UPWARD drift (loss increasing)
    // Rapid loss decrease (negative delta_mu) is GOOD, not pathological
    if (sigma_tilde < 1e-8f || delta_mu <= 0.0f) return false;
    const float drift_threshold = cfg->anchor_drift_sigma_multiplier * sigma_tilde;
    return delta_mu > drift_threshold;  // Positive drift only
}

//=============================================================================
// FUSED CONTROL DECISION KERNEL
//
// Single thread kernel - decision logic is sequential, not data-parallel.
// Benefits: 
//   - Reads telemetry directly from GPU memory (no D2H for vectors)
//   - Updates state GPU-resident
//   - Only final ControlDecision needs D2H transfer
//=============================================================================

__global__ void controlDecisionKernel(
    const LatticeLevelState* __restrict__ d_lattice_levels,  // Read-only observation
    int num_lattice_streams,
    const TelemetryControlConfig* __restrict__ cfg,
    TelemetryControlState_GPU* __restrict__ state,
    const ControlKernelInput* __restrict__ input,
    ControlDecision* __restrict__ decision
) {
    // Single-thread kernel
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // Initialize decision to defaults
    // CRITICAL: CUDA device memory is NOT automatically initialized!
    // Member initializers in C++ structs don't apply to cudaMalloc'd memory.
    // All fields MUST be explicitly set here to avoid garbage values.
    decision->action = ControlAction::Continue;
    decision->grad_scale_factor = 1.0f;
    decision->cooldown_extension = 0;
    decision->spike_severity = SpikeSeverity::None;
    decision->flags = 0;
    decision->spike_ratio = 0.0f;
    decision->volatility_damping = 1.0f;
    decision->decay_boost = 1.0f;
    decision->progress_boost = 1.0f;  // FIX Issue #19: Was uninitialized!
    decision->normalized_grad = 0.0f;
    decision->error_code = 0;
    decision->_pad0 = 0;
    
    // Increment step counter
    state->step_count++;
    
    // Unpack input
    const float raw_grad_norm = input->raw_grad_norm;
    const float loss = input->loss;
    const float avg_seq_len = input->avg_seq_len;
    const int valid_tokens = input->valid_tokens;
    
    // Compute normalized gradient
    decision->normalized_grad = d_computeNormalizedGrad(raw_grad_norm, valid_tokens, cfg->reference_tokens);
    
    //=========================================================================
    // CHECK 1: Accumulation Bug (FATAL)
    //=========================================================================
    
    if (d_checkAccumulationBug(raw_grad_norm, loss, cfg)) {
        state->consecutive_zero_grad_steps++;
        
        if (state->consecutive_zero_grad_steps >= cfg->max_consecutive_zero_grad_steps) {
            decision->flags |= ControlDecision::FLAG_ACCUMULATION_BUG;
            decision->action = ControlAction::FatalError;
            decision->error_code = 1;  // Accumulation bug
            return;
        }
    } else {
        state->consecutive_zero_grad_steps = 0;
    }
    
    //=========================================================================
    // If no lattice, return baseline decision
    //=========================================================================
    
    if (d_lattice_levels == nullptr) {

        return;
    }
    
    //=========================================================================
    // READ TELEMETRY VECTORS (direct GPU memory access - no transfer!)
    //=========================================================================
    
    // Compute array indices for direct read
    // Layout: levels[level * num_streams + stream]
    const int num_streams = num_lattice_streams;
    
    // Level 0 (immediate), stream indices: LOSS=0, GRAD_NORM_MEAN=1, TOKENS=4
    const int idx_L0_loss = 0 * num_streams + 0;
    const int idx_L0_grad = 0 * num_streams + 1;
    const int idx_L0_tokens = 0 * num_streams + 4;
    
    // Level 2 (4-step baseline)
    const int idx_L2_loss = 2 * num_streams + 0;
    const int idx_L2_grad = 2 * num_streams + 1;
    
    // Read telemetry states (GPU→GPU, no transfer)
    const TelemetryState& state_L0_loss = d_lattice_levels[idx_L0_loss].state;
    const TelemetryState& state_L0_grad = d_lattice_levels[idx_L0_grad].state;
    const TelemetryState& state_L0_tokens = d_lattice_levels[idx_L0_tokens].state;
    const TelemetryState& state_L2_loss = d_lattice_levels[idx_L2_loss].state;
    const TelemetryState& state_L2_grad = d_lattice_levels[idx_L2_grad].state;
    
    // Extract telemetry values we need (inline, no function call overhead)
    // Level 0 loss
    const float L0_loss_mu = state_L0_loss.mu;
    const float L0_loss_r_out = state_L0_loss.r_out;
    const float L0_loss_ell_out = state_L0_loss.ell_out;
    
    // Level 0 grad
    const float L0_grad_mu = state_L0_grad.mu;
    const float L0_grad_v_sigma = state_L0_grad.v_sigma;
    
    // Level 0 tokens
    const float L0_tokens_mu = state_L0_tokens.mu;
    
    // Level 2 loss
    const float L2_loss_sigma_tilde = state_L2_loss.sigma_tilde;
    const float L2_loss_delta_mu = state_L2_loss.delta_mu;
    
    // Level 2 grad
    const float L2_grad_mu = state_L2_grad.mu;
    
    //=========================================================================
    // CHECK 2: Regime Change Detection
    //=========================================================================
    
    const float current_tokens = static_cast<float>(valid_tokens);
    
    if (state->last_avg_tokens > 0.0f) {
        const float baseline_tokens = (L0_tokens_mu > 0.0f) ? L0_tokens_mu : state->last_avg_tokens;
        const float change = fabsf(current_tokens - baseline_tokens);
        const float threshold = cfg->seq_len_regime_change_threshold * baseline_tokens;
        
        if (change > threshold && threshold > 0.0f) {
            decision->flags |= ControlDecision::FLAG_REGIME_CHANGE;
            state->suppression_steps_remaining = cfg->regime_change_suppression_steps;
        }
    }
    
    // Update state
    state->last_avg_seq_len = avg_seq_len;
    state->last_avg_tokens = current_tokens;
    
    // Decrement suppression
    if (state->suppression_steps_remaining > 0) {
        state->suppression_steps_remaining--;
    }
    
    //=========================================================================
    // CHECK 3: Gradient Spike Detection
    //=========================================================================
    
    // CRITICAL: Compare normalized-to-normalized (both token-scaled)
    // Don't mix raw L2 baseline with token-normalized current grad
    const float raw_baseline = (L2_grad_mu > 1e-8f) ? L2_grad_mu : L0_grad_mu;
    const float normalized_baseline = d_computeNormalizedGrad(raw_baseline, cfg->reference_tokens, cfg->reference_tokens);
    const float baseline_grad = (normalized_baseline > 1e-8f) ? normalized_baseline : decision->normalized_grad;
    decision->spike_ratio = (baseline_grad > 1e-8f) 
        ? decision->normalized_grad / baseline_grad 
        : 1.0f;
    
    // Only detect spikes if not in suppression period
    if (state->suppression_steps_remaining == 0) {
        decision->spike_severity = d_computeSpikeSeverity(
            decision->normalized_grad, baseline_grad, cfg);
    }
    
    //=========================================================================
    // CHECK 4: Outlier Regime / Anchor Drift
    //=========================================================================
    
    // Skip outlier/drift detection during suppression and warmup
    // Token count changes cause distribution shifts that look like drift
    const uint32_t min_step_for_drift = cfg->warmup_steps + cfg->baseline_stabilization_steps;
    if (state->suppression_steps_remaining == 0 && input->global_step >= min_step_for_drift) {
        if (d_detectOutlierRegime(L0_loss_r_out, L0_loss_ell_out, cfg)) {
            decision->flags |= ControlDecision::FLAG_OUTLIER_REGIME;
        }
        
        if (d_detectAnchorDrift(L2_loss_delta_mu, L2_loss_sigma_tilde, cfg)) {
            decision->flags |= ControlDecision::FLAG_ANCHOR_DRIFT;
        }
    }
    
    //=========================================================================
    // CHECK 5: Gradient Decay Detection
    //=========================================================================
    
    // Now uses consistent normalization (fixed apples-to-oranges comparison bug)
    if (state->suppression_steps_remaining == 0 &&
        input->global_step >= min_step_for_drift &&
        decision->spike_ratio < cfg->gradient_decay_threshold && 
        decision->spike_ratio > 0.0f) {
        decision->flags |= ControlDecision::FLAG_GRADIENT_DECAY;
    }
    
    //=========================================================================
    // CHECK 6: Rapid Progress Detection (REWARD improvement)
    //=========================================================================
    
    // Detect when loss is dropping faster than baseline variance
    // Skip during warmup/suppression to avoid false positives
    const uint32_t min_step = cfg->warmup_steps + cfg->baseline_stabilization_steps;
    if (state->suppression_steps_remaining == 0 && input->global_step >= min_step) {
        const float progress_magnitude = (L2_loss_sigma_tilde > 1e-8f) 
            ? -L2_loss_delta_mu / L2_loss_sigma_tilde 
            : 0.0f;
        if (progress_magnitude > cfg->progress_boost_threshold) {
            decision->flags |= ControlDecision::FLAG_RAPID_PROGRESS;
        }
    }
    
    //=========================================================================
    // COMPUTE SCALING FACTORS
    //=========================================================================
    
    // Volatility damping
    decision->volatility_damping = d_computeVolatilityDamping(L0_grad_v_sigma, cfg);
    
    // Decay boost (only if not spiking)
    if (decision->spike_severity == SpikeSeverity::None && 
        (decision->flags & ControlDecision::FLAG_GRADIENT_DECAY)) {
        decision->decay_boost = d_computeDecayBoost(L0_grad_mu, L2_grad_mu, cfg);
    }
    
    // Progress boost (reward rapid improvement)
    if (decision->spike_severity == SpikeSeverity::None &&
        (decision->flags & ControlDecision::FLAG_RAPID_PROGRESS)) {
        decision->progress_boost = d_computeProgressBoost(L2_loss_delta_mu, L2_loss_sigma_tilde, cfg);
    }
    
    // Combine factors: damping reduces, decay/progress boost increase
    // BUT only if we have valid telemetry (L2 baseline established)
    // During warmup (first 150 steps), skip intervention entirely
    if (input->global_step >= min_step) {
        decision->grad_scale_factor = decision->volatility_damping * decision->decay_boost * decision->progress_boost;
    } else {
        // Warmup: no intervention, neutral scale
        decision->grad_scale_factor = 1.0f;
    }
    
    // CRITICAL FIX (Issue #18): Prevent grad_scale_factor from becoming 0
    // A scale of 0 would zero all gradients, halting training completely.
    // Minimum scale of 0.01 allows 1% gradient flow even in worst case.
    if (decision->grad_scale_factor < 0.01f) {
        decision->grad_scale_factor = 0.01f;
    }
    
    //=========================================================================
    // CHECK 7: Plateau Detection (using loss variance from TelemetryLattice)
    //=========================================================================
    
    // Plateau detection: if loss variance is very low for many consecutive batches,
    // we may be stuck in a local minimum. Use sigma_tilde from Level 2 loss stream.
    // The TelemetryLattice already computes rolling variance via coefficient of variation.
    
    if (cfg->plateau_noise_enabled && input->global_step >= min_step) {
        // sigma_tilde is coefficient of variation: std/mean
        // For plateau, we want very low variance: sigma_tilde * mean ≈ std
        const float loss_variance_approx = L2_loss_sigma_tilde * L0_loss_mu * L2_loss_sigma_tilde * L0_loss_mu;
        
        // Increment batches_since_noise_injection
        state->batches_since_noise_injection++;
        
        // Check if loss variance is below threshold
        if (loss_variance_approx < cfg->plateau_noise_variance_threshold && L0_loss_mu > 0.1f) {
            state->consecutive_low_variance_batches++;
        } else {
            state->consecutive_low_variance_batches = 0;
        }
        
        // Check if we should trigger noise injection
        const bool patience_reached = (state->consecutive_low_variance_batches >= cfg->plateau_noise_patience);
        const bool cooldown_satisfied = (state->batches_since_noise_injection >= cfg->plateau_noise_cooldown);
        const bool under_limit = (state->noise_injections_this_epoch < cfg->plateau_noise_max_per_epoch);
        
        if (patience_reached && cooldown_satisfied && under_limit) {
            decision->flags |= ControlDecision::FLAG_PLATEAU_NOISE;
        }
    }
    
    //=========================================================================
    // DETERMINE ACTION (priority: Severe > Moderate > PlateauNoise > Outlier > Drift > Scale)
    //=========================================================================
    
    // Check cooldown BEFORE making decision (don't decrement yet)
    const bool cooldown_active = (state->soft_restart_cooldown > 0);
    
    if (decision->spike_severity == SpikeSeverity::Severe) {
        decision->action = ControlAction::SkipStep;
    }
    else if (decision->spike_severity == SpikeSeverity::Moderate) {
        decision->action = ControlAction::ScaleGradients;
        // Apply moderate scaling (default 0.5), but enforce minimum 0.1 to prevent gradient death
        const float scale = fmaxf(cfg->moderate_grad_scale, 0.1f);
        decision->grad_scale_factor *= scale;
        decision->cooldown_extension = cfg->moderate_cooldown_extension;
    }
    else if ((decision->flags & ControlDecision::FLAG_PLATEAU_NOISE) &&
             !cooldown_active) {
        // Plateau noise has higher priority than soft restart - it's more aggressive
        decision->action = ControlAction::InjectPlateauNoise;
        // Reset plateau state after triggering
        state->consecutive_low_variance_batches = 0;
        state->batches_since_noise_injection = 0;
        state->noise_injections_this_epoch++;
        state->total_noise_injections++;
    }
    else if (((decision->flags & ControlDecision::FLAG_OUTLIER_REGIME) ||
              (decision->flags & ControlDecision::FLAG_ANCHOR_DRIFT)) &&
             !cooldown_active) {
        decision->action = ControlAction::TriggerSoftRestart;
        state->soft_restart_cooldown = cfg->soft_restart_cooldown_steps;
    }
    else if (fabsf(decision->grad_scale_factor - 1.0f) > 0.01f) {
        decision->action = ControlAction::ScaleGradients;
    }
    else {
        decision->action = ControlAction::Continue;
    }
    
    // Decrement cooldown AFTER decision is made
    if (state->soft_restart_cooldown > 0) {
        state->soft_restart_cooldown--;
    }
}

//=============================================================================
// KERNEL LAUNCH WRAPPER
//=============================================================================

void launchControlDecisionKernel(
    const LatticeLevelState* d_lattice_levels,
    int num_lattice_streams,
    const TelemetryControlConfig* d_config,
    TelemetryControlState_GPU* d_state,
    const ControlKernelInput* d_input,
    ControlDecision* d_decision,
    cudaStream_t stream
) {
    // Single thread kernel - decision logic is sequential
    controlDecisionKernel<<<1, 1, 0, stream>>>(
        d_lattice_levels, num_lattice_streams, d_config, d_state, d_input, d_decision
    );
}

//=============================================================================
// PLATEAU NOISE INJECTION KERNEL
//=============================================================================

#include <curand_kernel.h>

__global__ void plateauNoiseKernel(
    float* __restrict__ weights,
    size_t num_elements,
    float noise_std,
    bool proportional,
    unsigned long long seed
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    // Initialize per-thread RNG state (Philox for quality)
    curandStatePhilox4_32_10_t rng_state;
    curand_init(seed, idx, 0, &rng_state);
    
    // Generate Gaussian noise N(0, noise_std)
    float noise = curand_normal(&rng_state) * noise_std;
    
    // Scale by weight magnitude if proportional
    if (proportional) {
        const float weight = weights[idx];
        const float scale = fmaxf(fabsf(weight), 1e-6f);  // Min scale to perturb zero weights
        noise *= scale;
    }
    
    // Add noise to weight
    weights[idx] += noise;
}

void launchPlateauNoiseInjection(
    float* weights,
    size_t num_elements,
    float noise_std,
    bool proportional,
    uint64_t seed,
    cudaStream_t stream
) {
    // Rule 20: Fail loud on invalid inputs
    if (weights == nullptr) {
        throw std::runtime_error("[launchPlateauNoiseInjection] weights is NULL");
    }
    if (num_elements == 0) {
        throw std::runtime_error("[launchPlateauNoiseInjection] num_elements is 0");
    }
    
    constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
    const int grid_size = static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);
    
    plateauNoiseKernel<<<grid_size, kBlockSize, 0, stream>>>(
        weights, num_elements, noise_std, proportional, seed
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[launchPlateauNoiseInjection] kernel launch failed: ") +
            cudaGetErrorString(err));
    }
}

//=============================================================================
// CLASS IMPLEMENTATION
//=============================================================================

TelemetryControl::TelemetryControl(const TelemetryControlConfig& config)
    : config_(config)
{
}

TelemetryControl::~TelemetryControl() {
    freeGPU();
}

TelemetryControl::TelemetryControl(TelemetryControl&& other) noexcept
    : config_(other.config_)
    , d_config_(other.d_config_)
    , d_state_(other.d_state_)
    , d_decision_(other.d_decision_)
    , d_input_(other.d_input_)
    , gpu_state_initialized_(other.gpu_state_initialized_)
{
    other.d_config_ = nullptr;
    other.d_state_ = nullptr;
    other.d_decision_ = nullptr;
    other.d_input_ = nullptr;
    other.gpu_state_initialized_ = false;
}

TelemetryControl& TelemetryControl::operator=(TelemetryControl&& other) noexcept {
    if (this != &other) {
        freeGPU();
        config_ = other.config_;
        d_config_ = other.d_config_;
        d_state_ = other.d_state_;
        d_decision_ = other.d_decision_;
        d_input_ = other.d_input_;
        gpu_state_initialized_ = other.gpu_state_initialized_;
        other.d_config_ = nullptr;
        other.d_state_ = nullptr;
        other.d_decision_ = nullptr;
        other.d_input_ = nullptr;
        other.gpu_state_initialized_ = false;
    }
    return *this;
}

void TelemetryControl::initGPU() {
    if (d_config_ != nullptr) {
        throw std::runtime_error("TelemetryControl::initGPU called twice (double-init)");
    }
    
    // Allocate GPU buffers (no stream ops - pure allocation)
    cudaError_t err;
    
    err = cudaMalloc(&d_config_, sizeof(TelemetryControlConfig));
    if (err != cudaSuccess) {
        throw std::runtime_error("TelemetryControl: Failed to allocate d_config_");
    }
    
    err = cudaMalloc(&d_state_, sizeof(TelemetryControlState_GPU));
    if (err != cudaSuccess) {
        freeGPU();
        throw std::runtime_error("TelemetryControl: Failed to allocate d_state_");
    }
    
    err = cudaMalloc(&d_decision_, sizeof(ControlDecision));
    if (err != cudaSuccess) {
        freeGPU();
        throw std::runtime_error("TelemetryControl: Failed to allocate d_decision_");
    }
    
    err = cudaMalloc(&d_input_, sizeof(ControlKernelInput));
    if (err != cudaSuccess) {
        freeGPU();
        throw std::runtime_error("TelemetryControl: Failed to allocate d_input_");
    }
    
    // Config/state init deferred to first evaluate() call with proper stream
}

void TelemetryControl::freeGPU() {
    if (d_config_) { cudaFree(d_config_); d_config_ = nullptr; }
    if (d_state_) { cudaFree(d_state_); d_state_ = nullptr; }
    if (d_decision_) { cudaFree(d_decision_); d_decision_ = nullptr; }
    if (d_input_) { cudaFree(d_input_); d_input_ = nullptr; }
    gpu_state_initialized_ = false;
}

void TelemetryControl::reset(cudaStream_t stream) {
    if (!d_state_) {
        throw std::runtime_error("[TelemetryControl::reset] d_state_ is NULL - call initGPU first");
    }
    StreamController::fatalIfDefaultStream(stream, "TelemetryControl::reset");
    
    // Reset GPU state to zeros
    cudaError_t err = cudaMemsetAsync(d_state_, 0, sizeof(TelemetryControlState_GPU), stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[TelemetryControl::reset] cudaMemsetAsync failed: ") +
            cudaGetErrorString(err));
    }
    
    // Mark that config needs to be re-copied on next evaluate (in case config changed)
    gpu_state_initialized_ = false;
}

ControlDecision TelemetryControl::evaluate(
    const TelemetryLattice* lattice,
    float raw_grad_norm,
    float loss,
    int valid_tokens,
    float avg_seq_len,
    uint32_t global_step,
    cudaStream_t stream
) {
    if (!isInitialized()) {
        throw std::runtime_error("TelemetryControl::evaluate called before initGPU");
    }
    if (!lattice) {
        throw std::runtime_error("TelemetryControl::evaluate: lattice is NULL");
    }
    StreamController::fatalIfDefaultStream(stream, "TelemetryControl::evaluate");
    
    // Per-instance lazy init: copy config and zero state on first call
    // BUG FIX: Was using static bool which shared state across ALL instances!
    if (!gpu_state_initialized_) {
        cudaError_t err = cudaMemcpyAsync(d_config_, &config_, sizeof(TelemetryControlConfig),
                        cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("[TelemetryControl::evaluate] cudaMemcpyAsync config failed: ") +
                cudaGetErrorString(err));
        }
        err = cudaMemsetAsync(d_state_, 0, sizeof(TelemetryControlState_GPU), stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("[TelemetryControl::evaluate] cudaMemsetAsync state failed: ") +
                cudaGetErrorString(err));
        }
        gpu_state_initialized_ = true;
    }
    
    // Pack input
    ControlKernelInput input{};
    input.raw_grad_norm = raw_grad_norm;
    input.loss = loss;
    input.avg_seq_len = avg_seq_len;
    input.valid_tokens = valid_tokens;
    input.global_step = global_step;
    
    // Copy input to GPU (32 bytes H2D)
    cudaError_t err = cudaMemcpyAsync(d_input_, &input, sizeof(ControlKernelInput),
                    cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[TelemetryControl::evaluate] cudaMemcpyAsync input failed: ") +
            cudaGetErrorString(err));
    }
    
    // Extract device pointers from lattice (Pattern B: public accessors)
    const LatticeLevelState* d_lattice_levels = lattice->levels();
    const int num_lattice_streams = lattice->config().num_streams;
    
    // Launch kernel with device pointers (not host struct pointer!)
    launchControlDecisionKernel(d_lattice_levels, num_lattice_streams, d_config_, d_state_, d_input_, d_decision_, stream);
    
    // Check kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[TelemetryControl::evaluate] kernel launch failed: ") +
            cudaGetErrorString(err));
    }
    
    // Copy decision back (48 bytes D2H)
    ControlDecision result{};
    err = cudaMemcpyAsync(&result, d_decision_, sizeof(ControlDecision),
                    cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[TelemetryControl::evaluate] cudaMemcpyAsync decision failed: ") +
            cudaGetErrorString(err));
    }
    
    // Must sync to get result
    cudaStreamSynchronize(stream);
    
    return result;
}

std::string TelemetryControl::describeDecision(const ControlDecision& d) const {
    std::ostringstream oss;
    oss << "Action=" << getControlActionName(d.action);
    
    if (d.spike_severity != SpikeSeverity::None) {
        oss << " spike=" << getSpikeSeverityName(d.spike_severity) 
            << "(ratio=" << d.spike_ratio << ")";
    }
    
    if (d.grad_scale_factor != 1.0f) {
        oss << " scale=" << d.grad_scale_factor;
    }
    
    if (d.accumulationBugDetected()) oss << " [ACCUM_BUG]";
    if (d.regimeChangeDetected()) oss << " [REGIME_CHANGE]";
    if (d.outlierRegimeActive()) oss << " [OUTLIER]";
    if (d.anchorDriftDetected()) oss << " [DRIFT]";
    if (d.gradientDecayDetected()) oss << " [DECAY]";
    if (d.plateauNoiseTriggered()) oss << " [PLATEAU_NOISE]";
    
    return oss.str();
}

} // namespace GRIM::Telemetry
