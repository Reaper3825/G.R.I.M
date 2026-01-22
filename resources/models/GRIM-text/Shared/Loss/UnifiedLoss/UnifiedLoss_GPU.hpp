#pragma once
/**
 * @file UnifiedLoss_GPU.hpp
 * @brief Production-ready unified loss: Focal + LabelSmoothing + CrossEntropy
 * 
 * DESIGN PRINCIPLES:
 *   1. FAIL LOUD - No silent errors, no fallback paths
 *   2. SINGLE KERNEL - No overwrite bugs, no composition issues
 *   3. TELEMETRY READY - Exports all statistics for hierarchical tracking
 *   4. MATHEMATICALLY CORRECT - Proper focal gradient with smoothed targets
 * 
 * LOSS FORMULA (Focal CE with Label Smoothing):
 *   L = α * (1 - p_t)^γ * CE_smooth
 *   
 *   Where:
 *     CE_smooth = -(1-ε)*log(p_t) - ε/(V-1) * Σ_{i≠t} log(p_i)
 *     p_t = softmax probability of target class
 *     α = class balance weight (1.0 = no balance)
 *     γ = focusing parameter (0.0 = standard CE, 2.0 = strong focus)
 *     ε = label smoothing epsilon (0.0 = hard targets)
 * 
 * GRADIENT (derived correctly):
 *   ∂L/∂z_i = α * (1-p_t)^(γ-1) * [
 *       (1-p_t) * (p_i - q_i) +
 *       γ * p_t * CE_smooth * (p_i - δ_{i,t})
 *   ]
 *   
 *   Where q_i = smoothed target distribution
 */

#include <cuda_runtime.h>
#include <cstdint>

namespace GRIM::Loss {

//=============================================================================
// CONFIGURATION - All parameters in one place
//=============================================================================

struct UnifiedLossConfig {
    // Focal Loss parameters
    float focal_alpha = 1.0f;      // Class balance (1.0 = none)
    float focal_gamma = 0.0f;      // Focusing (0 = CE, 2 = strong focal)
    bool  focal_enabled = false;    // If false, gamma=0 (standard CE)
    
    // Label Smoothing parameters
    float smoothing_epsilon = 0.0f; // Target smoothing (0 = hard targets)
    bool  smoothing_enabled = false; // If false, epsilon=0
    
    // Issue #44 FIX: Entropy regularization to prevent mode collapse
    // reg = λ * Σ_v p_v²  (penalizes concentration, encourages diversity)
    // This is equivalent to negative entropy: high when one token dominates
    float entropy_reg_lambda = 1.0f;  // Regularization strength (0 = disabled, try 0.1-1.0)
    bool  entropy_reg_enabled = false; // If false, no entropy penalty
    
    // Validation
    bool  strict_mode = true;       // FATAL on any NaN/Inf (recommended)
};

//=============================================================================
// INPUT TENSORS - Everything the kernel needs
//=============================================================================

struct UnifiedLossInputs {
    // Required tensors (MUST NOT be null)
    const float* logits;            // [batch * seq, vocab] - Model output
    const int*   targets;           // [batch * seq] - Target indices (-1 = masked)
    
    // Dimensions (MUST be positive)
    int batch_size;
    int seq_len;
    int vocab_size;
    
    // Optional weighting
    const float* sequence_weights;  // [batch] - Per-sequence weight (nullptr = 1.0)
    int weight_count;               // Number of weights (0 = use batch_size)
    
    // Issue #38 FIX: Per-token class weighting to prevent mode collapse on frequent tokens
    // weight[token_id] = inverse frequency based weight (frequent tokens get lower weight)
    // This prevents SPACE token (277) from dominating due to being 15-22% of all targets
    const float* token_weights;     // [vocab_size] - Per-token weight (nullptr = 1.0 for all)
    
    // GRMT v6: Per-position byte length weights to prevent atom tokens from being "free"
    // Atoms represent multiple bytes but cost 1 position - weight by byte length so
    // learning signal is proportional to information content.
    const uint16_t* position_byte_lengths;  // [batch * seq] - Byte length per position (nullptr = 1)
    
    // Issue #39 FIX: Output logit bias correction to prevent mode collapse
    // Subtracts running EMA of mean logit per token BEFORE softmax.
    // This prevents tokens like SPACE from having systematically higher logits.
    const float* logit_bias;        // [vocab_size] - EMA of per-token mean logit (nullptr = no correction)
    float* logit_bias_update;       // [vocab_size] - OUTPUT: batch mean logit per token (nullptr = don't update)
    float logit_bias_ema_alpha;     // EMA decay rate (0.01 = slow adapt, 0.1 = fast adapt)
    
    // CUDA
    cudaStream_t stream;
};

//=============================================================================
// OUTPUT BUFFERS - Where results go
//=============================================================================

struct UnifiedLossOutputs {
    // Per-token outputs (MUST NOT be null)
    float* token_losses;            // [batch * seq] - Loss per token
    float* grad_logits;             // [batch * seq, vocab] - Gradient per logit
    
    // Reduction scratch (MUST NOT be null)
    float* loss_sum;                // [1] - Reduced total loss
};

//=============================================================================
// TELEMETRY - Statistics for monitoring (ready for hierarchical aggregation)
//=============================================================================

struct UnifiedLossTelemetry {
    // Magnitude statistics
    float loss_mean;                // μ: Mean loss across tokens
    float loss_variance;            // σ²: Variance of loss
    float loss_max;                 // Max token loss (outlier detection)
    float loss_min;                 // Min token loss
    
    // Gradient statistics  
    float grad_norm_mean;           // Mean L2 norm of per-token gradients
    float grad_norm_max;            // Max gradient norm (explosion detection)
    
    // Focal statistics
    float focal_weight_mean;        // Mean (1-p)^γ weight
    float hard_example_ratio;       // % tokens with p_t < 0.5
    
    // Health flags
    uint32_t valid_tokens;          // Tokens actually contributing to loss
    uint32_t masked_tokens;         // Tokens skipped (target = -1)
    uint32_t nan_count;             // NaN detections (should be 0)
    uint32_t inf_count;             // Inf detections (should be 0)
    uint32_t invalid_target_count;  // Targets outside [-1, vocab_size)
    
    // Error state (0 = success)
    int32_t error_code;
    
    // Error codes
    static constexpr int32_t OK = 0;
    static constexpr int32_t ERR_NULL_LOGITS = 1;
    static constexpr int32_t ERR_NULL_TARGETS = 2;
    static constexpr int32_t ERR_NULL_OUTPUTS = 3;
    static constexpr int32_t ERR_INVALID_DIMS = 4;
    static constexpr int32_t ERR_NAN_IN_LOGITS = 5;
    static constexpr int32_t ERR_NAN_IN_LOSS = 6;
    static constexpr int32_t ERR_INF_IN_LOSS = 7;
    static constexpr int32_t ERR_KERNEL_LAUNCH = 8;
    static constexpr int32_t ERR_CUDA_SYNC = 9;
    static constexpr int32_t ERR_INVALID_WEIGHTS = 10;
    static constexpr int32_t ERR_INVALID_TARGET = 11;
};

//=============================================================================
// DEVICE-SIDE TELEMETRY ACCUMULATOR (for atomic updates in kernel)
//=============================================================================

struct DeviceTelemetryAccum {
    // Atomically accumulated
    float loss_sum;
    float loss_sq_sum;
    float loss_max;
    float loss_min;
    float grad_norm_sum;
    float grad_norm_max;
    float focal_weight_sum;
    uint32_t hard_example_count;
    uint32_t valid_count;
    uint32_t masked_count;
    uint32_t nan_count;
    uint32_t inf_count;
    uint32_t invalid_target_count;
    // DEBUG: Track exit reasons
    uint32_t exit_max_logit_nan;    // Exit due to max_logit NaN/Inf
    uint32_t exit_sum_exp_zero;     // Exit due to sum_exp < epsilon
    uint32_t exit_loss_nan;         // Exit due to computed loss NaN/Inf
    uint32_t exit_success;          // Normal successful computation
    // DEBUG: Sample values from first valid token
    float debug_max_logit;
    float debug_sum_exp;
    float debug_p_t;
    float debug_p_277;
    float debug_ce_smooth;
    float debug_focal_weight;
    float debug_sample_weight;
    float debug_loss;
    float debug_focal_alpha;  // Config value for verification
    int debug_target;
    uint32_t debug_count;
    float debug_p_t_10[10];
    float debug_ce_smooth_10[10];
    // Token 277 gradient breakdown
    float grad_277_sum;          // Sum of grad_logit[277] across all positions
    float grad_277_sum_target;   // Sum where target==277 (should be negative)
    float grad_277_sum_nontarget;// Sum where target!=277 (will be positive when p_277 high)
    uint32_t target_277_count;   // Count of positions where target==277
};

//=============================================================================
// UNIFIED LOSS CONTEXT (Rule 22 compliant - owns GPU buffers)
//=============================================================================

/**
 * @brief Persistent context for unified loss computation
 * 
 * RULE 22 COMPLIANCE:
 *   - Allocates GPU buffers ONCE at construction
 *   - Reuses buffers across all calls
 *   - No per-call cudaMalloc/cudaFree
 */
class UnifiedLossContext {
public:
    UnifiedLossContext();
    ~UnifiedLossContext();
    
    // Non-copyable, movable
    UnifiedLossContext(const UnifiedLossContext&) = delete;
    UnifiedLossContext& operator=(const UnifiedLossContext&) = delete;
    UnifiedLossContext(UnifiedLossContext&&) noexcept;
    UnifiedLossContext& operator=(UnifiedLossContext&&) noexcept;
    
    /**
     * @brief Check if context is initialized
     */
    bool isInitialized() const { return d_telemetry_ != nullptr; }
    
    /**
     * @brief Compute loss with pre-allocated buffers
     */
    UnifiedLossTelemetry compute(
        const UnifiedLossConfig& config,
        const UnifiedLossInputs& inputs,
        UnifiedLossOutputs& outputs
    );
    
private:
    DeviceTelemetryAccum* d_telemetry_ = nullptr;  // Pre-allocated, reused
    
    void allocateGPU();
    void freeGPU();
};

/**
 * @brief Get human-readable error message
 */
const char* getErrorMessage(int32_t error_code);

} // namespace GRIM::Loss
