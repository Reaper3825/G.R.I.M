#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include "../UnigramByte/Unigram.hpp"

//======================================================//
// HyperParameters_GPU.hpp - Single Source of Truth
// 
// This header provides model architecture defaults and
// hyperparameter configuration for GRIM-text training.
//
// Structure:
// 1. Core constants (always available, even in CUDA)
// 2. Struct definitions (always available)
// 3. Helper functions that don't need JSON (always available)
// 4. JSON-based config loading (only in non-CUDA code)
//
// RULE: NO other file should define hyperparameters!
// All values come from here or ai_config.json via this header.
//======================================================//

namespace GRIM {
namespace HyperParameters {

//======================================================//
// CUDA Kernel Launch Configuration
// Standard block sizes used across all CUDA kernels
//======================================================//
constexpr int CUDA_BLOCK_SIZE_STANDARD = 256;     // Default for most kernels
constexpr int CUDA_BLOCK_SIZE_SMALL = 128;        // For memory-bound kernels
constexpr int CUDA_WARP_SIZE = 32;                // NVIDIA warp size (hardware constant)
constexpr int CUDA_MAX_GRID_SIZE = 65535;         // Maximum blocks per grid dimension (hardware)

// Derived CUDA constants - use primary constants for consistency
constexpr int CUDA_QUANTIZATION_THREADS = CUDA_BLOCK_SIZE_STANDARD;  // Quantization kernel threads
constexpr int CUDA_REDUCTION_MAX_BLOCKS = CUDA_BLOCK_SIZE_STANDARD;  // Cap grid size for reductions

//======================================================//
// Model Architecture Defaults (declare early for derived constants)
// Used when config values are missing or for validation
//======================================================//
constexpr int DEFAULT_D_MODEL = 768;
constexpr int DEFAULT_NUM_LAYERS = 6;
constexpr int DEFAULT_NUM_HEADS = 12;
constexpr int DEFAULT_D_FF_MULTIPLIER = 4;  // d_ff = d_model * multiplier
constexpr int DEFAULT_MAX_SEQ_LEN = 2048;
constexpr float DEFAULT_DROPOUT_RATE = 0.1f;
constexpr float DEFAULT_ATTENTION_DROPOUT = 0.1f;

// Derived model constants
constexpr int DEFAULT_D_FF = DEFAULT_D_MODEL * DEFAULT_D_FF_MULTIPLIER;  // 3072
constexpr int DEFAULT_HEAD_DIM = DEFAULT_D_MODEL / DEFAULT_NUM_HEADS;    // 64

// Sequence-length derived constants
constexpr size_t CUDA_FALLBACK_MAX_LOSS_TOKENS = DEFAULT_MAX_SEQ_LEN * 4;  // 4x max_seq_len for batching

//======================================================//
// Numerical Stability Constants
// Epsilon values for safe division and log operations
//======================================================//
constexpr float EPSILON_SAFE_DIV = 1e-8f;         // Division safety (AdamW, etc.)
constexpr float EPSILON_LOG_PROB = 1e-7f;         // Log probability clamping (loss)
constexpr float EPSILON_VARIANCE = 1e-4f;         // Variance computation (GRIM-TS)
constexpr float EPSILON_TEMPERATURE = 1e-6f;      // Temperature comparison threshold
constexpr float EPSILON_DYNAMIC_LR = 1e-12f;      // Dynamic LR controller precision
constexpr float EPSILON_RMSNORM = 1e-5f;          // RMSNorm numerical stability
constexpr float EPSILON_GRADIENT_CLIP = 1e-6f;    // Gradient clipping minimum threshold

constexpr float LOG_CLAMP_MIN = -100.0f;          // Minimum log value (prevents -inf)
constexpr float NEG_INF_ATTENTION = -1e9f;        // Attention masking value
constexpr float NEG_INF_THRESHOLD = -1e30f;       // Invalid/uninitialized threshold
constexpr float PROBABILITY_FLOOR = 1e-12f;       // Minimum probability (prevents log(0))
constexpr float SOFTMAX_CLIP_THRESHOLD = -20.0f;  // exp(x) ≈ 0 for x < this
constexpr float NORMALIZED_CLAMP = 4.0f;          // Clamp for normalized values (GRIM-TS)

//======================================================//
// Tensor/Matrix Operation Constants
// Configuration for CUDA tensor kernels
//======================================================//
constexpr int CUDA_TILE_DIM_TRANSPOSE = CUDA_WARP_SIZE;  // Tile dim = warp size for memory coalescing

//======================================================//
// Telemetry System Configuration
// Hierarchical streaming statistics
//======================================================//
constexpr int TELEMETRY_MAX_LEVELS = 8;           // TelemetryLattice temporal levels
constexpr int TELEMETRY_MAX_STREAMS = 16;         // TelemetryLattice metric streams

//======================================================//
// Equation Logging Configuration
// Controls diagnostic equation-based logging for training
//======================================================//
constexpr bool DEFAULT_EQ_LOG_ENABLED = true;     // Enable equation logging by default
constexpr int DEFAULT_EQ_LOG_INTERVAL = 1;       // Log every N batches (0 = every batch)

//======================================================//
// PyTorch Verification Configuration
// When GRIM_PYTORCH_VERIFY is defined at compile time, enables
// side-by-side comparison with PyTorch reference implementations.
// 
// To enable: Add -DGRIM_PYTORCH_VERIFY to CMake compile flags
// WARNING: This is SLOW (subprocess per op) - use DEBUG builds only!
//======================================================//
constexpr int DEFAULT_PYTORCH_VERIFY_INTERVAL = 1;  // Verify every N batches (reduces overhead)

//======================================================//
// Gradient Scaling Defaults
//======================================================//
constexpr bool DEFAULT_GRAD_SCALE_PER_TOKEN = false;  // Apply 1/valid_tokens in backward

//======================================================//
// UnigramLM Training Constants
// Parameters for vocabulary training/pruning
//======================================================//
constexpr int UNIGRAM_EM_ITERATIONS = 5;              // EM algorithm iterations
constexpr float UNIGRAM_PRUNE_THRESHOLD = 0.0001f;    // Tokens <0.01% usage get pruned
constexpr int UNIGRAM_MIN_VOCAB_SIZE = 8000;          // Don't prune below this
constexpr double UNIGRAM_MIN_COUNT = 1.0;             // Tokens used <1 time get pruned
constexpr size_t UNIGRAM_MAX_SUBWORD_BYTES = 100ULL * 1024 * 1024;  // 100MB limit for subword mining
constexpr size_t UNIGRAM_MAX_SEQUENCE_LENGTH = DEFAULT_MAX_SEQ_LEN * 4;  // 4x max_seq_len for Viterbi

//======================================================//
// GELU Activation Constants
// Mathematical constants for GELU approximation
//======================================================//
constexpr float GELU_SQRT_2_OVER_PI = 0.7978845608028654f;  // sqrt(2/pi)
constexpr float GELU_CUBIC_COEFF = 0.044715f;              // Coefficient in GELU approximation

//======================================================//
// AdamW Optimizer Hyperparameters
// Issue #134: Fixed β₂ and weight_decay magnitude problems:
//   β₂=0.9 gave 7-step half-life on v_t → unstable for small text gradients
//   β₂=0.999 gives ~700-step half-life → proper momentum smoothing (standard)
//   weight_decay=0.1 was 7000x larger than text gradient updates → optimizer
//   was just decaying weights, not learning. 0.01 is standard (GPT-2/3, LLaMA).
//======================================================//
constexpr float ADAMW_BETA1 = 0.9f;
constexpr float ADAMW_BETA2 = 0.999f;             // Issue #134: was 0.9 (7-step half-life!) → 0.999 (standard)
constexpr float ADAMW_EPSILON = 1e-8f;
constexpr float ADAMW_WEIGHT_DECAY = 0.01f;       // Issue #134: was 0.1 (7000x > grad update) → 0.01 (standard)

// Depth-Aware Upsilon (Υ) Regularization
// Formula: Υ_l = UPSILON_BASE * sqrt(L_ref / L)
// L_ref = reference layer count, L = current layer (1-indexed)
// Deeper layers get LESS regularization (smaller Υ)
constexpr float UPSILON_BASE = 0.1f;              // Base upsilon coefficient
constexpr int UPSILON_REFERENCE_LAYERS = 12;      // Reference layer count (L_ref)

//======================================================//
// Tokenizer / Atom Token Configuration
// Token ranges for byte fallback and atom types
//======================================================//
constexpr int BYTE_TOKEN_END = 256;               // Byte tokens: [0, 256) - one per byte value
constexpr int ATOM_TOKEN_START = BYTE_TOKEN_END;  // First atom token ID (immediately after bytes)
constexpr int NUM_ATOM_TYPES = Tokenizer::kAtomTypeCount;  // Number of distinct atom types
constexpr int ATOM_SLOTS_PER_TYPE = 1;            // One slot per atom type
constexpr int ATOM_TOKEN_END = ATOM_TOKEN_START + (NUM_ATOM_TYPES * ATOM_SLOTS_PER_TYPE);
constexpr uint32_t MAX_REASONABLE_VOCAB_SIZE = 2000000; // Sanity check for vocab detection

//======================================================//
// Flash Attention Constants
// These MUST match Flash_Attention_Kernal.cu exactly!
//======================================================//
constexpr int FLASH_ATTN_BLOCK_Q = CUDA_WARP_SIZE;    // Block size for Q tiles (= warp size)
constexpr int FLASH_ATTN_BLOCK_KV = CUDA_WARP_SIZE;   // Block size for K/V tiles (= warp size)
constexpr int FLASH_ATTN_NUM_THREADS = CUDA_BLOCK_SIZE_STANDARD;  // Threads per block

// Supported head dimensions for Flash Attention kernel (derived from architecture)
constexpr int FLASH_ATTN_HEAD_DIM_32 = 32;
constexpr int FLASH_ATTN_HEAD_DIM_64 = DEFAULT_HEAD_DIM;  // Primary supported head_dim

// Softmax temperature for attention (MUST match between forward and backward!)
// Standard value is 1.0 (no scaling). Lower values sharpen attention (risk saturation).
// Higher values spread attention more evenly (risk over-smoothing).
constexpr float SOFTMAX_TEMPERATURE = 1.0f;

// QK-Normalization: Normalize Q and K vectors to unit L2 norm before computing scores
// This prevents attention saturation when embeddings have large magnitudes
// Used by Gemma, ViT-22B, and other modern architectures
// 
// ❌ DISABLED - Extensive testing (Dec 22-27, 2025) showed QK-normalization does NOT fix
// training plateau. Loss still stalls at ~8.3-8.8 with this enabled. See:
// docs/PLATEAU_BUG_INVESTIGATION.md "🚫 CRITICAL: QK-NORMALIZATION DOES NOT FIX PLATEAU"
constexpr bool QK_NORMALIZATION_ENABLED = false;  // Master switch for QK-normalization
constexpr float QK_NORM_SCALE = 1.0f;  // Default scale factor (like sqrt(head_dim) but tunable)

// Learnable per-head QK-norm scales (nGPT-style)
// When enabled, each attention head has learnable alpha_q and alpha_k parameters
// Forward: q̂ = alpha_q * (q / ||q||), k̂ = alpha_k * (k / ||k||)
// This buffers the 1/||q|| division and allows heads to learn different scales
constexpr bool QK_NORM_LEARNABLE_SCALE = true;
constexpr float QK_NORM_ALPHA_INIT = 1.0f;  // Initial value for alpha_q and alpha_k

//======================================================//
// Positional Bias Method (PBM) Configuration
// Hybrid ALiBi + RoPE for position encoding
//======================================================//
constexpr float ROPE_THETA = 10000.0f;        // RoPE base frequency (standard: 10000)
constexpr float ROPE_SCALING = 1.0f;          // NTK scaling factor (1.0 = no scaling)
constexpr float ALIBI_SLOPE_EXPONENT = -8.0f; // Controls ALiBi slope decay across heads

// ISSUE #78 FIX: Maximum ALiBi bias cap (prevents gradient explosion in softmax backward)
// Standard ALiBi with max_seq_len=1024 can produce biases up to -256, causing:
// - exp(-256) ≈ 0 (underflow, near-zero attention probabilities)
// - Softmax backward amplifies dQ/dK by 100,000x vs dV (gradient explosion!)
// 
// With ALIBI_MAX_BIAS=-10:
// - exp(-10) ≈ 0.000045 (computable, not underflow)
// - Softmax backward stays numerically stable
// - Preserves ALiBi's distance-decay within the capped range
//
// Set to 0.0f to disable capping (NOT RECOMMENDED - causes Issue #78 gradient explosion)
constexpr float ALIBI_MAX_BIAS = 0.0f;

//======================================================//
// Grouped Query Attention (GQA) Configuration
// GQA reduces memory/compute by sharing K,V heads across query groups
// - MHA: num_kv_heads = num_heads (standard multi-head attention)
// - GQA: num_kv_heads < num_heads (grouped query attention)
// - MQA: num_kv_heads = 1 (multi-query attention, all Q share one K,V)
//
// Memory savings: KV cache reduced by factor of (num_heads / num_kv_heads)
// Quality tradeoff: GQA (4-8 KV heads) often matches MHA quality
//======================================================//
constexpr int DEFAULT_NUM_KV_HEADS = 4;        // GQA default: 4 KV heads for 12 Q heads (3:1 ratio)
constexpr bool GQA_ENABLED = true;            // Master switch for GQA

// Validate GQA configuration
inline bool isValidGQAConfig(int num_heads, int num_kv_heads) {
    if (num_kv_heads <= 0 || num_heads <= 0) return false;
    if (num_kv_heads > num_heads) return false;
    // num_heads must be divisible by num_kv_heads for proper grouping
    return (num_heads % num_kv_heads) == 0;
}

// Compute the number of Q heads per KV group
inline int computeHeadsPerKVGroup(int num_heads, int num_kv_heads) {
    return (num_kv_heads > 0) ? (num_heads / num_kv_heads) : num_heads;
}

// Compute KV projection size (reduced from Q projection in GQA)
inline int computeKVProjectionSize(int d_model, int num_heads, int num_kv_heads) {
    int head_dim = d_model / num_heads;
    return num_kv_heads * head_dim;  // K and V each have this size
}

// Validate head_dim at compile time or runtime
inline bool isValidFlashAttentionHeadDim(int head_dim) {
    return head_dim == FLASH_ATTN_HEAD_DIM_32 || head_dim == FLASH_ATTN_HEAD_DIM_64;
}

//======================================================//
// Positional Encoding Configuration
// Supports ALiBi, RoPE, and Hybrid (ALiBi+RoPE)
//======================================================//
enum class PositionalEncodingType {
    NONE,       // No positional encoding
    ALIBI,      // Attention with Linear Biases (bias-based, good for long-range)
    ROPE,       // Rotary Position Embedding (rotation-based, good for local patterns)
    ALIBI_ROPE  // Hybrid: ALiBi for long-range + RoPE for local patterns (recommended)
};

// Default positional encoding configuration
constexpr PositionalEncodingType DEFAULT_POSITIONAL_ENCODING = PositionalEncodingType::ALIBI_ROPE;

// RoPE base frequency (for computing rotation frequencies)
constexpr float ROPE_BASE_FREQ = 10000.0f;

// Helper to get string representation of positional encoding type
inline const char* positionalEncodingTypeToString(PositionalEncodingType type) {
    switch (type) {
        case PositionalEncodingType::NONE: return "NONE";
        case PositionalEncodingType::ALIBI: return "ALIBI";
        case PositionalEncodingType::ROPE: return "ROPE";
        case PositionalEncodingType::ALIBI_ROPE: return "ALIBI_ROPE";
        default: return "UNKNOWN";
    }
}

// Helper to parse positional encoding type from string
inline PositionalEncodingType parsePositionalEncodingType(const std::string& str) {
    if (str == "NONE" || str == "none") return PositionalEncodingType::NONE;
    if (str == "ALIBI" || str == "alibi") return PositionalEncodingType::ALIBI;
    if (str == "ROPE" || str == "rope") return PositionalEncodingType::ROPE;
    if (str == "ALIBI_ROPE" || str == "alibi_rope" || str == "hybrid" || str == "HYBRID") 
        return PositionalEncodingType::ALIBI_ROPE;
    
    // STRICT: Unknown encoding types are errors, not silent fallbacks
    throw std::runtime_error("parsePositionalEncodingType: unknown encoding type '" + str + 
                             "'. Valid: NONE, ALIBI, ROPE, ALIBI_ROPE/hybrid");
}

// Check if positional encoding type uses ALiBi
inline bool usesALiBi(PositionalEncodingType type) {
    return type == PositionalEncodingType::ALIBI || type == PositionalEncodingType::ALIBI_ROPE;
}

// Check if positional encoding type uses RoPE
inline bool usesRoPE(PositionalEncodingType type) {
    return type == PositionalEncodingType::ROPE || type == PositionalEncodingType::ALIBI_ROPE;
}

//======================================================//
// Loss Configuration Defaults
// Centralizes all loss-related hyperparameters
//======================================================//

// Label Smoothing: regularization technique that prevents overconfidence
constexpr bool DEFAULT_LOSS_LABEL_SMOOTHING_ENABLED = true;
constexpr float DEFAULT_LOSS_LABEL_SMOOTHING_EPSILON = 0.1f;  // Standard epsilon when enabled

// Focal Loss: down-weights easy examples, focuses on hard examples
constexpr bool DEFAULT_LOSS_FOCAL_ENABLED = false;
constexpr float DEFAULT_LOSS_FOCAL_GAMMA = 2.0f;   // Standard gamma when enabled (higher = more focus on hard examples)
constexpr float DEFAULT_LOSS_FOCAL_ALPHA = 1.0f;   // 1.0 = no class weighting (standard), <1.0 = downscale all losses

// Distillation: knowledge transfer from teacher model
constexpr bool DEFAULT_LOSS_DISTILLATION_ENABLED = false;
constexpr float DEFAULT_LOSS_DISTILLATION_TEMPERATURE = 1.0f;
constexpr float DEFAULT_LOSS_DISTILLATION_LAMBDA = 0.5f;  // Weight of distillation loss when enabled

// Preference (KL divergence for RLHF-style training)
constexpr bool DEFAULT_LOSS_PREFERENCE_ENABLED = false;
constexpr float DEFAULT_LOSS_PREFERENCE_BETA = 0.1f;  // KL penalty coefficient

// Token Masking: exclude specific tokens from loss
constexpr bool DEFAULT_LOSS_MASKING_ENABLED = true;

// Numeric head loss (side-channel regression for numeric atoms)
constexpr bool DEFAULT_NUMERIC_HEAD_ENABLED = false;
constexpr float DEFAULT_NUMERIC_HEAD_LOSS_WEIGHT = 1.0f;
constexpr float DEFAULT_NUMERIC_HEAD_HUBER_DELTA = 1.0f;
constexpr bool DEFAULT_NUMERIC_HEAD_LOG_SCALE = true;
constexpr float DEFAULT_NUMERIC_HEAD_LOG_MAX = 20.0f;

//======================================================//
// Scratch Block Configuration (ScratchBlock Reasoning Layer)
// Memory buffers for structured reasoning before text generation
//======================================================//
constexpr bool DEFAULT_SCRATCH_BLOCKS_ENABLED = true;
constexpr size_t DEFAULT_SCRATCH_MAX_TOKENS_PER_BLOCK = 16384;
constexpr size_t DEFAULT_SCRATCH_NUM_BLOCKS = 4;  // Calculated: (batch_size × max_seq_len × pipeline_depth) / tokens_per_block
constexpr bool DEFAULT_SCRATCH_WRITE_COMBINED = true;  // Safe default; enable for performance after validation

//======================================================//
// Model Architecture Validation & Computation
//======================================================//
struct ModelArchitecture {
    int d_model = DEFAULT_D_MODEL;
    int num_layers = DEFAULT_NUM_LAYERS;
    int num_heads = DEFAULT_NUM_HEADS;
    int num_kv_heads = DEFAULT_NUM_KV_HEADS;  // GQA: number of KV heads
    int d_ff = DEFAULT_D_FF;
    int max_seq_len = DEFAULT_MAX_SEQ_LEN;
    float dropout_rate = DEFAULT_DROPOUT_RATE;
    float attention_dropout = DEFAULT_ATTENTION_DROPOUT;
    bool tie_embeddings = true;  // Weight tying: share embedding/LM head weights
    
    // Derived (computed from above)
    int head_dim = DEFAULT_HEAD_DIM;
    
    // Compute derived values and validate
    bool validate(std::string* error_msg = nullptr) {
        // Compute head_dim
        if (num_heads <= 0) {
            if (error_msg) *error_msg = "num_heads must be > 0";
            return false;
        }
        head_dim = d_model / num_heads;
        
        // Check d_model divisibility
        if (d_model % num_heads != 0) {
            if (error_msg) *error_msg = "d_model must be divisible by num_heads";
            return false;
        }
        
        // Validate Flash Attention compatibility
        if (!isValidFlashAttentionHeadDim(head_dim)) {
            if (error_msg) {
                *error_msg = "head_dim=" + std::to_string(head_dim) + 
                             " not supported by Flash Attention (need 32 or 64)";
            }
            return false;
        }
        
        // Compute d_ff if not explicitly set
        if (d_ff <= 0) {
            d_ff = d_model * DEFAULT_D_FF_MULTIPLIER;
        }
        
        return true;
    }
};

// Helper to compute head_dim inline
inline int computeHeadDim(int d_model, int num_heads) {
    return (num_heads > 0) ? (d_model / num_heads) : DEFAULT_HEAD_DIM;
}

// Helper to compute d_ff inline
inline int computeDFF(int d_model, int multiplier = DEFAULT_D_FF_MULTIPLIER) {
    return d_model * multiplier;
}

struct DerivationContext {
    int train_sequence_count = 0;
    int validation_interval = 0;
};

struct DerivedScheduleInfo {
    int batches_per_epoch = 1;
    int total_training_steps = 1;
    int safe_last_step = 1;
};

using LogCallback = std::function<void(const std::string&)>;

} // namespace HyperParameters
} // namespace GRIM

//======================================================//
// Implementation Section
// 
// Functions below require full TrainingHyperparameters struct definition.
// These will only compile if ai_config_paths.hpp was included first.
// 
// For CUDA files: #include "ai_config_paths.hpp" BEFORE this header
// For regular C++: We include it automatically below
//======================================================//

#ifndef __CUDACC__
// Regular C++ compilation - include the config header
#include "../../../../../control/ai_config_paths.hpp"
#endif

// Only compile if TrainingHyperparameters is fully defined
// (checked via the marker defined in ai_config_paths.hpp)
#ifdef GRIM_CONFIG_AI_CONFIG_PATHS_HPP_INCLUDED

namespace GRIM {
namespace HyperParameters {

inline void autoPopulateDynamicLR(
    GRIM::Config::TrainingHyperparameters& params,
    const DerivedScheduleInfo& info) {
    if (!params.dynamic_lr_enabled || !params.dynamic_lr_autogenerate) {
        return;
    }

    const float base_lr = std::max(params.learning_rate, 5.0e-7f);
    const float batch_scale = std::clamp(static_cast<float>(params.batch_size) / 32.0f, 0.25f, 4.0f);
    const float grad_ref = std::max(params.grad_clip_norm, 1.0f);
    const float warmup_scale = static_cast<float>(params.warmup_steps) / 512.0f;

    // Learning rate envelope
    params.dynamic_lr_min = std::max(5.0e-7f, base_lr * 0.35f);
    params.dynamic_lr_max = std::min(5.0e-3f, std::max(params.dynamic_lr_min * 1.3f, base_lr * 1.9f));

    // Adjustment factors scale with batch size (smaller batches → gentler steps)
    params.dynamic_lr_increase_factor = std::clamp(1.0f + (0.05f / batch_scale), 1.02f, 1.18f);
    const float decrease_delta = std::clamp(0.04f * batch_scale, 0.08f, 0.35f);
    params.dynamic_lr_decrease_factor = std::clamp(1.0f - decrease_delta, 0.45f, 0.85f);

    params.dynamic_lr_max_step_up_ratio = 1.0f + (params.dynamic_lr_increase_factor - 1.0f) * 2.0f;
    params.dynamic_lr_max_step_down_ratio = 1.0f - (1.0f - params.dynamic_lr_decrease_factor) * 1.4f;

    // Gradient bounds derive from clip norm
    params.dynamic_lr_upper_grad_norm = grad_ref * 1.25f;
    params.dynamic_lr_lower_grad_norm = std::max(1.0f, grad_ref * 0.35f);

    // Loss spike sensitivity depends on plateau delta / high-loss tolerance
    const float plateau_delta = std::max(0.001f, params.auto_stop_plateau_min_delta);
    params.dynamic_lr_max_loss_jump = std::clamp(1.1f + plateau_delta * 40.0f, 1.2f, 3.0f);

    // Smoothing and cooldown align with warmup window
    params.dynamic_lr_smoothing = std::clamp(0.16f + warmup_scale * 0.3f, 0.12f, 0.55f);
    params.dynamic_lr_cooldown_steps = std::clamp(params.warmup_steps / 6, 1, 12);
    params.dynamic_lr_warmup_steps = params.warmup_steps;

    params.dynamic_lr_smoothing_min = std::max(0.08f, params.dynamic_lr_smoothing * 0.6f);
    params.dynamic_lr_smoothing_max = std::min(0.75f, params.dynamic_lr_smoothing * 1.4f);

    params.dynamic_lr_auto_band = true;
    params.dynamic_lr_band_sigma = std::clamp(1.0f + 0.2f * batch_scale, 1.1f, 2.5f);
    params.dynamic_lr_band_floor = std::max(2.0f, grad_ref * 0.3f);
    params.dynamic_lr_band_ceiling = std::max(params.dynamic_lr_band_floor + 6.0f, grad_ref * 1.8f);
    params.dynamic_lr_band_min_samples = std::clamp(params.batch_size / 2, 8, 64);
    params.dynamic_lr_band_min_span = std::max(0.5f, params.dynamic_lr_band_floor * 0.15f);

    params.dynamic_lr_adaptive_smoothing = true;
    params.dynamic_lr_variance_reference = grad_ref * grad_ref;

    params.dynamic_lr_adaptive_cooldown = true;
    params.dynamic_lr_cooldown_min = std::max(1, params.dynamic_lr_cooldown_steps / 2);
    params.dynamic_lr_cooldown_max = std::max(params.dynamic_lr_cooldown_steps, params.dynamic_lr_cooldown_steps * 2);

    params.dynamic_lr_adaptive_loss = true;
    params.dynamic_lr_loss_sigma = std::clamp(2.0f + params.auto_stop_high_loss_threshold / 12.0f, 2.0f, 4.5f);
    params.dynamic_lr_loss_min_samples = std::clamp(info.batches_per_epoch / 6, 6, 48);
    params.dynamic_lr_loss_floor = std::max(0.05f, plateau_delta * 8.0f);

    params.dynamic_lr_guard_logging = true;
    params.dynamic_lr_guard_floor_steps = std::max(info.batches_per_epoch * 2, params.warmup_steps);
    params.dynamic_lr_guard_grad_multiplier = std::clamp(1.15f + grad_ref / 25.0f, 1.2f, 2.5f);
    params.dynamic_lr_guard_loss_patience = std::max(8, info.batches_per_epoch / 3);
    params.dynamic_lr_guard_loss_multiplier = std::clamp(1.02f + plateau_delta * 4.0f, 1.02f, 1.25f);

    params.dynamic_lr_baseline_capture_steps = std::max(16, params.warmup_steps / 2);
    params.dynamic_lr_baseline_drift = 0.05f;
    params.dynamic_lr_momentum_interval = std::clamp(params.warmup_steps / 4, 6, 32);
    params.dynamic_lr_momentum_gain = std::clamp(0.25f + batch_scale * 0.05f, 0.25f, 0.45f);
    params.dynamic_lr_momentum_decay = 0.65f;
    params.dynamic_lr_safety_interval = std::max(1, params.dynamic_lr_momentum_interval / 3);
    params.dynamic_lr_safety_gain = 0.08f;
    params.dynamic_lr_safety_scale = std::clamp(1.8f + batch_scale * 0.4f, 2.0f, 3.8f);
}

inline DerivedScheduleInfo harmonizeTrainingHyperparameters(
    GRIM::Config::TrainingHyperparameters& params,
    const DerivationContext& context,
    LogCallback log_callback = {}) {
    DerivedScheduleInfo info;

    const int safe_batch_size = std::max(1, params.batch_size);
    const int sequence_count = std::max(0, context.train_sequence_count);

    info.batches_per_epoch = std::max(1, (sequence_count + safe_batch_size - 1) / safe_batch_size);
    info.total_training_steps = std::max(1, info.batches_per_epoch * std::max(1, params.epochs));
    info.safe_last_step = std::max(info.total_training_steps - 1, 1);

    autoPopulateDynamicLR(params, info);

    auto log_adjustment = [&](std::string_view label, auto before, auto after) {
        if (!log_callback || before == after) {
            return;
        }
        std::ostringstream oss;
        oss << "[ConfigAdjust] " << label << " " << before << " -> " << after;
        log_callback(oss.str());
    };

    auto ensure_ordered = [&](auto& low, auto& high, std::string_view low_label, std::string_view high_label) {
        if (low <= high) {
            return;
        }
        if (log_callback) {
            std::ostringstream oss;
            oss << "[ConfigAdjust] swapped " << low_label << " and " << high_label
                << " (expected " << low_label << " <= " << high_label << ")";
            log_callback(oss.str());
        }
        std::swap(low, high);
    };

    ensure_ordered(params.dynamic_lr_min, params.dynamic_lr_max, "dynamic_lr_min", "dynamic_lr_max");
    ensure_ordered(params.dynamic_lr_smoothing_min, params.dynamic_lr_smoothing_max,
                   "dynamic_lr_smoothing_min", "dynamic_lr_smoothing_max");
    ensure_ordered(params.dynamic_lr_cooldown_min, params.dynamic_lr_cooldown_max,
                   "dynamic_lr_cooldown_min", "dynamic_lr_cooldown_max");
    ensure_ordered(params.dynamic_lr_band_floor, params.dynamic_lr_band_ceiling,
                   "dynamic_lr_band_floor", "dynamic_lr_band_ceiling");

    const int cadence_reference = std::max(info.batches_per_epoch, std::max(0, context.validation_interval));

    const int original_warmup = params.warmup_steps;
    params.warmup_steps = std::min(params.warmup_steps, info.safe_last_step);
    log_adjustment("warmup_steps", original_warmup, params.warmup_steps);

    const int original_lr_warmup = params.dynamic_lr_warmup_steps;
    params.dynamic_lr_warmup_steps = std::max(params.dynamic_lr_warmup_steps, params.warmup_steps);
    log_adjustment("dynamic_lr_warmup_steps", original_lr_warmup, params.dynamic_lr_warmup_steps);

    const int original_validation_interval = params.validation_interval;
    params.validation_interval = std::max(1, params.validation_interval);
    log_adjustment("validation_interval", original_validation_interval, params.validation_interval);

    const int required_micro_min = params.warmup_steps + params.dynamic_lr_cooldown_max;
    const int original_micro_min = params.micro_validation_min_step;
    params.micro_validation_min_step = std::clamp(
        std::max(params.micro_validation_min_step, required_micro_min),
        0,
        info.safe_last_step);
    log_adjustment("micro_validation_min_step", original_micro_min, params.micro_validation_min_step);

    const int original_micro_interval = params.micro_validation_interval;
    params.micro_validation_interval = std::clamp(
        std::max(1, params.micro_validation_interval),
        1,
        info.safe_last_step);
    log_adjustment("micro_validation_interval", original_micro_interval, params.micro_validation_interval);

    const int original_micro_batches = params.micro_validation_batch_limit;
    params.micro_validation_batch_limit = std::max(1, params.micro_validation_batch_limit);
    log_adjustment("micro_validation_batch_limit", original_micro_batches, params.micro_validation_batch_limit);

    const int original_sr_window = params.soft_restart_max_step_window;
    params.soft_restart_max_step_window = std::max(params.soft_restart_max_step_window, cadence_reference);
    log_adjustment("soft_restart_max_step_window", original_sr_window, params.soft_restart_max_step_window);

    const int original_sr_cooldown = params.soft_restart_cooldown_steps;
    params.soft_restart_cooldown_steps = std::max(params.soft_restart_cooldown_steps, cadence_reference);
    log_adjustment("soft_restart_cooldown_steps", original_sr_cooldown, params.soft_restart_cooldown_steps);

    const float original_lr = params.learning_rate;
    params.learning_rate = std::clamp(params.learning_rate, params.dynamic_lr_min, params.dynamic_lr_max);
    log_adjustment("learning_rate_clamp", original_lr, params.learning_rate);

    const float original_smoothing = params.dynamic_lr_smoothing;
    params.dynamic_lr_smoothing = std::clamp(
        params.dynamic_lr_smoothing,
        params.dynamic_lr_smoothing_min,
        params.dynamic_lr_smoothing_max);
    log_adjustment("dynamic_lr_smoothing", original_smoothing, params.dynamic_lr_smoothing);

    const int original_cooldown_steps = params.dynamic_lr_cooldown_steps;
    params.dynamic_lr_cooldown_steps = std::clamp(
        params.dynamic_lr_cooldown_steps,
        params.dynamic_lr_cooldown_min,
        params.dynamic_lr_cooldown_max);
    log_adjustment("dynamic_lr_cooldown_steps", original_cooldown_steps, params.dynamic_lr_cooldown_steps);

    const int original_cache_batch = params.cache_max_batch;
    params.cache_max_batch = std::max(params.cache_max_batch, safe_batch_size);
    log_adjustment("cache_max_batch", original_cache_batch, params.cache_max_batch);

    const int original_cache_seq = params.cache_max_seq_len;
    const int max_allowed_seq = std::max(1, params.max_seq_len);
    params.cache_max_seq_len = std::clamp(params.cache_max_seq_len, 1, max_allowed_seq);
    log_adjustment("cache_max_seq_len", original_cache_seq, params.cache_max_seq_len);

    const int original_auto_plateau = params.auto_stop_plateau_patience;
    params.auto_stop_plateau_patience = std::clamp(params.auto_stop_plateau_patience, 0, params.epochs);
    log_adjustment("auto_stop_plateau_patience", original_auto_plateau, params.auto_stop_plateau_patience);

    const int original_auto_high_loss = params.auto_stop_high_loss_patience;
    params.auto_stop_high_loss_patience = std::clamp(params.auto_stop_high_loss_patience, 0, params.epochs);
    log_adjustment("auto_stop_high_loss_patience", original_auto_high_loss, params.auto_stop_high_loss_patience);

    return info;
}

} // namespace HyperParameters
} // namespace GRIM

#endif // GRIM_CONFIG_AI_CONFIG_PATHS_HPP_INCLUDED

//======================================================//
// JSON-based Config Loading (non-CUDA only)
// These functions parse JSON and cannot be compiled by nvcc
//======================================================//

#ifndef __CUDACC__

namespace GRIM {
namespace HyperParameters {

/**
 * @brief Populate model architecture values from ai_config.json
 * 
 * This is THE authoritative function for getting model architecture.
 * All other code should call this rather than using hardcoded values.
 * 
 * Priority order:
 * 1. ai_config.json training.config section (runtime overrides)
 * 2. HyperParameters::DEFAULT_* constants (compile-time fallbacks)
 * 
 * Note: vocab_size is NOT loaded here - it comes from the .grmt training data file
 * 
 * @param arch Output ModelArchitecture struct to populate
 * @param configPath Path to ai_config.json (defaults to auto-discovery)
 * @return true if config was loaded, false if using defaults only
 */
inline bool loadModelArchitecture(ModelArchitecture& arch, const std::string& configPath = "ai_config.json") {
    // Start with defaults
    arch.d_model = DEFAULT_D_MODEL;
    arch.num_layers = DEFAULT_NUM_LAYERS;
    arch.num_heads = DEFAULT_NUM_HEADS;
    arch.d_ff = DEFAULT_D_FF;
    arch.max_seq_len = DEFAULT_MAX_SEQ_LEN;
    arch.dropout_rate = DEFAULT_DROPOUT_RATE;
    arch.attention_dropout = DEFAULT_ATTENTION_DROPOUT;
    
    // Try to load from config
    auto snapshot = GRIM::Config::loadAiConfigSnapshot(configPath);
    if (!snapshot || !snapshot->has_training) {
        arch.validate();
        return false;
    }
    
    // Override with config values
    const auto& hp = snapshot->hyperparameters;
    
    // Check for explicit model architecture in training.config
    const auto& doc = snapshot->document;
    if (doc.contains("training") && doc["training"].contains("config")) {
        const auto& cfg = doc["training"]["config"];
        
        if (cfg.contains("d_model") && cfg["d_model"].is_number()) {
            arch.d_model = cfg["d_model"].get<int>();
        }
        if (cfg.contains("num_layers") && cfg["num_layers"].is_number()) {
            arch.num_layers = cfg["num_layers"].get<int>();
        }
        if (cfg.contains("num_heads") && cfg["num_heads"].is_number()) {
            arch.num_heads = cfg["num_heads"].get<int>();
        }
        if (cfg.contains("d_ff") && cfg["d_ff"].is_number()) {
            arch.d_ff = cfg["d_ff"].get<int>();
        } else {
            // Compute d_ff from d_model if not explicitly set
            arch.d_ff = arch.d_model * DEFAULT_D_FF_MULTIPLIER;
        }
        if (cfg.contains("dropout_rate") && cfg["dropout_rate"].is_number()) {
            arch.dropout_rate = cfg["dropout_rate"].get<float>();
        }
        if (cfg.contains("attention_dropout") && cfg["attention_dropout"].is_number()) {
            arch.attention_dropout = cfg["attention_dropout"].get<float>();
        }
    }
    
    // max_seq_len comes from hyperparameters
    arch.max_seq_len = hp.max_seq_len;
    
    // Validate and compute derived values (head_dim)
    arch.validate();
    
    return true;
}

/**
 * @brief Get model architecture with single call (convenience function)
 * 
 * Returns a fully populated ModelArchitecture struct.
 * Uses config if available, falls back to defaults.
 */
inline ModelArchitecture getModelArchitecture(const std::string& configPath = "ai_config.json") {
    ModelArchitecture arch;
    loadModelArchitecture(arch, configPath);
    return arch;
}

/**
 * @brief Print model architecture for debugging
 */
inline void printModelArchitecture(const ModelArchitecture& arch) {
    std::cout << "[HyperParameters] Model Architecture:" << std::endl;
    std::cout << "  d_model: " << arch.d_model << std::endl;
    std::cout << "  num_layers: " << arch.num_layers << std::endl;
    std::cout << "  num_heads: " << arch.num_heads << std::endl;
    std::cout << "  head_dim: " << arch.head_dim << std::endl;
    std::cout << "  d_ff: " << arch.d_ff << std::endl;
    std::cout << "  max_seq_len: " << arch.max_seq_len << std::endl;
    std::cout << "  dropout_rate: " << arch.dropout_rate << std::endl;
    std::cout << "  attention_dropout: " << arch.attention_dropout << std::endl;
}

} // namespace HyperParameters
} // namespace GRIM

#endif // __CUDACC__
