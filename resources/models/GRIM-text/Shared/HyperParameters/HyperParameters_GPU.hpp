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
// Residual and attention dropout are always derived from dropout_rate (no separate defaults)

// Derived model constants
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
constexpr int TELEMETRY_MAX_STREAMS = 40;         // TelemetryLattice metric streams

//======================================================//
// Equation Logging Configuration
// Controls diagnostic equation-based logging for training
//======================================================//
constexpr bool DEFAULT_EQ_LOG_ENABLED = false;    // Disabled: equation logging causes GPU sync + D2H every forward
constexpr int DEFAULT_EQ_LOG_INTERVAL = 1;       // Log every N batches (0 = every batch)

//======================================================//
// Gradient Scaling Defaults
//======================================================//
constexpr bool DEFAULT_GRAD_SCALE_PER_TOKEN = false;  // Apply 1/valid_tokens in backward

//======================================================//
// UnigramLM Training Constants
// Parameters for vocabulary training/pruning
//======================================================//
// EM_ITERATIONS removed — convergence-based loop in Unigram.cu (max 50, 0.01% threshold)
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
// Token layout: [0-3] Special, [4-259] Byte, [260-261] Atom (NONE+NUM), [262+] Unigram
//======================================================//
constexpr int BYTE_TOKEN_END = Tokenizer::BYTE_TOKEN_OFFSET + Tokenizer::BYTE_VOCAB_SIZE;  // 260
constexpr int ATOM_TOKEN_START = BYTE_TOKEN_END;  // First atom token ID (immediately after bytes)
// Atom type count: single source of truth is Tokenizer::kAtomTypeCount (TokenLayout.hpp)
constexpr int ATOM_TOKEN_END = ATOM_TOKEN_START + Tokenizer::kAtomTypeCount;  // 260 + kAtomTypeCount
constexpr uint32_t MAX_REASONABLE_VOCAB_SIZE = 2000000; // Sanity check for vocab detection

//======================================================//
// UnigramByte Tokenizer Initialization Constants
// Single source of truth for tokenizer configuration
// (Overridable per-run via ai_config.json [tokenizer] section)
//======================================================//
constexpr float TOKENIZER_CHARACTER_COVERAGE = 0.9995f;  // Character coverage for SentencePiece-style vocab building
constexpr bool TOKENIZER_PREFER_GPU = true;              // GPU acceleration for tokenization

// BOS/EOS Token Insertion Control
// These flags are loaded from ai_config.json [tokenizer] section:
//   add_bos: true/false - Controls whether to prepend BOS token to sequences
//   add_eos: true/false - Controls whether to append EOS token to sequences
// Used in Phase1_Startup.cu::loadTrainingData() to conditionally add boundary tokens
// See: ai_config.json [tokenizer] { "add_bos": true, "add_eos": true }

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

//======================================================//
// Positional Bias Method (PBM) Configuration
// Hybrid ALiBi + RoPE for position encoding
//======================================================//
constexpr float ROPE_THETA = 10000.0f;        // RoPE base frequency (standard: 10000)
constexpr float ROPE_SCALING = 1.0f;          // NTK scaling factor (1.0 = no scaling)
constexpr float ALIBI_SLOPE_EXPONENT = -8.0f; // Controls ALiBi slope decay across heads

// ISSUE #78: ALiBi bias capping (optional safety net for softmax backward stability)
// Issue #84 fixed the ROOT CAUSE of dQ/dK explosion (missing FlashAttention preprocessing
// kernel), so capping is no longer required for correctness.
// 
// Set to -10.0f to enable capping (limits exp(bias) ≥ exp(-10) ≈ 0.000045).
// Set to 0.0f to disable capping (current: disabled, safe after Issue #84 fix).
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
    NONE,       // Learned/additive position embeddings (no ALiBi/RoPE inside attention)
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

// Entropy Regularization: penalizes low entropy (overconfidence), encourages diversity
// reg = -λ * H(p) = λ * Σ p*log(p)
// Issue #133: Disabled by default (was masking true CE)
constexpr bool DEFAULT_LOSS_ENTROPY_REG_ENABLED = false;
constexpr float DEFAULT_LOSS_ENTROPY_REG_LAMBDA = 0.0f;  // Regularization strength when enabled (try 0.1-1.0)

// Class-balanced loss: w_v = 1/freq(v)^β reweights per-token loss
// β=0.5 → sqrt inverse frequency. Counteracts frequency-biased CE gradient.
constexpr bool DEFAULT_LOSS_CLASS_BALANCED_ENABLED = false;
constexpr float DEFAULT_LOSS_CLASS_BALANCED_BETA = 0.5f;  // sqrt inverse frequency (try 0.3-0.9)

//======================================================//
// Scratch Block Configuration (ScratchBlock Reasoning Layer)
// Pool block size is computed from max_cached_tokens in InitTrainingState
//======================================================//
constexpr bool DEFAULT_SCRATCH_BLOCKS_ENABLED = true;
constexpr size_t DEFAULT_SCRATCH_NUM_BLOCKS = 2;  // Double buffer for pinned memory staging
constexpr bool DEFAULT_SCRATCH_WRITE_COMBINED = false;

//======================================================//
// ExecutionBlock (training.config.execution_block in ai_config.json)
//
// Parsed into GRIM::Config::TrainingHyperparameters (ai_config_paths.hpp).
// Training startup maps those fields onto GRIM::LanguageModelConfig in Phase1_Startup.cu.
// LanguageModelConfig and EncoderConfig inherit from ModelArchitecture —
// architecture fields are defined ONLY here (Single Source of Truth).
//======================================================//

//======================================================//
// Model Architecture Validation & Computation
//======================================================//
struct ModelArchitecture {
    int d_model = DEFAULT_D_MODEL;
    int num_layers = DEFAULT_NUM_LAYERS;
    int num_heads = DEFAULT_NUM_HEADS;
    int num_kv_heads = DEFAULT_NUM_KV_HEADS;  // GQA: number of KV heads
    int d_ff = 0;  // Must be set from config or computed as d_model * multiplier
    int max_seq_len = DEFAULT_MAX_SEQ_LEN;
    float dropout_rate = DEFAULT_DROPOUT_RATE;
    float attention_dropout = DEFAULT_DROPOUT_RATE;        // Derived: always = dropout_rate
    bool tie_embeddings = true;  // Weight tying: share embedding/LM head weights
    PositionalEncodingType positional_encoding = DEFAULT_POSITIONAL_ENCODING;
    
    // Derived (computed from above)
    int head_dim = DEFAULT_HEAD_DIM;
    
    // Validate architecture and compute derived values. Throws on invalid config.
    void validate() {
        if (num_heads <= 0) {
            throw std::runtime_error("Invalid ModelArchitecture: num_heads must be > 0, got " + std::to_string(num_heads));
        }
        
        // Validate divisibility BEFORE computing head_dim
        if (d_model % num_heads != 0) {
            throw std::runtime_error("Invalid ModelArchitecture: d_model (" + std::to_string(d_model) +
                                     ") must be divisible by num_heads (" + std::to_string(num_heads) + ")");
        }
        head_dim = d_model / num_heads;
        
        // Validate Flash Attention compatibility
        if (!isValidFlashAttentionHeadDim(head_dim)) {
            throw std::runtime_error("Invalid ModelArchitecture: head_dim=" + std::to_string(head_dim) +
                                     " not supported by Flash Attention (need 32 or 64)");
        }
        
        // Validate GQA configuration
        if (!isValidGQAConfig(num_heads, num_kv_heads)) {
            throw std::runtime_error("Invalid GQA configuration: num_heads=" + std::to_string(num_heads) +
                                     " num_kv_heads=" + std::to_string(num_kv_heads) +
                                     " (num_heads must be divisible by num_kv_heads)");
        }
        
        // Compute d_ff if not explicitly set
        if (d_ff <= 0) {
            d_ff = d_model * DEFAULT_D_FF_MULTIPLIER;
        }
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

inline DerivedScheduleInfo harmonizeTrainingHyperparameters(
    GRIM::Config::TrainingHyperparameters& params,
    const DerivationContext& context,
    LogCallback log_callback = {}) {
    // Rule 20: Fail hard on invalid config instead of silently fixing
    if (params.batch_size <= 0) {
        throw std::runtime_error("FATAL: batch_size must be > 0, got " + std::to_string(params.batch_size));
    }
    if (params.epochs <= 0) {
        throw std::runtime_error("FATAL: epochs must be > 0, got " + std::to_string(params.epochs));
    }
    if (params.validation_interval <= 0) {
        throw std::runtime_error("FATAL: validation_interval must be > 0, got " + std::to_string(params.validation_interval));
    }

    DerivedScheduleInfo info;

    const int safe_batch_size = params.batch_size;
    const int sequence_count = std::max(0, context.train_sequence_count);

    info.batches_per_epoch = std::max(1, (sequence_count + safe_batch_size - 1) / safe_batch_size);
    info.total_training_steps = std::max(1, info.batches_per_epoch * params.epochs);
    info.safe_last_step = std::max(info.total_training_steps - 1, 1);

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

    const int cadence_reference = std::max(info.batches_per_epoch, std::max(0, context.validation_interval));

    // warmup_steps is derived in Phase2 from warmup_fraction * estimated_total_steps.
    // At harmonize time it is 0 — skip clamping here; it will be set correctly in Phase2.

    // validation_interval already validated > 0 above

    const int original_sr_window = params.soft_restart_max_step_window;
    params.soft_restart_max_step_window = std::max(params.soft_restart_max_step_window, cadence_reference);
    log_adjustment("soft_restart_max_step_window", original_sr_window, params.soft_restart_max_step_window);

    const int original_sr_cooldown = params.soft_restart_cooldown_steps;
    params.soft_restart_cooldown_steps = std::max(params.soft_restart_cooldown_steps, cadence_reference);
    log_adjustment("soft_restart_cooldown_steps", original_sr_cooldown, params.soft_restart_cooldown_steps);

    const int original_auto_plateau = params.auto_stop_plateau_patience;
    params.auto_stop_plateau_patience = std::max(0, params.auto_stop_plateau_patience);
    log_adjustment("auto_stop_plateau_patience", original_auto_plateau, params.auto_stop_plateau_patience);

    const int original_auto_high_loss = params.auto_stop_high_loss_patience;
    params.auto_stop_high_loss_patience = std::max(0, params.auto_stop_high_loss_patience);
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
    // Start with defaults (derived fields computed from base defaults)
    arch.d_model = DEFAULT_D_MODEL;
    arch.num_layers = DEFAULT_NUM_LAYERS;
    arch.num_heads = DEFAULT_NUM_HEADS;
    arch.d_ff = DEFAULT_D_MODEL * DEFAULT_D_FF_MULTIPLIER;
    arch.max_seq_len = DEFAULT_MAX_SEQ_LEN;
    arch.dropout_rate = DEFAULT_DROPOUT_RATE;
    arch.attention_dropout = DEFAULT_DROPOUT_RATE;       // = dropout_rate
    
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
        if (cfg.contains("num_kv_heads") && cfg["num_kv_heads"].is_number()) {
            arch.num_kv_heads = cfg["num_kv_heads"].get<int>();
        }
        // d_ff: always derived as d_model * DEFAULT_D_FF_MULTIPLIER (never read from JSON)
        arch.d_ff = arch.d_model * DEFAULT_D_FF_MULTIPLIER;
        if (cfg.contains("tie_embeddings") && cfg["tie_embeddings"].is_boolean()) {
            arch.tie_embeddings = cfg["tie_embeddings"].get<bool>();
        }
        if (cfg.contains("dropout_rate") && cfg["dropout_rate"].is_number()) {
            arch.dropout_rate = cfg["dropout_rate"].get<float>();
        }
        // attention_dropout: always derived from dropout_rate
        arch.attention_dropout = arch.dropout_rate;

        // Issue #142: Parse positional encoding from shared architecture loader.
        // This path is used by runtime inference/server startup and must match
        // Phase1 training config semantics.
        if (cfg.contains("positional_encoding")) {
            const auto& pe = cfg["positional_encoding"];
            if (pe.is_object()) {
                const bool use_rope = pe.value("use_rope", false);
                const bool use_alibi = pe.value("use_alibi", false);
                const bool use_learned = pe.value("use_learned", false);
                if (use_learned) {
                    arch.positional_encoding = PositionalEncodingType::NONE;
                } else if (use_rope && use_alibi) {
                    arch.positional_encoding = PositionalEncodingType::ALIBI_ROPE;
                } else if (use_rope) {
                    arch.positional_encoding = PositionalEncodingType::ROPE;
                } else if (use_alibi) {
                    arch.positional_encoding = PositionalEncodingType::ALIBI;
                } else {
                    arch.positional_encoding = PositionalEncodingType::NONE;
                }
            } else if (pe.is_string()) {
                arch.positional_encoding = parsePositionalEncodingType(pe.get<std::string>());
            } else {
                throw std::runtime_error(
                    "loadModelArchitecture: training.config.positional_encoding must be object or string");
            }
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
