#pragma once
// Traditional include guard in addition to #pragma once: nvcc dedupes by
// path-string, not by canonical inode, so the same header pulled in via two
// different ../ chains (e.g. .../GRIM/../Shared/... vs .../Layers/Embedding/../../Shared/...)
// would otherwise be expanded twice and produce "already defined" errors on
// the constexpr globals.
#ifndef GRIM_SHARED_HYPERPARAMETERS_GPU_HPP
#define GRIM_SHARED_HYPERPARAMETERS_GPU_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

// JSON loaders + their stdlib dependencies are host-only. By default we omit
// them so light CUDA TUs (Unigram.cu / UnigramTrainer.cu) that only need a
// couple of constexpr constants don't drag in <map>/<set>/<filesystem>/
// <nlohmann> (gcc 8 + nvcc choke on the locale machinery they pull in).
//
// Hosts (and CUDA TUs that DO need TrainingHyperparameters) opt in by defining
// GRIM_HP_HOST_TYPES_REQUIRED *before* including this header — ai_config_paths.hpp
// does this for everything that goes through it.
#if !defined(__CUDACC__) || defined(GRIM_HP_HOST_TYPES_REQUIRED)
#define GRIM_HP_HOST_TYPES_AVAILABLE 1
#include <fstream>
#include <filesystem>
#include <map>
#include <set>
#include <vector>
#include <nlohmann/json.hpp>
#endif

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
constexpr int DEFAULT_MAX_SEQ_LEN = 1024;
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
constexpr int TELEMETRY_MAX_STREAMS = 48;         // TelemetryLattice metric streams

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

//======================================================//
// RAdamW Optimizer Hyperparameters
// Rectified AdamW (Liu et al. 2019 + Loshchilov & Hutter 2019
// decoupled weight decay). Shares m/v moment buffers with AdamW so
// checkpoints stay format-compatible. Rectification is unconditional
// — it IS RAdamW; callers wanting plain AdamW use kind="adamw".
//======================================================//
constexpr float RADAMW_BETA1 = 0.9f;
constexpr float RADAMW_BETA2 = 0.999f;
constexpr float RADAMW_EPSILON = 1e-8f;

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


// Only compile if TrainingHyperparameters is fully defined
// (checked via the marker defined in ai_config_paths.hpp)

#ifdef GRIM_HP_HOST_TYPES_AVAILABLE


//======================================================//
// TrainingHyperparameters + JSON loaders (Phase A move)
//
// Previously lived in control/ai_config_paths.hpp. Moved here so that
// HyperParameters_GPU.hpp is the single source of truth and consumers
// no longer need to include the JSON-reader header directly.
//======================================================//


namespace GRIM {
namespace Config {

// ── LogRecorderConfig struct ──
/**
 * @brief Log Recorder configuration
 * 
 * Controls modular logging levels and overrides.
 */
struct LogRecorderConfig {
    bool enabled = true;
    std::string default_level = "Info";
    std::map<std::string, std::string> modules;
    
    // Layer logging enables (for RecordLayerLogHost)
    struct LayerEnables {
        bool embedding = true;
        bool rms_norm = true;
        bool attention = true;
        bool feed_forward = true;
        bool residual = true;
        bool encoding = true;     // Aggregate gradient logs
        bool serialization = true;
        bool execution_block = true;
    } layers;
};

// ── TapeLogConfig struct ──
/**
 * @brief Unified tape-based logging configuration (replaces LogRecorderConfig).
 * Parsed from training.config.logging in ai_config.json.
 */
struct TapeLogConfig {
    std::string default_level = "Info";
    bool equation_csv_enabled = true;
    bool stderr_enabled = true;
    size_t initial_capacity = 2048;
    std::map<std::string, std::string> group_overrides;
};

// ── TrainingHyperparameters struct ──
/**
 * @brief Load training configuration from ai_config.json
 * 
 * This loads the training hyperparameters (epochs, batch_size, learning_rate, etc.)
 * from the training.config section of ai_config.json.
 * 
 * REQUIRED FIELDS: ALL fields must be explicitly set in ai_config.json (Rule 20: fail loud)
 * NO DEFAULTS - every field must come from JSON configuration.
 */
struct TrainingHyperparameters {
    
    // Training run selectors — which model and curriculum to use
    std::string current_model_training;
    std::string current_curriculum;

    // Log Recorder configuration
    LogRecorderConfig log_recorder;

    // Unified tape-based logging configuration
    TapeLogConfig tape_logging;

    // Core training parameters - NO DEFAULTS
    int epochs;
    int64_t seed;
    int batch_size;
    int gradient_accumulation_steps;
    bool single_batch_overfit_enabled;
    int single_batch_overfit_max_steps;
    std::string batch_strategy;
    float learning_rate;
    // min_lr removed: cosine_decay_min_lr (from cosine_decay.min_lr) is the authoritative floor
    float weight_decay;
    float grad_clip_norm;
    bool per_token_grad_scale;
    float warmup_fraction;  // Fraction of total optimizer steps for warmup (e.g. 0.05 = 5%)
    int warmup_steps = 0;   // Derived in Phase2 from warmup_fraction * estimated_total_steps
    bool cosine_decay_enabled;
    bool cosine_warm_restarts;
    float cosine_decay_min_lr;
    int max_seq_len;
    int min_seq_valid_tokens;  // Minimum valid tokens required (after masking first/last positions)
    int log_interval;
    int atom_stats_interval;
    int atom_stats_max_seqs;
    int validation_interval;
    int checkpoint_interval;
    bool use_gpu;
    bool use_flash_attention;
    int min_seq_len_for_flash;
    
    // Soft restart - NO DEFAULTS
    bool soft_restart_enabled;
    float soft_restart_loss_increase_threshold;
    int soft_restart_max_step_window;
    int soft_restart_cooldown_steps;
    
    // Auto stop - NO DEFAULTS
    bool auto_stop_enabled;
    int auto_stop_plateau_patience;
    float auto_stop_plateau_min_delta;
    float auto_stop_high_loss_threshold;
    int auto_stop_high_loss_patience;
    
    // Guess aux - NO DEFAULTS
    bool guess_aux_enabled;
    float guess_aux_lambda;
    float guess_aux_min_confidence;
    
    // Shuffle - NO DEFAULTS
    bool shuffle_train_enabled;
    int shuffle_train_epochs;
    
    // Telemetry control - NO DEFAULTS
    bool telemetry_control_enabled;
    float telemetry_spike_mild_threshold;
    float telemetry_spike_moderate_threshold;
    float telemetry_spike_severe_threshold;
    float telemetry_moderate_grad_scale;
    int telemetry_moderate_cooldown_extension;
    float telemetry_min_grad_for_nonzero_loss;
    float telemetry_loss_threshold_for_grad_check;
    int telemetry_max_consecutive_zero_grad_steps;
    float telemetry_seq_len_regime_change_threshold;
    int telemetry_regime_change_suppression_steps;
    float telemetry_volatility_damping_threshold;
    float telemetry_max_volatility_damping;
    float telemetry_gradient_decay_threshold;
    float telemetry_max_decay_boost;
    float telemetry_progress_boost_threshold;
    float telemetry_max_progress_boost;
    float telemetry_outlier_frequency_trigger;
    float telemetry_outlier_persistence_trigger;
    float telemetry_anchor_drift_sigma_multiplier;
    int telemetry_soft_restart_cooldown_steps;
    int telemetry_warmup_steps;
    int telemetry_baseline_stabilization_steps;
    bool telemetry_verbose_logging;
    bool telemetry_fail_loud_on_accumulation_bug;
    bool telemetry_plateau_noise_enabled;
    int telemetry_plateau_noise_patience;
    float telemetry_plateau_noise_variance_threshold;
    float telemetry_plateau_noise_std;
    bool telemetry_plateau_noise_proportional;
    int telemetry_plateau_noise_cooldown;
    int telemetry_plateau_noise_max_per_epoch;
    
    // Loss options - NO DEFAULTS
    bool loss_label_smoothing_enabled;
    float loss_label_smoothing_epsilon;
    bool loss_focal_enabled;
    float loss_focal_gamma;
    float loss_focal_alpha;
    bool loss_preference_enabled;
    float loss_preference_beta;
    bool loss_distillation_enabled;
    float loss_distillation_temperature;
    float loss_distillation_lambda;
    bool loss_masking_enabled;
    std::string loss_masking_tag;
    
    // Issue #44 FIX: Entropy regularization to prevent mode collapse
    // reg = λ * Σ_v p_v² (penalizes logit concentration)
    bool loss_entropy_reg_enabled;
    float loss_entropy_reg_lambda;

    // Class-balanced loss: reweights per-token loss by 1/freq^β
    bool loss_class_balanced_enabled = false;
    float loss_class_balanced_beta = 0.5f;


    // LM Head centering (Issue #37 / #40) - NO DEFAULTS
    // When enabled, centers hidden states before LM head projection.
    // Centering backward handled by autograd GradFns (Issue #142).
    bool lm_head_centering_enabled;
    bool lm_head_center_hidden_states;
    bool lm_head_freeze_final_rms_gamma;  // Freeze γ_final at 1.0 (no requires_grad, no param group)
    bool center_logits;  // Center logits per position (row-wise) before softmax
    bool center_encoder_residuals;  // Center residuals INSIDE encoder layers (24 projections - can attenuate gradients)
    bool project_out_pc1;            // Issue #149: project out PC1 direction before LM head
    int  pc1_power_iters;            // Power iteration steps for PC1 (default 5)
    
    // Issue #109: LayerScale (learnable residual scaling from CaiT paper)
    // Reduces correlation buildup between layers by gating sublayer outputs
    // with learnable scalars (initialized to layer_scale_init, typically 0.1)
    bool use_layer_scale;
    float layer_scale_init;
    
    // QK-norm: Per-head RMSNorm on Q and K before RoPE (Gemma-2 style)
    // Bounds attention logit magnitudes, prevents entropy collapse
    bool qk_norm_enabled;
    
    // Hardcoded Hidden States Diagnostic (Issue #42) - NO DEFAULTS
    // When enabled, replaces encoder output with synthetic patterns to isolate
    // whether mode collapse is caused by encoder or LM head/gradient system.
    // Requires including grim_language_model_cuda.hpp for HardcodedPattern enum.
    int hardcoded_hidden_pattern;  // 0=DISABLED, 1=RANDOM_CENTERED, 2=ORTHOGONAL_W277, etc.
    int hardcoded_log_every_n_batches;
    
    // Embedding freeze guard - freezes embedding weights after N optimizer steps
    bool embedding_freeze_enabled = false;
    int embedding_freeze_after_step = 0;

    // Optimizer selection - NO DEFAULTS for kind, but betas/eps fall back to
    // HyperParameters constants if absent in JSON. Single source of truth for
    // the *constants* is HyperParameters_GPU.hpp; runtime values live here and
    // are passed to launch*Step(...) by signature.
    //
    // optimizer_kind ∈ {"adamw", "radamw"}.  Default "adamw" preserves prior behavior.
    // Selecting "radamw" runs the full Liu et al. 2019 rectified update with
    // decoupled weight decay (the "R" + "W" in RAdamW). There is no rectification
    // toggle — rectification IS RAdamW; for plain bias-corrected AdamW use "adamw".
    std::string optimizer_kind = "adamw";
    float optimizer_beta1   = 0.9f;
    float optimizer_beta2   = 0.999f;
    float optimizer_epsilon = 1e-8f;

    // Stability overrides - NO DEFAULTS
    bool stability_overrides_enabled;
    int stability_override_batch_size;
    int stability_override_max_seq_len;
    float stability_override_clip_per_token;
    float stability_override_lr_min;
    
    // Scratch blocks - NO DEFAULTS
    bool scratch_blocks_enabled;
    size_t scratch_max_tokens_per_block;
    size_t scratch_num_blocks;
    bool scratch_write_combined;
    
    // ScratchBlock reasoning - NO DEFAULTS
    bool scratch_block_reasoning_enabled;
    int scratch_block_reasoning_atom_embedding_dim;
    int scratch_block_reasoning_max_atoms;
    float scratch_block_reasoning_atom_scale;

    // ExecutionBlock + execution-first loss (training.config.execution_block) — NO DEFAULTS
    bool execution_block_enabled;
    /// When true with ExecutionBlock: ScratchBlock uses type embedding only (matches LanguageModelConfig).
    bool scratch_block_execution_first_type_only;
    int execution_block_layer;
    int execution_block_num_ops;
    int execution_block_num_slots;
    int execution_block_num_steps;
    int execution_block_d_key;
    int execution_block_d_type;
    int execution_block_cross_attn_head_dim;
    int execution_block_cross_attn_topk;
    float execution_block_usage_decay;
    float execution_block_diversity_kappa;
    float execution_block_temp_start;
    float execution_block_temp_end;
    int execution_block_temp_schedule;
    float execution_block_entropy_weight;
    float execution_step_x_multiplier;
    float execution_step_y_multiplier;
    bool execution_step_y_overrides_x;
    float execution_entropy_aux_weight;
    float execution_value_match_epsilon;
    float execution_final_slot_consistency_weight;

    // Causal state loss (Fixes 1-9)
    float execution_block_transition_hard_threshold;
    int   execution_block_gate_warmup_steps;
    float execution_block_causal_w1_transition;

    // Fix #6: Division invalid penalty (penalize selecting ÷ when |v2| < eps)
    float execution_div_invalid_penalty_weight;

    // Fix #8: Division magnitude penalty (penalize large |v_out| after clamped division)
    float execution_div_magnitude_penalty_weight;

    // Fix #7: Arg REINFORCE weight (0 = disabled)
    float execution_arg_reinforce_weight;
    float execution_arg_reinforce_baseline_decay;

    // Autograd structured CE
    bool  structured_ce_enabled;
    float structured_ce_weight;

    // Decode-time slot selector
    bool  selector_enabled;
    int   selector_d_selector;
    float selector_selection_margin;
    float selector_supervision_weight;

    
    // Activation quantization - NO DEFAULTS
    bool activation_quantization_enabled;
    bool activation_quantization_apply_to_embeddings;
    bool activation_quantization_apply_to_encoder_outputs;
    bool activation_quantization_apply_to_layer_caches;
    bool activation_quantization_apply_to_qkv_cache;
    bool activation_quantization_apply_to_logits;
    float activation_quantization_scale;
    float activation_quantization_clip_min;
    float activation_quantization_clip_max;
    int activation_quantization_zero_point;
    bool activation_quantization_symmetric;
    
    // CUDA execution mode - NO DEFAULTS
    bool single_stream_mode;
    bool disable_async_frees;
    bool synchronize_after_kernels;
    
    // Multi-token prediction (MTP) - auxiliary heads for trajectory learning
    bool mtp_enabled = false;
    int mtp_k = 0;
    float mtp_alpha = 0.2f;
    int mtp_alpha_warmup_steps = 500;
    bool mtp_log_ratio_monitor = true;

    // Prediction comparison - NO DEFAULTS
    bool prediction_comparison_enabled;
    int prediction_comparison_interval;
    int prediction_comparison_top_k;
    int prediction_comparison_max_positions;
    std::string prediction_comparison_log_path;
    
    // Logit update trace - NO DEFAULTS
    bool logit_update_trace_enabled;
    int logit_update_trace_interval;
    
    // Attention diagnostics - NO DEFAULTS
    bool attention_diag_enabled;
    int attention_diag_layer;
    int attention_diag_head;
    
    // Scratch block reasoning configuration - NO DEFAULTS
    bool tokenizer_enable_scratch_block_reasoning;
    bool tokenizer_detect_numbers;
};

// ── assignTrainingField ──
template <typename FieldType>
inline void assignTrainingField(FieldType& field, const nlohmann::json& node, const char* key) {
    auto it = node.find(key);
    if (it != node.end() && !it->is_null()) {
        field = it->get<std::decay_t<FieldType>>();
    }
}

// ── setDefaultHyperparameters ──
/**
 * @brief Set sensible defaults for all non-base hyperparameters.
 * Called BEFORE JSON parsing, so JSON values override these defaults.
 * Future: Replace this function body with a trained hyperparameter prediction model.
 * @param params The hyperparameters struct to populate with defaults
 */
inline void setDefaultHyperparameters(TrainingHyperparameters& params) {
    // ── Auto stop ──
    params.auto_stop_plateau_patience = 18;
    params.auto_stop_plateau_min_delta = 0.004f;
    params.auto_stop_high_loss_threshold = 6.0f;
    params.auto_stop_high_loss_patience = 12;

    // ── Soft restart ──
    params.soft_restart_loss_increase_threshold = 3.0f;
    params.soft_restart_max_step_window = 50;
    params.soft_restart_cooldown_steps = 200;

    // ── Single batch overfit ──
    params.single_batch_overfit_max_steps = 1000;

    // ── Guess aux ──
    params.guess_aux_lambda = 0.25f;
    params.guess_aux_min_confidence = 0.7f;

    // ── Shuffle ──
    params.shuffle_train_epochs = 1;

    // ── Embedding freeze ──
    params.embedding_freeze_after_step = 0;

    // ── Scratch blocks ──
    params.scratch_num_blocks = 4;
    params.scratch_write_combined = true;

    // ── Scratch block reasoning ──
    params.scratch_block_reasoning_max_atoms = 8192;
    params.scratch_block_reasoning_atom_scale = 1.0f;

    // ── Telemetry control ──
    params.telemetry_spike_mild_threshold = 3.0f;
    params.telemetry_spike_moderate_threshold = 5.0f;
    params.telemetry_spike_severe_threshold = 10.0f;
    params.telemetry_moderate_grad_scale = 0.5f;
    params.telemetry_moderate_cooldown_extension = 3;
    params.telemetry_min_grad_for_nonzero_loss = 1e-10f;
    params.telemetry_loss_threshold_for_grad_check = 0.01f;
    params.telemetry_max_consecutive_zero_grad_steps = 0;
    params.telemetry_seq_len_regime_change_threshold = 0.3f;
    params.telemetry_regime_change_suppression_steps = 2;
    params.telemetry_volatility_damping_threshold = 150.0f;
    params.telemetry_max_volatility_damping = 0.9f;
    params.telemetry_gradient_decay_threshold = 0.0f;
    params.telemetry_max_decay_boost = 1.0f;
    params.telemetry_progress_boost_threshold = 100.0f;
    params.telemetry_max_progress_boost = 1.0f;
    params.telemetry_outlier_frequency_trigger = 0.95f;
    params.telemetry_outlier_persistence_trigger = 0.9f;
    params.telemetry_anchor_drift_sigma_multiplier = 5.0f;
    params.telemetry_soft_restart_cooldown_steps = 100000;
    params.telemetry_baseline_stabilization_steps = 100;
    params.telemetry_plateau_noise_patience = 30;
    params.telemetry_plateau_noise_variance_threshold = 0.008f;
    params.telemetry_plateau_noise_std = 0.1f;
    params.telemetry_plateau_noise_proportional = true;
    params.telemetry_plateau_noise_cooldown = 10;
    params.telemetry_plateau_noise_max_per_epoch = 3;

    // ── Loss sub-parameters ──
    params.loss_label_smoothing_epsilon = 0.05f;
    params.loss_focal_gamma = 0.75f;
    params.loss_focal_alpha = 1.0f;
    params.loss_preference_beta = 0.1f;
    params.loss_distillation_temperature = 1.0f;
    params.loss_distillation_lambda = 0.5f;
    params.loss_entropy_reg_lambda = 0.0f;
    params.loss_class_balanced_beta = 0.5f;

    // ── LM head centering ──
    params.pc1_power_iters = 5;

    // ── Layer scale ──
    params.layer_scale_init = 1.0f;

    // ── Execution block tuning ──
    params.execution_block_cross_attn_topk = 2;
    params.execution_block_usage_decay = 0.95f;
    params.execution_block_diversity_kappa = 2.0f;
    params.execution_block_temp_start = 1.5f;
    params.execution_block_temp_end = 0.5f;
    params.execution_block_temp_schedule = 0;
    params.execution_block_entropy_weight = 0.01f;
    params.execution_step_x_multiplier = 2.0f;
    params.execution_step_y_multiplier = 3.0f;
    params.execution_step_y_overrides_x = false;
    params.execution_entropy_aux_weight = 0.01f;
    params.execution_value_match_epsilon = 0.0001f;
    params.execution_final_slot_consistency_weight = 0.1f;
    params.execution_block_transition_hard_threshold = 0.0f;
    params.execution_block_causal_w1_transition = 1.5f;
    params.execution_div_invalid_penalty_weight = 0.5f;
    params.execution_div_magnitude_penalty_weight = 0.1f;
    params.execution_arg_reinforce_weight = 0.0f;
    params.execution_arg_reinforce_baseline_decay = 0.99f;
    params.structured_ce_weight = 0.1f;
    params.selector_d_selector = 64;
    params.selector_selection_margin = 1.0f;
    params.selector_supervision_weight = 1.0f;

    // ── Activation quantization ──
    params.activation_quantization_apply_to_embeddings = false;
    params.activation_quantization_apply_to_encoder_outputs = false;
    params.activation_quantization_apply_to_layer_caches = false;
    params.activation_quantization_apply_to_qkv_cache = false;
    params.activation_quantization_apply_to_logits = false;
    params.activation_quantization_scale = 1.0f;
    params.activation_quantization_clip_min = -127.0f;
    params.activation_quantization_clip_max = 127.0f;
    params.activation_quantization_zero_point = 0;
    params.activation_quantization_symmetric = false;

    // ── Multi-token prediction ──
    params.mtp_k = 3;
    params.mtp_alpha = 0.2f;
    params.mtp_log_ratio_monitor = true;

    // ── Diagnostics ──
    params.hardcoded_hidden_pattern = 0;
    params.hardcoded_log_every_n_batches = 1;
    params.prediction_comparison_interval = 100;
    params.prediction_comparison_top_k = 5;
    params.prediction_comparison_max_positions = 8;
    params.prediction_comparison_log_path = "resources/models/GRIM-text/training/prediction_comparison.log";
    params.logit_update_trace_interval = 50;
    params.attention_diag_layer = -1;
    params.attention_diag_head = 0;
}

// ── validateTrainingConfigJson ──
/**
 * @brief Validate required (base+enable) fields exist in training.config JSON
 * @param trainConfig The training.config JSON object
 * @throws std::runtime_error listing all missing required fields
 *
 * Rule 20: Base parameters and feature enables MUST be explicitly set.
 * Non-base parameters get sensible defaults from setDefaultHyperparameters().
 */
inline void validateTrainingConfigJson(const nlohmann::json& trainConfig) {
    static const std::vector<std::string> REQUIRED = {
        // Core training
        "epochs", "seed", "batch_size", "gradient_accumulation_steps",
        "batch_strategy", "learning_rate", "weight_decay",
        "per_token_grad_scale", "warmup_fraction", "max_seq_len", "log_interval",
        "atom_stats_interval", "atom_stats_max_seqs",
        "validation_interval", "checkpoint_interval", "use_gpu", "use_flash_attention",
        
        // Feature enables
        "cosine_decay.enabled",
        "single_batch.enabled",
        "soft_restart.enabled",
        "auto_stop.enabled",
        "guess_aux.enabled",
        "shuffle.enabled",
        
        "telemetry_control.enabled",
        "stability_overrides_enabled",
        "activation_quantization.enabled",
        "multi_token_prediction.enabled",
        "execution_block.enabled",
        "scratch_blocks.enabled",
        "scratch_block_reasoning.enabled",
        "lm_head_centering.enabled",
        "layer_scale.enabled",
        "qk_norm.enabled",
        "embedding_freeze.enabled",
        "hardcoded_hidden_states.enabled",
        "prediction_comparison.enabled",
        "logit_update_trace.enabled",
        "attention_diagnostics.enabled",

        // Loss enables
        "loss.label_smoothing.enabled",
        "loss.focal.enabled",
        "loss.preference.enabled",
        "loss.distillation.enabled",
        "loss.masking.enabled", "loss.masking.tag",

        // Execution block structural (architecture choices)
        "execution_block.execution_first_type_only",
        "execution_block.layer",
        "execution_block.num_ops",
        "execution_block.num_slots",
        "execution_block.num_steps",
        "execution_block.d_type",
        "execution_block.structured_ce_enabled",
        "execution_block.selector.enabled",

        // LM head centering choices
        "lm_head_centering.center_hidden_states",
        "lm_head_centering.center_logits",
        "lm_head_centering.center_encoder_residuals",
        "lm_head_centering.project_out_pc1",

        // Logging
        "log_recorder.enabled", "log_recorder.default_level",
        "telemetry_control.logging.verbose",
        "telemetry_control.logging.fail_loud_on_accumulation_bug",
        "telemetry_control.plateau_noise.enabled",

        // CUDA execution
        "cuda_execution.single_stream_mode",
        "cuda_execution.disable_async_frees",
        "cuda_execution.synchronize_after_kernels",
    };
    
    std::vector<std::string> missing;
    
    for (const auto& path : REQUIRED) {
        if (!jsonPathExists(trainConfig, path)) {
            missing.push_back(path);
        }
    }
    
    // Check gradient clip (accepts either name)
    if (!trainConfig.contains("grad_clip_norm") && !trainConfig.contains("gradient_clip")) {
        missing.push_back("grad_clip_norm (or gradient_clip)");
    }
    
    if (!missing.empty()) {
        std::ostringstream oss;
        oss << "FATAL: ai_config.json training.config missing " << missing.size() << " required fields:\n";
        for (const auto& m : missing) {
            oss << "  - " << m << "\n";
        }
        oss << "\nRule 20: Base parameters and feature enables MUST be explicitly set.\n";
        oss << "Non-base parameters get sensible defaults from setDefaultHyperparameters().";
        throw std::runtime_error(oss.str());
    }
}

// ── applyTrainingConfigObject ──
inline void applyTrainingConfigObject(const nlohmann::json& trainConfig, TrainingHyperparameters& params) {
    if (!trainConfig.is_object()) {
        return;
    }
    
    assignTrainingField(params.epochs, trainConfig, "epochs");
    assignTrainingField(params.seed, trainConfig, "seed");
    assignTrainingField(params.batch_size, trainConfig, "batch_size");
    assignTrainingField(params.gradient_accumulation_steps, trainConfig, "gradient_accumulation_steps");
    assignTrainingField(params.batch_strategy, trainConfig, "batch_strategy");
    assignTrainingField(params.learning_rate, trainConfig, "learning_rate");
    assignTrainingField(params.weight_decay, trainConfig, "weight_decay");
    assignTrainingField(params.grad_clip_norm, trainConfig, "gradient_clip");
    assignTrainingField(params.grad_clip_norm, trainConfig, "grad_clip_norm");
    assignTrainingField(params.per_token_grad_scale, trainConfig, "per_token_grad_scale");
    assignTrainingField(params.max_seq_len, trainConfig, "max_seq_len");
    // min_seq_valid_tokens: derived as max_seq_len / 4 (see deriveComputedHyperparameters)
    // min_seq_len_for_flash: derived as max_seq_len / 4 (see deriveComputedHyperparameters)
    assignTrainingField(params.warmup_fraction, trainConfig, "warmup_fraction");
    if (auto it = trainConfig.find("cosine_decay"); it != trainConfig.end() && it->is_object()) {
        params.cosine_decay_enabled = it->value("enabled", false);
        params.cosine_warm_restarts = it->value("warm_restarts", false);
        // cosine_decay_min_lr: derived as learning_rate * 0.1 (see deriveComputedHyperparameters)
    } else {
        params.cosine_decay_enabled = false;
        params.cosine_warm_restarts = false;
    }
    assignTrainingField(params.log_interval, trainConfig, "log_interval");
    assignTrainingField(params.atom_stats_interval, trainConfig, "atom_stats_interval");
    assignTrainingField(params.atom_stats_max_seqs, trainConfig, "atom_stats_max_seqs");
    assignTrainingField(params.validation_interval, trainConfig, "validation_interval");
    assignTrainingField(params.checkpoint_interval, trainConfig, "checkpoint_interval");
    assignTrainingField(params.use_gpu, trainConfig, "use_gpu");
    assignTrainingField(params.use_flash_attention, trainConfig, "use_flash_attention");
    // min_seq_len_for_flash: derived from max_seq_len (see deriveComputedHyperparameters)

    if (auto it = trainConfig.find("soft_restart"); it != trainConfig.end()) {
        const auto& soft = *it;
        if (soft.is_boolean()) {
            params.soft_restart_enabled = soft.get<bool>();
        } else if (soft.is_object()) {
            params.soft_restart_enabled = soft.value("enabled", params.soft_restart_enabled);
            params.soft_restart_loss_increase_threshold =
                soft.value("loss_increase_threshold", params.soft_restart_loss_increase_threshold);
            params.soft_restart_max_step_window = soft.value("max_step_window", params.soft_restart_max_step_window);
            params.soft_restart_cooldown_steps = soft.value("cooldown_steps", params.soft_restart_cooldown_steps);
        }
    }

    if (auto it = trainConfig.find("auto_stop"); it != trainConfig.end()) {
        const auto& autoStop = *it;
        if (autoStop.is_boolean()) {
            params.auto_stop_enabled = autoStop.get<bool>();
        } else if (autoStop.is_object()) {
            params.auto_stop_enabled = autoStop.value("enabled", params.auto_stop_enabled);
            params.auto_stop_plateau_patience = autoStop.value("plateau_patience", params.auto_stop_plateau_patience);
            params.auto_stop_plateau_min_delta = autoStop.value("plateau_min_delta", params.auto_stop_plateau_min_delta);
            params.auto_stop_high_loss_threshold = autoStop.value("high_loss_threshold", params.auto_stop_high_loss_threshold);
            params.auto_stop_high_loss_patience = autoStop.value("high_loss_patience", params.auto_stop_high_loss_patience);
        }
    }

    if (auto it = trainConfig.find("atom_stats"); it != trainConfig.end() && it->is_object()) {
        const auto& atom_stats = *it;
        params.atom_stats_interval = atom_stats.value("interval", params.atom_stats_interval);
        params.atom_stats_max_seqs = atom_stats.value("max_seqs", params.atom_stats_max_seqs);
    }

    if (auto it = trainConfig.find("single_batch"); it != trainConfig.end()) {
        const auto& single = *it;
        if (single.is_boolean()) {
            params.single_batch_overfit_enabled = single.get<bool>();
        } else if (single.is_object()) {
            params.single_batch_overfit_enabled = single.value("enabled", params.single_batch_overfit_enabled);
            params.single_batch_overfit_max_steps = single.value("max_steps", params.single_batch_overfit_max_steps);
        }
    }

    if (auto it = trainConfig.find("shuffle"); it != trainConfig.end()) {
        const auto& shuffle = *it;
        if (shuffle.is_boolean()) {
            params.shuffle_train_enabled = shuffle.get<bool>();
        } else if (shuffle.is_object()) {
            params.shuffle_train_enabled = shuffle.value("enabled", params.shuffle_train_enabled);
            params.shuffle_train_epochs = shuffle.value("epochs", params.shuffle_train_epochs);
        }
        if (params.shuffle_train_epochs < 0) {
            params.shuffle_train_epochs = 0;
        }
    }

    if (auto it = trainConfig.find("guess_aux"); it != trainConfig.end()) {
        const auto& guess = *it;
        if (guess.is_boolean()) {
            params.guess_aux_enabled = guess.get<bool>();
        } else if (guess.is_object()) {
            params.guess_aux_enabled = guess.value("enabled", params.guess_aux_enabled);
            params.guess_aux_lambda = guess.value("lambda", params.guess_aux_lambda);
            params.guess_aux_min_confidence = guess.value("min_confidence", params.guess_aux_min_confidence);
        }
    }

    // Load telemetry control configuration (includes plateau noise)
    if (auto it = trainConfig.find("telemetry_control"); it != trainConfig.end()) {
        const auto& tc = *it;
        if (tc.is_boolean()) {
            params.telemetry_control_enabled = tc.get<bool>();
        } else if (tc.is_object()) {
            params.telemetry_control_enabled = tc.value("enabled", params.telemetry_control_enabled);

            if (auto spike_it = tc.find("spike_thresholds"); spike_it != tc.end() && spike_it->is_object()) {
                const auto& spike = *spike_it;
                params.telemetry_spike_mild_threshold = spike.value("mild", params.telemetry_spike_mild_threshold);
                params.telemetry_spike_moderate_threshold = spike.value("moderate", params.telemetry_spike_moderate_threshold);
                params.telemetry_spike_severe_threshold = spike.value("severe", params.telemetry_spike_severe_threshold);
            }

            if (auto resp_it = tc.find("response"); resp_it != tc.end() && resp_it->is_object()) {
                const auto& response = *resp_it;
                params.telemetry_moderate_grad_scale = response.value("moderate_grad_scale", params.telemetry_moderate_grad_scale);
                params.telemetry_moderate_cooldown_extension = response.value("moderate_cooldown_extension", params.telemetry_moderate_cooldown_extension);
            }

            if (auto acc_it = tc.find("accumulation_guard"); acc_it != tc.end() && acc_it->is_object()) {
                const auto& acc = *acc_it;
                params.telemetry_min_grad_for_nonzero_loss = acc.value("min_grad_for_nonzero_loss", params.telemetry_min_grad_for_nonzero_loss);
                params.telemetry_loss_threshold_for_grad_check = acc.value("loss_threshold", params.telemetry_loss_threshold_for_grad_check);
                params.telemetry_max_consecutive_zero_grad_steps = acc.value("max_consecutive_zero_grad_steps", params.telemetry_max_consecutive_zero_grad_steps);
            }

            if (auto regime_it = tc.find("regime_change"); regime_it != tc.end() && regime_it->is_object()) {
                const auto& regime = *regime_it;
                params.telemetry_seq_len_regime_change_threshold = regime.value("seq_len_threshold", params.telemetry_seq_len_regime_change_threshold);
                params.telemetry_regime_change_suppression_steps = regime.value("suppression_steps", params.telemetry_regime_change_suppression_steps);
            }

            if (auto vol_it = tc.find("volatility_damping"); vol_it != tc.end() && vol_it->is_object()) {
                const auto& vol = *vol_it;
                params.telemetry_volatility_damping_threshold = vol.value("threshold", params.telemetry_volatility_damping_threshold);
                params.telemetry_max_volatility_damping = vol.value("max_damping", params.telemetry_max_volatility_damping);
            }

            if (auto decay_it = tc.find("gradient_decay"); decay_it != tc.end() && decay_it->is_object()) {
                const auto& decay = *decay_it;
                params.telemetry_gradient_decay_threshold = decay.value("threshold", params.telemetry_gradient_decay_threshold);
                params.telemetry_max_decay_boost = decay.value("max_boost", params.telemetry_max_decay_boost);
            }

            if (auto boost_it = tc.find("progress_boost"); boost_it != tc.end() && boost_it->is_object()) {
                const auto& boost = *boost_it;
                params.telemetry_progress_boost_threshold = boost.value("threshold", params.telemetry_progress_boost_threshold);
                params.telemetry_max_progress_boost = boost.value("max_boost", params.telemetry_max_progress_boost);
            }

            if (auto out_it = tc.find("outlier"); out_it != tc.end() && out_it->is_object()) {
                const auto& outlier = *out_it;
                params.telemetry_outlier_frequency_trigger = outlier.value("frequency_trigger", params.telemetry_outlier_frequency_trigger);
                params.telemetry_outlier_persistence_trigger = outlier.value("persistence_trigger", params.telemetry_outlier_persistence_trigger);
            }

            if (auto drift_it = tc.find("drift"); drift_it != tc.end() && drift_it->is_object()) {
                const auto& drift = *drift_it;
                params.telemetry_anchor_drift_sigma_multiplier = drift.value("anchor_sigma_multiplier", params.telemetry_anchor_drift_sigma_multiplier);
            }

            if (auto sr_it = tc.find("soft_restart"); sr_it != tc.end() && sr_it->is_object()) {
                const auto& sr = *sr_it;
                params.telemetry_soft_restart_cooldown_steps = sr.value("cooldown_steps", params.telemetry_soft_restart_cooldown_steps);
            }

            if (auto base_it = tc.find("baseline"); base_it != tc.end() && base_it->is_object()) {
                const auto& base = *base_it;
                // telemetry_warmup_steps: derived as warmup_steps (see deriveComputedHyperparameters)
                params.telemetry_baseline_stabilization_steps = base.value("stabilization_steps", params.telemetry_baseline_stabilization_steps);
            }

            if (auto log_it = tc.find("logging"); log_it != tc.end() && log_it->is_object()) {
                const auto& logging = *log_it;
                params.telemetry_verbose_logging = logging.value("verbose", params.telemetry_verbose_logging);
                params.telemetry_fail_loud_on_accumulation_bug = logging.value("fail_loud_on_accumulation_bug", params.telemetry_fail_loud_on_accumulation_bug);
            }
            
            // Plateau noise sub-config
            if (auto pn_it = tc.find("plateau_noise"); pn_it != tc.end() && pn_it->is_object()) {
                const auto& pn = *pn_it;
                params.telemetry_plateau_noise_enabled = pn.value("enabled", params.telemetry_plateau_noise_enabled);
                params.telemetry_plateau_noise_patience = pn.value("patience", params.telemetry_plateau_noise_patience);
                params.telemetry_plateau_noise_variance_threshold = pn.value("variance_threshold", params.telemetry_plateau_noise_variance_threshold);
                params.telemetry_plateau_noise_std = pn.value("noise_std", params.telemetry_plateau_noise_std);
                params.telemetry_plateau_noise_proportional = pn.value("proportional", params.telemetry_plateau_noise_proportional);
                params.telemetry_plateau_noise_cooldown = pn.value("cooldown", params.telemetry_plateau_noise_cooldown);
                params.telemetry_plateau_noise_max_per_epoch = pn.value("max_per_epoch", params.telemetry_plateau_noise_max_per_epoch);
            }
        }
    }
    // Legacy: simple bool telemetry_control_enabled (backwards compat removed per Rule 20)
    // If you had "telemetry_control_enabled": true, migrate to "telemetry_control": { "enabled": true }

    // Load unified tape logging configuration
    if (auto it = trainConfig.find("logging"); it != trainConfig.end() && it->is_object()) {
        const auto& logCfg = *it;
        params.tape_logging.default_level = logCfg.value("default_level", params.tape_logging.default_level);
        params.tape_logging.equation_csv_enabled = logCfg.value("equation_csv_enabled", params.tape_logging.equation_csv_enabled);
        params.tape_logging.stderr_enabled = logCfg.value("stderr_enabled", params.tape_logging.stderr_enabled);
        params.tape_logging.initial_capacity = logCfg.value("initial_capacity", params.tape_logging.initial_capacity);
        if (logCfg.contains("group_overrides") && logCfg["group_overrides"].is_object()) {
            for (auto& [key, val] : logCfg["group_overrides"].items()) {
                if (val.is_string()) {
                    params.tape_logging.group_overrides[key] = val.get<std::string>();
                }
            }
        }
    }

    // Load Log Recorder configuration
    if (auto it = trainConfig.find("log_recorder"); it != trainConfig.end()) {
        const auto& logRec = *it;
        if (logRec.is_object()) {
            params.log_recorder.enabled = logRec.value("enabled", params.log_recorder.enabled);
            params.log_recorder.default_level = logRec.value("default_level", params.log_recorder.default_level);
            
            if (logRec.contains("modules") && logRec["modules"].is_object()) {
                for (auto& [key, val] : logRec["modules"].items()) {
                    if (val.is_string()) {
                        params.log_recorder.modules[key] = val.get<std::string>();
                    }
                }
            }
            
            // Parse layer logging enables
            if (logRec.contains("layers") && logRec["layers"].is_object()) {
                const auto& layers = logRec["layers"];
                params.log_recorder.layers.embedding = layers.value("embedding", params.log_recorder.layers.embedding);
                params.log_recorder.layers.rms_norm = layers.value("rms_norm", params.log_recorder.layers.rms_norm);
                params.log_recorder.layers.attention = layers.value("attention", params.log_recorder.layers.attention);
                params.log_recorder.layers.feed_forward = layers.value("feed_forward", params.log_recorder.layers.feed_forward);
                params.log_recorder.layers.residual = layers.value("residual", params.log_recorder.layers.residual);
                params.log_recorder.layers.encoding = layers.value("encoding", params.log_recorder.layers.encoding);
                params.log_recorder.layers.serialization = layers.value("serialization", params.log_recorder.layers.serialization);
                params.log_recorder.layers.execution_block = layers.value("execution_block", params.log_recorder.layers.execution_block);
            }
        }
    }
    
    // Load loss options
    if (auto it = trainConfig.find("loss"); it != trainConfig.end() && it->is_object()) {
        const auto& loss_cfg = *it;
        
        if (auto ls_it = loss_cfg.find("label_smoothing"); ls_it != loss_cfg.end()) {
            const auto& ls = *ls_it;
            if (ls.is_boolean()) {
                params.loss_label_smoothing_enabled = ls.get<bool>();
            } else if (ls.is_object()) {
                params.loss_label_smoothing_enabled = ls.value("enabled", params.loss_label_smoothing_enabled);
                params.loss_label_smoothing_epsilon = ls.value("epsilon", params.loss_label_smoothing_epsilon);
            }
        }
        
        if (auto fc_it = loss_cfg.find("focal"); fc_it != loss_cfg.end()) {
            const auto& fc = *fc_it;
            if (fc.is_boolean()) {
                params.loss_focal_enabled = fc.get<bool>();
            } else if (fc.is_object()) {
                params.loss_focal_enabled = fc.value("enabled", params.loss_focal_enabled);
                params.loss_focal_gamma = fc.value("gamma", params.loss_focal_gamma);
                params.loss_focal_alpha = fc.value("alpha", params.loss_focal_alpha);
            }
        }
        
        // Issue #44 FIX: Entropy regularization to prevent mode collapse
        if (auto er_it = loss_cfg.find("entropy_reg"); er_it != loss_cfg.end()) {
            const auto& er = *er_it;
            if (er.is_boolean()) {
                params.loss_entropy_reg_enabled = er.get<bool>();
            } else if (er.is_object()) {
                params.loss_entropy_reg_enabled = er.value("enabled", params.loss_entropy_reg_enabled);
                params.loss_entropy_reg_lambda = er.value("lambda", params.loss_entropy_reg_lambda);
            }
        }

        // Class-balanced loss: reweights per-token loss by 1/freq^β
        if (auto cb_it = loss_cfg.find("class_balanced"); cb_it != loss_cfg.end()) {
            const auto& cb = *cb_it;
            if (cb.is_boolean()) {
                params.loss_class_balanced_enabled = cb.get<bool>();
            } else if (cb.is_object()) {
                params.loss_class_balanced_enabled = cb.value("enabled", params.loss_class_balanced_enabled);
                params.loss_class_balanced_beta = cb.value("beta", params.loss_class_balanced_beta);
            }
        }

        
        if (auto pref_it = loss_cfg.find("preference"); pref_it != loss_cfg.end()) {
            const auto& pref = *pref_it;
            if (pref.is_boolean()) {
                params.loss_preference_enabled = pref.get<bool>();
            } else if (pref.is_object()) {
                params.loss_preference_enabled = pref.value("enabled", params.loss_preference_enabled);
                params.loss_preference_beta = pref.value("beta", params.loss_preference_beta);
            }
        }
        
        if (auto dist_it = loss_cfg.find("distillation"); dist_it != loss_cfg.end()) {
            const auto& dist = *dist_it;
            if (dist.is_boolean()) {
                params.loss_distillation_enabled = dist.get<bool>();
            } else if (dist.is_object()) {
                params.loss_distillation_enabled = dist.value("enabled", params.loss_distillation_enabled);
                params.loss_distillation_temperature = dist.value("temperature", params.loss_distillation_temperature);
                params.loss_distillation_lambda = dist.value("lambda", params.loss_distillation_lambda);
            }
        }
        
        if (auto mask_it = loss_cfg.find("masking"); mask_it != loss_cfg.end()) {
            const auto& mask = *mask_it;
            if (mask.is_boolean()) {
                params.loss_masking_enabled = mask.get<bool>();
            } else if (mask.is_object()) {
                params.loss_masking_enabled = mask.value("enabled", params.loss_masking_enabled);
                if (mask.contains("tag") && mask["tag"].is_string()) {
                    params.loss_masking_tag = mask["tag"].get<std::string>();
                }
            }
        }
    }
    
    // LM Head centering configuration (Issue #37 / #40)
    // When enabled, centers hidden states before LM head projection.
    // Set to false for standard PyTorch-style implementation.
    params.lm_head_centering_enabled = false;  // Default to disabled (standard implementation)
    params.lm_head_center_hidden_states = false;
    params.lm_head_freeze_final_rms_gamma = false;  // Default: γ_final is trainable
    params.center_logits = false;  // Default to disabled (standard implementation)
    params.center_encoder_residuals = false;  // Default: disabled. Enable to prevent ρ buildup across layers (mode collapse fix).
                                               // Gradient cost: (1-1/n_tokens)^24 ≈ 0.996 for n≈6000 — negligible.
    params.project_out_pc1 = false;  // Default: disabled (Issue #149)
    params.pc1_power_iters = 5;
    if (auto it = trainConfig.find("lm_head_centering"); it != trainConfig.end() && it->is_object()) {
        const auto& lmc = *it;
        params.lm_head_centering_enabled = lmc.value("enabled", false);
        params.lm_head_center_hidden_states = lmc.value("center_hidden_states", false);
        params.lm_head_freeze_final_rms_gamma = lmc.value("freeze_final_rms_gamma", false);
        params.center_logits = lmc.value("center_logits", false);
        params.center_encoder_residuals = lmc.value("center_encoder_residuals", false);
        params.project_out_pc1 = lmc.value("project_out_pc1", false);
        params.pc1_power_iters = lmc.value("pc1_power_iters", 5);
    }
    
    // Issue #109: LayerScale (learnable residual scaling from CaiT paper)
    // Reduces correlation buildup between layers by gating sublayer outputs
    params.use_layer_scale = false;   // Default: disabled (standard residual connections)
    params.layer_scale_init = 0.1f;   // CaiT paper recommends 0.1 for deeper networks
    if (auto it = trainConfig.find("layer_scale"); it != trainConfig.end() && it->is_object()) {
        const auto& ls = *it;
        params.use_layer_scale = ls.value("enabled", false);
        params.layer_scale_init = ls.value("init_value", 0.1f);
    }
    
    // QK-norm: Per-head RMSNorm applied to Q and K after QKV projection, before RoPE.
    // Bounds attention logit magnitudes, prevents entropy collapse in deeper models.
    params.qk_norm_enabled = false;   // Default: disabled (standard unscaled Q/K)
    if (auto it = trainConfig.find("qk_norm"); it != trainConfig.end() && it->is_object()) {
        const auto& qkn = *it;
        params.qk_norm_enabled = qkn.value("enabled", false);
    }
    
    // Hardcoded Hidden States Diagnostic (Issue #42)
    params.hardcoded_hidden_pattern = 0;  // 0 = DISABLED
    params.hardcoded_log_every_n_batches = 1;
    if (auto it = trainConfig.find("hardcoded_hidden_states"); it != trainConfig.end() && it->is_object()) {
        const auto& hcs = *it;
        if (hcs.value("enabled", false)) {
            std::string pattern_str = hcs.value("pattern", "random_centered");
            if (pattern_str == "random_centered") {
                params.hardcoded_hidden_pattern = 1;  // RANDOM_CENTERED
            } else if (pattern_str == "orthogonal_w277") {
                params.hardcoded_hidden_pattern = 2;  // ORTHOGONAL_W277
            } else if (pattern_str == "aligned_w277") {
                params.hardcoded_hidden_pattern = 3;  // ALIGNED_W277
            } else if (pattern_str == "constant_uniform") {
                params.hardcoded_hidden_pattern = 4;  // CONSTANT_UNIFORM
            } else if (pattern_str == "zero_mean_sine") {
                params.hardcoded_hidden_pattern = 5;  // ZERO_MEAN_SINE
            }
            params.hardcoded_log_every_n_batches = hcs.value("log_every_n_batches", 1);
        }
    }
    
    // Load embedding freeze guard
    if (auto it = trainConfig.find("embedding_freeze"); it != trainConfig.end() && it->is_object()) {
        const auto& ef = *it;
        params.embedding_freeze_enabled = ef.value("enabled", params.embedding_freeze_enabled);
        params.embedding_freeze_after_step = ef.value("freeze_after_step", params.embedding_freeze_after_step);
    }

    // Load optimizer selector.
    // JSON layout (all fields optional; struct defaults from HyperParameters apply):
    //   "optimizer": {
    //     "kind": "adamw" | "radamw",
    //     "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8
    //   }
    if (auto it = trainConfig.find("optimizer"); it != trainConfig.end() && it->is_object()) {
        const auto& opt = *it;
        params.optimizer_kind   = opt.value("kind",    params.optimizer_kind);
        params.optimizer_beta1  = opt.value("beta1",   params.optimizer_beta1);
        params.optimizer_beta2  = opt.value("beta2",   params.optimizer_beta2);
        params.optimizer_epsilon = opt.value("epsilon", params.optimizer_epsilon);
        // Rule 20: validate kind explicitly — fail loud on typo.
        if (params.optimizer_kind != "adamw" && params.optimizer_kind != "radamw") {
            throw std::runtime_error(
                "[ai_config] training.config.optimizer.kind must be \"adamw\" or \"radamw\", got \""
                + params.optimizer_kind + "\"");
        }
    }

    // Load stability overrides - ALWAYS parse values even if disabled
    // (Phase1_Startup copies them unconditionally, so they must be initialized)
    params.stability_overrides_enabled = trainConfig.value("stability_overrides_enabled", params.stability_overrides_enabled);
    if (auto it = trainConfig.find("stability_overrides"); it != trainConfig.end() && it->is_object()) {
        const auto& stab = *it;
        params.stability_override_batch_size = stab.value("batch_size", params.stability_override_batch_size);
        params.stability_override_max_seq_len = stab.value("max_seq_len", params.stability_override_max_seq_len);
        params.stability_override_clip_per_token = stab.value("clip_per_token", params.stability_override_clip_per_token);
        params.stability_override_lr_min = stab.value("lr_min", params.stability_override_lr_min);
    }
    
    // Load scratch blocks configuration
    if (auto it = trainConfig.find("scratch_blocks"); it != trainConfig.end()) {
        const auto& scratch = *it;
        if (scratch.is_boolean()) {
            params.scratch_blocks_enabled = scratch.get<bool>();
        } else if (scratch.is_object()) {
            params.scratch_blocks_enabled = scratch.value("enabled", params.scratch_blocks_enabled);
            // scratch_max_tokens_per_block: derived as max_seq_len (see deriveComputedHyperparameters)
            params.scratch_num_blocks = scratch.value("num_blocks", params.scratch_num_blocks);
            params.scratch_write_combined = scratch.value("use_write_combined", params.scratch_write_combined);
        }
    }
    
    // Load scratch_block_reasoning configuration (model config)
    if (auto it = trainConfig.find("scratch_block_reasoning"); it != trainConfig.end() && it->is_object()) {
        const auto& sbr = *it;
        params.scratch_block_reasoning_enabled = sbr.value("enabled", params.scratch_block_reasoning_enabled);
        params.scratch_block_reasoning_atom_embedding_dim = sbr.value("atom_embedding_dim", params.scratch_block_reasoning_atom_embedding_dim);
        params.scratch_block_reasoning_max_atoms = sbr.value("max_atoms", params.scratch_block_reasoning_max_atoms);
        params.scratch_block_reasoning_atom_scale = sbr.value("atom_scale", params.scratch_block_reasoning_atom_scale);
    }

    if (auto it = trainConfig.find("execution_block"); it != trainConfig.end() && it->is_object()) {
        const auto& eb = *it;
        assignTrainingField(params.execution_block_enabled, eb, "enabled");
        assignTrainingField(params.scratch_block_execution_first_type_only, eb, "execution_first_type_only");
        assignTrainingField(params.execution_block_layer, eb, "layer");
        assignTrainingField(params.execution_block_num_ops, eb, "num_ops");
        assignTrainingField(params.execution_block_num_slots, eb, "num_slots");
        assignTrainingField(params.execution_block_num_steps, eb, "num_steps");
        // execution_block_d_key: derived as d_model / num_heads (see deriveComputedHyperparameters)
        assignTrainingField(params.execution_block_d_type, eb, "d_type");
        // execution_block_cross_attn_head_dim: derived as d_model / num_heads (see deriveComputedHyperparameters)
        assignTrainingField(params.execution_block_cross_attn_topk, eb, "cross_attn_topk");
        assignTrainingField(params.execution_block_usage_decay, eb, "usage_decay");
        assignTrainingField(params.execution_block_diversity_kappa, eb, "diversity_kappa");
        assignTrainingField(params.execution_block_temp_start, eb, "temp_start");
        assignTrainingField(params.execution_block_temp_end, eb, "temp_end");
        assignTrainingField(params.execution_block_temp_schedule, eb, "temp_schedule");
        assignTrainingField(params.execution_block_entropy_weight, eb, "entropy_weight");
        assignTrainingField(params.execution_step_x_multiplier, eb, "step_x_multiplier");
        assignTrainingField(params.execution_step_y_multiplier, eb, "step_y_multiplier");
        assignTrainingField(params.execution_step_y_overrides_x, eb, "step_y_overrides_x");
        assignTrainingField(params.execution_entropy_aux_weight, eb, "entropy_aux_weight");
        assignTrainingField(params.execution_value_match_epsilon, eb, "value_match_epsilon");
        assignTrainingField(params.execution_final_slot_consistency_weight, eb, "final_slot_consistency_weight");
        assignTrainingField(params.execution_block_transition_hard_threshold, eb, "transition_hard_threshold");
        // execution_block_gate_warmup_steps: derived as warmup_steps (see deriveComputedHyperparameters)
        assignTrainingField(params.execution_block_causal_w1_transition, eb, "causal_w1_transition");
        assignTrainingField(params.execution_div_invalid_penalty_weight, eb, "div_invalid_penalty_weight");
        assignTrainingField(params.execution_div_magnitude_penalty_weight, eb, "div_magnitude_penalty_weight");
        assignTrainingField(params.execution_arg_reinforce_weight, eb, "arg_reinforce_weight");
        assignTrainingField(params.execution_arg_reinforce_baseline_decay, eb, "arg_reinforce_baseline_decay");
        assignTrainingField(params.structured_ce_enabled, eb, "structured_ce_enabled");
        assignTrainingField(params.structured_ce_weight, eb, "structured_ce_weight");

        // Decode-time slot selector (nested under execution_block)
        if (auto sit = eb.find("selector"); sit != eb.end() && sit->is_object()) {
            const auto& sel = *sit;
            assignTrainingField(params.selector_enabled, sel, "enabled");
            assignTrainingField(params.selector_d_selector, sel, "d_selector");
            assignTrainingField(params.selector_selection_margin, sel, "selection_margin");
            assignTrainingField(params.selector_supervision_weight, sel, "supervision_weight");
        }
    }
    
    // Load activation quantization configuration
    if (auto it = trainConfig.find("activation_quantization"); it != trainConfig.end() && it->is_object()) {
        const auto& quant = *it;
        params.activation_quantization_enabled = quant.value("enabled", params.activation_quantization_enabled);
        params.activation_quantization_apply_to_embeddings = quant.value("apply_to_embeddings", params.activation_quantization_apply_to_embeddings);
        params.activation_quantization_apply_to_encoder_outputs = quant.value("apply_to_encoder_outputs", params.activation_quantization_apply_to_encoder_outputs);
        params.activation_quantization_apply_to_layer_caches = quant.value("apply_to_layer_caches", params.activation_quantization_apply_to_layer_caches);
        params.activation_quantization_apply_to_qkv_cache = quant.value("apply_to_qkv_cache", params.activation_quantization_apply_to_qkv_cache);
        params.activation_quantization_apply_to_logits = quant.value("apply_to_logits", params.activation_quantization_apply_to_logits);
        params.activation_quantization_scale = quant.value("scale", params.activation_quantization_scale);
        params.activation_quantization_clip_min = quant.value("clip_min", params.activation_quantization_clip_min);
        params.activation_quantization_clip_max = quant.value("clip_max", params.activation_quantization_clip_max);
        params.activation_quantization_zero_point = quant.value("zero_point", params.activation_quantization_zero_point);
        params.activation_quantization_symmetric = quant.value("symmetric", params.activation_quantization_symmetric);
    }
    
    // Load CUDA execution mode configuration
    if (auto it = trainConfig.find("cuda_execution"); it != trainConfig.end() && it->is_object()) {
        const auto& cuda_exec = *it;
        params.single_stream_mode = cuda_exec.value("single_stream_mode", params.single_stream_mode);
        params.disable_async_frees = cuda_exec.value("disable_async_frees", params.disable_async_frees);
        params.synchronize_after_kernels = cuda_exec.value("synchronize_after_kernels", params.synchronize_after_kernels);
    }
    
    // Load multi_token_prediction (MTP) configuration
    if (auto it = trainConfig.find("multi_token_prediction"); it != trainConfig.end() && it->is_object()) {
        const auto& mtp = *it;
        params.mtp_enabled = mtp.value("enabled", params.mtp_enabled);
        params.mtp_k = mtp.value("k", params.mtp_k);
        params.mtp_alpha = mtp.value("alpha", params.mtp_alpha);
        // mtp_alpha_warmup_steps: derived as warmup_steps (see deriveComputedHyperparameters)
        params.mtp_log_ratio_monitor = mtp.value("log_ratio_monitor", params.mtp_log_ratio_monitor);
    }

    // Load prediction comparison configuration
    if (auto it = trainConfig.find("prediction_comparison"); it != trainConfig.end() && it->is_object()) {
        const auto& pred_cmp = *it;
        params.prediction_comparison_enabled = pred_cmp.value("enabled", params.prediction_comparison_enabled);
        params.prediction_comparison_interval = pred_cmp.value("interval", params.prediction_comparison_interval);
        params.prediction_comparison_top_k = pred_cmp.value("top_k", params.prediction_comparison_top_k);
        params.prediction_comparison_max_positions = pred_cmp.value("max_positions", params.prediction_comparison_max_positions);
        params.prediction_comparison_log_path = pred_cmp.value("log_path", params.prediction_comparison_log_path);
    }

    // Logit update trace configuration
    if (auto it = trainConfig.find("logit_update_trace"); it != trainConfig.end()) {
        const auto& trace = *it;
        if (trace.is_boolean()) {
            params.logit_update_trace_enabled = trace.get<bool>();
        } else if (trace.is_object()) {
            params.logit_update_trace_enabled = trace.value("enabled", params.logit_update_trace_enabled);
            params.logit_update_trace_interval = trace.value("interval", params.logit_update_trace_interval);
        }
    }
    
    // Load attention diagnostics configuration
    // Use this to diagnose training plateau (saturated attention, gradient collapse)
    if (auto it = trainConfig.find("attention_diagnostics"); it != trainConfig.end() && it->is_object()) {
        const auto& attn_diag = *it;
        params.attention_diag_enabled = attn_diag.value("enabled", params.attention_diag_enabled);
        params.attention_diag_layer = attn_diag.value("layer", params.attention_diag_layer);
        params.attention_diag_head = attn_diag.value("head", params.attention_diag_head);
    }
}

// ── deriveComputedHyperparameters ──
inline void deriveComputedHyperparameters(TrainingHyperparameters& params, const nlohmann::json& trainConfig) {
    // ── Sequence length derivations ──
    if (params.max_seq_len <= 0)
        throw std::runtime_error("deriveComputedHyperparameters: max_seq_len must be > 0, got " + std::to_string(params.max_seq_len));
    params.min_seq_valid_tokens = params.max_seq_len / 4;
    params.min_seq_len_for_flash = params.max_seq_len / 4;
    params.scratch_max_tokens_per_block = static_cast<size_t>(params.max_seq_len);

    // ── LR floor ──
    if (params.cosine_decay_enabled) {
        if (params.learning_rate <= 0.0f)
            throw std::runtime_error("deriveComputedHyperparameters: learning_rate must be > 0 when cosine_decay is enabled, got " + std::to_string(params.learning_rate));
        params.cosine_decay_min_lr = params.learning_rate * 0.1f;
    }

    // ── Head dimension propagation ──
    if (trainConfig.contains("d_model") && trainConfig.contains("num_heads")) {
        int d_model = trainConfig["d_model"].get<int>();
        int num_heads = trainConfig["num_heads"].get<int>();
        if (d_model > 0 && num_heads > 0) {
            int head_dim = d_model / num_heads;
            params.execution_block_d_key = head_dim;
            params.execution_block_cross_attn_head_dim = head_dim;
            params.scratch_block_reasoning_atom_embedding_dim = d_model / 8;
        }
    }

    // ── Warmup fraction validation (warmup_steps derived in Phase2 from warmup_fraction * total_steps) ──
    if (params.warmup_fraction <= 0.0f || params.warmup_fraction >= 1.0f)
        throw std::runtime_error("deriveComputedHyperparameters: warmup_fraction must be in (0, 1), got " + std::to_string(params.warmup_fraction));
    // warmup_steps, mtp_alpha_warmup_steps, telemetry_warmup_steps,
    // execution_block_gate_warmup_steps are all
    // derived in Phase2 via deriveWarmupSteps() once estimated_total_steps is known.

    // ── Stability overrides derived from base values ──
    params.stability_override_batch_size = params.batch_size * 2 / 3;
    if (params.stability_override_batch_size < 1) params.stability_override_batch_size = 1;
    params.stability_override_max_seq_len = params.max_seq_len;
    params.stability_override_clip_per_token = 0.02f;
    params.stability_override_lr_min = params.learning_rate * 0.83f;
}

// ── deriveWarmupSteps (inner, namespace detail) ──
inline void deriveWarmupSteps(TrainingHyperparameters& params, int estimated_total_steps) {
    if (estimated_total_steps <= 0)
        throw std::runtime_error("deriveWarmupSteps: estimated_total_steps must be > 0, got " + std::to_string(estimated_total_steps));
    if (params.warmup_fraction <= 0.0f || params.warmup_fraction >= 1.0f)
        throw std::runtime_error("deriveWarmupSteps: warmup_fraction must be in (0, 1), got " + std::to_string(params.warmup_fraction));

    params.warmup_steps = std::max(1, static_cast<int>(params.warmup_fraction * estimated_total_steps));
    params.mtp_alpha_warmup_steps = params.warmup_steps;
    params.telemetry_warmup_steps = params.warmup_steps;
    params.execution_block_gate_warmup_steps = params.warmup_steps;
}

// ── populateTrainingHyperparametersFromConfig ──
inline bool populateTrainingHyperparametersFromConfig(const nlohmann::json& config, TrainingHyperparameters& params) {
    if (config.contains("training") && config["training"].contains("config")) {
        const auto& training = config["training"];
        // Parse training-level selectors (sibling to "config")
        assignTrainingField(params.current_model_training, training, "current_model_training");
        assignTrainingField(params.current_curriculum, training, "current_curriculum");

        const auto& trainConfig = training["config"];
        // Phase 1: Set sensible defaults for all non-base params
        setDefaultHyperparameters(params);
        // Phase 2: Validate base+enable fields exist
        validateTrainingConfigJson(trainConfig);
        // Phase 3: Parse JSON (overwrites defaults only for keys present)
        applyTrainingConfigObject(trainConfig, params);
        // Phase 4: Compute formula-derived values (always overwrites)
        deriveComputedHyperparameters(params, trainConfig);
        return true;
    }
    return false;
}

// ── loadTrainingHyperparameters ──
// Reads ai_config.json directly (no AiConfigSnapshot dependency, since the
// snapshot loader lives in ai_config_paths.hpp which now depends on this
// header — the cycle is broken by inlining a minimal file read here).
inline bool loadTrainingHyperparameters(TrainingHyperparameters& params, const std::string& configPath = "ai_config.json") {
    std::ifstream in(configPath);
    if (!in.is_open()) {
        return false;
    }
    nlohmann::json doc;
    try {
        in >> doc;
    } catch (const std::exception&) {
        return false;
    }
    return populateTrainingHyperparametersFromConfig(doc, params);
}

namespace detail {
// (helpers nested under detail in the original file are now top-level in
//  Config; the empty detail namespace is preserved as a forward-compat
//  marker — REMOVE if not needed by Phase B.)
} // namespace detail

} // namespace Config
} // namespace GRIM

// (Validation/derivation/policy helpers below also reference
// GRIM::Config::TrainingHyperparameters and so must stay inside the gate.)
namespace GRIM {
namespace HyperParameters {

//======================================================//
// (A) Validation primitive.
// Throws std::runtime_error on any invalid required field.
// Pure: does not mutate, does not log.
//======================================================//
inline void validateTrainingHyperparameters(const GRIM::Config::TrainingHyperparameters& params) {
    if (params.batch_size <= 0) {
        throw std::runtime_error("FATAL: batch_size must be > 0, got " + std::to_string(params.batch_size));
    }
    if (params.epochs <= 0) {
        throw std::runtime_error("FATAL: epochs must be > 0, got " + std::to_string(params.epochs));
    }
    if (params.validation_interval <= 0) {
        throw std::runtime_error("FATAL: validation_interval must be > 0, got " + std::to_string(params.validation_interval));
    }
}

//======================================================//
// (B) Pure schedule derivation.
// Computes batches_per_epoch / total_training_steps / safe_last_step
// from the (sequence_count, batch_size, epochs) ratio. No mutation,
// no logging. warmup_steps is NOT computed here (set in Phase2 from
// warmup_fraction * total_training_steps).
// Caller MUST have already passed validateTrainingHyperparameters().
//======================================================//
inline DerivedScheduleInfo computeDerivedSchedule(
    const GRIM::Config::TrainingHyperparameters& params,
    const DerivationContext& context) {
    DerivedScheduleInfo info;
    const int safe_batch_size = params.batch_size;
    const int sequence_count  = std::max(0, context.train_sequence_count);
    info.batches_per_epoch    = std::max(1, (sequence_count + safe_batch_size - 1) / safe_batch_size);
    info.total_training_steps = std::max(1, info.batches_per_epoch * params.epochs);
    info.safe_last_step       = std::max(info.total_training_steps - 1, 1);
    return info;
}

//======================================================//
// (C) Cross-field policy / coercion.
// Mutates `params` to keep cadence-coupled fields consistent with the
// derived schedule, and floors non-negative integer knobs at zero.
// Each adjustment is logged through `log_callback` as
// `[ConfigAdjust] <field> <before> -> <after>` (only when the value
// actually changed). This is the ONLY function in the trio that mutates.
// Caller MUST have already called validate + computeDerivedSchedule.
//======================================================//
inline void applyTrainingHyperparameterPolicy(
    GRIM::Config::TrainingHyperparameters& params,
    const DerivedScheduleInfo& derived,
    const DerivationContext& context,
    LogCallback log_callback = {}) {
    auto log_adjustment = [&](std::string_view label, auto before, auto after) {
        if (!log_callback || before == after) {
            return;
        }
        std::ostringstream oss;
        oss << "[ConfigAdjust] " << label << " " << before << " -> " << after;
        log_callback(oss.str());
    };

    const int cadence_reference =
        std::max(derived.batches_per_epoch, std::max(0, context.validation_interval));

    // warmup_steps is derived in Phase2 from warmup_fraction * estimated_total_steps.
    // At policy time it is 0 — skip clamping here; it will be set correctly in Phase2.

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
}

} // namespace HyperParameters
} // namespace GRIM

#endif // GRIM_HP_HOST_TYPES_AVAILABLE (covers Config block + (A)/(B)/(C) helpers)


//======================================================//
// JSON-based Config Loading (host only)
// These functions parse JSON; require GRIM_HP_HOST_TYPES_AVAILABLE.
//======================================================//

#ifdef GRIM_HP_HOST_TYPES_AVAILABLE

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
    
    // Try to load from config (read JSON directly — AiConfigSnapshot lives in
    // ai_config_paths.hpp which now depends on this header; cycle broken).
    std::ifstream in(configPath);
    if (!in.is_open()) {
        arch.validate();
        return false;
    }
    nlohmann::json doc;
    try {
        in >> doc;
    } catch (const std::exception&) {
        arch.validate();
        return false;
    }
    if (!doc.contains("training") || !doc["training"].contains("config")) {
        arch.validate();
        return false;
    }

    // Populate hyperparameters (only need max_seq_len from it for this fn)
    GRIM::Config::TrainingHyperparameters hp{};
    GRIM::Config::populateTrainingHyperparametersFromConfig(doc, hp);

    // Check for explicit model architecture in training.config
    {
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
                if (use_rope && use_alibi) {
                    arch.positional_encoding = PositionalEncodingType::ALIBI_ROPE;
                } else if (use_rope) {
                    arch.positional_encoding = PositionalEncodingType::ROPE;
                } else if (use_alibi) {
                    arch.positional_encoding = PositionalEncodingType::ALIBI;
                } else {
                    // Rule 20: learned position embeddings were removed; require ALiBi and/or RoPE.
                    throw std::runtime_error(
                        "loadModelArchitecture: positional_encoding requires use_rope and/or use_alibi "
                        "(learned position embeddings have been removed)");
                }
            } else if (pe.is_string()) {
                arch.positional_encoding = parsePositionalEncodingType(pe.get<std::string>());
                if (arch.positional_encoding == PositionalEncodingType::NONE) {
                    throw std::runtime_error(
                        "loadModelArchitecture: positional_encoding=NONE is no longer supported "
                        "(learned position embeddings have been removed). Use ALIBI, ROPE, or ALIBI_ROPE.");
                }
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

#endif // GRIM_SHARED_HYPERPARAMETERS_GPU_HPP
