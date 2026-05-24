#pragma once
// Traditional include guard in addition to #pragma once: nvcc dedupes by
// path-string, not by canonical inode, so the same header pulled in via two
// different ../ chains (e.g. .../GRIM/../Shared/... vs .../Layers/Embedding/../../Shared/...)
// would otherwise be expanded twice and produce "already defined" errors on
// the constexpr globals.
#ifndef GRIM_SHARED_HYPERPARAMETERS_GPU_HPP
#define GRIM_SHARED_HYPERPARAMETERS_GPU_HPP

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../UnigramByte/Unigram.hpp"

namespace GRIM {
namespace Config {
struct TrainingHyperparameters;
}
}

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
//
// RULE: NO other file should define hyperparameters!
// All values come from here which is the preprocesseing layer for ai_config_paths.hpp.
// NO OTHER FILE SHOULD HAVE TRAINING CONFIGURABLE CONSTANTS OR DEFAULTS - this is the single source of truth.
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
// Model Architecture Formula Constants
// Architecture values themselves are authored in ai_config.json and loaded
// into ModelArchitecture. Constants in this section may only be formulas or
// static kernel capabilities, never fallback model defaults.
//======================================================//
constexpr int D_FF_MULTIPLIER = 4;  // d_ff = d_model * multiplier

//======================================================//
// Numerical Stability Constants
// Epsilon values for safe division and log operations
//======================================================//
constexpr float EPSILON_SAFE_DIV = 1e-8f;         // Division safety (AdamW, etc.)
constexpr float EPSILON_LOG_PROB = 1e-7f;         // Log probability clamping (loss)
constexpr float EPSILON_VARIANCE = 1e-4f;         // Variance computation
constexpr float EPSILON_TEMPERATURE = 1e-6f;      // Temperature comparison threshold
constexpr float EPSILON_RMSNORM = 1e-5f;          // RMSNorm numerical stability
constexpr float EPSILON_GRADIENT_CLIP = 1e-6f;    // Gradient clipping minimum threshold
constexpr float LOG_CLAMP_MIN = -100.0f;          // Minimum log value (prevents -inf)
constexpr float NEG_INF_ATTENTION = -1e9f;        // Attention masking value
constexpr float NEG_INF_THRESHOLD = -1e30f;       // Invalid/uninitialized threshold
constexpr float PROBABILITY_FLOOR = 1e-12f;       // Minimum probability (prevents log(0))
constexpr float SOFTMAX_CLIP_THRESHOLD = -20.0f;  // exp(x) ≈ 0 for x < this
constexpr float NORMALIZED_CLAMP = 4.0f;          // Clamp for normalized values

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
constexpr int TELEMETRY_MAX_STREAMS = 58;         // TelemetryLattice metric streams (0-46 dynamic, 47 logit-scale, 48-54 init invariants, 55-57 rho centered/signed-dot)

//======================================================//
// UnigramLM Training Constants
// Parameters for vocabulary training/pruning
//======================================================//
// EM_ITERATIONS removed — convergence-based loop in Unigram.cu (max 50, 0.01% threshold)
constexpr float UNIGRAM_PRUNE_THRESHOLD = 0.0001f;    // Tokens <0.01% usage get pruned
constexpr int UNIGRAM_MIN_VOCAB_SIZE = 8000;          // Don't prune below this
constexpr double UNIGRAM_MIN_COUNT = 1.0;             // Tokens used <1 time get pruned
constexpr size_t UNIGRAM_MAX_SUBWORD_BYTES = 100ULL * 1024 * 1024;  // 100MB limit for subword mining
constexpr size_t UNIGRAM_MAX_SEQUENCE_LENGTH = 4096;  // Static tokenizer workspace floor; not a model fallback

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

// BOS/EOS Token Insertion Control
// These flags are loaded from ai_config.json [tokenizer] section:
//   add_bos: true/false - Controls whether to prepend BOS token to sequences
//   add_eos: true/false - Controls whether to append EOS token to sequences
// Used by Startup/SlidingWindow after LoadTrainingData reads GRMT rows.
// See: ai_config.json [tokenizer] { "add_bos": true, "add_eos": true }

//======================================================//
// Flash Attention Constants
// These MUST match Flash_Attention_Kernal.cu exactly!
//======================================================//
constexpr int FLASH_ATTN_BLOCK_Q = CUDA_WARP_SIZE;    // Block size for Q tiles (= warp size)
constexpr int FLASH_ATTN_BLOCK_KV = CUDA_WARP_SIZE;   // Block size for K/V tiles (= warp size)
constexpr int FLASH_ATTN_NUM_THREADS = CUDA_BLOCK_SIZE_STANDARD;  // Threads per block

// Supported head dimensions for Flash Attention kernel
constexpr int FLASH_ATTN_HEAD_DIM_32 = 32;
constexpr int FLASH_ATTN_HEAD_DIM_64 = 64;  // Primary supported head_dim

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
// Validate GQA configuration
inline bool isValidGQAConfig(int num_heads, int num_kv_heads) {
    if (num_kv_heads <= 0 || num_heads <= 0) return false;
    if (num_kv_heads > num_heads) return false;
    // num_heads must be divisible by num_kv_heads for proper grouping
    return (num_heads % num_kv_heads) == 0;
}

// Compute the number of Q heads per KV group
inline int computeHeadsPerKVGroup(int num_heads, int num_kv_heads) {
    if (num_heads <= 0 || num_kv_heads <= 0) {
        throw std::runtime_error(
            "computeHeadsPerKVGroup: num_heads and num_kv_heads must be > 0");
    }
    return num_heads / num_kv_heads;
}

// Compute KV projection size (reduced from Q projection in GQA)
inline int computeKVProjectionSize(int d_model, int num_heads, int num_kv_heads) {
    int head_dim = d_model / num_heads;
    return num_kv_heads * head_dim;  // K and V each have this size
}

// Compute fused QKV projection size for packed [Q | K | V] output.
inline int computeQKVProjectionSize(int d_model, int num_heads, int num_kv_heads) {
    return d_model + 2 * computeKVProjectionSize(d_model, num_heads, num_kv_heads);
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
    UNSPECIFIED,// Fail-loud sentinel; JSON must author a real encoding type
    NONE,       // Learned/additive position embeddings (no ALiBi/RoPE inside attention)
    ALIBI,      // Attention with Linear Biases (bias-based, good for long-range)
    ROPE,       // Rotary Position Embedding (rotation-based, good for local patterns)
    ALIBI_ROPE  // Hybrid: ALiBi for long-range + RoPE for local patterns (recommended)
};

// Helper to get string representation of positional encoding type
inline const char* positionalEncodingTypeToString(PositionalEncodingType type) {
    switch (type) {
        case PositionalEncodingType::UNSPECIFIED: return "UNSPECIFIED";
        case PositionalEncodingType::NONE: return "NONE";
        case PositionalEncodingType::ALIBI: return "ALIBI";
        case PositionalEncodingType::ROPE: return "ROPE";
        case PositionalEncodingType::ALIBI_ROPE: return "ALIBI_ROPE";
        default: return "UNKNOWN";
    }
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
// Model Architecture Validation & Computation
//======================================================//
struct ModelArchitecture {
    int d_model = 0;
    int num_layers = 0;
    int num_heads = 0;
    int num_kv_heads = 0;  // GQA: number of KV heads
    int d_ff = 0;  // Must be set from config or computed as d_model * multiplier
    int max_seq_len = 0;
    float dropout_rate = 0.0f;
    float attention_dropout = 0.0f;        // Derived: always = dropout_rate
    bool tie_embeddings = true;  // Weight tying: share embedding/LM head weights
    PositionalEncodingType positional_encoding = PositionalEncodingType::UNSPECIFIED;

    // Positional Bias Method (PBM) authored by flat training.config PBM leaves.
    // These are policy/config values, not math constants.
    int rope_base_seq_len = 0;
    int alibi_min_locality_distance = 0;
    float alibi_slope_exponent = 0.0f;
    float alibi_max_bias = std::numeric_limits<float>::quiet_NaN();
    float rope_theta = 0.0f;
    float rope_scaling = 0.0f;

    // Flash Attention (architectural choice — affects kernel selection)
    bool use_flash_attention = true;
    int min_seq_len_for_flash = 0;     // Derived: max_seq_len / 4 (see deriveComputedTrainingHyperparameters)

    // Runtime device flag (GPU-only build forces this true; kept for plumbing)
    bool use_gpu = true;
    
    // Derived (computed from above)
    int head_dim = 0;

    // Mass-load architecture leaves from the finalized typed owner.
    // Defined after TrainingHyperparameters is complete.
    void loadFromTrainingHyperparameters(const GRIM::Config::TrainingHyperparameters& hp);
    
    // Validate architecture and compute derived values. Throws on invalid config.
    void validate() {
        if (d_model <= 0) {
            throw std::runtime_error("Invalid ModelArchitecture: d_model must be > 0, got " + std::to_string(d_model));
        }
        if (num_layers <= 0) {
            throw std::runtime_error("Invalid ModelArchitecture: num_layers must be > 0, got " + std::to_string(num_layers));
        }
        if (num_heads <= 0) {
            throw std::runtime_error("Invalid ModelArchitecture: num_heads must be > 0, got " + std::to_string(num_heads));
        }
        if (num_kv_heads <= 0) {
            throw std::runtime_error("Invalid ModelArchitecture: num_kv_heads must be > 0, got " + std::to_string(num_kv_heads));
        }
        if (max_seq_len <= 0) {
            throw std::runtime_error("Invalid ModelArchitecture: max_seq_len must be > 0, got " + std::to_string(max_seq_len));
        }
        if (!std::isfinite(dropout_rate) || dropout_rate < 0.0f || dropout_rate >= 1.0f) {
            throw std::runtime_error("Invalid ModelArchitecture: dropout_rate must be in [0, 1), got " + std::to_string(dropout_rate));
        }
        if (!std::isfinite(attention_dropout) || attention_dropout < 0.0f || attention_dropout >= 1.0f) {
            throw std::runtime_error("Invalid ModelArchitecture: attention_dropout must be in [0, 1), got " + std::to_string(attention_dropout));
        }
        if (positional_encoding == PositionalEncodingType::UNSPECIFIED) {
            throw std::runtime_error("Invalid ModelArchitecture: positional_encoding is UNSPECIFIED; training.config.use_rope/use_alibi must author it");
        }
        if (positional_encoding == PositionalEncodingType::NONE) {
            throw std::runtime_error("Invalid ModelArchitecture: positional_encoding=NONE is unsupported; learned position embeddings were removed");
        }
        if (rope_base_seq_len <= 0) {
            throw std::runtime_error("Invalid ModelArchitecture: rope_base_seq_len must be > 0, got " + std::to_string(rope_base_seq_len));
        }
        if (alibi_min_locality_distance <= 0) {
            throw std::runtime_error("Invalid ModelArchitecture: alibi_min_locality_distance must be > 0, got " + std::to_string(alibi_min_locality_distance));
        }
        if (!std::isfinite(alibi_slope_exponent) || alibi_slope_exponent == 0.0f) {
            throw std::runtime_error("Invalid ModelArchitecture: alibi_slope_exponent must be finite and non-zero, got " + std::to_string(alibi_slope_exponent));
        }
        if (!std::isfinite(alibi_max_bias) || alibi_max_bias > 0.0f) {
            throw std::runtime_error("Invalid ModelArchitecture: alibi_max_bias must be finite and <= 0, got " + std::to_string(alibi_max_bias));
        }
        if (!std::isfinite(rope_theta) || rope_theta <= 0.0f) {
            throw std::runtime_error("Invalid ModelArchitecture: rope_theta must be positive finite, got " + std::to_string(rope_theta));
        }
        if (!std::isfinite(rope_scaling) || rope_scaling <= 0.0f) {
            throw std::runtime_error("Invalid ModelArchitecture: rope_scaling must be positive finite, got " + std::to_string(rope_scaling));
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
            d_ff = d_model * D_FF_MULTIPLIER;
        }
    }
};

// Helper to compute head_dim inline
inline int computeHeadDim(int d_model, int num_heads) {
    if (d_model <= 0 || num_heads <= 0) {
        throw std::runtime_error("computeHeadDim: d_model and num_heads must be > 0");
    }
    return d_model / num_heads;
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

//======================================================//
// Generation / Sampling configuration (single source of truth)
//
// Owned here (NOT in grim_language_model_cuda.hpp) so that ai_config.json
// generation parsing can live in this same header next to the rest of
// the typed config helpers. Consumers (grim_text_server, training Phase1,
// LanguageModelConfig, generate*() APIs, Sampling bridge) refer to these
// as `GRIM::HyperParameters::SamplingStrategy` / `GenerationConfig`.
//======================================================//
enum class SamplingStrategy {
    GREEDY,
    TOP_K,
    TOP_P,
    MIN_P,           // Min-P (relative threshold)
    TYPICAL,         // Locally typical sampling
    TOP_K_TOP_P,     // Combined: Top-K first, then Top-P within survivors
    BEAM_SEARCH      // NOT SUPPORTED - exists only to give clear error
};

struct GenerationConfig {
    SamplingStrategy strategy = SamplingStrategy::TOP_P;
    int max_new_tokens = 100;
    int min_new_tokens = 0;
    float temperature = 1.0f;
    int top_k = 50;
    float top_p = 0.9f;
    float min_p = 0.0f;                    // Min-P threshold (0 = disabled)
    float typical_p = 1.0f;                // Typical sampling mass (1.0 = disabled)
    float repetition_penalty = 1.0f;
    int repetition_penalty_window = 64;
    float frequency_penalty = 0.0f;        // Additive penalty per occurrence (0 = disabled)
    float presence_penalty = 0.0f;         // Additive penalty if token appeared (0 = disabled)
    float length_penalty = 1.0f;
    int num_beams = 1;
    int num_return_sequences = 1;
    bool early_stopping = false;
    int eos_token_id = 0;
    int pad_token_id = 0;
    int bos_token_id = 2;
    int unk_token_id = 0;
    int no_repeat_ngram_size = 0;
    bool do_sample = true;
    float diversity_penalty = 0.0f;
    std::vector<int> bad_words_ids;
    /// Token IDs to mask at sampling (e.g. byte-level digit tokens); `<NUM>` must remain unmasked.
    std::vector<int> masked_numeric_literal_ids;
    unsigned int seed = 0;

    // ScratchBlock reasoning during inference
    // When true, generated atom tokens (numbers, URLs, etc.) are classified
    // and their metadata (numeric_value, atom_mask) is fed back into forwardStep()
    // so the ScratchBlock layer can inject structured reasoning embeddings.
    bool enable_scratchblock_reasoning = true;
};

using GenerationStreamCallback = std::function<void(int token_id, float score)>;

//======================================================//
// Model execution + runtime configuration (moved from
// grim_language_model_cuda.hpp — HyperParameters is the SoT
// for ALL model configuration, the model header is for the
// model class declarations only).
//======================================================//

// Determines memory allocation strategy at model construction time.
enum class ModelExecutionMode {
    TRAINING,    // Full training state with gradient buffers (~1GB+)
    INFERENCE    // Lightweight inference state with only forward caches (~385MB)
};

// Parameter-group precision policy. This is config plumbing only; kernels must
// explicitly opt in to BF16 in later changes. UNSPECIFIED is a fail-loud
// sentinel so registration cannot silently select FP32 for a missing field.
enum class ParameterGroupPrecision : uint8_t {
    UNSPECIFIED,
    FP32,
    BF16_COMPUTE
};

inline const char* parameterGroupPrecisionToString(ParameterGroupPrecision precision) {
    switch (precision) {
        case ParameterGroupPrecision::UNSPECIFIED:  return "UNSPECIFIED";
        case ParameterGroupPrecision::FP32:         return "FP32";
        case ParameterGroupPrecision::BF16_COMPUTE: return "BF16_COMPUTE";
    }
    throw std::runtime_error("parameterGroupPrecisionToString: unknown ParameterGroupPrecision enum value");
}

inline ParameterGroupPrecision parseParameterGroupPrecision(const std::string& value) {
    std::string precision = value;
    std::transform(precision.begin(), precision.end(), precision.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (precision == "fp32" || precision == "float32") {
        return ParameterGroupPrecision::FP32;
    }
    if (precision == "bf16_compute" || precision == "bf16" || precision == "bfloat16_compute") {
        return ParameterGroupPrecision::BF16_COMPUTE;
    }

    throw std::runtime_error(
        "parseParameterGroupPrecision: unknown precision value '" + value +
        "'. Valid values: fp32, bf16_compute");
}

inline void validateParameterGroupPrecision(ParameterGroupPrecision precision,
                                            const char* field,
                                            const char* caller) {
    if (precision == ParameterGroupPrecision::UNSPECIFIED) {
        throw std::runtime_error(
            std::string(caller) + ": " + field +
            " is UNSPECIFIED (valid: fp32, bf16_compute)");
    }
}

enum class HardcodedPattern {
    DISABLED,
    RANDOM_CENTERED,
    ORTHOGONAL_W277,
    ALIGNED_W277,
    CONSTANT_UNIFORM,
    ZERO_MEAN_SINE
};

struct LanguageModelConfig : public ModelArchitecture {
    // Architecture fields (d_model, num_heads, num_kv_heads, head_dim, d_ff,
    // num_layers, max_seq_len, dropout_rate, attention_dropout, positional_encoding,
    // tie_embeddings) inherited from ModelArchitecture

    int vocab_size = 0;        // Training: GRMT header. Inference: loaded tokenizer token-space size.

    // Validates architecture and computes derived values (head_dim = d_model / num_heads)
    // MUST be called after populating architecture fields
    void computeDerivedValues() {
        validate();  // ModelArchitecture::validate() computes head_dim and validates all arch fields
    }

    using ModelArchitecture::loadFromTrainingHyperparameters;
    void loadFromTrainingHyperparameters(const GRIM::Config::TrainingHyperparameters& hp,
                                         const GenerationConfig& generation_config);

    // Cache limits
    int max_cached_batch = 0;
    int max_cached_seq_len = 0;
    int max_tokens_per_batch = 0;  // Optional token budget for training logits/loss

    // Fixed config values (not architecture-dependent)
    // Issue #104 FIX: Changed from 1e-3 to 1e-5 (see TransformerConfig above for rationale)
    float rms_epsilon = 1e-5f;  // RMSNorm epsilon - shared across all RMSNorm layers
    bool causal_mask = true;
    bool use_pre_norm = true;
    bool fuse_qkv = true;
    bool use_simd = true;
    int num_threads = 4;
    bool use_bias = true;
    bool qk_norm_enabled = false;  // QK-Norm: RMSNorm applied to Q and K before attention scoring

    // Issue #109: LayerScale - per-channel learnable residual scaling vectors [1, d_model]
    bool use_layer_scale = false;
    float layer_scale_init = 1.0f;       // Issue #129: init=1.0 (NOT CaiT's 0.1)

    // use_gpu, use_flash_attention, min_seq_len_for_flash inherited from ModelArchitecture (Phase 3b)
    std::string vocab_path;

    // Execution mode - determines memory allocation strategy
    ModelExecutionMode execution_mode = ModelExecutionMode::INFERENCE;

    // Parameter-group precision policy. Registration reads these from the
    // actual LanguageModelConfig carried by LanguageModel::getConfig(); do not
    // pass a sidecar precision wrapper beside the model config.
    ParameterGroupPrecision parameter_precision_embedding = ParameterGroupPrecision::UNSPECIFIED;
    ParameterGroupPrecision parameter_precision_lm_head = ParameterGroupPrecision::UNSPECIFIED;
    ParameterGroupPrecision parameter_precision_attention = ParameterGroupPrecision::UNSPECIFIED;
    ParameterGroupPrecision parameter_precision_ffn = ParameterGroupPrecision::UNSPECIFIED;
    ParameterGroupPrecision parameter_precision_rmsnorm = ParameterGroupPrecision::UNSPECIFIED;
    ParameterGroupPrecision parameter_precision_scratchblock = ParameterGroupPrecision::UNSPECIFIED;
    ParameterGroupPrecision parameter_precision_mtp = ParameterGroupPrecision::UNSPECIFIED;
    ParameterGroupPrecision parameter_precision_reasoning_head = ParameterGroupPrecision::UNSPECIFIED;
    ParameterGroupPrecision parameter_precision_execution_block = ParameterGroupPrecision::UNSPECIFIED;
    ParameterGroupPrecision parameter_precision_slot_selector = ParameterGroupPrecision::UNSPECIFIED;

    // ScratchBlock reasoning layer config
    bool use_scratch_block = false;
    int scratch_block_atom_embedding_dim = 0;
    int scratch_block_max_atoms = 0;
    float scratch_block_atom_scale = 0.0f;
    bool scratch_block_execution_first_type_only = false;

    // ReasoningHead config
    bool reasoning_head_enabled = false;
    int reasoning_num_ops = 8;

    // ExecutionBlock config — differentiable register machine
    bool execution_block_enabled = false;
    int execution_block_layer = -1;
    int execution_block_num_ops = 0;
    int execution_block_num_slots = 0;
    int execution_block_num_scratch_slots = 0;
    int execution_block_num_steps = 0;
    int execution_block_value_decode_input_dim = 0;
    int execution_block_value_decode_hidden_dim = 0;
    int execution_block_d_key = 0;
    int execution_block_d_type = 0;
    int execution_block_cross_attn_head_dim = 0;
    int execution_block_cross_attn_topk = 0;
    float execution_block_usage_decay = 0.0f;
    float execution_block_inject_gate_temp = 0.0f;
    int execution_block_result_slot_mode = 0;
    int execution_block_result_slot_index = 0;
    bool execution_block_debug_mode = false;
    float execution_block_entropy_collapse_threshold = 0.0f;
    float execution_block_write_collapse_threshold = 0.0f;
    float execution_block_magnitude_limit = 0.0f;
    float execution_block_diversity_kappa = 0.0f;
    float execution_block_temp_start = 0.0f;
    float execution_block_temp_end = 0.0f;
    int   execution_block_temp_schedule = 0;
    float execution_block_entropy_weight = 0.0f;

    // Causal state loss weights (Fixes 1-9)
    float execution_block_transition_hard_threshold = 0.0f;
    int   execution_block_gate_warmup_steps = 0;
    float execution_block_causal_w1_transition = 0.0f;

    float div_invalid_penalty_weight = 0.0f;
    float div_magnitude_penalty_weight = 0.0f;

    float arg_reinforce_weight = 0.0f;
    float arg_reinforce_baseline_decay = 0.0f;

    bool  structured_ce_enabled = false;
    float structured_ce_weight  = 0.0f;

    // Decode-time slot selector config
    bool  selector_enabled = false;
    int   decode_time_slot_feature_dim = 0;
    int   selector_d_selector = 0;
    float selector_selection_margin = 0.0f;
    float selector_supervision_weight = 0.0f;

    // Execution-first structured CE loss config (Step X / Y multipliers)
    float step_x_multiplier = 0.0f;
    float step_y_multiplier = 0.0f;
    bool  step_y_overrides_x = false;
    float entropy_aux_weight = 0.0f;
    float value_match_epsilon = 0.0f;
    float final_slot_consistency_weight = 0.0f;

    // LM Head / RMSNorm gamma config (Issue #37 / #40 fixes)
    bool lm_head_center_hidden_states = false;
    bool freeze_learned_rms_gammas = false;
    bool project_out_pc1 = false;
    int  pc1_power_iters = 0;
    bool center_logits = false;
    bool center_encoder_residuals = false;

    HardcodedPattern hardcoded_hidden_pattern = HardcodedPattern::DISABLED;
    int hardcoded_log_every_n_batches = 0;

    GenerationConfig generation;

    // Multi-token prediction (MTP) - auxiliary heads (Gloeckle et al. 2024)
    bool mtp_enabled = false;
    int mtp_k = 0;
    float mtp_alpha = 0.0f;
    int mtp_alpha_warmup_steps = 0;
};

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

// LogRecorderConfig / TapeLogConfig / TrainingHyperparameters live here (the
// middle layer). They use <map>/<vector>/<string> in host context only;
// modern nvcc (CUDA 12+) handles these in host code paths fine, so they are
// no longer gated behind #ifndef __CUDACC__. The structs MUST be visible to
// CUDA TUs (e.g. Phase1_Startup.hpp / TelemetryUpdate.cu) — "hyperparameters
// live in HyperParameters_GPU.hpp" is the load-bearing invariant.
#include <map>
#include <string>
#include <vector>

namespace GRIM {
namespace Config {

/**
 * @brief Log Recorder configuration
 * 
 * Controls modular logging levels and overrides.
 */
struct LogRecorderConfig {
    bool enabled = false;
    std::string default_level;
    std::map<std::string, std::string> modules;
    
    // Layer logging enables (for RecordLayerLogHost)
    struct LayerEnables {
        bool embedding = false;
        bool rms_norm = false;
        bool attention = false;
        bool feed_forward = false;
        bool residual = false;
        bool encoding = false;     // Aggregate gradient logs
        bool serialization = false;
        bool execution_block = false;
    } layers;
};

/**
 * @brief Unified tape-based logging configuration (replaces LogRecorderConfig).
 * Parsed from training.config.logging in ai_config.json.
 */
struct TapeLogConfig {
    std::string default_level;
    bool equation_csv_enabled = false;
    bool stderr_enabled = false;
    size_t initial_capacity = 0;
    std::map<std::string, std::string> group_overrides;
};

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
    // Full model configuration fields are direct TrainingHyperparameters fields.
    // Do not add an `architecture` sub-object here: ai_config.json is authored
    // as one flat training.config object, and downstream immutable views slice
    // directly from this finalized owner.

    int d_model = 0;
    int num_layers = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int d_ff = 0;
    int max_seq_len = 0;
    float dropout_rate = 0.0f;
    float attention_dropout = 0.0f;
    bool tie_embeddings = true;
    GRIM::HyperParameters::PositionalEncodingType positional_encoding =
        GRIM::HyperParameters::PositionalEncodingType::UNSPECIFIED;
    int rope_base_seq_len = 0;
    int alibi_min_locality_distance = 0;
    float alibi_slope_exponent = 0.0f;
    float alibi_max_bias = std::numeric_limits<float>::quiet_NaN();
    float rope_theta = 0.0f;
    float rope_scaling = 0.0f;
    bool use_flash_attention = true;
    int min_seq_len_for_flash = 0;
    bool use_gpu = true;
    int head_dim = 0;

    int vocab_size = 0;
    int max_cached_batch = 0;
    int max_cached_seq_len = 0;
    int max_tokens_per_batch = 0;
    float rms_epsilon = 1e-5f;
    bool causal_mask = true;
    bool use_pre_norm = true;
    bool fuse_qkv = true;
    bool use_simd = true;
    int num_threads = 4;
    bool use_bias = true;
    bool qk_norm_enabled = false;
    bool use_layer_scale = false;
    float layer_scale_init = 1.0f;
    std::string vocab_path;
    GRIM::HyperParameters::ModelExecutionMode execution_mode =
        GRIM::HyperParameters::ModelExecutionMode::INFERENCE;

    GRIM::HyperParameters::ParameterGroupPrecision parameter_precision_embedding = GRIM::HyperParameters::ParameterGroupPrecision::UNSPECIFIED;
    GRIM::HyperParameters::ParameterGroupPrecision parameter_precision_lm_head = GRIM::HyperParameters::ParameterGroupPrecision::UNSPECIFIED;
    GRIM::HyperParameters::ParameterGroupPrecision parameter_precision_attention = GRIM::HyperParameters::ParameterGroupPrecision::UNSPECIFIED;
    GRIM::HyperParameters::ParameterGroupPrecision parameter_precision_ffn = GRIM::HyperParameters::ParameterGroupPrecision::UNSPECIFIED;
    GRIM::HyperParameters::ParameterGroupPrecision parameter_precision_rmsnorm = GRIM::HyperParameters::ParameterGroupPrecision::UNSPECIFIED;
    GRIM::HyperParameters::ParameterGroupPrecision parameter_precision_scratchblock = GRIM::HyperParameters::ParameterGroupPrecision::UNSPECIFIED;
    GRIM::HyperParameters::ParameterGroupPrecision parameter_precision_mtp = GRIM::HyperParameters::ParameterGroupPrecision::UNSPECIFIED;
    GRIM::HyperParameters::ParameterGroupPrecision parameter_precision_reasoning_head = GRIM::HyperParameters::ParameterGroupPrecision::UNSPECIFIED;
    GRIM::HyperParameters::ParameterGroupPrecision parameter_precision_execution_block = GRIM::HyperParameters::ParameterGroupPrecision::UNSPECIFIED;
    GRIM::HyperParameters::ParameterGroupPrecision parameter_precision_slot_selector = GRIM::HyperParameters::ParameterGroupPrecision::UNSPECIFIED;

    bool use_scratch_block = false;
    int scratch_block_atom_embedding_dim = 0;
    int scratch_block_max_atoms = 0;
    float scratch_block_atom_scale = 0.0f;
    bool scratch_block_execution_first_type_only = false;

    bool reasoning_head_enabled = false;
    int reasoning_num_ops = 8;

    bool execution_block_enabled = false;
    int execution_block_layer = -1;
    int execution_block_num_ops = 0;
    int execution_block_num_slots = 0;
    int execution_block_num_scratch_slots = 0;
    int execution_block_num_steps = 0;
    int execution_block_value_decode_input_dim = 0;
    int execution_block_value_decode_hidden_dim = 0;
    int execution_block_d_key = 0;
    int execution_block_d_type = 0;
    int execution_block_cross_attn_head_dim = 0;
    int execution_block_cross_attn_topk = 0;
    float execution_block_usage_decay = 0.0f;
    float execution_block_inject_gate_temp = 0.0f;
    int execution_block_result_slot_mode = 0;
    int execution_block_result_slot_index = 0;
    bool execution_block_debug_mode = false;
    float execution_block_entropy_collapse_threshold = 0.0f;
    float execution_block_write_collapse_threshold = 0.0f;
    float execution_block_magnitude_limit = 0.0f;
    float execution_block_diversity_kappa = 0.0f;
    float execution_block_temp_start = 0.0f;
    float execution_block_temp_end = 0.0f;
    int execution_block_temp_schedule = 0;
    float execution_block_entropy_weight = 0.0f;
    float execution_block_transition_hard_threshold = 0.0f;
    int execution_block_gate_warmup_steps = 0;
    float execution_block_causal_w1_transition = 0.0f;
    float div_invalid_penalty_weight = 0.0f;
    float div_magnitude_penalty_weight = 0.0f;
    float arg_reinforce_weight = 0.0f;
    float arg_reinforce_baseline_decay = 0.0f;
    bool structured_ce_enabled = false;
    float structured_ce_weight = 0.0f;
    bool selector_enabled = false;
    int decode_time_slot_feature_dim = 0;
    int selector_d_selector = 0;
    float selector_selection_margin = 0.0f;
    float selector_supervision_weight = 0.0f;
    float step_x_multiplier = 0.0f;
    float step_y_multiplier = 0.0f;
    bool step_y_overrides_x = false;
    float entropy_aux_weight = 0.0f;
    float value_match_epsilon = 0.0f;
    float final_slot_consistency_weight = 0.0f;

    bool lm_head_center_hidden_states = false;
    bool freeze_learned_rms_gammas = false;
    bool project_out_pc1 = false;
    int pc1_power_iters = 0;
    bool center_logits = false;
    bool center_encoder_residuals = false;
    GRIM::HyperParameters::HardcodedPattern hardcoded_hidden_pattern =
        GRIM::HyperParameters::HardcodedPattern::DISABLED;
    int hardcoded_log_every_n_batches = 0;
    bool mtp_enabled = false;
    int mtp_k = 0;
    float mtp_alpha = 0.0f;
    int mtp_alpha_warmup_steps = 0;

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
    int warmup_steps;   // Derived in Phase2 from warmup_fraction * estimated_total_steps
    bool force_rebuild_vocab;
    bool cosine_decay_enabled;
    bool cosine_warm_restarts;
    float cosine_decay_min_lr;
    int sliding_window_stride;  // Hop size between training windows; must be > 0 and <= effective max_seq_len
    int min_seq_valid_tokens;  // Minimum valid tokens required (after masking first/last positions)
    int log_interval;
    int atom_stats_interval;
    int atom_stats_max_seqs;
    int validation_interval;
    int checkpoint_interval;

    // Soft restart — NO DEFAULTS
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

    // Telemetry lattice (TelemetryLattice construction params) - NO DEFAULTS
    int telemetry_lattice_num_levels;
    int telemetry_lattice_num_streams;
    float telemetry_lattice_beta_mu;
    float telemetry_lattice_beta_a;
    float telemetry_lattice_beta_delta;
    float telemetry_lattice_beta_r;
    float telemetry_lattice_beta_run;
    float telemetry_lattice_beta_v;
    float telemetry_lattice_k_out0;
    float telemetry_lattice_alpha_v;
    float telemetry_lattice_epsilon;
    bool  telemetry_lattice_strict_mode;

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
    bool loss_class_balanced_enabled;
    float loss_class_balanced_beta;


    // LM Head centering master toggle (training-only, no LMC counterpart).
    bool lm_head_centering_enabled;
    
    // Embedding freeze guard - freezes embedding weights after N optimizer steps
    bool embedding_freeze_enabled;
    int embedding_freeze_after_step;

    // Optimizer selection - NO DEFAULTS.
    // optimizer_kind ∈ {"adamw", "radamw"}.
    // Selecting "radamw" runs the full Liu et al. 2019 rectified update with
    // decoupled weight decay (the "R" + "W" in RAdamW). There is no rectification
    // toggle — rectification IS RAdamW; for plain bias-corrected AdamW use "adamw".
    std::string optimizer_kind;
    float optimizer_beta1;
    float optimizer_beta2;
    float optimizer_epsilon;

    // Stability overrides - NO DEFAULTS
    bool stability_overrides_enabled;
    int stability_override_batch_size;
    int stability_override_max_seq_len;
    float stability_override_clip_per_token;
    
    // Scratch blocks - NO DEFAULTS
    bool scratch_blocks_enabled;
    size_t scratch_max_tokens_per_block;
    size_t scratch_num_blocks;
    bool scratch_write_combined;
    
    // CUDA execution mode - NO DEFAULTS
    bool single_stream_mode;
    bool disable_async_frees;
    bool synchronize_after_kernels;

    // Multi-token prediction (MTP) — log-ratio monitor is training-only.
    bool mtp_log_ratio_monitor;

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

} // namespace Config
} // namespace GRIM

namespace GRIM {
namespace HyperParameters {

inline void deriveComputedTrainingHyperparameters(GRIM::Config::TrainingHyperparameters& params);

} // namespace HyperParameters
} // namespace GRIM

// Sentinel: tells ai_config_paths.hpp that the structs it needs are now in
// scope. Without this, ai_config_paths.hpp #errors out (enforces that it can
// only be entered through this header).
#define GRIM_HP_GPU_DEFINED_TRAINING_STRUCTS 1

// Include the config header (JSON loaders + path utils + AiConfigSnapshot +
// AiConfigSnapshot GRIM-text startup leaves). ai_config_paths.hpp does NOT include
// this file back; it asserts via #error that the sentinel above is set, so
// it can only be entered through this header.
#include "../../../../../control/ai_config_paths.hpp"

// Only compile if TrainingHyperparameters is fully defined
// (checked via the marker defined in ai_config_paths.hpp)
#ifdef GRIM_CONFIG_AI_CONFIG_PATHS_HPP_INCLUDED

namespace GRIM {
namespace HyperParameters {

inline void deriveComputedTrainingHyperparameters(GRIM::Config::TrainingHyperparameters& params) {
    if (params.max_seq_len <= 0) {
        throw std::runtime_error(
            "deriveComputedTrainingHyperparameters: max_seq_len must be > 0, got " +
            std::to_string(params.max_seq_len));
    }
    params.min_seq_valid_tokens = params.max_seq_len / 4;
    params.min_seq_len_for_flash = params.max_seq_len / 4;
    params.scratch_max_tokens_per_block = static_cast<size_t>(params.max_seq_len);
    params.attention_dropout = params.dropout_rate;
    params.d_ff = params.d_model * D_FF_MULTIPLIER;

    if (params.cosine_decay_enabled) {
        if (params.learning_rate <= 0.0f) {
            throw std::runtime_error(
                "deriveComputedTrainingHyperparameters: learning_rate must be > 0 when cosine_decay is enabled, got " +
                std::to_string(params.learning_rate));
        }
        params.cosine_decay_min_lr = params.learning_rate * 0.1f;
    }

        if (params.d_model <= 0 || params.num_heads <= 0) {
        throw std::runtime_error(
            "deriveComputedTrainingHyperparameters: d_model and num_heads must be > 0");
    }
        if ((params.d_model % params.num_heads) != 0) {
        throw std::runtime_error(
            "deriveComputedTrainingHyperparameters: d_model (" +
            std::to_string(params.d_model) + ") must be divisible by num_heads (" +
            std::to_string(params.num_heads) + ")");
    }
        const int head_dim = params.d_model / params.num_heads;
        params.head_dim = head_dim;
        params.execution_block_d_key = head_dim;
        params.execution_block_cross_attn_head_dim = head_dim;

    if (params.warmup_fraction <= 0.0f || params.warmup_fraction >= 1.0f) {
        throw std::runtime_error(
            "deriveComputedTrainingHyperparameters: warmup_fraction must be in (0, 1), got " +
            std::to_string(params.warmup_fraction));
    }
}

inline void ModelArchitecture::loadFromTrainingHyperparameters(
    const GRIM::Config::TrainingHyperparameters& hp)
{
    d_model = hp.d_model;
    num_layers = hp.num_layers;
    num_heads = hp.num_heads;
    num_kv_heads = hp.num_kv_heads;
    d_ff = hp.d_ff;
    max_seq_len = hp.max_seq_len;
    dropout_rate = hp.dropout_rate;
    attention_dropout = hp.attention_dropout;
    tie_embeddings = hp.tie_embeddings;
    positional_encoding = hp.positional_encoding;
    rope_base_seq_len = hp.rope_base_seq_len;
    alibi_min_locality_distance = hp.alibi_min_locality_distance;
    alibi_slope_exponent = hp.alibi_slope_exponent;
    alibi_max_bias = hp.alibi_max_bias;
    rope_theta = hp.rope_theta;
    rope_scaling = hp.rope_scaling;
    use_flash_attention = hp.use_flash_attention;
    min_seq_len_for_flash = hp.min_seq_len_for_flash;
    use_gpu = hp.use_gpu;
    head_dim = hp.head_dim;
    validate();
}

inline void LanguageModelConfig::loadFromTrainingHyperparameters(
    const GRIM::Config::TrainingHyperparameters& hp,
    const GenerationConfig& generation_config)
{
    static_cast<ModelArchitecture&>(*this).loadFromTrainingHyperparameters(hp);
    vocab_size = hp.vocab_size;
    max_cached_batch = hp.max_cached_batch;
    max_cached_seq_len = hp.max_cached_seq_len;
    max_tokens_per_batch = hp.max_tokens_per_batch;
    rms_epsilon = hp.rms_epsilon;
    causal_mask = hp.causal_mask;
    use_pre_norm = hp.use_pre_norm;
    fuse_qkv = hp.fuse_qkv;
    use_simd = hp.use_simd;
    num_threads = hp.num_threads;
    use_bias = hp.use_bias;
    qk_norm_enabled = hp.qk_norm_enabled;
    use_layer_scale = hp.use_layer_scale;
    layer_scale_init = hp.layer_scale_init;
    vocab_path = hp.vocab_path;
    execution_mode = hp.execution_mode;
    parameter_precision_embedding = hp.parameter_precision_embedding;
    parameter_precision_lm_head = hp.parameter_precision_lm_head;
    parameter_precision_attention = hp.parameter_precision_attention;
    parameter_precision_ffn = hp.parameter_precision_ffn;
    parameter_precision_rmsnorm = hp.parameter_precision_rmsnorm;
    parameter_precision_scratchblock = hp.parameter_precision_scratchblock;
    parameter_precision_mtp = hp.parameter_precision_mtp;
    parameter_precision_reasoning_head = hp.parameter_precision_reasoning_head;
    parameter_precision_execution_block = hp.parameter_precision_execution_block;
    parameter_precision_slot_selector = hp.parameter_precision_slot_selector;
    use_scratch_block = hp.use_scratch_block;
    scratch_block_atom_embedding_dim = hp.scratch_block_atom_embedding_dim;
    scratch_block_max_atoms = hp.scratch_block_max_atoms;
    scratch_block_atom_scale = hp.scratch_block_atom_scale;
    scratch_block_execution_first_type_only = hp.scratch_block_execution_first_type_only;
    reasoning_head_enabled = hp.reasoning_head_enabled;
    reasoning_num_ops = hp.reasoning_num_ops;
    execution_block_enabled = hp.execution_block_enabled;
    execution_block_layer = hp.execution_block_layer;
    execution_block_num_ops = hp.execution_block_num_ops;
    execution_block_num_slots = hp.execution_block_num_slots;
    execution_block_num_scratch_slots = hp.execution_block_num_scratch_slots;
    execution_block_num_steps = hp.execution_block_num_steps;
    execution_block_value_decode_input_dim = hp.execution_block_value_decode_input_dim;
    execution_block_value_decode_hidden_dim = hp.execution_block_value_decode_hidden_dim;
    execution_block_d_key = hp.execution_block_d_key;
    execution_block_d_type = hp.execution_block_d_type;
    execution_block_cross_attn_head_dim = hp.execution_block_cross_attn_head_dim;
    execution_block_cross_attn_topk = hp.execution_block_cross_attn_topk;
    execution_block_usage_decay = hp.execution_block_usage_decay;
    execution_block_inject_gate_temp = hp.execution_block_inject_gate_temp;
    execution_block_result_slot_mode = hp.execution_block_result_slot_mode;
    execution_block_result_slot_index = hp.execution_block_result_slot_index;
    execution_block_debug_mode = hp.execution_block_debug_mode;
    execution_block_entropy_collapse_threshold = hp.execution_block_entropy_collapse_threshold;
    execution_block_write_collapse_threshold = hp.execution_block_write_collapse_threshold;
    execution_block_magnitude_limit = hp.execution_block_magnitude_limit;
    execution_block_diversity_kappa = hp.execution_block_diversity_kappa;
    execution_block_temp_start = hp.execution_block_temp_start;
    execution_block_temp_end = hp.execution_block_temp_end;
    execution_block_temp_schedule = hp.execution_block_temp_schedule;
    execution_block_entropy_weight = hp.execution_block_entropy_weight;
    execution_block_transition_hard_threshold = hp.execution_block_transition_hard_threshold;
    execution_block_gate_warmup_steps = hp.execution_block_gate_warmup_steps;
    execution_block_causal_w1_transition = hp.execution_block_causal_w1_transition;
    div_invalid_penalty_weight = hp.div_invalid_penalty_weight;
    div_magnitude_penalty_weight = hp.div_magnitude_penalty_weight;
    arg_reinforce_weight = hp.arg_reinforce_weight;
    arg_reinforce_baseline_decay = hp.arg_reinforce_baseline_decay;
    structured_ce_enabled = hp.structured_ce_enabled;
    structured_ce_weight = hp.structured_ce_weight;
    selector_enabled = hp.selector_enabled;
    decode_time_slot_feature_dim = hp.decode_time_slot_feature_dim;
    selector_d_selector = hp.selector_d_selector;
    selector_selection_margin = hp.selector_selection_margin;
    selector_supervision_weight = hp.selector_supervision_weight;
    step_x_multiplier = hp.step_x_multiplier;
    step_y_multiplier = hp.step_y_multiplier;
    step_y_overrides_x = hp.step_y_overrides_x;
    entropy_aux_weight = hp.entropy_aux_weight;
    value_match_epsilon = hp.value_match_epsilon;
    final_slot_consistency_weight = hp.final_slot_consistency_weight;
    lm_head_center_hidden_states = hp.lm_head_center_hidden_states;
    freeze_learned_rms_gammas = hp.freeze_learned_rms_gammas;
    project_out_pc1 = hp.project_out_pc1;
    pc1_power_iters = hp.pc1_power_iters;
    center_logits = hp.center_logits;
    center_encoder_residuals = hp.center_encoder_residuals;
    hardcoded_hidden_pattern = hp.hardcoded_hidden_pattern;
    hardcoded_log_every_n_batches = hp.hardcoded_log_every_n_batches;
    generation = generation_config;
    mtp_enabled = hp.mtp_enabled;
    mtp_k = hp.mtp_k;
    mtp_alpha = hp.mtp_alpha;
    mtp_alpha_warmup_steps = hp.mtp_alpha_warmup_steps;
}

inline void deriveWarmupSteps(GRIM::Config::TrainingHyperparameters& params, int estimated_total_steps) {
    if (estimated_total_steps <= 0) {
        throw std::runtime_error(
            "deriveWarmupSteps: estimated_total_steps must be > 0, got " +
            std::to_string(estimated_total_steps));
    }
    if (params.warmup_fraction <= 0.0f || params.warmup_fraction >= 1.0f) {
        throw std::runtime_error(
            "deriveWarmupSteps: warmup_fraction must be in (0, 1), got " +
            std::to_string(params.warmup_fraction));
    }

    params.warmup_steps = std::max(1, static_cast<int>(params.warmup_fraction * estimated_total_steps));
    params.mtp_alpha_warmup_steps = params.warmup_steps;
    params.telemetry_warmup_steps = params.warmup_steps;
    params.execution_block_gate_warmup_steps = params.warmup_steps;
}

inline void validateExecutionBlockHyperparameters(
    const GRIM::Config::TrainingHyperparameters& params)
{
    auto requirePositiveInt = [](int value, const char* field) {
        if (value <= 0) {
            throw std::runtime_error(std::string("validateExecutionBlockHyperparameters: ") +
                                     field + " must be > 0, got " + std::to_string(value));
        }
    };
    auto requireNonNegativeInt = [](int value, const char* field) {
        if (value < 0) {
            throw std::runtime_error(std::string("validateExecutionBlockHyperparameters: ") +
                                     field + " must be >= 0, got " + std::to_string(value));
        }
    };
    auto requirePositiveFinite = [](float value, const char* field) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error(std::string("validateExecutionBlockHyperparameters: ") +
                                     field + " must be a positive finite value, got " +
                                     std::to_string(value));
        }
    };
    auto requireNonNegativeFinite = [](float value, const char* field) {
        if (!std::isfinite(value) || value < 0.0f) {
            throw std::runtime_error(std::string("validateExecutionBlockHyperparameters: ") +
                                     field + " must be finite and >= 0, got " +
                                     std::to_string(value));
        }
    };
    auto requireUnitInterval = [](float value, const char* field) {
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
            throw std::runtime_error(std::string("validateExecutionBlockHyperparameters: ") +
                                     field + " must be finite and in [0,1], got " +
                                     std::to_string(value));
        }
    };

    if (params.use_scratch_block || params.execution_block_enabled) {
        requirePositiveInt(params.scratch_block_atom_embedding_dim,
                           "scratch_block_atom_embedding_dim");
        requirePositiveInt(params.scratch_block_max_atoms,
                           "scratch_block_max_atoms");
        requirePositiveFinite(params.scratch_block_atom_scale,
                              "scratch_block_atom_scale");
    }

    if (params.structured_ce_enabled) {
        requirePositiveFinite(params.structured_ce_weight,
                              "execution_block_structured_ce_weight");
    }
    requireNonNegativeFinite(params.step_x_multiplier,
                             "execution_block_step_x_multiplier");
    requireNonNegativeFinite(params.step_y_multiplier,
                             "execution_block_step_y_multiplier");
    requireNonNegativeFinite(params.entropy_aux_weight,
                             "execution_block_entropy_aux_weight");
    requirePositiveFinite(params.value_match_epsilon,
                          "execution_block_value_match_epsilon");
    requireNonNegativeFinite(params.final_slot_consistency_weight,
                             "execution_block_final_slot_consistency_weight");

    if (params.selector_enabled && !params.execution_block_enabled) {
        throw std::runtime_error(
            "validateExecutionBlockHyperparameters: selector_enabled=true requires execution_block_enabled=true");
    }

    if (!params.execution_block_enabled) {
        return;
    }

    if (!params.use_scratch_block) {
        throw std::runtime_error(
            "validateExecutionBlockHyperparameters: execution_block_enabled=true requires use_scratch_block=true");
    }
    if (params.execution_block_layer < -1) {
        throw std::runtime_error(
            "validateExecutionBlockHyperparameters: execution_block_layer must be -1 or >= 0, got " +
            std::to_string(params.execution_block_layer));
    }
    if (params.execution_block_layer >= params.num_layers) {
        throw std::runtime_error(
            "validateExecutionBlockHyperparameters: execution_block_layer=" +
            std::to_string(params.execution_block_layer) +
            " exceeds num_layers=" + std::to_string(params.num_layers));
    }

    requirePositiveInt(params.execution_block_num_ops, "execution_block_num_ops");
    requirePositiveInt(params.execution_block_num_slots, "execution_block_num_slots");
    requireNonNegativeInt(params.execution_block_num_scratch_slots,
                          "execution_block_num_scratch_slots");
    if (params.execution_block_num_scratch_slots >= params.execution_block_num_slots) {
        throw std::runtime_error(
            "validateExecutionBlockHyperparameters: execution_block_num_scratch_slots=" +
            std::to_string(params.execution_block_num_scratch_slots) +
            " must be < execution_block_num_slots=" +
            std::to_string(params.execution_block_num_slots));
    }
    requirePositiveInt(params.execution_block_num_steps, "execution_block_num_steps");
    requirePositiveInt(params.execution_block_value_decode_input_dim,
                       "execution_block_value_decode_input_dim");
    requirePositiveInt(params.execution_block_value_decode_hidden_dim,
                       "execution_block_value_decode_hidden_dim");
    if (params.execution_block_value_decode_input_dim + 16 > params.scratch_block_atom_embedding_dim) {
        throw std::runtime_error(
            "validateExecutionBlockHyperparameters: execution_block_value_decode_input_dim + 16 must fit scratch_block_atom_embedding_dim");
    }

    requirePositiveInt(params.execution_block_d_key, "execution_block_d_key");
    if (params.execution_block_d_key > 64) {
        throw std::runtime_error(
            "validateExecutionBlockHyperparameters: execution_block_d_key must be <= 64, got " +
            std::to_string(params.execution_block_d_key));
    }
    requirePositiveInt(params.execution_block_d_type, "execution_block_d_type");
    requirePositiveInt(params.execution_block_cross_attn_head_dim,
                       "execution_block_cross_attn_head_dim");
    requirePositiveInt(params.execution_block_cross_attn_topk,
                       "execution_block_cross_attn_topk");
    if (params.execution_block_cross_attn_topk > params.execution_block_num_slots) {
        throw std::runtime_error(
            "validateExecutionBlockHyperparameters: execution_block_cross_attn_topk=" +
            std::to_string(params.execution_block_cross_attn_topk) +
            " exceeds execution_block_num_slots=" +
            std::to_string(params.execution_block_num_slots));
    }

    requirePositiveFinite(params.execution_block_usage_decay,
                          "execution_block_usage_decay");
    if (params.execution_block_usage_decay > 1.0f) {
        throw std::runtime_error(
            "validateExecutionBlockHyperparameters: execution_block_usage_decay must be <= 1, got " +
            std::to_string(params.execution_block_usage_decay));
    }
    requirePositiveFinite(params.execution_block_inject_gate_temp,
                          "execution_block_inject_gate_temp");
    requireUnitInterval(params.execution_block_entropy_collapse_threshold,
                        "execution_block_entropy_collapse_threshold");
    requireUnitInterval(params.execution_block_write_collapse_threshold,
                        "execution_block_write_collapse_threshold");
    requirePositiveFinite(params.execution_block_magnitude_limit,
                          "execution_block_magnitude_limit");
    requirePositiveFinite(params.execution_block_diversity_kappa,
                          "execution_block_diversity_kappa");
    requirePositiveFinite(params.execution_block_temp_start,
                          "execution_block_temp_start");
    requirePositiveFinite(params.execution_block_temp_end,
                          "execution_block_temp_end");
    requireNonNegativeInt(params.execution_block_temp_schedule,
                          "execution_block_temp_schedule");
    requireNonNegativeFinite(params.execution_block_entropy_weight,
                             "execution_block_entropy_weight");
    requireNonNegativeFinite(params.execution_block_transition_hard_threshold,
                             "execution_block_transition_hard_threshold");
    requireNonNegativeInt(params.execution_block_gate_warmup_steps,
                          "execution_block_gate_warmup_steps");
    requireNonNegativeFinite(params.execution_block_causal_w1_transition,
                             "execution_block_causal_w1_transition");
    requireNonNegativeFinite(params.div_invalid_penalty_weight,
                             "execution_block_div_invalid_penalty_weight");
    requireNonNegativeFinite(params.div_magnitude_penalty_weight,
                             "execution_block_div_magnitude_penalty_weight");
    requireNonNegativeFinite(params.arg_reinforce_weight,
                             "execution_block_arg_reinforce_weight");
    requireUnitInterval(params.arg_reinforce_baseline_decay,
                        "execution_block_arg_reinforce_baseline_decay");

    if (params.execution_block_result_slot_index < -1 ||
        params.execution_block_result_slot_index >= params.execution_block_num_slots) {
        throw std::runtime_error(
            "validateExecutionBlockHyperparameters: execution_block_result_slot_index=" +
            std::to_string(params.execution_block_result_slot_index) +
            " out of range [-1, execution_block_num_slots)");
    }
    if (params.execution_block_result_slot_mode < 0 || params.execution_block_result_slot_mode > 1) {
        throw std::runtime_error(
            "validateExecutionBlockHyperparameters: execution_block_result_slot_mode must be 0 or 1, got " +
            std::to_string(params.execution_block_result_slot_mode));
    }

    if (params.selector_enabled) {
        requirePositiveInt(params.decode_time_slot_feature_dim,
                           "selector_d_slot_features");
        requirePositiveInt(params.selector_d_selector,
                           "selector_d_selector");
        requireNonNegativeFinite(params.selector_selection_margin,
                                 "selector_selection_margin");
        requireNonNegativeFinite(params.selector_supervision_weight,
                                 "selector_supervision_weight");
    }
}

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
    if (params.sliding_window_stride <= 0) {
        throw std::runtime_error("FATAL: sliding_window_stride must be > 0, got " +
                                 std::to_string(params.sliding_window_stride));
    }
    if (params.max_seq_len <= 0) {
        throw std::runtime_error("FATAL: max_seq_len must be > 0, got " +
                                 std::to_string(params.max_seq_len));
    }
    if (params.sliding_window_stride > params.max_seq_len) {
        throw std::runtime_error("FATAL: sliding_window_stride=" +
                                 std::to_string(params.sliding_window_stride) +
                                 " exceeds max_seq_len=" +
                                 std::to_string(params.max_seq_len));
    }
    validateParameterGroupPrecision(params.parameter_precision_embedding, "parameter_precision_embedding", "validateTrainingHyperparameters");
    validateParameterGroupPrecision(params.parameter_precision_lm_head, "parameter_precision_lm_head", "validateTrainingHyperparameters");
    validateParameterGroupPrecision(params.parameter_precision_attention, "parameter_precision_attention", "validateTrainingHyperparameters");
    validateParameterGroupPrecision(params.parameter_precision_ffn, "parameter_precision_ffn", "validateTrainingHyperparameters");
    validateParameterGroupPrecision(params.parameter_precision_rmsnorm, "parameter_precision_rmsnorm", "validateTrainingHyperparameters");
    validateParameterGroupPrecision(params.parameter_precision_scratchblock, "parameter_precision_scratchblock", "validateTrainingHyperparameters");
    validateParameterGroupPrecision(params.parameter_precision_mtp, "parameter_precision_mtp", "validateTrainingHyperparameters");
    validateParameterGroupPrecision(params.parameter_precision_reasoning_head, "parameter_precision_reasoning_head", "validateTrainingHyperparameters");
    validateParameterGroupPrecision(params.parameter_precision_execution_block, "parameter_precision_execution_block", "validateTrainingHyperparameters");
    validateParameterGroupPrecision(params.parameter_precision_slot_selector, "parameter_precision_slot_selector", "validateTrainingHyperparameters");
    validateExecutionBlockHyperparameters(params);
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

#endif // GRIM_CONFIG_AI_CONFIG_PATHS_HPP_INCLUDED

//======================================================//
// JSON-based Config Loading
// These inline helpers parse the ai_config.json snapshot. Modern nvcc
// (CUDA 12+) compiles them in host code paths, so they are NOT gated
// behind #ifndef __CUDACC__ — Phase1_Startup.cu (a .cu TU) calls them
// directly. This matches the policy used for TrainingHyperparameters
// above (see comment at the top of this file's middle section).
//======================================================//

namespace GRIM {
namespace HyperParameters {

inline HardcodedPattern parseHardcodedHiddenPattern(const std::string& pattern) {
    if (pattern == "disabled") return HardcodedPattern::DISABLED;
    if (pattern == "random_centered") return HardcodedPattern::RANDOM_CENTERED;
    if (pattern == "orthogonal_w277") return HardcodedPattern::ORTHOGONAL_W277;
    if (pattern == "aligned_w277") return HardcodedPattern::ALIGNED_W277;
    if (pattern == "constant_uniform") return HardcodedPattern::CONSTANT_UNIFORM;
    if (pattern == "zero_mean_sine") return HardcodedPattern::ZERO_MEAN_SINE;

    throw std::runtime_error(
        "ai_config.json: training.config.hardcoded_hidden_states_pattern has unknown value '" +
        pattern + "'");
}

inline bool loadTrainingHyperparameters(const GRIM::Config::AiConfigSnapshot& snapshot,
                                        GRIM::Config::TrainingHyperparameters& params) {
    params = GRIM::Config::TrainingHyperparameters{};
    params.current_model_training = snapshot.current_model_training;
    params.current_curriculum = snapshot.current_curriculum;
    params.epochs = snapshot.epochs;
    params.seed = snapshot.seed;
    params.batch_size = snapshot.batch_size;
    params.gradient_accumulation_steps = snapshot.gradient_accumulation_steps;
    params.batch_strategy = snapshot.batch_strategy;
    params.learning_rate = snapshot.learning_rate;
    params.weight_decay = snapshot.weight_decay;
    params.grad_clip_norm = snapshot.grad_clip_norm;
    params.per_token_grad_scale = snapshot.per_token_grad_scale;
    params.force_rebuild_vocab = snapshot.force_rebuild_vocab;
    params.d_model = snapshot.d_model;
    params.num_layers = snapshot.num_layers;
    params.num_heads = snapshot.num_heads;
    params.num_kv_heads = snapshot.num_kv_heads;
    params.max_seq_len = snapshot.max_seq_len;
    params.tie_embeddings = snapshot.tie_embeddings;
    params.dropout_rate = snapshot.dropout_rate;
    params.sliding_window_stride = snapshot.sliding_window_stride;
    params.warmup_fraction = snapshot.warmup_fraction;
    params.cosine_decay_enabled = snapshot.cosine_decay_enabled;
    params.cosine_warm_restarts = snapshot.cosine_warm_restarts;
    params.log_interval = snapshot.log_interval;
    params.atom_stats_interval = snapshot.atom_stats_interval;
    params.atom_stats_max_seqs = snapshot.atom_stats_max_seqs;
    params.validation_interval = snapshot.validation_interval;
    params.checkpoint_interval = snapshot.checkpoint_interval;
    params.use_gpu = snapshot.use_gpu;
    params.use_flash_attention = snapshot.use_flash_attention;

    params.parameter_precision_embedding = parseParameterGroupPrecision(snapshot.parameter_precision_embedding);
    params.parameter_precision_lm_head = parseParameterGroupPrecision(snapshot.parameter_precision_lm_head);
    params.parameter_precision_attention = parseParameterGroupPrecision(snapshot.parameter_precision_attention);
    params.parameter_precision_ffn = parseParameterGroupPrecision(snapshot.parameter_precision_ffn);
    params.parameter_precision_rmsnorm = parseParameterGroupPrecision(snapshot.parameter_precision_rmsnorm);
    params.parameter_precision_scratchblock = parseParameterGroupPrecision(snapshot.parameter_precision_scratchblock);
    params.parameter_precision_mtp = parseParameterGroupPrecision(snapshot.parameter_precision_mtp);
    params.parameter_precision_reasoning_head = parseParameterGroupPrecision(snapshot.parameter_precision_reasoning_head);
    params.parameter_precision_execution_block = parseParameterGroupPrecision(snapshot.parameter_precision_execution_block);
    params.parameter_precision_slot_selector = parseParameterGroupPrecision(snapshot.parameter_precision_slot_selector);

    if (snapshot.use_rope && snapshot.use_alibi) {
        params.positional_encoding = PositionalEncodingType::ALIBI_ROPE;
    } else if (snapshot.use_rope) {
        params.positional_encoding = PositionalEncodingType::ROPE;
    } else if (snapshot.use_alibi) {
        params.positional_encoding = PositionalEncodingType::ALIBI;
    } else {
        throw std::runtime_error("ai_config.json: training.config must enable at least one of use_rope or use_alibi");
    }
    params.rope_base_seq_len = snapshot.rope_base_seq_len;
    params.alibi_min_locality_distance = snapshot.alibi_min_locality_distance;
    params.alibi_slope_exponent = snapshot.alibi_slope_exponent;
    params.alibi_max_bias = snapshot.alibi_max_bias;
    params.rope_theta = snapshot.rope_theta;
    params.rope_scaling = snapshot.rope_scaling;

    params.soft_restart_enabled = snapshot.soft_restart_enabled;
    params.soft_restart_loss_increase_threshold = snapshot.soft_restart_loss_increase_threshold;
    params.soft_restart_max_step_window = snapshot.soft_restart_max_step_window;
    params.soft_restart_cooldown_steps = snapshot.soft_restart_cooldown_steps;
    params.auto_stop_enabled = snapshot.auto_stop_enabled;
    params.auto_stop_plateau_patience = snapshot.auto_stop_plateau_patience;
    params.auto_stop_plateau_min_delta = snapshot.auto_stop_plateau_min_delta;
    params.auto_stop_high_loss_threshold = snapshot.auto_stop_high_loss_threshold;
    params.auto_stop_high_loss_patience = snapshot.auto_stop_high_loss_patience;
    params.single_batch_overfit_enabled = snapshot.single_batch_overfit_enabled;
    params.single_batch_overfit_max_steps = snapshot.single_batch_overfit_max_steps;
    params.shuffle_train_enabled = snapshot.shuffle_train_enabled;
    params.shuffle_train_epochs = snapshot.shuffle_train_epochs;
    if (params.shuffle_train_epochs < 0) {
        throw std::runtime_error("ai_config.json: training.config.shuffle_train_epochs must be >= 0");
    }

    params.telemetry_control_enabled = snapshot.telemetry_control_enabled;
    params.telemetry_spike_mild_threshold = snapshot.telemetry_spike_mild_threshold;
    params.telemetry_spike_moderate_threshold = snapshot.telemetry_spike_moderate_threshold;
    params.telemetry_spike_severe_threshold = snapshot.telemetry_spike_severe_threshold;
    params.telemetry_moderate_grad_scale = snapshot.telemetry_moderate_grad_scale;
    params.telemetry_moderate_cooldown_extension = snapshot.telemetry_moderate_cooldown_extension;
    params.telemetry_min_grad_for_nonzero_loss = snapshot.telemetry_min_grad_for_nonzero_loss;
    params.telemetry_loss_threshold_for_grad_check = snapshot.telemetry_loss_threshold_for_grad_check;
    params.telemetry_max_consecutive_zero_grad_steps = snapshot.telemetry_max_consecutive_zero_grad_steps;
    params.telemetry_seq_len_regime_change_threshold = snapshot.telemetry_seq_len_regime_change_threshold;
    params.telemetry_regime_change_suppression_steps = snapshot.telemetry_regime_change_suppression_steps;
    params.telemetry_volatility_damping_threshold = snapshot.telemetry_volatility_damping_threshold;
    params.telemetry_max_volatility_damping = snapshot.telemetry_max_volatility_damping;
    params.telemetry_gradient_decay_threshold = snapshot.telemetry_gradient_decay_threshold;
    params.telemetry_max_decay_boost = snapshot.telemetry_max_decay_boost;
    params.telemetry_progress_boost_threshold = snapshot.telemetry_progress_boost_threshold;
    params.telemetry_max_progress_boost = snapshot.telemetry_max_progress_boost;
    params.telemetry_outlier_frequency_trigger = snapshot.telemetry_outlier_frequency_trigger;
    params.telemetry_outlier_persistence_trigger = snapshot.telemetry_outlier_persistence_trigger;
    params.telemetry_anchor_drift_sigma_multiplier = snapshot.telemetry_anchor_drift_sigma_multiplier;
    params.telemetry_soft_restart_cooldown_steps = snapshot.telemetry_soft_restart_cooldown_steps;
    params.telemetry_baseline_stabilization_steps = snapshot.telemetry_baseline_stabilization_steps;
    params.telemetry_verbose_logging = snapshot.telemetry_verbose_logging;
    params.telemetry_fail_loud_on_accumulation_bug = snapshot.telemetry_fail_loud_on_accumulation_bug;
    params.telemetry_plateau_noise_enabled = snapshot.telemetry_plateau_noise_enabled;
    params.telemetry_plateau_noise_patience = snapshot.telemetry_plateau_noise_patience;
    params.telemetry_plateau_noise_variance_threshold = snapshot.telemetry_plateau_noise_variance_threshold;
    params.telemetry_plateau_noise_std = snapshot.telemetry_plateau_noise_std;
    params.telemetry_plateau_noise_proportional = snapshot.telemetry_plateau_noise_proportional;
    params.telemetry_plateau_noise_cooldown = snapshot.telemetry_plateau_noise_cooldown;
    params.telemetry_plateau_noise_max_per_epoch = snapshot.telemetry_plateau_noise_max_per_epoch;
    params.telemetry_lattice_num_levels = snapshot.telemetry_lattice_num_levels;
    params.telemetry_lattice_num_streams = snapshot.telemetry_lattice_num_streams;
    params.telemetry_lattice_beta_mu = snapshot.telemetry_lattice_beta_mu;
    params.telemetry_lattice_beta_a = snapshot.telemetry_lattice_beta_a;
    params.telemetry_lattice_beta_delta = snapshot.telemetry_lattice_beta_delta;
    params.telemetry_lattice_beta_r = snapshot.telemetry_lattice_beta_r;
    params.telemetry_lattice_beta_run = snapshot.telemetry_lattice_beta_run;
    params.telemetry_lattice_beta_v = snapshot.telemetry_lattice_beta_v;
    params.telemetry_lattice_k_out0 = snapshot.telemetry_lattice_k_out0;
    params.telemetry_lattice_alpha_v = snapshot.telemetry_lattice_alpha_v;
    params.telemetry_lattice_epsilon = snapshot.telemetry_lattice_epsilon;
    params.telemetry_lattice_strict_mode = snapshot.telemetry_lattice_strict_mode;

    params.tape_logging.default_level = snapshot.logging_default_level;
    params.tape_logging.equation_csv_enabled = snapshot.logging_equation_csv_enabled;
    params.tape_logging.stderr_enabled = snapshot.logging_stderr_enabled;
    params.tape_logging.initial_capacity = snapshot.logging_initial_capacity;
    params.tape_logging.group_overrides = snapshot.logging_group_overrides;
    params.log_recorder.enabled = snapshot.log_recorder_enabled;
    params.log_recorder.default_level = snapshot.log_recorder_default_level;
    params.log_recorder.modules = snapshot.log_recorder_modules;
    params.log_recorder.layers.embedding = snapshot.log_recorder_layer_embedding;
    params.log_recorder.layers.rms_norm = snapshot.log_recorder_layer_rms_norm;
    params.log_recorder.layers.attention = snapshot.log_recorder_layer_attention;
    params.log_recorder.layers.feed_forward = snapshot.log_recorder_layer_feed_forward;
    params.log_recorder.layers.residual = snapshot.log_recorder_layer_residual;
    params.log_recorder.layers.encoding = snapshot.log_recorder_layer_encoding;
    params.log_recorder.layers.serialization = snapshot.log_recorder_layer_serialization;
    params.log_recorder.layers.execution_block = snapshot.log_recorder_layer_execution_block;

    params.loss_label_smoothing_enabled = snapshot.loss_label_smoothing_enabled;
    params.loss_label_smoothing_epsilon = snapshot.loss_label_smoothing_epsilon;
    params.loss_focal_enabled = snapshot.loss_focal_enabled;
    params.loss_focal_gamma = snapshot.loss_focal_gamma;
    params.loss_focal_alpha = snapshot.loss_focal_alpha;
    params.loss_entropy_reg_enabled = snapshot.loss_entropy_reg_enabled;
    params.loss_entropy_reg_lambda = snapshot.loss_entropy_reg_lambda;
    params.loss_class_balanced_enabled = snapshot.loss_class_balanced_enabled;
    params.loss_class_balanced_beta = snapshot.loss_class_balanced_beta;
    params.loss_preference_enabled = snapshot.loss_preference_enabled;
    params.loss_preference_beta = snapshot.loss_preference_beta;
    params.loss_distillation_enabled = snapshot.loss_distillation_enabled;
    params.loss_distillation_temperature = snapshot.loss_distillation_temperature;
    params.loss_distillation_lambda = snapshot.loss_distillation_lambda;
    params.loss_masking_enabled = snapshot.loss_masking_enabled;
    params.loss_masking_tag = snapshot.loss_masking_tag;
    params.lm_head_centering_enabled = snapshot.lm_head_centering_enabled;
    params.lm_head_center_hidden_states = snapshot.lm_head_center_hidden_states;
    params.freeze_learned_rms_gammas = snapshot.freeze_learned_rms_gammas;
    params.center_logits = snapshot.center_logits;
    params.center_encoder_residuals = snapshot.center_encoder_residuals;
    params.project_out_pc1 = snapshot.project_out_pc1;
    params.pc1_power_iters = snapshot.pc1_power_iters;
    params.use_layer_scale = snapshot.use_layer_scale;
    params.layer_scale_init = snapshot.layer_scale_init;
    params.qk_norm_enabled = snapshot.qk_norm_enabled;
    const HardcodedPattern parsed_pattern = parseHardcodedHiddenPattern(snapshot.hardcoded_hidden_states_pattern);
    params.hardcoded_hidden_pattern = snapshot.hardcoded_hidden_states_enabled
        ? parsed_pattern
        : HardcodedPattern::DISABLED;
    params.hardcoded_log_every_n_batches = snapshot.hardcoded_log_every_n_batches;
    params.embedding_freeze_enabled = snapshot.embedding_freeze_enabled;
    params.embedding_freeze_after_step = snapshot.embedding_freeze_after_step;
    params.optimizer_kind = snapshot.optimizer_kind;
    params.optimizer_beta1 = snapshot.optimizer_beta1;
    params.optimizer_beta2 = snapshot.optimizer_beta2;
    params.optimizer_epsilon = snapshot.optimizer_epsilon;
    if (params.optimizer_kind != "adamw" && params.optimizer_kind != "radamw") {
        throw std::runtime_error(
            "[ai_config] training.config.optimizer_kind must be \"adamw\" or \"radamw\", got \"" +
            params.optimizer_kind + "\"");
    }

    params.stability_overrides_enabled = snapshot.stability_overrides_enabled;
    params.stability_override_batch_size = snapshot.stability_override_batch_size;
    params.stability_override_max_seq_len = snapshot.stability_override_max_seq_len;
    params.stability_override_clip_per_token = snapshot.stability_override_clip_per_token;
    params.scratch_blocks_enabled = snapshot.scratch_blocks_enabled;
    params.scratch_num_blocks = snapshot.scratch_num_blocks;
    params.scratch_write_combined = snapshot.scratch_write_combined;
    params.use_scratch_block = snapshot.use_scratch_block;
    params.scratch_block_atom_embedding_dim = snapshot.scratch_block_atom_embedding_dim;
    params.scratch_block_max_atoms = snapshot.scratch_block_max_atoms;
    params.scratch_block_atom_scale = snapshot.scratch_block_atom_scale;
    params.execution_block_enabled = snapshot.execution_block_enabled;
    params.scratch_block_execution_first_type_only = snapshot.scratch_block_execution_first_type_only;
    params.execution_block_debug_mode = snapshot.execution_block_debug_mode;
    params.step_y_overrides_x = snapshot.step_y_overrides_x;
    params.structured_ce_enabled = snapshot.structured_ce_enabled;
    params.selector_enabled = snapshot.selector_enabled;
    params.execution_block_layer = snapshot.execution_block_layer;
    params.execution_block_num_ops = snapshot.execution_block_num_ops;
    params.execution_block_num_slots = snapshot.execution_block_num_slots;
    params.execution_block_num_scratch_slots = snapshot.execution_block_num_scratch_slots;
    params.execution_block_num_steps = snapshot.execution_block_num_steps;
    params.execution_block_value_decode_input_dim = snapshot.execution_block_value_decode_input_dim;
    params.execution_block_value_decode_hidden_dim = snapshot.execution_block_value_decode_hidden_dim;
    params.execution_block_d_type = snapshot.execution_block_d_type;
    params.execution_block_cross_attn_topk = snapshot.execution_block_cross_attn_topk;
    params.execution_block_result_slot_mode = snapshot.execution_block_result_slot_mode;
    params.execution_block_result_slot_index = snapshot.execution_block_result_slot_index;
    params.execution_block_temp_schedule = snapshot.execution_block_temp_schedule;
    params.decode_time_slot_feature_dim = snapshot.decode_time_slot_feature_dim;
    params.selector_d_selector = snapshot.selector_d_selector;
    params.execution_block_usage_decay = snapshot.execution_block_usage_decay;
    params.execution_block_inject_gate_temp = snapshot.execution_block_inject_gate_temp;
    params.execution_block_entropy_collapse_threshold = snapshot.execution_block_entropy_collapse_threshold;
    params.execution_block_write_collapse_threshold = snapshot.execution_block_write_collapse_threshold;
    params.execution_block_magnitude_limit = snapshot.execution_block_magnitude_limit;
    params.execution_block_diversity_kappa = snapshot.execution_block_diversity_kappa;
    params.execution_block_temp_start = snapshot.execution_block_temp_start;
    params.execution_block_temp_end = snapshot.execution_block_temp_end;
    params.execution_block_entropy_weight = snapshot.execution_block_entropy_weight;
    params.step_x_multiplier = snapshot.step_x_multiplier;
    params.step_y_multiplier = snapshot.step_y_multiplier;
    params.entropy_aux_weight = snapshot.entropy_aux_weight;
    params.value_match_epsilon = snapshot.value_match_epsilon;
    params.final_slot_consistency_weight = snapshot.final_slot_consistency_weight;
    params.execution_block_transition_hard_threshold = snapshot.execution_block_transition_hard_threshold;
    params.execution_block_causal_w1_transition = snapshot.execution_block_causal_w1_transition;
    params.div_invalid_penalty_weight = snapshot.div_invalid_penalty_weight;
    params.div_magnitude_penalty_weight = snapshot.div_magnitude_penalty_weight;
    params.arg_reinforce_weight = snapshot.arg_reinforce_weight;
    params.arg_reinforce_baseline_decay = snapshot.arg_reinforce_baseline_decay;
    params.structured_ce_weight = snapshot.structured_ce_weight;
    params.selector_selection_margin = snapshot.selector_selection_margin;
    params.selector_supervision_weight = snapshot.selector_supervision_weight;
    params.single_stream_mode = snapshot.single_stream_mode;
    params.disable_async_frees = snapshot.disable_async_frees;
    params.synchronize_after_kernels = snapshot.synchronize_after_kernels;
    params.mtp_enabled = snapshot.mtp_enabled;
    params.mtp_log_ratio_monitor = snapshot.mtp_log_ratio_monitor;
    params.mtp_k = snapshot.mtp_k;
    params.mtp_alpha = snapshot.mtp_alpha;
    params.prediction_comparison_enabled = snapshot.prediction_comparison_enabled;
    params.prediction_comparison_interval = snapshot.prediction_comparison_interval;
    params.prediction_comparison_top_k = snapshot.prediction_comparison_top_k;
    params.prediction_comparison_max_positions = snapshot.prediction_comparison_max_positions;
    params.prediction_comparison_log_path = snapshot.prediction_comparison_log_path;
    params.logit_update_trace_enabled = snapshot.logit_update_trace_enabled;
    params.logit_update_trace_interval = snapshot.logit_update_trace_interval;
    params.attention_diag_enabled = snapshot.attention_diag_enabled;
    params.attention_diag_layer = snapshot.attention_diag_layer;
    params.attention_diag_head = snapshot.attention_diag_head;
    params.tokenizer_enable_scratch_block_reasoning = snapshot.tokenizer_enable_scratch_block_reasoning;
    params.tokenizer_detect_numbers = snapshot.tokenizer_detect_numbers;

    deriveComputedTrainingHyperparameters(params);
    return true;
}

/**
 * @brief Populate model architecture values from ai_config.json
 * 
 * This is THE authoritative function for getting model architecture.
 * All other code should call this rather than using hardcoded values.
 * 
 * Rule 20: every architecture field is authored by ai_config.json
 * training.config. There is no compile-time fallback path.
 * 
 * Note: vocab_size is NOT loaded here - it comes from the .grmt training data file
 * 
 * @param arch Output ModelArchitecture struct to populate
 * @return true if config was loaded, false if using defaults only
 */
// Snapshot-based overload — preferred entry point. Avoids re-reading
// ai_config.json when the caller already loaded a snapshot (e.g. Phase1).
// All architecture fields come from flat AiConfigSnapshot leaves; typed
// TrainingHyperparameters derivation is owned by HyperParameters_GPU.hpp.
inline bool loadModelArchitecture(const GRIM::Config::AiConfigSnapshot& snapshot,
                                  ModelArchitecture& arch) {
    GRIM::Config::TrainingHyperparameters hp;
    loadTrainingHyperparameters(snapshot, hp);
    arch.loadFromTrainingHyperparameters(hp);
    return true;
}

// Convenience overload — loads a snapshot internally for callers that do not
// already hold the raw authored snapshot.
inline bool loadModelArchitecture(ModelArchitecture& arch) {
    auto snapshot = GRIM::Config::loadAiConfigSnapshot();
    if (!snapshot) {
        throw std::runtime_error("loadModelArchitecture: loadAiConfigSnapshot returned no snapshot");
    }
    return loadModelArchitecture(*snapshot, arch);
}

/**
 * @brief Get model architecture with single call (convenience function)
 * 
 * Returns a fully populated ModelArchitecture struct.
 * Uses config if available, falls back to defaults.
 */
inline ModelArchitecture getModelArchitecture() {
    ModelArchitecture arch;
    loadModelArchitecture(arch);
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

//======================================================//
// Generation config loader (snapshot-owned typed handoff)
//======================================================//
inline SamplingStrategy parseGenerationSamplingStrategy(const std::string& strategy) {
    if (strategy == "greedy") return SamplingStrategy::GREEDY;
    if (strategy == "top_k") return SamplingStrategy::TOP_K;
    if (strategy == "top_p") return SamplingStrategy::TOP_P;
    if (strategy == "min_p") return SamplingStrategy::MIN_P;
    if (strategy == "typical") return SamplingStrategy::TYPICAL;
    if (strategy == "top_k_top_p") return SamplingStrategy::TOP_K_TOP_P;

    throw std::runtime_error(
        "ai_config.json: training.config.generation_strategy has unknown value '" + strategy + "'");
}

inline bool loadGenerationConfig(const GRIM::Config::AiConfigSnapshot& snapshot,
                                 GenerationConfig& generation) {
    generation.strategy = parseGenerationSamplingStrategy(snapshot.generation_strategy);
    generation.max_new_tokens = snapshot.generation_max_new_tokens;
    generation.min_new_tokens = snapshot.generation_min_new_tokens;
    generation.top_k = snapshot.generation_top_k;
    generation.repetition_penalty_window = snapshot.generation_repetition_penalty_window;
    generation.no_repeat_ngram_size = snapshot.generation_no_repeat_ngram_size;
    generation.temperature = snapshot.generation_temperature;
    generation.top_p = snapshot.generation_top_p;
    generation.min_p = snapshot.generation_min_p;
    generation.typical_p = snapshot.generation_typical_p;
    generation.repetition_penalty = snapshot.generation_repetition_penalty;
    generation.frequency_penalty = snapshot.generation_frequency_penalty;
    generation.presence_penalty = snapshot.generation_presence_penalty;
    generation.do_sample = snapshot.generation_do_sample;
    generation.enable_scratchblock_reasoning = snapshot.generation_enable_scratchblock_reasoning;
    return true;
}

inline bool loadGenerationConfig(GenerationConfig& generation) {
    auto snapshot = GRIM::Config::loadAiConfigSnapshot();
    if (!snapshot) {
        throw std::runtime_error("loadGenerationConfig: loadAiConfigSnapshot returned no snapshot");
    }
    return loadGenerationConfig(*snapshot, generation);
}

//======================================================//
// Startup configuration — single owner.
//
// PathConfig + StartupConfig + loadStartupConfig() live here, NOT in
// the training/Phases/ layer. The earlier design scattered:
//   - JSON parsing                  → Phase1_Startup.cu
//   - architecture validation       → ModelArchitecture::validate()
//   - hyperparameter validation     → validateTrainingHyperparameters()
//   - max_seq_len / stride / batch  → Phase1_Startup.cu (manual)
//   - CLI overrides                 → Phase1_Startup.cu
// across multiple files, each independently re-asking the same
// HyperParameters_GPU helpers. By owning the entire load + derive +
// validate pipeline here, every consumer (training, server, future
// tools) gets one fully-validated `StartupConfig` from a single call.
//
// Side effects intentionally NOT performed here (caller does them):
//   - GRIM::Logging::InitLogRecorder()          → training log subsystem
// Keeping those out preserves the Shared/HyperParameters → Training
// layering rule (HyperParameters does not depend on training/).
//======================================================//

struct PathConfig {
    std::string data_path;
    std::string vocab_path;
    std::string output_model_path;
    std::string checkpoint_dir;
    std::string log_dir;
    std::string status_path;
    std::filesystem::path config_path;

    bool validate() const {
        return !data_path.empty() &&
               !vocab_path.empty() &&
               !output_model_path.empty() &&
               !checkpoint_dir.empty() &&
               !log_dir.empty();
    }
};

struct StartupConfig {
    PathConfig paths;
    // Single raw upload point for authored ai_config.json values. GRIM-text
    // startup may slice model/training/tokenizer/subprocess leaves from this
    // snapshot, but must not treat GRIM process-owned data_collection leaves as
    // training/inference startup config.
    GRIM::Config::AiConfigSnapshot ai_config_snapshot;
    GRIM::Config::TrainingHyperparameters hyperparameters;

    // Resolved startup values (populated by loadStartupConfig)
    int max_seq_len = 0;
    int sliding_window_stride = 0;

    // CLI flags
    bool save_test_mode = false;
};

/**
 * @brief Single entry point for startup configuration loading.
 *
 * Performs, in order:
 *   1. loadAiConfigSnapshot()       — single raw load + raw validation
 *   2. Copy paths from snapshot->grim_paths into PathConfig
 *   3. Store the raw snapshot and copy hyperparameters from it
 *   4. loadModelArchitecture()      — snapshot typed surface → ModelArchitecture
 *   5. Resolve max_seq_len (stability_override vs hyperparameters.max_seq_len)
 *   6. Resolve sliding_window_stride from training.config and validate it against effective max_seq_len
 *   7. Apply stability overrides to batch_size + grad_clip_norm
 *   8. Apply CLI overrides (--data, --vocab, --output, --epochs,
 *      --batch-size, --lr, --save-test, --help)
 *   9. validateTrainingHyperparameters() + ModelArchitecture::validate()
 *
 * Throws std::runtime_error on any failure (fail-loud, no fallbacks).
 */
inline StartupConfig loadStartupConfig(int argc, char** argv) {
    StartupConfig config;

    // 1. Load typed snapshot (single JSON parse for the entire program)
    auto snapshot = GRIM::Config::loadAiConfigSnapshot();
    if (!snapshot) {
        throw std::runtime_error("FATAL: ai_config.json not found or unreadable");
    }

    // 2. Paths
    if (!snapshot->hasRequiredGrimTextPaths()) {
        throw std::runtime_error(
            "FATAL: ai_config.json paths.grim_text missing required fields "
            "(at minimum vocab + training_data must be non-empty)");
    }
    config.paths.config_path       = snapshot->config_path;
    config.paths.data_path         = snapshot->grim_text_training_data;
    config.paths.vocab_path        = snapshot->grim_text_vocab;
    config.paths.output_model_path = snapshot->grim_text_model;
    config.paths.checkpoint_dir    = snapshot->grim_text_checkpoints;
    config.paths.log_dir           = snapshot->grim_text_logs;
    config.paths.status_path       = snapshot->grim_text_training_status;

    // 3. Retain the single raw snapshot and derive mutable startup hyperparameters.
    config.ai_config_snapshot = *snapshot;
    loadTrainingHyperparameters(config.ai_config_snapshot, config.hyperparameters);

    // 4. Model config validation — single source of truth for JSON keys.
    ModelArchitecture architecture_check;
    loadModelArchitecture(config.ai_config_snapshot, architecture_check);

    // 5. Derive max_seq_len (must be configured)
    {
        const auto& hp = config.hyperparameters;
        if (hp.stability_overrides_enabled && hp.stability_override_max_seq_len > 0) {
            config.max_seq_len = hp.stability_override_max_seq_len;
        } else if (hp.max_seq_len > 0) {
            config.max_seq_len = hp.max_seq_len;
        } else {
            throw std::runtime_error(
                "FATAL: max_seq_len not configured in ai_config.json "
                "(stability or hyperparameters)");
        }
    }

    // 7. Resolve configured stride against the effective max_seq_len.
    if (config.hyperparameters.sliding_window_stride <= 0) {
        throw std::runtime_error(
            "FATAL: sliding_window_stride must be > 0 in ai_config.json, got " +
            std::to_string(config.hyperparameters.sliding_window_stride));
    }
    config.sliding_window_stride = config.hyperparameters.sliding_window_stride;
    if (config.sliding_window_stride > config.max_seq_len) {
        throw std::runtime_error(
            "FATAL: sliding_window_stride=" +
            std::to_string(config.sliding_window_stride) +
            " exceeds effective max_seq_len=" +
            std::to_string(config.max_seq_len) +
            ". Update training.config.sliding_window_stride or stability_overrides.max_seq_len.");
    }

    // 8. Stability overrides (batch_size, grad_clip_norm)
    {
        const auto& hp = config.hyperparameters;
        if (hp.stability_overrides_enabled) {
            if (hp.stability_override_batch_size <= 0) {
                throw std::runtime_error(
                    "FATAL: stability_overrides enabled but "
                    "stability_override_batch_size=" +
                    std::to_string(hp.stability_override_batch_size) +
                    " (must be > 0)");
            }
            config.hyperparameters.batch_size = hp.stability_override_batch_size;
            if (hp.stability_override_clip_per_token > 0.0f) {
                config.hyperparameters.grad_clip_norm =
                    hp.stability_override_clip_per_token;
            }
        }
    }

    // 8. CLI overrides
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            config.paths.data_path = argv[++i];
        } else if (arg == "--vocab" && i + 1 < argc) {
            config.paths.vocab_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            config.paths.output_model_path = argv[++i];
        } else if (arg == "--epochs" && i + 1 < argc) {
            config.hyperparameters.epochs = std::atoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            config.hyperparameters.batch_size = std::atoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            config.hyperparameters.learning_rate =
                static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--save-test") {
            config.save_test_mode = true;
        } else if (arg == "--config") {
            throw std::runtime_error(
                "loadStartupConfig: --config is no longer supported; use the canonical ai_config.json");
        } else if (arg == "--help") {
            std::cout
                << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  --data <path>      Training data path\n"
                << "  --vocab <path>     Vocabulary path\n"
                << "  --output <path>    Output model path\n"
                << "  --epochs <n>       Number of epochs\n"
                << "  --batch-size <n>   Batch size\n"
                << "  --lr <rate>        Learning rate\n"
                << "  --save-test        Test serialization save and exit\n";
            std::exit(0);
        }
    }

    // 9. Final validation — fail loud if anything is inconsistent
    validateTrainingHyperparameters(config.hyperparameters);

    return config;
}

} // namespace HyperParameters
} // namespace GRIM

#endif // GRIM_SHARED_HYPERPARAMETERS_GPU_HPP
