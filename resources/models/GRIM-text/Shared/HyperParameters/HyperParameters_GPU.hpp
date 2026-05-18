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
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
constexpr int TELEMETRY_MAX_STREAMS = 55;         // TelemetryLattice metric streams (0-46 dynamic, 47 logit-scale, 48-54 init invariants)

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
constexpr int ROPE_BASE_SEQ_LEN = 2048;       // Base context length for NTK-aware RoPE scaling
constexpr float ALIBI_SLOPE_EXPONENT = -8.0f; // Controls ALiBi slope decay across heads
constexpr int ALIBI_MIN_LOCALITY_DISTANCE = 16; // Minimum locality distance for ALiBi slope calibration

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
constexpr bool DEFAULT_LOSS_ENTROPY_REG_ENABLED = false;
constexpr float DEFAULT_LOSS_ENTROPY_REG_LAMBDA = 0.0f;  // Regularization strength when enabled (try 0.1-1.0)

// Class-balanced loss: w_v = 1/freq(v)^β reweights per-token loss
// β=0.5 → sqrt inverse frequency. Counteracts frequency-biased CE gradient.
constexpr bool DEFAULT_LOSS_CLASS_BALANCED_ENABLED = false;
constexpr float DEFAULT_LOSS_CLASS_BALANCED_BETA = 0.5f;  // sqrt inverse frequency (try 0.3-0.9)

//======================================================//
// Scratch Block Configuration (ScratchBlock Reasoning Layer)
// Pool block size is computed from the RunCapacity-authored max token rectangle
// in InitTrainingState.
//======================================================//
constexpr bool DEFAULT_SCRATCH_BLOCKS_ENABLED = true;
constexpr size_t DEFAULT_SCRATCH_NUM_BLOCKS = 2;  // Double buffer for pinned memory staging
constexpr bool DEFAULT_SCRATCH_WRITE_COMBINED = false;

//======================================================//
// ExecutionBlock (training.config.execution_block in ai_config.json)
//
// Parsed into GRIM::Config::TrainingHyperparameters (ai_config_paths.hpp).
// Training startup maps those fields onto GRIM::HyperParameters::LanguageModelConfig in Phase1_Startup.cu.
// LanguageModelConfig inherits from ModelArchitecture — architecture fields
// are defined ONLY here (Single Source of Truth). The encoder consumes
// LanguageModelConfig directly; there is no intermediate EncoderConfig.
//======================================================//
constexpr int DECODE_TIME_SLOT_FEATURE_DIM = 5;
constexpr int EXECUTION_BLOCK_NUM_SCRATCH_SLOTS = 0;
constexpr int EXECUTION_BLOCK_VALUE_DECODE_INPUT_DIM = 24;
constexpr int EXECUTION_BLOCK_VALUE_DECODE_HIDDEN_DIM = 16;
constexpr float EXECUTION_BLOCK_INJECT_GATE_TEMP = 0.5f;
constexpr int EXECUTION_BLOCK_RESULT_SLOT_MODE = 0;
constexpr int EXECUTION_BLOCK_RESULT_SLOT_INDEX = -1;
constexpr bool EXECUTION_BLOCK_DEBUG_MODE = true;
constexpr float EXECUTION_BLOCK_ENTROPY_COLLAPSE_THRESHOLD = 0.01f;
constexpr float EXECUTION_BLOCK_WRITE_COLLAPSE_THRESHOLD = 0.98f;
constexpr float EXECUTION_BLOCK_MAGNITUDE_LIMIT = 1e6f;

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

    // Flash Attention (architectural choice — affects kernel selection)
    bool use_flash_attention = true;
    int min_seq_len_for_flash = 0;     // Derived: max_seq_len / 4 (see deriveComputedHyperparameters)

    // Runtime device flag (GPU-only build forces this true; kept for plumbing)
    bool use_gpu = true;
    
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

    // ScratchBlock reasoning layer config
    bool use_scratch_block = true;
    int scratch_block_atom_embedding_dim = 64;
    int scratch_block_max_atoms = 256;
    float scratch_block_atom_scale = 1.0f;
    bool scratch_block_execution_first_type_only = true;

    // ReasoningHead config
    bool reasoning_head_enabled = false;
    int reasoning_num_ops = 8;

    // ExecutionBlock config — differentiable register machine
    bool execution_block_enabled = false;
    int execution_block_layer = -1;
    int execution_block_num_ops = 4;
    int execution_block_num_slots = 4;
    int execution_block_num_steps = 2;
    int execution_block_d_key = 64;
    int execution_block_d_type = 8;
    int execution_block_cross_attn_head_dim = 64;
    int execution_block_cross_attn_topk = 1;
    float execution_block_usage_decay = 0.9f;
    float execution_block_diversity_kappa = 2.0f;
    float execution_block_temp_start = 2.0f;
    float execution_block_temp_end = 0.5f;
    int   execution_block_temp_schedule = 0;
    float execution_block_entropy_weight = 0.01f;

    // Causal state loss weights (Fixes 1-9)
    float execution_block_transition_hard_threshold = 0.0f;
    int   execution_block_gate_warmup_steps = 0;
    float execution_block_causal_w1_transition = 1.0f;

    float div_invalid_penalty_weight = 0.0f;
    float div_magnitude_penalty_weight = 0.0f;

    float arg_reinforce_weight = 0.0f;
    float arg_reinforce_baseline_decay = 0.99f;

    bool  structured_ce_enabled = false;
    float structured_ce_weight  = 0.0f;

    // Decode-time slot selector config
    bool  selector_enabled = false;
    int   selector_d_selector = 64;
    float selector_selection_margin = 1.0f;
    float selector_supervision_weight = 0.0f;

    // Execution-first structured CE loss config (Step X / Y multipliers)
    float step_x_multiplier = 2.0f;
    float step_y_multiplier = 2.0f;
    bool  step_y_overrides_x = false;
    float entropy_aux_weight = 0.0f;
    float value_match_epsilon = 1e-6f;
    float final_slot_consistency_weight = 0.0f;

    // LM Head centering config (Issue #37 / #40 fixes)
    bool lm_head_center_hidden_states = false;
    bool lm_head_freeze_final_rms_gamma = false;
    bool project_out_pc1 = false;
    int  pc1_power_iters = 5;
    bool center_logits = false;
    bool center_encoder_residuals = true;

    // Hardcoded Hidden States Diagnostic (Issue #42)
    enum class HardcodedPattern {
        DISABLED,
        RANDOM_CENTERED,
        ORTHOGONAL_W277,
        ALIGNED_W277,
        CONSTANT_UNIFORM,
        ZERO_MEAN_SINE
    };
    HardcodedPattern hardcoded_hidden_pattern = HardcodedPattern::DISABLED;
    int hardcoded_log_every_n_batches = 1;

    GenerationConfig generation;

    // Multi-token prediction (MTP) - auxiliary heads (Gloeckle et al. 2024)
    bool mtp_enabled = false;
    int mtp_k = 0;
    float mtp_alpha = 0.2f;
    int mtp_alpha_warmup_steps = 500;
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
    
    // Full model configuration (architecture + feature toggles). Single source
    // of truth — populated by loadModelArchitecture() (base ModelArchitecture
    // fields) plus the various model-feature parsers in ai_config_paths.hpp
    // (ScratchBlock, ExecutionBlock, LM-head centering, LayerScale, QK-norm,
    // hardcoded-hidden, MTP). Sliced into model_config in Phase1_Startup.
    //
    // Field name kept as `architecture` for backward compatibility with the
    // many call sites that read hp.architecture.<arch_field>.
    GRIM::HyperParameters::LanguageModelConfig architecture;

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
    bool force_rebuild_vocab;
    bool cosine_decay_enabled;
    bool cosine_warm_restarts;
    float cosine_decay_min_lr;
    // max_seq_len, use_gpu, use_flash_attention, min_seq_len_for_flash now live in `architecture` (Phase 3b)
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
    bool loss_class_balanced_enabled = false;
    float loss_class_balanced_beta = 0.5f;


    // LM Head centering master toggle (training-only, no LMC counterpart).
    // The per-knob fields (center_hidden_states / freeze_final_rms_gamma /
    // center_logits / center_encoder_residuals / project_out_pc1 /
    // pc1_power_iters) live in `architecture` (LanguageModelConfig).
    bool lm_head_centering_enabled;

    // LayerScale, QK-norm, hardcoded-hidden-states fields live in `architecture`.
    
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
    
    // ScratchBlock reasoning fields (use_scratch_block / atom_embedding_dim /
    // max_atoms / atom_scale / execution_first_type_only) live in `architecture`.

    // ExecutionBlock + execution-first loss fields (execution_block_*, step_x/y_*,
    // entropy_aux_weight, value_match_epsilon, final_slot_consistency_weight,
    // div_*, arg_*, structured_ce_*, selector_*) all live in `architecture`.

    // CUDA execution mode - NO DEFAULTS
    bool single_stream_mode;
    bool disable_async_frees;
    bool synchronize_after_kernels;

    // Multi-token prediction (MTP) — log-ratio monitor is training-only.
    // The other MTP fields (enabled, k, alpha, alpha_warmup_steps) live in `architecture`.
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

} // namespace Config
} // namespace GRIM

// Sentinel: tells ai_config_paths.hpp that the structs it needs are now in
// scope. Without this, ai_config_paths.hpp #errors out (enforces that it can
// only be entered through this header).
#define GRIM_HP_GPU_DEFINED_TRAINING_STRUCTS 1

// Include the config header (JSON loaders + path utils + TokenizerConfig +
// GrimTextPaths + AiConfigSnapshot). ai_config_paths.hpp does NOT include
// this file back; it asserts via #error that the sentinel above is set, so
// it can only be entered through this header.
#include "../../../../../control/ai_config_paths.hpp"

// Only compile if TrainingHyperparameters is fully defined
// (checked via the marker defined in ai_config_paths.hpp)
#ifdef GRIM_CONFIG_AI_CONFIG_PATHS_HPP_INCLUDED

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
// Snapshot-based overload — preferred entry point. Avoids re-reading
// ai_config.json when the caller already loaded a snapshot (e.g. Phase1).
// All architecture fields come from the snapshot's typed structs +
// training.config block; this is the SINGLE place where JSON keys like
// d_model / num_layers / positional_encoding are translated to the typed
// ModelArchitecture struct.
inline bool loadModelArchitecture(const GRIM::Config::AiConfigSnapshot& snapshot,
                                  ModelArchitecture& arch) {
    // Start with defaults for the fields THIS function owns (other fields like
    // max_seq_len / use_gpu / use_flash_attention are populated upstream by the
    // TrainingHyperparameters JSON loader — do NOT reset them here).
    arch.d_model = DEFAULT_D_MODEL;
    arch.num_layers = DEFAULT_NUM_LAYERS;
    arch.num_heads = DEFAULT_NUM_HEADS;
    arch.d_ff = DEFAULT_D_MODEL * DEFAULT_D_FF_MULTIPLIER;
    arch.dropout_rate = DEFAULT_DROPOUT_RATE;
    arch.attention_dropout = DEFAULT_DROPOUT_RATE;       // = dropout_rate

    if (!snapshot.has_training) {
        arch.validate();
        return false;
    }

    const auto& doc = snapshot.document;
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

        // Issue #142: Parse positional encoding. Used by both training (Phase1)
        // and runtime inference/server startup; semantics must match exactly.
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

    // max_seq_len is populated by the TrainingHyperparameters JSON loader (Phase 3b);
    // do not source from snapshot.hyperparameters here.

    // Validate and compute derived values (head_dim)
    arch.validate();
    return true;
}

// Path-based overload — loads a snapshot internally. Kept for callers that
// don't already hold a snapshot (e.g. grim_text_server startup).
inline bool loadModelArchitecture(ModelArchitecture& arch, const std::string& configPath = "ai_config.json") {
    auto snapshot = GRIM::Config::loadAiConfigSnapshot(configPath);
    if (!snapshot) {
        // No config available — populate defaults and bail.
        arch.d_model = DEFAULT_D_MODEL;
        arch.num_layers = DEFAULT_NUM_LAYERS;
        arch.num_heads = DEFAULT_NUM_HEADS;
        arch.d_ff = DEFAULT_D_MODEL * DEFAULT_D_FF_MULTIPLIER;
        arch.max_seq_len = DEFAULT_MAX_SEQ_LEN;
        arch.dropout_rate = DEFAULT_DROPOUT_RATE;
        arch.attention_dropout = DEFAULT_DROPOUT_RATE;
        arch.validate();
        return false;
    }
    return loadModelArchitecture(*snapshot, arch);
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

//======================================================//
// Generation config loader (single source of truth for
// translating training.config.generation JSON → GenerationConfig)
//======================================================//
inline bool loadGenerationConfig(const GRIM::Config::AiConfigSnapshot& snapshot,
                                 GenerationConfig& generation) {
    const auto& doc = snapshot.document;
    if (!doc.contains("training") || !doc["training"].contains("config")) {
        return false;
    }
    const auto& cfg = doc["training"]["config"];
    if (!cfg.contains("generation") || !cfg["generation"].is_object()) {
        return false;
    }
    const auto& gen = cfg["generation"];

    if (gen.contains("strategy") && gen["strategy"].is_string()) {
        const std::string strat = gen["strategy"].get<std::string>();
        if (strat == "greedy") {
            generation.strategy = SamplingStrategy::GREEDY;
            generation.do_sample = false;
        } else if (strat == "top_k")       generation.strategy = SamplingStrategy::TOP_K;
        else if (strat == "top_p")       generation.strategy = SamplingStrategy::TOP_P;
        else if (strat == "min_p")       generation.strategy = SamplingStrategy::MIN_P;
        else if (strat == "typical")     generation.strategy = SamplingStrategy::TYPICAL;
        else if (strat == "top_k_top_p") generation.strategy = SamplingStrategy::TOP_K_TOP_P;
        else throw std::runtime_error("loadGenerationConfig: unknown generation.strategy: " + strat);
    }
    if (gen.contains("max_new_tokens") && gen["max_new_tokens"].is_number())
        generation.max_new_tokens = gen["max_new_tokens"].get<int>();
    if (gen.contains("min_new_tokens") && gen["min_new_tokens"].is_number())
        generation.min_new_tokens = gen["min_new_tokens"].get<int>();
    if (gen.contains("temperature") && gen["temperature"].is_number())
        generation.temperature = gen["temperature"].get<float>();
    if (gen.contains("top_k") && gen["top_k"].is_number())
        generation.top_k = gen["top_k"].get<int>();
    if (gen.contains("top_p") && gen["top_p"].is_number())
        generation.top_p = gen["top_p"].get<float>();
    if (gen.contains("min_p") && gen["min_p"].is_number())
        generation.min_p = gen["min_p"].get<float>();
    if (gen.contains("typical_p") && gen["typical_p"].is_number())
        generation.typical_p = gen["typical_p"].get<float>();
    if (gen.contains("repetition_penalty") && gen["repetition_penalty"].is_number())
        generation.repetition_penalty = gen["repetition_penalty"].get<float>();
    if (gen.contains("repetition_penalty_window") && gen["repetition_penalty_window"].is_number())
        generation.repetition_penalty_window = gen["repetition_penalty_window"].get<int>();
    if (gen.contains("frequency_penalty") && gen["frequency_penalty"].is_number())
        generation.frequency_penalty = gen["frequency_penalty"].get<float>();
    if (gen.contains("presence_penalty") && gen["presence_penalty"].is_number())
        generation.presence_penalty = gen["presence_penalty"].get<float>();
    if (gen.contains("no_repeat_ngram_size") && gen["no_repeat_ngram_size"].is_number())
        generation.no_repeat_ngram_size = gen["no_repeat_ngram_size"].get<int>();
    if (gen.contains("do_sample") && gen["do_sample"].is_boolean())
        generation.do_sample = gen["do_sample"].get<bool>();
    if (gen.contains("enable_scratchblock_reasoning") && gen["enable_scratchblock_reasoning"].is_boolean())
        generation.enable_scratchblock_reasoning = gen["enable_scratchblock_reasoning"].get<bool>();
    return true;
}

inline bool loadGenerationConfig(GenerationConfig& generation,
                                 const std::string& configPath = "ai_config.json") {
    auto snapshot = GRIM::Config::loadAiConfigSnapshot(configPath);
    if (!snapshot) return false;
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
//   - generation config translation → loadGenerationConfig()
//   - max_seq_len / stride / batch  → Phase1_Startup.cu (manual)
//   - CLI overrides                 → Phase1_Startup.cu
// across multiple files, each independently re-asking the same
// HyperParameters_GPU helpers. By owning the entire load + derive +
// validate pipeline here, every consumer (training, server, future
// tools) gets one fully-validated `StartupConfig` from a single call.
//
// Side effects intentionally NOT performed here (caller does them):
//   - GRIM::Tokenizer::configureTokenLayout()   → Tokenizer subsystem
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
    GRIM::Config::TrainingHyperparameters hyperparameters;
    GRIM::Config::TokenizerConfig tokenizer_config;
    GenerationConfig generation;

    // Convenience accessor: architecture lives inside hyperparameters now.
    ModelArchitecture& architecture() { return hyperparameters.architecture; }
    const ModelArchitecture& architecture() const { return hyperparameters.architecture; }

    // Derived values (populated by loadStartupConfig)
    int max_seq_len = 0;
    int sliding_window_stride = 0;

    // Set later by training data loader (kept here for ctx.config.* ergonomics)
    uint32_t actual_vocab_size = 0;

    // CLI flags
    bool save_test_mode = false;
};

/**
 * @brief Single entry point for startup configuration loading.
 *
 * Performs, in order:
 *   1. Resolve --config path from argv (default: ai_config.json)
 *   2. loadAiConfigSnapshot()       — typed JSON parse
 *   3. Copy paths from snapshot->grim_paths into PathConfig
 *   4. Copy hyperparameters + tokenizer_config from snapshot
 *   5. loadModelArchitecture()      — JSON → ModelArchitecture
 *   6. loadGenerationConfig()       — JSON → GenerationConfig
 *   7. Resolve max_seq_len (stability_override vs hyperparameters.architecture.max_seq_len)
 *   8. Derive sliding_window_stride = max_seq_len * 7/8
 *   9. Apply stability overrides to batch_size + grad_clip_norm
 *  10. Apply CLI overrides (--data, --vocab, --output, --epochs,
 *      --batch-size, --lr, --save-test, --help)
 *  11. validateTrainingHyperparameters() + ModelArchitecture::validate()
 *
 * Throws std::runtime_error on any failure (fail-loud, no fallbacks).
 */
inline StartupConfig loadStartupConfig(int argc, char** argv) {
    StartupConfig config;

    // 1. Resolve --config (early scan, single pass below for the rest)
    std::string config_path = "ai_config.json";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--config" && i + 1 < argc) {
            config_path = argv[++i];
            break;
        }
    }

    // 2. Load typed snapshot (single JSON parse for the entire program)
    auto snapshot = GRIM::Config::loadAiConfigSnapshot(config_path);
    if (!snapshot) {
        throw std::runtime_error(
            "FATAL: ai_config.json not found or unreadable at: " + config_path);
    }

    // 3. Paths
    if (!snapshot->grim_paths.isValid()) {
        throw std::runtime_error(
            "FATAL: ai_config.json paths.grim_text missing required fields "
            "(at minimum vocab + training_data must be non-empty)");
    }
    config.paths.config_path       = snapshot->config_path;
    const auto& gp                 = snapshot->grim_paths;
    config.paths.data_path         = gp.training_data;
    config.paths.vocab_path        = gp.vocab;
    config.paths.output_model_path = gp.model;
    config.paths.checkpoint_dir    = gp.checkpoints;
    config.paths.log_dir           = gp.logs;
    config.paths.status_path       = gp.training_status;

    // 4. Hyperparameters & tokenizer config
    if (snapshot->has_training) {
        config.hyperparameters = snapshot->hyperparameters;
    }
    config.tokenizer_config = snapshot->tokenizer_config;

    // 5. + 6. Architecture and generation — single source of truth for JSON keys.
    //         max_seq_len/use_gpu/use_flash_attention were populated into
    //         hyperparameters.architecture by the TrainingHyperparameters loader
    //         (Phase 3b); loadModelArchitecture fills the remaining arch fields.
    loadModelArchitecture(*snapshot, config.hyperparameters.architecture);
    loadGenerationConfig(*snapshot, config.generation);

    // 7. Derive max_seq_len (no fallback — must be configured)
    {
        const auto& hp = config.hyperparameters;
        if (hp.stability_overrides_enabled && hp.stability_override_max_seq_len > 0) {
            config.max_seq_len = hp.stability_override_max_seq_len;
        } else if (hp.architecture.max_seq_len > 0) {
            config.max_seq_len = hp.architecture.max_seq_len;
        } else {
            throw std::runtime_error(
                "FATAL: max_seq_len not configured in ai_config.json "
                "(stability or hyperparameters)");
        }
    }

    // 8. Stride: 7/8 of max_seq_len → 12.5% sliding-window overlap
    //    (50% overlap wastes ~30% of GPU compute on masked tokens; 12.5% is
    //    the standard balance between context continuity and token budget.)
    config.sliding_window_stride = std::max(1, config.max_seq_len * 7 / 8);

    // 9. Stability overrides (batch_size, grad_clip_norm)
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

    // 10. CLI overrides
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
        } else if (arg == "--config" && i + 1 < argc) {
            ++i;  // consumed in step 1
        } else if (arg == "--help") {
            std::cout
                << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  --config <path>    Config file (ai_config.json)\n"
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

    // 11. Final validation — fail loud if anything is inconsistent
    validateTrainingHyperparameters(config.hyperparameters);

    return config;
}

} // namespace HyperParameters
} // namespace GRIM

#endif // GRIM_SHARED_HYPERPARAMETERS_GPU_HPP
