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
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "../UnigramByte/Unigram.hpp"
#include "HyperparameterEnums.hpp"

//======================================================//
// HyperParameters_GPU.hpp - typed config owner / computation boundary
// 
// This header owns the first authoritative typed handoff produced from the
// raw ai_config.json snapshot loaded by control/ai_config_paths.hpp.
//
// Structure:
// 1. Core constants (always available, even in CUDA)
// 2. Struct definitions (always available)
// 3. Helper functions that don't need JSON (always available)
//
// RULE: NO other file should define hyperparameters!
// Raw authored values enter through loadAiConfigSnapshot().document; validation
// and formula-derived fields are owned here.
// NO OTHER FILE SHOULD HAVE TRAINING CONFIGURABLE CONSTANTS OR DEFAULTS.
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
// Architecture values themselves are authored in ai_config.json and consumed
// through the HyperParameters snapshot-document finalization boundary. Constants in
// this section may only be formulas or static kernel capabilities, never
// authored model defaults.
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
constexpr float NORMALIZED_CLAMP = 1.0f;          // Clamp for normalized values

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
constexpr int TELEMETRY_MAX_STREAMS = 69;         // TelemetryLattice metric streams (0-60 existing diagnostics, 61-68 raw loss components)

//======================================================//
// UnigramLM Training Constants
// Parameters for vocabulary training/pruning
//======================================================//
constexpr float UNIGRAM_PRUNE_THRESHOLD = 0.0001f;    // Tokens <0.01% usage get pruned
constexpr int UNIGRAM_MIN_VOCAB_SIZE = 8000;          // Don't prune below this
constexpr double UNIGRAM_MIN_COUNT = 1.0;             // Tokens used <1 time get pruned
constexpr size_t UNIGRAM_MAX_SUBWORD_BYTES = 100ULL * 1024 * 1024;  // 100MB limit for subword mining
constexpr size_t UNIGRAM_MAX_SEQUENCE_LENGTH = 4096;  // Static tokenizer workspace floor

//======================================================//
// GELU Activation Constants
// Mathematical constants for GELU approximation
//======================================================//
constexpr float GELU_SQRT_2_OVER_PI = 0.7978845608028654f;  // sqrt(2/pi)
constexpr float GELU_CUBIC_COEFF = 0.044715f;              // Coefficient in GELU approximation

//======================================================//
// AdamW Optimizer Hyperparameters
//======================================================//
constexpr float ADAMW_BETA1 = 0.9f;
constexpr float ADAMW_BETA2 = 0.999f;
constexpr float ADAMW_EPSILON = 1e-8f;
constexpr float ADAMW_WEIGHT_DECAY = 0.01f;

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
constexpr float UPSILON_BASE = 1.0f;              // Base upsilon coefficient
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

inline float computeAttentionSoftmaxScale(int head_dim, const char* caller) {
    if (head_dim <= 0) {
        throw std::runtime_error(std::string(caller) + ": head_dim must be > 0, got " +
                                 std::to_string(head_dim));
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    if (!std::isfinite(scale) || scale <= 0.0f) {
        throw std::runtime_error(std::string(caller) +
                                 ": computed attention_softmax_scale must be finite and > 0, got " +
                                 std::to_string(scale));
    }
    return scale;
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

using GenerationStreamCallback = std::function<void(int token_id, float score)>;

//======================================================//
// Model execution + runtime configuration.
//======================================================//

// Determines memory allocation strategy at model construction time.
enum class ModelExecutionMode {
    TRAINING,    // Full training state with gradient buffers (~1GB+)
    INFERENCE    // Lightweight inference state with only forward caches (~385MB)
};

struct LanguageModelConfig;
inline std::vector<float> computeDerivedPBMAlibiSlopes(
    const LanguageModelConfig& params,
    const char* caller);
inline std::vector<float> computeDerivedPBMRopeInvFreq(
    const LanguageModelConfig& params,
    const char* caller);
inline void populateDerivedPBMTables(
    LanguageModelConfig& params,
    const char* caller);

struct LanguageModelConfig {
    int d_model = 0;
    int num_layers = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int d_ff = 0;
    int max_seq_len = 0;
    float dropout_rate = 0.0f;
    float embedding_scale = 1.0f;
    float attention_dropout = 0.0f;
    float attention_softmax_scale = 0.0f;
    bool tie_embeddings = true;
    PositionalEncodingType positional_encoding = PositionalEncodingType::UNSPECIFIED;

    int rope_base_seq_len = 0;
    int alibi_min_locality_distance = 0;
    float alibi_slope_exponent = 0.0f;
    float alibi_max_bias = std::numeric_limits<float>::quiet_NaN();
    float rope_theta = 0.0f;
    float rope_scaling = 0.0f;
    std::vector<float> pbm_alibi_slopes;
    std::vector<float> pbm_rope_inv_freq;

    bool use_flash_attention = true;
    int min_seq_len_for_flash = 0;
    bool use_gpu = true;

    int head_dim = 0;
    int heads_per_kv_group = 0;
    int kv_dim = 0;
    int qkv_dim = 0;
    int rotary_dim = 0;
    bool is_gqa = false;
    float residual_projection_init_gain = 0.0f;

    int vocab_size = 0;        // Configured model token-space width / tokenizer target vocab size.

    void computeDerivedValues() {
        if (d_model <= 0 || num_heads <= 0 || num_kv_heads <= 0) {
            throw std::runtime_error("LanguageModelConfig::computeDerivedValues: d_model, num_heads, and num_kv_heads must be > 0");
        }
        if ((d_model % num_heads) != 0) {
            throw std::runtime_error(
                "LanguageModelConfig::computeDerivedValues: d_model (" +
                std::to_string(d_model) + ") must be divisible by num_heads (" +
                std::to_string(num_heads) + ")");
        }
        if (!isValidGQAConfig(num_heads, num_kv_heads)) {
            throw std::runtime_error(
                "LanguageModelConfig::computeDerivedValues: invalid GQA config num_heads=" +
                std::to_string(num_heads) + " num_kv_heads=" + std::to_string(num_kv_heads));
        }
        if (num_layers <= 0) {
            throw std::runtime_error(
                "LanguageModelConfig::computeDerivedValues: num_layers must be > 0, got " +
                std::to_string(num_layers));
        }

        head_dim = d_model / num_heads;
        heads_per_kv_group = computeHeadsPerKVGroup(num_heads, num_kv_heads);
        kv_dim = computeKVProjectionSize(d_model, num_heads, num_kv_heads);
        qkv_dim = computeQKVProjectionSize(d_model, num_heads, num_kv_heads);
        rotary_dim = head_dim;
        attention_softmax_scale = computeAttentionSoftmaxScale(
            head_dim, "LanguageModelConfig::computeDerivedValues");
        is_gqa = num_kv_heads < num_heads;
        populateDerivedPBMTables(*this, "LanguageModelConfig::computeDerivedValues");
        if (d_ff <= 0) {
            d_ff = d_model * D_FF_MULTIPLIER;
        }
        residual_projection_init_gain =
            1.0f / std::sqrt(2.0f * static_cast<float>(num_layers));
    }

    // Cache limits
    int max_cached_seq_len = 0;
    int max_tokens_per_batch = 0;  // Optional token budget for training logits/loss

    // Fixed config values (not architecture-dependent)
    float rms_epsilon = 1e-5f;  // RMSNorm epsilon - shared across all RMSNorm layers
    bool causal_mask = true;
    bool use_pre_norm = true;
    bool fuse_qkv = true;
    bool use_simd = true;
    int num_threads = 4;
    bool use_bias = true;
    // Dedicated LM-head output bias initialized to log p(v) (unigram marginal),
    // decoupled from use_bias so it can be enabled WITHOUT the attention/FFN
    // biases (b_o/b2), which are themselves shared-direction injectors. Houses
    // the unigram marginal outside the residual stream to prevent the
    // common-mode (rho) buildup that drives representation collapse.
    bool lm_head_unigram_bias = false;
    bool qk_norm_enabled = false;  // QK-Norm: RMSNorm applied to Q and K before attention scoring
    // Off-by-one attention (softmax1 / zero-value sink, Miller 2023): denominator
    // gains a phantom logit at 0 with a ZERO value vector, so a head may attend to
    // "nothing" and emit ~0 instead of mean-pooling prior tokens. Removes the
    // softmax-sum-to-1 forcing that injects the shared residual common-mode (rho)
    // direction at the first attention layer. Parameterless: applied as an exact
    // post-process of the FlashAttention result, no learnable tensor.
    bool attention_off_by_one = false;

    // LayerScale - per-channel learnable residual scaling vectors [1, d_model]
    bool use_layer_scale = false;
    float layer_scale_init = 1.0f;

    std::string data_path;
    std::string vocab_path;
    std::string output_model_path;
    std::string checkpoint_dir;
    std::string log_dir;
    std::string status_path;
    bool save_test_mode = false;

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
    ParameterGroupPrecision parameter_precision_execution_block = ParameterGroupPrecision::UNSPECIFIED;

    // Atom-data pipeline config (consumed by ExecutionBlock)
    bool use_atom_data = false;
    int atom_embedding_dim = 0;

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

    // Causal state loss weights
    float execution_block_transition_hard_threshold = 0.0f;
    int   execution_block_gate_warmup_steps = 0;
    float execution_block_causal_w1_transition = 0.0f;

    float div_invalid_penalty_weight = 0.0f;
    float div_magnitude_penalty_weight = 0.0f;

    float arg_reinforce_weight = 0.0f;
    float arg_reinforce_baseline_decay = 0.0f;

    bool  structured_ce_enabled = false;
    float structured_ce_weight  = 0.0f;

    // NumberEncoder (numeric-meaning input path) config.
    // Digit-place contribution encoding per docs/ATOM_SELECTOR_IMPLEMENTATION_PLAN.md.
    bool  number_encoder_enabled = false;
    int   number_encoder_max_digit_slots = 0;
    int   number_encoder_d_hidden = 0;
    int   number_encoder_max_abs_pow10 = 0;

    // Execution-first structured CE loss config (Step X / Y multipliers)
    float step_x_multiplier = 0.0f;
    float step_y_multiplier = 0.0f;
    bool  step_y_overrides_x = false;
    float entropy_aux_weight = 0.0f;
    float value_match_epsilon = 0.0f;
    float final_slot_consistency_weight = 0.0f;

    // LM Head / RMSNorm gamma config
    bool lm_head_center_hidden_states = false;
    bool freeze_learned_rms_gammas = false;
    bool project_out_pc1 = false;
    int  pc1_power_iters = 0;
    bool center_logits = false;
    bool center_encoder_residuals = false;

    // LM-head residual SwiGLU adapter (head capacity expansion):
    //   u = z + lm_head_mlp_alpha * SwiGLU_MLP(z), z = RMSNorm(encoder_output)
    bool lm_head_mlp_enabled = false;
    int  lm_head_mlp_d_ff = 0;
    float lm_head_mlp_alpha = 0.0f;

    HardcodedPattern hardcoded_hidden_pattern = HardcodedPattern::DISABLED;
    int hardcoded_log_every_n_batches = 0;

    // Generation / sampling leaves (authored as flat training.config.generation_* fields).
    SamplingStrategy generation_strategy = SamplingStrategy::UNSPECIFIED;
    int generation_max_new_tokens = 0;
    int generation_min_new_tokens = 0;
    float generation_temperature = 0.0f;
    int generation_top_k = 0;
    float generation_top_p = 0.0f;
    float generation_min_p = 0.0f;
    float generation_typical_p = 0.0f;
    float generation_repetition_penalty = 0.0f;
    int generation_repetition_penalty_window = 0;
    float generation_frequency_penalty = 0.0f;
    float generation_presence_penalty = 0.0f;
    int generation_no_repeat_ngram_size = 0;
    bool generation_do_sample = false;
    bool generation_enable_scratchblock_reasoning = false;
    int generation_num_return_sequences = 0;
    int generation_eos_token_id = -1;
    int generation_pad_token_id = -1;
    int generation_bos_token_id = -1;
    int generation_unk_token_id = -1;
    std::vector<int> generation_bad_words_ids;
    /// Token IDs to mask at sampling (e.g. byte-level digit tokens); `<NUM>` must remain unmasked.
    std::vector<int> generation_masked_numeric_literal_ids;
    unsigned int generation_seed = 0;


    // Training run selectors — which model and curriculum to use
    std::string current_model_training;
    std::string current_curriculum;

    // Log recorder/tape logging leaves
    bool log_recorder_enabled = false;
    std::string log_recorder_default_level;
    std::map<std::string, std::string> log_recorder_modules;
    bool log_recorder_layer_embedding = false;
    bool log_recorder_layer_rms_norm = false;
    bool log_recorder_layer_attention = false;
    bool log_recorder_layer_feed_forward = false;
    bool log_recorder_layer_residual = false;
    bool log_recorder_layer_encoding = false;
    bool log_recorder_layer_serialization = false;
    bool log_recorder_layer_execution_block = false;
    std::string logging_default_level;
    bool logging_equation_csv_enabled = false;
    bool logging_stderr_enabled = false;
    std::size_t logging_initial_capacity = 0;
    std::map<std::string, std::string> logging_group_overrides;

    // Core training parameters
    int epochs = 0;
    int64_t seed = 0;
    int batch_size = 0;
    int gradient_accumulation_steps = 0;
    bool single_batch_overfit_enabled = false;
    int single_batch_overfit_max_steps = 0;
    std::string batch_strategy;
    float learning_rate = 0.0f;
    float weight_decay = 0.0f;
    bool use_depth_aware_upsilon = false;
    float grad_clip_norm = 0.0f;
    bool grad_clip_enabled = false;
    float effective_per_token_grad_limit = EPSILON_GRADIENT_CLIP;
    bool per_token_grad_scale = false;
    float warmup_fraction = 0.0f;
    int warmup_steps = 0;
    bool force_rebuild_vocab = false;
    bool cosine_decay_enabled = false;
    bool cosine_warm_restarts = false;
    float cosine_decay_min_lr = 0.0f;
    int sliding_window_stride = 0;
    int min_seq_valid_tokens = 0;
    int log_interval = 0;
    int atom_stats_interval = 0;
    int atom_stats_max_seqs = 0;
    bool inference_diagnostic_enabled = false;
    int inference_diagnostic_interval = 0;
    int validation_interval = 0;
    int checkpoint_interval = 0;

    bool soft_restart_enabled = false;
    float soft_restart_loss_increase_threshold = 0.0f;
    int soft_restart_max_step_window = 0;
    int soft_restart_cooldown_steps = 0;

    bool auto_stop_enabled = false;
    int auto_stop_plateau_patience = 0;
    float auto_stop_plateau_min_delta = 0.0f;
    float auto_stop_high_loss_threshold = 0.0f;
    int auto_stop_high_loss_patience = 0;

    bool shuffle_train_enabled = false;
    int shuffle_train_epochs = 0;

    bool telemetry_control_enabled = false;
    float telemetry_spike_mild_threshold = 0.0f;
    float telemetry_spike_moderate_threshold = 0.0f;
    float telemetry_spike_severe_threshold = 0.0f;
    float telemetry_moderate_grad_scale = 0.0f;
    int telemetry_moderate_cooldown_extension = 0;
    float telemetry_min_grad_for_nonzero_loss = 0.0f;
    float telemetry_loss_threshold_for_grad_check = 0.0f;
    int telemetry_max_consecutive_zero_grad_steps = 0;
    float telemetry_seq_len_regime_change_threshold = 0.0f;
    int telemetry_regime_change_suppression_steps = 0;
    float telemetry_volatility_damping_threshold = 0.0f;
    float telemetry_max_volatility_damping = 0.0f;
    float telemetry_gradient_decay_threshold = 0.0f;
    float telemetry_max_decay_boost = 0.0f;
    float telemetry_progress_boost_threshold = 0.0f;
    float telemetry_max_progress_boost = 0.0f;
    float telemetry_outlier_frequency_trigger = 0.0f;
    float telemetry_outlier_persistence_trigger = 0.0f;
    float telemetry_anchor_drift_sigma_multiplier = 0.0f;
    int telemetry_soft_restart_cooldown_steps = 0;
    int telemetry_warmup_steps = 0;
    int telemetry_baseline_stabilization_steps = 0;
    bool telemetry_verbose_logging = false;
    bool telemetry_fail_loud_on_accumulation_bug = false;
    bool telemetry_plateau_noise_enabled = false;
    int telemetry_plateau_noise_patience = 0;
    float telemetry_plateau_noise_variance_threshold = 0.0f;
    float telemetry_plateau_noise_std = 0.0f;
    bool telemetry_plateau_noise_proportional = false;
    int telemetry_plateau_noise_cooldown = 0;
    int telemetry_plateau_noise_max_per_epoch = 0;

    int telemetry_lattice_num_levels = 0;
    int telemetry_lattice_num_streams = 0;
    float telemetry_lattice_beta_mu = 0.0f;
    float telemetry_lattice_beta_a = 0.0f;
    float telemetry_lattice_beta_delta = 0.0f;
    float telemetry_lattice_beta_r = 0.0f;
    float telemetry_lattice_beta_run = 0.0f;
    float telemetry_lattice_beta_v = 0.0f;
    float telemetry_lattice_k_out0 = 0.0f;
    float telemetry_lattice_alpha_v = 0.0f;
    float telemetry_lattice_epsilon = 0.0f;
    bool telemetry_lattice_strict_mode = false;

    bool loss_label_smoothing_enabled = false;
    float loss_label_smoothing_epsilon = 0.0f;
    bool loss_focal_enabled = false;
    float loss_focal_gamma = 0.0f;
    float loss_focal_alpha = 0.0f;
    bool loss_preference_enabled = false;
    float loss_preference_beta = 0.0f;
    bool loss_distillation_enabled = false;
    float loss_distillation_temperature = 0.0f;
    float loss_distillation_lambda = 0.0f;
    bool loss_masking_enabled = true;
    std::string loss_masking_tag;
    bool loss_entropy_reg_enabled = false;
    float loss_entropy_reg_lambda = 0.0f;
    bool loss_class_balanced_enabled = false;
    float loss_class_balanced_beta = 0.0f;

    bool lm_head_centering_enabled = false;
    bool embedding_freeze_enabled = false;
    int embedding_freeze_after_step = 0;

    OptimizerKind optimizer_kind = OptimizerKind::UNSPECIFIED;
    float optimizer_beta1 = 0.0f;
    float optimizer_beta2 = 0.0f;
    float optimizer_epsilon = 0.0f;
    int optimizer_embedding_freeze_after_step = -1;

    bool stability_overrides_enabled = false;
    int stability_override_batch_size = 0;
    int stability_override_max_seq_len = 0;
    float stability_override_clip_per_token = 0.0f;

    bool scratch_blocks_enabled = false;
    std::size_t scratch_max_tokens_per_block = 0;
    std::size_t scratch_num_blocks = 0;
    bool scratch_write_combined = false;

    bool single_stream_mode = false;
    bool disable_async_frees = false;
    bool synchronize_after_kernels = false;

    bool prediction_comparison_enabled = false;
    int prediction_comparison_interval = 0;
    int prediction_comparison_top_k = 0;
    int prediction_comparison_max_positions = 0;
    std::string prediction_comparison_log_path;

    bool logit_update_trace_enabled = false;
    int logit_update_trace_interval = 0;

    bool attention_diag_enabled = false;
    int attention_diag_layer = 0;
    int attention_diag_head = 0;

    bool tokenizer_enable_atom_reasoning = false;
    bool tokenizer_detect_numbers = false;
    int tokenizer_target_vocab_size = 0;
    int tokenizer_max_vocab_size = 0;
    int tokenizer_max_length = 0;
    float tokenizer_character_coverage = 0.0f;
    int tokenizer_min_cleaned_text_length = 0;
    int tokenizer_min_subword_freq = 0;
    bool tokenizer_prune_during_mining = false;
    bool tokenizer_enable_parallel_subword_mining = false;
    int tokenizer_subword_mining_workers = 0;
    std::size_t tokenizer_subword_mining_max_bytes = 0;
    std::string tokenizer_model_type;
    std::vector<std::string> tokenizer_special_tokens;
    bool tokenizer_add_bos = false;
    bool tokenizer_add_eos = false;
    std::string tokenizer_unk_token;
    std::string tokenizer_pad_token;
    std::string tokenizer_bos_token;
    std::string tokenizer_eos_token;
    bool tokenizer_enable_nfkc_normalization = false;
    bool tokenizer_enable_lowercasing = false;
    bool tokenizer_enable_parallel_tokenization = false;
    int tokenizer_parallel_threshold = 0;
    bool tokenizer_enable_byte_fallback = false;
    uint32_t tokenizer_expected_checksum = 0;
    bool tokenizer_save_text_vocab = false;
    float tokenizer_vocab_score_multiplier = 0.0f;
    bool clear_merged_cache_on_merge = false;
    bool subprocess_tokenizer_only_mode = false;
};

//======================================================//
// Registry validation primitives
//
// Each primitive below validates one rule type against one owning container.
// Callers provide a field registry for that rule type; no primitive reaches
// into unrelated config objects or performs derived-value computation.
//======================================================//

template <typename Container, typename Value>
struct ValidationField {
    const char* name;
    Value Container::* member;
};

template <typename Container, typename Value>
inline ValidationField<Container, Value> validationField(
    const char* name,
    Value Container::* member)
{
    return ValidationField<Container, Value>{name, member};
}

template <typename Value>
inline std::string validationValueToString(const Value& value)
{
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

template <typename Value>
inline bool validationIsFinite(const Value& value)
{
    if constexpr (std::is_floating_point_v<Value>) {
        return std::isfinite(value);
    } else {
        return true;
    }
}

template <typename Value>
inline void throwValidationFieldError(
    const char* caller,
    const char* field,
    const char* rule,
    const Value& value)
{
    throw std::runtime_error(std::string(caller) + ": " + field + " " + rule +
                             ", got " + validationValueToString(value));
}

template <typename Container, typename Value>
inline void validatePositiveFields(
    const Container& container,
    std::initializer_list<ValidationField<Container, Value>> fields,
    const char* caller)
{
    for (const auto& field : fields) {
        const auto& value = container.*(field.member);
        if (!(value > static_cast<Value>(0))) {
            throwValidationFieldError(caller, field.name, "must be > 0", value);
        }
    }
}

template <typename Container, typename Value>
inline void validateNonNegativeFields(
    const Container& container,
    std::initializer_list<ValidationField<Container, Value>> fields,
    const char* caller)
{
    for (const auto& field : fields) {
        const auto& value = container.*(field.member);
        if (value < static_cast<Value>(0)) {
            throwValidationFieldError(caller, field.name, "must be >= 0", value);
        }
    }
}

template <typename Container, typename Value>
inline void validatePositiveFiniteFields(
    const Container& container,
    std::initializer_list<ValidationField<Container, Value>> fields,
    const char* caller)
{
    for (const auto& field : fields) {
        const auto& value = container.*(field.member);
        if (!validationIsFinite(value) || !(value > static_cast<Value>(0))) {
            throwValidationFieldError(caller, field.name,
                                      "must be a positive finite value", value);
        }
    }
}

template <typename Container, typename Value>
inline void validateNonNegativeFiniteFields(
    const Container& container,
    std::initializer_list<ValidationField<Container, Value>> fields,
    const char* caller)
{
    for (const auto& field : fields) {
        const auto& value = container.*(field.member);
        if (!validationIsFinite(value) || value < static_cast<Value>(0)) {
            throwValidationFieldError(caller, field.name,
                                      "must be finite and >= 0", value);
        }
    }
}

template <typename Container, typename Value>
inline void validateFiniteNonZeroFields(
    const Container& container,
    std::initializer_list<ValidationField<Container, Value>> fields,
    const char* caller)
{
    for (const auto& field : fields) {
        const auto& value = container.*(field.member);
        if (!validationIsFinite(value) || value == static_cast<Value>(0)) {
            throwValidationFieldError(caller, field.name,
                                      "must be finite and non-zero", value);
        }
    }
}

template <typename Container, typename Value>
inline void validateClosedUnitIntervalFields(
    const Container& container,
    std::initializer_list<ValidationField<Container, Value>> fields,
    const char* caller)
{
    for (const auto& field : fields) {
        const auto& value = container.*(field.member);
        if (!validationIsFinite(value) || value < static_cast<Value>(0) || value > static_cast<Value>(1)) {
            throwValidationFieldError(caller, field.name,
                                      "must be finite and in [0,1]", value);
        }
    }
}

template <typename Container, typename Value>
inline void validateHalfOpenUnitIntervalFields(
    const Container& container,
    std::initializer_list<ValidationField<Container, Value>> fields,
    const char* caller)
{
    for (const auto& field : fields) {
        const auto& value = container.*(field.member);
        if (!validationIsFinite(value) || value < static_cast<Value>(0) || value >= static_cast<Value>(1)) {
            throwValidationFieldError(caller, field.name,
                                      "must be finite and in [0,1)", value);
        }
    }
}

template <typename Container, typename Value>
inline void validateOpenUnitIntervalFields(
    const Container& container,
    std::initializer_list<ValidationField<Container, Value>> fields,
    const char* caller)
{
    for (const auto& field : fields) {
        const auto& value = container.*(field.member);
        if (!validationIsFinite(value) || value <= static_cast<Value>(0) || value >= static_cast<Value>(1)) {
            throwValidationFieldError(caller, field.name,
                                      "must be finite and in (0,1)", value);
        }
    }
}

template <typename Container, typename Value>
struct ValidationConstantBound {
    const char* name;
    Value Container::* member;
    Value bound;
    const char* bound_name;
};

template <typename Container, typename Value>
inline ValidationConstantBound<Container, Value> validationConstantBound(
    const char* name,
    Value Container::* member,
    Value bound,
    const char* bound_name)
{
    return ValidationConstantBound<Container, Value>{name, member, bound, bound_name};
}

template <typename Container, typename Value>
inline void validateFieldsAtLeast(
    const Container& container,
    std::initializer_list<ValidationConstantBound<Container, Value>> fields,
    const char* caller)
{
    for (const auto& field : fields) {
        const auto& value = container.*(field.member);
        if (value < field.bound) {
            throw std::runtime_error(std::string(caller) + ": " + field.name +
                                     " must be >= " + field.bound_name + ", got " +
                                     validationValueToString(value));
        }
    }
}

template <typename Container, typename Value>
inline void validateFieldsAtMost(
    const Container& container,
    std::initializer_list<ValidationConstantBound<Container, Value>> fields,
    const char* caller)
{
    for (const auto& field : fields) {
        const auto& value = container.*(field.member);
        if (value > field.bound) {
            throw std::runtime_error(std::string(caller) + ": " + field.name +
                                     " must be <= " + field.bound_name + ", got " +
                                     validationValueToString(value));
        }
    }
}

template <typename Container, typename Value>
struct ValidationFieldRelation {
    const char* left_name;
    Value Container::* left;
    const char* right_name;
    Value Container::* right;
};

template <typename Container, typename Value>
inline ValidationFieldRelation<Container, Value> validationFieldRelation(
    const char* left_name,
    Value Container::* left,
    const char* right_name,
    Value Container::* right)
{
    return ValidationFieldRelation<Container, Value>{left_name, left, right_name, right};
}

template <typename Container, typename Value>
inline void validateDivisibleFields(
    const Container& container,
    std::initializer_list<ValidationFieldRelation<Container, Value>> fields,
    const char* caller)
{
    for (const auto& field : fields) {
        const auto& dividend = container.*(field.left);
        const auto& divisor = container.*(field.right);
        if (divisor == static_cast<Value>(0) || dividend % divisor != static_cast<Value>(0)) {
            throw std::runtime_error(std::string(caller) + ": " + field.left_name + "=" +
                                     validationValueToString(dividend) +
                                     " must be divisible by " + field.right_name + "=" +
                                     validationValueToString(divisor));
        }
    }
}

template <typename Container, typename Value>
inline void validateLessThanFields(
    const Container& container,
    std::initializer_list<ValidationFieldRelation<Container, Value>> fields,
    const char* caller)
{
    for (const auto& field : fields) {
        const auto& left = container.*(field.left);
        const auto& right = container.*(field.right);
        if (!(left < right)) {
            throw std::runtime_error(std::string(caller) + ": " + field.left_name + "=" +
                                     validationValueToString(left) +
                                     " must be < " + field.right_name + "=" +
                                     validationValueToString(right));
        }
    }
}

} // namespace HyperParameters
} // namespace GRIM

//======================================================//
// Implementation Section
//
// Functions below consume the single root config object.
// ai_config_paths.hpp is deliberately entered only through this header after
// the typed structs are in scope; do not include ai_config_paths.hpp directly.
//======================================================//

namespace GRIM {
namespace HyperParameters {

inline void deriveComputedLanguageModelConfig(LanguageModelConfig& params);
inline int computeMaxTokensPerBatch(int batch_size, int max_seq_len, const char* caller);
inline void refreshMutableTrainingDerivedValues(LanguageModelConfig& params,
                                                int effective_max_seq_len,
                                                const char* caller);

} // namespace HyperParameters
} // namespace GRIM

// Sentinel: tells ai_config_paths.hpp that the structs it needs are now in
// scope. Without this, ai_config_paths.hpp #errors out (enforces that it can
// only be entered through this header).
#define GRIM_HP_GPU_DEFINED_TRAINING_STRUCTS 1

// Include the config header (raw JSON loader + AiConfigSnapshot document).
// ai_config_paths.hpp does NOT include
// this file back; it asserts via #error that the sentinel above is set, so
// it can only be entered through this header.
#include "../../../../../control/ai_config_paths.hpp"

// Only compile after the typed root config is in scope.
// (checked via the marker defined in ai_config_paths.hpp)
#ifdef GRIM_CONFIG_AI_CONFIG_PATHS_HPP_INCLUDED

namespace GRIM {
namespace HyperParameters {

inline LanguageModelConfig finalizeLanguageModelConfig(
    const nlohmann::json& document,
    int argc,
    char** argv,
    ModelExecutionMode execution_mode);

inline int computeMaxTokensPerBatch(int batch_size, int max_seq_len, const char* caller) {
    if (batch_size <= 0) {
        throw std::runtime_error(std::string(caller) + ": batch_size must be > 0, got " +
                                 std::to_string(batch_size));
    }
    if (max_seq_len <= 0) {
        throw std::runtime_error(std::string(caller) + ": max_seq_len must be > 0, got " +
                                 std::to_string(max_seq_len));
    }
    const std::int64_t token_budget =
        static_cast<std::int64_t>(batch_size) * static_cast<std::int64_t>(max_seq_len);
    if (token_budget > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(caller) + ": max_tokens_per_batch overflow: batch_size=" +
                                 std::to_string(batch_size) + " max_seq_len=" +
                                 std::to_string(max_seq_len) + " product=" +
                                 std::to_string(token_budget) + " exceeds int max");
    }
    return static_cast<int>(token_budget);
}

inline std::vector<float> computeDerivedPBMAlibiSlopes(
    const LanguageModelConfig& params,
    const char* caller)
{
    if (params.num_heads <= 0) {
        throw std::runtime_error(std::string(caller) + ": PBM num_heads must be > 0, got " +
                                 std::to_string(params.num_heads));
    }
    if (params.max_seq_len <= 0) {
        throw std::runtime_error(std::string(caller) + ": PBM max_seq_len must be > 0, got " +
                                 std::to_string(params.max_seq_len));
    }
    if (params.rotary_dim <= 0 || (params.rotary_dim & 1) != 0) {
        throw std::runtime_error(std::string(caller) + ": PBM rotary_dim must be positive and even, got " +
                                 std::to_string(params.rotary_dim));
    }

    const int d_max = params.max_seq_len;
    const int locality_floor = std::min(params.alibi_min_locality_distance, d_max);
    const int d_min = std::max(locality_floor, std::min(params.rotary_dim / 2, d_max));
    const float target_bias = std::abs(params.alibi_slope_exponent);
    if (target_bias == 0.0f) {
        throw std::runtime_error(std::string(caller) + ": alibi_slope_exponent must be non-zero");
    }

    // Strongest head reaches `target_bias` of ALiBi penalty at distance d_min;
    // weakest head reaches it at d_max. The earlier `d_min/d_max` rescale that
    // multiplied both bounds crushed every slope ~30x below standard ALiBi,
    // turning low-slope heads into near-uniform causal mean-pools (the shared
    // direction that drove the L1 rho collapse) — removed.
    const float m_max = target_bias / static_cast<float>(d_min);
    const float m_min = target_bias / static_cast<float>(d_max);
    if (!(m_min <= m_max)) {
        throw std::runtime_error(std::string(caller) + ": computed ALiBi slope range is inverted");
    }
    if (!std::isfinite(params.alibi_max_bias) || params.alibi_max_bias > 0.0f) {
        throw std::runtime_error(std::string(caller) + ": alibi_max_bias must be finite and <= 0, got " +
                                 std::to_string(params.alibi_max_bias));
    }

    std::vector<float> slopes(static_cast<size_t>(params.num_heads));
    const float max_bias_magnitude = std::abs(params.alibi_max_bias);
    const float max_slope_magnitude = (params.alibi_max_bias != 0.0f)
        ? (max_bias_magnitude / static_cast<float>(params.max_seq_len))
        : 0.0f;

    // SIGN CONVENTION (Dao FA2 kernel contract): slopes MUST be POSITIVE magnitudes.
    // The vendored flash-attention alibi.h applies, for causal attention:
    //     score(row, col) += slope * col_idx
    // which (up to a per-row constant that softmax cancels) equals the standard
    // ALiBi penalty  -slope * (row - col)  ONLY when slope > 0.
    // Passing NEGATIVE slopes inverts the bias: the most DISTANT keys (col 0,
    // i.e. the earliest tokens) receive the highest relative score in every head,
    // so early tokens get systematically over-attended in forward AND backward
    // (flash_bwd_kernel.h recomputes the same biased scores for gradients).
    // The "penalty is negative" framing lives in alibi_max_bias (<= 0) only;
    // the slope table handed to the kernel is positive.
    if (params.num_heads == 1) {
        float slope = m_max;
        if (max_slope_magnitude > 0.0f && m_max > max_slope_magnitude) {
            slope = max_slope_magnitude;
        }
        slopes[0] = slope;
        return slopes;
    }

    const float log_mmax = std::log(m_max);
    const float log_mmin = std::log(m_min);
    for (int h = 0; h < params.num_heads; ++h) {
        const float t = static_cast<float>(h) / static_cast<float>(params.num_heads - 1);
        const float log_m = log_mmax + t * (log_mmin - log_mmax);
        float slope_magnitude = std::exp(log_m);
        if (max_slope_magnitude > 0.0f && slope_magnitude > max_slope_magnitude) {
            slope_magnitude = max_slope_magnitude;
        }
        slopes[static_cast<size_t>(h)] = slope_magnitude;
    }

    return slopes;
}

inline std::vector<float> computeDerivedPBMRopeInvFreq(
    const LanguageModelConfig& params,
    const char* caller)
{
    if (params.rotary_dim <= 0 || (params.rotary_dim & 1) != 0) {
        throw std::runtime_error(std::string(caller) + ": PBM rotary_dim must be positive and even, got " +
                                 std::to_string(params.rotary_dim));
    }
    if (params.max_seq_len <= 0) {
        throw std::runtime_error(std::string(caller) + ": PBM max_seq_len must be > 0, got " +
                                 std::to_string(params.max_seq_len));
    }
    if (params.rope_base_seq_len <= 0) {
        throw std::runtime_error(std::string(caller) + ": rope_base_seq_len must be > 0, got " +
                                 std::to_string(params.rope_base_seq_len));
    }
    if (!(params.rope_theta > 0.0f) || !std::isfinite(params.rope_theta)) {
        throw std::runtime_error(std::string(caller) + ": rope_theta must be a positive finite value, got " +
                                 std::to_string(params.rope_theta));
    }
    if (!(params.rope_scaling > 0.0f) || !std::isfinite(params.rope_scaling)) {
        throw std::runtime_error(std::string(caller) + ": rope_scaling must be a positive finite value, got " +
                                 std::to_string(params.rope_scaling));
    }

    const int half_dim = params.rotary_dim / 2;
    std::vector<float> inv_freq(static_cast<size_t>(half_dim));

    float effective_theta = params.rope_theta;
    if (params.max_seq_len > params.rope_base_seq_len && params.rotary_dim > 2) {
        const float ctx_ratio = static_cast<float>(params.max_seq_len) /
                                static_cast<float>(params.rope_base_seq_len);
        const float ntk_exponent = static_cast<float>(params.rotary_dim) /
                                   static_cast<float>(params.rotary_dim - 2);
        effective_theta = params.rope_theta * std::pow(ctx_ratio, ntk_exponent);
    }
    if (!(effective_theta > 0.0f) || !std::isfinite(effective_theta)) {
        throw std::runtime_error(std::string(caller) + ": effective RoPE theta must be a positive finite value, got " +
                                 std::to_string(effective_theta));
    }

    for (int i = 0; i < half_dim; ++i) {
        const float exp_arg = static_cast<float>(2 * i) / static_cast<float>(params.rotary_dim);
        inv_freq[static_cast<size_t>(i)] = params.rope_scaling / std::pow(effective_theta, exp_arg);
    }

    return inv_freq;
}

inline void populateDerivedPBMTables(
    LanguageModelConfig& params,
    const char* caller)
{
    params.pbm_alibi_slopes = computeDerivedPBMAlibiSlopes(params, caller);
    params.pbm_rope_inv_freq = computeDerivedPBMRopeInvFreq(params, caller);
}

inline void refreshMutableTrainingDerivedValues(LanguageModelConfig& params,
                                                int effective_max_seq_len,
                                                const char* caller) {
    params.max_tokens_per_batch = computeMaxTokensPerBatch(
        params.batch_size, effective_max_seq_len, caller);

    if (params.cosine_decay_enabled) {
        if (params.learning_rate <= 0.0f) {
            throw std::runtime_error(
                std::string(caller) + ": learning_rate must be > 0 when cosine_decay is enabled, got " +
                std::to_string(params.learning_rate));
        }
        params.cosine_decay_min_lr = params.learning_rate * 0.1f;
    }

    params.grad_clip_enabled = params.grad_clip_norm > 0.0f;
    params.effective_per_token_grad_limit = std::max(
        params.grad_clip_norm, EPSILON_GRADIENT_CLIP);

    if (params.optimizer_kind != OptimizerKind::ADAMW &&
        params.optimizer_kind != OptimizerKind::RADAMW) {
        throw std::runtime_error(
            std::string(caller) + ": optimizer_kind must be adamw or radamw, got '" +
            std::string(optimizerKindToString(params.optimizer_kind)) + "'");
    }
    if (params.embedding_freeze_enabled) {
        if (params.embedding_freeze_after_step < 0) {
            throw std::runtime_error(
                std::string(caller) + ": embedding_freeze_enabled=true but embedding_freeze_after_step=" +
                std::to_string(params.embedding_freeze_after_step));
        }
        params.optimizer_embedding_freeze_after_step = params.embedding_freeze_after_step;
    } else {
        params.optimizer_embedding_freeze_after_step = -1;
    }
}

inline void deriveComputedLanguageModelConfig(LanguageModelConfig& params) {
    if (params.max_seq_len <= 0) {
        throw std::runtime_error(
            "deriveComputedLanguageModelConfig: max_seq_len must be > 0, got " +
            std::to_string(params.max_seq_len));
    }
    params.min_seq_valid_tokens = params.max_seq_len / 4;
    params.min_seq_len_for_flash = params.max_seq_len / 4;
    params.scratch_max_tokens_per_block = static_cast<size_t>(params.max_seq_len);
    params.attention_dropout = params.dropout_rate;
    params.d_ff = params.d_model * D_FF_MULTIPLIER;
    refreshMutableTrainingDerivedValues(
        params, params.max_seq_len, "deriveComputedLanguageModelConfig");

    if (params.d_model <= 0 || params.num_heads <= 0 || params.num_kv_heads <= 0) {
        throw std::runtime_error(
            "deriveComputedLanguageModelConfig: d_model, num_heads, and num_kv_heads must be > 0");
    }
    if ((params.d_model % params.num_heads) != 0) {
        throw std::runtime_error(
            "deriveComputedLanguageModelConfig: d_model (" +
            std::to_string(params.d_model) + ") must be divisible by num_heads (" +
            std::to_string(params.num_heads) + ")");
    }
    if (!isValidGQAConfig(params.num_heads, params.num_kv_heads)) {
        throw std::runtime_error(
            "deriveComputedLanguageModelConfig: invalid GQA config num_heads=" +
            std::to_string(params.num_heads) + " num_kv_heads=" +
            std::to_string(params.num_kv_heads));
    }
    const int head_dim = params.d_model / params.num_heads;
    params.head_dim = head_dim;
    params.heads_per_kv_group = computeHeadsPerKVGroup(params.num_heads, params.num_kv_heads);
    params.kv_dim = computeKVProjectionSize(params.d_model, params.num_heads, params.num_kv_heads);
    params.qkv_dim = computeQKVProjectionSize(params.d_model, params.num_heads, params.num_kv_heads);
    params.rotary_dim = head_dim;
    params.attention_softmax_scale = computeAttentionSoftmaxScale(
        head_dim, "deriveComputedLanguageModelConfig");
    params.is_gqa = params.num_kv_heads < params.num_heads;
    populateDerivedPBMTables(params, "deriveComputedLanguageModelConfig");
    if (params.num_layers <= 0) {
        throw std::runtime_error(
            "deriveComputedLanguageModelConfig: num_layers must be > 0, got " +
            std::to_string(params.num_layers));
    }
    params.residual_projection_init_gain =
        1.0f / std::sqrt(2.0f * static_cast<float>(params.num_layers));
    params.execution_block_d_key = head_dim;
    params.execution_block_cross_attn_head_dim = head_dim;

    if (params.tokenizer_max_vocab_size > 0 &&
        params.tokenizer_target_vocab_size > params.tokenizer_max_vocab_size) {
        params.tokenizer_target_vocab_size = params.tokenizer_max_vocab_size;
    }

    params.generation_num_return_sequences = 1;
    params.generation_eos_token_id = Tokenizer::EOS_TOKEN_ID;
    params.generation_pad_token_id = Tokenizer::PAD_TOKEN_ID;
    params.generation_bos_token_id = Tokenizer::BOS_TOKEN_ID;
    params.generation_unk_token_id = Tokenizer::UNK_TOKEN_ID;
    params.generation_masked_numeric_literal_ids.clear();
    params.generation_masked_numeric_literal_ids.reserve(10);
    for (int b = 0x30; b <= 0x39; ++b) {
        params.generation_masked_numeric_literal_ids.push_back(Tokenizer::BYTE_TOKEN_OFFSET + b);
    }

    if (params.warmup_fraction <= 0.0f || params.warmup_fraction >= 1.0f) {
        throw std::runtime_error(
            "deriveComputedLanguageModelConfig: warmup_fraction must be in (0, 1), got " +
            std::to_string(params.warmup_fraction));
    }
}

//======================================================//
// (A) Validation primitive.
// Throws std::runtime_error on any invalid required field.
// Pure: does not mutate, does not log.
//======================================================//
inline void validateRootConfigDocument(
    const LanguageModelConfig& params,
    const char* caller)
{
    validatePositiveFields(params, {
        validationField("batch_size", &LanguageModelConfig::batch_size),
        validationField("epochs", &LanguageModelConfig::epochs),
        validationField("validation_interval", &LanguageModelConfig::validation_interval),
        validationField("sliding_window_stride", &LanguageModelConfig::sliding_window_stride),
        validationField("max_seq_len", &LanguageModelConfig::max_seq_len),
        validationField("d_model", &LanguageModelConfig::d_model),
        validationField("num_heads", &LanguageModelConfig::num_heads),
        validationField("num_kv_heads", &LanguageModelConfig::num_kv_heads),
        validationField("head_dim", &LanguageModelConfig::head_dim),
        validationField("heads_per_kv_group", &LanguageModelConfig::heads_per_kv_group),
        validationField("kv_dim", &LanguageModelConfig::kv_dim),
        validationField("qkv_dim", &LanguageModelConfig::qkv_dim),
        validationField("num_layers", &LanguageModelConfig::num_layers),
        validationField("d_ff", &LanguageModelConfig::d_ff),
        validationField("rotary_dim", &LanguageModelConfig::rotary_dim),
        validationField("rope_base_seq_len", &LanguageModelConfig::rope_base_seq_len),
        validationField("alibi_min_locality_distance", &LanguageModelConfig::alibi_min_locality_distance)
    }, caller);
    if (params.sliding_window_stride > params.max_seq_len) {
        throw std::runtime_error(std::string(caller) + ": sliding_window_stride=" +
                                 std::to_string(params.sliding_window_stride) +
                                 " exceeds max_seq_len=" +
                                 std::to_string(params.max_seq_len));
    }
    if (!isValidGQAConfig(params.num_heads, params.num_kv_heads)) {
        throw std::runtime_error(std::string(caller) + ": invalid GQA config num_heads=" +
                                 std::to_string(params.num_heads) + " num_kv_heads=" +
                                 std::to_string(params.num_kv_heads));
    }
    validateDivisibleFields(params, {
        validationFieldRelation("d_model", &LanguageModelConfig::d_model,
                                "num_heads", &LanguageModelConfig::num_heads)
    }, caller);
    const int expected_head_dim = params.d_model / params.num_heads;
    if (params.head_dim != expected_head_dim) {
        throw std::runtime_error(std::string(caller) + ": head_dim=" +
                                 std::to_string(params.head_dim) +
                                 " does not match d_model/num_heads=" +
                                 std::to_string(expected_head_dim));
    }
    const int expected_heads_per_kv_group = params.num_heads / params.num_kv_heads;
    if (params.heads_per_kv_group != expected_heads_per_kv_group) {
        throw std::runtime_error(std::string(caller) + ": heads_per_kv_group=" +
                                 std::to_string(params.heads_per_kv_group) +
                                 " does not match num_heads/num_kv_heads=" +
                                 std::to_string(expected_heads_per_kv_group));
    }
    const int expected_kv_dim = computeKVProjectionSize(
        params.d_model, params.num_heads, params.num_kv_heads);
    if (params.kv_dim != expected_kv_dim) {
        throw std::runtime_error(std::string(caller) + ": kv_dim=" +
                                 std::to_string(params.kv_dim) +
                                 " does not match num_kv_heads*head_dim=" +
                                 std::to_string(expected_kv_dim));
    }
    const int expected_qkv_dim = computeQKVProjectionSize(
        params.d_model, params.num_heads, params.num_kv_heads);
    if (params.qkv_dim != expected_qkv_dim) {
        throw std::runtime_error(std::string(caller) + ": qkv_dim=" +
                                 std::to_string(params.qkv_dim) +
                                 " does not match d_model+2*kv_dim=" +
                                 std::to_string(expected_qkv_dim));
    }
    const float expected_attention_softmax_scale =
        computeAttentionSoftmaxScale(params.head_dim, caller);
    if (std::abs(params.attention_softmax_scale - expected_attention_softmax_scale) > 1e-8f) {
        throw std::runtime_error(std::string(caller) + ": attention_softmax_scale=" +
                                 std::to_string(params.attention_softmax_scale) +
                                 " does not match 1/sqrt(head_dim)=" +
                                 std::to_string(expected_attention_softmax_scale));
    }
    const bool expected_is_gqa = params.num_kv_heads < params.num_heads;
    if (params.is_gqa != expected_is_gqa) {
        throw std::runtime_error(std::string(caller) + ": is_gqa=" +
                                 std::to_string(static_cast<int>(params.is_gqa)) +
                                 " does not match num_kv_heads<num_heads=" +
                                 std::to_string(static_cast<int>(expected_is_gqa)));
    }
    validatePositiveFiniteFields(params, {
        validationField("residual_projection_init_gain", &LanguageModelConfig::residual_projection_init_gain),
        validationField("rms_epsilon", &LanguageModelConfig::rms_epsilon),
        validationField("attention_softmax_scale", &LanguageModelConfig::attention_softmax_scale),
        validationField("rope_theta", &LanguageModelConfig::rope_theta),
        validationField("rope_scaling", &LanguageModelConfig::rope_scaling)
    }, caller);
    validateHalfOpenUnitIntervalFields(params, {
        validationField("dropout_rate", &LanguageModelConfig::dropout_rate),
        validationField("attention_dropout", &LanguageModelConfig::attention_dropout)
    }, caller);
    if ((params.rotary_dim & 1) != 0 || params.rotary_dim > params.head_dim) {
        throw std::runtime_error(std::string(caller) + ": rotary_dim=" +
                                 std::to_string(params.rotary_dim) +
                                 " must be even and <= head_dim=" +
                                 std::to_string(params.head_dim));
    }
    if (params.pbm_alibi_slopes.size() != static_cast<size_t>(params.num_heads)) {
        throw std::runtime_error(std::string(caller) + ": pbm_alibi_slopes.size()=" +
                                 std::to_string(params.pbm_alibi_slopes.size()) +
                                 " does not match num_heads=" +
                                 std::to_string(params.num_heads));
    }
    if (params.pbm_rope_inv_freq.size() != static_cast<size_t>(params.rotary_dim / 2)) {
        throw std::runtime_error(std::string(caller) + ": pbm_rope_inv_freq.size()=" +
                                 std::to_string(params.pbm_rope_inv_freq.size()) +
                                 " does not match rotary_dim/2=" +
                                 std::to_string(params.rotary_dim / 2));
    }
    validateFiniteNonZeroFields(params, {
        validationField("alibi_slope_exponent", &LanguageModelConfig::alibi_slope_exponent)
    }, caller);
    if (!std::isfinite(params.alibi_max_bias) || params.alibi_max_bias > 0.0f) {
        throw std::runtime_error(std::string(caller) +
                                 ": alibi_max_bias must be finite and <= 0, got " +
                                 std::to_string(params.alibi_max_bias));
    }
    if (params.use_flash_attention) {
        validatePositiveFields(params, {
            validationField("min_seq_len_for_flash", &LanguageModelConfig::min_seq_len_for_flash)
        }, caller);
        if (!isValidFlashAttentionHeadDim(params.head_dim)) {
            throw std::runtime_error(std::string(caller) + ": head_dim=" +
                                     std::to_string(params.head_dim) +
                                     " is not supported by FlashAttention");
        }
#ifdef GRIM_FLASHATTN_HDIM64_ONLY
        if (params.head_dim != FLASH_ATTN_HEAD_DIM_64) {
            throw std::runtime_error(std::string(caller) +
                                     ": binary was compiled with GRIM_FLASHATTN_HDIM64_ONLY but head_dim=" +
                                     std::to_string(params.head_dim));
        }
#endif
#ifdef GRIM_FLASHATTN_CAUSAL_ONLY
        if (!params.causal_mask) {
            throw std::runtime_error(std::string(caller) +
                                     ": binary was compiled with GRIM_FLASHATTN_CAUSAL_ONLY but causal_mask=false");
        }
#endif
    }
    if (params.use_layer_scale) {
        validatePositiveFiniteFields(params, {
            validationField("layer_scale_init", &LanguageModelConfig::layer_scale_init)
        }, caller);
    }
    if (params.project_out_pc1) {
        validatePositiveFields(params, {
            validationField("pc1_power_iters", &LanguageModelConfig::pc1_power_iters)
        }, caller);
    }
    if (params.lm_head_mlp_enabled) {
        validatePositiveFields(params, {
            validationField("lm_head_mlp_d_ff", &LanguageModelConfig::lm_head_mlp_d_ff)
        }, caller);
        validatePositiveFiniteFields(params, {
            validationField("lm_head_mlp_alpha", &LanguageModelConfig::lm_head_mlp_alpha)
        }, caller);
    }
    validateParameterGroupPrecision(params.parameter_precision_embedding, "parameter_precision_embedding", caller);
    validateParameterGroupPrecision(params.parameter_precision_lm_head, "parameter_precision_lm_head", caller);
    validateParameterGroupPrecision(params.parameter_precision_attention, "parameter_precision_attention", caller);
    validateParameterGroupPrecision(params.parameter_precision_ffn, "parameter_precision_ffn", caller);
    validateParameterGroupPrecision(params.parameter_precision_rmsnorm, "parameter_precision_rmsnorm", caller);
    validateParameterGroupPrecision(params.parameter_precision_execution_block, "parameter_precision_execution_block", caller);

    validateNonNegativeFiniteFields(params, {
        validationField("loss_focal_gamma", &LanguageModelConfig::loss_focal_gamma),
        validationField("loss_entropy_reg_lambda", &LanguageModelConfig::loss_entropy_reg_lambda),
        validationField("weight_decay", &LanguageModelConfig::weight_decay)
    }, caller);
    if (params.loss_focal_enabled) {
        validatePositiveFiniteFields(params, {
            validationField("loss_focal_alpha", &LanguageModelConfig::loss_focal_alpha)
        }, caller);
    }
    if (params.loss_class_balanced_enabled) {
        validatePositiveFiniteFields(params, {
            validationField("loss_class_balanced_beta", &LanguageModelConfig::loss_class_balanced_beta)
        }, caller);
    }
    validateHalfOpenUnitIntervalFields(params, {
        validationField("loss_label_smoothing_epsilon", &LanguageModelConfig::loss_label_smoothing_epsilon)
    }, caller);
    validateOpenUnitIntervalFields(params, {
        validationField("optimizer_beta1", &LanguageModelConfig::optimizer_beta1),
        validationField("optimizer_beta2", &LanguageModelConfig::optimizer_beta2)
    }, caller);
    if (params.optimizer_kind == OptimizerKind::UNSPECIFIED) {
        throw std::runtime_error(std::string(caller) + ": optimizer_kind is UNSPECIFIED");
    }
    validatePositiveFiniteFields(params, {
        validationField("optimizer_epsilon", &LanguageModelConfig::optimizer_epsilon),
        validationField("tokenizer_character_coverage", &LanguageModelConfig::tokenizer_character_coverage),
        validationField("tokenizer_vocab_score_multiplier", &LanguageModelConfig::tokenizer_vocab_score_multiplier)
    }, caller);
    validateFieldsAtLeast(params, {
        validationConstantBound("optimizer_embedding_freeze_after_step", &LanguageModelConfig::optimizer_embedding_freeze_after_step, -1, "-1")
    }, caller);
    validatePositiveFields(params, {
        validationField("tokenizer_target_vocab_size", &LanguageModelConfig::tokenizer_target_vocab_size),
        validationField("tokenizer_min_cleaned_text_length", &LanguageModelConfig::tokenizer_min_cleaned_text_length),
        validationField("tokenizer_min_subword_freq", &LanguageModelConfig::tokenizer_min_subword_freq)
    }, caller);
    if (params.tokenizer_character_coverage > 1.0f) {
        throw std::runtime_error(std::string(caller) + ": tokenizer_character_coverage must be <= 1, got " +
                                 std::to_string(params.tokenizer_character_coverage));
    }
    validateNonNegativeFields(params, {
        validationField("soft_restart_max_step_window", &LanguageModelConfig::soft_restart_max_step_window),
        validationField("soft_restart_cooldown_steps", &LanguageModelConfig::soft_restart_cooldown_steps),
        validationField("auto_stop_plateau_patience", &LanguageModelConfig::auto_stop_plateau_patience),
        validationField("auto_stop_high_loss_patience", &LanguageModelConfig::auto_stop_high_loss_patience),
        validationField("tokenizer_subword_mining_workers", &LanguageModelConfig::tokenizer_subword_mining_workers),
        validationField("execution_block_num_steps", &LanguageModelConfig::execution_block_num_steps)
    }, caller);

    if (params.use_atom_data || params.execution_block_enabled) {
        validatePositiveFields(params, {
            validationField("atom_embedding_dim", &LanguageModelConfig::atom_embedding_dim)
        }, caller);
    }
    if (params.structured_ce_enabled) {
        validatePositiveFiniteFields(params, {
            validationField("execution_block_structured_ce_weight", &LanguageModelConfig::structured_ce_weight)
        }, caller);
    }
    validateNonNegativeFiniteFields(params, {
        validationField("execution_block_step_x_multiplier", &LanguageModelConfig::step_x_multiplier),
        validationField("execution_block_step_y_multiplier", &LanguageModelConfig::step_y_multiplier),
        validationField("execution_block_entropy_aux_weight", &LanguageModelConfig::entropy_aux_weight),
        validationField("execution_block_final_slot_consistency_weight", &LanguageModelConfig::final_slot_consistency_weight)
    }, caller);
    validatePositiveFiniteFields(params, {
        validationField("execution_block_value_match_epsilon", &LanguageModelConfig::value_match_epsilon)
    }, caller);
    if (params.execution_block_enabled) {
        if (!params.use_atom_data) {
            throw std::runtime_error(
                std::string(caller) + ": execution_block_enabled=true requires use_atom_data=true");
        }
        validateFieldsAtLeast(params, {
            validationConstantBound("execution_block_layer", &LanguageModelConfig::execution_block_layer, -1, "-1"),
            validationConstantBound("execution_block_num_scratch_slots", &LanguageModelConfig::execution_block_num_scratch_slots, 0, "0"),
            validationConstantBound("execution_block_temp_schedule", &LanguageModelConfig::execution_block_temp_schedule, 0, "0"),
            validationConstantBound("execution_block_gate_warmup_steps", &LanguageModelConfig::execution_block_gate_warmup_steps, 0, "0"),
            validationConstantBound("execution_block_result_slot_index", &LanguageModelConfig::execution_block_result_slot_index, -1, "-1"),
            validationConstantBound("execution_block_result_slot_mode", &LanguageModelConfig::execution_block_result_slot_mode, 0, "0")
        }, caller);
        validateFieldsAtMost(params, {
            validationConstantBound("execution_block_d_key", &LanguageModelConfig::execution_block_d_key, 64, "64"),
            validationConstantBound("execution_block_result_slot_mode", &LanguageModelConfig::execution_block_result_slot_mode, 1, "1")
        }, caller);
        validateLessThanFields(params, {
            validationFieldRelation("execution_block_num_scratch_slots", &LanguageModelConfig::execution_block_num_scratch_slots,
                                    "execution_block_num_slots", &LanguageModelConfig::execution_block_num_slots),
            validationFieldRelation("execution_block_result_slot_index", &LanguageModelConfig::execution_block_result_slot_index,
                                    "execution_block_num_slots", &LanguageModelConfig::execution_block_num_slots)
        }, caller);
        if (params.execution_block_layer >= params.num_layers) {
            throw std::runtime_error(
                std::string(caller) + ": execution_block_layer=" +
                std::to_string(params.execution_block_layer) +
                " exceeds num_layers=" + std::to_string(params.num_layers));
        }
        validatePositiveFields(params, {
            validationField("execution_block_num_ops", &LanguageModelConfig::execution_block_num_ops),
            validationField("execution_block_num_slots", &LanguageModelConfig::execution_block_num_slots),
            validationField("execution_block_num_steps", &LanguageModelConfig::execution_block_num_steps),
            validationField("execution_block_value_decode_input_dim", &LanguageModelConfig::execution_block_value_decode_input_dim),
            validationField("execution_block_value_decode_hidden_dim", &LanguageModelConfig::execution_block_value_decode_hidden_dim),
            validationField("execution_block_d_key", &LanguageModelConfig::execution_block_d_key),
            validationField("execution_block_d_type", &LanguageModelConfig::execution_block_d_type),
            validationField("execution_block_cross_attn_head_dim", &LanguageModelConfig::execution_block_cross_attn_head_dim),
            validationField("execution_block_cross_attn_topk", &LanguageModelConfig::execution_block_cross_attn_topk)
        }, caller);
        if (params.execution_block_value_decode_input_dim + 16 > params.atom_embedding_dim) {
            throw std::runtime_error(
                std::string(caller) + ": execution_block_value_decode_input_dim + 16 must fit atom_embedding_dim");
        }
        if (params.execution_block_cross_attn_topk > params.execution_block_num_slots) {
            throw std::runtime_error(
                std::string(caller) + ": execution_block_cross_attn_topk=" +
                std::to_string(params.execution_block_cross_attn_topk) +
                " exceeds execution_block_num_slots=" +
                std::to_string(params.execution_block_num_slots));
        }
        validatePositiveFiniteFields(params, {
            validationField("execution_block_usage_decay", &LanguageModelConfig::execution_block_usage_decay),
            validationField("execution_block_inject_gate_temp", &LanguageModelConfig::execution_block_inject_gate_temp),
            validationField("execution_block_magnitude_limit", &LanguageModelConfig::execution_block_magnitude_limit),
            validationField("execution_block_diversity_kappa", &LanguageModelConfig::execution_block_diversity_kappa),
            validationField("execution_block_temp_start", &LanguageModelConfig::execution_block_temp_start),
            validationField("execution_block_temp_end", &LanguageModelConfig::execution_block_temp_end)
        }, caller);
        if (params.execution_block_usage_decay > 1.0f) {
            throw std::runtime_error(
                std::string(caller) + ": execution_block_usage_decay must be <= 1, got " +
                std::to_string(params.execution_block_usage_decay));
        }
        validateNonNegativeFiniteFields(params, {
            validationField("execution_block_entropy_weight", &LanguageModelConfig::execution_block_entropy_weight),
            validationField("execution_block_transition_hard_threshold", &LanguageModelConfig::execution_block_transition_hard_threshold),
            validationField("execution_block_causal_w1_transition", &LanguageModelConfig::execution_block_causal_w1_transition),
            validationField("execution_block_div_invalid_penalty_weight", &LanguageModelConfig::div_invalid_penalty_weight),
            validationField("execution_block_div_magnitude_penalty_weight", &LanguageModelConfig::div_magnitude_penalty_weight),
            validationField("execution_block_arg_reinforce_weight", &LanguageModelConfig::arg_reinforce_weight)
        }, caller);
        validateClosedUnitIntervalFields(params, {
            validationField("execution_block_entropy_collapse_threshold", &LanguageModelConfig::execution_block_entropy_collapse_threshold),
            validationField("execution_block_write_collapse_threshold", &LanguageModelConfig::execution_block_write_collapse_threshold),
            validationField("execution_block_arg_reinforce_baseline_decay", &LanguageModelConfig::arg_reinforce_baseline_decay)
        }, caller);
    }
    if (params.number_encoder_enabled) {
        if (!params.use_atom_data) {
            throw std::runtime_error(
                std::string(caller) + ": number_encoder_enabled=true requires use_atom_data=true");
        }
        validatePositiveFields(params, {
            validationField("number_encoder_max_digit_slots", &LanguageModelConfig::number_encoder_max_digit_slots),
            validationField("number_encoder_d_hidden", &LanguageModelConfig::number_encoder_d_hidden),
            validationField("number_encoder_max_abs_pow10", &LanguageModelConfig::number_encoder_max_abs_pow10)
        }, caller);
        // arg_number stores pow10 as int16; the bucket table must stay inside that range.
        if (params.number_encoder_max_abs_pow10 > 32766) {
            throw std::runtime_error(
                std::string(caller) + ": number_encoder_max_abs_pow10=" +
                std::to_string(params.number_encoder_max_abs_pow10) +
                " exceeds the int16 pow10 capacity of arg_number digit bindings");
        }
    }
    if (params.generation_strategy == SamplingStrategy::UNSPECIFIED) {
        throw std::runtime_error(std::string(caller) + ": generation_strategy is UNSPECIFIED");
    }
    validatePositiveFields(params, {
        validationField("generation_max_new_tokens", &LanguageModelConfig::generation_max_new_tokens),
        validationField("generation_num_return_sequences", &LanguageModelConfig::generation_num_return_sequences),
        validationField("generation_repetition_penalty_window", &LanguageModelConfig::generation_repetition_penalty_window)
    }, caller);
    validateNonNegativeFields(params, {
        validationField("generation_min_new_tokens", &LanguageModelConfig::generation_min_new_tokens),
        validationField("generation_top_k", &LanguageModelConfig::generation_top_k),
        validationField("generation_no_repeat_ngram_size", &LanguageModelConfig::generation_no_repeat_ngram_size),
        validationField("generation_eos_token_id", &LanguageModelConfig::generation_eos_token_id),
        validationField("generation_pad_token_id", &LanguageModelConfig::generation_pad_token_id),
        validationField("generation_bos_token_id", &LanguageModelConfig::generation_bos_token_id),
        validationField("generation_unk_token_id", &LanguageModelConfig::generation_unk_token_id)
    }, caller);
    if (params.generation_min_new_tokens > params.generation_max_new_tokens) {
        throw std::runtime_error(std::string(caller) + ": generation_min_new_tokens=" +
                                 std::to_string(params.generation_min_new_tokens) +
                                 " exceeds generation_max_new_tokens=" +
                                 std::to_string(params.generation_max_new_tokens));
    }
    validatePositiveFiniteFields(params, {
        validationField("generation_temperature", &LanguageModelConfig::generation_temperature),
        validationField("generation_repetition_penalty", &LanguageModelConfig::generation_repetition_penalty)
    }, caller);
    validateNonNegativeFiniteFields(params, {
        validationField("generation_min_p", &LanguageModelConfig::generation_min_p),
        validationField("generation_frequency_penalty", &LanguageModelConfig::generation_frequency_penalty),
        validationField("generation_presence_penalty", &LanguageModelConfig::generation_presence_penalty)
    }, caller);
    validateClosedUnitIntervalFields(params, {
        validationField("generation_top_p", &LanguageModelConfig::generation_top_p),
        validationField("generation_typical_p", &LanguageModelConfig::generation_typical_p)
    }, caller);
    if (params.generation_min_p > 1.0f) {
        throw std::runtime_error(std::string(caller) + ": generation_min_p must be <= 1, got " +
                                 std::to_string(params.generation_min_p));
    }
    if (params.generation_masked_numeric_literal_ids.empty()) {
        throw std::runtime_error(std::string(caller) + ": generation_masked_numeric_literal_ids is empty");
    }
    validatePositiveFields(params, {
        validationField("vocab_size", &LanguageModelConfig::vocab_size),
        validationField("max_cached_seq_len", &LanguageModelConfig::max_cached_seq_len),
        validationField("max_tokens_per_batch", &LanguageModelConfig::max_tokens_per_batch)
    }, caller);
    if (params.data_path.empty()) {
        throw std::runtime_error(std::string(caller) + ": data_path is empty");
    }
    if (params.vocab_path.empty()) {
        throw std::runtime_error(std::string(caller) + ": vocab_path is empty");
    }
    if (params.output_model_path.empty()) {
        throw std::runtime_error(std::string(caller) + ": output_model_path is empty");
    }
    if (params.checkpoint_dir.empty()) {
        throw std::runtime_error(std::string(caller) + ": checkpoint_dir is empty");
    }
    if (params.log_dir.empty()) {
        throw std::runtime_error(std::string(caller) + ": log_dir is empty");
    }
    const std::int64_t expected_tokens =
        static_cast<std::int64_t>(params.batch_size) *
        static_cast<std::int64_t>(params.max_cached_seq_len);
    if (expected_tokens > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(caller) +
                                 ": batch_size * max_cached_seq_len overflowed int capacity (batch=" +
                                 std::to_string(params.batch_size) + " seq_len=" +
                                 std::to_string(params.max_cached_seq_len) + " product=" +
                                 std::to_string(expected_tokens) + ")");
    }
    if (params.max_tokens_per_batch != static_cast<int>(expected_tokens)) {
        throw std::runtime_error(std::string(caller) + ": max_tokens_per_batch=" +
                                 std::to_string(params.max_tokens_per_batch) +
                                 " does not match batch_size * max_cached_seq_len=" +
                                 std::to_string(expected_tokens));
    }
    if (params.execution_mode != ModelExecutionMode::TRAINING &&
        params.execution_mode != ModelExecutionMode::INFERENCE) {
        throw std::runtime_error(std::string(caller) + ": execution_mode is neither TRAINING nor INFERENCE");
    }
    if (params.structured_ce_enabled && params.structured_ce_weight <= 0.0f) {
        throw std::runtime_error(std::string(caller) + ": structured_ce_enabled=true but structured_ce_weight=" +
                                 std::to_string(params.structured_ce_weight) + " (must be > 0)");
    }
}

//======================================================//
// (B) Pure schedule derivation.
// Computes batches_per_epoch / total_training_steps / safe_last_step
// from the (sequence_count, batch_size, epochs) ratio. No mutation,
// no logging. warmup_steps is NOT computed here; Phase1 epoch-plan
// finalization stamps it later once estimated_total_steps exists.
// Caller MUST have already passed validateRootConfigDocument().
//======================================================//
template <typename T>
inline T snapshotTrainingConfigField(
    const GRIM::Config::AiConfigSnapshot& snapshot,
    const char* name);

inline DerivedScheduleInfo computeDerivedSchedule(
    const LanguageModelConfig& params,
    const DerivationContext& context) {
    DerivedScheduleInfo info;
    const int safe_batch_size = params.batch_size;
    const int sequence_count  = std::max(0, context.train_sequence_count);
    info.batches_per_epoch    = std::max(1, (sequence_count + safe_batch_size - 1) / safe_batch_size);
    info.total_training_steps = std::max(1, info.batches_per_epoch * params.epochs);
    info.safe_last_step       = std::max(info.total_training_steps - 1, 1);
    return info;
}

inline DerivedScheduleInfo computeDerivedSchedule(
    const GRIM::Config::AiConfigSnapshot& snapshot,
    const DerivationContext& context) {
    DerivedScheduleInfo info;
    const int safe_batch_size = snapshotTrainingConfigField<int>(snapshot, "batch_size");
    const int sequence_count  = std::max(0, context.train_sequence_count);
    const int epochs          = snapshotTrainingConfigField<int>(snapshot, "epochs");
    info.batches_per_epoch    = std::max(1, (sequence_count + safe_batch_size - 1) / safe_batch_size);
    info.total_training_steps = std::max(1, info.batches_per_epoch * epochs);
    info.safe_last_step       = std::max(info.total_training_steps - 1, 1);
    return info;
}

} // namespace HyperParameters
} // namespace GRIM

#endif // GRIM_CONFIG_AI_CONFIG_PATHS_HPP_INCLUDED

//======================================================//
// Config loading from the canonical ai_config.json snapshot.
//======================================================//

namespace GRIM {
namespace HyperParameters {
inline LanguageModelConfig loadLanguageModelConfig(
    const nlohmann::json& document) {
    const auto& config = document.at("training").at("config");
    LanguageModelConfig params{};
    auto load = [&](const char* name, auto member) {
        using Member = std::remove_cv_t<std::remove_reference_t<decltype(params.*member)>>;
        params.*member = config.at(name).get<Member>();
    };

#define GRIM_LOAD_CONFIG_FIELD(member) load(#member, &LanguageModelConfig::member)
#define GRIM_LOAD_CONFIG_LEAF(leaf, member) load(leaf, &LanguageModelConfig::member)

    GRIM_LOAD_CONFIG_FIELD(current_model_training);
    GRIM_LOAD_CONFIG_FIELD(current_curriculum);
    GRIM_LOAD_CONFIG_FIELD(clear_merged_cache_on_merge);
    GRIM_LOAD_CONFIG_FIELD(epochs);
    GRIM_LOAD_CONFIG_FIELD(seed);
    GRIM_LOAD_CONFIG_FIELD(batch_size);
    GRIM_LOAD_CONFIG_FIELD(gradient_accumulation_steps);
    GRIM_LOAD_CONFIG_FIELD(batch_strategy);
    GRIM_LOAD_CONFIG_FIELD(learning_rate);
    GRIM_LOAD_CONFIG_FIELD(weight_decay);
    GRIM_LOAD_CONFIG_FIELD(use_depth_aware_upsilon);
    GRIM_LOAD_CONFIG_LEAF("gradient_clip", grad_clip_norm);
    GRIM_LOAD_CONFIG_FIELD(per_token_grad_scale);
    GRIM_LOAD_CONFIG_FIELD(force_rebuild_vocab);
    GRIM_LOAD_CONFIG_FIELD(d_model);
    GRIM_LOAD_CONFIG_FIELD(num_layers);
    GRIM_LOAD_CONFIG_FIELD(num_heads);
    GRIM_LOAD_CONFIG_FIELD(num_kv_heads);
    GRIM_LOAD_CONFIG_FIELD(max_seq_len);
    GRIM_LOAD_CONFIG_FIELD(tie_embeddings);
    GRIM_LOAD_CONFIG_FIELD(use_bias);
    GRIM_LOAD_CONFIG_FIELD(lm_head_unigram_bias);
    GRIM_LOAD_CONFIG_FIELD(dropout_rate);
    GRIM_LOAD_CONFIG_FIELD(embedding_scale);
    GRIM_LOAD_CONFIG_FIELD(sliding_window_stride);
    GRIM_LOAD_CONFIG_FIELD(warmup_fraction);
    GRIM_LOAD_CONFIG_FIELD(cosine_decay_enabled);
    GRIM_LOAD_CONFIG_FIELD(cosine_warm_restarts);
    GRIM_LOAD_CONFIG_FIELD(log_interval);
    GRIM_LOAD_CONFIG_FIELD(atom_stats_interval);
    GRIM_LOAD_CONFIG_FIELD(atom_stats_max_seqs);
    GRIM_LOAD_CONFIG_FIELD(inference_diagnostic_enabled);
    GRIM_LOAD_CONFIG_FIELD(inference_diagnostic_interval);
    GRIM_LOAD_CONFIG_FIELD(validation_interval);
    GRIM_LOAD_CONFIG_FIELD(checkpoint_interval);
    GRIM_LOAD_CONFIG_FIELD(use_gpu);
    GRIM_LOAD_CONFIG_FIELD(use_flash_attention);
    GRIM_LOAD_CONFIG_FIELD(parameter_precision_embedding);
    GRIM_LOAD_CONFIG_FIELD(parameter_precision_lm_head);
    GRIM_LOAD_CONFIG_FIELD(parameter_precision_attention);
    GRIM_LOAD_CONFIG_FIELD(parameter_precision_ffn);
    GRIM_LOAD_CONFIG_FIELD(parameter_precision_rmsnorm);
    GRIM_LOAD_CONFIG_FIELD(parameter_precision_execution_block);

    params.positional_encoding = parsePositionalEncodingFlags(
        config.at("use_rope").get<bool>(),
        config.at("use_alibi").get<bool>());
    GRIM_LOAD_CONFIG_FIELD(rope_base_seq_len);
    GRIM_LOAD_CONFIG_FIELD(alibi_min_locality_distance);
    GRIM_LOAD_CONFIG_FIELD(alibi_slope_exponent);
    GRIM_LOAD_CONFIG_FIELD(alibi_max_bias);
    GRIM_LOAD_CONFIG_FIELD(rope_theta);
    GRIM_LOAD_CONFIG_FIELD(rope_scaling);
    GRIM_LOAD_CONFIG_FIELD(soft_restart_enabled);
    GRIM_LOAD_CONFIG_FIELD(soft_restart_loss_increase_threshold);
    GRIM_LOAD_CONFIG_FIELD(soft_restart_max_step_window);
    GRIM_LOAD_CONFIG_FIELD(soft_restart_cooldown_steps);
    GRIM_LOAD_CONFIG_FIELD(auto_stop_enabled);
    GRIM_LOAD_CONFIG_FIELD(auto_stop_plateau_patience);
    GRIM_LOAD_CONFIG_FIELD(auto_stop_plateau_min_delta);
    GRIM_LOAD_CONFIG_FIELD(auto_stop_high_loss_threshold);
    GRIM_LOAD_CONFIG_FIELD(auto_stop_high_loss_patience);
    GRIM_LOAD_CONFIG_FIELD(single_batch_overfit_enabled);
    GRIM_LOAD_CONFIG_FIELD(single_batch_overfit_max_steps);
    GRIM_LOAD_CONFIG_FIELD(shuffle_train_enabled);
    GRIM_LOAD_CONFIG_FIELD(shuffle_train_epochs);
    GRIM_LOAD_CONFIG_FIELD(telemetry_control_enabled);
    GRIM_LOAD_CONFIG_FIELD(telemetry_spike_mild_threshold);
    GRIM_LOAD_CONFIG_FIELD(telemetry_spike_moderate_threshold);
    GRIM_LOAD_CONFIG_FIELD(telemetry_spike_severe_threshold);
    GRIM_LOAD_CONFIG_FIELD(telemetry_moderate_grad_scale);
    GRIM_LOAD_CONFIG_FIELD(telemetry_moderate_cooldown_extension);
    GRIM_LOAD_CONFIG_FIELD(telemetry_min_grad_for_nonzero_loss);
    GRIM_LOAD_CONFIG_FIELD(telemetry_loss_threshold_for_grad_check);
    GRIM_LOAD_CONFIG_FIELD(telemetry_max_consecutive_zero_grad_steps);
    GRIM_LOAD_CONFIG_FIELD(telemetry_seq_len_regime_change_threshold);
    GRIM_LOAD_CONFIG_FIELD(telemetry_regime_change_suppression_steps);
    GRIM_LOAD_CONFIG_FIELD(telemetry_volatility_damping_threshold);
    GRIM_LOAD_CONFIG_FIELD(telemetry_max_volatility_damping);
    GRIM_LOAD_CONFIG_FIELD(telemetry_gradient_decay_threshold);
    GRIM_LOAD_CONFIG_FIELD(telemetry_max_decay_boost);
    GRIM_LOAD_CONFIG_FIELD(telemetry_progress_boost_threshold);
    GRIM_LOAD_CONFIG_FIELD(telemetry_max_progress_boost);
    GRIM_LOAD_CONFIG_FIELD(telemetry_outlier_frequency_trigger);
    GRIM_LOAD_CONFIG_FIELD(telemetry_outlier_persistence_trigger);
    GRIM_LOAD_CONFIG_FIELD(telemetry_anchor_drift_sigma_multiplier);
    GRIM_LOAD_CONFIG_FIELD(telemetry_baseline_stabilization_steps);
    GRIM_LOAD_CONFIG_FIELD(telemetry_verbose_logging);
    GRIM_LOAD_CONFIG_FIELD(telemetry_fail_loud_on_accumulation_bug);
    GRIM_LOAD_CONFIG_FIELD(telemetry_plateau_noise_enabled);
    GRIM_LOAD_CONFIG_FIELD(telemetry_plateau_noise_patience);
    GRIM_LOAD_CONFIG_FIELD(telemetry_plateau_noise_variance_threshold);
    GRIM_LOAD_CONFIG_FIELD(telemetry_plateau_noise_std);
    GRIM_LOAD_CONFIG_FIELD(telemetry_plateau_noise_proportional);
    GRIM_LOAD_CONFIG_FIELD(telemetry_plateau_noise_cooldown);
    GRIM_LOAD_CONFIG_FIELD(telemetry_plateau_noise_max_per_epoch);
    GRIM_LOAD_CONFIG_FIELD(telemetry_lattice_num_levels);
    GRIM_LOAD_CONFIG_FIELD(telemetry_lattice_num_streams);
    GRIM_LOAD_CONFIG_FIELD(telemetry_lattice_beta_mu);
    GRIM_LOAD_CONFIG_FIELD(telemetry_lattice_beta_a);
    GRIM_LOAD_CONFIG_FIELD(telemetry_lattice_beta_delta);
    GRIM_LOAD_CONFIG_FIELD(telemetry_lattice_beta_r);
    GRIM_LOAD_CONFIG_FIELD(telemetry_lattice_beta_run);
    GRIM_LOAD_CONFIG_FIELD(telemetry_lattice_beta_v);
    GRIM_LOAD_CONFIG_FIELD(telemetry_lattice_k_out0);
    GRIM_LOAD_CONFIG_FIELD(telemetry_lattice_alpha_v);
    GRIM_LOAD_CONFIG_FIELD(telemetry_lattice_epsilon);
    GRIM_LOAD_CONFIG_FIELD(telemetry_lattice_strict_mode);
    GRIM_LOAD_CONFIG_FIELD(logging_default_level);
    GRIM_LOAD_CONFIG_FIELD(logging_equation_csv_enabled);
    GRIM_LOAD_CONFIG_FIELD(logging_stderr_enabled);
    GRIM_LOAD_CONFIG_FIELD(logging_initial_capacity);
    GRIM_LOAD_CONFIG_FIELD(logging_group_overrides);
    GRIM_LOAD_CONFIG_FIELD(log_recorder_enabled);
    GRIM_LOAD_CONFIG_FIELD(log_recorder_default_level);
    GRIM_LOAD_CONFIG_FIELD(log_recorder_modules);
    GRIM_LOAD_CONFIG_FIELD(log_recorder_layer_embedding);
    GRIM_LOAD_CONFIG_FIELD(log_recorder_layer_rms_norm);
    GRIM_LOAD_CONFIG_FIELD(log_recorder_layer_attention);
    GRIM_LOAD_CONFIG_FIELD(log_recorder_layer_feed_forward);
    GRIM_LOAD_CONFIG_FIELD(log_recorder_layer_residual);
    GRIM_LOAD_CONFIG_FIELD(log_recorder_layer_encoding);
    GRIM_LOAD_CONFIG_FIELD(log_recorder_layer_serialization);
    GRIM_LOAD_CONFIG_FIELD(log_recorder_layer_execution_block);
    GRIM_LOAD_CONFIG_FIELD(loss_label_smoothing_enabled);
    GRIM_LOAD_CONFIG_FIELD(loss_label_smoothing_epsilon);
    GRIM_LOAD_CONFIG_FIELD(loss_focal_enabled);
    GRIM_LOAD_CONFIG_FIELD(loss_focal_gamma);
    GRIM_LOAD_CONFIG_FIELD(loss_focal_alpha);
    GRIM_LOAD_CONFIG_FIELD(loss_entropy_reg_enabled);
    GRIM_LOAD_CONFIG_FIELD(loss_entropy_reg_lambda);
    GRIM_LOAD_CONFIG_FIELD(loss_class_balanced_enabled);
    GRIM_LOAD_CONFIG_FIELD(loss_class_balanced_beta);
    GRIM_LOAD_CONFIG_FIELD(loss_preference_enabled);
    GRIM_LOAD_CONFIG_FIELD(loss_preference_beta);
    GRIM_LOAD_CONFIG_FIELD(loss_distillation_enabled);
    GRIM_LOAD_CONFIG_FIELD(loss_distillation_temperature);
    GRIM_LOAD_CONFIG_FIELD(loss_distillation_lambda);
    GRIM_LOAD_CONFIG_FIELD(loss_masking_enabled);
    GRIM_LOAD_CONFIG_FIELD(loss_masking_tag);
    GRIM_LOAD_CONFIG_FIELD(lm_head_centering_enabled);
    GRIM_LOAD_CONFIG_FIELD(lm_head_center_hidden_states);
    GRIM_LOAD_CONFIG_FIELD(lm_head_mlp_enabled);
    GRIM_LOAD_CONFIG_FIELD(lm_head_mlp_d_ff);
    GRIM_LOAD_CONFIG_FIELD(lm_head_mlp_alpha);
    GRIM_LOAD_CONFIG_FIELD(freeze_learned_rms_gammas);
    GRIM_LOAD_CONFIG_FIELD(center_logits);
    GRIM_LOAD_CONFIG_FIELD(center_encoder_residuals);
    GRIM_LOAD_CONFIG_FIELD(project_out_pc1);
    GRIM_LOAD_CONFIG_FIELD(pc1_power_iters);
    GRIM_LOAD_CONFIG_FIELD(use_layer_scale);
    GRIM_LOAD_CONFIG_FIELD(layer_scale_init);
    GRIM_LOAD_CONFIG_FIELD(qk_norm_enabled);
    GRIM_LOAD_CONFIG_FIELD(attention_off_by_one);
    if (config.at("hardcoded_hidden_states_enabled").get<bool>()) {
        params.hardcoded_hidden_pattern = config.at("hardcoded_hidden_states_pattern").get<HardcodedPattern>();
    }
    GRIM_LOAD_CONFIG_FIELD(hardcoded_log_every_n_batches);
    GRIM_LOAD_CONFIG_FIELD(generation_strategy);
    GRIM_LOAD_CONFIG_FIELD(generation_max_new_tokens);
    GRIM_LOAD_CONFIG_FIELD(generation_min_new_tokens);
    GRIM_LOAD_CONFIG_FIELD(generation_top_k);
    GRIM_LOAD_CONFIG_FIELD(generation_repetition_penalty_window);
    GRIM_LOAD_CONFIG_FIELD(generation_no_repeat_ngram_size);
    GRIM_LOAD_CONFIG_FIELD(generation_temperature);
    GRIM_LOAD_CONFIG_FIELD(generation_top_p);
    GRIM_LOAD_CONFIG_FIELD(generation_min_p);
    GRIM_LOAD_CONFIG_FIELD(generation_typical_p);
    GRIM_LOAD_CONFIG_FIELD(generation_repetition_penalty);
    GRIM_LOAD_CONFIG_FIELD(generation_frequency_penalty);
    GRIM_LOAD_CONFIG_FIELD(generation_presence_penalty);
    GRIM_LOAD_CONFIG_FIELD(generation_do_sample);
    GRIM_LOAD_CONFIG_FIELD(generation_enable_scratchblock_reasoning);
    GRIM_LOAD_CONFIG_FIELD(embedding_freeze_enabled);
    GRIM_LOAD_CONFIG_FIELD(embedding_freeze_after_step);
    GRIM_LOAD_CONFIG_FIELD(optimizer_kind);
    GRIM_LOAD_CONFIG_FIELD(optimizer_beta1);
    GRIM_LOAD_CONFIG_FIELD(optimizer_beta2);
    GRIM_LOAD_CONFIG_FIELD(optimizer_epsilon);

    GRIM_LOAD_CONFIG_FIELD(stability_overrides_enabled);
    GRIM_LOAD_CONFIG_FIELD(stability_override_batch_size);
    GRIM_LOAD_CONFIG_FIELD(stability_override_max_seq_len);
    GRIM_LOAD_CONFIG_FIELD(stability_override_clip_per_token);
    GRIM_LOAD_CONFIG_FIELD(scratch_blocks_enabled);
    GRIM_LOAD_CONFIG_FIELD(scratch_num_blocks);
    GRIM_LOAD_CONFIG_FIELD(scratch_write_combined);
    GRIM_LOAD_CONFIG_FIELD(use_atom_data);
    GRIM_LOAD_CONFIG_FIELD(atom_embedding_dim);
    GRIM_LOAD_CONFIG_FIELD(execution_block_enabled);
    GRIM_LOAD_CONFIG_FIELD(execution_block_debug_mode);
    GRIM_LOAD_CONFIG_LEAF("execution_block_step_y_overrides_x", step_y_overrides_x);
    GRIM_LOAD_CONFIG_LEAF("execution_block_structured_ce_enabled", structured_ce_enabled);
    GRIM_LOAD_CONFIG_FIELD(execution_block_layer);
    GRIM_LOAD_CONFIG_FIELD(execution_block_num_ops);
    GRIM_LOAD_CONFIG_FIELD(execution_block_num_slots);
    GRIM_LOAD_CONFIG_FIELD(execution_block_num_scratch_slots);
    GRIM_LOAD_CONFIG_FIELD(execution_block_num_steps);
    GRIM_LOAD_CONFIG_FIELD(execution_block_value_decode_input_dim);
    GRIM_LOAD_CONFIG_FIELD(execution_block_value_decode_hidden_dim);
    GRIM_LOAD_CONFIG_FIELD(execution_block_d_type);
    GRIM_LOAD_CONFIG_FIELD(execution_block_cross_attn_topk);
    GRIM_LOAD_CONFIG_FIELD(execution_block_result_slot_mode);
    GRIM_LOAD_CONFIG_FIELD(execution_block_result_slot_index);
    GRIM_LOAD_CONFIG_FIELD(execution_block_temp_schedule);
    GRIM_LOAD_CONFIG_FIELD(execution_block_usage_decay);
    GRIM_LOAD_CONFIG_FIELD(execution_block_inject_gate_temp);
    GRIM_LOAD_CONFIG_FIELD(execution_block_entropy_collapse_threshold);
    GRIM_LOAD_CONFIG_FIELD(execution_block_write_collapse_threshold);
    GRIM_LOAD_CONFIG_FIELD(execution_block_magnitude_limit);
    GRIM_LOAD_CONFIG_FIELD(execution_block_diversity_kappa);
    GRIM_LOAD_CONFIG_FIELD(execution_block_temp_start);
    GRIM_LOAD_CONFIG_FIELD(execution_block_temp_end);
    GRIM_LOAD_CONFIG_FIELD(execution_block_entropy_weight);
    GRIM_LOAD_CONFIG_LEAF("execution_block_step_x_multiplier", step_x_multiplier);
    GRIM_LOAD_CONFIG_LEAF("execution_block_step_y_multiplier", step_y_multiplier);
    GRIM_LOAD_CONFIG_LEAF("execution_block_entropy_aux_weight", entropy_aux_weight);
    GRIM_LOAD_CONFIG_LEAF("execution_block_value_match_epsilon", value_match_epsilon);
    GRIM_LOAD_CONFIG_LEAF("execution_block_final_slot_consistency_weight", final_slot_consistency_weight);
    GRIM_LOAD_CONFIG_FIELD(execution_block_transition_hard_threshold);
    GRIM_LOAD_CONFIG_FIELD(execution_block_causal_w1_transition);
    GRIM_LOAD_CONFIG_LEAF("execution_block_div_invalid_penalty_weight", div_invalid_penalty_weight);
    GRIM_LOAD_CONFIG_LEAF("execution_block_div_magnitude_penalty_weight", div_magnitude_penalty_weight);
    GRIM_LOAD_CONFIG_LEAF("execution_block_arg_reinforce_weight", arg_reinforce_weight);
    GRIM_LOAD_CONFIG_LEAF("execution_block_arg_reinforce_baseline_decay", arg_reinforce_baseline_decay);
    GRIM_LOAD_CONFIG_LEAF("execution_block_structured_ce_weight", structured_ce_weight);
    GRIM_LOAD_CONFIG_FIELD(number_encoder_enabled);
    GRIM_LOAD_CONFIG_FIELD(number_encoder_max_digit_slots);
    GRIM_LOAD_CONFIG_FIELD(number_encoder_d_hidden);
    GRIM_LOAD_CONFIG_FIELD(number_encoder_max_abs_pow10);
    GRIM_LOAD_CONFIG_FIELD(single_stream_mode);
    GRIM_LOAD_CONFIG_FIELD(disable_async_frees);
    GRIM_LOAD_CONFIG_FIELD(synchronize_after_kernels);
    GRIM_LOAD_CONFIG_FIELD(prediction_comparison_enabled);
    GRIM_LOAD_CONFIG_FIELD(prediction_comparison_interval);
    GRIM_LOAD_CONFIG_FIELD(prediction_comparison_top_k);
    GRIM_LOAD_CONFIG_FIELD(prediction_comparison_max_positions);
    GRIM_LOAD_CONFIG_FIELD(prediction_comparison_log_path);
    GRIM_LOAD_CONFIG_FIELD(logit_update_trace_enabled);
    GRIM_LOAD_CONFIG_FIELD(logit_update_trace_interval);
    GRIM_LOAD_CONFIG_FIELD(attention_diag_enabled);
    GRIM_LOAD_CONFIG_FIELD(attention_diag_layer);
    GRIM_LOAD_CONFIG_FIELD(attention_diag_head);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_enable_atom_reasoning);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_detect_numbers);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_target_vocab_size);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_max_vocab_size);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_max_length);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_min_cleaned_text_length);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_min_subword_freq);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_subword_mining_workers);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_parallel_threshold);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_character_coverage);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_vocab_score_multiplier);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_subword_mining_max_bytes);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_prune_during_mining);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_enable_parallel_subword_mining);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_add_bos);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_add_eos);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_enable_nfkc_normalization);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_enable_lowercasing);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_enable_parallel_tokenization);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_enable_byte_fallback);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_save_text_vocab);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_model_type);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_unk_token);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_pad_token);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_bos_token);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_eos_token);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_expected_checksum);
    GRIM_LOAD_CONFIG_FIELD(tokenizer_special_tokens);
    GRIM_LOAD_CONFIG_FIELD(subprocess_tokenizer_only_mode);

#undef GRIM_LOAD_CONFIG_LEAF
#undef GRIM_LOAD_CONFIG_FIELD

    deriveComputedLanguageModelConfig(params);
    return params;
}

//======================================================//
// Startup configuration finalization.
//
// control/ai_config_paths.hpp::loadAiConfigSnapshot() is the raw authored
// source. This header consumes AiConfigSnapshot::document directly, computes
// and validates the startup/runtime values, and returns the transitional typed
// handoff still required by downstream CUDA/model consumers.
//
// Side effects intentionally NOT performed here (caller does them):
//   - GRIM::Logging::InitLogRecorder()          → training log subsystem
// Keeping those out preserves the Shared/HyperParameters → Training
// layering rule (HyperParameters does not depend on training/).
//======================================================//

inline std::string resolveMappedPath(const std::string& raw_path) {
    std::filesystem::path path(raw_path);
    if (!path.is_relative()) {
        return raw_path;
    }
    return (GRIM::Config::detail::resolveAiConfigPath().parent_path() / path).string();
}

inline void loadResolvedPathFields(
    LanguageModelConfig& config,
    const nlohmann::json& document)
{
    const auto& doc_config = document.at("training").at("config");
    config.data_path = resolveMappedPath(doc_config.at("grim_text_training_data").get<std::string>());
    config.vocab_path = resolveMappedPath(doc_config.at("grim_text_vocab").get<std::string>());
    config.output_model_path = resolveMappedPath(doc_config.at("grim_text_model").get<std::string>());
    config.checkpoint_dir = resolveMappedPath(doc_config.at("grim_text_checkpoints").get<std::string>());
    config.log_dir = resolveMappedPath(doc_config.at("grim_text_logs").get<std::string>());
    config.status_path = resolveMappedPath(doc_config.at("grim_text_training_status").get<std::string>());
    if (config.data_path.empty() ||
        config.vocab_path.empty() ||
        config.output_model_path.empty() ||
        config.checkpoint_dir.empty() ||
        config.log_dir.empty()) {
        throw std::runtime_error(
            "FATAL: ai_config.json training.config grim_text_* path fields must all be non-empty");
    }
}

inline const nlohmann::json& snapshotTrainingConfig(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    return snapshot.document.at("training").at("config");
}

inline nlohmann::json& mutableSnapshotTrainingConfig(
    GRIM::Config::AiConfigSnapshot& snapshot)
{
    return snapshot.document.at("training").at("config");
}

template <typename T>
inline T snapshotTrainingConfigField(
    const GRIM::Config::AiConfigSnapshot& snapshot,
    const char* name)
{
    return snapshotTrainingConfig(snapshot).at(name).get<T>();
}

inline void validateRootBiasConfig(
    const GRIM::Config::AiConfigSnapshot& snapshot,
    const char* caller)
{
    const bool use_bias = snapshotTrainingConfigField<bool>(snapshot, "use_bias");
    const auto require_global = [&](const char* child_name) {
        if (snapshotTrainingConfigField<bool>(snapshot, child_name) && !use_bias) {
            throw std::runtime_error(std::string(caller) + ": " + child_name +
                                     "=true requires use_bias=true");
        }
    };

    require_global("attention_qkv_bias_enabled");
    require_global("attention_output_bias_enabled");
    require_global("ffn_output_bias_enabled");
    require_global("number_encoder_contribution_bias_enabled");
    require_global("number_encoder_global_bias_enabled");
    require_global("lm_head_bias_enabled");
    require_global("execution_block_decode_bias_enabled");
    require_global("execution_block_value_embedding_bias_enabled");
    require_global("execution_block_scalar_bias_enabled");
    require_global("execution_block_trace_bias_enabled");
    require_global("latent_trajectory_hidden_bias_enabled");
    require_global("latent_trajectory_fuse_bias_enabled");
    require_global("latent_trajectory_down_bias_enabled");
    require_global("latent_trajectory_up_bias_enabled");
    require_global("latent_trajectory_gate_bias_enabled");

    if (snapshotTrainingConfigField<bool>(snapshot, "lm_head_unigram_bias") &&
        !snapshotTrainingConfigField<bool>(snapshot, "lm_head_bias_enabled")) {
        throw std::runtime_error(std::string(caller) +
                                 ": lm_head_unigram_bias=true requires lm_head_bias_enabled=true");
    }
}

inline int snapshotEffectiveBatchSize(
    const GRIM::Config::AiConfigSnapshot& snapshot,
    const char* caller)
{
    const int batch_size = snapshotTrainingConfigField<int>(snapshot, "batch_size");
    if (!snapshotTrainingConfigField<bool>(snapshot, "stability_overrides_enabled")) {
        return batch_size;
    }

    const int override_batch_size =
        snapshotTrainingConfigField<int>(snapshot, "stability_override_batch_size");
    if (override_batch_size <= 0) {
        throw std::runtime_error(
            std::string(caller) +
            ": stability_overrides_enabled=true but stability_override_batch_size=" +
            std::to_string(override_batch_size));
    }
    return override_batch_size;
}

inline int snapshotEffectiveMaxSeqLen(
    const GRIM::Config::AiConfigSnapshot& snapshot,
    const char* caller)
{
    int max_seq_len = snapshotTrainingConfigField<int>(snapshot, "max_seq_len");
    if (snapshotTrainingConfigField<bool>(snapshot, "stability_overrides_enabled")) {
        const int override_max_seq_len =
            snapshotTrainingConfigField<int>(snapshot, "stability_override_max_seq_len");
        if (override_max_seq_len > 0) {
            max_seq_len = override_max_seq_len;
        }
    }

    if (max_seq_len <= 0) {
        throw std::runtime_error(
            std::string(caller) +
            ": max_seq_len not configured in ai_config.json "
            "(stability overrides or root config)");
    }
    return max_seq_len;
}

inline float snapshotEffectiveGradClipNorm(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    float grad_clip_norm = snapshotTrainingConfigField<float>(snapshot, "gradient_clip");
    if (snapshotTrainingConfigField<bool>(snapshot, "stability_overrides_enabled")) {
        const float override_clip =
            snapshotTrainingConfigField<float>(snapshot, "stability_override_clip_per_token");
        if (override_clip > 0.0f) {
            grad_clip_norm = override_clip;
        }
    }
    return grad_clip_norm;
}

inline int snapshotTokenizerTargetVocabSize(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    int target_vocab_size = snapshotTrainingConfigField<int>(snapshot, "tokenizer_target_vocab_size");
    const int max_vocab_size = snapshotTrainingConfigField<int>(snapshot, "tokenizer_max_vocab_size");
    if (max_vocab_size > 0 && target_vocab_size > max_vocab_size) {
        target_vocab_size = max_vocab_size;
    }
    return target_vocab_size;
}

inline std::vector<int> snapshotMaskedNumericLiteralIds()
{
    std::vector<int> ids;
    ids.reserve(10);
    for (int byte_value = 0x30; byte_value <= 0x39; ++byte_value) {
        ids.push_back(Tokenizer::BYTE_TOKEN_OFFSET + byte_value);
    }
    return ids;
}

inline const char* modelExecutionModeToJsonString(ModelExecutionMode mode)
{
    switch (mode) {
        case ModelExecutionMode::TRAINING: return "training";
        case ModelExecutionMode::INFERENCE: return "inference";
    }
    throw std::runtime_error("modelExecutionModeToJsonString: unsupported ModelExecutionMode value");
}

inline ModelExecutionMode parseModelExecutionMode(const std::string& value)
{
    const std::string normalized = normalizeHyperparameterEnumToken(value);
    if (normalized == "training") {
        return ModelExecutionMode::TRAINING;
    }
    if (normalized == "inference") {
        return ModelExecutionMode::INFERENCE;
    }
    throw std::runtime_error("parseModelExecutionMode: expected training or inference, got '" + value + "'");
}

inline ModelExecutionMode snapshotExecutionMode(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    return parseModelExecutionMode(snapshotTrainingConfigField<std::string>(snapshot, "execution_mode"));
}

inline nlohmann::json buildFinalizedTrainingConfigDocument(
    const nlohmann::json& document,
    int argc,
    char** argv,
    ModelExecutionMode execution_mode)
{
    const LanguageModelConfig config = finalizeLanguageModelConfig(
        document,
        argc,
        argv,
        execution_mode);
    nlohmann::json finalized_config = nlohmann::json::object();

#define GRIM_WRITE_FINAL_CONFIG_FIELD(member) finalized_config[#member] = config.member
    GRIM_WRITE_FINAL_CONFIG_FIELD(d_model);
    GRIM_WRITE_FINAL_CONFIG_FIELD(num_layers);
    GRIM_WRITE_FINAL_CONFIG_FIELD(num_heads);
    GRIM_WRITE_FINAL_CONFIG_FIELD(num_kv_heads);
    GRIM_WRITE_FINAL_CONFIG_FIELD(d_ff);
    GRIM_WRITE_FINAL_CONFIG_FIELD(max_seq_len);
    GRIM_WRITE_FINAL_CONFIG_FIELD(dropout_rate);
    GRIM_WRITE_FINAL_CONFIG_FIELD(embedding_scale);
    GRIM_WRITE_FINAL_CONFIG_FIELD(attention_dropout);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tie_embeddings);
    GRIM_WRITE_FINAL_CONFIG_FIELD(positional_encoding);
    GRIM_WRITE_FINAL_CONFIG_FIELD(rope_base_seq_len);
    GRIM_WRITE_FINAL_CONFIG_FIELD(alibi_min_locality_distance);
    GRIM_WRITE_FINAL_CONFIG_FIELD(alibi_slope_exponent);
    GRIM_WRITE_FINAL_CONFIG_FIELD(alibi_max_bias);
    GRIM_WRITE_FINAL_CONFIG_FIELD(rope_theta);
    GRIM_WRITE_FINAL_CONFIG_FIELD(rope_scaling);
    GRIM_WRITE_FINAL_CONFIG_FIELD(pbm_alibi_slopes);
    GRIM_WRITE_FINAL_CONFIG_FIELD(pbm_rope_inv_freq);
    GRIM_WRITE_FINAL_CONFIG_FIELD(use_flash_attention);
    GRIM_WRITE_FINAL_CONFIG_FIELD(min_seq_len_for_flash);
    GRIM_WRITE_FINAL_CONFIG_FIELD(use_gpu);
    GRIM_WRITE_FINAL_CONFIG_FIELD(head_dim);
    GRIM_WRITE_FINAL_CONFIG_FIELD(heads_per_kv_group);
    GRIM_WRITE_FINAL_CONFIG_FIELD(kv_dim);
    GRIM_WRITE_FINAL_CONFIG_FIELD(qkv_dim);
    GRIM_WRITE_FINAL_CONFIG_FIELD(rotary_dim);
    GRIM_WRITE_FINAL_CONFIG_FIELD(is_gqa);
    GRIM_WRITE_FINAL_CONFIG_FIELD(residual_projection_init_gain);
    GRIM_WRITE_FINAL_CONFIG_FIELD(vocab_size);
    GRIM_WRITE_FINAL_CONFIG_FIELD(max_cached_seq_len);
    GRIM_WRITE_FINAL_CONFIG_FIELD(max_tokens_per_batch);
    GRIM_WRITE_FINAL_CONFIG_FIELD(rms_epsilon);
    GRIM_WRITE_FINAL_CONFIG_FIELD(causal_mask);
    GRIM_WRITE_FINAL_CONFIG_FIELD(use_pre_norm);
    GRIM_WRITE_FINAL_CONFIG_FIELD(fuse_qkv);
    GRIM_WRITE_FINAL_CONFIG_FIELD(use_bias);
    GRIM_WRITE_FINAL_CONFIG_FIELD(lm_head_unigram_bias);
    GRIM_WRITE_FINAL_CONFIG_FIELD(qk_norm_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(attention_off_by_one);
    GRIM_WRITE_FINAL_CONFIG_FIELD(use_layer_scale);
    GRIM_WRITE_FINAL_CONFIG_FIELD(layer_scale_init);
    GRIM_WRITE_FINAL_CONFIG_FIELD(data_path);
    GRIM_WRITE_FINAL_CONFIG_FIELD(vocab_path);
    GRIM_WRITE_FINAL_CONFIG_FIELD(output_model_path);
    GRIM_WRITE_FINAL_CONFIG_FIELD(checkpoint_dir);
    GRIM_WRITE_FINAL_CONFIG_FIELD(log_dir);
    GRIM_WRITE_FINAL_CONFIG_FIELD(status_path);
    GRIM_WRITE_FINAL_CONFIG_FIELD(save_test_mode);
    GRIM_WRITE_FINAL_CONFIG_FIELD(parameter_precision_embedding);
    GRIM_WRITE_FINAL_CONFIG_FIELD(parameter_precision_lm_head);
    GRIM_WRITE_FINAL_CONFIG_FIELD(parameter_precision_attention);
    GRIM_WRITE_FINAL_CONFIG_FIELD(parameter_precision_ffn);
    GRIM_WRITE_FINAL_CONFIG_FIELD(parameter_precision_rmsnorm);
    GRIM_WRITE_FINAL_CONFIG_FIELD(parameter_precision_execution_block);
    GRIM_WRITE_FINAL_CONFIG_FIELD(use_atom_data);
    GRIM_WRITE_FINAL_CONFIG_FIELD(atom_embedding_dim);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_layer);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_num_ops);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_num_slots);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_num_scratch_slots);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_num_steps);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_value_decode_input_dim);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_value_decode_hidden_dim);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_d_key);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_d_type);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_cross_attn_head_dim);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_cross_attn_topk);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_usage_decay);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_inject_gate_temp);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_result_slot_mode);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_result_slot_index);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_debug_mode);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_entropy_collapse_threshold);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_write_collapse_threshold);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_magnitude_limit);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_diversity_kappa);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_temp_start);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_temp_end);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_temp_schedule);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_entropy_weight);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_transition_hard_threshold);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_gate_warmup_steps);
    GRIM_WRITE_FINAL_CONFIG_FIELD(execution_block_causal_w1_transition);
    GRIM_WRITE_FINAL_CONFIG_FIELD(div_invalid_penalty_weight);
    GRIM_WRITE_FINAL_CONFIG_FIELD(div_magnitude_penalty_weight);
    GRIM_WRITE_FINAL_CONFIG_FIELD(arg_reinforce_weight);
    GRIM_WRITE_FINAL_CONFIG_FIELD(arg_reinforce_baseline_decay);
    GRIM_WRITE_FINAL_CONFIG_FIELD(structured_ce_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(structured_ce_weight);
    GRIM_WRITE_FINAL_CONFIG_FIELD(number_encoder_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(number_encoder_max_digit_slots);
    GRIM_WRITE_FINAL_CONFIG_FIELD(number_encoder_d_hidden);
    GRIM_WRITE_FINAL_CONFIG_FIELD(number_encoder_max_abs_pow10);
    GRIM_WRITE_FINAL_CONFIG_FIELD(step_x_multiplier);
    GRIM_WRITE_FINAL_CONFIG_FIELD(step_y_multiplier);
    GRIM_WRITE_FINAL_CONFIG_FIELD(step_y_overrides_x);
    GRIM_WRITE_FINAL_CONFIG_FIELD(entropy_aux_weight);
    GRIM_WRITE_FINAL_CONFIG_FIELD(value_match_epsilon);
    GRIM_WRITE_FINAL_CONFIG_FIELD(final_slot_consistency_weight);
    GRIM_WRITE_FINAL_CONFIG_FIELD(lm_head_center_hidden_states);
    GRIM_WRITE_FINAL_CONFIG_FIELD(lm_head_mlp_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(lm_head_mlp_d_ff);
    GRIM_WRITE_FINAL_CONFIG_FIELD(lm_head_mlp_alpha);
    GRIM_WRITE_FINAL_CONFIG_FIELD(freeze_learned_rms_gammas);
    GRIM_WRITE_FINAL_CONFIG_FIELD(project_out_pc1);
    GRIM_WRITE_FINAL_CONFIG_FIELD(pc1_power_iters);
    GRIM_WRITE_FINAL_CONFIG_FIELD(center_logits);
    GRIM_WRITE_FINAL_CONFIG_FIELD(center_encoder_residuals);
    GRIM_WRITE_FINAL_CONFIG_FIELD(hardcoded_hidden_pattern);
    GRIM_WRITE_FINAL_CONFIG_FIELD(hardcoded_log_every_n_batches);
    GRIM_WRITE_FINAL_CONFIG_FIELD(generation_strategy);
    GRIM_WRITE_FINAL_CONFIG_FIELD(generation_max_new_tokens);
    GRIM_WRITE_FINAL_CONFIG_FIELD(generation_min_new_tokens);
    GRIM_WRITE_FINAL_CONFIG_FIELD(generation_temperature);
    GRIM_WRITE_FINAL_CONFIG_FIELD(generation_top_k);
    GRIM_WRITE_FINAL_CONFIG_FIELD(generation_top_p);
    GRIM_WRITE_FINAL_CONFIG_FIELD(generation_min_p);
    GRIM_WRITE_FINAL_CONFIG_FIELD(generation_typical_p);
    GRIM_WRITE_FINAL_CONFIG_FIELD(generation_repetition_penalty);
    GRIM_WRITE_FINAL_CONFIG_FIELD(generation_repetition_penalty_window);
    GRIM_WRITE_FINAL_CONFIG_FIELD(generation_frequency_penalty);
    GRIM_WRITE_FINAL_CONFIG_FIELD(generation_presence_penalty);
    GRIM_WRITE_FINAL_CONFIG_FIELD(generation_no_repeat_ngram_size);
    GRIM_WRITE_FINAL_CONFIG_FIELD(generation_do_sample);
    GRIM_WRITE_FINAL_CONFIG_FIELD(generation_num_return_sequences);
    GRIM_WRITE_FINAL_CONFIG_FIELD(generation_eos_token_id);
    GRIM_WRITE_FINAL_CONFIG_FIELD(generation_pad_token_id);
    GRIM_WRITE_FINAL_CONFIG_FIELD(generation_bos_token_id);
    GRIM_WRITE_FINAL_CONFIG_FIELD(generation_unk_token_id);
    GRIM_WRITE_FINAL_CONFIG_FIELD(generation_bad_words_ids);
    GRIM_WRITE_FINAL_CONFIG_FIELD(generation_masked_numeric_literal_ids);
    GRIM_WRITE_FINAL_CONFIG_FIELD(generation_seed);
    GRIM_WRITE_FINAL_CONFIG_FIELD(generation_enable_scratchblock_reasoning);
    GRIM_WRITE_FINAL_CONFIG_FIELD(current_model_training);
    GRIM_WRITE_FINAL_CONFIG_FIELD(current_curriculum);
    GRIM_WRITE_FINAL_CONFIG_FIELD(log_recorder_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(log_recorder_default_level);
    GRIM_WRITE_FINAL_CONFIG_FIELD(log_recorder_modules);
    GRIM_WRITE_FINAL_CONFIG_FIELD(log_recorder_layer_embedding);
    GRIM_WRITE_FINAL_CONFIG_FIELD(log_recorder_layer_rms_norm);
    GRIM_WRITE_FINAL_CONFIG_FIELD(log_recorder_layer_attention);
    GRIM_WRITE_FINAL_CONFIG_FIELD(log_recorder_layer_feed_forward);
    GRIM_WRITE_FINAL_CONFIG_FIELD(log_recorder_layer_residual);
    GRIM_WRITE_FINAL_CONFIG_FIELD(log_recorder_layer_encoding);
    GRIM_WRITE_FINAL_CONFIG_FIELD(log_recorder_layer_serialization);
    GRIM_WRITE_FINAL_CONFIG_FIELD(log_recorder_layer_execution_block);
    GRIM_WRITE_FINAL_CONFIG_FIELD(logging_default_level);
    GRIM_WRITE_FINAL_CONFIG_FIELD(logging_equation_csv_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(logging_stderr_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(logging_initial_capacity);
    GRIM_WRITE_FINAL_CONFIG_FIELD(logging_group_overrides);
    GRIM_WRITE_FINAL_CONFIG_FIELD(epochs);
    GRIM_WRITE_FINAL_CONFIG_FIELD(seed);
    GRIM_WRITE_FINAL_CONFIG_FIELD(batch_size);
    GRIM_WRITE_FINAL_CONFIG_FIELD(gradient_accumulation_steps);
    GRIM_WRITE_FINAL_CONFIG_FIELD(single_batch_overfit_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(single_batch_overfit_max_steps);
    GRIM_WRITE_FINAL_CONFIG_FIELD(batch_strategy);
    GRIM_WRITE_FINAL_CONFIG_FIELD(learning_rate);
    GRIM_WRITE_FINAL_CONFIG_FIELD(weight_decay);
    GRIM_WRITE_FINAL_CONFIG_FIELD(use_depth_aware_upsilon);
    GRIM_WRITE_FINAL_CONFIG_FIELD(grad_clip_norm);
    GRIM_WRITE_FINAL_CONFIG_FIELD(grad_clip_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(effective_per_token_grad_limit);
    GRIM_WRITE_FINAL_CONFIG_FIELD(per_token_grad_scale);
    GRIM_WRITE_FINAL_CONFIG_FIELD(warmup_fraction);
    GRIM_WRITE_FINAL_CONFIG_FIELD(warmup_steps);
    GRIM_WRITE_FINAL_CONFIG_FIELD(force_rebuild_vocab);
    GRIM_WRITE_FINAL_CONFIG_FIELD(cosine_decay_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(cosine_warm_restarts);
    GRIM_WRITE_FINAL_CONFIG_FIELD(cosine_decay_min_lr);
    GRIM_WRITE_FINAL_CONFIG_FIELD(sliding_window_stride);
    GRIM_WRITE_FINAL_CONFIG_FIELD(min_seq_valid_tokens);
    GRIM_WRITE_FINAL_CONFIG_FIELD(log_interval);
    GRIM_WRITE_FINAL_CONFIG_FIELD(atom_stats_interval);
    GRIM_WRITE_FINAL_CONFIG_FIELD(atom_stats_max_seqs);
    GRIM_WRITE_FINAL_CONFIG_FIELD(inference_diagnostic_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(inference_diagnostic_interval);
    GRIM_WRITE_FINAL_CONFIG_FIELD(validation_interval);
    GRIM_WRITE_FINAL_CONFIG_FIELD(checkpoint_interval);
    GRIM_WRITE_FINAL_CONFIG_FIELD(soft_restart_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(soft_restart_loss_increase_threshold);
    GRIM_WRITE_FINAL_CONFIG_FIELD(soft_restart_max_step_window);
    GRIM_WRITE_FINAL_CONFIG_FIELD(soft_restart_cooldown_steps);
    GRIM_WRITE_FINAL_CONFIG_FIELD(auto_stop_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(auto_stop_plateau_patience);
    GRIM_WRITE_FINAL_CONFIG_FIELD(auto_stop_plateau_min_delta);
    GRIM_WRITE_FINAL_CONFIG_FIELD(auto_stop_high_loss_threshold);
    GRIM_WRITE_FINAL_CONFIG_FIELD(auto_stop_high_loss_patience);
    GRIM_WRITE_FINAL_CONFIG_FIELD(shuffle_train_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(shuffle_train_epochs);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_control_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_spike_mild_threshold);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_spike_moderate_threshold);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_spike_severe_threshold);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_moderate_grad_scale);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_moderate_cooldown_extension);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_min_grad_for_nonzero_loss);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_loss_threshold_for_grad_check);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_max_consecutive_zero_grad_steps);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_seq_len_regime_change_threshold);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_regime_change_suppression_steps);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_volatility_damping_threshold);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_max_volatility_damping);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_gradient_decay_threshold);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_max_decay_boost);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_progress_boost_threshold);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_max_progress_boost);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_outlier_frequency_trigger);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_outlier_persistence_trigger);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_anchor_drift_sigma_multiplier);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_soft_restart_cooldown_steps);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_warmup_steps);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_baseline_stabilization_steps);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_verbose_logging);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_fail_loud_on_accumulation_bug);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_plateau_noise_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_plateau_noise_patience);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_plateau_noise_variance_threshold);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_plateau_noise_std);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_plateau_noise_proportional);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_plateau_noise_cooldown);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_plateau_noise_max_per_epoch);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_lattice_num_levels);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_lattice_num_streams);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_lattice_beta_mu);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_lattice_beta_a);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_lattice_beta_delta);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_lattice_beta_r);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_lattice_beta_run);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_lattice_beta_v);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_lattice_k_out0);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_lattice_alpha_v);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_lattice_epsilon);
    GRIM_WRITE_FINAL_CONFIG_FIELD(telemetry_lattice_strict_mode);
    GRIM_WRITE_FINAL_CONFIG_FIELD(loss_label_smoothing_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(loss_label_smoothing_epsilon);
    GRIM_WRITE_FINAL_CONFIG_FIELD(loss_focal_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(loss_focal_gamma);
    GRIM_WRITE_FINAL_CONFIG_FIELD(loss_focal_alpha);
    GRIM_WRITE_FINAL_CONFIG_FIELD(loss_preference_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(loss_preference_beta);
    GRIM_WRITE_FINAL_CONFIG_FIELD(loss_distillation_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(loss_distillation_temperature);
    GRIM_WRITE_FINAL_CONFIG_FIELD(loss_distillation_lambda);
    GRIM_WRITE_FINAL_CONFIG_FIELD(loss_masking_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(loss_masking_tag);
    GRIM_WRITE_FINAL_CONFIG_FIELD(loss_entropy_reg_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(loss_entropy_reg_lambda);
    GRIM_WRITE_FINAL_CONFIG_FIELD(loss_class_balanced_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(loss_class_balanced_beta);
    GRIM_WRITE_FINAL_CONFIG_FIELD(lm_head_centering_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(embedding_freeze_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(embedding_freeze_after_step);
    GRIM_WRITE_FINAL_CONFIG_FIELD(optimizer_kind);
    GRIM_WRITE_FINAL_CONFIG_FIELD(optimizer_beta1);
    GRIM_WRITE_FINAL_CONFIG_FIELD(optimizer_beta2);
    GRIM_WRITE_FINAL_CONFIG_FIELD(optimizer_epsilon);
    GRIM_WRITE_FINAL_CONFIG_FIELD(optimizer_embedding_freeze_after_step);
    GRIM_WRITE_FINAL_CONFIG_FIELD(stability_overrides_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(stability_override_batch_size);
    GRIM_WRITE_FINAL_CONFIG_FIELD(stability_override_max_seq_len);
    GRIM_WRITE_FINAL_CONFIG_FIELD(stability_override_clip_per_token);
    GRIM_WRITE_FINAL_CONFIG_FIELD(single_stream_mode);
    GRIM_WRITE_FINAL_CONFIG_FIELD(disable_async_frees);
    GRIM_WRITE_FINAL_CONFIG_FIELD(synchronize_after_kernels);
    GRIM_WRITE_FINAL_CONFIG_FIELD(prediction_comparison_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(prediction_comparison_interval);
    GRIM_WRITE_FINAL_CONFIG_FIELD(prediction_comparison_top_k);
    GRIM_WRITE_FINAL_CONFIG_FIELD(prediction_comparison_max_positions);
    GRIM_WRITE_FINAL_CONFIG_FIELD(prediction_comparison_log_path);
    GRIM_WRITE_FINAL_CONFIG_FIELD(logit_update_trace_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(logit_update_trace_interval);
    GRIM_WRITE_FINAL_CONFIG_FIELD(attention_diag_enabled);
    GRIM_WRITE_FINAL_CONFIG_FIELD(attention_diag_layer);
    GRIM_WRITE_FINAL_CONFIG_FIELD(attention_diag_head);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_enable_atom_reasoning);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_detect_numbers);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_target_vocab_size);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_max_vocab_size);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_max_length);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_character_coverage);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_min_cleaned_text_length);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_min_subword_freq);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_prune_during_mining);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_enable_parallel_subword_mining);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_subword_mining_workers);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_subword_mining_max_bytes);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_model_type);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_special_tokens);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_add_bos);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_add_eos);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_unk_token);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_pad_token);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_bos_token);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_eos_token);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_enable_nfkc_normalization);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_enable_lowercasing);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_enable_parallel_tokenization);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_parallel_threshold);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_enable_byte_fallback);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_expected_checksum);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_save_text_vocab);
    GRIM_WRITE_FINAL_CONFIG_FIELD(tokenizer_vocab_score_multiplier);
    GRIM_WRITE_FINAL_CONFIG_FIELD(clear_merged_cache_on_merge);
    GRIM_WRITE_FINAL_CONFIG_FIELD(subprocess_tokenizer_only_mode);
#undef GRIM_WRITE_FINAL_CONFIG_FIELD

    finalized_config["execution_mode"] = modelExecutionModeToJsonString(config.execution_mode);
    finalized_config["grim_text_training_data"] = config.data_path;
    finalized_config["grim_text_vocab"] = config.vocab_path;
    finalized_config["grim_text_model"] = config.output_model_path;
    finalized_config["grim_text_checkpoints"] = config.checkpoint_dir;
    finalized_config["grim_text_logs"] = config.log_dir;
    finalized_config["grim_text_training_status"] = config.status_path;
    return finalized_config;
}

inline void writeFinalizedTrainingConfigDocumentToSnapshot(
    GRIM::Config::AiConfigSnapshot& snapshot,
    const nlohmann::json& finalized_config)
{
    auto& cfg = mutableSnapshotTrainingConfig(snapshot);
    for (const auto& entry : finalized_config.items()) {
        cfg[entry.key()] = entry.value();
    }
}

inline LanguageModelConfig finalizeLanguageModelConfig(
    const nlohmann::json& document,
    int argc,
    char** argv,
    ModelExecutionMode execution_mode)
{
    // 1. Root registry + resolved path leaves from the raw snapshot document.
    LanguageModelConfig config = loadLanguageModelConfig(document);
    loadResolvedPathFields(config, document);

    // 2. Resolve max_seq_len (must be configured)
    if (config.stability_overrides_enabled &&
        config.stability_override_max_seq_len > 0) {
        config.max_seq_len = config.stability_override_max_seq_len;
    } else if (config.max_seq_len > 0) {
        // already authored on the root; keep it
    } else {
        throw std::runtime_error(
            "FATAL: max_seq_len not configured in ai_config.json "
            "(stability overrides or root config)");
    }

    // 3. Resolve configured stride against the effective max_seq_len.
    if (config.sliding_window_stride <= 0) {
        throw std::runtime_error(
            "FATAL: sliding_window_stride must be > 0 in ai_config.json, got " +
            std::to_string(config.sliding_window_stride));
    }
    if (config.sliding_window_stride > config.max_seq_len) {
        throw std::runtime_error(
            "FATAL: sliding_window_stride=" +
            std::to_string(config.sliding_window_stride) +
            " exceeds effective max_seq_len=" +
            std::to_string(config.max_seq_len) +
            ". Update training.config.sliding_window_stride or stability_overrides.max_seq_len.");
    }

    // 4. Stability overrides (batch_size, grad_clip_norm)
    if (config.stability_overrides_enabled) {
        if (config.stability_override_batch_size <= 0) {
            throw std::runtime_error(
                "FATAL: stability_overrides enabled but "
                "stability_override_batch_size=" +
                std::to_string(config.stability_override_batch_size) +
                " (must be > 0)");
        }
        config.batch_size = config.stability_override_batch_size;
        if (config.stability_override_clip_per_token > 0.0f) {
            config.grad_clip_norm = config.stability_override_clip_per_token;
        }
    }
    // 5. CLI overrides
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            config.data_path = argv[++i];
        } else if (arg == "--vocab" && i + 1 < argc) {
            config.vocab_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_model_path = argv[++i];
        } else if (arg == "--epochs" && i + 1 < argc) {
            config.epochs = std::atoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            config.batch_size = std::atoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            config.learning_rate =
                static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--save-test") {
            config.save_test_mode = true;
        } else if (arg == "--config") {
            throw std::runtime_error(
                "finalizeLanguageModelConfig: --config is no longer supported; use the canonical ai_config.json");
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

    refreshMutableTrainingDerivedValues(
        config,
        config.max_seq_len,
        "finalizeLanguageModelConfig");

    config.vocab_size = config.tokenizer_target_vocab_size;
    config.execution_mode = execution_mode;
    config.causal_mask = true;
    config.use_pre_norm = true;
    config.fuse_qkv = true;

    const char* caller = nullptr;
    switch (execution_mode) {
        case ModelExecutionMode::TRAINING:
            caller = "finalizeLanguageModelConfig(TRAINING)";
            config.max_cached_seq_len = config.max_seq_len;
            break;
        case ModelExecutionMode::INFERENCE:
            caller = "finalizeLanguageModelConfig(INFERENCE)";
            config.use_gpu = true;
            config.batch_size = 1;
            config.max_cached_seq_len = config.max_seq_len;
            config.max_tokens_per_batch = config.max_seq_len;
            break;
        default:
            throw std::runtime_error("finalizeLanguageModelConfig: unsupported ModelExecutionMode value");
    }

    config.computeDerivedValues();
    validateRootConfigDocument(config, caller);
    return config;
}

inline GRIM::Config::AiConfigSnapshot finalizeAiConfigSnapshot(
    GRIM::Config::AiConfigSnapshot snapshot,
    int argc,
    char** argv,
    ModelExecutionMode execution_mode)
{
    writeFinalizedTrainingConfigDocumentToSnapshot(
        snapshot,
        buildFinalizedTrainingConfigDocument(
            snapshot.document,
            argc,
            argv,
            execution_mode));
    return snapshot;
}

inline void setSnapshotRuntimeVocabSize(
    GRIM::Config::AiConfigSnapshot& snapshot,
    int vocab_size,
    const char* caller)
{
    if (vocab_size < static_cast<int>(Tokenizer::UNIGRAM_VOCAB_OFFSET)) {
        throw std::runtime_error(std::string(caller) +
            ": vocab_size must include special+byte+atom ranges (>= " +
            std::to_string(Tokenizer::UNIGRAM_VOCAB_OFFSET) + "), got " +
            std::to_string(vocab_size));
    }
    mutableSnapshotTrainingConfig(snapshot).at("vocab_size") = vocab_size;
}

inline LanguageModelConfig loadStartupConfig(
    int argc,
    char** argv,
    ModelExecutionMode execution_mode)
{
    const auto snapshot = GRIM::Config::loadAiConfigSnapshot();
    return finalizeLanguageModelConfig(
        snapshot.document,
        argc,
        argv,
        execution_mode);
}
} // namespace HyperParameters
} // namespace GRIM

#endif // GRIM_SHARED_HYPERPARAMETERS_GPU_HPP
