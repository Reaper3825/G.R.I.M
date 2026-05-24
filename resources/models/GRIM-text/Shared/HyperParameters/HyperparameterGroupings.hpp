#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "HyperParameters_GPU.hpp"
#include "../Dynamic_LR/LRSchedule.hpp"

namespace GRIM::HyperParameters {

// Groupings are read views over HyperParameters_GPU.hpp-owned config and constants.
// Do not introduce authored defaults here; add them to HyperParameters_GPU.hpp first.

struct DataLoadingHP {
    int min_seq_valid_tokens = 0;
    int sliding_window_stride = 0;
};

struct TokenizerHP {
    std::string data_path;
    std::string vocab_path;

    int target_vocab_size = 0;
    float character_coverage = 0.0f;
    int min_cleaned_text_length = 0;
    int min_subword_freq = 0;
    bool prune_during_mining = false;
    bool enable_parallel_subword_mining = false;
    int subword_mining_workers = 0;
    std::size_t subword_mining_max_bytes = 0;

    bool enable_scratch_block_reasoning = false;
    bool detect_numbers = false;
    bool enable_byte_fallback = false;

    bool add_bos = false;
    bool add_eos = false;
    bool force_rebuild_vocab = false;
    bool save_text_vocab = false;
    float vocab_score_multiplier = 0.0f;

    std::string current_curriculum;
    std::string current_model_training;
    int execution_block_num_steps = 0;
};

struct TokenizerSubprocessHP {
    std::string config_path;
    TokenizerHP tokenizer;
    bool only_mode = false;
};

struct LearningRateScheduleInputs {
    float learning_rate = 0.0f;
    float cosine_decay_min_lr = 0.0f;
    int warmup_steps = 0;
    bool cosine_decay_enabled = false;
    bool cosine_warm_restarts = false;
};

enum class OptimizerKind {
    ADAMW,
    RADAMW
};

struct OptimizerUpdateHP {
    OptimizerKind kind = OptimizerKind::ADAMW;
    float weight_decay = 0.0f;
    float beta1 = 0.0f;
    float beta2 = 0.0f;
    float epsilon = 0.0f;
    int embedding_freeze_after_step = -1;
};

struct GradientClippingHP {
    bool enabled = false;
    float configured_clip_norm = 0.0f;
    float effective_per_token_limit = EPSILON_GRADIENT_CLIP;
};

struct LossConfigHP {
    float focal_alpha = 0.0f;
    float focal_gamma = 0.0f;
    bool focal_enabled = false;

    float smoothing_epsilon = 0.0f;
    bool smoothing_enabled = false;

    float entropy_reg_lambda = 0.0f;
    bool entropy_reg_enabled = false;

    bool class_balanced_enabled = false;
    float class_balanced_beta = 0.0f;
};

struct TrainingFixedShapeHP {
    int batch_size = 0;
    int max_seq_len = 0;
    int max_tokens_per_batch = 0;
};

struct GpuModelInitializationHP {
    bool use_gpu = false;
    int num_layers = 0;
    bool use_flash_attention = false;
    int min_seq_len_for_flash = 0;
};

struct PBMConstructionHP {
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int rotary_dim = 0;
    int max_seq_len = 0;
    int rope_base_seq_len = 0;
    int alibi_min_locality_distance = 0;
    float alibi_slope_exponent = 0.0f;
    float alibi_max_bias = 0.0f;
    float rope_theta = 0.0f;
    float rope_scaling = 0.0f;
};

struct EncoderLayerConstructionHP {
    int num_layers = 0;
    int d_model = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int heads_per_kv_group = 0;
    int kv_dim = 0;
    int qkv_dim = 0;
    int d_ff = 0;
    float rms_epsilon = 0.0f;
    bool causal_mask = false;
    bool use_flash_attention = false;
    int min_seq_len_for_flash = 0;
    bool use_layer_scale = false;
    float layer_scale_init = 0.0f;
    bool center_encoder_residuals = false;
    bool use_bias = false;
    float dropout_rate = 0.0f;
    float attention_dropout = 0.0f;
    bool qk_norm_enabled = false;
    float residual_projection_init_gain = 0.0f;
    bool is_gqa = false;
    bool freeze_learned_rms_gammas = false;
};

struct EncoderSelfAttentionHP {
    int d_model = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int heads_per_kv_group = 0;
    int kv_dim = 0;
    int qkv_dim = 0;
    bool causal_mask = false;
    bool use_flash_attention = false;
    int min_seq_len_for_flash = 0;
    bool use_bias = false;
    float attention_dropout = 0.0f;
    bool qk_norm_enabled = false;
    bool is_gqa = false;
};

struct FlashAttentionRuntimeHP {
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    bool causal = false;
    bool is_bf16 = false;
    bool requires_alibi = false;
};

struct FeedForwardLayerConstructionHP {
    int d_model = 0;
    int d_ff = 0;
    bool use_bias = false;
    float dropout_rate = 0.0f;
    float residual_projection_init_gain = 0.0f;
};

struct EmbeddingLayerConstructionHP {
    int vocab_size = 0;
    int d_model = 0;
};

struct LMHeadLayerConstructionHP {
    int d_model = 0;
    int vocab_size = 0;
    int training_batch_size = 0;
    int training_rows_per_sequence = 0;
    bool use_bias = false;
    bool tie_embeddings = false;
    bool center_hidden_states = false;
    bool project_out_pc1 = false;
    int pc1_power_iters = 0;
    bool center_logits = false;
    bool freeze_learned_rms_gammas = false;
    float rms_epsilon = 0.0f;
};

struct ScratchBlockConstructionHP {
    bool enabled = false;
    int d_model = 0;
    int max_atoms = 0;
    int atom_embedding_dim = 0;
    int atom_token_start = ATOM_TOKEN_START;
    int atom_token_end = ATOM_TOKEN_END;
    float atom_scale = 0.0f;
};

struct ReasoningHeadConstructionHP {
    bool enabled = false;
    int d_model = 0;
    int atom_embedding_dim = 0;
    int num_ops = 0;
};

struct ExecutionBlockConstructionHP {
    bool enabled = false;
    int layer = -1;
    int d_model = 0;
    int atom_embedding_dim = 0;
    int num_ops = 0;
    int num_slots = 0;
    int num_scratch_slots = 0;
    int num_exec_steps = 0;
    int value_decode_input_dim = 0;
    int value_decode_hidden_dim = 0;
    int d_key = 0;
    int d_type = 0;
    int cross_attn_head_dim = 0;
    int cross_attn_topk = 0;
    float usage_decay = 0.0f;
    float inject_gate_temp = 0.0f;
    int result_slot_mode = 0;
    int result_slot_index = 0;
    bool debug_mode = false;
    float entropy_collapse_threshold = 0.0f;
    float write_collapse_threshold = 0.0f;
    float magnitude_limit = 0.0f;
    float diversity_kappa = 0.0f;
    float temp_start = 0.0f;
    float temp_end = 0.0f;
    int temp_schedule = 0;
    float entropy_weight = 0.0f;
    float transition_hard_threshold = 0.0f;
    int gate_warmup_steps = 0;
    float causal_w1_transition = 0.0f;
    float div_invalid_penalty_weight = 0.0f;
    float div_magnitude_penalty_weight = 0.0f;
    float arg_reinforce_weight = 0.0f;
    float arg_reinforce_baseline_decay = 0.0f;
};

struct DecodeTimeSelectorConstructionHP {
    bool enabled = false;
    int d_model = 0;
    int d_selector = 0;
    int d_slot_features = 0;
    int num_slots = 0;
    int scratch_slots = 0;
    float selection_margin = 0.0f;
    float supervision_weight = 0.0f;
};

struct MTPConstructionHP {
    bool enabled = false;
    int k = 0;
    int vocab_size = 0;
    int d_model = 0;
    float alpha = 0.0f;
    int alpha_warmup_steps = 0;
};

struct MTPFeatureHP {
    bool enabled = false;
    int k = 0;
};

inline void requirePositiveGroupingValue(int value,
                                         const char* field,
                                         const char* caller)
{
    if (value <= 0) {
        throw std::runtime_error(std::string(caller) + ": " + field +
                                 " must be > 0, got " + std::to_string(value));
    }
}

inline void requirePositiveFiniteGroupingValue(float value,
                                               const char* field,
                                               const char* caller)
{
    if (!std::isfinite(value) || value <= 0.0f) {
        throw std::runtime_error(std::string(caller) + ": " + field +
                                 " must be a positive finite value, got " +
                                 std::to_string(value));
    }
}

inline void requireDropoutProbability(float value,
                                      const char* field,
                                      const char* caller)
{
    if (value < 0.0f || value >= 1.0f) {
        throw std::runtime_error(std::string(caller) + ": " + field +
                                 " must be in [0, 1), got " + std::to_string(value));
    }
}

inline void requireValidGQAGrouping(const LanguageModelConfig& cfg,
                                    const char* caller)
{
    requirePositiveGroupingValue(cfg.d_model, "d_model", caller);
    requirePositiveGroupingValue(cfg.num_heads, "num_heads", caller);
    requirePositiveGroupingValue(cfg.num_kv_heads, "num_kv_heads", caller);
    if (!isValidGQAConfig(cfg.num_heads, cfg.num_kv_heads)) {
        throw std::runtime_error(std::string(caller) + ": invalid GQA config num_heads=" +
                                 std::to_string(cfg.num_heads) + " num_kv_heads=" +
                                 std::to_string(cfg.num_kv_heads));
    }
    if (cfg.d_model % cfg.num_heads != 0) {
        throw std::runtime_error(std::string(caller) + ": d_model=" +
                                 std::to_string(cfg.d_model) +
                                 " must be divisible by num_heads=" +
                                 std::to_string(cfg.num_heads));
    }
    const int computed_head_dim = cfg.d_model / cfg.num_heads;
    if (cfg.head_dim != computed_head_dim) {
        throw std::runtime_error(std::string(caller) + ": head_dim=" +
                                 std::to_string(cfg.head_dim) +
                                 " does not match d_model/num_heads=" +
                                 std::to_string(computed_head_dim));
    }
}

inline void validateLossConfigHP(
    const LossConfigHP& hp,
    const char* caller)
{
    if (!std::isfinite(hp.focal_alpha) || hp.focal_alpha <= 0.0f) {
        throw std::runtime_error(std::string(caller) + ": focal_alpha must be a positive finite value, got " +
                                 std::to_string(hp.focal_alpha));
    }
    if (!std::isfinite(hp.focal_gamma) || hp.focal_gamma < 0.0f) {
        throw std::runtime_error(std::string(caller) + ": focal_gamma must be finite and >= 0, got " +
                                 std::to_string(hp.focal_gamma));
    }
    if (!std::isfinite(hp.smoothing_epsilon) || hp.smoothing_epsilon < 0.0f || hp.smoothing_epsilon >= 1.0f) {
        throw std::runtime_error(std::string(caller) + ": smoothing_epsilon must be in [0, 1), got " +
                                 std::to_string(hp.smoothing_epsilon));
    }
    if (!std::isfinite(hp.entropy_reg_lambda) || hp.entropy_reg_lambda < 0.0f) {
        throw std::runtime_error(std::string(caller) + ": entropy_reg_lambda must be finite and >= 0, got " +
                                 std::to_string(hp.entropy_reg_lambda));
    }
    if (!std::isfinite(hp.class_balanced_beta) || hp.class_balanced_beta <= 0.0f) {
        throw std::runtime_error(std::string(caller) + ": class_balanced_beta must be a positive finite value, got " +
                                 std::to_string(hp.class_balanced_beta));
    }
}

inline void validateEncoderLayerConstructionHP(
    const EncoderLayerConstructionHP& hp,
    const char* caller)
{
    requirePositiveGroupingValue(hp.num_layers, "num_layers", caller);
    requirePositiveGroupingValue(hp.d_model, "d_model", caller);
    requirePositiveGroupingValue(hp.num_heads, "num_heads", caller);
    requirePositiveGroupingValue(hp.num_kv_heads, "num_kv_heads", caller);
    requirePositiveGroupingValue(hp.d_ff, "d_ff", caller);
    requirePositiveFiniteGroupingValue(hp.rms_epsilon, "rms_epsilon", caller);
    requireDropoutProbability(hp.dropout_rate, "dropout_rate", caller);
    requireDropoutProbability(hp.attention_dropout, "attention_dropout", caller);
    requirePositiveFiniteGroupingValue(hp.residual_projection_init_gain, "residual_projection_init_gain", caller);

    if (!isValidGQAConfig(hp.num_heads, hp.num_kv_heads)) {
        throw std::runtime_error(std::string(caller) + ": invalid GQA config num_heads=" +
                                 std::to_string(hp.num_heads) + " num_kv_heads=" +
                                 std::to_string(hp.num_kv_heads));
    }
    if (hp.d_model % hp.num_heads != 0) {
        throw std::runtime_error(std::string(caller) + ": d_model=" +
                                 std::to_string(hp.d_model) +
                                 " must be divisible by num_heads=" +
                                 std::to_string(hp.num_heads));
    }

    const int expected_head_dim = hp.d_model / hp.num_heads;
    if (hp.head_dim != expected_head_dim) {
        throw std::runtime_error(std::string(caller) + ": head_dim=" +
                                 std::to_string(hp.head_dim) +
                                 " does not match d_model/num_heads=" +
                                 std::to_string(expected_head_dim));
    }

    const int expected_heads_per_kv_group = hp.num_heads / hp.num_kv_heads;
    if (hp.heads_per_kv_group != expected_heads_per_kv_group) {
        throw std::runtime_error(std::string(caller) + ": heads_per_kv_group=" +
                                 std::to_string(hp.heads_per_kv_group) +
                                 " does not match num_heads/num_kv_heads=" +
                                 std::to_string(expected_heads_per_kv_group));
    }

    const int expected_kv_dim = computeKVProjectionSize(
        hp.d_model, hp.num_heads, hp.num_kv_heads);
    if (hp.kv_dim != expected_kv_dim) {
        throw std::runtime_error(std::string(caller) + ": kv_dim=" +
                                 std::to_string(hp.kv_dim) +
                                 " does not match num_kv_heads*head_dim=" +
                                 std::to_string(expected_kv_dim));
    }

    const int expected_qkv_dim = computeQKVProjectionSize(
        hp.d_model, hp.num_heads, hp.num_kv_heads);
    if (hp.qkv_dim != expected_qkv_dim) {
        throw std::runtime_error(std::string(caller) + ": qkv_dim=" +
                                 std::to_string(hp.qkv_dim) +
                                 " does not match d_model+2*kv_dim=" +
                                 std::to_string(expected_qkv_dim));
    }

    const bool expected_is_gqa = hp.num_kv_heads < hp.num_heads;
    if (hp.is_gqa != expected_is_gqa) {
        throw std::runtime_error(std::string(caller) + ": is_gqa=" +
                                 std::to_string(static_cast<int>(hp.is_gqa)) +
                                 " does not match num_kv_heads<num_heads=" +
                                 std::to_string(static_cast<int>(expected_is_gqa)));
    }

    if (hp.use_flash_attention) {
        requirePositiveGroupingValue(hp.min_seq_len_for_flash,
                                     "min_seq_len_for_flash",
                                     caller);
    }
    if (hp.use_layer_scale) {
        requirePositiveFiniteGroupingValue(hp.layer_scale_init,
                                           "layer_scale_init",
                                           caller);
    }
}

inline void validateEncoderSelfAttentionHP(
    const EncoderSelfAttentionHP& hp,
    const char* caller)
{
    requirePositiveGroupingValue(hp.d_model, "d_model", caller);
    requirePositiveGroupingValue(hp.num_heads, "num_heads", caller);
    requirePositiveGroupingValue(hp.num_kv_heads, "num_kv_heads", caller);
    requirePositiveGroupingValue(hp.head_dim, "head_dim", caller);
    requirePositiveGroupingValue(hp.heads_per_kv_group, "heads_per_kv_group", caller);
    requirePositiveGroupingValue(hp.kv_dim, "kv_dim", caller);
    requirePositiveGroupingValue(hp.qkv_dim, "qkv_dim", caller);
    requireDropoutProbability(hp.attention_dropout, "attention_dropout", caller);

    if (!isValidGQAConfig(hp.num_heads, hp.num_kv_heads)) {
        throw std::runtime_error(std::string(caller) + ": invalid GQA config num_heads=" +
                                 std::to_string(hp.num_heads) + " num_kv_heads=" +
                                 std::to_string(hp.num_kv_heads));
    }
    if (hp.d_model % hp.num_heads != 0) {
        throw std::runtime_error(std::string(caller) + ": d_model=" +
                                 std::to_string(hp.d_model) +
                                 " must be divisible by num_heads=" +
                                 std::to_string(hp.num_heads));
    }

    const int expected_head_dim = hp.d_model / hp.num_heads;
    if (hp.head_dim != expected_head_dim) {
        throw std::runtime_error(std::string(caller) + ": head_dim=" +
                                 std::to_string(hp.head_dim) +
                                 " does not match d_model/num_heads=" +
                                 std::to_string(expected_head_dim));
    }

    const int expected_heads_per_kv_group = hp.num_heads / hp.num_kv_heads;
    if (hp.heads_per_kv_group != expected_heads_per_kv_group) {
        throw std::runtime_error(std::string(caller) + ": heads_per_kv_group=" +
                                 std::to_string(hp.heads_per_kv_group) +
                                 " does not match num_heads/num_kv_heads=" +
                                 std::to_string(expected_heads_per_kv_group));
    }

    const int expected_kv_dim = computeKVProjectionSize(
        hp.d_model, hp.num_heads, hp.num_kv_heads);
    if (hp.kv_dim != expected_kv_dim) {
        throw std::runtime_error(std::string(caller) + ": kv_dim=" +
                                 std::to_string(hp.kv_dim) +
                                 " does not match num_kv_heads*head_dim=" +
                                 std::to_string(expected_kv_dim));
    }

    const int expected_qkv_dim = computeQKVProjectionSize(
        hp.d_model, hp.num_heads, hp.num_kv_heads);
    if (hp.qkv_dim != expected_qkv_dim) {
        throw std::runtime_error(std::string(caller) + ": qkv_dim=" +
                                 std::to_string(hp.qkv_dim) +
                                 " does not match d_model+2*kv_dim=" +
                                 std::to_string(expected_qkv_dim));
    }

    const bool expected_is_gqa = hp.num_kv_heads < hp.num_heads;
    if (hp.is_gqa != expected_is_gqa) {
        throw std::runtime_error(std::string(caller) + ": is_gqa=" +
                                 std::to_string(static_cast<int>(hp.is_gqa)) +
                                 " does not match num_kv_heads<num_heads=" +
                                 std::to_string(static_cast<int>(expected_is_gqa)));
    }

    if (hp.use_flash_attention) {
        requirePositiveGroupingValue(hp.min_seq_len_for_flash,
                                     "min_seq_len_for_flash",
                                     caller);
    }
}

inline void validateFlashAttentionRuntimeHP(
    const FlashAttentionRuntimeHP& hp,
    const char* caller)
{
    requirePositiveGroupingValue(hp.num_heads, "num_heads", caller);
    requirePositiveGroupingValue(hp.num_kv_heads, "num_kv_heads", caller);
    requirePositiveGroupingValue(hp.head_dim, "head_dim", caller);

    if (!isValidGQAConfig(hp.num_heads, hp.num_kv_heads)) {
        throw std::runtime_error(std::string(caller) + ": invalid FlashAttention GQA config num_heads=" +
                                 std::to_string(hp.num_heads) + " num_kv_heads=" +
                                 std::to_string(hp.num_kv_heads));
    }
    if (!isValidFlashAttentionHeadDim(hp.head_dim)) {
        throw std::runtime_error(std::string(caller) + ": head_dim=" +
                                 std::to_string(hp.head_dim) +
                                 " is not supported by FlashAttention grouping");
    }
    if (!hp.requires_alibi) {
        throw std::runtime_error(std::string(caller) +
                                 ": requires_alibi=false but GRIM FlashAttention consumes PBM ALiBi slopes");
    }

#ifdef GRIM_FLASHATTN_HDIM64_ONLY
    if (hp.head_dim != FLASH_ATTN_HEAD_DIM_64) {
        throw std::runtime_error(std::string(caller) +
                                 ": binary was compiled with GRIM_FLASHATTN_HDIM64_ONLY but grouped head_dim=" +
                                 std::to_string(hp.head_dim));
    }
#endif

#ifdef GRIM_FLASHATTN_CAUSAL_ONLY
    if (!hp.causal) {
        throw std::runtime_error(std::string(caller) +
                                 ": binary was compiled with GRIM_FLASHATTN_CAUSAL_ONLY but grouped causal=false");
    }
#endif

#ifdef GRIM_FLASHATTN_BF16_ONLY
    if (!hp.is_bf16) {
        throw std::runtime_error(std::string(caller) +
                                 ": binary was compiled with GRIM_FLASHATTN_BF16_ONLY but grouped is_bf16=false");
    }
#endif
}

inline void validateFlashAttentionRuntimeForEncoderSelfAttentionHP(
    const FlashAttentionRuntimeHP& flash_hp,
    const EncoderSelfAttentionHP& attention_hp,
    const char* caller)
{
    validateEncoderSelfAttentionHP(attention_hp, caller);
    validateFlashAttentionRuntimeHP(flash_hp, caller);

    if (!attention_hp.use_flash_attention) {
        throw std::runtime_error(std::string(caller) +
                                 ": use_flash_attention=false but GRIM encoder attention is FlashAttention-backed");
    }
    if (flash_hp.num_heads != attention_hp.num_heads) {
        throw std::runtime_error(std::string(caller) + ": FlashAttention num_heads=" +
                                 std::to_string(flash_hp.num_heads) +
                                 " does not match EncoderSelfAttentionHP num_heads=" +
                                 std::to_string(attention_hp.num_heads));
    }
    if (flash_hp.num_kv_heads != attention_hp.num_kv_heads) {
        throw std::runtime_error(std::string(caller) + ": FlashAttention num_kv_heads=" +
                                 std::to_string(flash_hp.num_kv_heads) +
                                 " does not match EncoderSelfAttentionHP num_kv_heads=" +
                                 std::to_string(attention_hp.num_kv_heads));
    }
    if (flash_hp.head_dim != attention_hp.head_dim) {
        throw std::runtime_error(std::string(caller) + ": FlashAttention head_dim=" +
                                 std::to_string(flash_hp.head_dim) +
                                 " does not match EncoderSelfAttentionHP head_dim=" +
                                 std::to_string(attention_hp.head_dim));
    }
    if (flash_hp.causal != attention_hp.causal_mask) {
        throw std::runtime_error(std::string(caller) + ": FlashAttention causal=" +
                                 std::to_string(static_cast<int>(flash_hp.causal)) +
                                 " does not match EncoderSelfAttentionHP causal_mask=" +
                                 std::to_string(static_cast<int>(attention_hp.causal_mask)));
    }
}

inline void validatePBMConstructionHP(
    const PBMConstructionHP& hp,
    const char* caller)
{
    requirePositiveGroupingValue(hp.num_heads, "num_heads", caller);
    requirePositiveGroupingValue(hp.num_kv_heads, "num_kv_heads", caller);
    requirePositiveGroupingValue(hp.head_dim, "head_dim", caller);
    requirePositiveGroupingValue(hp.rotary_dim, "rotary_dim", caller);
    requirePositiveGroupingValue(hp.max_seq_len, "max_seq_len", caller);
    requirePositiveGroupingValue(hp.rope_base_seq_len, "rope_base_seq_len", caller);
    requirePositiveGroupingValue(hp.alibi_min_locality_distance,
                                 "alibi_min_locality_distance",
                                 caller);
    requirePositiveFiniteGroupingValue(hp.rope_theta, "rope_theta", caller);
    requirePositiveFiniteGroupingValue(hp.rope_scaling, "rope_scaling", caller);

    if (!isValidGQAConfig(hp.num_heads, hp.num_kv_heads)) {
        throw std::runtime_error(std::string(caller) + ": invalid GQA config num_heads=" +
                                 std::to_string(hp.num_heads) + " num_kv_heads=" +
                                 std::to_string(hp.num_kv_heads));
    }
    if ((hp.rotary_dim & 1) != 0 || hp.rotary_dim > hp.head_dim) {
        throw std::runtime_error(std::string(caller) + ": rotary_dim=" +
                                 std::to_string(hp.rotary_dim) +
                                 " must be even and <= head_dim=" +
                                 std::to_string(hp.head_dim));
    }
    if (!std::isfinite(hp.alibi_slope_exponent) || hp.alibi_slope_exponent == 0.0f) {
        throw std::runtime_error(std::string(caller) +
                                 ": alibi_slope_exponent must be finite and non-zero, got " +
                                 std::to_string(hp.alibi_slope_exponent));
    }
    if (!std::isfinite(hp.alibi_max_bias) || hp.alibi_max_bias > 0.0f) {
        throw std::runtime_error(std::string(caller) +
                                 ": alibi_max_bias must be finite and <= 0, got " +
                                 std::to_string(hp.alibi_max_bias));
    }
}

inline void validateTrainingFixedShapeHP(
    const TrainingFixedShapeHP& hp,
    const char* caller)
{
    requirePositiveGroupingValue(hp.batch_size, "batch_size", caller);
    requirePositiveGroupingValue(hp.max_seq_len, "max_seq_len", caller);
    requirePositiveGroupingValue(hp.max_tokens_per_batch, "max_tokens_per_batch", caller);

    const std::int64_t expected_tokens =
        static_cast<std::int64_t>(hp.batch_size) * static_cast<std::int64_t>(hp.max_seq_len);
    if (expected_tokens > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(caller) +
                                 ": batch_size * max_seq_len overflowed int capacity (batch_size=" +
                                 std::to_string(hp.batch_size) + " max_seq_len=" +
                                 std::to_string(hp.max_seq_len) + " product=" +
                                 std::to_string(expected_tokens) + ")");
    }
    if (hp.max_tokens_per_batch != static_cast<int>(expected_tokens)) {
        throw std::runtime_error(std::string(caller) +
                                 ": max_tokens_per_batch=" +
                                 std::to_string(hp.max_tokens_per_batch) +
                                 " does not match batch_size * max_seq_len=" +
                                 std::to_string(expected_tokens));
    }
}

inline TrainingFixedShapeHP trainingFixedShapeHP(
    const StartupConfig& config)
{
    TrainingFixedShapeHP view;
    view.batch_size = config.hyperparameters.batch_size;
    view.max_seq_len = config.max_seq_len;

    const std::int64_t token_budget =
        static_cast<std::int64_t>(view.batch_size) * static_cast<std::int64_t>(view.max_seq_len);
    if (token_budget > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("trainingFixedShapeHP: max_tokens_per_batch overflow: batch_size=" +
                                 std::to_string(view.batch_size) +
                                 " max_seq_len=" + std::to_string(view.max_seq_len) +
                                 " product=" + std::to_string(token_budget) +
                                 " exceeds int max");
    }

    view.max_tokens_per_batch = static_cast<int>(token_budget);
    validateTrainingFixedShapeHP(view, "trainingFixedShapeHP");
    return view;
}

inline DataLoadingHP dataLoadingHP(const StartupConfig& config) {
    DataLoadingHP view;
    view.min_seq_valid_tokens = config.hyperparameters.min_seq_valid_tokens;
    view.sliding_window_stride = config.sliding_window_stride;
    return view;
}

inline TokenizerHP tokenizerHP(const StartupConfig& config) {
    const ::GRIM::Config::TrainingHyperparameters& hp = config.hyperparameters;
    // Tokenizer JSON leaves are owned by AiConfigSnapshot; StartupConfig only carries the snapshot.
    const ::GRIM::Config::AiConfigSnapshot& snapshot = config.ai_config_snapshot;
    if (!snapshot.has_tokenizer) {
        throw std::runtime_error("tokenizerHP: AiConfigSnapshot.has_tokenizer is false");
    }

    TokenizerHP view;
    view.data_path = config.paths.data_path;
    view.vocab_path = config.paths.vocab_path;
    view.target_vocab_size = snapshot.tokenizer_vocab_size;
    if (snapshot.tokenizer_max_vocab_size > 0 && view.target_vocab_size > snapshot.tokenizer_max_vocab_size) {
        view.target_vocab_size = snapshot.tokenizer_max_vocab_size;
    }
    view.character_coverage = snapshot.tokenizer_character_coverage;
    view.min_cleaned_text_length = snapshot.tokenizer_min_cleaned_text_length;
    view.min_subword_freq = snapshot.tokenizer_min_subword_freq;
    view.prune_during_mining = snapshot.tokenizer_prune_during_mining;
    view.enable_parallel_subword_mining = snapshot.tokenizer_enable_parallel_subword_mining;
    view.subword_mining_workers = snapshot.tokenizer_subword_mining_workers;
    view.subword_mining_max_bytes = snapshot.tokenizer_subword_mining_max_bytes;
    view.enable_scratch_block_reasoning = hp.tokenizer_enable_scratch_block_reasoning;
    view.detect_numbers = hp.tokenizer_detect_numbers;
    view.enable_byte_fallback = snapshot.tokenizer_enable_byte_fallback;
    view.add_bos = snapshot.tokenizer_add_bos;
    view.add_eos = snapshot.tokenizer_add_eos;
    view.force_rebuild_vocab = hp.force_rebuild_vocab;
    view.save_text_vocab = snapshot.tokenizer_save_text_vocab;
    view.vocab_score_multiplier = snapshot.tokenizer_vocab_score_multiplier;
    view.current_curriculum = hp.current_curriculum;
    view.current_model_training = hp.current_model_training;
    view.execution_block_num_steps = hp.execution_block_num_steps;

    if (view.data_path.empty()) {
        throw std::runtime_error("tokenizerHP: data_path is empty");
    }
    if (view.vocab_path.empty()) {
        throw std::runtime_error("tokenizerHP: vocab_path is empty");
    }
    requirePositiveGroupingValue(view.target_vocab_size, "target_vocab_size", "tokenizerHP");
    requirePositiveFiniteGroupingValue(view.character_coverage, "character_coverage", "tokenizerHP");
    if (view.character_coverage > 1.0f) {
        throw std::runtime_error("tokenizerHP: character_coverage must be <= 1, got " +
                                 std::to_string(view.character_coverage));
    }
    requirePositiveGroupingValue(view.min_cleaned_text_length, "min_cleaned_text_length", "tokenizerHP");
    requirePositiveGroupingValue(view.min_subword_freq, "min_subword_freq", "tokenizerHP");
    if (view.subword_mining_workers < 0) {
        throw std::runtime_error("tokenizerHP: subword_mining_workers must be >= 0, got " +
                                 std::to_string(view.subword_mining_workers));
    }
    requirePositiveFiniteGroupingValue(view.vocab_score_multiplier,
                                       "vocab_score_multiplier",
                                       "tokenizerHP");
    if (view.execution_block_num_steps < 0) {
        throw std::runtime_error("tokenizerHP: execution_block_num_steps must be >= 0, got " +
                                 std::to_string(view.execution_block_num_steps));
    }
    return view;
}

inline TokenizerSubprocessHP tokenizerSubprocessHP(const StartupConfig& config)
{
    TokenizerSubprocessHP view;
    const ::GRIM::Config::AiConfigSnapshot& snapshot = config.ai_config_snapshot;
    if (config.paths.config_path.empty()) {
        throw std::runtime_error("tokenizerSubprocessHP: config_path is empty");
    }
    if (!snapshot.has_subprocess) {
        throw std::runtime_error("tokenizerSubprocessHP: AiConfigSnapshot.has_subprocess is false");
    }

    view.config_path = config.paths.config_path.string();
    view.tokenizer = tokenizerHP(config);
    view.only_mode = snapshot.subprocess_tokenizer_only_mode;
    return view;
}

inline LearningRateScheduleInputs learningRateScheduleInputs(
    const ::GRIM::Config::TrainingHyperparameters& hp)
{
    LearningRateScheduleInputs inputs;
    inputs.learning_rate = hp.learning_rate;
    inputs.cosine_decay_min_lr = hp.cosine_decay_min_lr;
    inputs.warmup_steps = hp.warmup_steps;
    inputs.cosine_decay_enabled = hp.cosine_decay_enabled;
    inputs.cosine_warm_restarts = hp.cosine_warm_restarts;
    return inputs;
}

inline OptimizerUpdateHP optimizerUpdateHP(
    const ::GRIM::Config::TrainingHyperparameters& hp)
{
    OptimizerUpdateHP view;
    if (hp.optimizer_kind == "adamw") {
        view.kind = OptimizerKind::ADAMW;
    } else if (hp.optimizer_kind == "radamw") {
        view.kind = OptimizerKind::RADAMW;
    } else {
        throw std::runtime_error("optimizerUpdateHP: optimizer_kind must be 'adamw' or 'radamw', got '" +
                                 hp.optimizer_kind + "'");
    }

    if (!std::isfinite(hp.weight_decay) || hp.weight_decay < 0.0f) {
        throw std::runtime_error("optimizerUpdateHP: weight_decay must be finite and >= 0, got " +
                                 std::to_string(hp.weight_decay));
    }
    if (!(hp.optimizer_beta1 > 0.0f && hp.optimizer_beta1 < 1.0f)) {
        throw std::runtime_error("optimizerUpdateHP: optimizer_beta1 must be in (0,1), got " +
                                 std::to_string(hp.optimizer_beta1));
    }
    if (!(hp.optimizer_beta2 > 0.0f && hp.optimizer_beta2 < 1.0f)) {
        throw std::runtime_error("optimizerUpdateHP: optimizer_beta2 must be in (0,1), got " +
                                 std::to_string(hp.optimizer_beta2));
    }
    requirePositiveFiniteGroupingValue(hp.optimizer_epsilon,
                                       "optimizer_epsilon",
                                       "optimizerUpdateHP");
    if (hp.embedding_freeze_enabled && hp.embedding_freeze_after_step < 0) {
        throw std::runtime_error("optimizerUpdateHP: embedding_freeze_enabled=true but embedding_freeze_after_step=" +
                                 std::to_string(hp.embedding_freeze_after_step));
    }

    view.weight_decay = hp.weight_decay;
    view.beta1 = hp.optimizer_beta1;
    view.beta2 = hp.optimizer_beta2;
    view.epsilon = hp.optimizer_epsilon;
    view.embedding_freeze_after_step = hp.embedding_freeze_enabled
        ? hp.embedding_freeze_after_step
        : -1;
    return view;
}

inline GradientClippingHP gradientClippingHP(
    const ::GRIM::Config::TrainingHyperparameters& hp)
{
    GradientClippingHP view;
    view.configured_clip_norm = hp.grad_clip_norm;
    view.enabled = hp.grad_clip_norm > 0.0f;
    view.effective_per_token_limit = std::max(
        hp.grad_clip_norm, EPSILON_GRADIENT_CLIP);
    return view;
}

inline LossConfigHP lossConfigHP(
    const ::GRIM::Config::TrainingHyperparameters& hp)
{
    LossConfigHP view;
    view.focal_enabled = hp.loss_focal_enabled;
    view.focal_alpha = hp.loss_focal_alpha;
    view.focal_gamma = hp.loss_focal_gamma;
    view.smoothing_enabled = hp.loss_label_smoothing_enabled;
    view.smoothing_epsilon = hp.loss_label_smoothing_epsilon;
    view.entropy_reg_enabled = hp.loss_entropy_reg_enabled;
    view.entropy_reg_lambda = hp.loss_entropy_reg_lambda;
    view.class_balanced_enabled = hp.loss_class_balanced_enabled;
    view.class_balanced_beta = hp.loss_class_balanced_beta;
    validateLossConfigHP(view, "lossConfigHP");
    return view;
}

inline void validateLanguageModelCacheCapacity(
    const LanguageModelConfig& cfg,
    const char* caller = "validateLanguageModelCacheCapacity")
{
    requirePositiveGroupingValue(cfg.max_cached_batch, "max_cached_batch", caller);
    requirePositiveGroupingValue(cfg.max_tokens_per_batch, "max_tokens_per_batch", caller);
}

inline void validateStartupLanguageModelConfig(
    const LanguageModelConfig& cfg)
{
    if (cfg.vocab_size <= 0) {
        throw std::runtime_error("startupLanguageModelConfig: vocab_size must be > 0, got " +
                                 std::to_string(cfg.vocab_size));
    }
    if (cfg.vocab_path.empty()) {
        throw std::runtime_error("startupLanguageModelConfig: vocab_path is empty");
    }
    validateLanguageModelCacheCapacity(cfg, "startupLanguageModelConfig");
    if (cfg.execution_mode != ModelExecutionMode::TRAINING) {
        throw std::runtime_error("startupLanguageModelConfig: training startup produced a non-training execution_mode");
    }
    if (cfg.structured_ce_enabled && cfg.structured_ce_weight <= 0.0f) {
        throw std::runtime_error("startupLanguageModelConfig: structured_ce_enabled=true but structured_ce_weight=" +
                                 std::to_string(cfg.structured_ce_weight) + " (must be > 0)");
    }
    if (cfg.mtp_enabled && (cfg.mtp_k <= 0 || cfg.mtp_alpha <= 0.0f)) {
        throw std::runtime_error("startupLanguageModelConfig: multi_token_prediction enabled but k/alpha invalid (k=" +
                                 std::to_string(cfg.mtp_k) + " alpha=" +
                                 std::to_string(cfg.mtp_alpha) + ")");
    }
}

inline void validateInferenceLanguageModelConfig(
    const LanguageModelConfig& cfg)
{
    if (cfg.vocab_size <= 0) {
        throw std::runtime_error("inferenceLanguageModelConfig: vocab_size must be > 0, got " +
                                 std::to_string(cfg.vocab_size));
    }
    if (cfg.vocab_path.empty()) {
        throw std::runtime_error("inferenceLanguageModelConfig: vocab_path is empty");
    }
    if (cfg.execution_mode != ModelExecutionMode::INFERENCE) {
        throw std::runtime_error("inferenceLanguageModelConfig: produced a non-inference execution_mode");
    }
    validateLanguageModelCacheCapacity(cfg, "inferenceLanguageModelConfig");
}

inline LanguageModelConfig languageModelConfigFromTrainingHyperparameters(
    const ::GRIM::Config::TrainingHyperparameters& hp,
    const GenerationConfig& generation)
{
    LanguageModelConfig cfg;
    cfg.d_model = hp.d_model;
    cfg.num_layers = hp.num_layers;
    cfg.num_heads = hp.num_heads;
    cfg.num_kv_heads = hp.num_kv_heads;
    cfg.d_ff = hp.d_ff;
    cfg.max_seq_len = hp.max_seq_len;
    cfg.dropout_rate = hp.dropout_rate;
    cfg.attention_dropout = hp.attention_dropout;
    cfg.tie_embeddings = hp.tie_embeddings;
    cfg.positional_encoding = hp.positional_encoding;
    cfg.rope_base_seq_len = hp.rope_base_seq_len;
    cfg.alibi_min_locality_distance = hp.alibi_min_locality_distance;
    cfg.alibi_slope_exponent = hp.alibi_slope_exponent;
    cfg.alibi_max_bias = hp.alibi_max_bias;
    cfg.rope_theta = hp.rope_theta;
    cfg.rope_scaling = hp.rope_scaling;
    cfg.use_flash_attention = hp.use_flash_attention;
    cfg.min_seq_len_for_flash = hp.min_seq_len_for_flash;
    cfg.use_gpu = hp.use_gpu;
    cfg.head_dim = hp.head_dim;
    cfg.vocab_size = hp.vocab_size;
    cfg.max_cached_batch = hp.max_cached_batch;
    cfg.max_cached_seq_len = hp.max_cached_seq_len;
    cfg.max_tokens_per_batch = hp.max_tokens_per_batch;
    cfg.rms_epsilon = hp.rms_epsilon;
    cfg.causal_mask = hp.causal_mask;
    cfg.use_pre_norm = hp.use_pre_norm;
    cfg.fuse_qkv = hp.fuse_qkv;
    cfg.use_simd = hp.use_simd;
    cfg.num_threads = hp.num_threads;
    cfg.use_bias = hp.use_bias;
    cfg.qk_norm_enabled = hp.qk_norm_enabled;
    cfg.use_layer_scale = hp.use_layer_scale;
    cfg.layer_scale_init = hp.layer_scale_init;
    cfg.vocab_path = hp.vocab_path;
    cfg.execution_mode = hp.execution_mode;
    cfg.parameter_precision_embedding = hp.parameter_precision_embedding;
    cfg.parameter_precision_lm_head = hp.parameter_precision_lm_head;
    cfg.parameter_precision_attention = hp.parameter_precision_attention;
    cfg.parameter_precision_ffn = hp.parameter_precision_ffn;
    cfg.parameter_precision_rmsnorm = hp.parameter_precision_rmsnorm;
    cfg.parameter_precision_scratchblock = hp.parameter_precision_scratchblock;
    cfg.parameter_precision_mtp = hp.parameter_precision_mtp;
    cfg.parameter_precision_reasoning_head = hp.parameter_precision_reasoning_head;
    cfg.parameter_precision_execution_block = hp.parameter_precision_execution_block;
    cfg.parameter_precision_slot_selector = hp.parameter_precision_slot_selector;
    cfg.use_scratch_block = hp.use_scratch_block;
    cfg.scratch_block_atom_embedding_dim = hp.scratch_block_atom_embedding_dim;
    cfg.scratch_block_max_atoms = hp.scratch_block_max_atoms;
    cfg.scratch_block_atom_scale = hp.scratch_block_atom_scale;
    cfg.scratch_block_execution_first_type_only = hp.scratch_block_execution_first_type_only;
    cfg.reasoning_head_enabled = hp.reasoning_head_enabled;
    cfg.reasoning_num_ops = hp.reasoning_num_ops;
    cfg.execution_block_enabled = hp.execution_block_enabled;
    cfg.execution_block_layer = hp.execution_block_layer;
    cfg.execution_block_num_ops = hp.execution_block_num_ops;
    cfg.execution_block_num_slots = hp.execution_block_num_slots;
    cfg.execution_block_num_scratch_slots = hp.execution_block_num_scratch_slots;
    cfg.execution_block_num_steps = hp.execution_block_num_steps;
    cfg.execution_block_value_decode_input_dim = hp.execution_block_value_decode_input_dim;
    cfg.execution_block_value_decode_hidden_dim = hp.execution_block_value_decode_hidden_dim;
    cfg.execution_block_d_key = hp.execution_block_d_key;
    cfg.execution_block_d_type = hp.execution_block_d_type;
    cfg.execution_block_cross_attn_head_dim = hp.execution_block_cross_attn_head_dim;
    cfg.execution_block_cross_attn_topk = hp.execution_block_cross_attn_topk;
    cfg.execution_block_usage_decay = hp.execution_block_usage_decay;
    cfg.execution_block_inject_gate_temp = hp.execution_block_inject_gate_temp;
    cfg.execution_block_result_slot_mode = hp.execution_block_result_slot_mode;
    cfg.execution_block_result_slot_index = hp.execution_block_result_slot_index;
    cfg.execution_block_debug_mode = hp.execution_block_debug_mode;
    cfg.execution_block_entropy_collapse_threshold = hp.execution_block_entropy_collapse_threshold;
    cfg.execution_block_write_collapse_threshold = hp.execution_block_write_collapse_threshold;
    cfg.execution_block_magnitude_limit = hp.execution_block_magnitude_limit;
    cfg.execution_block_diversity_kappa = hp.execution_block_diversity_kappa;
    cfg.execution_block_temp_start = hp.execution_block_temp_start;
    cfg.execution_block_temp_end = hp.execution_block_temp_end;
    cfg.execution_block_temp_schedule = hp.execution_block_temp_schedule;
    cfg.execution_block_entropy_weight = hp.execution_block_entropy_weight;
    cfg.execution_block_transition_hard_threshold = hp.execution_block_transition_hard_threshold;
    cfg.execution_block_gate_warmup_steps = hp.execution_block_gate_warmup_steps;
    cfg.execution_block_causal_w1_transition = hp.execution_block_causal_w1_transition;
    cfg.div_invalid_penalty_weight = hp.div_invalid_penalty_weight;
    cfg.div_magnitude_penalty_weight = hp.div_magnitude_penalty_weight;
    cfg.arg_reinforce_weight = hp.arg_reinforce_weight;
    cfg.arg_reinforce_baseline_decay = hp.arg_reinforce_baseline_decay;
    cfg.structured_ce_enabled = hp.structured_ce_enabled;
    cfg.structured_ce_weight = hp.structured_ce_weight;
    cfg.selector_enabled = hp.selector_enabled;
    cfg.decode_time_slot_feature_dim = hp.decode_time_slot_feature_dim;
    cfg.selector_d_selector = hp.selector_d_selector;
    cfg.selector_selection_margin = hp.selector_selection_margin;
    cfg.selector_supervision_weight = hp.selector_supervision_weight;
    cfg.step_x_multiplier = hp.step_x_multiplier;
    cfg.step_y_multiplier = hp.step_y_multiplier;
    cfg.step_y_overrides_x = hp.step_y_overrides_x;
    cfg.entropy_aux_weight = hp.entropy_aux_weight;
    cfg.value_match_epsilon = hp.value_match_epsilon;
    cfg.final_slot_consistency_weight = hp.final_slot_consistency_weight;
    cfg.lm_head_center_hidden_states = hp.lm_head_center_hidden_states;
    cfg.freeze_learned_rms_gammas = hp.freeze_learned_rms_gammas;
    cfg.project_out_pc1 = hp.project_out_pc1;
    cfg.pc1_power_iters = hp.pc1_power_iters;
    cfg.center_logits = hp.center_logits;
    cfg.center_encoder_residuals = hp.center_encoder_residuals;
    cfg.hardcoded_hidden_pattern = hp.hardcoded_hidden_pattern;
    cfg.hardcoded_log_every_n_batches = hp.hardcoded_log_every_n_batches;
    cfg.generation = generation;
    cfg.mtp_enabled = hp.mtp_enabled;
    cfg.mtp_k = hp.mtp_k;
    cfg.mtp_alpha = hp.mtp_alpha;
    cfg.mtp_alpha_warmup_steps = hp.mtp_alpha_warmup_steps;
    return cfg;
}

inline LanguageModelConfig startupLanguageModelConfig(
    const StartupConfig& config,
    std::uint32_t vocab_size,
    const TrainingFixedShapeHP& fixed_shape)
{
    GenerationConfig generation;
    loadGenerationConfig(config.ai_config_snapshot, generation);
    LanguageModelConfig cfg = languageModelConfigFromTrainingHyperparameters(config.hyperparameters, generation);

    // Runtime startup facts authored outside ai_config.json but still mapped
    // through this grouping layer so model startup has one config read path.
    cfg.max_seq_len = config.max_seq_len;
    cfg.vocab_size = static_cast<int>(vocab_size);
    cfg.vocab_path = config.paths.vocab_path;

    // GRIM-text training invariants. These are not caller fallbacks; they are
    // the single startup policy for this executable.
    cfg.execution_mode = ModelExecutionMode::TRAINING;
    cfg.causal_mask = true;
    cfg.use_pre_norm = true;
    cfg.fuse_qkv = true;

    validateTrainingFixedShapeHP(fixed_shape, "startupLanguageModelConfig");
    cfg.max_cached_batch = fixed_shape.batch_size;
    cfg.max_cached_seq_len = fixed_shape.max_seq_len;
    cfg.max_tokens_per_batch = fixed_shape.max_tokens_per_batch;

    cfg.computeDerivedValues();
    validateStartupLanguageModelConfig(cfg);
    return cfg;
}

inline LanguageModelConfig inferenceLanguageModelConfig(
    const StartupConfig& config,
    std::uint32_t vocab_size,
    const std::string& vocab_path)
{
    requirePositiveGroupingValue(config.max_seq_len,
                                 "StartupConfig.max_seq_len",
                                 "inferenceLanguageModelConfig");

    GenerationConfig generation;
    loadGenerationConfig(config.ai_config_snapshot, generation);
    LanguageModelConfig cfg = languageModelConfigFromTrainingHyperparameters(config.hyperparameters, generation);
    cfg.max_seq_len = config.max_seq_len;
    cfg.vocab_size = static_cast<int>(vocab_size);
    cfg.vocab_path = vocab_path;

    cfg.execution_mode = ModelExecutionMode::INFERENCE;
    cfg.causal_mask = true;
    cfg.use_pre_norm = true;
    cfg.fuse_qkv = true;
    cfg.use_gpu = true;

    cfg.max_cached_batch = 1;
    cfg.max_cached_seq_len = config.max_seq_len;
    cfg.max_tokens_per_batch = config.max_seq_len;

    cfg.computeDerivedValues();
    validateInferenceLanguageModelConfig(cfg);
    return cfg;
}

inline GpuModelInitializationHP gpuModelInitializationHP(
    const LanguageModelConfig& cfg)
{
    GpuModelInitializationHP view;
    requirePositiveGroupingValue(cfg.num_layers, "num_layers", "gpuModelInitializationHP");
    if (cfg.use_flash_attention) {
        requirePositiveGroupingValue(cfg.min_seq_len_for_flash,
                                     "min_seq_len_for_flash",
                                     "gpuModelInitializationHP");
    }
    view.use_gpu = cfg.use_gpu;
    view.num_layers = cfg.num_layers;
    view.use_flash_attention = cfg.use_flash_attention;
    view.min_seq_len_for_flash = cfg.min_seq_len_for_flash;
    return view;
}

inline PBMConstructionHP pbmConstructionHP(
    const LanguageModelConfig& cfg)
{
    requireValidGQAGrouping(cfg, "pbmConstructionHP");
    requirePositiveGroupingValue(cfg.max_seq_len, "max_seq_len", "pbmConstructionHP");

    PBMConstructionHP view;
    view.num_heads = cfg.num_heads;
    view.num_kv_heads = cfg.num_kv_heads;
    view.head_dim = cfg.head_dim;
    view.rotary_dim = cfg.head_dim;
    view.max_seq_len = cfg.max_seq_len;
    view.rope_base_seq_len = cfg.rope_base_seq_len;
    view.alibi_min_locality_distance = cfg.alibi_min_locality_distance;
    view.alibi_slope_exponent = cfg.alibi_slope_exponent;
    view.alibi_max_bias = cfg.alibi_max_bias;
    view.rope_theta = cfg.rope_theta;
    view.rope_scaling = cfg.rope_scaling;
    validatePBMConstructionHP(view, "pbmConstructionHP");
    return view;
}

inline EncoderLayerConstructionHP encoderLayerConstructionHP(
    const LanguageModelConfig& cfg)
{
    EncoderLayerConstructionHP view;
    requireValidGQAGrouping(cfg, "encoderLayerConstructionHP");
    requirePositiveGroupingValue(cfg.num_layers, "num_layers", "encoderLayerConstructionHP");
    requirePositiveGroupingValue(cfg.d_ff, "d_ff", "encoderLayerConstructionHP");
    requirePositiveFiniteGroupingValue(cfg.rms_epsilon, "rms_epsilon", "encoderLayerConstructionHP");
    requireDropoutProbability(cfg.dropout_rate, "dropout_rate", "encoderLayerConstructionHP");
    requireDropoutProbability(cfg.attention_dropout, "attention_dropout", "encoderLayerConstructionHP");
    if (cfg.use_flash_attention) {
        requirePositiveGroupingValue(cfg.min_seq_len_for_flash,
                                     "min_seq_len_for_flash",
                                     "encoderLayerConstructionHP");
    }
    view.num_layers = cfg.num_layers;
    view.d_model = cfg.d_model;
    view.num_heads = cfg.num_heads;
    view.num_kv_heads = cfg.num_kv_heads;
    view.head_dim = cfg.head_dim;
    view.heads_per_kv_group = computeHeadsPerKVGroup(cfg.num_heads, cfg.num_kv_heads);
    view.kv_dim = computeKVProjectionSize(cfg.d_model, cfg.num_heads, cfg.num_kv_heads);
    view.qkv_dim = computeQKVProjectionSize(cfg.d_model, cfg.num_heads, cfg.num_kv_heads);
    view.d_ff = cfg.d_ff;
    view.rms_epsilon = cfg.rms_epsilon;
    view.causal_mask = cfg.causal_mask;
    view.use_flash_attention = cfg.use_flash_attention;
    view.min_seq_len_for_flash = cfg.min_seq_len_for_flash;
    view.use_layer_scale = cfg.use_layer_scale;
    view.layer_scale_init = cfg.layer_scale_init;
    view.center_encoder_residuals = cfg.center_encoder_residuals;
    view.use_bias = cfg.use_bias;
    view.dropout_rate = cfg.dropout_rate;
    view.attention_dropout = cfg.attention_dropout;
    view.qk_norm_enabled = cfg.qk_norm_enabled;
    view.residual_projection_init_gain =
        1.0f / std::sqrt(2.0f * static_cast<float>(cfg.num_layers));
    view.is_gqa = cfg.num_kv_heads < cfg.num_heads;
    view.freeze_learned_rms_gammas = cfg.freeze_learned_rms_gammas;
    validateEncoderLayerConstructionHP(view, "encoderLayerConstructionHP");
    return view;
}

inline FeedForwardLayerConstructionHP feedForwardLayerConstructionHP(
    const EncoderLayerConstructionHP& encoder_hp)
{
    FeedForwardLayerConstructionHP view;
    requirePositiveGroupingValue(encoder_hp.d_model, "d_model", "feedForwardLayerConstructionHP");
    requirePositiveGroupingValue(encoder_hp.d_ff, "d_ff", "feedForwardLayerConstructionHP");
    requireDropoutProbability(encoder_hp.dropout_rate, "dropout_rate", "feedForwardLayerConstructionHP");
    if (!std::isfinite(encoder_hp.residual_projection_init_gain) || encoder_hp.residual_projection_init_gain <= 0.0f) {
        throw std::runtime_error("feedForwardLayerConstructionHP: residual_projection_init_gain must be a positive finite value from encoderLayerConstructionHP");
    }
    view.d_model = encoder_hp.d_model;
    view.d_ff = encoder_hp.d_ff;
    view.use_bias = encoder_hp.use_bias;
    view.dropout_rate = encoder_hp.dropout_rate;
    view.residual_projection_init_gain = encoder_hp.residual_projection_init_gain;
    return view;
}

inline EncoderSelfAttentionHP encoderSelfAttentionHP(
    const EncoderLayerConstructionHP& encoder_hp)
{
    validateEncoderLayerConstructionHP(encoder_hp, "encoderSelfAttentionHP");

    EncoderSelfAttentionHP view;
    view.d_model = encoder_hp.d_model;
    view.num_heads = encoder_hp.num_heads;
    view.num_kv_heads = encoder_hp.num_kv_heads;
    view.head_dim = encoder_hp.head_dim;
    view.heads_per_kv_group = encoder_hp.heads_per_kv_group;
    view.kv_dim = encoder_hp.kv_dim;
    view.qkv_dim = encoder_hp.qkv_dim;
    view.causal_mask = encoder_hp.causal_mask;
    view.use_flash_attention = encoder_hp.use_flash_attention;
    view.min_seq_len_for_flash = encoder_hp.min_seq_len_for_flash;
    view.use_bias = encoder_hp.use_bias;
    view.attention_dropout = encoder_hp.attention_dropout;
    view.qk_norm_enabled = encoder_hp.qk_norm_enabled;
    view.is_gqa = encoder_hp.is_gqa;
    validateEncoderSelfAttentionHP(view, "encoderSelfAttentionHP");
    return view;
}

inline FlashAttentionRuntimeHP flashAttentionRuntimeHP(
    const EncoderSelfAttentionHP& attention_hp)
{
    validateEncoderSelfAttentionHP(attention_hp, "flashAttentionRuntimeHP");
    if (!attention_hp.use_flash_attention) {
        throw std::runtime_error(
            "flashAttentionRuntimeHP: use_flash_attention=false but GRIM encoder attention is FlashAttention-backed");
    }

    FlashAttentionRuntimeHP view;
    view.num_heads = attention_hp.num_heads;
    view.num_kv_heads = attention_hp.num_kv_heads;
    view.head_dim = attention_hp.head_dim;
    view.causal = attention_hp.causal_mask;
    view.is_bf16 = true;
    view.requires_alibi = true;
    validateFlashAttentionRuntimeForEncoderSelfAttentionHP(
        view, attention_hp, "flashAttentionRuntimeHP");
    return view;
}

inline EmbeddingLayerConstructionHP embeddingLayerConstructionHP(
    const LanguageModelConfig& cfg)
{
    EmbeddingLayerConstructionHP view;
    requirePositiveGroupingValue(cfg.vocab_size, "vocab_size", "embeddingLayerConstructionHP");
    requirePositiveGroupingValue(cfg.d_model, "d_model", "embeddingLayerConstructionHP");
    view.vocab_size = cfg.vocab_size;
    view.d_model = cfg.d_model;
    return view;
}

inline LMHeadLayerConstructionHP lmHeadLayerConstructionHP(
    const LanguageModelConfig& cfg)
{
    LMHeadLayerConstructionHP view;
    requirePositiveGroupingValue(cfg.d_model, "d_model", "lmHeadLayerConstructionHP");
    requirePositiveFiniteGroupingValue(cfg.rms_epsilon, "rms_epsilon", "lmHeadLayerConstructionHP");
    requirePositiveGroupingValue(cfg.vocab_size, "vocab_size", "lmHeadLayerConstructionHP");
    if (cfg.project_out_pc1) {
        requirePositiveGroupingValue(cfg.pc1_power_iters,
                                     "pc1_power_iters",
                                     "lmHeadLayerConstructionHP");
    }
    view.d_model = cfg.d_model;
    view.vocab_size = cfg.vocab_size;
    if (cfg.execution_mode == ModelExecutionMode::TRAINING) {
        requirePositiveGroupingValue(cfg.max_cached_batch,
                                     "max_cached_batch",
                                     "lmHeadLayerConstructionHP");
        requirePositiveGroupingValue(cfg.max_cached_seq_len,
                                     "max_cached_seq_len",
                                     "lmHeadLayerConstructionHP");
        view.training_batch_size = cfg.max_cached_batch;
        view.training_rows_per_sequence = cfg.max_cached_seq_len;
    }
    view.use_bias = cfg.use_bias;
    view.tie_embeddings = cfg.tie_embeddings;
    view.center_hidden_states = cfg.lm_head_center_hidden_states;
    view.project_out_pc1 = cfg.project_out_pc1;
    view.pc1_power_iters = cfg.pc1_power_iters;
    view.center_logits = cfg.center_logits;
    view.freeze_learned_rms_gammas = cfg.freeze_learned_rms_gammas;
    view.rms_epsilon = cfg.rms_epsilon;
    return view;
}

inline ScratchBlockConstructionHP scratchBlockConstructionHP(
    const LanguageModelConfig& cfg)
{
    ScratchBlockConstructionHP view;
    view.enabled = cfg.use_scratch_block;
    view.d_model = cfg.d_model;
    view.max_atoms = cfg.scratch_block_max_atoms;
    view.atom_embedding_dim = cfg.scratch_block_atom_embedding_dim;
    view.atom_token_start = ATOM_TOKEN_START;
    view.atom_token_end = ATOM_TOKEN_END;
    view.atom_scale = cfg.scratch_block_atom_scale;

    if (view.enabled) {
        requirePositiveGroupingValue(view.d_model, "d_model", "scratchBlockConstructionHP");
        requirePositiveGroupingValue(view.max_atoms,
                                     "scratch_block_max_atoms",
                                     "scratchBlockConstructionHP");
        requirePositiveGroupingValue(view.atom_embedding_dim,
                                     "scratch_block_atom_embedding_dim",
                                     "scratchBlockConstructionHP");
        requirePositiveFiniteGroupingValue(view.atom_scale,
                                           "scratch_block_atom_scale",
                                           "scratchBlockConstructionHP");
        if (view.atom_token_start < 0 || view.atom_token_end <= view.atom_token_start) {
            throw std::runtime_error("scratchBlockConstructionHP: invalid atom token range start=" +
                                     std::to_string(view.atom_token_start) + " end=" +
                                     std::to_string(view.atom_token_end));
        }
    }
    return view;
}

inline ReasoningHeadConstructionHP reasoningHeadConstructionHP(
    const LanguageModelConfig& cfg)
{
    ReasoningHeadConstructionHP view;
    view.enabled = cfg.reasoning_head_enabled;
    view.d_model = cfg.d_model;
    view.atom_embedding_dim = cfg.scratch_block_atom_embedding_dim;
    view.num_ops = cfg.reasoning_num_ops;
    if (view.enabled) {
        requirePositiveGroupingValue(view.d_model, "d_model", "reasoningHeadConstructionHP");
        requirePositiveGroupingValue(view.atom_embedding_dim,
                                     "scratch_block_atom_embedding_dim",
                                     "reasoningHeadConstructionHP");
        requirePositiveGroupingValue(view.num_ops, "reasoning_num_ops", "reasoningHeadConstructionHP");
    }
    return view;
}

inline ExecutionBlockConstructionHP executionBlockConstructionHP(
    const LanguageModelConfig& cfg)
{
    ExecutionBlockConstructionHP view;
    view.enabled = cfg.execution_block_enabled;
    view.layer = cfg.execution_block_layer;
    view.d_model = cfg.d_model;
    view.atom_embedding_dim = cfg.scratch_block_atom_embedding_dim;
    view.num_ops = cfg.execution_block_num_ops;
    view.num_slots = cfg.execution_block_num_slots;
    view.num_scratch_slots = cfg.execution_block_num_scratch_slots;
    view.num_exec_steps = cfg.execution_block_num_steps;
    view.value_decode_input_dim = cfg.execution_block_value_decode_input_dim;
    view.value_decode_hidden_dim = cfg.execution_block_value_decode_hidden_dim;
    view.d_key = cfg.execution_block_d_key;
    view.d_type = cfg.execution_block_d_type;
    view.cross_attn_head_dim = cfg.execution_block_cross_attn_head_dim;
    view.cross_attn_topk = cfg.execution_block_cross_attn_topk;
    view.usage_decay = cfg.execution_block_usage_decay;
    view.inject_gate_temp = cfg.execution_block_inject_gate_temp;
    view.result_slot_mode = cfg.execution_block_result_slot_mode;
    view.result_slot_index = cfg.execution_block_result_slot_index;
    view.debug_mode = cfg.execution_block_debug_mode;
    view.entropy_collapse_threshold = cfg.execution_block_entropy_collapse_threshold;
    view.write_collapse_threshold = cfg.execution_block_write_collapse_threshold;
    view.magnitude_limit = cfg.execution_block_magnitude_limit;
    view.diversity_kappa = cfg.execution_block_diversity_kappa;
    view.temp_start = cfg.execution_block_temp_start;
    view.temp_end = cfg.execution_block_temp_end;
    view.temp_schedule = cfg.execution_block_temp_schedule;
    view.entropy_weight = cfg.execution_block_entropy_weight;
    view.transition_hard_threshold = cfg.execution_block_transition_hard_threshold;
    view.gate_warmup_steps = cfg.execution_block_gate_warmup_steps;
    view.causal_w1_transition = cfg.execution_block_causal_w1_transition;
    view.div_invalid_penalty_weight = cfg.div_invalid_penalty_weight;
    view.div_magnitude_penalty_weight = cfg.div_magnitude_penalty_weight;
    view.arg_reinforce_weight = cfg.arg_reinforce_weight;
    view.arg_reinforce_baseline_decay = cfg.arg_reinforce_baseline_decay;
    if (view.enabled) {
        requirePositiveGroupingValue(view.d_model, "d_model", "executionBlockConstructionHP");
        requirePositiveGroupingValue(view.atom_embedding_dim,
                                     "scratch_block_atom_embedding_dim",
                                     "executionBlockConstructionHP");
        if (view.layer < -1) {
            throw std::runtime_error(
                "executionBlockConstructionHP: execution_block_layer must be -1 or >= 0, got " +
                std::to_string(view.layer));
        }
        requirePositiveGroupingValue(view.num_ops, "execution_block_num_ops", "executionBlockConstructionHP");
        requirePositiveGroupingValue(view.num_slots, "execution_block_num_slots", "executionBlockConstructionHP");
        if (view.num_scratch_slots < 0 || view.num_scratch_slots >= view.num_slots) {
            throw std::runtime_error(
                "executionBlockConstructionHP: num_scratch_slots=" +
                std::to_string(view.num_scratch_slots) +
                " out of range [0, num_slots)");
        }
        requirePositiveGroupingValue(view.num_exec_steps,
                                     "execution_block_num_steps",
                                     "executionBlockConstructionHP");
        requirePositiveGroupingValue(view.value_decode_input_dim,
                                     "value_decode_input_dim",
                                     "executionBlockConstructionHP");
        requirePositiveGroupingValue(view.value_decode_hidden_dim,
                                     "value_decode_hidden_dim",
                                     "executionBlockConstructionHP");
        if (view.value_decode_input_dim + 16 > view.atom_embedding_dim) {
            throw std::runtime_error(
                "executionBlockConstructionHP: value_decode_input_dim + 16 must fit scratch_block_atom_embedding_dim");
        }
        requirePositiveGroupingValue(view.d_key, "execution_block_d_key", "executionBlockConstructionHP");
        if (view.d_key > 64) {
            throw std::runtime_error("executionBlockConstructionHP: execution_block_d_key must be <= 64, got " +
                                     std::to_string(view.d_key));
        }
        requirePositiveGroupingValue(view.d_type, "execution_block_d_type", "executionBlockConstructionHP");
        requirePositiveGroupingValue(view.cross_attn_head_dim,
                                     "execution_block_cross_attn_head_dim",
                                     "executionBlockConstructionHP");
        requirePositiveGroupingValue(view.cross_attn_topk,
                                     "execution_block_cross_attn_topk",
                                     "executionBlockConstructionHP");
        if (view.cross_attn_topk > view.num_slots) {
            throw std::runtime_error(
                "executionBlockConstructionHP: execution_block_cross_attn_topk=" +
                std::to_string(view.cross_attn_topk) +
                " exceeds execution_block_num_slots=" +
                std::to_string(view.num_slots));
        }
        requirePositiveFiniteGroupingValue(view.usage_decay,
                                           "execution_block_usage_decay",
                                           "executionBlockConstructionHP");
        if (view.usage_decay > 1.0f) {
            throw std::runtime_error("executionBlockConstructionHP: execution_block_usage_decay must be <= 1, got " +
                                     std::to_string(view.usage_decay));
        }
        requirePositiveFiniteGroupingValue(view.inject_gate_temp,
                                           "execution_block_inject_gate_temp",
                                           "executionBlockConstructionHP");
        requirePositiveFiniteGroupingValue(view.magnitude_limit,
                                           "execution_block_magnitude_limit",
                                           "executionBlockConstructionHP");
        requirePositiveFiniteGroupingValue(view.diversity_kappa,
                                           "execution_block_diversity_kappa",
                                           "executionBlockConstructionHP");
        requirePositiveFiniteGroupingValue(view.temp_start,
                                           "execution_block_temp_start",
                                           "executionBlockConstructionHP");
        requirePositiveFiniteGroupingValue(view.temp_end,
                                           "execution_block_temp_end",
                                           "executionBlockConstructionHP");
        if (view.temp_schedule < 0) {
            throw std::runtime_error("executionBlockConstructionHP: execution_block_temp_schedule must be >= 0, got " +
                                     std::to_string(view.temp_schedule));
        }
        if (!std::isfinite(view.entropy_weight) || view.entropy_weight < 0.0f) {
            throw std::runtime_error("executionBlockConstructionHP: execution_block_entropy_weight must be finite and >= 0, got " +
                                     std::to_string(view.entropy_weight));
        }
        if (!std::isfinite(view.transition_hard_threshold) || view.transition_hard_threshold < 0.0f) {
            throw std::runtime_error("executionBlockConstructionHP: execution_block_transition_hard_threshold must be finite and >= 0, got " +
                                     std::to_string(view.transition_hard_threshold));
        }
        if (view.gate_warmup_steps < 0) {
            throw std::runtime_error("executionBlockConstructionHP: execution_block_gate_warmup_steps must be >= 0, got " +
                                     std::to_string(view.gate_warmup_steps));
        }
        if (!std::isfinite(view.causal_w1_transition) || view.causal_w1_transition < 0.0f) {
            throw std::runtime_error("executionBlockConstructionHP: execution_block_causal_w1_transition must be finite and >= 0, got " +
                                     std::to_string(view.causal_w1_transition));
        }
        if (!std::isfinite(view.div_invalid_penalty_weight) || view.div_invalid_penalty_weight < 0.0f) {
            throw std::runtime_error("executionBlockConstructionHP: execution_block_div_invalid_penalty_weight must be finite and >= 0, got " +
                                     std::to_string(view.div_invalid_penalty_weight));
        }
        if (!std::isfinite(view.div_magnitude_penalty_weight) || view.div_magnitude_penalty_weight < 0.0f) {
            throw std::runtime_error("executionBlockConstructionHP: execution_block_div_magnitude_penalty_weight must be finite and >= 0, got " +
                                     std::to_string(view.div_magnitude_penalty_weight));
        }
        if (!std::isfinite(view.arg_reinforce_weight) || view.arg_reinforce_weight < 0.0f) {
            throw std::runtime_error("executionBlockConstructionHP: execution_block_arg_reinforce_weight must be finite and >= 0, got " +
                                     std::to_string(view.arg_reinforce_weight));
        }
        if (!std::isfinite(view.arg_reinforce_baseline_decay) ||
            view.arg_reinforce_baseline_decay < 0.0f ||
            view.arg_reinforce_baseline_decay > 1.0f) {
            throw std::runtime_error("executionBlockConstructionHP: execution_block_arg_reinforce_baseline_decay must be finite and in [0,1], got " +
                                     std::to_string(view.arg_reinforce_baseline_decay));
        }
        if (view.entropy_collapse_threshold < 0.0f || view.entropy_collapse_threshold > 1.0f) {
            throw std::runtime_error(
                "executionBlockConstructionHP: entropy_collapse_threshold must be in [0,1], got " +
                std::to_string(view.entropy_collapse_threshold));
        }
        if (view.write_collapse_threshold < 0.0f || view.write_collapse_threshold > 1.0f) {
            throw std::runtime_error(
                "executionBlockConstructionHP: write_collapse_threshold must be in [0,1], got " +
                std::to_string(view.write_collapse_threshold));
        }
        if (view.result_slot_index < -1 || view.result_slot_index >= view.num_slots) {
            throw std::runtime_error(
                "executionBlockConstructionHP: result_slot_index=" +
                std::to_string(view.result_slot_index) +
                " out of range [-1, num_slots)");
        }
        if (view.result_slot_mode < 0 || view.result_slot_mode > 1) {
            throw std::runtime_error(
                "executionBlockConstructionHP: execution_block_result_slot_mode must be 0 or 1, got " +
                std::to_string(view.result_slot_mode));
        }
    }
    return view;
}

inline DecodeTimeSelectorConstructionHP decodeTimeSelectorConstructionHP(
    const LanguageModelConfig& cfg)
{
    DecodeTimeSelectorConstructionHP view;
    view.enabled = cfg.selector_enabled;
    view.d_model = cfg.d_model;
    view.d_selector = cfg.selector_d_selector;
    view.d_slot_features = cfg.decode_time_slot_feature_dim;
    view.num_slots = cfg.execution_block_num_slots;
    view.scratch_slots = cfg.execution_block_num_scratch_slots;
    view.selection_margin = cfg.selector_selection_margin;
    view.supervision_weight = cfg.selector_supervision_weight;
    if (view.enabled) {
        if (!cfg.execution_block_enabled) {
            throw std::runtime_error(
                "decodeTimeSelectorConstructionHP: selector_enabled=true requires execution_block_enabled=true");
        }
        requirePositiveGroupingValue(view.d_model, "d_model", "decodeTimeSelectorConstructionHP");
        requirePositiveGroupingValue(view.d_selector,
                                     "selector_d_selector",
                                     "decodeTimeSelectorConstructionHP");
        requirePositiveGroupingValue(view.d_slot_features,
                                     "decode_time_slot_feature_dim",
                                     "decodeTimeSelectorConstructionHP");
        requirePositiveGroupingValue(view.num_slots,
                                     "execution_block_num_slots",
                                     "decodeTimeSelectorConstructionHP");
        if (view.scratch_slots < 0 || view.scratch_slots >= view.num_slots) {
            throw std::runtime_error(
                "decodeTimeSelectorConstructionHP: scratch_slots=" +
                std::to_string(view.scratch_slots) +
                " out of range [0, num_slots)");
        }
        if (!std::isfinite(view.selection_margin) || view.selection_margin < 0.0f) {
            throw std::runtime_error(
                "decodeTimeSelectorConstructionHP: selector_selection_margin must be finite and >= 0, got " +
                std::to_string(view.selection_margin));
        }
        if (!std::isfinite(view.supervision_weight) || view.supervision_weight < 0.0f) {
            throw std::runtime_error(
                "decodeTimeSelectorConstructionHP: selector_supervision_weight must be finite and >= 0, got " +
                std::to_string(view.supervision_weight));
        }
    }
    return view;
}

inline MTPConstructionHP mtpConstructionHP(const LanguageModelConfig& cfg)
{
    MTPConstructionHP view;
    view.enabled = cfg.mtp_enabled;
    view.k = cfg.mtp_k;
    view.vocab_size = cfg.vocab_size;
    view.d_model = cfg.d_model;
    view.alpha = cfg.mtp_alpha;
    view.alpha_warmup_steps = cfg.mtp_alpha_warmup_steps;
    if (view.enabled) {
        requirePositiveGroupingValue(view.k, "mtp_k", "mtpConstructionHP");
        requirePositiveGroupingValue(view.vocab_size, "vocab_size", "mtpConstructionHP");
        requirePositiveGroupingValue(view.d_model, "d_model", "mtpConstructionHP");
        if (view.alpha <= 0.0f) {
            throw std::runtime_error("mtpConstructionHP: mtp_alpha must be > 0, got " +
                                     std::to_string(view.alpha));
        }
    }
    return view;
}

inline MTPFeatureHP mtpFeatureHP(const LanguageModelConfig& cfg)
{
    MTPFeatureHP view;
    view.enabled = cfg.mtp_enabled;
    view.k = cfg.mtp_k;
    if (view.enabled) {
        requirePositiveGroupingValue(view.k, "mtp_k", "mtpFeatureHP");
    }
    return view;
}

inline ::GRIM::LR::LRScheduleConfig makeLRScheduleConfig(
    const LearningRateScheduleInputs& inputs,
    int total_steps,
    int steps_per_epoch)
{
    ::GRIM::LR::LRScheduleConfig cfg;
    cfg.base_lr = inputs.learning_rate;
    cfg.cosine_decay_min_lr = inputs.cosine_decay_min_lr;
    cfg.warmup_steps = inputs.warmup_steps;
    cfg.total_steps = total_steps;
    cfg.steps_per_epoch = steps_per_epoch;
    cfg.cosine_decay_enabled = inputs.cosine_decay_enabled;
    cfg.warm_restarts = inputs.cosine_warm_restarts;
    return cfg;
}

} // namespace GRIM::HyperParameters

