#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include "HyperParameters_GPU.hpp"
#include "../Dynamic_LR/LRSchedule.hpp"

namespace GRIM::HyperParameters {

// Groupings are read views over HyperParameters_GPU.hpp-owned config and constants.
// Do not introduce authored defaults here; add them to HyperParameters_GPU.hpp first.

struct CoreRunHP {
    int epochs = 0;
    int64_t seed = 0;
    int gradient_accumulation_steps = 0;
    bool single_batch_overfit_enabled = false;
    int single_batch_overfit_max_steps = 0;
};

struct CapacityHP {
    int batch_size = 0;
    int max_seq_len = 0;
    int gradient_accumulation_steps = 0;
};

struct DataLoadingHP {
    int min_seq_valid_tokens = 0;
    int sliding_window_stride = 0;
};

struct LearningRateScheduleInputs {
    float learning_rate = 0.0f;
    float cosine_decay_min_lr = 0.0f;
    int warmup_steps = 0;
    bool cosine_decay_enabled = false;
    bool cosine_warm_restarts = false;
};

struct GradientClippingHP {
    bool enabled = false;
    float configured_clip_norm = 0.0f;
    float effective_per_token_limit = EPSILON_GRADIENT_CLIP;
};

struct StartupModelCapacityHP {
    int max_cached_batch = 0;
    int max_cached_seq_len = 0;
    int max_tokens_per_batch = 0;
};

struct ParameterRegistrationHP {
    int num_layers = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int d_model = 0;
    int vocab_size = 0;

    bool register_untied_embedding = false;
    bool register_lm_head_bias = false;
    bool register_encoder_biases = false;
    bool register_layer_scale = false;
    bool register_scratch_block = false;
    bool register_reasoning_head = false;
    bool register_execution_block = false;
    bool register_slot_selector = false;
    bool register_mtp = false;
    int mtp_k = 0;
    bool register_final_rms_gamma = false;
};

struct GpuModelInitializationHP {
    bool use_gpu = false;
    int num_layers = 0;
    bool use_flash_attention = false;
    int min_seq_len_for_flash = 0;
};

struct RMSNormConstructionHP {
    int hidden_dim = 0;
    float epsilon = 0.0f;
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
    float residual_scale = 0.0f;
    bool is_gqa = false;
};

struct FeedForwardLayerConstructionHP {
    int d_model = 0;
    int d_ff = 0;
    bool use_bias = false;
    float dropout_rate = 0.0f;
    float residual_scale = 0.0f;
};

struct EmbeddingLayerConstructionHP {
    int vocab_size = 0;
    int d_model = 0;
};

struct LMHeadLayerConstructionHP {
    int d_model = 0;
    int vocab_size = 0;
    bool use_bias = false;
    bool tie_embeddings = false;
    bool center_hidden_states = false;
    bool project_out_pc1 = false;
    int pc1_power_iters = 0;
    bool center_logits = false;
    bool freeze_final_rms_gamma = false;
    float rms_epsilon = 0.0f;
};

struct ReasoningHeadConstructionHP {
    bool enabled = false;
    int d_model = 0;
    int atom_embedding_dim = 0;
    int num_ops = 0;
};

struct ExecutionBlockConstructionHP {
    bool enabled = false;
    int d_model = 0;
    int atom_embedding_dim = 0;
    int num_ops = 0;
    int num_slots = 0;
    int num_scratch_slots = EXECUTION_BLOCK_NUM_SCRATCH_SLOTS;
    int num_exec_steps = 0;
    int value_decode_input_dim = EXECUTION_BLOCK_VALUE_DECODE_INPUT_DIM;
    int value_decode_hidden_dim = EXECUTION_BLOCK_VALUE_DECODE_HIDDEN_DIM;
    int d_key = 0;
    int d_type = 0;
    int cross_attn_head_dim = 0;
    int cross_attn_topk = 0;
    float usage_decay = 0.0f;
    float inject_gate_temp = EXECUTION_BLOCK_INJECT_GATE_TEMP;
    int result_slot_mode = EXECUTION_BLOCK_RESULT_SLOT_MODE;
    int result_slot_index = EXECUTION_BLOCK_RESULT_SLOT_INDEX;
    bool debug_mode = EXECUTION_BLOCK_DEBUG_MODE;
    float entropy_collapse_threshold = EXECUTION_BLOCK_ENTROPY_COLLAPSE_THRESHOLD;
    float write_collapse_threshold = EXECUTION_BLOCK_WRITE_COLLAPSE_THRESHOLD;
    float magnitude_limit = EXECUTION_BLOCK_MAGNITUDE_LIMIT;
    float transition_hard_threshold = 0.0f;
    float div_invalid_penalty_weight = 0.0f;
    float div_magnitude_penalty_weight = 0.0f;
    float arg_reinforce_weight = 0.0f;
    float arg_reinforce_baseline_decay = 0.0f;
};

struct DecodeTimeSelectorConstructionHP {
    bool enabled = false;
    int d_model = 0;
    int d_selector = 0;
    int d_slot_features = DECODE_TIME_SLOT_FEATURE_DIM;
    int num_slots = 0;
    int scratch_slots = EXECUTION_BLOCK_NUM_SCRATCH_SLOTS;
    float selection_margin = 0.0f;
};

struct MTPConstructionHP {
    bool enabled = false;
    int k = 0;
    int vocab_size = 0;
    int d_model = 0;
    float alpha = 0.0f;
    int alpha_warmup_steps = 0;
};

struct InferenceCacheConstructionHP {
    int num_layers = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int d_model = 0;
    int d_ff = 0;
    int vocab_size = 0;
    std::size_t max_batch_size = 0;
    std::size_t max_seq_len_cache = 0;
    std::size_t max_tokens = 0;
    bool use_scratch_block = false;
    int scratch_block_atom_embedding_dim = 0;
    int scratch_block_max_atoms = 0;
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
    requirePositiveFiniteGroupingValue(hp.residual_scale, "residual_scale", caller);

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

inline CoreRunHP coreRunHP(const StartupConfig& config) {
    const auto& hp = config.hyperparameters;
    CoreRunHP view;
    view.epochs = hp.epochs;
    view.seed = hp.seed;
    view.gradient_accumulation_steps = hp.gradient_accumulation_steps;
    view.single_batch_overfit_enabled = hp.single_batch_overfit_enabled;
    view.single_batch_overfit_max_steps = hp.single_batch_overfit_max_steps;
    return view;
}

inline CapacityHP capacityHP(const StartupConfig& config) {
    CapacityHP view;
    view.batch_size = config.hyperparameters.batch_size;
    view.max_seq_len = config.max_seq_len;
    view.gradient_accumulation_steps = config.hyperparameters.gradient_accumulation_steps;
    return view;
}

inline DataLoadingHP dataLoadingHP(const StartupConfig& config) {
    DataLoadingHP view;
    view.min_seq_valid_tokens = config.hyperparameters.min_seq_valid_tokens;
    view.sliding_window_stride = config.sliding_window_stride;
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
    if (cfg.max_cached_batch <= 0) {
        throw std::runtime_error("startupLanguageModelConfig: max_cached_batch must be > 0, got " +
                                 std::to_string(cfg.max_cached_batch));
    }
    if (cfg.max_cached_seq_len <= 0) {
        throw std::runtime_error("startupLanguageModelConfig: max_cached_seq_len must be > 0, got " +
                                 std::to_string(cfg.max_cached_seq_len));
    }
    if (cfg.max_tokens_per_batch <= 0) {
        throw std::runtime_error("startupLanguageModelConfig: max_tokens_per_batch must be > 0, got " +
                                 std::to_string(cfg.max_tokens_per_batch));
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

inline LanguageModelConfig startupLanguageModelConfig(
    const StartupConfig& config,
    std::uint32_t vocab_size,
    const StartupModelCapacityHP& capacity)
{
    LanguageModelConfig cfg = config.hyperparameters.architecture;

    // Runtime startup facts authored outside ai_config.json but still mapped
    // through this grouping layer so model startup has one config read path.
    cfg.max_seq_len = config.max_seq_len;
    cfg.vocab_size = static_cast<int>(vocab_size);
    cfg.vocab_path = config.paths.vocab_path;
    cfg.infer_vocab_from_file = true;

    // GRIM-text training invariants. These are not caller fallbacks; they are
    // the single startup policy for this executable.
    cfg.causal_mask = true;
    cfg.use_pre_norm = true;
    cfg.fuse_qkv = true;

    cfg.max_cached_batch = capacity.max_cached_batch;
    cfg.max_cached_seq_len = capacity.max_cached_seq_len;
    cfg.max_tokens_per_batch = capacity.max_tokens_per_batch;

    cfg.computeDerivedValues();
    validateStartupLanguageModelConfig(cfg);
    return cfg;
}

inline ParameterRegistrationHP parameterRegistrationHP(
    const LanguageModelConfig& cfg)
{
    ParameterRegistrationHP view;
    view.num_layers = cfg.num_layers;
    view.num_heads = cfg.num_heads;
    view.num_kv_heads = cfg.num_kv_heads;
    view.head_dim = cfg.head_dim;
    view.d_model = cfg.d_model;
    view.vocab_size = cfg.vocab_size;
    view.register_untied_embedding = !cfg.tie_embeddings;
    view.register_lm_head_bias = cfg.use_bias;
    view.register_encoder_biases = cfg.use_bias;
    view.register_layer_scale = cfg.use_layer_scale;
    view.register_scratch_block = cfg.use_scratch_block;
    view.register_reasoning_head = cfg.reasoning_head_enabled;
    view.register_execution_block = cfg.execution_block_enabled;
    view.register_slot_selector = cfg.selector_enabled;
    view.register_mtp = cfg.mtp_enabled;
    view.mtp_k = cfg.mtp_k;
    view.register_final_rms_gamma = !cfg.lm_head_freeze_final_rms_gamma;
    return view;
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

inline RMSNormConstructionHP rmsNormConstructionHP(int hidden_dim,
                                                   float epsilon,
                                                   const char* caller)
{
    requirePositiveGroupingValue(hidden_dim, "hidden_dim", caller);
    requirePositiveFiniteGroupingValue(epsilon, "epsilon", caller);

    RMSNormConstructionHP view;
    view.hidden_dim = hidden_dim;
    view.epsilon = epsilon;
    return view;
}

inline EncoderLayerConstructionHP encoderLayerConstructionHP(
    const LanguageModelConfig& cfg)
{
    EncoderLayerConstructionHP view;
    const RMSNormConstructionHP rms_hp = rmsNormConstructionHP(
        cfg.d_model, cfg.rms_epsilon, "encoderLayerConstructionHP");
    requireValidGQAGrouping(cfg, "encoderLayerConstructionHP");
    requirePositiveGroupingValue(cfg.num_layers, "num_layers", "encoderLayerConstructionHP");
    requirePositiveGroupingValue(cfg.d_ff, "d_ff", "encoderLayerConstructionHP");
    requireDropoutProbability(cfg.dropout_rate, "dropout_rate", "encoderLayerConstructionHP");
    requireDropoutProbability(cfg.attention_dropout, "attention_dropout", "encoderLayerConstructionHP");
    if (cfg.use_flash_attention) {
        requirePositiveGroupingValue(cfg.min_seq_len_for_flash,
                                     "min_seq_len_for_flash",
                                     "encoderLayerConstructionHP");
    }
    view.num_layers = cfg.num_layers;
    view.d_model = rms_hp.hidden_dim;
    view.num_heads = cfg.num_heads;
    view.num_kv_heads = cfg.num_kv_heads;
    view.head_dim = cfg.head_dim;
    view.heads_per_kv_group = computeHeadsPerKVGroup(cfg.num_heads, cfg.num_kv_heads);
    view.kv_dim = computeKVProjectionSize(cfg.d_model, cfg.num_heads, cfg.num_kv_heads);
    view.qkv_dim = computeQKVProjectionSize(cfg.d_model, cfg.num_heads, cfg.num_kv_heads);
    view.d_ff = cfg.d_ff;
    view.rms_epsilon = rms_hp.epsilon;
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
    view.residual_scale =
        1.0f / std::sqrt(2.0f * static_cast<float>(cfg.num_layers));
    view.is_gqa = cfg.num_kv_heads < cfg.num_heads;
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
    if (!std::isfinite(encoder_hp.residual_scale) || encoder_hp.residual_scale <= 0.0f) {
        throw std::runtime_error("feedForwardLayerConstructionHP: residual_scale must be a positive finite value from encoderLayerConstructionHP");
    }
    view.d_model = encoder_hp.d_model;
    view.d_ff = encoder_hp.d_ff;
    view.use_bias = encoder_hp.use_bias;
    view.dropout_rate = encoder_hp.dropout_rate;
    view.residual_scale = encoder_hp.residual_scale;
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
    const RMSNormConstructionHP rms_hp = rmsNormConstructionHP(
        cfg.d_model, cfg.rms_epsilon, "lmHeadLayerConstructionHP");
    requirePositiveGroupingValue(cfg.vocab_size, "vocab_size", "lmHeadLayerConstructionHP");
    if (cfg.project_out_pc1) {
        requirePositiveGroupingValue(cfg.pc1_power_iters,
                                     "pc1_power_iters",
                                     "lmHeadLayerConstructionHP");
    }
    view.d_model = rms_hp.hidden_dim;
    view.vocab_size = cfg.vocab_size;
    view.use_bias = cfg.use_bias;
    view.tie_embeddings = cfg.tie_embeddings;
    view.center_hidden_states = cfg.lm_head_center_hidden_states;
    view.project_out_pc1 = cfg.project_out_pc1;
    view.pc1_power_iters = cfg.pc1_power_iters;
    view.center_logits = cfg.center_logits;
    view.freeze_final_rms_gamma = cfg.lm_head_freeze_final_rms_gamma;
    view.rms_epsilon = rms_hp.epsilon;
    return view;
}

inline RMSNormConstructionHP encoderRMSNormConstructionHP(
    const EncoderLayerConstructionHP& encoder_hp)
{
    return rmsNormConstructionHP(
        encoder_hp.d_model, encoder_hp.rms_epsilon, "encoderRMSNormConstructionHP");
}

inline RMSNormConstructionHP lmHeadRMSNormConstructionHP(
    const LMHeadLayerConstructionHP& lm_head_hp)
{
    return rmsNormConstructionHP(
        lm_head_hp.d_model, lm_head_hp.rms_epsilon, "lmHeadRMSNormConstructionHP");
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
    view.d_model = cfg.d_model;
    view.atom_embedding_dim = cfg.scratch_block_atom_embedding_dim;
    view.num_ops = cfg.execution_block_num_ops;
    view.num_slots = cfg.execution_block_num_slots;
    view.num_scratch_slots = EXECUTION_BLOCK_NUM_SCRATCH_SLOTS;
    view.num_exec_steps = cfg.execution_block_num_steps;
    view.value_decode_input_dim = EXECUTION_BLOCK_VALUE_DECODE_INPUT_DIM;
    view.value_decode_hidden_dim = EXECUTION_BLOCK_VALUE_DECODE_HIDDEN_DIM;
    view.d_key = cfg.execution_block_d_key;
    view.d_type = cfg.execution_block_d_type;
    view.cross_attn_head_dim = cfg.execution_block_cross_attn_head_dim;
    view.cross_attn_topk = cfg.execution_block_cross_attn_topk;
    view.usage_decay = cfg.execution_block_usage_decay;
    view.inject_gate_temp = EXECUTION_BLOCK_INJECT_GATE_TEMP;
    view.result_slot_mode = EXECUTION_BLOCK_RESULT_SLOT_MODE;
    view.result_slot_index = EXECUTION_BLOCK_RESULT_SLOT_INDEX;
    view.debug_mode = EXECUTION_BLOCK_DEBUG_MODE;
    view.entropy_collapse_threshold = EXECUTION_BLOCK_ENTROPY_COLLAPSE_THRESHOLD;
    view.write_collapse_threshold = EXECUTION_BLOCK_WRITE_COLLAPSE_THRESHOLD;
    view.magnitude_limit = EXECUTION_BLOCK_MAGNITUDE_LIMIT;
    view.transition_hard_threshold = cfg.execution_block_transition_hard_threshold;
    view.div_invalid_penalty_weight = cfg.div_invalid_penalty_weight;
    view.div_magnitude_penalty_weight = cfg.div_magnitude_penalty_weight;
    view.arg_reinforce_weight = cfg.arg_reinforce_weight;
    view.arg_reinforce_baseline_decay = cfg.arg_reinforce_baseline_decay;
    if (view.enabled) {
        requirePositiveGroupingValue(view.d_model, "d_model", "executionBlockConstructionHP");
        requirePositiveGroupingValue(view.atom_embedding_dim,
                                     "scratch_block_atom_embedding_dim",
                                     "executionBlockConstructionHP");
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
        requirePositiveGroupingValue(view.d_key, "execution_block_d_key", "executionBlockConstructionHP");
        requirePositiveGroupingValue(view.d_type, "execution_block_d_type", "executionBlockConstructionHP");
        requirePositiveGroupingValue(view.cross_attn_head_dim,
                                     "execution_block_cross_attn_head_dim",
                                     "executionBlockConstructionHP");
        requirePositiveGroupingValue(view.cross_attn_topk,
                                     "execution_block_cross_attn_topk",
                                     "executionBlockConstructionHP");
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
    view.d_slot_features = DECODE_TIME_SLOT_FEATURE_DIM;
    view.num_slots = cfg.execution_block_num_slots;
    view.scratch_slots = EXECUTION_BLOCK_NUM_SCRATCH_SLOTS;
    view.selection_margin = cfg.selector_selection_margin;
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

inline InferenceCacheConstructionHP inferenceCacheConstructionHP(
    const LanguageModelConfig& cfg)
{
    InferenceCacheConstructionHP view;
    requireValidGQAGrouping(cfg, "inferenceCacheConstructionHP");
    requirePositiveGroupingValue(cfg.num_layers, "num_layers", "inferenceCacheConstructionHP");
    requirePositiveGroupingValue(cfg.d_ff, "d_ff", "inferenceCacheConstructionHP");
    requirePositiveGroupingValue(cfg.vocab_size, "vocab_size", "inferenceCacheConstructionHP");
    requirePositiveGroupingValue(cfg.max_cached_batch,
                                 "max_cached_batch",
                                 "inferenceCacheConstructionHP");
    requirePositiveGroupingValue(cfg.max_seq_len, "max_seq_len", "inferenceCacheConstructionHP");
    requirePositiveGroupingValue(cfg.max_cached_seq_len,
                                 "max_cached_seq_len",
                                 "inferenceCacheConstructionHP");

    view.num_layers = cfg.num_layers;
    view.num_heads = cfg.num_heads;
    view.num_kv_heads = cfg.num_kv_heads;
    view.head_dim = cfg.head_dim;
    view.d_model = cfg.d_model;
    view.d_ff = cfg.d_ff;
    view.vocab_size = cfg.vocab_size;
    view.max_batch_size = static_cast<std::size_t>(cfg.max_cached_batch);
    view.max_seq_len_cache = static_cast<std::size_t>(
        std::min(cfg.max_seq_len, cfg.max_cached_seq_len));
    view.max_tokens = view.max_batch_size * view.max_seq_len_cache;
    if (view.max_tokens == 0 ||
        view.max_tokens > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("inferenceCacheConstructionHP: max_tokens=" +
                                 std::to_string(view.max_tokens) +
                                 " is outside TensorShape int capacity");
    }
    view.use_scratch_block = cfg.use_scratch_block;
    view.scratch_block_atom_embedding_dim = cfg.scratch_block_atom_embedding_dim;
    view.scratch_block_max_atoms = cfg.scratch_block_max_atoms;
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

