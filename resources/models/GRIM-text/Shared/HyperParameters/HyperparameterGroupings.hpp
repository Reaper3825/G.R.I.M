#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "HyperParameters_GPU.hpp"
#include "../Dynamic_LR/LRSchedule.hpp"

namespace GRIM::HyperParameters {

// Groupings are immutable read views over finalized HyperParameters_GPU.hpp-owned
// config, constants, and payload structs. Do not introduce authored defaults,
// validation policy, or derived-value computation here; add those to
// HyperParameters_GPU.hpp first.

template <typename T>
struct ImmutableArrayView {
    const T* data = nullptr;
    std::size_t size = 0;

    bool empty() const noexcept { return size == 0; }
    const T* begin() const noexcept { return data; }
    const T* end() const noexcept { return data ? data + size : nullptr; }
};

template <typename T>
inline ImmutableArrayView<T> immutableArrayView(const std::vector<T>& values) {
    return ImmutableArrayView<T>{values.empty() ? nullptr : values.data(), values.size()};
}

struct DataLoadingHP {
    int min_seq_valid_tokens = 0;
    int sliding_window_stride = 0;
};

struct PathsHP {
    std::string data_path;
    std::string vocab_path;
    std::string output_model_path;
    std::string checkpoint_dir;
    std::string log_dir;
    std::string status_path;
};

struct CheckpointLoadHP {
    std::string checkpoint_dir;
    std::string checkpoint_path;
    ModelExecutionMode execution_mode = ModelExecutionMode::INFERENCE;
};

struct TokenizerHP {
    std::string data_path;
    std::string vocab_path;
    int max_seq_len = 0;
    int target_vocab_size = 0;
    float character_coverage = 0.0f;
    int min_cleaned_text_length = 0;
    int min_subword_freq = 0;
    bool prune_during_mining = false;
    bool enable_parallel_subword_mining = false;
    int subword_mining_workers = 0;
    std::size_t subword_mining_max_bytes = 0;

    bool enable_atom_reasoning = false;
    bool detect_numbers = false;
    bool enable_byte_fallback = false;

    bool add_bos = false;
    bool add_eos = false;
    int number_encoder_max_digit_slots = 0;
    int number_encoder_max_abs_pow10 = 0;
    bool force_rebuild_vocab = false;
    bool only_mode = false;
    bool save_text_vocab = false;
    float vocab_score_multiplier = 0.0f;

    std::string current_curriculum;
    std::string current_model_training;
    int execution_block_num_steps = 0;
};

struct LearningRateScheduleInputs {
    float learning_rate = 0.0f;
    float cosine_decay_min_lr = 0.0f;
    int warmup_steps = 0;
    bool cosine_decay_enabled = false;
    bool cosine_warm_restarts = false;
};

struct TrainingSeedHP {
    int64_t seed = 0;
};

struct GenerationHP {
    SamplingStrategy strategy = SamplingStrategy::UNSPECIFIED;
    int max_new_tokens = 0;
    int min_new_tokens = 0;
    float temperature = 0.0f;
    int top_k = 0;
    float top_p = 0.0f;
    float min_p = 0.0f;
    float typical_p = 0.0f;
    float repetition_penalty = 0.0f;
    int repetition_penalty_window = 0;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    int num_return_sequences = 0;
    int eos_token_id = -1;
    int pad_token_id = -1;
    int bos_token_id = -1;
    int unk_token_id = -1;
    int no_repeat_ngram_size = 0;
    bool do_sample = false;
    std::vector<int> bad_words_ids;
    /// Token IDs to mask at sampling (e.g. byte-level digit tokens); `<NUM>` must remain unmasked.
    std::vector<int> masked_numeric_literal_ids;
    unsigned int seed = 0;
    bool enable_scratchblock_reasoning = false;
};

struct TapeLogHP {
    std::string default_level;
    bool equation_csv_enabled = false;
    bool stderr_enabled = false;
    std::size_t initial_capacity = 0;
    std::map<std::string, std::string> group_overrides;
};

struct LogRecorderLayerEnablesHP {
    bool embedding = false;
    bool rms_norm = false;
    bool attention = false;
    bool feed_forward = false;
    bool residual = false;
    bool encoding = false;
    bool serialization = false;
    bool execution_block = false;
};

struct LogRecorderHP {
    bool enabled = false;
    std::string default_level;
    std::map<std::string, std::string> modules;
    LogRecorderLayerEnablesHP layers;
};

struct OptimizerUpdateHP {
    OptimizerKind kind = OptimizerKind::ADAMW;
    float weight_decay = 0.0f;
    bool use_depth_aware_upsilon = false;
    float beta1 = 0.0f;
    float beta2 = 0.0f;
    float epsilon = 0.0f;
    int embedding_freeze_after_step = -1;
};

struct GradientClippingHP {
    bool enabled = true;
    float configured_clip_norm = 1.0f;
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

struct AutoStopHP {
    bool enabled = false;
    int plateau_patience = 0;
    float plateau_min_delta = 0.0f;
    float high_loss_threshold = 0.0f;
    int high_loss_patience = 0;
};

struct TrainingScheduleHP {
    int epochs = 0;
    int gradient_accumulation_steps = 0;
    float accumulation_normalization_scale = 1.0f;
    bool single_batch_overfit_enabled = false;
    int single_batch_overfit_max_steps = 0;
    bool shuffle_train_enabled = false;
    int shuffle_train_epochs = 0;
};

struct SoftRestartHP {
    bool enabled = false;
    float loss_increase_threshold = 0.0f;
    int max_step_window = 0;
    int cooldown_steps = 0;
};

struct TrainingRuntimeControlHP {
    int log_interval = 0;
    int atom_stats_interval = 0;
    int atom_stats_max_seqs = 0;
    int validation_interval = 0;
    int checkpoint_interval = 0;
    bool logit_update_trace_enabled = false;
    int logit_update_trace_interval = 0;
};

struct TelemetryControlHP {
    bool enabled = false;
    float spike_mild_threshold = 0.0f;
    float spike_moderate_threshold = 0.0f;
    float spike_severe_threshold = 0.0f;
    float moderate_grad_scale = 0.0f;
    int moderate_cooldown_extension = 0;
    float min_grad_for_nonzero_loss = 0.0f;
    float loss_threshold_for_grad_check = 0.0f;
    int max_consecutive_zero_grad_steps = 0;
    float seq_len_regime_change_threshold = 0.0f;
    int regime_change_suppression_steps = 0;
    float volatility_damping_threshold = 0.0f;
    float max_volatility_damping = 0.0f;
    float gradient_decay_threshold = 0.0f;
    float max_decay_boost = 0.0f;
    float progress_boost_threshold = 0.0f;
    float max_progress_boost = 0.0f;
    float outlier_frequency_trigger = 0.0f;
    float outlier_persistence_trigger = 0.0f;
    float anchor_drift_sigma_multiplier = 0.0f;
    int soft_restart_cooldown_steps = 0;
    int warmup_steps = 0;
    int baseline_stabilization_steps = 0;
    bool verbose_logging = false;
    bool fail_loud_on_accumulation_bug = false;
    bool plateau_noise_enabled = false;
    int plateau_noise_patience = 0;
    float plateau_noise_variance_threshold = 0.0f;
    float plateau_noise_std = 0.0f;
    bool plateau_noise_proportional = false;
    int plateau_noise_cooldown = 0;
    int plateau_noise_max_per_epoch = 0;
};

struct TelemetryLatticeHP {
    int num_levels = 0;
    int num_streams = 0;
    float beta_mu = 0.0f;
    float beta_a = 0.0f;
    float beta_delta = 0.0f;
    float beta_r = 0.0f;
    float beta_run = 0.0f;
    float beta_v = 0.0f;
    float k_out0 = 0.0f;
    float alpha_v = 0.0f;
    float epsilon = 0.0f;
    bool strict_mode = false;
};

struct StabilityOverrideHP {
    bool enabled = false;
    int batch_size = 0;
    int max_seq_len = 0;
    float clip_per_token = 0.0f;
};

struct TrainingFixedShapeHP {
    int batch_size = 0;
    int max_seq_len = 0;
    int max_tokens_per_batch = 0;
};

struct TrainingStateWorkspaceHP {
    int batch_size = 0;
    int max_tokens_per_batch = 0;
    bool mtp_enabled = false;
    int mtp_k = 0;
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
    ImmutableArrayView<float> alibi_slopes{};
    ImmutableArrayView<float> rope_inv_freq{};
};

struct EncoderLayerConstructionHP {
    int num_layers = 0;
    int d_model = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int rotary_dim = 0;
    float attention_softmax_scale = 0.0f;
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
    bool attention_off_by_one = false;
    float residual_projection_init_gain = 0.0f;
    bool is_gqa = false;
    bool freeze_learned_rms_gammas = false;
};

struct EncoderSelfAttentionHP {
    int d_model = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int rotary_dim = 0;
    float attention_softmax_scale = 0.0f;
    int heads_per_kv_group = 0;
    int kv_dim = 0;
    int qkv_dim = 0;
    bool causal_mask = false;
    bool use_flash_attention = false;
    int min_seq_len_for_flash = 0;
    bool use_bias = false;
    float attention_dropout = 0.0f;
    bool dropout_enabled = false;
    bool qk_norm_enabled = false;
    bool attention_off_by_one = false;
    bool is_gqa = false;
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
    float embedding_scale = 1.0f;
};

struct LMHeadLayerConstructionHP {
    int d_model = 0;
    int vocab_size = 0;
    int training_batch_size = 0;
    int training_rows_per_sequence = 0;
    bool use_bias = false;
    bool unigram_bias = false;
    bool tie_embeddings = false;
    bool center_hidden_states = false;
    bool project_out_pc1 = false;
    int pc1_power_iters = 0;
    bool center_logits = false;
    bool freeze_learned_rms_gammas = false;
    float rms_epsilon = 0.0f;
    // Head-side residual SwiGLU adapter: u = z + mlp_alpha * SwiGLU_MLP(z)
    bool mlp_enabled = false;
    int mlp_d_ff = 0;
    float mlp_alpha = 0.0f;
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

// NumberEncoder construction view — numeric-meaning input path.
// Encodes (digit, pow10) contribution slots plus a global mantissa/exponent
// feature head per docs/ATOM_SELECTOR_IMPLEMENTATION_PLAN.md. Selection-side
// representation; execution consumes downstream results only.
struct NumberEncoderConstructionHP {
    bool enabled = false;
    int d_model = 0;
    int d_hidden = 0;          // contribution-MLP hidden width
    int max_digit_slots = 0;   // fixed digit-slot capacity per numeric atom
    int max_abs_pow10 = 0;     // place-exponent range; buckets span [-max, +max]
    int pow10_buckets = 0;     // derived: 2 * max_abs_pow10 + 1
    bool use_bias = true;      // gates the contribution/global MLP hidden biases (b_c1/b_g1)
};

struct MTPConstructionHP {
    bool enabled = false;
    int k = 0;
    int vocab_size = 0;
    int d_model = 0;
    float alpha = 0.0f;
};

struct MTPFeatureHP {
    bool enabled = false;
    int k = 0;
};

struct MTPDiagnosticHP {
    bool log_ratio_monitor = false;
};

struct LatentTrajectoryPresetHP {
    bool enabled = false;

    int d_model = 0;
    int mtp_k = 0;
    int vocab_size = 0;
    int fuse_dim = 0;
    int preset_dim = 0;
    int gate_dim = 0;

    float preset_scale = 0.0f;
    float gate_bias_init = 0.0f;

    bool use_token_mean = false;
    bool use_mtp_logits = false;
    bool use_mtp_hidden = false;
    bool use_entropy = false;
    bool use_delta_target = false;
    bool use_consistency_loss = false;
    bool use_diversity_loss = false;
    bool use_gate_sparsity_loss = false;

    float lambda_traj = 0.0f;
    float lambda_delta = 0.0f;
    float lambda_consistency = 0.0f;
    float lambda_diversity = 0.0f;
    float lambda_gate = 0.0f;
};

// Unified model-side config payload for future Phase2 handoff -> Phase2 training.
// Immutable read view rooted on AiConfigSnapshot (raw document owner) and
// assembled from training.config authored leaves plus explicit derived formulas.
struct ModelHP {
    int training_batch_size = 0;
    int training_max_seq_len = 0;
    int training_max_tokens_per_batch = 0;

    int embedding_vocab_size = 0;
    int embedding_d_model = 0;
    float embedding_scale = 1.0f;

    int encoder_num_layers = 0;
    int encoder_d_model = 0;
    int encoder_num_heads = 0;
    int encoder_num_kv_heads = 0;
    int encoder_head_dim = 0;
    int encoder_rotary_dim = 0;
    float encoder_attention_softmax_scale = 0.0f;
    int encoder_heads_per_kv_group = 0;
    int encoder_kv_dim = 0;
    int encoder_qkv_dim = 0;
    int encoder_d_ff = 0;
    float encoder_rms_epsilon = 0.0f;
    bool encoder_causal_mask = false;
    bool encoder_use_flash_attention = false;
    int encoder_min_seq_len_for_flash = 0;
    bool encoder_use_layer_scale = false;
    float encoder_layer_scale_init = 0.0f;
    bool encoder_center_encoder_residuals = false;
    bool encoder_use_bias = false;
    float encoder_dropout_rate = 0.0f;
    float encoder_attention_dropout = 0.0f;
    bool encoder_qk_norm_enabled = false;
    bool encoder_attention_off_by_one = false;
    float encoder_residual_projection_init_gain = 0.0f;
    bool encoder_is_gqa = false;
    bool encoder_freeze_learned_rms_gammas = false;

    int lm_head_d_model = 0;
    int lm_head_vocab_size = 0;
    int lm_head_training_batch_size = 0;
    int lm_head_training_rows_per_sequence = 0;
    bool lm_head_use_bias = false;
    bool lm_head_unigram_bias = false;
    bool lm_head_tie_embeddings = false;
    bool lm_head_center_hidden_states = false;
    bool lm_head_project_out_pc1 = false;
    int lm_head_pc1_power_iters = 0;
    bool lm_head_center_logits = false;
    bool lm_head_freeze_learned_rms_gammas = false;
    float lm_head_rms_epsilon = 0.0f;
    bool lm_head_mlp_enabled = false;
    int lm_head_mlp_d_ff = 0;
    float lm_head_mlp_alpha = 0.0f;

    int atom_embedding_dim = 0;

    bool execution_block_enabled = false;
    int execution_block_layer = -1;
    int execution_block_d_model = 0;
    int execution_block_num_ops = 0;
    int execution_block_num_slots = 0;
    int execution_block_num_scratch_slots = 0;
    int execution_block_num_exec_steps = 0;
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
    float execution_block_div_invalid_penalty_weight = 0.0f;
    float execution_block_div_magnitude_penalty_weight = 0.0f;
    float execution_block_arg_reinforce_weight = 0.0f;
    float execution_block_arg_reinforce_baseline_decay = 0.0f;
    float execution_block_entropy_aux_weight = 0.0f;
    float execution_block_structured_ce_weight = 0.0f;

    bool number_encoder_enabled = false;
    int number_encoder_d_model = 0;
    int number_encoder_d_hidden = 0;
    int number_encoder_max_digit_slots = 0;
    int number_encoder_max_abs_pow10 = 0;

    bool mtp_enabled = false;
    int mtp_k = 0;
    int mtp_vocab_size = 0;
    int mtp_d_model = 0;
    float mtp_alpha = 0.0f;

    bool latent_trajectory_preset_enabled = false;
    int latent_trajectory_preset_d_model = 0;
    int latent_trajectory_preset_mtp_k = 0;
    int latent_trajectory_preset_vocab_size = 0;
    int latent_trajectory_preset_fuse_dim = 0;
    int latent_trajectory_preset_dim = 0;
    int latent_trajectory_preset_gate_dim = 0;
    float latent_trajectory_preset_scale = 0.0f;
    float latent_trajectory_preset_gate_bias_init = 0.0f;
    bool latent_trajectory_preset_use_token_mean = false;
    bool latent_trajectory_preset_use_mtp_logits = false;
    bool latent_trajectory_preset_use_mtp_hidden = false;
    bool latent_trajectory_preset_use_entropy = false;
    bool latent_trajectory_preset_use_delta_target = false;
    bool latent_trajectory_preset_use_consistency_loss = false;
    bool latent_trajectory_preset_use_diversity_loss = false;
    bool latent_trajectory_preset_use_gate_sparsity_loss = false;
    float latent_trajectory_preset_lambda_traj = 0.0f;
    float latent_trajectory_preset_lambda_delta = 0.0f;
    float latent_trajectory_preset_lambda_consistency = 0.0f;
    float latent_trajectory_preset_lambda_diversity = 0.0f;
    float latent_trajectory_preset_lambda_gate = 0.0f;

    PositionalEncodingType positional_encoding = PositionalEncodingType::UNSPECIFIED;
    bool structured_ce_enabled = false;
};

inline TrainingFixedShapeHP trainingFixedShapeHP(
    const LanguageModelConfig& config)
{
    TrainingFixedShapeHP view;
    view.batch_size = config.batch_size;
    view.max_seq_len = config.max_seq_len;
    view.max_tokens_per_batch = config.max_tokens_per_batch;
    return view;
}

inline TrainingStateWorkspaceHP trainingStateWorkspaceHP(
    const LanguageModelConfig& config)
{
    TrainingStateWorkspaceHP view;
    view.batch_size = config.batch_size;
    view.max_tokens_per_batch = config.max_tokens_per_batch;
    view.mtp_enabled = config.mtp_enabled;
    view.mtp_k = config.mtp_k;
    return view;
}

inline DataLoadingHP dataLoadingHP(const LanguageModelConfig& config) {
    DataLoadingHP view;
    view.min_seq_valid_tokens = config.min_seq_valid_tokens;
    view.sliding_window_stride = config.sliding_window_stride;
    return view;
}

inline PathsHP pathsHP(const LanguageModelConfig& config)
{
    PathsHP view;
    view.data_path = config.data_path;
    view.vocab_path = config.vocab_path;
    view.output_model_path = config.output_model_path;
    view.checkpoint_dir = config.checkpoint_dir;
    view.log_dir = config.log_dir;
    view.status_path = config.status_path;
    return view;
}


inline LearningRateScheduleInputs learningRateScheduleInputs(
    const LanguageModelConfig& hp)
{
    LearningRateScheduleInputs inputs;
    inputs.learning_rate = hp.learning_rate;
    inputs.cosine_decay_min_lr = hp.cosine_decay_min_lr;
    inputs.warmup_steps = hp.warmup_steps;
    inputs.cosine_decay_enabled = hp.cosine_decay_enabled;
    inputs.cosine_warm_restarts = hp.cosine_warm_restarts;
    return inputs;
}

inline LearningRateScheduleInputs learningRateScheduleInputs(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    LearningRateScheduleInputs inputs;
    inputs.learning_rate = snapshotTrainingConfigField<float>(snapshot, "learning_rate");
    inputs.cosine_decay_min_lr = snapshotTrainingConfigField<float>(snapshot, "cosine_decay_min_lr");
    inputs.warmup_steps = snapshotTrainingConfigField<int>(snapshot, "warmup_steps");
    inputs.cosine_decay_enabled = snapshotTrainingConfigField<bool>(snapshot, "cosine_decay_enabled");
    inputs.cosine_warm_restarts = snapshotTrainingConfigField<bool>(snapshot, "cosine_warm_restarts");
    return inputs;
}

inline TrainingSeedHP trainingSeedHP(const LanguageModelConfig& config)
{
    TrainingSeedHP view;
    view.seed = config.seed;
    return view;
}

inline TapeLogHP tapeLogHP(
    const LanguageModelConfig& hp)
{
    TapeLogHP view;
    view.default_level = hp.logging_default_level;
    view.equation_csv_enabled = hp.logging_equation_csv_enabled;
    view.stderr_enabled = hp.logging_stderr_enabled;
    view.initial_capacity = hp.logging_initial_capacity;
    view.group_overrides = hp.logging_group_overrides;
    return view;
}

inline LogRecorderHP logRecorderHP(
    const LanguageModelConfig& hp)
{
    LogRecorderHP view;
    view.enabled = hp.log_recorder_enabled;
    view.default_level = hp.log_recorder_default_level;
    view.modules = hp.log_recorder_modules;
    view.layers.embedding = hp.log_recorder_layer_embedding;
    view.layers.rms_norm = hp.log_recorder_layer_rms_norm;
    view.layers.attention = hp.log_recorder_layer_attention;
    view.layers.feed_forward = hp.log_recorder_layer_feed_forward;
    view.layers.residual = hp.log_recorder_layer_residual;
    view.layers.encoding = hp.log_recorder_layer_encoding;
    view.layers.serialization = hp.log_recorder_layer_serialization;
    view.layers.execution_block = hp.log_recorder_layer_execution_block;
    return view;
}

inline OptimizerUpdateHP optimizerUpdateHP(
    const LanguageModelConfig& hp)
{
    OptimizerUpdateHP view;
    view.kind = hp.optimizer_kind;
    view.weight_decay = hp.weight_decay;
    view.use_depth_aware_upsilon = hp.use_depth_aware_upsilon;
    view.beta1 = hp.optimizer_beta1;
    view.beta2 = hp.optimizer_beta2;
    view.epsilon = hp.optimizer_epsilon;
    view.embedding_freeze_after_step = hp.optimizer_embedding_freeze_after_step;
    return view;
}

inline GradientClippingHP gradientClippingHP(
    const LanguageModelConfig& hp)
{
    GradientClippingHP view;
    view.configured_clip_norm = hp.grad_clip_norm;
    view.enabled = hp.grad_clip_enabled;
    view.effective_per_token_limit = hp.effective_per_token_grad_limit;
    return view;
}

inline LossConfigHP lossConfigHP(
    const LanguageModelConfig& hp)
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
    return view;
}

inline AutoStopHP autoStopHP(
    const LanguageModelConfig& hp)
{
    AutoStopHP view;
    view.enabled = hp.auto_stop_enabled;
    view.plateau_patience = hp.auto_stop_plateau_patience;
    view.plateau_min_delta = hp.auto_stop_plateau_min_delta;
    view.high_loss_threshold = hp.auto_stop_high_loss_threshold;
    view.high_loss_patience = hp.auto_stop_high_loss_patience;
    return view;
}

inline TrainingScheduleHP trainingScheduleHP(
    const LanguageModelConfig& hp)
{
    TrainingScheduleHP view;
    view.epochs = hp.epochs;
    view.gradient_accumulation_steps = hp.gradient_accumulation_steps;
    view.accumulation_normalization_scale = (hp.gradient_accumulation_steps > 0)
        ? 1.0f / static_cast<float>(hp.gradient_accumulation_steps)
        : 1.0f;
    view.single_batch_overfit_enabled = hp.single_batch_overfit_enabled;
    view.single_batch_overfit_max_steps = hp.single_batch_overfit_max_steps;
    view.shuffle_train_enabled = hp.shuffle_train_enabled;
    view.shuffle_train_epochs = hp.shuffle_train_epochs;
    return view;
}

inline SoftRestartHP softRestartHP(
    const LanguageModelConfig& hp)
{
    SoftRestartHP view;
    view.enabled = hp.soft_restart_enabled;
    view.loss_increase_threshold = hp.soft_restart_loss_increase_threshold;
    view.max_step_window = hp.soft_restart_max_step_window;
    view.cooldown_steps = hp.soft_restart_cooldown_steps;
    return view;
}

inline TrainingRuntimeControlHP trainingRuntimeControlHP(
    const LanguageModelConfig& hp)
{
    TrainingRuntimeControlHP view;
    view.log_interval = hp.log_interval;
    view.atom_stats_interval = hp.atom_stats_interval;
    view.atom_stats_max_seqs = hp.atom_stats_max_seqs;
    view.validation_interval = hp.validation_interval;
    view.checkpoint_interval = hp.checkpoint_interval;
    view.logit_update_trace_enabled = hp.logit_update_trace_enabled;
    view.logit_update_trace_interval = hp.logit_update_trace_interval;
    return view;
}

inline TelemetryControlHP telemetryControlHP(
    const LanguageModelConfig& hp)
{
    TelemetryControlHP view;
    view.enabled = hp.telemetry_control_enabled;
    view.spike_mild_threshold = hp.telemetry_spike_mild_threshold;
    view.spike_moderate_threshold = hp.telemetry_spike_moderate_threshold;
    view.spike_severe_threshold = hp.telemetry_spike_severe_threshold;
    view.moderate_grad_scale = hp.telemetry_moderate_grad_scale;
    view.moderate_cooldown_extension = hp.telemetry_moderate_cooldown_extension;
    view.min_grad_for_nonzero_loss = hp.telemetry_min_grad_for_nonzero_loss;
    view.loss_threshold_for_grad_check = hp.telemetry_loss_threshold_for_grad_check;
    view.max_consecutive_zero_grad_steps = hp.telemetry_max_consecutive_zero_grad_steps;
    view.seq_len_regime_change_threshold = hp.telemetry_seq_len_regime_change_threshold;
    view.regime_change_suppression_steps = hp.telemetry_regime_change_suppression_steps;
    view.volatility_damping_threshold = hp.telemetry_volatility_damping_threshold;
    view.max_volatility_damping = hp.telemetry_max_volatility_damping;
    view.gradient_decay_threshold = hp.telemetry_gradient_decay_threshold;
    view.max_decay_boost = hp.telemetry_max_decay_boost;
    view.progress_boost_threshold = hp.telemetry_progress_boost_threshold;
    view.max_progress_boost = hp.telemetry_max_progress_boost;
    view.outlier_frequency_trigger = hp.telemetry_outlier_frequency_trigger;
    view.outlier_persistence_trigger = hp.telemetry_outlier_persistence_trigger;
    view.anchor_drift_sigma_multiplier = hp.telemetry_anchor_drift_sigma_multiplier;
    view.soft_restart_cooldown_steps = hp.telemetry_soft_restart_cooldown_steps;
    view.warmup_steps = hp.telemetry_warmup_steps;
    view.baseline_stabilization_steps = hp.telemetry_baseline_stabilization_steps;
    view.verbose_logging = hp.telemetry_verbose_logging;
    view.fail_loud_on_accumulation_bug = hp.telemetry_fail_loud_on_accumulation_bug;
    view.plateau_noise_enabled = hp.telemetry_plateau_noise_enabled;
    view.plateau_noise_patience = hp.telemetry_plateau_noise_patience;
    view.plateau_noise_variance_threshold = hp.telemetry_plateau_noise_variance_threshold;
    view.plateau_noise_std = hp.telemetry_plateau_noise_std;
    view.plateau_noise_proportional = hp.telemetry_plateau_noise_proportional;
    view.plateau_noise_cooldown = hp.telemetry_plateau_noise_cooldown;
    view.plateau_noise_max_per_epoch = hp.telemetry_plateau_noise_max_per_epoch;
    return view;
}

inline TelemetryLatticeHP telemetryLatticeHP(
    const LanguageModelConfig& hp)
{
    TelemetryLatticeHP view;
    view.num_levels = hp.telemetry_lattice_num_levels;
    view.num_streams = hp.telemetry_lattice_num_streams;
    view.beta_mu = hp.telemetry_lattice_beta_mu;
    view.beta_a = hp.telemetry_lattice_beta_a;
    view.beta_delta = hp.telemetry_lattice_beta_delta;
    view.beta_r = hp.telemetry_lattice_beta_r;
    view.beta_run = hp.telemetry_lattice_beta_run;
    view.beta_v = hp.telemetry_lattice_beta_v;
    view.k_out0 = hp.telemetry_lattice_k_out0;
    view.alpha_v = hp.telemetry_lattice_alpha_v;
    view.epsilon = hp.telemetry_lattice_epsilon;
    view.strict_mode = hp.telemetry_lattice_strict_mode;
    return view;
}

inline StabilityOverrideHP stabilityOverrideHP(
    const LanguageModelConfig& hp)
{
    StabilityOverrideHP view;
    view.enabled = hp.stability_overrides_enabled;
    view.batch_size = hp.stability_override_batch_size;
    view.max_seq_len = hp.stability_override_max_seq_len;
    view.clip_per_token = hp.stability_override_clip_per_token;
    return view;
}

inline PBMConstructionHP pbmConstructionHP(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    PBMConstructionHP view;
    view.num_heads = snapshotTrainingConfigField<int>(snapshot, "num_heads");
    view.num_kv_heads = snapshotTrainingConfigField<int>(snapshot, "num_kv_heads");
    view.head_dim = snapshotTrainingConfigField<int>(snapshot, "head_dim");
    view.rotary_dim = snapshotTrainingConfigField<int>(snapshot, "rotary_dim");
    view.max_seq_len = snapshotTrainingConfigField<int>(snapshot, "max_seq_len");
    view.rope_base_seq_len = snapshotTrainingConfigField<int>(snapshot, "rope_base_seq_len");
    view.alibi_min_locality_distance = snapshotTrainingConfigField<int>(snapshot, "alibi_min_locality_distance");
    view.alibi_slope_exponent = snapshotTrainingConfigField<float>(snapshot, "alibi_slope_exponent");
    view.alibi_max_bias = snapshotTrainingConfigField<float>(snapshot, "alibi_max_bias");
    view.rope_theta = snapshotTrainingConfigField<float>(snapshot, "rope_theta");
    view.rope_scaling = snapshotTrainingConfigField<float>(snapshot, "rope_scaling");
    const auto alibi_slopes = snapshotTrainingConfigField<std::vector<float>>(snapshot, "pbm_alibi_slopes");
    const auto rope_inv_freq = snapshotTrainingConfigField<std::vector<float>>(snapshot, "pbm_rope_inv_freq");
    static thread_local std::vector<float> alibi_storage;
    static thread_local std::vector<float> rope_storage;
    alibi_storage = alibi_slopes;
    rope_storage = rope_inv_freq;
    view.alibi_slopes = immutableArrayView(alibi_storage);
    view.rope_inv_freq = immutableArrayView(rope_storage);
    return view;
}

inline FeedForwardLayerConstructionHP feedForwardLayerConstructionHP(
    const EncoderLayerConstructionHP& encoder_hp)
{
    FeedForwardLayerConstructionHP view;
    view.d_model = encoder_hp.d_model;
    view.d_ff = encoder_hp.d_ff;
    view.use_bias = encoder_hp.use_bias;
    view.dropout_rate = encoder_hp.dropout_rate;
    view.residual_projection_init_gain = encoder_hp.residual_projection_init_gain;
    return view;
}

inline EncoderSelfAttentionHP encoderSelfAttentionHP(
    const EncoderLayerConstructionHP& encoder_hp,
    bool dropout_enabled)
{
    EncoderSelfAttentionHP view;
    view.d_model = encoder_hp.d_model;
    view.num_heads = encoder_hp.num_heads;
    view.num_kv_heads = encoder_hp.num_kv_heads;
    view.head_dim = encoder_hp.head_dim;
    view.rotary_dim = encoder_hp.rotary_dim;
    view.attention_softmax_scale = encoder_hp.attention_softmax_scale;
    view.heads_per_kv_group = encoder_hp.heads_per_kv_group;
    view.kv_dim = encoder_hp.kv_dim;
    view.qkv_dim = encoder_hp.qkv_dim;
    view.causal_mask = encoder_hp.causal_mask;
    view.use_flash_attention = encoder_hp.use_flash_attention;
    view.min_seq_len_for_flash = encoder_hp.min_seq_len_for_flash;
    view.use_bias = encoder_hp.use_bias;
    view.attention_dropout = encoder_hp.attention_dropout;
    view.dropout_enabled = dropout_enabled;
    view.qk_norm_enabled = encoder_hp.qk_norm_enabled;
    view.attention_off_by_one = encoder_hp.attention_off_by_one;
    view.is_gqa = encoder_hp.is_gqa;
    return view;
}

inline TrainingFixedShapeHP trainingFixedShapeHP(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    TrainingFixedShapeHP view;
    view.batch_size = snapshotEffectiveBatchSize(snapshot, "trainingFixedShapeHP");
    view.max_seq_len = snapshotEffectiveMaxSeqLen(snapshot, "trainingFixedShapeHP");
    view.max_tokens_per_batch = computeMaxTokensPerBatch(
        view.batch_size,
        view.max_seq_len,
        "trainingFixedShapeHP");
    return view;
}

inline TrainingStateWorkspaceHP trainingStateWorkspaceHP(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    const auto fixed_shape = trainingFixedShapeHP(snapshot);

    TrainingStateWorkspaceHP view;
    view.batch_size = fixed_shape.batch_size;
    view.max_tokens_per_batch = fixed_shape.max_tokens_per_batch;
    view.mtp_enabled = snapshotTrainingConfigField<bool>(snapshot, "mtp_enabled");
    view.mtp_k = snapshotTrainingConfigField<int>(snapshot, "mtp_k");
    return view;
}

inline DataLoadingHP dataLoadingHP(const GRIM::Config::AiConfigSnapshot& snapshot) {
    const auto fixed_shape = trainingFixedShapeHP(snapshot);

    DataLoadingHP view;
    view.min_seq_valid_tokens = fixed_shape.max_seq_len / 4;
    view.sliding_window_stride = snapshotTrainingConfigField<int>(snapshot, "sliding_window_stride");
    return view;
}

inline PathsHP pathsHP(const GRIM::Config::AiConfigSnapshot& snapshot)
{
    PathsHP view;
    view.data_path = resolveMappedPath(snapshotTrainingConfigField<std::string>(snapshot, "grim_text_training_data"));
    view.vocab_path = resolveMappedPath(snapshotTrainingConfigField<std::string>(snapshot, "grim_text_vocab"));
    view.output_model_path = resolveMappedPath(snapshotTrainingConfigField<std::string>(snapshot, "grim_text_model"));
    view.checkpoint_dir = resolveMappedPath(snapshotTrainingConfigField<std::string>(snapshot, "grim_text_checkpoints"));
    view.log_dir = resolveMappedPath(snapshotTrainingConfigField<std::string>(snapshot, "grim_text_logs"));
    view.status_path = resolveMappedPath(snapshotTrainingConfigField<std::string>(snapshot, "grim_text_training_status"));
    return view;
}

inline CheckpointLoadHP checkpointLoadHP(
    const GRIM::Config::AiConfigSnapshot& snapshot,
    const std::string& checkpoint_path,
    ModelExecutionMode execution_mode)
{
    CheckpointLoadHP view;
    view.checkpoint_dir = pathsHP(snapshot).checkpoint_dir;
    view.checkpoint_path = checkpoint_path;
    view.execution_mode = execution_mode;
    return view;
}

inline TokenizerHP tokenizerHP(const GRIM::Config::AiConfigSnapshot& snapshot) {
    const auto paths = pathsHP(snapshot);

    TokenizerHP view;
    view.data_path = paths.data_path;
    view.vocab_path = paths.vocab_path;
    view.target_vocab_size = snapshotTokenizerTargetVocabSize(snapshot);
    view.character_coverage = snapshotTrainingConfigField<float>(snapshot, "tokenizer_character_coverage");
    view.min_cleaned_text_length = snapshotTrainingConfigField<int>(snapshot, "tokenizer_min_cleaned_text_length");
    view.min_subword_freq = snapshotTrainingConfigField<int>(snapshot, "tokenizer_min_subword_freq");
    view.prune_during_mining = snapshotTrainingConfigField<bool>(snapshot, "tokenizer_prune_during_mining");
    view.enable_parallel_subword_mining = snapshotTrainingConfigField<bool>(snapshot, "tokenizer_enable_parallel_subword_mining");
    view.subword_mining_workers = snapshotTrainingConfigField<int>(snapshot, "tokenizer_subword_mining_workers");
    view.subword_mining_max_bytes = snapshotTrainingConfigField<std::size_t>(snapshot, "tokenizer_subword_mining_max_bytes");
    view.enable_atom_reasoning = snapshotTrainingConfigField<bool>(snapshot, "tokenizer_enable_atom_reasoning");
    view.detect_numbers = snapshotTrainingConfigField<bool>(snapshot, "tokenizer_detect_numbers");
    view.enable_byte_fallback = snapshotTrainingConfigField<bool>(snapshot, "tokenizer_enable_byte_fallback");
    view.add_bos = snapshotTrainingConfigField<bool>(snapshot, "tokenizer_add_bos");
    view.add_eos = snapshotTrainingConfigField<bool>(snapshot, "tokenizer_add_eos");
    view.number_encoder_max_digit_slots = snapshotTrainingConfigField<int>(snapshot, "number_encoder_max_digit_slots");
    view.number_encoder_max_abs_pow10 = snapshotTrainingConfigField<int>(snapshot, "number_encoder_max_abs_pow10");
    view.force_rebuild_vocab = snapshotTrainingConfigField<bool>(snapshot, "force_rebuild_vocab");
    view.only_mode = snapshotTrainingConfigField<bool>(snapshot, "subprocess_tokenizer_only_mode");
    view.save_text_vocab = snapshotTrainingConfigField<bool>(snapshot, "tokenizer_save_text_vocab");
    view.vocab_score_multiplier = snapshotTrainingConfigField<float>(snapshot, "tokenizer_vocab_score_multiplier");
    view.current_curriculum = snapshotTrainingConfigField<std::string>(snapshot, "current_curriculum");
    view.current_model_training = snapshotTrainingConfigField<std::string>(snapshot, "current_model_training");
    view.execution_block_num_steps = snapshotTrainingConfigField<int>(snapshot, "execution_block_num_steps");
    view.max_seq_len = snapshotTrainingConfigField<int>(snapshot, "max_seq_len");
    return view;
}

inline TrainingSeedHP trainingSeedHP(const GRIM::Config::AiConfigSnapshot& snapshot)
{
    TrainingSeedHP view;
    view.seed = snapshotTrainingConfigField<int64_t>(snapshot, "seed");
    return view;
}

inline TapeLogHP tapeLogHP(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    TapeLogHP view;
    view.default_level = snapshotTrainingConfigField<std::string>(snapshot, "logging_default_level");
    view.equation_csv_enabled = snapshotTrainingConfigField<bool>(snapshot, "logging_equation_csv_enabled");
    view.stderr_enabled = snapshotTrainingConfigField<bool>(snapshot, "logging_stderr_enabled");
    view.initial_capacity = snapshotTrainingConfigField<std::size_t>(snapshot, "logging_initial_capacity");
    view.group_overrides = snapshotTrainingConfigField<std::map<std::string, std::string>>(snapshot, "logging_group_overrides");
    return view;
}

inline LogRecorderHP logRecorderHP(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    LogRecorderHP view;
    view.enabled = snapshotTrainingConfigField<bool>(snapshot, "log_recorder_enabled");
    view.default_level = snapshotTrainingConfigField<std::string>(snapshot, "log_recorder_default_level");
    view.modules = snapshotTrainingConfigField<std::map<std::string, std::string>>(snapshot, "log_recorder_modules");
    view.layers.embedding = snapshotTrainingConfigField<bool>(snapshot, "log_recorder_layer_embedding");
    view.layers.rms_norm = snapshotTrainingConfigField<bool>(snapshot, "log_recorder_layer_rms_norm");
    view.layers.attention = snapshotTrainingConfigField<bool>(snapshot, "log_recorder_layer_attention");
    view.layers.feed_forward = snapshotTrainingConfigField<bool>(snapshot, "log_recorder_layer_feed_forward");
    view.layers.residual = snapshotTrainingConfigField<bool>(snapshot, "log_recorder_layer_residual");
    view.layers.encoding = snapshotTrainingConfigField<bool>(snapshot, "log_recorder_layer_encoding");
    view.layers.serialization = snapshotTrainingConfigField<bool>(snapshot, "log_recorder_layer_serialization");
    view.layers.execution_block = snapshotTrainingConfigField<bool>(snapshot, "log_recorder_layer_execution_block");
    return view;
}

inline OptimizerUpdateHP optimizerUpdateHP(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    OptimizerUpdateHP view;
    view.kind = snapshotTrainingConfigField<OptimizerKind>(snapshot, "optimizer_kind");
    view.weight_decay = snapshotTrainingConfigField<float>(snapshot, "weight_decay");
    view.use_depth_aware_upsilon = snapshotTrainingConfigField<bool>(snapshot, "use_depth_aware_upsilon");
    view.beta1 = snapshotTrainingConfigField<float>(snapshot, "optimizer_beta1");
    view.beta2 = snapshotTrainingConfigField<float>(snapshot, "optimizer_beta2");
    view.epsilon = snapshotTrainingConfigField<float>(snapshot, "optimizer_epsilon");
    if (snapshotTrainingConfigField<bool>(snapshot, "embedding_freeze_enabled")) {
        view.embedding_freeze_after_step = snapshotTrainingConfigField<int>(snapshot, "embedding_freeze_after_step");
    } else {
        view.embedding_freeze_after_step = -1;
    }
    return view;
}

inline GradientClippingHP gradientClippingHP(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    const auto configured_clip_norm = snapshotEffectiveGradClipNorm(snapshot);

    GradientClippingHP view;
    view.configured_clip_norm = configured_clip_norm;
    view.enabled = configured_clip_norm > 0.0f;
    view.effective_per_token_limit = std::max(configured_clip_norm, EPSILON_GRADIENT_CLIP);
    return view;
}

inline LossConfigHP lossConfigHP(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    LossConfigHP view;
    view.focal_enabled = snapshotTrainingConfigField<bool>(snapshot, "loss_focal_enabled");
    view.focal_alpha = snapshotTrainingConfigField<float>(snapshot, "loss_focal_alpha");
    view.focal_gamma = snapshotTrainingConfigField<float>(snapshot, "loss_focal_gamma");
    view.smoothing_enabled = snapshotTrainingConfigField<bool>(snapshot, "loss_label_smoothing_enabled");
    view.smoothing_epsilon = snapshotTrainingConfigField<float>(snapshot, "loss_label_smoothing_epsilon");
    view.entropy_reg_enabled = snapshotTrainingConfigField<bool>(snapshot, "loss_entropy_reg_enabled");
    view.entropy_reg_lambda = snapshotTrainingConfigField<float>(snapshot, "loss_entropy_reg_lambda");
    view.class_balanced_enabled = snapshotTrainingConfigField<bool>(snapshot, "loss_class_balanced_enabled");
    view.class_balanced_beta = snapshotTrainingConfigField<float>(snapshot, "loss_class_balanced_beta");
    return view;
}

inline AutoStopHP autoStopHP(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    AutoStopHP view;
    view.enabled = snapshotTrainingConfigField<bool>(snapshot, "auto_stop_enabled");
    view.plateau_patience = snapshotTrainingConfigField<int>(snapshot, "auto_stop_plateau_patience");
    view.plateau_min_delta = snapshotTrainingConfigField<float>(snapshot, "auto_stop_plateau_min_delta");
    view.high_loss_threshold = snapshotTrainingConfigField<float>(snapshot, "auto_stop_high_loss_threshold");
    view.high_loss_patience = snapshotTrainingConfigField<int>(snapshot, "auto_stop_high_loss_patience");
    return view;
}

inline TrainingScheduleHP trainingScheduleHP(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    TrainingScheduleHP view;
    view.epochs = snapshotTrainingConfigField<int>(snapshot, "epochs");
    view.gradient_accumulation_steps = snapshotTrainingConfigField<int>(snapshot, "gradient_accumulation_steps");
    view.accumulation_normalization_scale = (view.gradient_accumulation_steps > 0)
        ? 1.0f / static_cast<float>(view.gradient_accumulation_steps)
        : 1.0f;
    view.single_batch_overfit_enabled = snapshotTrainingConfigField<bool>(snapshot, "single_batch_overfit_enabled");
    view.single_batch_overfit_max_steps = snapshotTrainingConfigField<int>(snapshot, "single_batch_overfit_max_steps");
    view.shuffle_train_enabled = snapshotTrainingConfigField<bool>(snapshot, "shuffle_train_enabled");
    view.shuffle_train_epochs = snapshotTrainingConfigField<int>(snapshot, "shuffle_train_epochs");
    return view;
}

inline SoftRestartHP softRestartHP(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    SoftRestartHP view;
    view.enabled = snapshotTrainingConfigField<bool>(snapshot, "soft_restart_enabled");
    view.loss_increase_threshold = snapshotTrainingConfigField<float>(snapshot, "soft_restart_loss_increase_threshold");
    view.max_step_window = snapshotTrainingConfigField<int>(snapshot, "soft_restart_max_step_window");
    view.cooldown_steps = snapshotTrainingConfigField<int>(snapshot, "soft_restart_cooldown_steps");
    return view;
}

inline TrainingRuntimeControlHP trainingRuntimeControlHP(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    TrainingRuntimeControlHP view;
    view.log_interval = snapshotTrainingConfigField<int>(snapshot, "log_interval");
    view.atom_stats_interval = snapshotTrainingConfigField<int>(snapshot, "atom_stats_interval");
    view.atom_stats_max_seqs = snapshotTrainingConfigField<int>(snapshot, "atom_stats_max_seqs");
    view.validation_interval = snapshotTrainingConfigField<int>(snapshot, "validation_interval");
    view.checkpoint_interval = snapshotTrainingConfigField<int>(snapshot, "checkpoint_interval");
    view.logit_update_trace_enabled = snapshotTrainingConfigField<bool>(snapshot, "logit_update_trace_enabled");
    view.logit_update_trace_interval = snapshotTrainingConfigField<int>(snapshot, "logit_update_trace_interval");
    return view;
}

inline TelemetryControlHP telemetryControlHP(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    TelemetryControlHP view;
    view.enabled = snapshotTrainingConfigField<bool>(snapshot, "telemetry_control_enabled");
    view.spike_mild_threshold = snapshotTrainingConfigField<float>(snapshot, "telemetry_spike_mild_threshold");
    view.spike_moderate_threshold = snapshotTrainingConfigField<float>(snapshot, "telemetry_spike_moderate_threshold");
    view.spike_severe_threshold = snapshotTrainingConfigField<float>(snapshot, "telemetry_spike_severe_threshold");
    view.moderate_grad_scale = snapshotTrainingConfigField<float>(snapshot, "telemetry_moderate_grad_scale");
    view.moderate_cooldown_extension = snapshotTrainingConfigField<int>(snapshot, "telemetry_moderate_cooldown_extension");
    view.min_grad_for_nonzero_loss = snapshotTrainingConfigField<float>(snapshot, "telemetry_min_grad_for_nonzero_loss");
    view.loss_threshold_for_grad_check = snapshotTrainingConfigField<float>(snapshot, "telemetry_loss_threshold_for_grad_check");
    view.max_consecutive_zero_grad_steps = snapshotTrainingConfigField<int>(snapshot, "telemetry_max_consecutive_zero_grad_steps");
    view.seq_len_regime_change_threshold = snapshotTrainingConfigField<float>(snapshot, "telemetry_seq_len_regime_change_threshold");
    view.regime_change_suppression_steps = snapshotTrainingConfigField<int>(snapshot, "telemetry_regime_change_suppression_steps");
    view.volatility_damping_threshold = snapshotTrainingConfigField<float>(snapshot, "telemetry_volatility_damping_threshold");
    view.max_volatility_damping = snapshotTrainingConfigField<float>(snapshot, "telemetry_max_volatility_damping");
    view.gradient_decay_threshold = snapshotTrainingConfigField<float>(snapshot, "telemetry_gradient_decay_threshold");
    view.max_decay_boost = snapshotTrainingConfigField<float>(snapshot, "telemetry_max_decay_boost");
    view.progress_boost_threshold = snapshotTrainingConfigField<float>(snapshot, "telemetry_progress_boost_threshold");
    view.max_progress_boost = snapshotTrainingConfigField<float>(snapshot, "telemetry_max_progress_boost");
    view.outlier_frequency_trigger = snapshotTrainingConfigField<float>(snapshot, "telemetry_outlier_frequency_trigger");
    view.outlier_persistence_trigger = snapshotTrainingConfigField<float>(snapshot, "telemetry_outlier_persistence_trigger");
    view.anchor_drift_sigma_multiplier = snapshotTrainingConfigField<float>(snapshot, "telemetry_anchor_drift_sigma_multiplier");
    view.soft_restart_cooldown_steps = snapshotTrainingConfigField<int>(snapshot, "telemetry_soft_restart_cooldown_steps");
    view.warmup_steps = 0;
    view.baseline_stabilization_steps = snapshotTrainingConfigField<int>(snapshot, "telemetry_baseline_stabilization_steps");
    view.verbose_logging = snapshotTrainingConfigField<bool>(snapshot, "telemetry_verbose_logging");
    view.fail_loud_on_accumulation_bug = snapshotTrainingConfigField<bool>(snapshot, "telemetry_fail_loud_on_accumulation_bug");
    view.plateau_noise_enabled = snapshotTrainingConfigField<bool>(snapshot, "telemetry_plateau_noise_enabled");
    view.plateau_noise_patience = snapshotTrainingConfigField<int>(snapshot, "telemetry_plateau_noise_patience");
    view.plateau_noise_variance_threshold = snapshotTrainingConfigField<float>(snapshot, "telemetry_plateau_noise_variance_threshold");
    view.plateau_noise_std = snapshotTrainingConfigField<float>(snapshot, "telemetry_plateau_noise_std");
    view.plateau_noise_proportional = snapshotTrainingConfigField<bool>(snapshot, "telemetry_plateau_noise_proportional");
    view.plateau_noise_cooldown = snapshotTrainingConfigField<int>(snapshot, "telemetry_plateau_noise_cooldown");
    view.plateau_noise_max_per_epoch = snapshotTrainingConfigField<int>(snapshot, "telemetry_plateau_noise_max_per_epoch");
    return view;
}

inline TelemetryLatticeHP telemetryLatticeHP(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    TelemetryLatticeHP view;
    view.num_levels = snapshotTrainingConfigField<int>(snapshot, "telemetry_lattice_num_levels");
    view.num_streams = snapshotTrainingConfigField<int>(snapshot, "telemetry_lattice_num_streams");
    view.beta_mu = snapshotTrainingConfigField<float>(snapshot, "telemetry_lattice_beta_mu");
    view.beta_a = snapshotTrainingConfigField<float>(snapshot, "telemetry_lattice_beta_a");
    view.beta_delta = snapshotTrainingConfigField<float>(snapshot, "telemetry_lattice_beta_delta");
    view.beta_r = snapshotTrainingConfigField<float>(snapshot, "telemetry_lattice_beta_r");
    view.beta_run = snapshotTrainingConfigField<float>(snapshot, "telemetry_lattice_beta_run");
    view.beta_v = snapshotTrainingConfigField<float>(snapshot, "telemetry_lattice_beta_v");
    view.k_out0 = snapshotTrainingConfigField<float>(snapshot, "telemetry_lattice_k_out0");
    view.alpha_v = snapshotTrainingConfigField<float>(snapshot, "telemetry_lattice_alpha_v");
    view.epsilon = snapshotTrainingConfigField<float>(snapshot, "telemetry_lattice_epsilon");
    view.strict_mode = snapshotTrainingConfigField<bool>(snapshot, "telemetry_lattice_strict_mode");
    return view;
}

inline StabilityOverrideHP stabilityOverrideHP(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    StabilityOverrideHP view;
    view.enabled = snapshotTrainingConfigField<bool>(snapshot, "stability_overrides_enabled");
    view.batch_size = snapshotTrainingConfigField<int>(snapshot, "stability_override_batch_size");
    view.max_seq_len = snapshotTrainingConfigField<int>(snapshot, "stability_override_max_seq_len");
    view.clip_per_token = snapshotTrainingConfigField<float>(snapshot, "stability_override_clip_per_token");
    return view;
}

inline ModelHP modelHP(const GRIM::Config::AiConfigSnapshot& snapshot)
{
    const auto& cfg = snapshotTrainingConfig(snapshot);
    const auto requireInt = [&](const char* name) -> int {
        return cfg.at(name).get<int>();
    };
    const auto requireFloat = [&](const char* name) -> float {
        return cfg.at(name).get<float>();
    };
    const auto requireBool = [&](const char* name) -> bool {
        return cfg.at(name).get<bool>();
    };

    const int batch_size = snapshotEffectiveBatchSize(snapshot, "modelHP(snapshot)");
    const int max_seq_len = snapshotEffectiveMaxSeqLen(snapshot, "modelHP(snapshot)");
    const int d_model = requireInt("d_model");
    const int num_layers = requireInt("num_layers");
    const int num_heads = requireInt("num_heads");
    const int num_kv_heads = requireInt("num_kv_heads");

    if (d_model <= 0 || num_layers <= 0 || num_heads <= 0 || num_kv_heads <= 0) {
        throw std::runtime_error(
            "modelHP(snapshot): d_model, num_layers, num_heads, and num_kv_heads must all be > 0");
    }
    if ((d_model % num_heads) != 0) {
        throw std::runtime_error(
            "modelHP(snapshot): d_model must be divisible by num_heads (d_model=" +
            std::to_string(d_model) + ", num_heads=" + std::to_string(num_heads) + ")");
    }
    if (!isValidGQAConfig(num_heads, num_kv_heads)) {
        throw std::runtime_error(
            "modelHP(snapshot): invalid GQA config (num_heads=" +
            std::to_string(num_heads) + ", num_kv_heads=" + std::to_string(num_kv_heads) + ")");
    }

    const int head_dim = d_model / num_heads;
    const int heads_per_kv_group = computeHeadsPerKVGroup(num_heads, num_kv_heads);
    const int kv_dim = computeKVProjectionSize(d_model, num_heads, num_kv_heads);
    const int qkv_dim = computeQKVProjectionSize(d_model, num_heads, num_kv_heads);
    const int rotary_dim = head_dim;
    const float attention_softmax_scale = computeAttentionSoftmaxScale(
        head_dim, "modelHP(snapshot)");
    const int d_ff = d_model * D_FF_MULTIPLIER;
    const int min_seq_len_for_flash = max_seq_len / 4;
    const float dropout_rate = requireFloat("dropout_rate");
    const float attention_dropout = dropout_rate;
    const float residual_projection_init_gain =
        1.0f / std::sqrt(2.0f * static_cast<float>(num_layers));
    const bool is_gqa = num_kv_heads < num_heads;
    const int vocab_size = snapshotTrainingConfigField<int>(snapshot, "vocab_size");

    ModelHP view;
    view.training_batch_size = batch_size;
    view.training_max_seq_len = max_seq_len;
    view.training_max_tokens_per_batch =
        computeMaxTokensPerBatch(batch_size, max_seq_len, "modelHP(snapshot)");

    view.embedding_vocab_size = vocab_size;
    view.embedding_d_model = d_model;
    view.embedding_scale = requireFloat("embedding_scale");

    view.encoder_num_layers = num_layers;
    view.encoder_d_model = d_model;
    view.encoder_num_heads = num_heads;
    view.encoder_num_kv_heads = num_kv_heads;
    view.encoder_head_dim = head_dim;
    view.encoder_rotary_dim = rotary_dim;
    view.encoder_attention_softmax_scale = attention_softmax_scale;
    view.encoder_heads_per_kv_group = heads_per_kv_group;
    view.encoder_kv_dim = kv_dim;
    view.encoder_qkv_dim = qkv_dim;
    view.encoder_d_ff = d_ff;
    view.encoder_rms_epsilon = EPSILON_RMSNORM;
    view.encoder_causal_mask = true;
    view.encoder_use_flash_attention = requireBool("use_flash_attention");
    view.encoder_min_seq_len_for_flash = min_seq_len_for_flash;
    view.encoder_use_layer_scale = requireBool("use_layer_scale");
    view.encoder_layer_scale_init = requireFloat("layer_scale_init");
    view.encoder_center_encoder_residuals = requireBool("center_encoder_residuals");
    view.encoder_use_bias = requireBool("use_bias");
    view.encoder_dropout_rate = dropout_rate;
    view.encoder_attention_dropout = attention_dropout;
    view.encoder_qk_norm_enabled = requireBool("qk_norm_enabled");
    view.encoder_attention_off_by_one = requireBool("attention_off_by_one");
    view.encoder_residual_projection_init_gain = residual_projection_init_gain;
    view.encoder_is_gqa = is_gqa;
    view.encoder_freeze_learned_rms_gammas = requireBool("freeze_learned_rms_gammas");

    view.lm_head_d_model = d_model;
    view.lm_head_vocab_size = vocab_size;
    view.lm_head_training_batch_size = batch_size;
    view.lm_head_training_rows_per_sequence = max_seq_len;
    view.lm_head_use_bias = requireBool("use_bias");
    view.lm_head_unigram_bias = requireBool("lm_head_unigram_bias");
    view.lm_head_tie_embeddings = requireBool("tie_embeddings");
    view.lm_head_center_hidden_states = requireBool("lm_head_center_hidden_states");
    view.lm_head_project_out_pc1 = requireBool("project_out_pc1");
    view.lm_head_pc1_power_iters = requireInt("pc1_power_iters");
    view.lm_head_center_logits = requireBool("center_logits");
    view.lm_head_freeze_learned_rms_gammas = requireBool("freeze_learned_rms_gammas");
    view.lm_head_rms_epsilon = EPSILON_RMSNORM;
    view.lm_head_mlp_enabled = requireBool("lm_head_mlp_enabled");
    view.lm_head_mlp_d_ff = requireInt("lm_head_mlp_d_ff");
    view.lm_head_mlp_alpha = requireFloat("lm_head_mlp_alpha");

    view.atom_embedding_dim = requireInt("atom_embedding_dim");

    view.execution_block_enabled = requireBool("execution_block_enabled");
    view.execution_block_layer = requireInt("execution_block_layer");
    view.execution_block_d_model = d_model;
    view.execution_block_num_ops = requireInt("execution_block_num_ops");
    view.execution_block_num_slots = requireInt("execution_block_num_slots");
    view.execution_block_num_scratch_slots = requireInt("execution_block_num_scratch_slots");
    view.execution_block_num_exec_steps = requireInt("execution_block_num_steps");
    view.execution_block_value_decode_input_dim = requireInt("execution_block_value_decode_input_dim");
    view.execution_block_value_decode_hidden_dim = requireInt("execution_block_value_decode_hidden_dim");
    view.execution_block_d_key = head_dim;
    view.execution_block_d_type = requireInt("execution_block_d_type");
    view.execution_block_cross_attn_head_dim = head_dim;
    view.execution_block_cross_attn_topk = requireInt("execution_block_cross_attn_topk");
    view.execution_block_usage_decay = requireFloat("execution_block_usage_decay");
    view.execution_block_inject_gate_temp = requireFloat("execution_block_inject_gate_temp");
    view.execution_block_result_slot_mode = requireInt("execution_block_result_slot_mode");
    view.execution_block_result_slot_index = requireInt("execution_block_result_slot_index");
    view.execution_block_debug_mode = requireBool("execution_block_debug_mode");
    view.execution_block_entropy_collapse_threshold = requireFloat("execution_block_entropy_collapse_threshold");
    view.execution_block_write_collapse_threshold = requireFloat("execution_block_write_collapse_threshold");
    view.execution_block_magnitude_limit = requireFloat("execution_block_magnitude_limit");
    view.execution_block_diversity_kappa = requireFloat("execution_block_diversity_kappa");
    view.execution_block_temp_start = requireFloat("execution_block_temp_start");
    view.execution_block_temp_end = requireFloat("execution_block_temp_end");
    view.execution_block_temp_schedule = requireInt("execution_block_temp_schedule");
    view.execution_block_entropy_weight = requireFloat("execution_block_entropy_weight");
    view.execution_block_transition_hard_threshold = requireFloat("execution_block_transition_hard_threshold");
    view.execution_block_gate_warmup_steps = 0;
    view.execution_block_causal_w1_transition = requireFloat("execution_block_causal_w1_transition");
    view.execution_block_div_invalid_penalty_weight = requireFloat("execution_block_div_invalid_penalty_weight");
    view.execution_block_div_magnitude_penalty_weight = requireFloat("execution_block_div_magnitude_penalty_weight");
    view.execution_block_arg_reinforce_weight = requireFloat("execution_block_arg_reinforce_weight");
    view.execution_block_arg_reinforce_baseline_decay = requireFloat("execution_block_arg_reinforce_baseline_decay");
    view.execution_block_entropy_aux_weight = requireFloat("execution_block_entropy_aux_weight");
    view.execution_block_structured_ce_weight = requireFloat("execution_block_structured_ce_weight");

    view.mtp_enabled = requireBool("mtp_enabled");
    view.mtp_k = requireInt("mtp_k");
    view.mtp_vocab_size = vocab_size;
    view.mtp_d_model = d_model;
    view.mtp_alpha = requireFloat("mtp_alpha");

    view.latent_trajectory_preset_enabled = requireBool("latent_trajectory_preset_enabled");
    view.latent_trajectory_preset_d_model = d_model;
    view.latent_trajectory_preset_mtp_k = view.mtp_k;
    view.latent_trajectory_preset_vocab_size = vocab_size;
    view.latent_trajectory_preset_fuse_dim = view.latent_trajectory_preset_enabled
        ? computeLatentTrajectoryPresetFuseDim(d_model, "modelHP(snapshot)")
        : 0;
    view.latent_trajectory_preset_dim = view.latent_trajectory_preset_enabled
        ? computeLatentTrajectoryPresetDim(d_model, "modelHP(snapshot)")
        : 0;
    view.latent_trajectory_preset_gate_dim = view.latent_trajectory_preset_enabled
        ? computeLatentTrajectoryPresetGateDim(d_model, "modelHP(snapshot)")
        : 0;
    view.latent_trajectory_preset_scale = requireFloat("latent_trajectory_preset_scale");
    view.latent_trajectory_preset_gate_bias_init = requireFloat("latent_trajectory_preset_gate_bias_init");
    view.latent_trajectory_preset_use_token_mean = requireBool("latent_trajectory_preset_use_token_mean");
    view.latent_trajectory_preset_use_mtp_logits = requireBool("latent_trajectory_preset_use_mtp_logits");
    view.latent_trajectory_preset_use_mtp_hidden = requireBool("latent_trajectory_preset_use_mtp_hidden");
    view.latent_trajectory_preset_use_entropy = requireBool("latent_trajectory_preset_use_entropy");
    view.latent_trajectory_preset_use_delta_target = requireBool("latent_trajectory_preset_use_delta_target");
    view.latent_trajectory_preset_use_consistency_loss = requireBool("latent_trajectory_preset_use_consistency_loss");
    view.latent_trajectory_preset_use_diversity_loss = requireBool("latent_trajectory_preset_use_diversity_loss");
    view.latent_trajectory_preset_use_gate_sparsity_loss = requireBool("latent_trajectory_preset_use_gate_sparsity_loss");
    view.latent_trajectory_preset_lambda_traj = requireFloat("latent_trajectory_preset_lambda_traj");
    view.latent_trajectory_preset_lambda_delta = requireFloat("latent_trajectory_preset_lambda_delta");
    view.latent_trajectory_preset_lambda_consistency = requireFloat("latent_trajectory_preset_lambda_consistency");
    view.latent_trajectory_preset_lambda_diversity = requireFloat("latent_trajectory_preset_lambda_diversity");
    view.latent_trajectory_preset_lambda_gate = requireFloat("latent_trajectory_preset_lambda_gate");

    view.number_encoder_enabled = requireBool("number_encoder_enabled");
    view.number_encoder_d_model = d_model;
    view.number_encoder_d_hidden = requireInt("number_encoder_d_hidden");
    view.number_encoder_max_digit_slots = requireInt("number_encoder_max_digit_slots");
    view.number_encoder_max_abs_pow10 = requireInt("number_encoder_max_abs_pow10");

    view.positional_encoding = parsePositionalEncodingFlags(
        requireBool("use_rope"), requireBool("use_alibi"));
    view.structured_ce_enabled = requireBool("execution_block_structured_ce_enabled");
    return view;
}

inline GpuModelInitializationHP gpuModelInitializationHP(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    const auto model = modelHP(snapshot);

    GpuModelInitializationHP view;
    view.use_gpu = snapshotTrainingConfigField<bool>(snapshot, "use_gpu");
    view.num_layers = model.encoder_num_layers;
    view.use_flash_attention = model.encoder_use_flash_attention;
    view.min_seq_len_for_flash = model.encoder_min_seq_len_for_flash;
    return view;
}

inline EncoderLayerConstructionHP encoderLayerConstructionHP(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    const auto model = modelHP(snapshot);

    EncoderLayerConstructionHP view;
    view.num_layers = model.encoder_num_layers;
    view.d_model = model.encoder_d_model;
    view.num_heads = model.encoder_num_heads;
    view.num_kv_heads = model.encoder_num_kv_heads;
    view.head_dim = model.encoder_head_dim;
    view.rotary_dim = model.encoder_rotary_dim;
    view.attention_softmax_scale = model.encoder_attention_softmax_scale;
    view.heads_per_kv_group = model.encoder_heads_per_kv_group;
    view.kv_dim = model.encoder_kv_dim;
    view.qkv_dim = model.encoder_qkv_dim;
    view.d_ff = model.encoder_d_ff;
    view.rms_epsilon = model.encoder_rms_epsilon;
    view.causal_mask = model.encoder_causal_mask;
    view.use_flash_attention = model.encoder_use_flash_attention;
    view.min_seq_len_for_flash = model.encoder_min_seq_len_for_flash;
    view.use_layer_scale = model.encoder_use_layer_scale;
    view.layer_scale_init = model.encoder_layer_scale_init;
    view.center_encoder_residuals = model.encoder_center_encoder_residuals;
    view.use_bias = model.encoder_use_bias;
    view.dropout_rate = model.encoder_dropout_rate;
    view.attention_dropout = model.encoder_attention_dropout;
    view.qk_norm_enabled = model.encoder_qk_norm_enabled;
    view.attention_off_by_one = model.encoder_attention_off_by_one;
    view.residual_projection_init_gain = model.encoder_residual_projection_init_gain;
    view.is_gqa = model.encoder_is_gqa;
    view.freeze_learned_rms_gammas = model.encoder_freeze_learned_rms_gammas;
    return view;
}

inline EmbeddingLayerConstructionHP embeddingLayerConstructionHP(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    const auto model = modelHP(snapshot);

    EmbeddingLayerConstructionHP view;
    view.vocab_size = model.embedding_vocab_size;
    view.d_model = model.embedding_d_model;
    view.embedding_scale = model.embedding_scale;
    return view;
}

inline LMHeadLayerConstructionHP lmHeadLayerConstructionHP(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    const auto model = modelHP(snapshot);

    LMHeadLayerConstructionHP view;
    view.d_model = model.lm_head_d_model;
    view.vocab_size = model.lm_head_vocab_size;
    view.training_batch_size = model.lm_head_training_batch_size;
    view.training_rows_per_sequence = model.lm_head_training_rows_per_sequence;
    view.use_bias = model.lm_head_use_bias;
    view.unigram_bias = model.lm_head_unigram_bias;
    view.tie_embeddings = model.lm_head_tie_embeddings;
    view.center_hidden_states = model.lm_head_center_hidden_states;
    view.project_out_pc1 = model.lm_head_project_out_pc1;
    view.pc1_power_iters = model.lm_head_pc1_power_iters;
    view.center_logits = model.lm_head_center_logits;
    view.freeze_learned_rms_gammas = model.lm_head_freeze_learned_rms_gammas;
    view.rms_epsilon = model.lm_head_rms_epsilon;
    view.mlp_enabled = model.lm_head_mlp_enabled;
    view.mlp_d_ff = model.lm_head_mlp_d_ff;
    view.mlp_alpha = model.lm_head_mlp_alpha;
    return view;
}

inline ExecutionBlockConstructionHP executionBlockConstructionHP(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    const auto model = modelHP(snapshot);

    ExecutionBlockConstructionHP view;
    view.enabled = model.execution_block_enabled;
    view.layer = model.execution_block_layer;
    view.d_model = model.execution_block_d_model;
    view.atom_embedding_dim = model.atom_embedding_dim;
    view.num_ops = model.execution_block_num_ops;
    view.num_slots = model.execution_block_num_slots;
    view.num_scratch_slots = model.execution_block_num_scratch_slots;
    view.num_exec_steps = model.execution_block_num_exec_steps;
    view.value_decode_input_dim = model.execution_block_value_decode_input_dim;
    view.value_decode_hidden_dim = model.execution_block_value_decode_hidden_dim;
    view.d_key = model.execution_block_d_key;
    view.d_type = model.execution_block_d_type;
    view.cross_attn_head_dim = model.execution_block_cross_attn_head_dim;
    view.cross_attn_topk = model.execution_block_cross_attn_topk;
    view.usage_decay = model.execution_block_usage_decay;
    view.inject_gate_temp = model.execution_block_inject_gate_temp;
    view.result_slot_mode = model.execution_block_result_slot_mode;
    view.result_slot_index = model.execution_block_result_slot_index;
    view.debug_mode = model.execution_block_debug_mode;
    view.entropy_collapse_threshold = model.execution_block_entropy_collapse_threshold;
    view.write_collapse_threshold = model.execution_block_write_collapse_threshold;
    view.magnitude_limit = model.execution_block_magnitude_limit;
    view.diversity_kappa = model.execution_block_diversity_kappa;
    view.temp_start = model.execution_block_temp_start;
    view.temp_end = model.execution_block_temp_end;
    view.temp_schedule = model.execution_block_temp_schedule;
    view.entropy_weight = model.execution_block_entropy_weight;
    view.transition_hard_threshold = model.execution_block_transition_hard_threshold;
    view.gate_warmup_steps = model.execution_block_gate_warmup_steps;
    view.causal_w1_transition = model.execution_block_causal_w1_transition;
    view.div_invalid_penalty_weight = model.execution_block_div_invalid_penalty_weight;
    view.div_magnitude_penalty_weight = model.execution_block_div_magnitude_penalty_weight;
    view.arg_reinforce_weight = model.execution_block_arg_reinforce_weight;
    view.arg_reinforce_baseline_decay = model.execution_block_arg_reinforce_baseline_decay;
    return view;
}

inline MTPConstructionHP mtpConstructionHP(const GRIM::Config::AiConfigSnapshot& snapshot)
{
    const auto model = modelHP(snapshot);

    MTPConstructionHP view;
    view.enabled = model.mtp_enabled;
    view.k = model.mtp_k;
    view.vocab_size = model.mtp_vocab_size;
    view.d_model = model.mtp_d_model;
    view.alpha = model.mtp_alpha;
    return view;
}

inline NumberEncoderConstructionHP numberEncoderConstructionHP(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    const auto model = modelHP(snapshot);

    NumberEncoderConstructionHP view;
    view.enabled = model.number_encoder_enabled;
    view.d_model = model.number_encoder_d_model;
    view.d_hidden = model.number_encoder_d_hidden;
    view.max_digit_slots = model.number_encoder_max_digit_slots;
    view.max_abs_pow10 = model.number_encoder_max_abs_pow10;
    view.pow10_buckets = 2 * model.number_encoder_max_abs_pow10 + 1;
    view.use_bias = model.encoder_use_bias;
    return view;
}

inline MTPFeatureHP mtpFeatureHP(const GRIM::Config::AiConfigSnapshot& snapshot)
{
    const auto model = modelHP(snapshot);

    MTPFeatureHP view;
    view.enabled = model.mtp_enabled;
    view.k = model.mtp_k;
    return view;
}

inline LatentTrajectoryPresetHP latentTrajectoryPresetHP(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    const auto model = modelHP(snapshot);

    LatentTrajectoryPresetHP view;
    view.enabled = model.latent_trajectory_preset_enabled;
    view.d_model = model.latent_trajectory_preset_d_model;
    view.mtp_k = model.latent_trajectory_preset_mtp_k;
    view.vocab_size = model.latent_trajectory_preset_vocab_size;
    view.fuse_dim = model.latent_trajectory_preset_fuse_dim;
    view.preset_dim = model.latent_trajectory_preset_dim;
    view.gate_dim = model.latent_trajectory_preset_gate_dim;
    view.preset_scale = model.latent_trajectory_preset_scale;
    view.gate_bias_init = model.latent_trajectory_preset_gate_bias_init;
    view.use_token_mean = model.latent_trajectory_preset_use_token_mean;
    view.use_mtp_logits = model.latent_trajectory_preset_use_mtp_logits;
    view.use_mtp_hidden = model.latent_trajectory_preset_use_mtp_hidden;
    view.use_entropy = model.latent_trajectory_preset_use_entropy;
    view.use_delta_target = model.latent_trajectory_preset_use_delta_target;
    view.use_consistency_loss = model.latent_trajectory_preset_use_consistency_loss;
    view.use_diversity_loss = model.latent_trajectory_preset_use_diversity_loss;
    view.use_gate_sparsity_loss = model.latent_trajectory_preset_use_gate_sparsity_loss;
    view.lambda_traj = model.latent_trajectory_preset_lambda_traj;
    view.lambda_delta = model.latent_trajectory_preset_lambda_delta;
    view.lambda_consistency = model.latent_trajectory_preset_lambda_consistency;
    view.lambda_diversity = model.latent_trajectory_preset_lambda_diversity;
    view.lambda_gate = model.latent_trajectory_preset_lambda_gate;
    return view;
}

inline MTPDiagnosticHP mtpDiagnosticHP(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    MTPDiagnosticHP view;
    view.log_ratio_monitor = snapshotTrainingConfigField<bool>(snapshot, "mtp_log_ratio_monitor");
    return view;
}

// True when MTP supervision consumes the latent-trajectory predicted future
// hidden states (latent_preset_mtp_hidden slice k) projected through the
// SHARED LM head, instead of standalone per-horizon MTP head parameters.
// In this mode the standalone MTP heads are never assembled, registered,
// initialized, or serialized — the MTP CE gradient flows through
// W_hidden_traj into the trunk, coupling the trajectory objective to
// token-space future prediction.
inline bool mtpUsesLatentTrajectoryLogits(const GRIM::Config::AiConfigSnapshot& snapshot)
{
    const auto mtp = mtpFeatureHP(snapshot);
    const auto latent = latentTrajectoryPresetHP(snapshot);
    return mtp.enabled && mtp.k > 0 && latent.enabled && latent.use_mtp_logits;
}

inline GenerationHP generationHP(const LanguageModelConfig& cfg)
{
    GenerationHP view;
    view.strategy = cfg.generation_strategy;
    view.max_new_tokens = cfg.generation_max_new_tokens;
    view.min_new_tokens = cfg.generation_min_new_tokens;
    view.temperature = cfg.generation_temperature;
    view.top_k = cfg.generation_top_k;
    view.top_p = cfg.generation_top_p;
    view.min_p = cfg.generation_min_p;
    view.typical_p = cfg.generation_typical_p;
    view.repetition_penalty = cfg.generation_repetition_penalty;
    view.repetition_penalty_window = cfg.generation_repetition_penalty_window;
    view.frequency_penalty = cfg.generation_frequency_penalty;
    view.presence_penalty = cfg.generation_presence_penalty;
    view.num_return_sequences = cfg.generation_num_return_sequences;
    view.eos_token_id = cfg.generation_eos_token_id;
    view.pad_token_id = cfg.generation_pad_token_id;
    view.bos_token_id = cfg.generation_bos_token_id;
    view.unk_token_id = cfg.generation_unk_token_id;
    view.no_repeat_ngram_size = cfg.generation_no_repeat_ngram_size;
    view.do_sample = cfg.generation_do_sample;
    view.bad_words_ids = cfg.generation_bad_words_ids;
    view.masked_numeric_literal_ids = cfg.generation_masked_numeric_literal_ids;
    view.seed = cfg.generation_seed;
    view.enable_scratchblock_reasoning = cfg.generation_enable_scratchblock_reasoning;
    return view;
}

inline GenerationHP generationHP(const GRIM::Config::AiConfigSnapshot& snapshot)
{
    GenerationHP view;
    view.strategy = snapshotTrainingConfigField<SamplingStrategy>(snapshot, "generation_strategy");
    view.max_new_tokens = snapshotTrainingConfigField<int>(snapshot, "generation_max_new_tokens");
    view.min_new_tokens = snapshotTrainingConfigField<int>(snapshot, "generation_min_new_tokens");
    view.temperature = snapshotTrainingConfigField<float>(snapshot, "generation_temperature");
    view.top_k = snapshotTrainingConfigField<int>(snapshot, "generation_top_k");
    view.top_p = snapshotTrainingConfigField<float>(snapshot, "generation_top_p");
    view.min_p = snapshotTrainingConfigField<float>(snapshot, "generation_min_p");
    view.typical_p = snapshotTrainingConfigField<float>(snapshot, "generation_typical_p");
    view.repetition_penalty = snapshotTrainingConfigField<float>(snapshot, "generation_repetition_penalty");
    view.repetition_penalty_window = snapshotTrainingConfigField<int>(snapshot, "generation_repetition_penalty_window");
    view.frequency_penalty = snapshotTrainingConfigField<float>(snapshot, "generation_frequency_penalty");
    view.presence_penalty = snapshotTrainingConfigField<float>(snapshot, "generation_presence_penalty");
    view.num_return_sequences = 1;
    view.eos_token_id = Tokenizer::EOS_TOKEN_ID;
    view.pad_token_id = Tokenizer::PAD_TOKEN_ID;
    view.bos_token_id = Tokenizer::BOS_TOKEN_ID;
    view.unk_token_id = Tokenizer::UNK_TOKEN_ID;
    view.no_repeat_ngram_size = snapshotTrainingConfigField<int>(snapshot, "generation_no_repeat_ngram_size");
    view.do_sample = snapshotTrainingConfigField<bool>(snapshot, "generation_do_sample");
    view.masked_numeric_literal_ids = snapshotMaskedNumericLiteralIds();
    view.seed = 0;
    view.enable_scratchblock_reasoning =
        snapshotTrainingConfigField<bool>(snapshot, "generation_enable_scratchblock_reasoning");
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

