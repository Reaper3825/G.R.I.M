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

struct MTPDiagnosticHP {
    bool log_ratio_monitor = false;
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

inline CheckpointLoadHP checkpointLoadHP(
    const LanguageModelConfig& config,
    const std::string& checkpoint_path,
    ModelExecutionMode execution_mode)
{
    CheckpointLoadHP view;
    view.checkpoint_dir = config.checkpoint_dir;
    view.checkpoint_path = checkpoint_path;
    view.execution_mode = execution_mode;
    return view;
}

inline TokenizerHP tokenizerHP(const LanguageModelConfig& config) {
    TokenizerHP view;
    const auto paths = pathsHP(config);
    view.data_path = paths.data_path;
    view.vocab_path = paths.vocab_path;
    view.target_vocab_size = config.tokenizer_target_vocab_size;
    view.character_coverage = config.tokenizer_character_coverage;
    view.min_cleaned_text_length = config.tokenizer_min_cleaned_text_length;
    view.min_subword_freq = config.tokenizer_min_subword_freq;
    view.prune_during_mining = config.tokenizer_prune_during_mining;
    view.enable_parallel_subword_mining = config.tokenizer_enable_parallel_subword_mining;
    view.subword_mining_workers = config.tokenizer_subword_mining_workers;
    view.subword_mining_max_bytes = config.tokenizer_subword_mining_max_bytes;
    view.enable_scratch_block_reasoning = config.tokenizer_enable_scratch_block_reasoning;
    view.detect_numbers = config.tokenizer_detect_numbers;
    view.enable_byte_fallback = config.tokenizer_enable_byte_fallback;
    view.add_bos = config.tokenizer_add_bos;
    view.add_eos = config.tokenizer_add_eos;
    view.force_rebuild_vocab = config.force_rebuild_vocab;
    view.save_text_vocab = config.tokenizer_save_text_vocab;
    view.vocab_score_multiplier = config.tokenizer_vocab_score_multiplier;
    view.current_curriculum = config.current_curriculum;
    view.current_model_training = config.current_model_training;
    view.execution_block_num_steps = config.execution_block_num_steps;
    return view;
}

inline TokenizerSubprocessHP tokenizerSubprocessHP(const LanguageModelConfig& config)
{
    TokenizerSubprocessHP view;
    view.tokenizer = tokenizerHP(config);
    view.only_mode = config.subprocess_tokenizer_only_mode;
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

inline GpuModelInitializationHP gpuModelInitializationHP(
    const LanguageModelConfig& cfg)
{
    GpuModelInitializationHP view;
    view.use_gpu = cfg.use_gpu;
    view.num_layers = cfg.num_layers;
    view.use_flash_attention = cfg.use_flash_attention;
    view.min_seq_len_for_flash = cfg.min_seq_len_for_flash;
    return view;
}

inline PBMConstructionHP pbmConstructionHP(
    const LanguageModelConfig& cfg)
{
    PBMConstructionHP view;
    view.num_heads = cfg.num_heads;
    view.num_kv_heads = cfg.num_kv_heads;
    view.head_dim = cfg.head_dim;
    view.rotary_dim = cfg.rotary_dim;
    view.max_seq_len = cfg.max_seq_len;
    view.rope_base_seq_len = cfg.rope_base_seq_len;
    view.alibi_min_locality_distance = cfg.alibi_min_locality_distance;
    view.alibi_slope_exponent = cfg.alibi_slope_exponent;
    view.alibi_max_bias = cfg.alibi_max_bias;
    view.rope_theta = cfg.rope_theta;
    view.rope_scaling = cfg.rope_scaling;
    return view;
}

inline EncoderLayerConstructionHP encoderLayerConstructionHP(
    const LanguageModelConfig& cfg)
{
    EncoderLayerConstructionHP view;
    view.num_layers = cfg.num_layers;
    view.d_model = cfg.d_model;
    view.num_heads = cfg.num_heads;
    view.num_kv_heads = cfg.num_kv_heads;
    view.head_dim = cfg.head_dim;
    view.heads_per_kv_group = cfg.heads_per_kv_group;
    view.kv_dim = cfg.kv_dim;
    view.qkv_dim = cfg.qkv_dim;
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
    view.residual_projection_init_gain = cfg.residual_projection_init_gain;
    view.is_gqa = cfg.is_gqa;
    view.freeze_learned_rms_gammas = cfg.freeze_learned_rms_gammas;
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
    const EncoderLayerConstructionHP& encoder_hp)
{
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
    return view;
}

inline FlashAttentionRuntimeHP flashAttentionRuntimeHP(
    const EncoderSelfAttentionHP& attention_hp)
{
    FlashAttentionRuntimeHP view;
    view.num_heads = attention_hp.num_heads;
    view.num_kv_heads = attention_hp.num_kv_heads;
    view.head_dim = attention_hp.head_dim;
    view.causal = attention_hp.causal_mask;
    view.is_bf16 = true;
    view.requires_alibi = true;
    return view;
}

inline EmbeddingLayerConstructionHP embeddingLayerConstructionHP(
    const LanguageModelConfig& cfg)
{
    EmbeddingLayerConstructionHP view;
    view.vocab_size = cfg.vocab_size;
    view.d_model = cfg.d_model;
    return view;
}

inline LMHeadLayerConstructionHP lmHeadLayerConstructionHP(
    const LanguageModelConfig& cfg)
{
    LMHeadLayerConstructionHP view;
    view.d_model = cfg.d_model;
    view.vocab_size = cfg.vocab_size;
    if (cfg.execution_mode == ModelExecutionMode::TRAINING) {
        view.training_batch_size = cfg.batch_size;
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
    return view;
}

inline MTPFeatureHP mtpFeatureHP(const LanguageModelConfig& cfg)
{
    MTPFeatureHP view;
    view.enabled = cfg.mtp_enabled;
    view.k = cfg.mtp_k;
    return view;
}

inline MTPDiagnosticHP mtpDiagnosticHP(
    const LanguageModelConfig& hp)
{
    MTPDiagnosticHP view;
    view.log_ratio_monitor = hp.mtp_log_ratio_monitor;
    return view;
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

