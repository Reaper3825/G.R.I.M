#pragma once

//======================================================//
// AI CONFIG ORGANIZATION
//
// This file defines the C++ structs that parse ai_config.json.
// The JSON→C++ mapping is:
//
// JSON Section                → C++ Owner
// ---------------------------------------
// paths.grim_text            → AiConfigSnapshot grim_text_* fields
// training.config            → AiConfigSnapshot flat training_* / authored policy leaves
// training.config current_* leaves → AiConfigSnapshot current_* fields
// training.config tokenizer_* leaves → AiConfigSnapshot tokenizer_* fields
// training.config.subprocess_tokenizer_only_mode → AiConfigSnapshot subprocess_tokenizer_only_mode
// training.config.clear_merged_cache_on_merge → AiConfigSnapshot clear_merged_cache_on_merge
//
// RULE: All runtime defaults in TrainingHyperparameters MUST
// be authored in ai_config.json flat snapshot leaves or derived in HyperParameters_GPU.hpp.
// HyperParameters_GPU.hpp may keep only formulas/static kernel capabilities,
// never runtime policy fallbacks.
//
// For compile-time constants (CUDA blocks, epsilons, etc.),
// see HyperParameters_GPU.hpp - DO NOT duplicate them here.
//======================================================//

// Include guard macro for detection by other headers
#define GRIM_CONFIG_AI_CONFIG_PATHS_HPP_INCLUDED

#include <string>
#include <array>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string_view>
#include <type_traits>
#include <map>
#include <vector>

// Strict layering: this file is the JSON reader and is allowed to be
// included by EXACTLY ONE place — HyperParameters_GPU.hpp. AiConfigSnapshot
// stays raw/flat; downstream typed config construction lives in HP_GPU.hpp.
// Including this header from anywhere else is a layering violation; HP_GPU.hpp
// is the single entry point.
#ifndef GRIM_HP_GPU_DEFINED_TRAINING_STRUCTS
# error "ai_config_paths.hpp must only be included via HyperParameters_GPU.hpp. Include HyperParameters_GPU.hpp instead."
#endif

#ifdef _WIN32
    #include <windows.h>  // GetModuleFileNameW, etc.
#elif defined(__APPLE__)
    #include <mach-o/dyld.h>
#else
    #include <unistd.h>
    #include <limits.h>
#endif

namespace GRIM {
namespace Config {

inline const nlohmann::json& requireJsonField(const nlohmann::json& node,
                                             const char* key,
                                             const std::string& parentPath) {
    if (!node.is_object()) {
        throw std::runtime_error(
            std::string("ai_config.json: expected object at '") + parentPath + "'");
    }

    auto it = node.find(key);
    if (it == node.end() || it->is_null()) {
        throw std::runtime_error(
            std::string("ai_config.json: missing required field '") + parentPath + "." + key + "'");
    }
    return *it;
}

inline const nlohmann::json& requireJsonObjectField(const nlohmann::json& node,
                                                    const char* key,
                                                    const std::string& parentPath) {
    const auto& field = requireJsonField(node, key, parentPath);
    if (!field.is_object()) {
        throw std::runtime_error(
            std::string("ai_config.json: field '") + parentPath + "." + key + "' must be an object");
    }
    return field;
}

inline const nlohmann::json& requireJsonArrayField(const nlohmann::json& node,
                                                   const char* key,
                                                   const std::string& parentPath) {
    const auto& field = requireJsonField(node, key, parentPath);
    if (!field.is_array()) {
        throw std::runtime_error(
            std::string("ai_config.json: field '") + parentPath + "." + key + "' must be an array");
    }
    return field;
}

template <typename FieldType>
inline std::decay_t<FieldType> getRequiredJsonValue(const nlohmann::json& node,
                                                    const char* key,
                                                    const std::string& parentPath) {
    const auto& field = requireJsonField(node, key, parentPath);
    try {
        return field.get<std::decay_t<FieldType>>();
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("ai_config.json: invalid value for field '") + parentPath + "." + key + "': " +
            e.what());
    }
}

inline std::string childJsonPath(const std::string& parentPath, const char* childPath) {
    if (childPath == nullptr || childPath[0] == '\0') {
        return parentPath;
    }
    return parentPath + "." + childPath;
}

struct AiConfigSnapshot {
    std::filesystem::path config_path;
    nlohmann::json document;
    std::string grim_text_vocab;
    std::string grim_text_model;
    std::string grim_text_training_data;
    std::string grim_text_checkpoints;
    std::string grim_text_collected;
    std::string grim_text_directory_collection;
    std::string grim_text_verified;
    std::string grim_text_logs;
    std::string grim_text_training_status;
    std::string grim_text_collector_log;
    std::string grim_text_source_config;
    std::string grim_text_model_store;
    std::string current_model_training;
    std::string current_curriculum;
    int epochs = 0;
    int64_t seed = 0;
    int batch_size = 0;
    int gradient_accumulation_steps = 0;
    std::string batch_strategy;
    float learning_rate = 0.0f;
    float weight_decay = 0.0f;
    float grad_clip_norm = 0.0f;
    bool per_token_grad_scale = false;
    bool force_rebuild_vocab = false;
    int d_model = 0;
    int num_layers = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int max_seq_len = 0;
    bool tie_embeddings = true;
    float dropout_rate = 0.0f;
    int sliding_window_stride = 0;
    float warmup_fraction = 0.0f;
    bool cosine_decay_enabled = false;
    bool cosine_warm_restarts = false;
    int log_interval = 0;
    int atom_stats_interval = 0;
    int atom_stats_max_seqs = 0;
    int validation_interval = 0;
    int checkpoint_interval = 0;
    bool use_gpu = true;
    bool use_flash_attention = true;
    std::string parameter_precision_embedding;
    std::string parameter_precision_lm_head;
    std::string parameter_precision_attention;
    std::string parameter_precision_ffn;
    std::string parameter_precision_rmsnorm;
    std::string parameter_precision_scratchblock;
    std::string parameter_precision_mtp;
    std::string parameter_precision_reasoning_head;
    std::string parameter_precision_execution_block;
    std::string parameter_precision_slot_selector;
    bool use_rope = false;
    bool use_alibi = false;
    int rope_base_seq_len = 0;
    int alibi_min_locality_distance = 0;
    float alibi_slope_exponent = 0.0f;
    float alibi_max_bias = 0.0f;
    float rope_theta = 0.0f;
    float rope_scaling = 0.0f;
    bool soft_restart_enabled = false;
    float soft_restart_loss_increase_threshold = 0.0f;
    int soft_restart_max_step_window = 0;
    int soft_restart_cooldown_steps = 0;
    bool auto_stop_enabled = false;
    int auto_stop_plateau_patience = 0;
    float auto_stop_plateau_min_delta = 0.0f;
    float auto_stop_high_loss_threshold = 0.0f;
    int auto_stop_high_loss_patience = 0;
    bool single_batch_overfit_enabled = false;
    int single_batch_overfit_max_steps = 0;
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
    std::string logging_default_level;
    bool logging_equation_csv_enabled = false;
    bool logging_stderr_enabled = false;
    size_t logging_initial_capacity = 0;
    std::map<std::string, std::string> logging_group_overrides;
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
    bool loss_label_smoothing_enabled = false;
    float loss_label_smoothing_epsilon = 0.0f;
    bool loss_focal_enabled = false;
    float loss_focal_gamma = 0.0f;
    float loss_focal_alpha = 0.0f;
    bool loss_entropy_reg_enabled = false;
    float loss_entropy_reg_lambda = 0.0f;
    bool loss_class_balanced_enabled = false;
    float loss_class_balanced_beta = 0.0f;
    bool loss_preference_enabled = false;
    float loss_preference_beta = 0.0f;
    bool loss_distillation_enabled = false;
    float loss_distillation_temperature = 0.0f;
    float loss_distillation_lambda = 0.0f;
    bool loss_masking_enabled = false;
    std::string loss_masking_tag;
    bool lm_head_centering_enabled = false;
    bool lm_head_center_hidden_states = false;
    bool freeze_learned_rms_gammas = false;
    bool center_logits = false;
    bool center_encoder_residuals = false;
    bool project_out_pc1 = false;
    int pc1_power_iters = 0;
    bool use_layer_scale = false;
    float layer_scale_init = 0.0f;
    bool qk_norm_enabled = false;
    bool hardcoded_hidden_states_enabled = false;
    std::string hardcoded_hidden_states_pattern;
    int hardcoded_log_every_n_batches = 0;
    bool embedding_freeze_enabled = false;
    int embedding_freeze_after_step = 0;
    std::string optimizer_kind;
    float optimizer_beta1 = 0.0f;
    float optimizer_beta2 = 0.0f;
    float optimizer_epsilon = 0.0f;
    bool stability_overrides_enabled = false;
    int stability_override_batch_size = 0;
    int stability_override_max_seq_len = 0;
    float stability_override_clip_per_token = 0.0f;
    bool scratch_blocks_enabled = false;
    size_t scratch_num_blocks = 0;
    bool scratch_write_combined = false;
    bool use_scratch_block = false;
    int scratch_block_atom_embedding_dim = 0;
    int scratch_block_max_atoms = 0;
    float scratch_block_atom_scale = 0.0f;
    bool execution_block_enabled = false;
    bool scratch_block_execution_first_type_only = false;
    bool execution_block_debug_mode = false;
    bool step_y_overrides_x = false;
    bool structured_ce_enabled = false;
    bool selector_enabled = false;
    int execution_block_layer = 0;
    int execution_block_num_ops = 0;
    int execution_block_num_slots = 0;
    int execution_block_num_scratch_slots = 0;
    int execution_block_num_steps = 0;
    int execution_block_value_decode_input_dim = 0;
    int execution_block_value_decode_hidden_dim = 0;
    int execution_block_d_type = 0;
    int execution_block_cross_attn_topk = 0;
    int execution_block_result_slot_mode = 0;
    int execution_block_result_slot_index = 0;
    int execution_block_temp_schedule = 0;
    int decode_time_slot_feature_dim = 0;
    int selector_d_selector = 0;
    float execution_block_usage_decay = 0.0f;
    float execution_block_inject_gate_temp = 0.0f;
    float execution_block_entropy_collapse_threshold = 0.0f;
    float execution_block_write_collapse_threshold = 0.0f;
    float execution_block_magnitude_limit = 0.0f;
    float execution_block_diversity_kappa = 0.0f;
    float execution_block_temp_start = 0.0f;
    float execution_block_temp_end = 0.0f;
    float execution_block_entropy_weight = 0.0f;
    float step_x_multiplier = 0.0f;
    float step_y_multiplier = 0.0f;
    float entropy_aux_weight = 0.0f;
    float value_match_epsilon = 0.0f;
    float final_slot_consistency_weight = 0.0f;
    float execution_block_transition_hard_threshold = 0.0f;
    float execution_block_causal_w1_transition = 0.0f;
    float div_invalid_penalty_weight = 0.0f;
    float div_magnitude_penalty_weight = 0.0f;
    float arg_reinforce_weight = 0.0f;
    float arg_reinforce_baseline_decay = 0.0f;
    float structured_ce_weight = 0.0f;
    float selector_selection_margin = 0.0f;
    float selector_supervision_weight = 0.0f;
    bool single_stream_mode = false;
    bool disable_async_frees = false;
    bool synchronize_after_kernels = false;
    bool mtp_enabled = false;
    bool mtp_log_ratio_monitor = false;
    int mtp_k = 0;
    float mtp_alpha = 0.0f;
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
    bool tokenizer_enable_scratch_block_reasoning = false;
    bool tokenizer_detect_numbers = false;
    std::string generation_strategy;
    int generation_max_new_tokens = 0;
    int generation_min_new_tokens = 0;
    int generation_top_k = 0;
    int generation_repetition_penalty_window = 0;
    int generation_no_repeat_ngram_size = 0;
    float generation_temperature = 0.0f;
    float generation_top_p = 0.0f;
    float generation_min_p = 0.0f;
    float generation_typical_p = 0.0f;
    float generation_repetition_penalty = 0.0f;
    float generation_frequency_penalty = 0.0f;
    float generation_presence_penalty = 0.0f;
    bool generation_do_sample = false;
    bool generation_enable_scratchblock_reasoning = false;
    int tokenizer_vocab_size = 50000;
    int tokenizer_max_vocab_size = 0;
    int tokenizer_max_length = 8192;
    float tokenizer_character_coverage = 0.0f;
    int tokenizer_min_cleaned_text_length = 0;
    int tokenizer_min_subword_freq = 3;
    bool tokenizer_prune_during_mining = false;
    bool tokenizer_enable_parallel_subword_mining = true;
    int tokenizer_subword_mining_workers = 0;
    size_t tokenizer_subword_mining_max_bytes = 0;
    std::string tokenizer_model_type = "unibytes";
    std::vector<std::string> tokenizer_special_tokens = {"<pad>", "<unk>", "<s>", "</s>"};
    bool tokenizer_add_bos = true;
    bool tokenizer_add_eos = true;
    std::string tokenizer_unk_token = "<unk>";
    std::string tokenizer_pad_token = "<pad>";
    std::string tokenizer_bos_token = "<s>";
    std::string tokenizer_eos_token = "</s>";
    bool tokenizer_enable_nfkc_normalization = true;
    bool tokenizer_enable_lowercasing = true;
    bool tokenizer_enable_parallel_tokenization = true;
    int tokenizer_parallel_threshold = 1000;
    bool tokenizer_enable_byte_fallback = true;
    uint32_t tokenizer_expected_checksum = 0;
    bool tokenizer_save_text_vocab = true;
    float tokenizer_vocab_score_multiplier = 1.0f;
    bool clear_merged_cache_on_merge = false;
    bool subprocess_tokenizer_only_mode = false;

    bool hasRequiredGrimTextPaths() const {
        return !grim_text_vocab.empty() && !grim_text_training_data.empty();
    }
};

inline std::optional<AiConfigSnapshot> loadAiConfigSnapshot();

namespace detail {

inline constexpr const char* kRootConfigPath = "ai_config.json";
inline constexpr const char* kTrainingPath = "training";
inline constexpr const char* kTrainingConfigPath = "training.config";
inline constexpr const char* kGrimTextConfigPath = "paths.grim_text";

struct GrimTextPathBinding {
    const char* json_key;
    std::string AiConfigSnapshot::* snapshot_field;
};

inline constexpr std::array<GrimTextPathBinding, 12> kGrimTextPathBindings{{
    {"vocab", &AiConfigSnapshot::grim_text_vocab},
    {"model", &AiConfigSnapshot::grim_text_model},
    {"training_data", &AiConfigSnapshot::grim_text_training_data},
    {"checkpoints", &AiConfigSnapshot::grim_text_checkpoints},
    {"collected", &AiConfigSnapshot::grim_text_collected},
    {"directory_collection", &AiConfigSnapshot::grim_text_directory_collection},
    {"verified", &AiConfigSnapshot::grim_text_verified},
    {"logs", &AiConfigSnapshot::grim_text_logs},
    {"training_status", &AiConfigSnapshot::grim_text_training_status},
    {"collector_log", &AiConfigSnapshot::grim_text_collector_log},
    {"source_config", &AiConfigSnapshot::grim_text_source_config},
    {"model_store", &AiConfigSnapshot::grim_text_model_store},
}};

inline std::filesystem::path resolveAiConfigPath() {
    namespace fs = std::filesystem;
    fs::path defaultCandidate = fs::current_path() / "ai_config.json";
    if (fs::exists(defaultCandidate)) {
        return defaultCandidate;
    }

    fs::path searchPath = fs::current_path();
    for (int i = 0; i < 5 && searchPath.has_parent_path(); ++i) {
        searchPath = searchPath.parent_path();
        fs::path parentCandidate = searchPath / "ai_config.json";
        if (fs::exists(parentCandidate)) {
            return parentCandidate;
        }
    }

    throw std::runtime_error(
        "resolveAiConfigPath: ai_config.json not found in current directory or parent directories");
}

inline std::filesystem::path resolveGrimRoot() {
    namespace fs = std::filesystem;
    
#ifdef GRIM_ROOT_DIR
    fs::path root = fs::path(GRIM_ROOT_DIR);
    if (fs::exists(root)) {
        return root;
    }
#endif

    // Try to find GRIM root by walking up from current directory or executable location
    fs::path searchPath = fs::current_path();
    
    // Also try executable directory
    fs::path exeDir;
#ifdef _WIN32
    char buffer[MAX_PATH];
    if (GetModuleFileNameA(nullptr, buffer, MAX_PATH)) {
        exeDir = fs::path(buffer).parent_path();
    }
#elif defined(__APPLE__)
    char buffer[PATH_MAX];
    uint32_t size = sizeof(buffer);
    if (_NSGetExecutablePath(buffer, &size) == 0) {
        exeDir = fs::path(buffer).parent_path();
    }
#else
    char buffer[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len > 0) {
        buffer[len] = '\0';
        exeDir = fs::path(buffer).parent_path();
    }
#endif
    
    std::vector<fs::path> searchPaths = {searchPath, exeDir};
    
    for (auto& base : searchPaths) {
        if (base.empty()) continue;
        
        fs::path probe = base;
        for (int i = 0; i < 10 && probe.has_parent_path(); ++i) {
            if (fs::exists(probe / "control") && fs::exists(probe / "resources")) {
                return probe;
            }
            if (!probe.has_parent_path()) break;
            probe = probe.parent_path();
        }
    }
    
    // Fallback: return current directory
    return fs::current_path();
}

inline void assignGrimTextPathFields(const nlohmann::json& config, AiConfigSnapshot& snapshot) {
    const auto& grimTextPaths =
        requireJsonObjectField(requireJsonObjectField(config, "paths", kRootConfigPath),
                               "grim_text",
                               "paths");
    const std::filesystem::path grimRoot = resolveGrimRoot();
    const auto assignPath = [&](std::string& field, const char* key) {
        std::string pathStr = getRequiredJsonValue<std::string>(grimTextPaths, key, kGrimTextConfigPath);
        std::filesystem::path path(pathStr);
        field = path.is_relative() ? (grimRoot / path).string() : pathStr;
    };

    for (const auto& binding : kGrimTextPathBindings) {
        assignPath(snapshot.*(binding.snapshot_field), binding.json_key);
    }
}

inline const std::string* findGrimTextPathField(const AiConfigSnapshot& snapshot,
                                                std::string_view key) {
    for (const auto& binding : kGrimTextPathBindings) {
        if (key == binding.json_key) {
            return &(snapshot.*(binding.snapshot_field));
        }
    }
    return nullptr;
}

template <typename FieldType>
inline void assignTrainingField(FieldType& field, const nlohmann::json& node, const char* key) {
    field = getRequiredJsonValue<FieldType>(node, key, kTrainingConfigPath);
}

inline std::map<std::string, std::string> getRequiredStringMap(const nlohmann::json& node,
                                                               const char* key,
                                                               const std::string& parentPath) {
    std::map<std::string, std::string> values;
    const auto& jsonMap = requireJsonObjectField(node, key, parentPath);
    for (const auto& [entryKey, entryValue] : jsonMap.items()) {
        if (!entryValue.is_string()) {
            throw std::runtime_error(
                std::string("ai_config.json: ") + childJsonPath(childJsonPath(parentPath, key), entryKey.c_str()) +
                " must be a string");
        }
        values.emplace(entryKey, entryValue.get<std::string>());
    }
    return values;
}

inline std::vector<std::string> getRequiredStringArray(const nlohmann::json& node,
                                                       const char* key,
                                                       const std::string& parentPath) {
    std::vector<std::string> values;
    const auto& jsonArray = requireJsonArrayField(node, key, parentPath);
    values.reserve(jsonArray.size());
    for (const auto& entryValue : jsonArray) {
        if (!entryValue.is_string()) {
            throw std::runtime_error(
                std::string("ai_config.json: ") + childJsonPath(parentPath, key) +
                " entries must all be strings");
        }
        values.push_back(entryValue.get<std::string>());
    }
    return values;
}

inline void assignSnapshotTrainingConfigFields(const nlohmann::json& config,
                                               AiConfigSnapshot& snapshot) {
    const auto& trainConfig =
        requireJsonObjectField(requireJsonObjectField(config, "training", kRootConfigPath),
                               "config",
                               kTrainingPath);

    assignTrainingField(snapshot.clear_merged_cache_on_merge, trainConfig, "clear_merged_cache_on_merge");
    assignTrainingField(snapshot.current_model_training, trainConfig, "current_model_training");
    assignTrainingField(snapshot.current_curriculum, trainConfig, "current_curriculum");

    assignTrainingField(snapshot.epochs, trainConfig, "epochs");
    assignTrainingField(snapshot.seed, trainConfig, "seed");
    assignTrainingField(snapshot.batch_size, trainConfig, "batch_size");
    assignTrainingField(snapshot.gradient_accumulation_steps, trainConfig, "gradient_accumulation_steps");
    assignTrainingField(snapshot.batch_strategy, trainConfig, "batch_strategy");
    assignTrainingField(snapshot.learning_rate, trainConfig, "learning_rate");
    assignTrainingField(snapshot.weight_decay, trainConfig, "weight_decay");
    assignTrainingField(snapshot.grad_clip_norm, trainConfig, "gradient_clip");
    assignTrainingField(snapshot.per_token_grad_scale, trainConfig, "per_token_grad_scale");
    assignTrainingField(snapshot.force_rebuild_vocab, trainConfig, "force_rebuild_vocab");
    assignTrainingField(snapshot.d_model, trainConfig, "d_model");
    assignTrainingField(snapshot.num_layers, trainConfig, "num_layers");
    assignTrainingField(snapshot.num_heads, trainConfig, "num_heads");
    assignTrainingField(snapshot.num_kv_heads, trainConfig, "num_kv_heads");
    assignTrainingField(snapshot.max_seq_len, trainConfig, "max_seq_len");
    assignTrainingField(snapshot.tie_embeddings, trainConfig, "tie_embeddings");
    assignTrainingField(snapshot.dropout_rate, trainConfig, "dropout_rate");
    assignTrainingField(snapshot.sliding_window_stride, trainConfig, "sliding_window_stride");
    assignTrainingField(snapshot.warmup_fraction, trainConfig, "warmup_fraction");
    assignTrainingField(snapshot.cosine_decay_enabled, trainConfig, "cosine_decay_enabled");
    assignTrainingField(snapshot.cosine_warm_restarts, trainConfig, "cosine_warm_restarts");
    assignTrainingField(snapshot.log_interval, trainConfig, "log_interval");
    assignTrainingField(snapshot.atom_stats_interval, trainConfig, "atom_stats_interval");
    assignTrainingField(snapshot.atom_stats_max_seqs, trainConfig, "atom_stats_max_seqs");
    assignTrainingField(snapshot.validation_interval, trainConfig, "validation_interval");
    assignTrainingField(snapshot.checkpoint_interval, trainConfig, "checkpoint_interval");
    assignTrainingField(snapshot.use_gpu, trainConfig, "use_gpu");
    assignTrainingField(snapshot.use_flash_attention, trainConfig, "use_flash_attention");
    assignTrainingField(snapshot.parameter_precision_embedding, trainConfig, "parameter_precision_embedding");
    assignTrainingField(snapshot.parameter_precision_lm_head, trainConfig, "parameter_precision_lm_head");
    assignTrainingField(snapshot.parameter_precision_attention, trainConfig, "parameter_precision_attention");
    assignTrainingField(snapshot.parameter_precision_ffn, trainConfig, "parameter_precision_ffn");
    assignTrainingField(snapshot.parameter_precision_rmsnorm, trainConfig, "parameter_precision_rmsnorm");
    assignTrainingField(snapshot.parameter_precision_scratchblock, trainConfig, "parameter_precision_scratchblock");
    assignTrainingField(snapshot.parameter_precision_mtp, trainConfig, "parameter_precision_mtp");
    assignTrainingField(snapshot.parameter_precision_reasoning_head, trainConfig, "parameter_precision_reasoning_head");
    assignTrainingField(snapshot.parameter_precision_execution_block, trainConfig, "parameter_precision_execution_block");
    assignTrainingField(snapshot.parameter_precision_slot_selector, trainConfig, "parameter_precision_slot_selector");
    assignTrainingField(snapshot.use_rope, trainConfig, "use_rope");
    assignTrainingField(snapshot.use_alibi, trainConfig, "use_alibi");
    assignTrainingField(snapshot.rope_base_seq_len, trainConfig, "rope_base_seq_len");
    assignTrainingField(snapshot.alibi_min_locality_distance, trainConfig, "alibi_min_locality_distance");
    assignTrainingField(snapshot.alibi_slope_exponent, trainConfig, "alibi_slope_exponent");
    assignTrainingField(snapshot.alibi_max_bias, trainConfig, "alibi_max_bias");
    assignTrainingField(snapshot.rope_theta, trainConfig, "rope_theta");
    assignTrainingField(snapshot.rope_scaling, trainConfig, "rope_scaling");
    assignTrainingField(snapshot.soft_restart_enabled, trainConfig, "soft_restart_enabled");
    assignTrainingField(snapshot.soft_restart_loss_increase_threshold, trainConfig, "soft_restart_loss_increase_threshold");
    assignTrainingField(snapshot.soft_restart_max_step_window, trainConfig, "soft_restart_max_step_window");
    assignTrainingField(snapshot.soft_restart_cooldown_steps, trainConfig, "soft_restart_cooldown_steps");
    assignTrainingField(snapshot.auto_stop_enabled, trainConfig, "auto_stop_enabled");
    assignTrainingField(snapshot.auto_stop_plateau_patience, trainConfig, "auto_stop_plateau_patience");
    assignTrainingField(snapshot.auto_stop_plateau_min_delta, trainConfig, "auto_stop_plateau_min_delta");
    assignTrainingField(snapshot.auto_stop_high_loss_threshold, trainConfig, "auto_stop_high_loss_threshold");
    assignTrainingField(snapshot.auto_stop_high_loss_patience, trainConfig, "auto_stop_high_loss_patience");
    assignTrainingField(snapshot.single_batch_overfit_enabled, trainConfig, "single_batch_overfit_enabled");
    assignTrainingField(snapshot.single_batch_overfit_max_steps, trainConfig, "single_batch_overfit_max_steps");
    assignTrainingField(snapshot.shuffle_train_enabled, trainConfig, "shuffle_train_enabled");
    assignTrainingField(snapshot.shuffle_train_epochs, trainConfig, "shuffle_train_epochs");
    assignTrainingField(snapshot.telemetry_control_enabled, trainConfig, "telemetry_control_enabled");
    assignTrainingField(snapshot.telemetry_spike_mild_threshold, trainConfig, "telemetry_spike_mild_threshold");
    assignTrainingField(snapshot.telemetry_spike_moderate_threshold, trainConfig, "telemetry_spike_moderate_threshold");
    assignTrainingField(snapshot.telemetry_spike_severe_threshold, trainConfig, "telemetry_spike_severe_threshold");
    assignTrainingField(snapshot.telemetry_moderate_grad_scale, trainConfig, "telemetry_moderate_grad_scale");
    assignTrainingField(snapshot.telemetry_moderate_cooldown_extension, trainConfig, "telemetry_moderate_cooldown_extension");
    assignTrainingField(snapshot.telemetry_min_grad_for_nonzero_loss, trainConfig, "telemetry_min_grad_for_nonzero_loss");
    assignTrainingField(snapshot.telemetry_loss_threshold_for_grad_check, trainConfig, "telemetry_loss_threshold_for_grad_check");
    assignTrainingField(snapshot.telemetry_max_consecutive_zero_grad_steps, trainConfig, "telemetry_max_consecutive_zero_grad_steps");
    assignTrainingField(snapshot.telemetry_seq_len_regime_change_threshold, trainConfig, "telemetry_seq_len_regime_change_threshold");
    assignTrainingField(snapshot.telemetry_regime_change_suppression_steps, trainConfig, "telemetry_regime_change_suppression_steps");
    assignTrainingField(snapshot.telemetry_volatility_damping_threshold, trainConfig, "telemetry_volatility_damping_threshold");
    assignTrainingField(snapshot.telemetry_max_volatility_damping, trainConfig, "telemetry_max_volatility_damping");
    assignTrainingField(snapshot.telemetry_gradient_decay_threshold, trainConfig, "telemetry_gradient_decay_threshold");
    assignTrainingField(snapshot.telemetry_max_decay_boost, trainConfig, "telemetry_max_decay_boost");
    assignTrainingField(snapshot.telemetry_progress_boost_threshold, trainConfig, "telemetry_progress_boost_threshold");
    assignTrainingField(snapshot.telemetry_max_progress_boost, trainConfig, "telemetry_max_progress_boost");
    assignTrainingField(snapshot.telemetry_outlier_frequency_trigger, trainConfig, "telemetry_outlier_frequency_trigger");
    assignTrainingField(snapshot.telemetry_outlier_persistence_trigger, trainConfig, "telemetry_outlier_persistence_trigger");
    assignTrainingField(snapshot.telemetry_anchor_drift_sigma_multiplier, trainConfig, "telemetry_anchor_drift_sigma_multiplier");
    assignTrainingField(snapshot.telemetry_soft_restart_cooldown_steps, trainConfig, "telemetry_soft_restart_cooldown_steps");
    assignTrainingField(snapshot.telemetry_baseline_stabilization_steps, trainConfig, "telemetry_baseline_stabilization_steps");
    assignTrainingField(snapshot.telemetry_verbose_logging, trainConfig, "telemetry_verbose_logging");
    assignTrainingField(snapshot.telemetry_fail_loud_on_accumulation_bug, trainConfig, "telemetry_fail_loud_on_accumulation_bug");
    assignTrainingField(snapshot.telemetry_plateau_noise_enabled, trainConfig, "telemetry_plateau_noise_enabled");
    assignTrainingField(snapshot.telemetry_plateau_noise_patience, trainConfig, "telemetry_plateau_noise_patience");
    assignTrainingField(snapshot.telemetry_plateau_noise_variance_threshold, trainConfig, "telemetry_plateau_noise_variance_threshold");
    assignTrainingField(snapshot.telemetry_plateau_noise_std, trainConfig, "telemetry_plateau_noise_std");
    assignTrainingField(snapshot.telemetry_plateau_noise_proportional, trainConfig, "telemetry_plateau_noise_proportional");
    assignTrainingField(snapshot.telemetry_plateau_noise_cooldown, trainConfig, "telemetry_plateau_noise_cooldown");
    assignTrainingField(snapshot.telemetry_plateau_noise_max_per_epoch, trainConfig, "telemetry_plateau_noise_max_per_epoch");
    assignTrainingField(snapshot.telemetry_lattice_num_levels, trainConfig, "telemetry_lattice_num_levels");
    assignTrainingField(snapshot.telemetry_lattice_num_streams, trainConfig, "telemetry_lattice_num_streams");
    assignTrainingField(snapshot.telemetry_lattice_beta_mu, trainConfig, "telemetry_lattice_beta_mu");
    assignTrainingField(snapshot.telemetry_lattice_beta_a, trainConfig, "telemetry_lattice_beta_a");
    assignTrainingField(snapshot.telemetry_lattice_beta_delta, trainConfig, "telemetry_lattice_beta_delta");
    assignTrainingField(snapshot.telemetry_lattice_beta_r, trainConfig, "telemetry_lattice_beta_r");
    assignTrainingField(snapshot.telemetry_lattice_beta_run, trainConfig, "telemetry_lattice_beta_run");
    assignTrainingField(snapshot.telemetry_lattice_beta_v, trainConfig, "telemetry_lattice_beta_v");
    assignTrainingField(snapshot.telemetry_lattice_k_out0, trainConfig, "telemetry_lattice_k_out0");
    assignTrainingField(snapshot.telemetry_lattice_alpha_v, trainConfig, "telemetry_lattice_alpha_v");
    assignTrainingField(snapshot.telemetry_lattice_epsilon, trainConfig, "telemetry_lattice_epsilon");
    assignTrainingField(snapshot.telemetry_lattice_strict_mode, trainConfig, "telemetry_lattice_strict_mode");
    assignTrainingField(snapshot.logging_default_level, trainConfig, "logging_default_level");
    assignTrainingField(snapshot.logging_equation_csv_enabled, trainConfig, "logging_equation_csv_enabled");
    assignTrainingField(snapshot.logging_stderr_enabled, trainConfig, "logging_stderr_enabled");
    assignTrainingField(snapshot.logging_initial_capacity, trainConfig, "logging_initial_capacity");
    snapshot.logging_group_overrides =
        getRequiredStringMap(trainConfig, "logging_group_overrides", kTrainingConfigPath);
    assignTrainingField(snapshot.log_recorder_enabled, trainConfig, "log_recorder_enabled");
    assignTrainingField(snapshot.log_recorder_default_level, trainConfig, "log_recorder_default_level");
    snapshot.log_recorder_modules =
        getRequiredStringMap(trainConfig, "log_recorder_modules", kTrainingConfigPath);
    assignTrainingField(snapshot.log_recorder_layer_embedding, trainConfig, "log_recorder_layer_embedding");
    assignTrainingField(snapshot.log_recorder_layer_rms_norm, trainConfig, "log_recorder_layer_rms_norm");
    assignTrainingField(snapshot.log_recorder_layer_attention, trainConfig, "log_recorder_layer_attention");
    assignTrainingField(snapshot.log_recorder_layer_feed_forward, trainConfig, "log_recorder_layer_feed_forward");
    assignTrainingField(snapshot.log_recorder_layer_residual, trainConfig, "log_recorder_layer_residual");
    assignTrainingField(snapshot.log_recorder_layer_encoding, trainConfig, "log_recorder_layer_encoding");
    assignTrainingField(snapshot.log_recorder_layer_serialization, trainConfig, "log_recorder_layer_serialization");
    assignTrainingField(snapshot.log_recorder_layer_execution_block, trainConfig, "log_recorder_layer_execution_block");
    assignTrainingField(snapshot.loss_label_smoothing_enabled, trainConfig, "loss_label_smoothing_enabled");
    assignTrainingField(snapshot.loss_label_smoothing_epsilon, trainConfig, "loss_label_smoothing_epsilon");
    assignTrainingField(snapshot.loss_focal_enabled, trainConfig, "loss_focal_enabled");
    assignTrainingField(snapshot.loss_focal_gamma, trainConfig, "loss_focal_gamma");
    assignTrainingField(snapshot.loss_focal_alpha, trainConfig, "loss_focal_alpha");
    assignTrainingField(snapshot.loss_entropy_reg_enabled, trainConfig, "loss_entropy_reg_enabled");
    assignTrainingField(snapshot.loss_entropy_reg_lambda, trainConfig, "loss_entropy_reg_lambda");
    assignTrainingField(snapshot.loss_class_balanced_enabled, trainConfig, "loss_class_balanced_enabled");
    assignTrainingField(snapshot.loss_class_balanced_beta, trainConfig, "loss_class_balanced_beta");
    assignTrainingField(snapshot.loss_preference_enabled, trainConfig, "loss_preference_enabled");
    assignTrainingField(snapshot.loss_preference_beta, trainConfig, "loss_preference_beta");
    assignTrainingField(snapshot.loss_distillation_enabled, trainConfig, "loss_distillation_enabled");
    assignTrainingField(snapshot.loss_distillation_temperature, trainConfig, "loss_distillation_temperature");
    assignTrainingField(snapshot.loss_distillation_lambda, trainConfig, "loss_distillation_lambda");
    assignTrainingField(snapshot.loss_masking_enabled, trainConfig, "loss_masking_enabled");
    assignTrainingField(snapshot.loss_masking_tag, trainConfig, "loss_masking_tag");
    assignTrainingField(snapshot.lm_head_centering_enabled, trainConfig, "lm_head_centering_enabled");
    assignTrainingField(snapshot.lm_head_center_hidden_states, trainConfig, "lm_head_center_hidden_states");
    assignTrainingField(snapshot.freeze_learned_rms_gammas, trainConfig, "freeze_learned_rms_gammas");
    assignTrainingField(snapshot.center_logits, trainConfig, "center_logits");
    assignTrainingField(snapshot.center_encoder_residuals, trainConfig, "center_encoder_residuals");
    assignTrainingField(snapshot.project_out_pc1, trainConfig, "project_out_pc1");
    assignTrainingField(snapshot.pc1_power_iters, trainConfig, "pc1_power_iters");
    assignTrainingField(snapshot.use_layer_scale, trainConfig, "use_layer_scale");
    assignTrainingField(snapshot.layer_scale_init, trainConfig, "layer_scale_init");
    assignTrainingField(snapshot.qk_norm_enabled, trainConfig, "qk_norm_enabled");
    assignTrainingField(snapshot.hardcoded_hidden_states_enabled, trainConfig, "hardcoded_hidden_states_enabled");
    assignTrainingField(snapshot.hardcoded_hidden_states_pattern, trainConfig, "hardcoded_hidden_states_pattern");
    assignTrainingField(snapshot.hardcoded_log_every_n_batches, trainConfig, "hardcoded_log_every_n_batches");
    assignTrainingField(snapshot.embedding_freeze_enabled, trainConfig, "embedding_freeze_enabled");
    assignTrainingField(snapshot.embedding_freeze_after_step, trainConfig, "embedding_freeze_after_step");
    assignTrainingField(snapshot.optimizer_kind, trainConfig, "optimizer_kind");
    assignTrainingField(snapshot.optimizer_beta1, trainConfig, "optimizer_beta1");
    assignTrainingField(snapshot.optimizer_beta2, trainConfig, "optimizer_beta2");
    assignTrainingField(snapshot.optimizer_epsilon, trainConfig, "optimizer_epsilon");
    assignTrainingField(snapshot.stability_overrides_enabled, trainConfig, "stability_overrides_enabled");
    assignTrainingField(snapshot.stability_override_batch_size, trainConfig, "stability_override_batch_size");
    assignTrainingField(snapshot.stability_override_max_seq_len, trainConfig, "stability_override_max_seq_len");
    assignTrainingField(snapshot.stability_override_clip_per_token, trainConfig, "stability_override_clip_per_token");
    assignTrainingField(snapshot.scratch_blocks_enabled, trainConfig, "scratch_blocks_enabled");
    assignTrainingField(snapshot.scratch_num_blocks, trainConfig, "scratch_num_blocks");
    assignTrainingField(snapshot.scratch_write_combined, trainConfig, "scratch_write_combined");
    assignTrainingField(snapshot.use_scratch_block, trainConfig, "use_scratch_block");
    assignTrainingField(snapshot.scratch_block_atom_embedding_dim, trainConfig, "scratch_block_atom_embedding_dim");
    assignTrainingField(snapshot.scratch_block_max_atoms, trainConfig, "scratch_block_max_atoms");
    assignTrainingField(snapshot.scratch_block_atom_scale, trainConfig, "scratch_block_atom_scale");
    assignTrainingField(snapshot.execution_block_enabled, trainConfig, "execution_block_enabled");
    assignTrainingField(snapshot.scratch_block_execution_first_type_only, trainConfig, "execution_block_execution_first_type_only");
    assignTrainingField(snapshot.execution_block_debug_mode, trainConfig, "execution_block_debug_mode");
    assignTrainingField(snapshot.step_y_overrides_x, trainConfig, "execution_block_step_y_overrides_x");
    assignTrainingField(snapshot.structured_ce_enabled, trainConfig, "execution_block_structured_ce_enabled");
    assignTrainingField(snapshot.selector_enabled, trainConfig, "selector_enabled");
    assignTrainingField(snapshot.execution_block_layer, trainConfig, "execution_block_layer");
    assignTrainingField(snapshot.execution_block_num_ops, trainConfig, "execution_block_num_ops");
    assignTrainingField(snapshot.execution_block_num_slots, trainConfig, "execution_block_num_slots");
    assignTrainingField(snapshot.execution_block_num_scratch_slots, trainConfig, "execution_block_num_scratch_slots");
    assignTrainingField(snapshot.execution_block_num_steps, trainConfig, "execution_block_num_steps");
    assignTrainingField(snapshot.execution_block_value_decode_input_dim, trainConfig, "execution_block_value_decode_input_dim");
    assignTrainingField(snapshot.execution_block_value_decode_hidden_dim, trainConfig, "execution_block_value_decode_hidden_dim");
    assignTrainingField(snapshot.execution_block_d_type, trainConfig, "execution_block_d_type");
    assignTrainingField(snapshot.execution_block_cross_attn_topk, trainConfig, "execution_block_cross_attn_topk");
    assignTrainingField(snapshot.execution_block_result_slot_mode, trainConfig, "execution_block_result_slot_mode");
    assignTrainingField(snapshot.execution_block_result_slot_index, trainConfig, "execution_block_result_slot_index");
    assignTrainingField(snapshot.execution_block_temp_schedule, trainConfig, "execution_block_temp_schedule");
    assignTrainingField(snapshot.decode_time_slot_feature_dim, trainConfig, "selector_d_slot_features");
    assignTrainingField(snapshot.selector_d_selector, trainConfig, "selector_d_selector");
    assignTrainingField(snapshot.execution_block_usage_decay, trainConfig, "execution_block_usage_decay");
    assignTrainingField(snapshot.execution_block_inject_gate_temp, trainConfig, "execution_block_inject_gate_temp");
    assignTrainingField(snapshot.execution_block_entropy_collapse_threshold, trainConfig, "execution_block_entropy_collapse_threshold");
    assignTrainingField(snapshot.execution_block_write_collapse_threshold, trainConfig, "execution_block_write_collapse_threshold");
    assignTrainingField(snapshot.execution_block_magnitude_limit, trainConfig, "execution_block_magnitude_limit");
    assignTrainingField(snapshot.execution_block_diversity_kappa, trainConfig, "execution_block_diversity_kappa");
    assignTrainingField(snapshot.execution_block_temp_start, trainConfig, "execution_block_temp_start");
    assignTrainingField(snapshot.execution_block_temp_end, trainConfig, "execution_block_temp_end");
    assignTrainingField(snapshot.execution_block_entropy_weight, trainConfig, "execution_block_entropy_weight");
    assignTrainingField(snapshot.step_x_multiplier, trainConfig, "execution_block_step_x_multiplier");
    assignTrainingField(snapshot.step_y_multiplier, trainConfig, "execution_block_step_y_multiplier");
    assignTrainingField(snapshot.entropy_aux_weight, trainConfig, "execution_block_entropy_aux_weight");
    assignTrainingField(snapshot.value_match_epsilon, trainConfig, "execution_block_value_match_epsilon");
    assignTrainingField(snapshot.final_slot_consistency_weight, trainConfig, "execution_block_final_slot_consistency_weight");
    assignTrainingField(snapshot.execution_block_transition_hard_threshold, trainConfig, "execution_block_transition_hard_threshold");
    assignTrainingField(snapshot.execution_block_causal_w1_transition, trainConfig, "execution_block_causal_w1_transition");
    assignTrainingField(snapshot.div_invalid_penalty_weight, trainConfig, "execution_block_div_invalid_penalty_weight");
    assignTrainingField(snapshot.div_magnitude_penalty_weight, trainConfig, "execution_block_div_magnitude_penalty_weight");
    assignTrainingField(snapshot.arg_reinforce_weight, trainConfig, "execution_block_arg_reinforce_weight");
    assignTrainingField(snapshot.arg_reinforce_baseline_decay, trainConfig, "execution_block_arg_reinforce_baseline_decay");
    assignTrainingField(snapshot.structured_ce_weight, trainConfig, "execution_block_structured_ce_weight");
    assignTrainingField(snapshot.selector_selection_margin, trainConfig, "selector_selection_margin");
    assignTrainingField(snapshot.selector_supervision_weight, trainConfig, "selector_supervision_weight");
    assignTrainingField(snapshot.single_stream_mode, trainConfig, "single_stream_mode");
    assignTrainingField(snapshot.disable_async_frees, trainConfig, "disable_async_frees");
    assignTrainingField(snapshot.synchronize_after_kernels, trainConfig, "synchronize_after_kernels");
    assignTrainingField(snapshot.mtp_enabled, trainConfig, "mtp_enabled");
    assignTrainingField(snapshot.mtp_log_ratio_monitor, trainConfig, "mtp_log_ratio_monitor");
    assignTrainingField(snapshot.mtp_k, trainConfig, "mtp_k");
    assignTrainingField(snapshot.mtp_alpha, trainConfig, "mtp_alpha");
    assignTrainingField(snapshot.prediction_comparison_enabled, trainConfig, "prediction_comparison_enabled");
    assignTrainingField(snapshot.prediction_comparison_interval, trainConfig, "prediction_comparison_interval");
    assignTrainingField(snapshot.prediction_comparison_top_k, trainConfig, "prediction_comparison_top_k");
    assignTrainingField(snapshot.prediction_comparison_max_positions, trainConfig, "prediction_comparison_max_positions");
    assignTrainingField(snapshot.prediction_comparison_log_path, trainConfig, "prediction_comparison_log_path");
    assignTrainingField(snapshot.logit_update_trace_enabled, trainConfig, "logit_update_trace_enabled");
    assignTrainingField(snapshot.logit_update_trace_interval, trainConfig, "logit_update_trace_interval");
    assignTrainingField(snapshot.attention_diag_enabled, trainConfig, "attention_diag_enabled");
    assignTrainingField(snapshot.attention_diag_layer, trainConfig, "attention_diag_layer");
    assignTrainingField(snapshot.attention_diag_head, trainConfig, "attention_diag_head");
    assignTrainingField(snapshot.generation_strategy, trainConfig, "generation_strategy");
    assignTrainingField(snapshot.generation_max_new_tokens, trainConfig, "generation_max_new_tokens");
    assignTrainingField(snapshot.generation_min_new_tokens, trainConfig, "generation_min_new_tokens");
    assignTrainingField(snapshot.generation_top_k, trainConfig, "generation_top_k");
    assignTrainingField(snapshot.generation_repetition_penalty_window, trainConfig, "generation_repetition_penalty_window");
    assignTrainingField(snapshot.generation_no_repeat_ngram_size, trainConfig, "generation_no_repeat_ngram_size");
    assignTrainingField(snapshot.generation_temperature, trainConfig, "generation_temperature");
    assignTrainingField(snapshot.generation_top_p, trainConfig, "generation_top_p");
    assignTrainingField(snapshot.generation_min_p, trainConfig, "generation_min_p");
    assignTrainingField(snapshot.generation_typical_p, trainConfig, "generation_typical_p");
    assignTrainingField(snapshot.generation_repetition_penalty, trainConfig, "generation_repetition_penalty");
    assignTrainingField(snapshot.generation_frequency_penalty, trainConfig, "generation_frequency_penalty");
    assignTrainingField(snapshot.generation_presence_penalty, trainConfig, "generation_presence_penalty");
    assignTrainingField(snapshot.generation_do_sample, trainConfig, "generation_do_sample");
    assignTrainingField(snapshot.generation_enable_scratchblock_reasoning, trainConfig, "generation_enable_scratchblock_reasoning");
    assignTrainingField(snapshot.tokenizer_vocab_size, trainConfig, "tokenizer_vocab_size");
    assignTrainingField(snapshot.tokenizer_max_vocab_size, trainConfig, "tokenizer_max_vocab_size");
    assignTrainingField(snapshot.tokenizer_max_length, trainConfig, "tokenizer_max_length");
    assignTrainingField(snapshot.tokenizer_min_cleaned_text_length, trainConfig, "tokenizer_min_cleaned_text_length");
    assignTrainingField(snapshot.tokenizer_min_subword_freq, trainConfig, "tokenizer_min_subword_freq");
    assignTrainingField(snapshot.tokenizer_subword_mining_workers, trainConfig, "tokenizer_subword_mining_workers");
    assignTrainingField(snapshot.tokenizer_parallel_threshold, trainConfig, "tokenizer_parallel_threshold");
    assignTrainingField(snapshot.tokenizer_character_coverage, trainConfig, "tokenizer_character_coverage");
    assignTrainingField(snapshot.tokenizer_vocab_score_multiplier, trainConfig, "tokenizer_vocab_score_multiplier");
    assignTrainingField(snapshot.tokenizer_subword_mining_max_bytes, trainConfig, "tokenizer_subword_mining_max_bytes");
    assignTrainingField(snapshot.tokenizer_prune_during_mining, trainConfig, "tokenizer_prune_during_mining");
    assignTrainingField(snapshot.tokenizer_enable_parallel_subword_mining, trainConfig, "tokenizer_enable_parallel_subword_mining");
    assignTrainingField(snapshot.tokenizer_add_bos, trainConfig, "tokenizer_add_bos");
    assignTrainingField(snapshot.tokenizer_add_eos, trainConfig, "tokenizer_add_eos");
    assignTrainingField(snapshot.tokenizer_enable_nfkc_normalization, trainConfig, "tokenizer_enable_nfkc_normalization");
    assignTrainingField(snapshot.tokenizer_enable_lowercasing, trainConfig, "tokenizer_enable_lowercasing");
    assignTrainingField(snapshot.tokenizer_enable_parallel_tokenization, trainConfig, "tokenizer_enable_parallel_tokenization");
    assignTrainingField(snapshot.tokenizer_enable_byte_fallback, trainConfig, "tokenizer_enable_byte_fallback");
    assignTrainingField(snapshot.tokenizer_save_text_vocab, trainConfig, "tokenizer_save_text_vocab");
    assignTrainingField(snapshot.tokenizer_model_type, trainConfig, "tokenizer_model_type");
    assignTrainingField(snapshot.tokenizer_unk_token, trainConfig, "tokenizer_unk_token");
    assignTrainingField(snapshot.tokenizer_pad_token, trainConfig, "tokenizer_pad_token");
    assignTrainingField(snapshot.tokenizer_bos_token, trainConfig, "tokenizer_bos_token");
    assignTrainingField(snapshot.tokenizer_eos_token, trainConfig, "tokenizer_eos_token");
    assignTrainingField(snapshot.tokenizer_expected_checksum, trainConfig, "tokenizer_expected_checksum");
    assignTrainingField(snapshot.tokenizer_enable_scratch_block_reasoning, trainConfig, "tokenizer_enable_scratch_block_reasoning");
    assignTrainingField(snapshot.tokenizer_detect_numbers, trainConfig, "tokenizer_detect_numbers");
    snapshot.tokenizer_special_tokens =
        getRequiredStringArray(trainConfig, "tokenizer_special_tokens", kTrainingConfigPath);
    assignTrainingField(snapshot.subprocess_tokenizer_only_mode, trainConfig, "subprocess_tokenizer_only_mode");
}

} // namespace detail

inline std::optional<AiConfigSnapshot> loadAiConfigSnapshot() {
    auto resolved_path = detail::resolveAiConfigPath();

    std::ifstream configFile(resolved_path);
    if (!configFile.is_open()) {
        throw std::runtime_error(
            "loadAiConfigSnapshot: could not open ai_config.json at: " + resolved_path.string());
    }

    AiConfigSnapshot snapshot;
    snapshot.config_path = resolved_path;
    try {
        configFile >> snapshot.document;
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "loadAiConfigSnapshot: failed to parse ai_config.json at: " + resolved_path.string() +
            ": " + e.what());
    }

    if (!snapshot.document.is_object()) {
        throw std::runtime_error("ai_config.json: root document must be an object");
    }

    detail::assignGrimTextPathFields(snapshot.document, snapshot);
    detail::assignSnapshotTrainingConfigFields(snapshot.document, snapshot);
    return snapshot;
}

inline std::string getRequiredGrimTextPath(const char* key) {
    auto snapshot = loadAiConfigSnapshot();
    if (!snapshot) {
        throw std::runtime_error("getRequiredGrimTextPath: failed to load ai_config.json");
    }

    if (key == nullptr || key[0] == '\0') {
        throw std::runtime_error("getRequiredGrimTextPath: key must be non-empty");
    }

    const std::string* path = detail::findGrimTextPathField(*snapshot, key);

    if (path != nullptr && !path->empty()) {
        return *path;
    }
    if (path != nullptr) {
        throw std::runtime_error(
            std::string("getRequiredGrimTextPath: ") + childJsonPath(detail::kGrimTextConfigPath, key) +
            " missing from ai_config.json");
    }

    throw std::runtime_error(std::string("getRequiredGrimTextPath: unknown paths.grim_text key: ") + key);
}

/**
 * @brief Get actual vocab size from vocab.bin file
 * 
 * Loads the tokenizer and returns its actual vocabulary size.
 * This is the authoritative source for vocab size, not ai_config.json.
 * 
 * @param vocabPath Path to vocab.bin file (optional, auto-detects if not provided)
 * @return Vocab size from tokenizer
 * @throws std::runtime_error on any missing config, unreadable file, or invalid vocab format
 */
inline uint32_t getActualVocabSize(const std::string& vocabPath = "") {
    std::string path = vocabPath;
    
    if (path.empty()) {
        path = getRequiredGrimTextPath("vocab");
    }
    
    // Load tokenizer to get actual vocab size
    // Note: This requires linking against the tokenizer library
    // For a header-only solution, we could parse the binary format directly
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("getActualVocabSize: failed to open vocab file: " + path);
    }
    
    // Read GMKT header format (version 2)
    char magic[4];
    file.read(magic, 4);
    if (!file || magic[0] != 'K' || magic[1] != 'T' || magic[2] != 'M' || magic[3] != 'G') {
        throw std::runtime_error("getActualVocabSize: invalid vocab file magic in: " + path);
    }
    
    uint16_t version;
    file.read(reinterpret_cast<char*>(&version), 2);
    if (!file) {
        throw std::runtime_error("getActualVocabSize: failed to read vocab version from: " + path);
    }
    
    if (version >= 2) {
        // Skip checksum (4 bytes)
        file.seekg(4, std::ios::cur);
        
        // Skip config vocab_size (4 bytes) and max_length (4 bytes)
        file.seekg(8, std::ios::cur);
        
        // Skip 3 bools (3 bytes)
        file.seekg(3, std::ios::cur);
    }
    
    // Read actual vocab size (number of tokens that follow)
    uint32_t vocab_size;
    file.read(reinterpret_cast<char*>(&vocab_size), 4);
    if (!file) {
        throw std::runtime_error("getActualVocabSize: failed to read vocab size from: " + path);
    }
    
    return vocab_size;
}

} // namespace Config
} // namespace GRIM
