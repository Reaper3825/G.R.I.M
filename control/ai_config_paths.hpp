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
// training.config tokenizer_* leaves → AiConfigSnapshot tokenizer_* fields
// training.config.subprocess_tokenizer_only_mode → AiConfigSnapshot subprocess_tokenizer_only_mode
// clear_merged_cache_on_merge → AiConfigSnapshot clear_merged_cache_on_merge
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
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <type_traits>
#include <map>
#include <vector>

// Strict layering: this file is the JSON reader and is allowed to be
// included by EXACTLY ONE place — HyperParameters_GPU.hpp. AiConfigSnapshot
// stays raw/flat; downstream typed config construction lives in HP_GPU.hpp.
// Including this header from anywhere else is a layering violation; HP_GPU.hpp
// is the single entry point.
#ifndef GRIM_HP_GPU_DEFINED_TRAINING_STRUCTS
#  error "ai_config_paths.hpp must only be included via HyperParameters_GPU.hpp. Include HyperParameters_GPU.hpp instead."
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
    bool has_grim_paths = false;
    bool has_training = false;
    bool has_tokenizer = false;
    bool has_subprocess = false;

    bool hasRequiredGrimTextPaths() const {
        return !grim_text_vocab.empty() && !grim_text_training_data.empty();
    }
};

inline std::optional<AiConfigSnapshot> loadAiConfigSnapshot(const std::string& configPath = "ai_config.json");

namespace GrimTextPathKey {
inline constexpr const char* Vocab = "vocab";
inline constexpr const char* Model = "model";
inline constexpr const char* TrainingData = "training_data";
inline constexpr const char* Checkpoints = "checkpoints";
inline constexpr const char* Collected = "collected";
inline constexpr const char* DirectoryCollection = "directory_collection";
inline constexpr const char* Verified = "verified";
inline constexpr const char* Logs = "logs";
inline constexpr const char* TrainingStatus = "training_status";
inline constexpr const char* CollectorLog = "collector_log";
inline constexpr const char* SourceConfig = "source_config";
inline constexpr const char* ModelStore = "model_store";
}

namespace detail {

inline constexpr const char* kRootConfigPath = "ai_config.json";
inline constexpr const char* kTrainingPath = "training";
inline constexpr const char* kTrainingConfigPath = "training.config";
inline constexpr const char* kGrimTextConfigPath = "paths.grim_text";

inline std::string trainingConfigPath(const char* relativePath) {
    return childJsonPath(kTrainingConfigPath, relativePath);
}

inline const nlohmann::json& requireTrainingObject(const nlohmann::json& config) {
    return requireJsonObjectField(config, "training", kRootConfigPath);
}

inline const nlohmann::json& requireTrainingConfigObject(const nlohmann::json& config) {
    return requireJsonObjectField(requireTrainingObject(config), "config", kTrainingPath);
}

inline std::filesystem::path resolveAiConfigPath(const std::string& configPath) {
    namespace fs = std::filesystem;
    fs::path candidate(configPath);
    if (!configPath.empty() && fs::exists(candidate)) {
        return fs::absolute(candidate);
    }

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
        "resolveAiConfigPath: ai_config.json not found at: " + configPath +
        " (also searched current directory and parent directories)");
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

inline std::string resolveRequiredGrimTextPath(const nlohmann::json& grimTextPaths,
                                               const char* key,
                                               const std::filesystem::path& grimRoot) {
    std::string pathStr = getRequiredJsonValue<std::string>(grimTextPaths, key, kGrimTextConfigPath);
    std::filesystem::path path(pathStr);
    if (path.is_relative()) {
        return (grimRoot / path).string();
    }
    return pathStr;
}

inline bool assignGrimTextPathFields(const nlohmann::json& config, AiConfigSnapshot& snapshot) {
    const auto& pathsNode = requireJsonObjectField(config, "paths", "ai_config.json");
    const auto& grimTextPaths = requireJsonObjectField(pathsNode, "grim_text", "paths");
    std::filesystem::path grimRoot = resolveGrimRoot();

    snapshot.grim_text_vocab = resolveRequiredGrimTextPath(grimTextPaths, GrimTextPathKey::Vocab, grimRoot);
    snapshot.grim_text_model = resolveRequiredGrimTextPath(grimTextPaths, GrimTextPathKey::Model, grimRoot);
    snapshot.grim_text_training_data = resolveRequiredGrimTextPath(grimTextPaths, GrimTextPathKey::TrainingData, grimRoot);
    snapshot.grim_text_checkpoints = resolveRequiredGrimTextPath(grimTextPaths, GrimTextPathKey::Checkpoints, grimRoot);
    snapshot.grim_text_collected = resolveRequiredGrimTextPath(grimTextPaths, GrimTextPathKey::Collected, grimRoot);
    snapshot.grim_text_directory_collection = resolveRequiredGrimTextPath(grimTextPaths, GrimTextPathKey::DirectoryCollection, grimRoot);
    snapshot.grim_text_verified = resolveRequiredGrimTextPath(grimTextPaths, GrimTextPathKey::Verified, grimRoot);
    snapshot.grim_text_logs = resolveRequiredGrimTextPath(grimTextPaths, GrimTextPathKey::Logs, grimRoot);
    snapshot.grim_text_training_status = resolveRequiredGrimTextPath(grimTextPaths, GrimTextPathKey::TrainingStatus, grimRoot);
    snapshot.grim_text_collector_log = resolveRequiredGrimTextPath(grimTextPaths, GrimTextPathKey::CollectorLog, grimRoot);
    snapshot.grim_text_source_config = resolveRequiredGrimTextPath(grimTextPaths, GrimTextPathKey::SourceConfig, grimRoot);
    snapshot.grim_text_model_store = resolveRequiredGrimTextPath(grimTextPaths, GrimTextPathKey::ModelStore, grimRoot);

    return true;
}

inline const std::string& requireGrimTextPath(const AiConfigSnapshot& snapshot,
                                              const char* key,
                                              const std::string& caller) {
    if (!snapshot.has_grim_paths) {
        throw std::runtime_error(
            caller + ": " + kGrimTextConfigPath + " missing from ai_config.json");
    }

    const std::string* path = nullptr;
    if (std::string(key) == GrimTextPathKey::Vocab) path = &snapshot.grim_text_vocab;
    if (std::string(key) == GrimTextPathKey::Model) path = &snapshot.grim_text_model;
    if (std::string(key) == GrimTextPathKey::TrainingData) path = &snapshot.grim_text_training_data;
    if (std::string(key) == GrimTextPathKey::Checkpoints) path = &snapshot.grim_text_checkpoints;
    if (std::string(key) == GrimTextPathKey::Collected) path = &snapshot.grim_text_collected;
    if (std::string(key) == GrimTextPathKey::DirectoryCollection) path = &snapshot.grim_text_directory_collection;
    if (std::string(key) == GrimTextPathKey::Verified) path = &snapshot.grim_text_verified;
    if (std::string(key) == GrimTextPathKey::Logs) path = &snapshot.grim_text_logs;
    if (std::string(key) == GrimTextPathKey::TrainingStatus) path = &snapshot.grim_text_training_status;
    if (std::string(key) == GrimTextPathKey::CollectorLog) path = &snapshot.grim_text_collector_log;
    if (std::string(key) == GrimTextPathKey::SourceConfig) path = &snapshot.grim_text_source_config;
    if (std::string(key) == GrimTextPathKey::ModelStore) path = &snapshot.grim_text_model_store;

    if (path != nullptr && !path->empty()) {
        return *path;
    }
    if (path != nullptr) {
        throw std::runtime_error(
            caller + ": " + childJsonPath(kGrimTextConfigPath, key) + " missing from ai_config.json");
    }

    throw std::runtime_error(std::string("requireGrimTextPath: unknown paths.grim_text key: ") + key);
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

inline void assignSnapshotFromDocument(const nlohmann::json& config, AiConfigSnapshot& snapshot) {
    if (!config.is_object()) {
        throw std::runtime_error("ai_config.json: root document must be an object");
    }

    const auto& training = requireTrainingObject(config);
    requireJsonField(training, "current_model_training", "training");
    requireJsonField(training, "current_curriculum", "training");
    const auto& trainConfig = requireTrainingConfigObject(config);

    requireJsonArrayField(trainConfig, "tokenizer_special_tokens", kTrainingConfigPath);
    requireJsonField(config, "clear_merged_cache_on_merge", kRootConfigPath);

    snapshot.has_grim_paths = assignGrimTextPathFields(config, snapshot);
    snapshot.clear_merged_cache_on_merge =
        getRequiredJsonValue<bool>(config, "clear_merged_cache_on_merge", kRootConfigPath);

    if (!trainConfig.is_object()) {
        throw std::runtime_error(std::string("ai_config.json: ") + kTrainingConfigPath + " must be an object");
    }

    AiConfigSnapshot& params = snapshot;
    params.current_model_training =
        getRequiredJsonValue<std::string>(training, "current_model_training", "training");
    params.current_curriculum =
        getRequiredJsonValue<std::string>(training, "current_curriculum", "training");
    snapshot.has_training = true;
    snapshot.has_tokenizer = true;
    snapshot.has_subprocess = true;

    assignTrainingField(params.epochs, trainConfig, "epochs");
    assignTrainingField(params.seed, trainConfig, "seed");
    assignTrainingField(params.batch_size, trainConfig, "batch_size");
    assignTrainingField(params.gradient_accumulation_steps, trainConfig, "gradient_accumulation_steps");
    assignTrainingField(params.batch_strategy, trainConfig, "batch_strategy");
    assignTrainingField(params.learning_rate, trainConfig, "learning_rate");
    assignTrainingField(params.weight_decay, trainConfig, "weight_decay");
    params.grad_clip_norm = getRequiredJsonValue<float>(trainConfig, "gradient_clip", "training.config");
    assignTrainingField(params.per_token_grad_scale, trainConfig, "per_token_grad_scale");
    assignTrainingField(params.force_rebuild_vocab, trainConfig, "force_rebuild_vocab");
    assignTrainingField(params.d_model, trainConfig, "d_model");
    assignTrainingField(params.num_layers, trainConfig, "num_layers");
    assignTrainingField(params.num_heads, trainConfig, "num_heads");
    assignTrainingField(params.num_kv_heads, trainConfig, "num_kv_heads");
    assignTrainingField(params.max_seq_len, trainConfig, "max_seq_len");
    assignTrainingField(params.tie_embeddings, trainConfig, "tie_embeddings");
    assignTrainingField(params.dropout_rate, trainConfig, "dropout_rate");
    assignTrainingField(params.sliding_window_stride, trainConfig, "sliding_window_stride");
    // min_seq_valid_tokens: derived in HyperParameters_GPU.hpp after raw parse.
    // architecture.min_seq_len_for_flash: derived in HyperParameters_GPU.hpp after raw parse.
    assignTrainingField(params.warmup_fraction, trainConfig, "warmup_fraction");
    params.cosine_decay_enabled =
        getRequiredJsonValue<bool>(trainConfig, "cosine_decay_enabled", kTrainingConfigPath);
    params.cosine_warm_restarts =
        getRequiredJsonValue<bool>(trainConfig, "cosine_warm_restarts", kTrainingConfigPath);
    // cosine_decay_min_lr: derived in HyperParameters_GPU.hpp after raw parse.
    assignTrainingField(params.log_interval, trainConfig, "log_interval");
    assignTrainingField(params.atom_stats_interval, trainConfig, "atom_stats_interval");
    assignTrainingField(params.atom_stats_max_seqs, trainConfig, "atom_stats_max_seqs");
    assignTrainingField(params.validation_interval, trainConfig, "validation_interval");
    assignTrainingField(params.checkpoint_interval, trainConfig, "checkpoint_interval");
    assignTrainingField(params.use_gpu, trainConfig, "use_gpu");
    assignTrainingField(params.use_flash_attention, trainConfig, "use_flash_attention");
    // architecture.min_seq_len_for_flash: derived from max_seq_len in HyperParameters_GPU.hpp.

    assignTrainingField(params.parameter_precision_embedding, trainConfig, "parameter_precision_embedding");
    assignTrainingField(params.parameter_precision_lm_head, trainConfig, "parameter_precision_lm_head");
    assignTrainingField(params.parameter_precision_attention, trainConfig, "parameter_precision_attention");
    assignTrainingField(params.parameter_precision_ffn, trainConfig, "parameter_precision_ffn");
    assignTrainingField(params.parameter_precision_rmsnorm, trainConfig, "parameter_precision_rmsnorm");
    assignTrainingField(params.parameter_precision_scratchblock, trainConfig, "parameter_precision_scratchblock");
    assignTrainingField(params.parameter_precision_mtp, trainConfig, "parameter_precision_mtp");
    assignTrainingField(params.parameter_precision_reasoning_head, trainConfig, "parameter_precision_reasoning_head");
    assignTrainingField(params.parameter_precision_execution_block, trainConfig, "parameter_precision_execution_block");
    assignTrainingField(params.parameter_precision_slot_selector, trainConfig, "parameter_precision_slot_selector");

    {
        params.use_rope = getRequiredJsonValue<bool>(trainConfig, "use_rope", kTrainingConfigPath);
        params.use_alibi = getRequiredJsonValue<bool>(trainConfig, "use_alibi", kTrainingConfigPath);
        params.rope_base_seq_len =
            getRequiredJsonValue<int>(trainConfig, "rope_base_seq_len", kTrainingConfigPath);
        params.alibi_min_locality_distance =
            getRequiredJsonValue<int>(trainConfig, "alibi_min_locality_distance", kTrainingConfigPath);
        params.alibi_slope_exponent =
            getRequiredJsonValue<float>(trainConfig, "alibi_slope_exponent", kTrainingConfigPath);
        params.alibi_max_bias =
            getRequiredJsonValue<float>(trainConfig, "alibi_max_bias", kTrainingConfigPath);
        params.rope_theta =
            getRequiredJsonValue<float>(trainConfig, "rope_theta", kTrainingConfigPath);
        params.rope_scaling =
            getRequiredJsonValue<float>(trainConfig, "rope_scaling", kTrainingConfigPath);
    }

    {
        params.soft_restart_enabled =
            getRequiredJsonValue<bool>(trainConfig, "soft_restart_enabled", kTrainingConfigPath);
        params.soft_restart_loss_increase_threshold =
            getRequiredJsonValue<float>(trainConfig, "soft_restart_loss_increase_threshold", kTrainingConfigPath);
        params.soft_restart_max_step_window =
            getRequiredJsonValue<int>(trainConfig, "soft_restart_max_step_window", kTrainingConfigPath);
        params.soft_restart_cooldown_steps =
            getRequiredJsonValue<int>(trainConfig, "soft_restart_cooldown_steps", kTrainingConfigPath);
    }

    {
        params.auto_stop_enabled =
            getRequiredJsonValue<bool>(trainConfig, "auto_stop_enabled", kTrainingConfigPath);
        params.auto_stop_plateau_patience =
            getRequiredJsonValue<int>(trainConfig, "auto_stop_plateau_patience", kTrainingConfigPath);
        params.auto_stop_plateau_min_delta =
            getRequiredJsonValue<float>(trainConfig, "auto_stop_plateau_min_delta", kTrainingConfigPath);
        params.auto_stop_high_loss_threshold =
            getRequiredJsonValue<float>(trainConfig, "auto_stop_high_loss_threshold", kTrainingConfigPath);
        params.auto_stop_high_loss_patience =
            getRequiredJsonValue<int>(trainConfig, "auto_stop_high_loss_patience", kTrainingConfigPath);
    }

    {
        params.single_batch_overfit_enabled =
            getRequiredJsonValue<bool>(trainConfig, "single_batch_overfit_enabled", kTrainingConfigPath);
        params.single_batch_overfit_max_steps =
            getRequiredJsonValue<int>(trainConfig, "single_batch_overfit_max_steps", kTrainingConfigPath);
    }

    {
        params.shuffle_train_enabled =
            getRequiredJsonValue<bool>(trainConfig, "shuffle_train_enabled", kTrainingConfigPath);
        params.shuffle_train_epochs =
            getRequiredJsonValue<int>(trainConfig, "shuffle_train_epochs", kTrainingConfigPath);
    }

    {
        params.telemetry_control_enabled =
            getRequiredJsonValue<bool>(trainConfig, "telemetry_control_enabled", kTrainingConfigPath);
        params.telemetry_spike_mild_threshold =
            getRequiredJsonValue<float>(trainConfig, "telemetry_spike_mild_threshold", kTrainingConfigPath);
        params.telemetry_spike_moderate_threshold =
            getRequiredJsonValue<float>(trainConfig, "telemetry_spike_moderate_threshold", kTrainingConfigPath);
        params.telemetry_spike_severe_threshold =
            getRequiredJsonValue<float>(trainConfig, "telemetry_spike_severe_threshold", kTrainingConfigPath);
        params.telemetry_moderate_grad_scale =
            getRequiredJsonValue<float>(trainConfig, "telemetry_moderate_grad_scale", kTrainingConfigPath);
        params.telemetry_moderate_cooldown_extension =
            getRequiredJsonValue<int>(trainConfig, "telemetry_moderate_cooldown_extension", kTrainingConfigPath);
        params.telemetry_min_grad_for_nonzero_loss =
            getRequiredJsonValue<float>(trainConfig, "telemetry_min_grad_for_nonzero_loss", kTrainingConfigPath);
        params.telemetry_loss_threshold_for_grad_check =
            getRequiredJsonValue<float>(trainConfig, "telemetry_loss_threshold_for_grad_check", kTrainingConfigPath);
        params.telemetry_max_consecutive_zero_grad_steps =
            getRequiredJsonValue<int>(trainConfig, "telemetry_max_consecutive_zero_grad_steps", kTrainingConfigPath);
        params.telemetry_seq_len_regime_change_threshold =
            getRequiredJsonValue<float>(trainConfig, "telemetry_seq_len_regime_change_threshold", kTrainingConfigPath);
        params.telemetry_regime_change_suppression_steps =
            getRequiredJsonValue<int>(trainConfig, "telemetry_regime_change_suppression_steps", kTrainingConfigPath);
        params.telemetry_volatility_damping_threshold =
            getRequiredJsonValue<float>(trainConfig, "telemetry_volatility_damping_threshold", kTrainingConfigPath);
        params.telemetry_max_volatility_damping =
            getRequiredJsonValue<float>(trainConfig, "telemetry_max_volatility_damping", kTrainingConfigPath);
        params.telemetry_gradient_decay_threshold =
            getRequiredJsonValue<float>(trainConfig, "telemetry_gradient_decay_threshold", kTrainingConfigPath);
        params.telemetry_max_decay_boost =
            getRequiredJsonValue<float>(trainConfig, "telemetry_max_decay_boost", kTrainingConfigPath);
        params.telemetry_progress_boost_threshold =
            getRequiredJsonValue<float>(trainConfig, "telemetry_progress_boost_threshold", kTrainingConfigPath);
        params.telemetry_max_progress_boost =
            getRequiredJsonValue<float>(trainConfig, "telemetry_max_progress_boost", kTrainingConfigPath);
        params.telemetry_outlier_frequency_trigger =
            getRequiredJsonValue<float>(trainConfig, "telemetry_outlier_frequency_trigger", kTrainingConfigPath);
        params.telemetry_outlier_persistence_trigger =
            getRequiredJsonValue<float>(trainConfig, "telemetry_outlier_persistence_trigger", kTrainingConfigPath);
        params.telemetry_anchor_drift_sigma_multiplier =
            getRequiredJsonValue<float>(trainConfig, "telemetry_anchor_drift_sigma_multiplier", kTrainingConfigPath);
        params.telemetry_soft_restart_cooldown_steps =
            getRequiredJsonValue<int>(trainConfig, "telemetry_soft_restart_cooldown_steps", kTrainingConfigPath);
        params.telemetry_baseline_stabilization_steps =
            getRequiredJsonValue<int>(trainConfig, "telemetry_baseline_stabilization_steps", kTrainingConfigPath);
        params.telemetry_verbose_logging =
            getRequiredJsonValue<bool>(trainConfig, "telemetry_verbose_logging", kTrainingConfigPath);
        params.telemetry_fail_loud_on_accumulation_bug =
            getRequiredJsonValue<bool>(trainConfig, "telemetry_fail_loud_on_accumulation_bug", kTrainingConfigPath);
        params.telemetry_plateau_noise_enabled =
            getRequiredJsonValue<bool>(trainConfig, "telemetry_plateau_noise_enabled", kTrainingConfigPath);
        params.telemetry_plateau_noise_patience =
            getRequiredJsonValue<int>(trainConfig, "telemetry_plateau_noise_patience", kTrainingConfigPath);
        params.telemetry_plateau_noise_variance_threshold =
            getRequiredJsonValue<float>(trainConfig, "telemetry_plateau_noise_variance_threshold", kTrainingConfigPath);
        params.telemetry_plateau_noise_std =
            getRequiredJsonValue<float>(trainConfig, "telemetry_plateau_noise_std", kTrainingConfigPath);
        params.telemetry_plateau_noise_proportional =
            getRequiredJsonValue<bool>(trainConfig, "telemetry_plateau_noise_proportional", kTrainingConfigPath);
        params.telemetry_plateau_noise_cooldown =
            getRequiredJsonValue<int>(trainConfig, "telemetry_plateau_noise_cooldown", kTrainingConfigPath);
        params.telemetry_plateau_noise_max_per_epoch =
            getRequiredJsonValue<int>(trainConfig, "telemetry_plateau_noise_max_per_epoch", kTrainingConfigPath);
        params.telemetry_lattice_num_levels =
            getRequiredJsonValue<int>(trainConfig, "telemetry_lattice_num_levels", kTrainingConfigPath);
        params.telemetry_lattice_num_streams =
            getRequiredJsonValue<int>(trainConfig, "telemetry_lattice_num_streams", kTrainingConfigPath);
        params.telemetry_lattice_beta_mu =
            getRequiredJsonValue<float>(trainConfig, "telemetry_lattice_beta_mu", kTrainingConfigPath);
        params.telemetry_lattice_beta_a =
            getRequiredJsonValue<float>(trainConfig, "telemetry_lattice_beta_a", kTrainingConfigPath);
        params.telemetry_lattice_beta_delta =
            getRequiredJsonValue<float>(trainConfig, "telemetry_lattice_beta_delta", kTrainingConfigPath);
        params.telemetry_lattice_beta_r =
            getRequiredJsonValue<float>(trainConfig, "telemetry_lattice_beta_r", kTrainingConfigPath);
        params.telemetry_lattice_beta_run =
            getRequiredJsonValue<float>(trainConfig, "telemetry_lattice_beta_run", kTrainingConfigPath);
        params.telemetry_lattice_beta_v =
            getRequiredJsonValue<float>(trainConfig, "telemetry_lattice_beta_v", kTrainingConfigPath);
        params.telemetry_lattice_k_out0 =
            getRequiredJsonValue<float>(trainConfig, "telemetry_lattice_k_out0", kTrainingConfigPath);
        params.telemetry_lattice_alpha_v =
            getRequiredJsonValue<float>(trainConfig, "telemetry_lattice_alpha_v", kTrainingConfigPath);
        params.telemetry_lattice_epsilon =
            getRequiredJsonValue<float>(trainConfig, "telemetry_lattice_epsilon", kTrainingConfigPath);
        params.telemetry_lattice_strict_mode =
            getRequiredJsonValue<bool>(trainConfig, "telemetry_lattice_strict_mode", kTrainingConfigPath);
    }

    params.logging_default_level =
        getRequiredJsonValue<std::string>(trainConfig, "logging_default_level", kTrainingConfigPath);
    params.logging_equation_csv_enabled =
        getRequiredJsonValue<bool>(trainConfig, "logging_equation_csv_enabled", kTrainingConfigPath);
    params.logging_stderr_enabled =
        getRequiredJsonValue<bool>(trainConfig, "logging_stderr_enabled", kTrainingConfigPath);
    params.logging_initial_capacity =
        getRequiredJsonValue<size_t>(trainConfig, "logging_initial_capacity", kTrainingConfigPath);
    params.logging_group_overrides =
        getRequiredStringMap(trainConfig, "logging_group_overrides", kTrainingConfigPath);

    params.log_recorder_enabled =
        getRequiredJsonValue<bool>(trainConfig, "log_recorder_enabled", kTrainingConfigPath);
    params.log_recorder_default_level =
        getRequiredJsonValue<std::string>(trainConfig, "log_recorder_default_level", kTrainingConfigPath);
    params.log_recorder_modules =
        getRequiredStringMap(trainConfig, "log_recorder_modules", kTrainingConfigPath);

    params.log_recorder_layer_embedding = getRequiredJsonValue<bool>(trainConfig, "log_recorder_layer_embedding", kTrainingConfigPath);
    params.log_recorder_layer_rms_norm = getRequiredJsonValue<bool>(trainConfig, "log_recorder_layer_rms_norm", kTrainingConfigPath);
    params.log_recorder_layer_attention = getRequiredJsonValue<bool>(trainConfig, "log_recorder_layer_attention", kTrainingConfigPath);
    params.log_recorder_layer_feed_forward = getRequiredJsonValue<bool>(trainConfig, "log_recorder_layer_feed_forward", kTrainingConfigPath);
    params.log_recorder_layer_residual = getRequiredJsonValue<bool>(trainConfig, "log_recorder_layer_residual", kTrainingConfigPath);
    params.log_recorder_layer_encoding = getRequiredJsonValue<bool>(trainConfig, "log_recorder_layer_encoding", kTrainingConfigPath);
    params.log_recorder_layer_serialization = getRequiredJsonValue<bool>(trainConfig, "log_recorder_layer_serialization", kTrainingConfigPath);
    params.log_recorder_layer_execution_block = getRequiredJsonValue<bool>(trainConfig, "log_recorder_layer_execution_block", kTrainingConfigPath);

    {
        params.loss_label_smoothing_enabled =
            getRequiredJsonValue<bool>(trainConfig, "loss_label_smoothing_enabled", kTrainingConfigPath);
        params.loss_label_smoothing_epsilon =
            getRequiredJsonValue<float>(trainConfig, "loss_label_smoothing_epsilon", kTrainingConfigPath);
        params.loss_focal_enabled =
            getRequiredJsonValue<bool>(trainConfig, "loss_focal_enabled", kTrainingConfigPath);
        params.loss_focal_gamma =
            getRequiredJsonValue<float>(trainConfig, "loss_focal_gamma", kTrainingConfigPath);
        params.loss_focal_alpha =
            getRequiredJsonValue<float>(trainConfig, "loss_focal_alpha", kTrainingConfigPath);
        params.loss_entropy_reg_enabled =
            getRequiredJsonValue<bool>(trainConfig, "loss_entropy_reg_enabled", kTrainingConfigPath);
        params.loss_entropy_reg_lambda =
            getRequiredJsonValue<float>(trainConfig, "loss_entropy_reg_lambda", kTrainingConfigPath);
        params.loss_class_balanced_enabled =
            getRequiredJsonValue<bool>(trainConfig, "loss_class_balanced_enabled", kTrainingConfigPath);
        params.loss_class_balanced_beta =
            getRequiredJsonValue<float>(trainConfig, "loss_class_balanced_beta", kTrainingConfigPath);
        params.loss_preference_enabled =
            getRequiredJsonValue<bool>(trainConfig, "loss_preference_enabled", kTrainingConfigPath);
        params.loss_preference_beta =
            getRequiredJsonValue<float>(trainConfig, "loss_preference_beta", kTrainingConfigPath);
        params.loss_distillation_enabled =
            getRequiredJsonValue<bool>(trainConfig, "loss_distillation_enabled", kTrainingConfigPath);
        params.loss_distillation_temperature =
            getRequiredJsonValue<float>(trainConfig, "loss_distillation_temperature", kTrainingConfigPath);
        params.loss_distillation_lambda =
            getRequiredJsonValue<float>(trainConfig, "loss_distillation_lambda", kTrainingConfigPath);
        params.loss_masking_enabled =
            getRequiredJsonValue<bool>(trainConfig, "loss_masking_enabled", kTrainingConfigPath);
        params.loss_masking_tag =
            getRequiredJsonValue<std::string>(trainConfig, "loss_masking_tag", kTrainingConfigPath);
    }

    {
        params.lm_head_centering_enabled =
            getRequiredJsonValue<bool>(trainConfig, "lm_head_centering_enabled", kTrainingConfigPath);
        params.lm_head_center_hidden_states =
            getRequiredJsonValue<bool>(trainConfig, "lm_head_center_hidden_states", kTrainingConfigPath);
        params.freeze_learned_rms_gammas =
            getRequiredJsonValue<bool>(trainConfig, "freeze_learned_rms_gammas", kTrainingConfigPath);
        params.center_logits =
            getRequiredJsonValue<bool>(trainConfig, "center_logits", kTrainingConfigPath);
        params.center_encoder_residuals =
            getRequiredJsonValue<bool>(trainConfig, "center_encoder_residuals", kTrainingConfigPath);
        params.project_out_pc1 =
            getRequiredJsonValue<bool>(trainConfig, "project_out_pc1", kTrainingConfigPath);
        params.pc1_power_iters =
            getRequiredJsonValue<int>(trainConfig, "pc1_power_iters", kTrainingConfigPath);
    }

    {
        params.use_layer_scale =
            getRequiredJsonValue<bool>(trainConfig, "use_layer_scale", kTrainingConfigPath);
        params.layer_scale_init =
            getRequiredJsonValue<float>(trainConfig, "layer_scale_init", kTrainingConfigPath);
    }

    {
        params.qk_norm_enabled =
            getRequiredJsonValue<bool>(trainConfig, "qk_norm_enabled", kTrainingConfigPath);
    }

    {
        params.hardcoded_hidden_states_enabled =
            getRequiredJsonValue<bool>(trainConfig, "hardcoded_hidden_states_enabled", kTrainingConfigPath);
        params.hardcoded_hidden_states_pattern =
            getRequiredJsonValue<std::string>(trainConfig, "hardcoded_hidden_states_pattern", kTrainingConfigPath);
        params.hardcoded_log_every_n_batches =
            getRequiredJsonValue<int>(trainConfig, "hardcoded_log_every_n_batches", kTrainingConfigPath);
    }

    {
        params.embedding_freeze_enabled =
            getRequiredJsonValue<bool>(trainConfig, "embedding_freeze_enabled", kTrainingConfigPath);
        params.embedding_freeze_after_step =
            getRequiredJsonValue<int>(trainConfig, "embedding_freeze_after_step", kTrainingConfigPath);
    }

    {
        params.optimizer_kind =
            getRequiredJsonValue<std::string>(trainConfig, "optimizer_kind", kTrainingConfigPath);
        params.optimizer_beta1 =
            getRequiredJsonValue<float>(trainConfig, "optimizer_beta1", kTrainingConfigPath);
        params.optimizer_beta2 =
            getRequiredJsonValue<float>(trainConfig, "optimizer_beta2", kTrainingConfigPath);
        params.optimizer_epsilon =
            getRequiredJsonValue<float>(trainConfig, "optimizer_epsilon", kTrainingConfigPath);
    }

    assignTrainingField(params.stability_overrides_enabled, trainConfig, "stability_overrides_enabled");
    assignTrainingField(params.stability_override_batch_size, trainConfig, "stability_override_batch_size");
    assignTrainingField(params.stability_override_max_seq_len, trainConfig, "stability_override_max_seq_len");
    assignTrainingField(params.stability_override_clip_per_token, trainConfig, "stability_override_clip_per_token");

    assignTrainingField(params.scratch_blocks_enabled, trainConfig, "scratch_blocks_enabled");
    assignTrainingField(params.scratch_num_blocks, trainConfig, "scratch_num_blocks");
    assignTrainingField(params.scratch_write_combined, trainConfig, "scratch_write_combined");
    assignTrainingField(params.use_scratch_block, trainConfig, "use_scratch_block");
    assignTrainingField(params.scratch_block_atom_embedding_dim, trainConfig, "scratch_block_atom_embedding_dim");
    assignTrainingField(params.scratch_block_max_atoms, trainConfig, "scratch_block_max_atoms");
    assignTrainingField(params.scratch_block_atom_scale, trainConfig, "scratch_block_atom_scale");

    assignTrainingField(params.execution_block_enabled, trainConfig, "execution_block_enabled");
    assignTrainingField(params.scratch_block_execution_first_type_only, trainConfig, "execution_block_execution_first_type_only");
    assignTrainingField(params.execution_block_debug_mode, trainConfig, "execution_block_debug_mode");
    assignTrainingField(params.step_y_overrides_x, trainConfig, "execution_block_step_y_overrides_x");
    assignTrainingField(params.structured_ce_enabled, trainConfig, "execution_block_structured_ce_enabled");
    assignTrainingField(params.selector_enabled, trainConfig, "selector_enabled");
    assignTrainingField(params.execution_block_layer, trainConfig, "execution_block_layer");
    assignTrainingField(params.execution_block_num_ops, trainConfig, "execution_block_num_ops");
    assignTrainingField(params.execution_block_num_slots, trainConfig, "execution_block_num_slots");
    assignTrainingField(params.execution_block_num_scratch_slots, trainConfig, "execution_block_num_scratch_slots");
    assignTrainingField(params.execution_block_num_steps, trainConfig, "execution_block_num_steps");
    assignTrainingField(params.execution_block_value_decode_input_dim, trainConfig, "execution_block_value_decode_input_dim");
    assignTrainingField(params.execution_block_value_decode_hidden_dim, trainConfig, "execution_block_value_decode_hidden_dim");
    assignTrainingField(params.execution_block_d_type, trainConfig, "execution_block_d_type");
    assignTrainingField(params.execution_block_cross_attn_topk, trainConfig, "execution_block_cross_attn_topk");
    assignTrainingField(params.execution_block_result_slot_mode, trainConfig, "execution_block_result_slot_mode");
    assignTrainingField(params.execution_block_result_slot_index, trainConfig, "execution_block_result_slot_index");
    assignTrainingField(params.execution_block_temp_schedule, trainConfig, "execution_block_temp_schedule");
    assignTrainingField(params.decode_time_slot_feature_dim, trainConfig, "selector_d_slot_features");
    assignTrainingField(params.selector_d_selector, trainConfig, "selector_d_selector");
    assignTrainingField(params.execution_block_usage_decay, trainConfig, "execution_block_usage_decay");
    assignTrainingField(params.execution_block_inject_gate_temp, trainConfig, "execution_block_inject_gate_temp");
    assignTrainingField(params.execution_block_entropy_collapse_threshold, trainConfig, "execution_block_entropy_collapse_threshold");
    assignTrainingField(params.execution_block_write_collapse_threshold, trainConfig, "execution_block_write_collapse_threshold");
    assignTrainingField(params.execution_block_magnitude_limit, trainConfig, "execution_block_magnitude_limit");
    assignTrainingField(params.execution_block_diversity_kappa, trainConfig, "execution_block_diversity_kappa");
    assignTrainingField(params.execution_block_temp_start, trainConfig, "execution_block_temp_start");
    assignTrainingField(params.execution_block_temp_end, trainConfig, "execution_block_temp_end");
    assignTrainingField(params.execution_block_entropy_weight, trainConfig, "execution_block_entropy_weight");
    assignTrainingField(params.step_x_multiplier, trainConfig, "execution_block_step_x_multiplier");
    assignTrainingField(params.step_y_multiplier, trainConfig, "execution_block_step_y_multiplier");
    assignTrainingField(params.entropy_aux_weight, trainConfig, "execution_block_entropy_aux_weight");
    assignTrainingField(params.value_match_epsilon, trainConfig, "execution_block_value_match_epsilon");
    assignTrainingField(params.final_slot_consistency_weight, trainConfig, "execution_block_final_slot_consistency_weight");
    assignTrainingField(params.execution_block_transition_hard_threshold, trainConfig, "execution_block_transition_hard_threshold");
    assignTrainingField(params.execution_block_causal_w1_transition, trainConfig, "execution_block_causal_w1_transition");
    assignTrainingField(params.div_invalid_penalty_weight, trainConfig, "execution_block_div_invalid_penalty_weight");
    assignTrainingField(params.div_magnitude_penalty_weight, trainConfig, "execution_block_div_magnitude_penalty_weight");
    assignTrainingField(params.arg_reinforce_weight, trainConfig, "execution_block_arg_reinforce_weight");
    assignTrainingField(params.arg_reinforce_baseline_decay, trainConfig, "execution_block_arg_reinforce_baseline_decay");
    assignTrainingField(params.structured_ce_weight, trainConfig, "execution_block_structured_ce_weight");
    assignTrainingField(params.selector_selection_margin, trainConfig, "selector_selection_margin");
    assignTrainingField(params.selector_supervision_weight, trainConfig, "selector_supervision_weight");

    assignTrainingField(params.single_stream_mode, trainConfig, "single_stream_mode");
    assignTrainingField(params.disable_async_frees, trainConfig, "disable_async_frees");
    assignTrainingField(params.synchronize_after_kernels, trainConfig, "synchronize_after_kernels");

    assignTrainingField(params.mtp_enabled, trainConfig, "mtp_enabled");
    assignTrainingField(params.mtp_log_ratio_monitor, trainConfig, "mtp_log_ratio_monitor");
    assignTrainingField(params.mtp_k, trainConfig, "mtp_k");
    assignTrainingField(params.mtp_alpha, trainConfig, "mtp_alpha");

    assignTrainingField(params.prediction_comparison_enabled, trainConfig, "prediction_comparison_enabled");
    assignTrainingField(params.prediction_comparison_interval, trainConfig, "prediction_comparison_interval");
    assignTrainingField(params.prediction_comparison_top_k, trainConfig, "prediction_comparison_top_k");
    assignTrainingField(params.prediction_comparison_max_positions, trainConfig, "prediction_comparison_max_positions");
    assignTrainingField(params.prediction_comparison_log_path, trainConfig, "prediction_comparison_log_path");

    assignTrainingField(params.logit_update_trace_enabled, trainConfig, "logit_update_trace_enabled");
    assignTrainingField(params.logit_update_trace_interval, trainConfig, "logit_update_trace_interval");
    assignTrainingField(params.attention_diag_enabled, trainConfig, "attention_diag_enabled");
    assignTrainingField(params.attention_diag_layer, trainConfig, "attention_diag_layer");
    assignTrainingField(params.attention_diag_head, trainConfig, "attention_diag_head");

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

inline std::optional<AiConfigSnapshot> loadAiConfigSnapshot(const std::string& configPath) {
    auto resolved_path = detail::resolveAiConfigPath(configPath);

    std::ifstream configFile(resolved_path);
    if (!configFile.is_open()) {
        throw std::runtime_error(
            "loadAiConfigSnapshot: could not open ai_config.json at: " + resolved_path.string());
    }

    nlohmann::json config;
    configFile >> config;

    AiConfigSnapshot snapshot;
    snapshot.config_path = resolved_path;
    snapshot.document = std::move(config);
    detail::assignSnapshotFromDocument(snapshot.document, snapshot);
    return snapshot;
}

inline std::string getRequiredGrimTextPath(const char* key) {
    auto snapshot = loadAiConfigSnapshot();
    if (!snapshot) {
        throw std::runtime_error("getRequiredGrimTextPath: failed to load ai_config.json");
    }
    return detail::requireGrimTextPath(*snapshot, key, "getRequiredGrimTextPath");
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
