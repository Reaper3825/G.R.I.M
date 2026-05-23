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
// training.config            → TrainingHyperparameters (incl. generation, execution_block, scratch_blocks, …)
// tokenizer                  → AiConfigSnapshot tokenizer_* fields
// data_collection            → AiConfigSnapshot data_collection_* fields (GRIM process config, not training/inference startup)
//
// RULE: All runtime defaults in TrainingHyperparameters MUST
// be authored in ai_config.json or derived in HyperParameters_GPU.hpp.
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
#include <iostream>
#include <optional>
#include <type_traits>
#include <map>
#include <set>
#include <sstream>
#include <vector>

// Strict layering: this file is the JSON reader and is allowed to be
// included by EXACTLY ONE place — HyperParameters_GPU.hpp, after it has
// defined LogRecorderConfig / TapeLogConfig / TrainingHyperparameters in
// namespace GRIM::Config. Including this header from anywhere else is a
// layering violation; HP_GPU.hpp is the single entry point.
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
                                             const char* parentPath) {
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
                                                    const char* parentPath) {
    const auto& field = requireJsonField(node, key, parentPath);
    if (!field.is_object()) {
        throw std::runtime_error(
            std::string("ai_config.json: field '") + parentPath + "." + key + "' must be an object");
    }
    return field;
}

inline const nlohmann::json& requireJsonArrayField(const nlohmann::json& node,
                                                   const char* key,
                                                   const char* parentPath) {
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
                                                    const char* parentPath) {
    const auto& field = requireJsonField(node, key, parentPath);
    try {
        return field.get<std::decay_t<FieldType>>();
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("ai_config.json: invalid value for field '") + parentPath + "." + key + "': " +
            e.what());
    }
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
    TrainingHyperparameters hyperparameters;
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
    bool data_collection_clear_merged_cache_on_merge = false;
    int data_collection_max_new_entries_per_run = 5000;
    // subprocess.tokenizer.only_mode
    // When true, train_gpu spawns the train_tokenizer subprocess, waits for
    // completion, and exits cleanly without entering Phase 1. When false,
    // tokenizer training runs as a normal pre-Phase-1 step and training
    // continues on success.
    bool subprocess_tokenizer_only_mode = false;
    bool has_grim_paths = false;
    bool has_training = false;
    bool has_tokenizer = false;
    bool has_data_collection = false;
    bool has_subprocess = false;

    template <typename FieldType>
    void assignSnapshotField(FieldType& field, const nlohmann::json& node, const char* key) {
        field = getRequiredJsonValue<FieldType>(node, key, "tokenizer");
    }

    void assignTokenizerFields(const nlohmann::json& tok) {
        has_tokenizer = true;

        assignSnapshotField(tokenizer_vocab_size, tok, "vocab_size");
        assignSnapshotField(tokenizer_max_vocab_size, tok, "max_vocab_size");
        assignSnapshotField(tokenizer_max_length, tok, "max_length");
        assignSnapshotField(tokenizer_character_coverage, tok, "character_coverage");
        assignSnapshotField(tokenizer_min_cleaned_text_length, tok, "min_cleaned_text_length");
        assignSnapshotField(tokenizer_min_subword_freq, tok, "min_subword_freq");
        assignSnapshotField(tokenizer_prune_during_mining, tok, "prune_during_mining");
        assignSnapshotField(tokenizer_enable_parallel_subword_mining, tok, "enable_parallel_subword_mining");
        assignSnapshotField(tokenizer_subword_mining_workers, tok, "subword_mining_workers");
        assignSnapshotField(tokenizer_subword_mining_max_bytes, tok, "subword_mining_max_bytes");
        assignSnapshotField(tokenizer_model_type, tok, "model_type");
        assignSnapshotField(tokenizer_add_bos, tok, "add_bos");
        assignSnapshotField(tokenizer_add_eos, tok, "add_eos");
        assignSnapshotField(tokenizer_unk_token, tok, "unk_token");
        assignSnapshotField(tokenizer_pad_token, tok, "pad_token");
        assignSnapshotField(tokenizer_bos_token, tok, "bos_token");
        assignSnapshotField(tokenizer_eos_token, tok, "eos_token");
        assignSnapshotField(tokenizer_enable_nfkc_normalization, tok, "enable_nfkc_normalization");
        assignSnapshotField(tokenizer_enable_lowercasing, tok, "enable_lowercasing");
        assignSnapshotField(tokenizer_enable_parallel_tokenization, tok, "enable_parallel_tokenization");
        assignSnapshotField(tokenizer_parallel_threshold, tok, "parallel_threshold");
        assignSnapshotField(tokenizer_enable_byte_fallback, tok, "enable_byte_fallback");
        assignSnapshotField(tokenizer_expected_checksum, tok, "expected_checksum");
        assignSnapshotField(tokenizer_save_text_vocab, tok, "save_text_vocab");
        assignSnapshotField(tokenizer_vocab_score_multiplier, tok, "vocab_score_multiplier");

        const auto& sbr = requireJsonObjectField(tok, "scratch_block_reasoning", "tokenizer");
        hyperparameters.tokenizer_enable_scratch_block_reasoning =
            getRequiredJsonValue<bool>(sbr, "enabled", "tokenizer.scratch_block_reasoning");
        hyperparameters.tokenizer_detect_numbers =
            getRequiredJsonValue<bool>(sbr, "detect_numbers", "tokenizer.scratch_block_reasoning");

        const auto& special_tokens = requireJsonArrayField(tok, "special_tokens", "tokenizer");
        tokenizer_special_tokens.clear();
        for (const auto& token : special_tokens) {
            if (!token.is_string()) {
                throw std::runtime_error(
                    "ai_config.json: tokenizer.special_tokens entries must all be strings");
            }
            tokenizer_special_tokens.push_back(token.get<std::string>());
        }
    }

    bool hasRequiredGrimTextPaths() const {
        return !grim_text_vocab.empty() && !grim_text_training_data.empty();
    }

    void printGrimTextPaths() const {
        std::cout << "[Config] GRIM-text paths from ai_config.json:" << std::endl;
        std::cout << "  vocab: " << grim_text_vocab << std::endl;
        std::cout << "  model: " << grim_text_model << std::endl;
        std::cout << "  training_data: " << grim_text_training_data << std::endl;
        std::cout << "  checkpoints: " << grim_text_checkpoints << std::endl;
        std::cout << "  collected: " << grim_text_collected << std::endl;
        std::cout << "  directory_collection: " << grim_text_directory_collection << std::endl;
        std::cout << "  verified: " << grim_text_verified << std::endl;
        std::cout << "  logs: " << grim_text_logs << std::endl;
        std::cout << "  training_status: " << grim_text_training_status << std::endl;
        std::cout << "  collector_log: " << grim_text_collector_log << std::endl;
        std::cout << "  source_config: " << grim_text_source_config << std::endl;
        std::cout << "  model_store: " << grim_text_model_store << std::endl;
    }
};

inline std::optional<AiConfigSnapshot> loadAiConfigSnapshot(const std::string& configPath = "ai_config.json");
inline void validateAiConfigDocument(const nlohmann::json& config);

namespace detail {

inline constexpr const char* kRootConfigPath = "ai_config.json";
inline constexpr const char* kTrainingPath = "training";
inline constexpr const char* kTrainingConfigPath = "training.config";
inline constexpr const char* kTrainingGenerationPath = "training.config.generation";

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

inline bool assignGrimTextPathFields(const nlohmann::json& config, AiConfigSnapshot& snapshot) {
    const auto& pathsNode = requireJsonObjectField(config, "paths", "ai_config.json");
    const auto& grimTextPaths = requireJsonObjectField(pathsNode, "grim_text", "paths");

    // Get GRIM root for resolving relative paths
    std::filesystem::path grimRoot = resolveGrimRoot();
    
    auto assignRequiredPath = [&](const char* key, std::string& field) {
        std::string pathStr = getRequiredJsonValue<std::string>(grimTextPaths, key, "paths.grim_text");
        std::filesystem::path path(pathStr);

        // If path is relative, resolve it relative to GRIM root
        if (path.is_relative()) {
            field = (grimRoot / path).string();
        } else {
            // Path is already absolute, use as-is
            field = pathStr;
        }
    };

    assignRequiredPath("vocab", snapshot.grim_text_vocab);
    assignRequiredPath("model", snapshot.grim_text_model);
    assignRequiredPath("training_data", snapshot.grim_text_training_data);
    assignRequiredPath("checkpoints", snapshot.grim_text_checkpoints);
    assignRequiredPath("collected", snapshot.grim_text_collected);
    assignRequiredPath("directory_collection", snapshot.grim_text_directory_collection);
    assignRequiredPath("verified", snapshot.grim_text_verified);
    assignRequiredPath("logs", snapshot.grim_text_logs);
    assignRequiredPath("training_status", snapshot.grim_text_training_status);
    assignRequiredPath("collector_log", snapshot.grim_text_collector_log);
    assignRequiredPath("source_config", snapshot.grim_text_source_config);
    assignRequiredPath("model_store", snapshot.grim_text_model_store);

    return true;
}

template <typename FieldType>
inline void assignTrainingField(FieldType& field, const nlohmann::json& node, const char* key) {
    field = getRequiredJsonValue<FieldType>(node, key, kTrainingConfigPath);
}

/**
 * @brief Check if a nested JSON path exists
 * @param json The root JSON object
 * @param path Dot-separated path (e.g., "loss.focal.enabled")
 * @return true if path exists
 */
inline bool jsonPathExists(const nlohmann::json& json, const std::string& path) {
    const nlohmann::json* current = &json;
    std::istringstream stream(path);
    std::string segment;
    
    while (std::getline(stream, segment, '.')) {
        if (!current->is_object() || !current->contains(segment)) {
            return false;
        }
        current = &(*current)[segment];
    }
    return true;
}

/**
 * @brief Validate required (base+enable) fields exist in training.config JSON
 * @param trainConfig The training.config JSON object
 * @throws std::runtime_error listing all missing required fields
 *
 * Rule 20: authored runtime parameters and feature enables MUST be explicitly set.
 * Formula-derived values are computed after parsing; there are no runtime defaults here.
 */
inline void appendMissingTrainingConfigPaths(const nlohmann::json& trainConfig,
                                             std::vector<std::string>& missing) {
    static const std::vector<std::string> REQUIRED = {
        // Core training
        "epochs", "seed", "batch_size", "gradient_accumulation_steps",
        "batch_strategy", "learning_rate", "weight_decay",
        "per_token_grad_scale", "warmup_fraction", "force_rebuild_vocab", "max_seq_len",
        "d_model", "num_layers", "num_heads", "num_kv_heads",
        "tie_embeddings", "dropout_rate",
        "positional_encoding.use_rope",
        "positional_encoding.use_alibi",
        "positional_encoding.rope_base_seq_len",
        "positional_encoding.alibi_min_locality_distance",
        "positional_encoding.alibi_slope_exponent",
        "positional_encoding.alibi_max_bias",
        "positional_encoding.rope_theta",
        "positional_encoding.rope_scaling",
        "sliding_window_stride", "log_interval",
        "atom_stats_interval", "atom_stats_max_seqs",
        "validation_interval", "checkpoint_interval", "use_gpu", "use_flash_attention",
        "precision.parameter_groups.embedding",
        "precision.parameter_groups.lm_head",
        "precision.parameter_groups.attention",
        "precision.parameter_groups.ffn",
        "precision.parameter_groups.rmsnorm",
        "precision.parameter_groups.scratchblock",
        "precision.parameter_groups.mtp",
        "precision.parameter_groups.reasoning_head",
        "precision.parameter_groups.execution_block",
        "precision.parameter_groups.slot_selector",
        
        // Feature enables
        "cosine_decay.enabled", "cosine_decay.warm_restarts",
        "single_batch.enabled", "single_batch.max_steps",
        "soft_restart.enabled", "soft_restart.loss_increase_threshold",
        "soft_restart.max_step_window", "soft_restart.cooldown_steps",
        "auto_stop.enabled", "auto_stop.plateau_patience",
        "auto_stop.plateau_min_delta", "auto_stop.high_loss_threshold",
        "auto_stop.high_loss_patience",
        "shuffle.enabled", "shuffle.epochs",
        
        "telemetry_control.enabled",
        "stability_overrides_enabled",
        "stability_overrides.batch_size",
        "stability_overrides.max_seq_len",
        "stability_overrides.clip_per_token",
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
        "loss.label_smoothing.enabled", "loss.label_smoothing.epsilon",
        "loss.focal.enabled", "loss.focal.gamma", "loss.focal.alpha",
        "loss.preference.enabled", "loss.preference.beta",
        "loss.distillation.enabled", "loss.distillation.temperature", "loss.distillation.lambda",
        "loss.entropy_reg.enabled", "loss.entropy_reg.lambda",
        "loss.class_balanced.enabled", "loss.class_balanced.beta",
        "loss.masking.enabled", "loss.masking.tag",

        // Execution block structural (architecture choices)
        "execution_block.execution_first_type_only",
        "execution_block.layer",
        "execution_block.num_ops",
        "execution_block.num_slots",
        "execution_block.num_scratch_slots",
        "execution_block.num_steps",
        "execution_block.value_decode_input_dim",
        "execution_block.value_decode_hidden_dim",
        "execution_block.d_type",
        "execution_block.cross_attn_topk",
        "execution_block.usage_decay",
        "execution_block.inject_gate_temp",
        "execution_block.result_slot_mode",
        "execution_block.result_slot_index",
        "execution_block.debug_mode",
        "execution_block.entropy_collapse_threshold",
        "execution_block.write_collapse_threshold",
        "execution_block.magnitude_limit",
        "execution_block.diversity_kappa",
        "execution_block.temp_start",
        "execution_block.temp_end",
        "execution_block.temp_schedule",
        "execution_block.entropy_weight",
        "execution_block.step_x_multiplier",
        "execution_block.step_y_multiplier",
        "execution_block.step_y_overrides_x",
        "execution_block.entropy_aux_weight",
        "execution_block.value_match_epsilon",
        "execution_block.final_slot_consistency_weight",
        "execution_block.transition_hard_threshold",
        "execution_block.causal_w1_transition",
        "execution_block.div_invalid_penalty_weight",
        "execution_block.div_magnitude_penalty_weight",
        "execution_block.arg_reinforce_weight",
        "execution_block.arg_reinforce_baseline_decay",
        "execution_block.structured_ce_enabled",
        "execution_block.structured_ce_weight",
        "execution_block.selector.enabled",
        "execution_block.selector.d_slot_features",
        "execution_block.selector.d_selector",
        "execution_block.selector.selection_margin",
        "execution_block.selector.supervision_weight",

        // LM head centering choices
        "lm_head_centering.center_hidden_states",
        "lm_head_centering.freeze_learned_rms_gammas",
        "lm_head_centering.center_logits",
        "lm_head_centering.center_encoder_residuals",
        "lm_head_centering.project_out_pc1",
        "lm_head_centering.pc1_power_iters",
        "layer_scale.init_value",
        "hardcoded_hidden_states.pattern",
        "hardcoded_hidden_states.log_every_n_batches",
        "embedding_freeze.freeze_after_step",
        "optimizer.kind", "optimizer.beta1", "optimizer.beta2", "optimizer.epsilon",
        "scratch_blocks.num_blocks", "scratch_blocks.use_write_combined",
        "scratch_block_reasoning.atom_embedding_dim",
        "scratch_block_reasoning.max_atoms",
        "scratch_block_reasoning.atom_scale",
        "multi_token_prediction.k",
        "multi_token_prediction.alpha",
        "multi_token_prediction.log_ratio_monitor",
        "prediction_comparison.interval",
        "prediction_comparison.top_k",
        "prediction_comparison.max_positions",
        "prediction_comparison.log_path",
        "logit_update_trace.interval",
        "attention_diagnostics.layer",
        "attention_diagnostics.head",

        // Logging
        "logging.default_level", "logging.equation_csv_enabled",
        "logging.stderr_enabled", "logging.initial_capacity", "logging.group_overrides",
        "log_recorder.enabled", "log_recorder.default_level",
        "log_recorder.modules",
        "log_recorder.layers.embedding",
        "log_recorder.layers.rms_norm",
        "log_recorder.layers.attention",
        "log_recorder.layers.feed_forward",
        "log_recorder.layers.residual",
        "log_recorder.layers.encoding",
        "log_recorder.layers.serialization",
        "log_recorder.layers.execution_block",
        "telemetry_control.logging.verbose",
        "telemetry_control.logging.fail_loud_on_accumulation_bug",
        "telemetry_control.plateau_noise.enabled",
        "telemetry_control.spike_thresholds.mild",
        "telemetry_control.spike_thresholds.moderate",
        "telemetry_control.spike_thresholds.severe",
        "telemetry_control.response.moderate_grad_scale",
        "telemetry_control.response.moderate_cooldown_extension",
        "telemetry_control.accumulation_guard.min_grad_for_nonzero_loss",
        "telemetry_control.accumulation_guard.loss_threshold",
        "telemetry_control.accumulation_guard.max_consecutive_zero_grad_steps",
        "telemetry_control.regime_change.seq_len_threshold",
        "telemetry_control.regime_change.suppression_steps",
        "telemetry_control.volatility_damping.threshold",
        "telemetry_control.volatility_damping.max_damping",
        "telemetry_control.gradient_decay.threshold",
        "telemetry_control.gradient_decay.max_boost",
        "telemetry_control.progress_boost.threshold",
        "telemetry_control.progress_boost.max_boost",
        "telemetry_control.outlier.frequency_trigger",
        "telemetry_control.outlier.persistence_trigger",
        "telemetry_control.drift.anchor_sigma_multiplier",
        "telemetry_control.soft_restart.cooldown_steps",
        "telemetry_control.baseline.stabilization_steps",
        "telemetry_control.plateau_noise.patience",
        "telemetry_control.plateau_noise.variance_threshold",
        "telemetry_control.plateau_noise.noise_std",
        "telemetry_control.plateau_noise.proportional",
        "telemetry_control.plateau_noise.cooldown",
        "telemetry_control.plateau_noise.max_per_epoch",
        "telemetry_control.lattice.num_levels",
        "telemetry_control.lattice.num_streams",
        "telemetry_control.lattice.beta_mu",
        "telemetry_control.lattice.beta_a",
        "telemetry_control.lattice.beta_delta",
        "telemetry_control.lattice.beta_r",
        "telemetry_control.lattice.beta_run",
        "telemetry_control.lattice.beta_v",
        "telemetry_control.lattice.k_out0",
        "telemetry_control.lattice.alpha_v",
        "telemetry_control.lattice.epsilon",
        "telemetry_control.lattice.strict_mode",
        "generation.strategy",
        "generation.max_new_tokens",
        "generation.min_new_tokens",
        "generation.temperature",
        "generation.top_k",
        "generation.top_p",
        "generation.min_p",
        "generation.typical_p",
        "generation.repetition_penalty",
        "generation.repetition_penalty_window",
        "generation.frequency_penalty",
        "generation.presence_penalty",
        "generation.no_repeat_ngram_size",
        "generation.do_sample",
        "generation.enable_scratchblock_reasoning",

        // CUDA execution
        "cuda_execution.single_stream_mode",
        "cuda_execution.disable_async_frees",
        "cuda_execution.synchronize_after_kernels",
    };
    
    for (const auto& path : REQUIRED) {
        if (!jsonPathExists(trainConfig, path)) {
            missing.push_back(path);
        }
    }

    if (!jsonPathExists(trainConfig, "gradient_clip")) {
        missing.push_back("gradient_clip");
    }
}

inline void applyTrainingConfigObject(const nlohmann::json& trainConfig, TrainingHyperparameters& params) {
    if (!trainConfig.is_object()) {
        throw std::runtime_error(std::string("ai_config.json: ") + kTrainingConfigPath + " must be an object");
    }
    
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
    assignTrainingField(params.architecture.d_model, trainConfig, "d_model");
    assignTrainingField(params.architecture.num_layers, trainConfig, "num_layers");
    assignTrainingField(params.architecture.num_heads, trainConfig, "num_heads");
    assignTrainingField(params.architecture.num_kv_heads, trainConfig, "num_kv_heads");
    assignTrainingField(params.architecture.max_seq_len, trainConfig, "max_seq_len");
    assignTrainingField(params.architecture.tie_embeddings, trainConfig, "tie_embeddings");
    assignTrainingField(params.architecture.dropout_rate, trainConfig, "dropout_rate");
    assignTrainingField(params.sliding_window_stride, trainConfig, "sliding_window_stride");
    // min_seq_valid_tokens: derived in HyperParameters_GPU.hpp after raw parse.
    // architecture.min_seq_len_for_flash: derived in HyperParameters_GPU.hpp after raw parse.
    assignTrainingField(params.warmup_fraction, trainConfig, "warmup_fraction");
    const auto& cosine_decay = requireJsonObjectField(trainConfig, "cosine_decay", "training.config");
    params.cosine_decay_enabled =
        getRequiredJsonValue<bool>(cosine_decay, "enabled", "training.config.cosine_decay");
    params.cosine_warm_restarts =
        getRequiredJsonValue<bool>(cosine_decay, "warm_restarts", "training.config.cosine_decay");
    // cosine_decay_min_lr: derived in HyperParameters_GPU.hpp after raw parse.
    assignTrainingField(params.log_interval, trainConfig, "log_interval");
    assignTrainingField(params.atom_stats_interval, trainConfig, "atom_stats_interval");
    assignTrainingField(params.atom_stats_max_seqs, trainConfig, "atom_stats_max_seqs");
    assignTrainingField(params.validation_interval, trainConfig, "validation_interval");
    assignTrainingField(params.checkpoint_interval, trainConfig, "checkpoint_interval");
    assignTrainingField(params.architecture.use_gpu, trainConfig, "use_gpu");
    assignTrainingField(params.architecture.use_flash_attention, trainConfig, "use_flash_attention");
    // architecture.min_seq_len_for_flash: derived from max_seq_len in HyperParameters_GPU.hpp.

    {
        const auto& precision = requireJsonObjectField(trainConfig, "precision", "training.config");
        const auto& groups = requireJsonObjectField(precision, "parameter_groups", "training.config.precision");
        auto parse_group_precision = [&](const char* key) {
            if (!groups.contains(key) || !groups[key].is_string()) {
                throw std::runtime_error(std::string("[ai_config] training.config.precision.parameter_groups.") +
                                         key + " must be a string");
            }
            return ::GRIM::HyperParameters::parseParameterGroupPrecision(groups[key].get<std::string>());
        };

        params.architecture.parameter_precision_embedding = parse_group_precision("embedding");
        params.architecture.parameter_precision_lm_head = parse_group_precision("lm_head");
        params.architecture.parameter_precision_attention = parse_group_precision("attention");
        params.architecture.parameter_precision_ffn = parse_group_precision("ffn");
        params.architecture.parameter_precision_rmsnorm = parse_group_precision("rmsnorm");
        params.architecture.parameter_precision_scratchblock = parse_group_precision("scratchblock");
        params.architecture.parameter_precision_mtp = parse_group_precision("mtp");
        params.architecture.parameter_precision_reasoning_head = parse_group_precision("reasoning_head");
        params.architecture.parameter_precision_execution_block = parse_group_precision("execution_block");
        params.architecture.parameter_precision_slot_selector = parse_group_precision("slot_selector");
    }

    {
        const auto& soft = requireJsonObjectField(trainConfig, "soft_restart", "training.config");
        params.soft_restart_enabled =
            getRequiredJsonValue<bool>(soft, "enabled", "training.config.soft_restart");
        params.soft_restart_loss_increase_threshold =
            getRequiredJsonValue<float>(soft, "loss_increase_threshold", "training.config.soft_restart");
        params.soft_restart_max_step_window =
            getRequiredJsonValue<int>(soft, "max_step_window", "training.config.soft_restart");
        params.soft_restart_cooldown_steps =
            getRequiredJsonValue<int>(soft, "cooldown_steps", "training.config.soft_restart");
    }

    {
        const auto& autoStop = requireJsonObjectField(trainConfig, "auto_stop", "training.config");
        params.auto_stop_enabled =
            getRequiredJsonValue<bool>(autoStop, "enabled", "training.config.auto_stop");
        params.auto_stop_plateau_patience =
            getRequiredJsonValue<int>(autoStop, "plateau_patience", "training.config.auto_stop");
        params.auto_stop_plateau_min_delta =
            getRequiredJsonValue<float>(autoStop, "plateau_min_delta", "training.config.auto_stop");
        params.auto_stop_high_loss_threshold =
            getRequiredJsonValue<float>(autoStop, "high_loss_threshold", "training.config.auto_stop");
        params.auto_stop_high_loss_patience =
            getRequiredJsonValue<int>(autoStop, "high_loss_patience", "training.config.auto_stop");
    }

    {
        const auto& single = requireJsonObjectField(trainConfig, "single_batch", "training.config");
        params.single_batch_overfit_enabled =
            getRequiredJsonValue<bool>(single, "enabled", "training.config.single_batch");
        params.single_batch_overfit_max_steps =
            getRequiredJsonValue<int>(single, "max_steps", "training.config.single_batch");
    }

    {
        const auto& shuffle = requireJsonObjectField(trainConfig, "shuffle", "training.config");
        params.shuffle_train_enabled =
            getRequiredJsonValue<bool>(shuffle, "enabled", "training.config.shuffle");
        params.shuffle_train_epochs =
            getRequiredJsonValue<int>(shuffle, "epochs", "training.config.shuffle");
        if (params.shuffle_train_epochs < 0) {
            throw std::runtime_error(
                "ai_config.json: training.config.shuffle.epochs must be >= 0");
        }
    }

    {
        const auto& tc = requireJsonObjectField(trainConfig, "telemetry_control", "training.config");
        params.telemetry_control_enabled =
            getRequiredJsonValue<bool>(tc, "enabled", "training.config.telemetry_control");

        const auto& spike = requireJsonObjectField(tc, "spike_thresholds", "training.config.telemetry_control");
        params.telemetry_spike_mild_threshold =
            getRequiredJsonValue<float>(spike, "mild", "training.config.telemetry_control.spike_thresholds");
        params.telemetry_spike_moderate_threshold =
            getRequiredJsonValue<float>(spike, "moderate", "training.config.telemetry_control.spike_thresholds");
        params.telemetry_spike_severe_threshold =
            getRequiredJsonValue<float>(spike, "severe", "training.config.telemetry_control.spike_thresholds");

        const auto& response = requireJsonObjectField(tc, "response", "training.config.telemetry_control");
        params.telemetry_moderate_grad_scale =
            getRequiredJsonValue<float>(response, "moderate_grad_scale", "training.config.telemetry_control.response");
        params.telemetry_moderate_cooldown_extension =
            getRequiredJsonValue<int>(response, "moderate_cooldown_extension", "training.config.telemetry_control.response");

        const auto& acc = requireJsonObjectField(tc, "accumulation_guard", "training.config.telemetry_control");
        params.telemetry_min_grad_for_nonzero_loss =
            getRequiredJsonValue<float>(acc, "min_grad_for_nonzero_loss", "training.config.telemetry_control.accumulation_guard");
        params.telemetry_loss_threshold_for_grad_check =
            getRequiredJsonValue<float>(acc, "loss_threshold", "training.config.telemetry_control.accumulation_guard");
        params.telemetry_max_consecutive_zero_grad_steps =
            getRequiredJsonValue<int>(acc, "max_consecutive_zero_grad_steps", "training.config.telemetry_control.accumulation_guard");

        const auto& regime = requireJsonObjectField(tc, "regime_change", "training.config.telemetry_control");
        params.telemetry_seq_len_regime_change_threshold =
            getRequiredJsonValue<float>(regime, "seq_len_threshold", "training.config.telemetry_control.regime_change");
        params.telemetry_regime_change_suppression_steps =
            getRequiredJsonValue<int>(regime, "suppression_steps", "training.config.telemetry_control.regime_change");

        const auto& vol = requireJsonObjectField(tc, "volatility_damping", "training.config.telemetry_control");
        params.telemetry_volatility_damping_threshold =
            getRequiredJsonValue<float>(vol, "threshold", "training.config.telemetry_control.volatility_damping");
        params.telemetry_max_volatility_damping =
            getRequiredJsonValue<float>(vol, "max_damping", "training.config.telemetry_control.volatility_damping");

        const auto& decay = requireJsonObjectField(tc, "gradient_decay", "training.config.telemetry_control");
        params.telemetry_gradient_decay_threshold =
            getRequiredJsonValue<float>(decay, "threshold", "training.config.telemetry_control.gradient_decay");
        params.telemetry_max_decay_boost =
            getRequiredJsonValue<float>(decay, "max_boost", "training.config.telemetry_control.gradient_decay");

        const auto& boost = requireJsonObjectField(tc, "progress_boost", "training.config.telemetry_control");
        params.telemetry_progress_boost_threshold =
            getRequiredJsonValue<float>(boost, "threshold", "training.config.telemetry_control.progress_boost");
        params.telemetry_max_progress_boost =
            getRequiredJsonValue<float>(boost, "max_boost", "training.config.telemetry_control.progress_boost");

        const auto& outlier = requireJsonObjectField(tc, "outlier", "training.config.telemetry_control");
        params.telemetry_outlier_frequency_trigger =
            getRequiredJsonValue<float>(outlier, "frequency_trigger", "training.config.telemetry_control.outlier");
        params.telemetry_outlier_persistence_trigger =
            getRequiredJsonValue<float>(outlier, "persistence_trigger", "training.config.telemetry_control.outlier");

        const auto& drift = requireJsonObjectField(tc, "drift", "training.config.telemetry_control");
        params.telemetry_anchor_drift_sigma_multiplier =
            getRequiredJsonValue<float>(drift, "anchor_sigma_multiplier", "training.config.telemetry_control.drift");

        const auto& sr = requireJsonObjectField(tc, "soft_restart", "training.config.telemetry_control");
        params.telemetry_soft_restart_cooldown_steps =
            getRequiredJsonValue<int>(sr, "cooldown_steps", "training.config.telemetry_control.soft_restart");

        const auto& base = requireJsonObjectField(tc, "baseline", "training.config.telemetry_control");
        params.telemetry_baseline_stabilization_steps =
            getRequiredJsonValue<int>(base, "stabilization_steps", "training.config.telemetry_control.baseline");

        const auto& logging = requireJsonObjectField(tc, "logging", "training.config.telemetry_control");
        params.telemetry_verbose_logging =
            getRequiredJsonValue<bool>(logging, "verbose", "training.config.telemetry_control.logging");
        params.telemetry_fail_loud_on_accumulation_bug =
            getRequiredJsonValue<bool>(logging, "fail_loud_on_accumulation_bug", "training.config.telemetry_control.logging");

        const auto& pn = requireJsonObjectField(tc, "plateau_noise", "training.config.telemetry_control");
        params.telemetry_plateau_noise_enabled =
            getRequiredJsonValue<bool>(pn, "enabled", "training.config.telemetry_control.plateau_noise");
        params.telemetry_plateau_noise_patience =
            getRequiredJsonValue<int>(pn, "patience", "training.config.telemetry_control.plateau_noise");
        params.telemetry_plateau_noise_variance_threshold =
            getRequiredJsonValue<float>(pn, "variance_threshold", "training.config.telemetry_control.plateau_noise");
        params.telemetry_plateau_noise_std =
            getRequiredJsonValue<float>(pn, "noise_std", "training.config.telemetry_control.plateau_noise");
        params.telemetry_plateau_noise_proportional =
            getRequiredJsonValue<bool>(pn, "proportional", "training.config.telemetry_control.plateau_noise");
        params.telemetry_plateau_noise_cooldown =
            getRequiredJsonValue<int>(pn, "cooldown", "training.config.telemetry_control.plateau_noise");
        params.telemetry_plateau_noise_max_per_epoch =
            getRequiredJsonValue<int>(pn, "max_per_epoch", "training.config.telemetry_control.plateau_noise");

        const auto& lat = requireJsonObjectField(tc, "lattice", "training.config.telemetry_control");
        params.telemetry_lattice_num_levels =
            getRequiredJsonValue<int>(lat, "num_levels", "training.config.telemetry_control.lattice");
        params.telemetry_lattice_num_streams =
            getRequiredJsonValue<int>(lat, "num_streams", "training.config.telemetry_control.lattice");
        params.telemetry_lattice_beta_mu =
            getRequiredJsonValue<float>(lat, "beta_mu", "training.config.telemetry_control.lattice");
        params.telemetry_lattice_beta_a =
            getRequiredJsonValue<float>(lat, "beta_a", "training.config.telemetry_control.lattice");
        params.telemetry_lattice_beta_delta =
            getRequiredJsonValue<float>(lat, "beta_delta", "training.config.telemetry_control.lattice");
        params.telemetry_lattice_beta_r =
            getRequiredJsonValue<float>(lat, "beta_r", "training.config.telemetry_control.lattice");
        params.telemetry_lattice_beta_run =
            getRequiredJsonValue<float>(lat, "beta_run", "training.config.telemetry_control.lattice");
        params.telemetry_lattice_beta_v =
            getRequiredJsonValue<float>(lat, "beta_v", "training.config.telemetry_control.lattice");
        params.telemetry_lattice_k_out0 =
            getRequiredJsonValue<float>(lat, "k_out0", "training.config.telemetry_control.lattice");
        params.telemetry_lattice_alpha_v =
            getRequiredJsonValue<float>(lat, "alpha_v", "training.config.telemetry_control.lattice");
        params.telemetry_lattice_epsilon =
            getRequiredJsonValue<float>(lat, "epsilon", "training.config.telemetry_control.lattice");
        params.telemetry_lattice_strict_mode =
            getRequiredJsonValue<bool>(lat, "strict_mode", "training.config.telemetry_control.lattice");
    }

    {
        const auto& logCfg = requireJsonObjectField(trainConfig, "logging", "training.config");
        params.tape_logging.default_level =
            getRequiredJsonValue<std::string>(logCfg, "default_level", "training.config.logging");
        params.tape_logging.equation_csv_enabled =
            getRequiredJsonValue<bool>(logCfg, "equation_csv_enabled", "training.config.logging");
        params.tape_logging.stderr_enabled =
            getRequiredJsonValue<bool>(logCfg, "stderr_enabled", "training.config.logging");
        params.tape_logging.initial_capacity =
            getRequiredJsonValue<size_t>(logCfg, "initial_capacity", "training.config.logging");
        params.tape_logging.group_overrides.clear();
        const auto& group_overrides = requireJsonObjectField(logCfg, "group_overrides", "training.config.logging");
        for (auto& [key, val] : group_overrides.items()) {
            if (!val.is_string()) {
                throw std::runtime_error(
                    std::string("ai_config.json: training.config.logging.group_overrides.") + key +
                    " must be a string");
            }
            params.tape_logging.group_overrides[key] = val.get<std::string>();
        }
    }

    {
        const auto& logRec = requireJsonObjectField(trainConfig, "log_recorder", "training.config");
        params.log_recorder.enabled =
            getRequiredJsonValue<bool>(logRec, "enabled", "training.config.log_recorder");
        params.log_recorder.default_level =
            getRequiredJsonValue<std::string>(logRec, "default_level", "training.config.log_recorder");

        params.log_recorder.modules.clear();
        const auto& modules = requireJsonObjectField(logRec, "modules", "training.config.log_recorder");
        for (auto& [key, val] : modules.items()) {
            if (!val.is_string()) {
                throw std::runtime_error(
                    std::string("ai_config.json: training.config.log_recorder.modules.") + key +
                    " must be a string");
            }
            params.log_recorder.modules[key] = val.get<std::string>();
        }

        const auto& layers = requireJsonObjectField(logRec, "layers", "training.config.log_recorder");
        params.log_recorder.layers.embedding =
            getRequiredJsonValue<bool>(layers, "embedding", "training.config.log_recorder.layers");
        params.log_recorder.layers.rms_norm =
            getRequiredJsonValue<bool>(layers, "rms_norm", "training.config.log_recorder.layers");
        params.log_recorder.layers.attention =
            getRequiredJsonValue<bool>(layers, "attention", "training.config.log_recorder.layers");
        params.log_recorder.layers.feed_forward =
            getRequiredJsonValue<bool>(layers, "feed_forward", "training.config.log_recorder.layers");
        params.log_recorder.layers.residual =
            getRequiredJsonValue<bool>(layers, "residual", "training.config.log_recorder.layers");
        params.log_recorder.layers.encoding =
            getRequiredJsonValue<bool>(layers, "encoding", "training.config.log_recorder.layers");
        params.log_recorder.layers.serialization =
            getRequiredJsonValue<bool>(layers, "serialization", "training.config.log_recorder.layers");
        params.log_recorder.layers.execution_block =
            getRequiredJsonValue<bool>(layers, "execution_block", "training.config.log_recorder.layers");
    }

    {
        const auto& loss_cfg = requireJsonObjectField(trainConfig, "loss", "training.config");

        const auto& ls = requireJsonObjectField(loss_cfg, "label_smoothing", "training.config.loss");
        params.loss_label_smoothing_enabled =
            getRequiredJsonValue<bool>(ls, "enabled", "training.config.loss.label_smoothing");
        params.loss_label_smoothing_epsilon =
            getRequiredJsonValue<float>(ls, "epsilon", "training.config.loss.label_smoothing");

        const auto& fc = requireJsonObjectField(loss_cfg, "focal", "training.config.loss");
        params.loss_focal_enabled =
            getRequiredJsonValue<bool>(fc, "enabled", "training.config.loss.focal");
        params.loss_focal_gamma =
            getRequiredJsonValue<float>(fc, "gamma", "training.config.loss.focal");
        params.loss_focal_alpha =
            getRequiredJsonValue<float>(fc, "alpha", "training.config.loss.focal");

        const auto& er = requireJsonObjectField(loss_cfg, "entropy_reg", "training.config.loss");
        params.loss_entropy_reg_enabled =
            getRequiredJsonValue<bool>(er, "enabled", "training.config.loss.entropy_reg");
        params.loss_entropy_reg_lambda =
            getRequiredJsonValue<float>(er, "lambda", "training.config.loss.entropy_reg");

        const auto& cb = requireJsonObjectField(loss_cfg, "class_balanced", "training.config.loss");
        params.loss_class_balanced_enabled =
            getRequiredJsonValue<bool>(cb, "enabled", "training.config.loss.class_balanced");
        params.loss_class_balanced_beta =
            getRequiredJsonValue<float>(cb, "beta", "training.config.loss.class_balanced");

        const auto& pref = requireJsonObjectField(loss_cfg, "preference", "training.config.loss");
        params.loss_preference_enabled =
            getRequiredJsonValue<bool>(pref, "enabled", "training.config.loss.preference");
        params.loss_preference_beta =
            getRequiredJsonValue<float>(pref, "beta", "training.config.loss.preference");

        const auto& dist = requireJsonObjectField(loss_cfg, "distillation", "training.config.loss");
        params.loss_distillation_enabled =
            getRequiredJsonValue<bool>(dist, "enabled", "training.config.loss.distillation");
        params.loss_distillation_temperature =
            getRequiredJsonValue<float>(dist, "temperature", "training.config.loss.distillation");
        params.loss_distillation_lambda =
            getRequiredJsonValue<float>(dist, "lambda", "training.config.loss.distillation");

        const auto& mask = requireJsonObjectField(loss_cfg, "masking", "training.config.loss");
        params.loss_masking_enabled =
            getRequiredJsonValue<bool>(mask, "enabled", "training.config.loss.masking");
        params.loss_masking_tag =
            getRequiredJsonValue<std::string>(mask, "tag", "training.config.loss.masking");
    }

    {
        const auto& lmc = requireJsonObjectField(trainConfig, "lm_head_centering", "training.config");
        params.lm_head_centering_enabled =
            getRequiredJsonValue<bool>(lmc, "enabled", "training.config.lm_head_centering");
        params.architecture.lm_head_center_hidden_states =
            getRequiredJsonValue<bool>(lmc, "center_hidden_states", "training.config.lm_head_centering");
        params.architecture.freeze_learned_rms_gammas =
            getRequiredJsonValue<bool>(lmc, "freeze_learned_rms_gammas", "training.config.lm_head_centering");
        params.architecture.center_logits =
            getRequiredJsonValue<bool>(lmc, "center_logits", "training.config.lm_head_centering");
        params.architecture.center_encoder_residuals =
            getRequiredJsonValue<bool>(lmc, "center_encoder_residuals", "training.config.lm_head_centering");
        params.architecture.project_out_pc1 =
            getRequiredJsonValue<bool>(lmc, "project_out_pc1", "training.config.lm_head_centering");
        params.architecture.pc1_power_iters =
            getRequiredJsonValue<int>(lmc, "pc1_power_iters", "training.config.lm_head_centering");
    }

    {
        const auto& ls = requireJsonObjectField(trainConfig, "layer_scale", "training.config");
        params.architecture.use_layer_scale =
            getRequiredJsonValue<bool>(ls, "enabled", "training.config.layer_scale");
        params.architecture.layer_scale_init =
            getRequiredJsonValue<float>(ls, "init_value", "training.config.layer_scale");
    }

    {
        const auto& qkn = requireJsonObjectField(trainConfig, "qk_norm", "training.config");
        params.architecture.qk_norm_enabled =
            getRequiredJsonValue<bool>(qkn, "enabled", "training.config.qk_norm");
    }

    {
        const auto& hcs = requireJsonObjectField(trainConfig, "hardcoded_hidden_states", "training.config");
        const bool hardcoded_enabled =
            getRequiredJsonValue<bool>(hcs, "enabled", "training.config.hardcoded_hidden_states");
        const std::string pattern_str =
            getRequiredJsonValue<std::string>(hcs, "pattern", "training.config.hardcoded_hidden_states");
        params.architecture.hardcoded_log_every_n_batches =
            getRequiredJsonValue<int>(hcs, "log_every_n_batches", "training.config.hardcoded_hidden_states");

        using HCP = ::GRIM::HyperParameters::LanguageModelConfig::HardcodedPattern;
        HCP parsed_pattern = HCP::DISABLED;
        if (pattern_str == "disabled") {
            parsed_pattern = HCP::DISABLED;
        } else if (pattern_str == "random_centered") {
            parsed_pattern = HCP::RANDOM_CENTERED;
        } else if (pattern_str == "orthogonal_w277") {
            parsed_pattern = HCP::ORTHOGONAL_W277;
        } else if (pattern_str == "aligned_w277") {
            parsed_pattern = HCP::ALIGNED_W277;
        } else if (pattern_str == "constant_uniform") {
            parsed_pattern = HCP::CONSTANT_UNIFORM;
        } else if (pattern_str == "zero_mean_sine") {
            parsed_pattern = HCP::ZERO_MEAN_SINE;
        } else {
            throw std::runtime_error(
                "ai_config.json: training.config.hardcoded_hidden_states.pattern has unknown value '" +
                pattern_str + "'");
        }
        if (hardcoded_enabled) {
            params.architecture.hardcoded_hidden_pattern = parsed_pattern;
        } else {
            params.architecture.hardcoded_hidden_pattern = HCP::DISABLED;
        }
    }

    {
        const auto& ef = requireJsonObjectField(trainConfig, "embedding_freeze", "training.config");
        params.embedding_freeze_enabled =
            getRequiredJsonValue<bool>(ef, "enabled", "training.config.embedding_freeze");
        params.embedding_freeze_after_step =
            getRequiredJsonValue<int>(ef, "freeze_after_step", "training.config.embedding_freeze");
    }

    {
        const auto& opt = requireJsonObjectField(trainConfig, "optimizer", "training.config");
        params.optimizer_kind =
            getRequiredJsonValue<std::string>(opt, "kind", "training.config.optimizer");
        params.optimizer_beta1 =
            getRequiredJsonValue<float>(opt, "beta1", "training.config.optimizer");
        params.optimizer_beta2 =
            getRequiredJsonValue<float>(opt, "beta2", "training.config.optimizer");
        params.optimizer_epsilon =
            getRequiredJsonValue<float>(opt, "epsilon", "training.config.optimizer");
        if (params.optimizer_kind != "adamw" && params.optimizer_kind != "radamw") {
            throw std::runtime_error(
                "[ai_config] training.config.optimizer.kind must be \"adamw\" or \"radamw\", got \"" +
                params.optimizer_kind + "\"");
        }
    }

    assignTrainingField(params.stability_overrides_enabled, trainConfig, "stability_overrides_enabled");
    {
        const auto& stability = requireJsonObjectField(trainConfig, "stability_overrides", "training.config");
        params.stability_override_batch_size =
            getRequiredJsonValue<int>(stability, "batch_size", "training.config.stability_overrides");
        params.stability_override_max_seq_len =
            getRequiredJsonValue<int>(stability, "max_seq_len", "training.config.stability_overrides");
        params.stability_override_clip_per_token =
            getRequiredJsonValue<float>(stability, "clip_per_token", "training.config.stability_overrides");
    }

    {
        const auto& scratch = requireJsonObjectField(trainConfig, "scratch_blocks", "training.config");
        params.scratch_blocks_enabled =
            getRequiredJsonValue<bool>(scratch, "enabled", "training.config.scratch_blocks");
        params.scratch_num_blocks =
            getRequiredJsonValue<size_t>(scratch, "num_blocks", "training.config.scratch_blocks");
        params.scratch_write_combined =
            getRequiredJsonValue<bool>(scratch, "use_write_combined", "training.config.scratch_blocks");
    }

    {
        const auto& sbr = requireJsonObjectField(trainConfig, "scratch_block_reasoning", "training.config");
        params.architecture.use_scratch_block =
            getRequiredJsonValue<bool>(sbr, "enabled", "training.config.scratch_block_reasoning");
        params.architecture.scratch_block_atom_embedding_dim =
            getRequiredJsonValue<int>(sbr, "atom_embedding_dim", "training.config.scratch_block_reasoning");
        params.architecture.scratch_block_max_atoms =
            getRequiredJsonValue<int>(sbr, "max_atoms", "training.config.scratch_block_reasoning");
        params.architecture.scratch_block_atom_scale =
            getRequiredJsonValue<float>(sbr, "atom_scale", "training.config.scratch_block_reasoning");
    }

    {
        const auto& eb = requireJsonObjectField(trainConfig, "execution_block", "training.config");
        params.architecture.execution_block_enabled =
            getRequiredJsonValue<bool>(eb, "enabled", "training.config.execution_block");
        params.architecture.scratch_block_execution_first_type_only =
            getRequiredJsonValue<bool>(eb, "execution_first_type_only", "training.config.execution_block");
        params.architecture.execution_block_layer =
            getRequiredJsonValue<int>(eb, "layer", "training.config.execution_block");
        params.architecture.execution_block_num_ops =
            getRequiredJsonValue<int>(eb, "num_ops", "training.config.execution_block");
        params.architecture.execution_block_num_slots =
            getRequiredJsonValue<int>(eb, "num_slots", "training.config.execution_block");
        params.architecture.execution_block_num_scratch_slots =
            getRequiredJsonValue<int>(eb, "num_scratch_slots", "training.config.execution_block");
        params.architecture.execution_block_num_steps =
            getRequiredJsonValue<int>(eb, "num_steps", "training.config.execution_block");
        params.architecture.execution_block_value_decode_input_dim =
            getRequiredJsonValue<int>(eb, "value_decode_input_dim", "training.config.execution_block");
        params.architecture.execution_block_value_decode_hidden_dim =
            getRequiredJsonValue<int>(eb, "value_decode_hidden_dim", "training.config.execution_block");
        params.architecture.execution_block_d_type =
            getRequiredJsonValue<int>(eb, "d_type", "training.config.execution_block");
        params.architecture.execution_block_cross_attn_topk =
            getRequiredJsonValue<int>(eb, "cross_attn_topk", "training.config.execution_block");
        params.architecture.execution_block_usage_decay =
            getRequiredJsonValue<float>(eb, "usage_decay", "training.config.execution_block");
        params.architecture.execution_block_inject_gate_temp =
            getRequiredJsonValue<float>(eb, "inject_gate_temp", "training.config.execution_block");
        params.architecture.execution_block_result_slot_mode =
            getRequiredJsonValue<int>(eb, "result_slot_mode", "training.config.execution_block");
        params.architecture.execution_block_result_slot_index =
            getRequiredJsonValue<int>(eb, "result_slot_index", "training.config.execution_block");
        params.architecture.execution_block_debug_mode =
            getRequiredJsonValue<bool>(eb, "debug_mode", "training.config.execution_block");
        params.architecture.execution_block_entropy_collapse_threshold =
            getRequiredJsonValue<float>(eb, "entropy_collapse_threshold", "training.config.execution_block");
        params.architecture.execution_block_write_collapse_threshold =
            getRequiredJsonValue<float>(eb, "write_collapse_threshold", "training.config.execution_block");
        params.architecture.execution_block_magnitude_limit =
            getRequiredJsonValue<float>(eb, "magnitude_limit", "training.config.execution_block");
        params.architecture.execution_block_diversity_kappa =
            getRequiredJsonValue<float>(eb, "diversity_kappa", "training.config.execution_block");
        params.architecture.execution_block_temp_start =
            getRequiredJsonValue<float>(eb, "temp_start", "training.config.execution_block");
        params.architecture.execution_block_temp_end =
            getRequiredJsonValue<float>(eb, "temp_end", "training.config.execution_block");
        params.architecture.execution_block_temp_schedule =
            getRequiredJsonValue<int>(eb, "temp_schedule", "training.config.execution_block");
        params.architecture.execution_block_entropy_weight =
            getRequiredJsonValue<float>(eb, "entropy_weight", "training.config.execution_block");
        params.architecture.step_x_multiplier =
            getRequiredJsonValue<float>(eb, "step_x_multiplier", "training.config.execution_block");
        params.architecture.step_y_multiplier =
            getRequiredJsonValue<float>(eb, "step_y_multiplier", "training.config.execution_block");
        params.architecture.step_y_overrides_x =
            getRequiredJsonValue<bool>(eb, "step_y_overrides_x", "training.config.execution_block");
        params.architecture.entropy_aux_weight =
            getRequiredJsonValue<float>(eb, "entropy_aux_weight", "training.config.execution_block");
        params.architecture.value_match_epsilon =
            getRequiredJsonValue<float>(eb, "value_match_epsilon", "training.config.execution_block");
        params.architecture.final_slot_consistency_weight =
            getRequiredJsonValue<float>(eb, "final_slot_consistency_weight", "training.config.execution_block");
        params.architecture.execution_block_transition_hard_threshold =
            getRequiredJsonValue<float>(eb, "transition_hard_threshold", "training.config.execution_block");
        params.architecture.execution_block_causal_w1_transition =
            getRequiredJsonValue<float>(eb, "causal_w1_transition", "training.config.execution_block");
        params.architecture.div_invalid_penalty_weight =
            getRequiredJsonValue<float>(eb, "div_invalid_penalty_weight", "training.config.execution_block");
        params.architecture.div_magnitude_penalty_weight =
            getRequiredJsonValue<float>(eb, "div_magnitude_penalty_weight", "training.config.execution_block");
        params.architecture.arg_reinforce_weight =
            getRequiredJsonValue<float>(eb, "arg_reinforce_weight", "training.config.execution_block");
        params.architecture.arg_reinforce_baseline_decay =
            getRequiredJsonValue<float>(eb, "arg_reinforce_baseline_decay", "training.config.execution_block");
        params.architecture.structured_ce_enabled =
            getRequiredJsonValue<bool>(eb, "structured_ce_enabled", "training.config.execution_block");
        params.architecture.structured_ce_weight =
            getRequiredJsonValue<float>(eb, "structured_ce_weight", "training.config.execution_block");

        const auto& sel = requireJsonObjectField(eb, "selector", "training.config.execution_block");
        params.architecture.selector_enabled =
            getRequiredJsonValue<bool>(sel, "enabled", "training.config.execution_block.selector");
        params.architecture.decode_time_slot_feature_dim =
            getRequiredJsonValue<int>(sel, "d_slot_features", "training.config.execution_block.selector");
        params.architecture.selector_d_selector =
            getRequiredJsonValue<int>(sel, "d_selector", "training.config.execution_block.selector");
        params.architecture.selector_selection_margin =
            getRequiredJsonValue<float>(sel, "selection_margin", "training.config.execution_block.selector");
        params.architecture.selector_supervision_weight =
            getRequiredJsonValue<float>(sel, "supervision_weight", "training.config.execution_block.selector");
    }

    {
        const auto& cuda_exec = requireJsonObjectField(trainConfig, "cuda_execution", "training.config");
        params.single_stream_mode =
            getRequiredJsonValue<bool>(cuda_exec, "single_stream_mode", "training.config.cuda_execution");
        params.disable_async_frees =
            getRequiredJsonValue<bool>(cuda_exec, "disable_async_frees", "training.config.cuda_execution");
        params.synchronize_after_kernels =
            getRequiredJsonValue<bool>(cuda_exec, "synchronize_after_kernels", "training.config.cuda_execution");
    }

    {
        const auto& mtp = requireJsonObjectField(trainConfig, "multi_token_prediction", "training.config");
        params.architecture.mtp_enabled =
            getRequiredJsonValue<bool>(mtp, "enabled", "training.config.multi_token_prediction");
        params.architecture.mtp_k =
            getRequiredJsonValue<int>(mtp, "k", "training.config.multi_token_prediction");
        params.architecture.mtp_alpha =
            getRequiredJsonValue<float>(mtp, "alpha", "training.config.multi_token_prediction");
        params.mtp_log_ratio_monitor =
            getRequiredJsonValue<bool>(mtp, "log_ratio_monitor", "training.config.multi_token_prediction");
    }

    {
        const auto& pred_cmp = requireJsonObjectField(trainConfig, "prediction_comparison", "training.config");
        params.prediction_comparison_enabled =
            getRequiredJsonValue<bool>(pred_cmp, "enabled", "training.config.prediction_comparison");
        params.prediction_comparison_interval =
            getRequiredJsonValue<int>(pred_cmp, "interval", "training.config.prediction_comparison");
        params.prediction_comparison_top_k =
            getRequiredJsonValue<int>(pred_cmp, "top_k", "training.config.prediction_comparison");
        params.prediction_comparison_max_positions =
            getRequiredJsonValue<int>(pred_cmp, "max_positions", "training.config.prediction_comparison");
        params.prediction_comparison_log_path =
            getRequiredJsonValue<std::string>(pred_cmp, "log_path", "training.config.prediction_comparison");
    }

    {
        const auto& trace = requireJsonObjectField(trainConfig, "logit_update_trace", "training.config");
        params.logit_update_trace_enabled =
            getRequiredJsonValue<bool>(trace, "enabled", "training.config.logit_update_trace");
        params.logit_update_trace_interval =
            getRequiredJsonValue<int>(trace, "interval", "training.config.logit_update_trace");
    }

    {
        const auto& attn_diag = requireJsonObjectField(trainConfig, "attention_diagnostics", "training.config");
        params.attention_diag_enabled =
            getRequiredJsonValue<bool>(attn_diag, "enabled", "training.config.attention_diagnostics");
        params.attention_diag_layer =
            getRequiredJsonValue<int>(attn_diag, "layer", "training.config.attention_diagnostics");
        params.attention_diag_head =
            getRequiredJsonValue<int>(attn_diag, "head", "training.config.attention_diagnostics");
    }

    {
        const auto& gen = requireJsonObjectField(trainConfig, "generation", kTrainingConfigPath);
        const std::string strategy =
            getRequiredJsonValue<std::string>(gen, "strategy", kTrainingGenerationPath);

        using SamplingStrategy = ::GRIM::HyperParameters::SamplingStrategy;
        if (strategy == "greedy") {
            params.architecture.generation.strategy = SamplingStrategy::GREEDY;
        } else if (strategy == "top_k") {
            params.architecture.generation.strategy = SamplingStrategy::TOP_K;
        } else if (strategy == "top_p") {
            params.architecture.generation.strategy = SamplingStrategy::TOP_P;
        } else if (strategy == "min_p") {
            params.architecture.generation.strategy = SamplingStrategy::MIN_P;
        } else if (strategy == "typical") {
            params.architecture.generation.strategy = SamplingStrategy::TYPICAL;
        } else if (strategy == "top_k_top_p") {
            params.architecture.generation.strategy = SamplingStrategy::TOP_K_TOP_P;
        } else {
            throw std::runtime_error("ai_config.json: training.config.generation.strategy has unknown value '" +
                                     strategy + "'");
        }

        params.architecture.generation.max_new_tokens =
            getRequiredJsonValue<int>(gen, "max_new_tokens", kTrainingGenerationPath);
        params.architecture.generation.min_new_tokens =
            getRequiredJsonValue<int>(gen, "min_new_tokens", kTrainingGenerationPath);
        params.architecture.generation.temperature =
            getRequiredJsonValue<float>(gen, "temperature", kTrainingGenerationPath);
        params.architecture.generation.top_k =
            getRequiredJsonValue<int>(gen, "top_k", kTrainingGenerationPath);
        params.architecture.generation.top_p =
            getRequiredJsonValue<float>(gen, "top_p", kTrainingGenerationPath);
        params.architecture.generation.min_p =
            getRequiredJsonValue<float>(gen, "min_p", kTrainingGenerationPath);
        params.architecture.generation.typical_p =
            getRequiredJsonValue<float>(gen, "typical_p", kTrainingGenerationPath);
        params.architecture.generation.repetition_penalty =
            getRequiredJsonValue<float>(gen, "repetition_penalty", kTrainingGenerationPath);
        params.architecture.generation.repetition_penalty_window =
            getRequiredJsonValue<int>(gen, "repetition_penalty_window", kTrainingGenerationPath);
        params.architecture.generation.frequency_penalty =
            getRequiredJsonValue<float>(gen, "frequency_penalty", kTrainingGenerationPath);
        params.architecture.generation.presence_penalty =
            getRequiredJsonValue<float>(gen, "presence_penalty", kTrainingGenerationPath);
        params.architecture.generation.no_repeat_ngram_size =
            getRequiredJsonValue<int>(gen, "no_repeat_ngram_size", kTrainingGenerationPath);
        params.architecture.generation.do_sample =
            getRequiredJsonValue<bool>(gen, "do_sample", kTrainingGenerationPath);
        params.architecture.generation.enable_scratchblock_reasoning =
            getRequiredJsonValue<bool>(gen, "enable_scratchblock_reasoning", kTrainingGenerationPath);
    }
}

inline bool assignTrainingHyperparametersFromDocument(const nlohmann::json& config, TrainingHyperparameters& params) {
    const auto& training = requireTrainingObject(config);
    const auto& trainConfig = requireTrainingConfigObject(config);
    // Phase 1: Reset to value-initialized sentinels so repeated loads cannot
    // retain stale caller state. Authored runtime values are assigned below;
    // formula-derived values are computed in HyperParameters_GPU.hpp after raw parsing.
    params = TrainingHyperparameters{};
    params.current_model_training = getRequiredJsonValue<std::string>(training, "current_model_training", "training");
    params.current_curriculum = getRequiredJsonValue<std::string>(training, "current_curriculum", "training");
    // Phase 2: Parse authored JSON after the top-level validator has already run.
    applyTrainingConfigObject(trainConfig, params);
    return true;
}

inline bool assignDataCollectionFields(const nlohmann::json& config, AiConfigSnapshot& snapshot) {
    const auto& dc = requireJsonObjectField(config, "data_collection", "ai_config.json");

    snapshot.data_collection_clear_merged_cache_on_merge =
        getRequiredJsonValue<bool>(dc, "clear_merged_cache_on_merge", "data_collection");
    snapshot.data_collection_max_new_entries_per_run =
        getRequiredJsonValue<int>(dc, "max_new_entries_per_run", "data_collection");

    return true;
}

// Populate subprocess-owned snapshot fields from ai_config.json.
// Throws std::runtime_error on a type mismatch (Rule 20: fail loud — wrong
// type is NEVER silently coerced). Missing fields default to false. Returns
// true if any subprocess field was found.
inline bool assignSubprocessFields(const nlohmann::json& config, AiConfigSnapshot& snapshot) {
    const auto& subp = requireJsonObjectField(config, "subprocess", "ai_config.json");
    const auto& tok = requireJsonObjectField(subp, "tokenizer", "subprocess");
    snapshot.subprocess_tokenizer_only_mode =
        getRequiredJsonValue<bool>(tok, "only_mode", "subprocess.tokenizer");
    return true;
}

} // namespace detail

inline void validateAiConfigDocument(const nlohmann::json& config) {
    if (!config.is_object()) {
        throw std::runtime_error("ai_config.json: root document must be an object");
    }

    const auto& pathsNode = requireJsonObjectField(config, "paths", "ai_config.json");
    const auto& grimTextPaths = requireJsonObjectField(pathsNode, "grim_text", "paths");
    static const char* REQUIRED_GRIM_PATHS[] = {
        "vocab", "model", "training_data", "checkpoints", "collected",
        "directory_collection", "verified", "logs", "training_status",
        "collector_log", "source_config", "model_store"
    };
    for (const char* key : REQUIRED_GRIM_PATHS) {
        requireJsonField(grimTextPaths, key, "paths.grim_text");
    }

    const auto& training = detail::requireTrainingObject(config);
    requireJsonField(training, "current_model_training", "training");
    requireJsonField(training, "current_curriculum", "training");
    const auto& trainConfig = detail::requireTrainingConfigObject(config);

    std::vector<std::string> missing;
    detail::appendMissingTrainingConfigPaths(trainConfig, missing);
    if (!missing.empty()) {
        std::ostringstream oss;
        oss << "FATAL: ai_config.json training.config missing " << missing.size() << " required fields:\n";
        for (const auto& m : missing) {
            oss << "  - " << m << "\n";
        }
        oss << "\nRule 20: authored runtime parameters and feature enables MUST be explicitly set.\n";
        oss << "Formula-derived values belong to HyperParameters_GPU.hpp; runtime defaults are forbidden.";
        throw std::runtime_error(oss.str());
    }

    const auto& tok = requireJsonObjectField(config, "tokenizer", "ai_config.json");
    static const char* REQUIRED_TOKENIZER_FIELDS[] = {
        "vocab_size", "max_vocab_size", "max_length", "character_coverage",
        "min_cleaned_text_length", "min_subword_freq", "prune_during_mining",
        "enable_parallel_subword_mining", "subword_mining_workers",
        "subword_mining_max_bytes", "model_type", "add_bos", "add_eos",
        "unk_token", "pad_token", "bos_token", "eos_token",
        "enable_nfkc_normalization", "enable_lowercasing",
        "enable_parallel_tokenization", "parallel_threshold",
        "enable_byte_fallback", "expected_checksum", "save_text_vocab",
        "vocab_score_multiplier"
    };
    for (const char* key : REQUIRED_TOKENIZER_FIELDS) {
        requireJsonField(tok, key, "tokenizer");
    }
    requireJsonArrayField(tok, "special_tokens", "tokenizer");
    const auto& tokenizerScratch = requireJsonObjectField(tok, "scratch_block_reasoning", "tokenizer");
    requireJsonField(tokenizerScratch, "enabled", "tokenizer.scratch_block_reasoning");
    requireJsonField(tokenizerScratch, "detect_numbers", "tokenizer.scratch_block_reasoning");

    const auto& dc = requireJsonObjectField(config, "data_collection", "ai_config.json");
    requireJsonField(dc, "clear_merged_cache_on_merge", "data_collection");
    requireJsonField(dc, "max_new_entries_per_run", "data_collection");

    const auto& subp = requireJsonObjectField(config, "subprocess", "ai_config.json");
    const auto& subTok = requireJsonObjectField(subp, "tokenizer", "subprocess");
    requireJsonField(subTok, "only_mode", "subprocess.tokenizer");
}

/// Derive warmup_steps and dependent fields once estimated_total_steps is known (Phase2).
/// Must be called after loadAiConfigSnapshot() and before the training loop.
inline void deriveWarmupSteps(TrainingHyperparameters& params, int estimated_total_steps) {
    if (estimated_total_steps <= 0)
        throw std::runtime_error("deriveWarmupSteps: estimated_total_steps must be > 0, got " + std::to_string(estimated_total_steps));
    if (params.warmup_fraction <= 0.0f || params.warmup_fraction >= 1.0f)
        throw std::runtime_error("deriveWarmupSteps: warmup_fraction must be in (0, 1), got " + std::to_string(params.warmup_fraction));

    params.warmup_steps = std::max(1, static_cast<int>(params.warmup_fraction * estimated_total_steps));
    params.architecture.mtp_alpha_warmup_steps = params.warmup_steps;
    params.telemetry_warmup_steps = params.warmup_steps;
    params.architecture.execution_block_gate_warmup_steps = params.warmup_steps;
}

inline std::optional<AiConfigSnapshot> loadAiConfigSnapshot(const std::string& configPath) {
    auto resolved_path = detail::resolveAiConfigPath(configPath);

    std::ifstream configFile(resolved_path);
    if (!configFile.is_open()) {
        throw std::runtime_error(
            "loadAiConfigSnapshot: could not open ai_config.json at: " + resolved_path.string());
    }

    nlohmann::json config;
    configFile >> config;
    validateAiConfigDocument(config);

    AiConfigSnapshot snapshot;
    snapshot.config_path = resolved_path;
    snapshot.document = std::move(config);
    snapshot.has_grim_paths = detail::assignGrimTextPathFields(snapshot.document, snapshot);
    snapshot.has_training = detail::assignTrainingHyperparametersFromDocument(snapshot.document, snapshot.hyperparameters);
    if (snapshot.has_training) {
        ::GRIM::HyperParameters::deriveComputedTrainingHyperparameters(snapshot.hyperparameters);
    }
    const auto& tok = requireJsonObjectField(snapshot.document, "tokenizer", "ai_config.json");
    snapshot.assignTokenizerFields(tok);
    snapshot.has_data_collection = detail::assignDataCollectionFields(snapshot.document, snapshot);
    snapshot.has_subprocess = detail::assignSubprocessFields(snapshot.document, snapshot);
    return snapshot;
}

/**
 * @brief Get the training status file path
 * 
 * This returns the path where train_gpu.exe writes its real-time status
 * and where training_control_server.exe reads it from.
 * 
 * Loads from ai_config.json paths.grim_text.training_status if available,
 * otherwise falls back to automatic discovery.
 * 
 * @return Absolute path to training_status.fb
 */
inline std::string getTrainingStatusFilePath() {
    auto snapshot = loadAiConfigSnapshot();
    if (!snapshot) {
        throw std::runtime_error("getTrainingStatusFilePath: failed to load ai_config.json");
    }
    if (!snapshot->has_grim_paths || snapshot->grim_text_training_status.empty()) {
        throw std::runtime_error(
            "getTrainingStatusFilePath: paths.grim_text.training_status missing from ai_config.json");
    }
    return snapshot->grim_text_training_status;
}

/**
 * @brief Get the checkpoints directory path
 * 
 * This returns the directory where data collection checkpoints are stored.
 * 
 * Loads from ai_config.json paths.grim_text.checkpoints if available,
 * otherwise falls back to automatic discovery.
 * 
 * @return Absolute path to checkpoints directory
 */
inline std::string getCheckpointDir() {
    auto snapshot = loadAiConfigSnapshot();
    if (!snapshot) {
        throw std::runtime_error("getCheckpointDir: failed to load ai_config.json");
    }
    if (!snapshot->has_grim_paths || snapshot->grim_text_checkpoints.empty()) {
        throw std::runtime_error(
            "getCheckpointDir: paths.grim_text.checkpoints missing from ai_config.json");
    }
    return snapshot->grim_text_checkpoints;
}

/**
 * Get the collector log file path from ai_config.json.
 * Falls back to automatic discovery if not in config.
 */
inline std::string getCollectorLogPath() {
    auto snapshot = loadAiConfigSnapshot();
    if (!snapshot) {
        throw std::runtime_error("getCollectorLogPath: failed to load ai_config.json");
    }
    if (!snapshot->has_grim_paths || snapshot->grim_text_collector_log.empty()) {
        throw std::runtime_error(
            "getCollectorLogPath: paths.grim_text.collector_log missing from ai_config.json");
    }
    return snapshot->grim_text_collector_log;
}

/**
 * Get the collected data directory from ai_config.json.
 * Falls back to automatic discovery if not in config.
 */
inline std::string getCollectedDir() {
    auto snapshot = loadAiConfigSnapshot();
    if (!snapshot) {
        throw std::runtime_error("getCollectedDir: failed to load ai_config.json");
    }
    if (!snapshot->has_grim_paths || snapshot->grim_text_collected.empty()) {
        throw std::runtime_error(
            "getCollectedDir: paths.grim_text.collected missing from ai_config.json");
    }
    return snapshot->grim_text_collected;
}

/**
 * Get the verified data directory from ai_config.json.
 * Falls back to automatic discovery if not in config.
 */
inline std::string getVerifiedDir() {
    auto snapshot = loadAiConfigSnapshot();
    if (!snapshot) {
        throw std::runtime_error("getVerifiedDir: failed to load ai_config.json");
    }
    if (!snapshot->has_grim_paths || snapshot->grim_text_verified.empty()) {
        throw std::runtime_error(
            "getVerifiedDir: paths.grim_text.verified missing from ai_config.json");
    }
    return snapshot->grim_text_verified;
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
        auto snapshot = loadAiConfigSnapshot();
        if (!snapshot) {
            throw std::runtime_error("getActualVocabSize: failed to load ai_config.json");
        }
        if (!snapshot->has_grim_paths || snapshot->grim_text_vocab.empty()) {
            throw std::runtime_error(
                "getActualVocabSize: paths.grim_text.vocab missing from ai_config.json");
        }
        path = snapshot->grim_text_vocab;
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
