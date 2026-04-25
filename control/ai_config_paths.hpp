#pragma once

//======================================================//
// AI CONFIG ORGANIZATION
//
// This file defines the C++ structs that parse ai_config.json.
// The JSON→C++ mapping is:
//
// JSON Section                → C++ Struct
// ----------------------------------------
// paths.grim_text            → GrimTextPaths
// training.config            → TrainingHyperparameters (incl. execution_block, scratch_blocks, …)
// tokenizer                  → TokenizerConfig
// data_collection            → DataCollectionConfig
//
// RULE: All runtime defaults in TrainingHyperparameters MUST
// match compile-time defaults in HyperParameters_GPU.hpp.
// The JSON overrides those defaults at runtime.
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

/**
 * @brief Load GRIM-text paths from ai_config.json
 * 
 * This function reads the centralized ai_config.json file that is written by
 * the UI Training Panel in GRIM.exe and provides the absolute paths for:
 * - vocab: Vocabulary file path
 * - model: Model file path
 * - training_data: Training data file path (.grmt)
 * - checkpoints: Checkpoints directory path
 * - logs: Logs directory path
 * 
 * All components (training_control_server, train_gpu, data_collection_server)
 * should use this function to ensure they read the same paths that the UI writes.
 * 
 * @param configPath Path to ai_config.json (defaults to "./ai_config.json")
 * @return true if paths were loaded successfully, false otherwise
 */
struct GrimTextPaths {
    std::string vocab;
    std::string model;
    std::string training_data;
    std::string checkpoints;
    std::string collected;
    std::string directory_collection;
    std::string verified;
    std::string logs;
    std::string training_status;
    std::string merge_checkpoints_exe;
    std::string collector_log;
    std::string source_config;  // Path to source_data.json for data collection
    std::string model_store;   // Path to model store directory for per-model configs
    
    bool isValid() const {
        // At minimum, we need vocab and training_data to do any work
        return !vocab.empty() && !training_data.empty();
    }
    
    void printPaths() const {
        std::cout << "[Config] GRIM-text paths from ai_config.json:" << std::endl;
        std::cout << "  vocab: " << (vocab.empty() ? "(not set)" : vocab) << std::endl;
        std::cout << "  model: " << (model.empty() ? "(not set)" : model) << std::endl;
        std::cout << "  training_data: " << (training_data.empty() ? "(not set)" : training_data) << std::endl;
        std::cout << "  checkpoints: " << (checkpoints.empty() ? "(not set)" : checkpoints) << std::endl;
        std::cout << "  collected: " << (collected.empty() ? "(not set)" : collected) << std::endl;
        std::cout << "  directory_collection: " << (directory_collection.empty() ? "(not set)" : directory_collection) << std::endl;
        std::cout << "  verified: " << (verified.empty() ? "(not set)" : verified) << std::endl;
        std::cout << "  logs: " << (logs.empty() ? "(not set)" : logs) << std::endl;
        std::cout << "  training_status: " << (training_status.empty() ? "(not set)" : training_status) << std::endl;
        std::cout << "  merge_checkpoints_exe: " << (merge_checkpoints_exe.empty() ? "(not set)" : merge_checkpoints_exe) << std::endl;
        std::cout << "  collector_log: " << (collector_log.empty() ? "(not set)" : collector_log) << std::endl;
        std::cout << "  source_config: " << (source_config.empty() ? "(not set)" : source_config) << std::endl;
        std::cout << "  model_store: " << (model_store.empty() ? "(not set)" : model_store) << std::endl;
    }
};

/**
 * @brief Data collection configuration
 * 
 * Controls behavior of data collection pipeline (collection, verification, merging)
 */
struct DataCollectionConfig {
    bool clear_merged_cache_on_merge = false;  // Whether to clear merged_verified_cache.jsonl on merge
    int max_new_entries_per_run = 5000;        // Global limit: stop after collecting this many NEW entries
};

/**
 * @brief Tokenizer configuration from ai_config.json
 * 
 * This loads the tokenizer configuration (vocab_size, max_length, model_type, etc.)
 * from the tokenizer section of ai_config.json.
 * 
 * Note: This mirrors GRIM::TokenizerConfig from Tokenizer_GPU.hpp
 * All values MUST be loaded from ai_config.json - no hardcoded defaults
 */
struct TokenizerConfig {
    int vocab_size = 50000;  // Target vocab size for training new tokenizer (actual size comes from vocab.bin)
    int max_vocab_size = 0;  // Hard cap on loaded vocab (0 = no cap, >0 = keep top-K most frequent tokens)
    int max_length = 8192;
    int min_subword_freq = 3;  // Minimum frequency for subwords to be included in vocab
    bool prune_during_mining = false;  // Enable memory pruning during subword mining (disable if RAM is plentiful)
    bool enable_parallel_subword_mining = true;  // Parallelize subword mining during vocab training
    int subword_mining_workers = 0;  // 0 = auto, >0 fixed worker count
    size_t subword_mining_max_bytes = 0;  // 0 = use tokenizer default ceap
    std::string model_type = "unibytes";
    std::vector<std::string> special_tokens = {"<pad>", "<unk>", "<s>", "</s>"};
    bool add_bos = true;
    bool add_eos = true;
    std::string unk_token = "<unk>";
    std::string pad_token = "<pad>";
    std::string bos_token = "<s>";
    std::string eos_token = "</s>";
    bool enable_nfkc_normalization = true;
    bool enable_lowercasing = true;
    bool enable_parallel_tokenization = true;
    int parallel_threshold = 1000;
    bool enable_byte_fallback = true;
    uint32_t expected_checksum = 0;
    bool save_text_vocab = true;  // Also save human-readable .txt alongside .bin
    float vocab_score_multiplier = 1.0f;  // Multiply all vocab scores by this value on save (experiment knob)
};

/**
 * @brief Subprocess coordinator configuration
 *
 * Read by the primitive subprocess coordinator (training/Subprocess/) BEFORE
 * any training phase begins. The fields here aggregate the JSON keys consumed
 * by the coordinator across multiple top-level sections (subprocess.* and
 * training.config.*). This is the single, centralized C++ view those flags;
 * Subprocess/ code MUST go through this struct (via loadAiConfigSnapshot /
 * loadSubprocessConfig) and MUST NOT raw-parse ai_config.json itself.
 */
struct SubprocessConfig {
    // subprocess.tokenizer.only_mode
    // When true, train_gpu spawns the train_tokenizer subprocess, waits for
    // completion, and exits cleanly without entering Phase 1. When false,
    // tokenizer training runs as a normal pre-Phase-1 step and training
    // continues on success.
    bool tokenizer_only_mode = false;

    // training.config.force_rebuild_vocab
    // Mirrored here (and ONLY here, post-refactor) because the train_tokenizer
    // subprocess wrapper is the sole consumer.
    bool tokenizer_force_rebuild_vocab = false;
};

struct AiConfigSnapshot {
    std::filesystem::path config_path;
    nlohmann::json document;
    GrimTextPaths grim_paths;
    TrainingHyperparameters hyperparameters;
    TokenizerConfig tokenizer_config;
    DataCollectionConfig data_collection_config;
    SubprocessConfig subprocess_config;
    bool has_grim_paths = false;
    bool has_training = false;
    bool has_tokenizer = false;
    bool has_data_collection = false;
    bool has_subprocess = false;
};

inline std::optional<AiConfigSnapshot> loadAiConfigSnapshot(const std::string& configPath = "ai_config.json");

namespace detail {

inline std::optional<std::filesystem::path> resolveAiConfigPath(const std::string& configPath) {
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

    return std::nullopt;
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

inline bool populateGrimTextPathsFromConfig(const nlohmann::json& config, GrimTextPaths& paths) {
    if (!config.contains("paths")) {
        std::cerr << "[Config] WARNING: ai_config.json does not contain 'paths' section" << std::endl;
        return false;
    }

    const auto& pathsNode = config["paths"];
    if (!pathsNode.contains("grim_text")) {
        std::cerr << "[Config] WARNING: ai_config.json does not contain 'paths.grim_text' section" << std::endl;
        return false;
    }

    const auto& grimTextPaths = pathsNode["grim_text"];
    if (!grimTextPaths.is_object()) {
        std::cerr << "[Config] WARNING: 'paths.grim_text' is not an object" << std::endl;
        return false;
    }

    // Get GRIM root for resolving relative paths
    std::filesystem::path grimRoot = resolveGrimRoot();
    
    auto assignIfPresent = [&](const char* key, std::string& field) {
        if (grimTextPaths.contains(key) && grimTextPaths[key].is_string()) {
            std::string pathStr = grimTextPaths[key].get<std::string>();
            std::filesystem::path path(pathStr);
            
            // If path is relative, resolve it relative to GRIM root
            if (path.is_relative()) {
                field = (grimRoot / path).string();
            } else {
                // Path is already absolute, use as-is
                field = pathStr;
            }
        }
    };

    assignIfPresent("vocab", paths.vocab);
    assignIfPresent("model", paths.model);
    assignIfPresent("training_data", paths.training_data);
    assignIfPresent("checkpoints", paths.checkpoints);
    assignIfPresent("collected", paths.collected);
    assignIfPresent("directory_collection", paths.directory_collection);
    assignIfPresent("verified", paths.verified);
    assignIfPresent("logs", paths.logs);
    assignIfPresent("training_status", paths.training_status);
    assignIfPresent("merge_checkpoints_exe", paths.merge_checkpoints_exe);
    assignIfPresent("collector_log", paths.collector_log);
    assignIfPresent("source_config", paths.source_config);
    assignIfPresent("model_store", paths.model_store);

    return true;
}

template <typename FieldType>
inline void assignTrainingField(FieldType& field, const nlohmann::json& node, const char* key) {
    auto it = node.find(key);
    if (it != node.end() && !it->is_null()) {
        field = it->get<std::decay_t<FieldType>>();
    }
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
 * @brief Set sensible defaults for all non-base hyperparameters.
 * Called BEFORE JSON parsing, so JSON values override these defaults.
 * Future: Replace this function body with a trained hyperparameter prediction model.
 * @param params The hyperparameters struct to populate with defaults
 */
inline void setDefaultHyperparameters(TrainingHyperparameters& params) {
    // ── Auto stop ──
    params.auto_stop_plateau_patience = 18;
    params.auto_stop_plateau_min_delta = 0.004f;
    params.auto_stop_high_loss_threshold = 6.0f;
    params.auto_stop_high_loss_patience = 12;

    // ── Soft restart ──
    params.soft_restart_loss_increase_threshold = 3.0f;
    params.soft_restart_max_step_window = 50;
    params.soft_restart_cooldown_steps = 200;

    // ── Single batch overfit ──
    params.single_batch_overfit_max_steps = 1000;

    // ── Guess aux ──
    params.guess_aux_lambda = 0.25f;
    params.guess_aux_min_confidence = 0.7f;

    // ── Shuffle ──
    params.shuffle_train_epochs = 1;

    // ── Embedding freeze ──
    params.embedding_freeze_after_step = 0;

    // ── Scratch blocks ──
    params.scratch_num_blocks = 4;
    params.scratch_write_combined = true;

    // ── Scratch block reasoning (model fields, live on architecture) ──
    params.architecture.scratch_block_max_atoms = 8192;
    params.architecture.scratch_block_atom_scale = 1.0f;

    // ── Telemetry control ──
    params.telemetry_spike_mild_threshold = 3.0f;
    params.telemetry_spike_moderate_threshold = 5.0f;
    params.telemetry_spike_severe_threshold = 10.0f;
    params.telemetry_moderate_grad_scale = 0.5f;
    params.telemetry_moderate_cooldown_extension = 3;
    params.telemetry_min_grad_for_nonzero_loss = 1e-10f;
    params.telemetry_loss_threshold_for_grad_check = 0.01f;
    params.telemetry_max_consecutive_zero_grad_steps = 0;
    params.telemetry_seq_len_regime_change_threshold = 0.3f;
    params.telemetry_regime_change_suppression_steps = 2;
    params.telemetry_volatility_damping_threshold = 150.0f;
    params.telemetry_max_volatility_damping = 0.9f;
    params.telemetry_gradient_decay_threshold = 0.0f;
    params.telemetry_max_decay_boost = 1.0f;
    params.telemetry_progress_boost_threshold = 100.0f;
    params.telemetry_max_progress_boost = 1.0f;
    params.telemetry_outlier_frequency_trigger = 0.95f;
    params.telemetry_outlier_persistence_trigger = 0.9f;
    params.telemetry_anchor_drift_sigma_multiplier = 5.0f;
    params.telemetry_soft_restart_cooldown_steps = 100000;
    params.telemetry_baseline_stabilization_steps = 100;
    params.telemetry_plateau_noise_patience = 30;
    params.telemetry_plateau_noise_variance_threshold = 0.008f;
    params.telemetry_plateau_noise_std = 0.1f;
    params.telemetry_plateau_noise_proportional = true;
    params.telemetry_plateau_noise_cooldown = 10;
    params.telemetry_plateau_noise_max_per_epoch = 3;

    // ── Telemetry lattice (TelemetryLattice construction params) ──
    // num_streams=48: 0-4 core, 5-8 rho, 9-13 adam, 14-20 exec block, 21-26 EB/SB injection,
    // 27-30 PBM, 31-34 rho raw, 35-37 RMS gamma, 38 rho rms-spread, 39-44 h<->W alignment,
    // 45-46 unigram-dir cosine, 47 lm_head_w_rms_rms
    params.telemetry_lattice_num_levels = 8;     // k in [0,7]: strides [1,2,4,8,16,32,64,128]
    params.telemetry_lattice_num_streams = 48;
    params.telemetry_lattice_beta_mu = 0.95f;
    params.telemetry_lattice_beta_a = 0.995f;
    params.telemetry_lattice_beta_delta = 0.90f;
    params.telemetry_lattice_beta_r = 0.85f;
    params.telemetry_lattice_beta_run = 0.80f;
    params.telemetry_lattice_beta_v = 0.90f;
    params.telemetry_lattice_k_out0 = 2.5f;
    params.telemetry_lattice_alpha_v = 1.5f;
    params.telemetry_lattice_epsilon = 1e-7f;
    params.telemetry_lattice_strict_mode = true;  // Rule 20: fail loud on NaN/Inf

    // ── Loss sub-parameters ──
    params.loss_label_smoothing_epsilon = 0.05f;
    params.loss_focal_gamma = 0.75f;
    params.loss_focal_alpha = 1.0f;
    params.loss_preference_beta = 0.1f;
    params.loss_distillation_temperature = 1.0f;
    params.loss_distillation_lambda = 0.5f;
    params.loss_entropy_reg_lambda = 0.0f;
    params.loss_class_balanced_beta = 0.5f;

    // ── LM head centering (model fields, live on architecture) ──
    params.architecture.pc1_power_iters = 5;

    // ── Layer scale (model field, lives on architecture) ──
    params.architecture.layer_scale_init = 1.0f;

    // ── Execution block tuning (model fields, live on architecture) ──
    params.architecture.execution_block_cross_attn_topk = 2;
    params.architecture.execution_block_usage_decay = 0.95f;
    params.architecture.execution_block_diversity_kappa = 2.0f;
    params.architecture.execution_block_temp_start = 1.5f;
    params.architecture.execution_block_temp_end = 0.5f;
    params.architecture.execution_block_temp_schedule = 0;
    params.architecture.execution_block_entropy_weight = 0.01f;
    params.architecture.step_x_multiplier = 2.0f;
    params.architecture.step_y_multiplier = 3.0f;
    params.architecture.step_y_overrides_x = false;
    params.architecture.entropy_aux_weight = 0.01f;
    params.architecture.value_match_epsilon = 0.0001f;
    params.architecture.final_slot_consistency_weight = 0.1f;
    params.architecture.execution_block_transition_hard_threshold = 0.0f;
    params.architecture.execution_block_causal_w1_transition = 1.5f;
    params.architecture.div_invalid_penalty_weight = 0.5f;
    params.architecture.div_magnitude_penalty_weight = 0.1f;
    params.architecture.arg_reinforce_weight = 0.0f;
    params.architecture.arg_reinforce_baseline_decay = 0.99f;
    params.architecture.structured_ce_weight = 0.1f;
    params.architecture.selector_d_selector = 64;
    params.architecture.selector_selection_margin = 1.0f;
    params.architecture.selector_supervision_weight = 1.0f;

    // ── Multi-token prediction (k/alpha live on architecture; monitor flag is training-only) ──
    params.architecture.mtp_k = 3;
    params.architecture.mtp_alpha = 0.2f;
    params.mtp_log_ratio_monitor = true;

    // ── Diagnostics (hardcoded-hidden lives on architecture as enum) ──
    params.architecture.hardcoded_hidden_pattern = ::GRIM::HyperParameters::LanguageModelConfig::HardcodedPattern::DISABLED;
    params.architecture.hardcoded_log_every_n_batches = 1;
    params.prediction_comparison_interval = 100;
    params.prediction_comparison_top_k = 5;
    params.prediction_comparison_max_positions = 8;
    params.prediction_comparison_log_path = "resources/models/GRIM-text/training/prediction_comparison.log";
    params.logit_update_trace_interval = 50;
    params.attention_diag_layer = -1;
    params.attention_diag_head = 0;
}

/**
 * @brief Validate required (base+enable) fields exist in training.config JSON
 * @param trainConfig The training.config JSON object
 * @throws std::runtime_error listing all missing required fields
 *
 * Rule 20: Base parameters and feature enables MUST be explicitly set.
 * Non-base parameters get sensible defaults from setDefaultHyperparameters().
 */
inline void validateTrainingConfigJson(const nlohmann::json& trainConfig) {
    static const std::vector<std::string> REQUIRED = {
        // Core training
        "epochs", "seed", "batch_size", "gradient_accumulation_steps",
        "batch_strategy", "learning_rate", "weight_decay",
        "per_token_grad_scale", "warmup_fraction", "max_seq_len", "log_interval",
        "atom_stats_interval", "atom_stats_max_seqs",
        "validation_interval", "checkpoint_interval", "use_gpu", "use_flash_attention",
        
        // Feature enables
        "cosine_decay.enabled",
        "single_batch.enabled",
        "soft_restart.enabled",
        "auto_stop.enabled",
        "guess_aux.enabled",
        "shuffle.enabled",
        
        "telemetry_control.enabled",
        "stability_overrides_enabled",
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
        "loss.label_smoothing.enabled",
        "loss.focal.enabled",
        "loss.preference.enabled",
        "loss.distillation.enabled",
        "loss.masking.enabled", "loss.masking.tag",

        // Execution block structural (architecture choices)
        "execution_block.execution_first_type_only",
        "execution_block.layer",
        "execution_block.num_ops",
        "execution_block.num_slots",
        "execution_block.num_steps",
        "execution_block.d_type",
        "execution_block.structured_ce_enabled",
        "execution_block.selector.enabled",

        // LM head centering choices
        "lm_head_centering.center_hidden_states",
        "lm_head_centering.center_logits",
        "lm_head_centering.center_encoder_residuals",
        "lm_head_centering.project_out_pc1",

        // Logging
        "log_recorder.enabled", "log_recorder.default_level",
        "telemetry_control.logging.verbose",
        "telemetry_control.logging.fail_loud_on_accumulation_bug",
        "telemetry_control.plateau_noise.enabled",

        // CUDA execution
        "cuda_execution.single_stream_mode",
        "cuda_execution.disable_async_frees",
        "cuda_execution.synchronize_after_kernels",
    };
    
    std::vector<std::string> missing;
    
    for (const auto& path : REQUIRED) {
        if (!jsonPathExists(trainConfig, path)) {
            missing.push_back(path);
        }
    }
    
    // Check gradient clip (accepts either name)
    if (!trainConfig.contains("grad_clip_norm") && !trainConfig.contains("gradient_clip")) {
        missing.push_back("grad_clip_norm (or gradient_clip)");
    }
    
    if (!missing.empty()) {
        std::ostringstream oss;
        oss << "FATAL: ai_config.json training.config missing " << missing.size() << " required fields:\n";
        for (const auto& m : missing) {
            oss << "  - " << m << "\n";
        }
        oss << "\nRule 20: Base parameters and feature enables MUST be explicitly set.\n";
        oss << "Non-base parameters get sensible defaults from setDefaultHyperparameters().";
        throw std::runtime_error(oss.str());
    }
}

inline void applyTrainingConfigObject(const nlohmann::json& trainConfig, TrainingHyperparameters& params) {
    if (!trainConfig.is_object()) {
        return;
    }
    
    assignTrainingField(params.epochs, trainConfig, "epochs");
    assignTrainingField(params.seed, trainConfig, "seed");
    assignTrainingField(params.batch_size, trainConfig, "batch_size");
    assignTrainingField(params.gradient_accumulation_steps, trainConfig, "gradient_accumulation_steps");
    assignTrainingField(params.batch_strategy, trainConfig, "batch_strategy");
    assignTrainingField(params.learning_rate, trainConfig, "learning_rate");
    assignTrainingField(params.weight_decay, trainConfig, "weight_decay");
    assignTrainingField(params.grad_clip_norm, trainConfig, "gradient_clip");
    assignTrainingField(params.grad_clip_norm, trainConfig, "grad_clip_norm");
    assignTrainingField(params.per_token_grad_scale, trainConfig, "per_token_grad_scale");
    assignTrainingField(params.architecture.max_seq_len, trainConfig, "max_seq_len");
    // min_seq_valid_tokens: derived as max_seq_len / 4 (see deriveComputedHyperparameters)
    // architecture.min_seq_len_for_flash: derived as max_seq_len / 4 (see deriveComputedHyperparameters)
    assignTrainingField(params.warmup_fraction, trainConfig, "warmup_fraction");
    if (auto it = trainConfig.find("cosine_decay"); it != trainConfig.end() && it->is_object()) {
        params.cosine_decay_enabled = it->value("enabled", false);
        params.cosine_warm_restarts = it->value("warm_restarts", false);
        // cosine_decay_min_lr: derived as learning_rate * 0.1 (see deriveComputedHyperparameters)
    } else {
        params.cosine_decay_enabled = false;
        params.cosine_warm_restarts = false;
    }
    assignTrainingField(params.log_interval, trainConfig, "log_interval");
    assignTrainingField(params.atom_stats_interval, trainConfig, "atom_stats_interval");
    assignTrainingField(params.atom_stats_max_seqs, trainConfig, "atom_stats_max_seqs");
    assignTrainingField(params.validation_interval, trainConfig, "validation_interval");
    assignTrainingField(params.checkpoint_interval, trainConfig, "checkpoint_interval");
    assignTrainingField(params.architecture.use_gpu, trainConfig, "use_gpu");
    assignTrainingField(params.architecture.use_flash_attention, trainConfig, "use_flash_attention");
    // architecture.min_seq_len_for_flash: derived from max_seq_len (see deriveComputedHyperparameters)

    if (auto it = trainConfig.find("soft_restart"); it != trainConfig.end()) {
        const auto& soft = *it;
        if (soft.is_boolean()) {
            params.soft_restart_enabled = soft.get<bool>();
        } else if (soft.is_object()) {
            params.soft_restart_enabled = soft.value("enabled", params.soft_restart_enabled);
            params.soft_restart_loss_increase_threshold =
                soft.value("loss_increase_threshold", params.soft_restart_loss_increase_threshold);
            params.soft_restart_max_step_window = soft.value("max_step_window", params.soft_restart_max_step_window);
            params.soft_restart_cooldown_steps = soft.value("cooldown_steps", params.soft_restart_cooldown_steps);
        }
    }

    if (auto it = trainConfig.find("auto_stop"); it != trainConfig.end()) {
        const auto& autoStop = *it;
        if (autoStop.is_boolean()) {
            params.auto_stop_enabled = autoStop.get<bool>();
        } else if (autoStop.is_object()) {
            params.auto_stop_enabled = autoStop.value("enabled", params.auto_stop_enabled);
            params.auto_stop_plateau_patience = autoStop.value("plateau_patience", params.auto_stop_plateau_patience);
            params.auto_stop_plateau_min_delta = autoStop.value("plateau_min_delta", params.auto_stop_plateau_min_delta);
            params.auto_stop_high_loss_threshold = autoStop.value("high_loss_threshold", params.auto_stop_high_loss_threshold);
            params.auto_stop_high_loss_patience = autoStop.value("high_loss_patience", params.auto_stop_high_loss_patience);
        }
    }

    if (auto it = trainConfig.find("atom_stats"); it != trainConfig.end() && it->is_object()) {
        const auto& atom_stats = *it;
        params.atom_stats_interval = atom_stats.value("interval", params.atom_stats_interval);
        params.atom_stats_max_seqs = atom_stats.value("max_seqs", params.atom_stats_max_seqs);
    }

    if (auto it = trainConfig.find("single_batch"); it != trainConfig.end()) {
        const auto& single = *it;
        if (single.is_boolean()) {
            params.single_batch_overfit_enabled = single.get<bool>();
        } else if (single.is_object()) {
            params.single_batch_overfit_enabled = single.value("enabled", params.single_batch_overfit_enabled);
            params.single_batch_overfit_max_steps = single.value("max_steps", params.single_batch_overfit_max_steps);
        }
    }

    if (auto it = trainConfig.find("shuffle"); it != trainConfig.end()) {
        const auto& shuffle = *it;
        if (shuffle.is_boolean()) {
            params.shuffle_train_enabled = shuffle.get<bool>();
        } else if (shuffle.is_object()) {
            params.shuffle_train_enabled = shuffle.value("enabled", params.shuffle_train_enabled);
            params.shuffle_train_epochs = shuffle.value("epochs", params.shuffle_train_epochs);
        }
        if (params.shuffle_train_epochs < 0) {
            params.shuffle_train_epochs = 0;
        }
    }

    if (auto it = trainConfig.find("guess_aux"); it != trainConfig.end()) {
        const auto& guess = *it;
        if (guess.is_boolean()) {
            params.guess_aux_enabled = guess.get<bool>();
        } else if (guess.is_object()) {
            params.guess_aux_enabled = guess.value("enabled", params.guess_aux_enabled);
            params.guess_aux_lambda = guess.value("lambda", params.guess_aux_lambda);
            params.guess_aux_min_confidence = guess.value("min_confidence", params.guess_aux_min_confidence);
        }
    }

    // Load telemetry control configuration (includes plateau noise)
    if (auto it = trainConfig.find("telemetry_control"); it != trainConfig.end()) {
        const auto& tc = *it;
        if (tc.is_boolean()) {
            params.telemetry_control_enabled = tc.get<bool>();
        } else if (tc.is_object()) {
            params.telemetry_control_enabled = tc.value("enabled", params.telemetry_control_enabled);

            if (auto spike_it = tc.find("spike_thresholds"); spike_it != tc.end() && spike_it->is_object()) {
                const auto& spike = *spike_it;
                params.telemetry_spike_mild_threshold = spike.value("mild", params.telemetry_spike_mild_threshold);
                params.telemetry_spike_moderate_threshold = spike.value("moderate", params.telemetry_spike_moderate_threshold);
                params.telemetry_spike_severe_threshold = spike.value("severe", params.telemetry_spike_severe_threshold);
            }

            if (auto resp_it = tc.find("response"); resp_it != tc.end() && resp_it->is_object()) {
                const auto& response = *resp_it;
                params.telemetry_moderate_grad_scale = response.value("moderate_grad_scale", params.telemetry_moderate_grad_scale);
                params.telemetry_moderate_cooldown_extension = response.value("moderate_cooldown_extension", params.telemetry_moderate_cooldown_extension);
            }

            if (auto acc_it = tc.find("accumulation_guard"); acc_it != tc.end() && acc_it->is_object()) {
                const auto& acc = *acc_it;
                params.telemetry_min_grad_for_nonzero_loss = acc.value("min_grad_for_nonzero_loss", params.telemetry_min_grad_for_nonzero_loss);
                params.telemetry_loss_threshold_for_grad_check = acc.value("loss_threshold", params.telemetry_loss_threshold_for_grad_check);
                params.telemetry_max_consecutive_zero_grad_steps = acc.value("max_consecutive_zero_grad_steps", params.telemetry_max_consecutive_zero_grad_steps);
            }

            if (auto regime_it = tc.find("regime_change"); regime_it != tc.end() && regime_it->is_object()) {
                const auto& regime = *regime_it;
                params.telemetry_seq_len_regime_change_threshold = regime.value("seq_len_threshold", params.telemetry_seq_len_regime_change_threshold);
                params.telemetry_regime_change_suppression_steps = regime.value("suppression_steps", params.telemetry_regime_change_suppression_steps);
            }

            if (auto vol_it = tc.find("volatility_damping"); vol_it != tc.end() && vol_it->is_object()) {
                const auto& vol = *vol_it;
                params.telemetry_volatility_damping_threshold = vol.value("threshold", params.telemetry_volatility_damping_threshold);
                params.telemetry_max_volatility_damping = vol.value("max_damping", params.telemetry_max_volatility_damping);
            }

            if (auto decay_it = tc.find("gradient_decay"); decay_it != tc.end() && decay_it->is_object()) {
                const auto& decay = *decay_it;
                params.telemetry_gradient_decay_threshold = decay.value("threshold", params.telemetry_gradient_decay_threshold);
                params.telemetry_max_decay_boost = decay.value("max_boost", params.telemetry_max_decay_boost);
            }

            if (auto boost_it = tc.find("progress_boost"); boost_it != tc.end() && boost_it->is_object()) {
                const auto& boost = *boost_it;
                params.telemetry_progress_boost_threshold = boost.value("threshold", params.telemetry_progress_boost_threshold);
                params.telemetry_max_progress_boost = boost.value("max_boost", params.telemetry_max_progress_boost);
            }

            if (auto out_it = tc.find("outlier"); out_it != tc.end() && out_it->is_object()) {
                const auto& outlier = *out_it;
                params.telemetry_outlier_frequency_trigger = outlier.value("frequency_trigger", params.telemetry_outlier_frequency_trigger);
                params.telemetry_outlier_persistence_trigger = outlier.value("persistence_trigger", params.telemetry_outlier_persistence_trigger);
            }

            if (auto drift_it = tc.find("drift"); drift_it != tc.end() && drift_it->is_object()) {
                const auto& drift = *drift_it;
                params.telemetry_anchor_drift_sigma_multiplier = drift.value("anchor_sigma_multiplier", params.telemetry_anchor_drift_sigma_multiplier);
            }

            if (auto sr_it = tc.find("soft_restart"); sr_it != tc.end() && sr_it->is_object()) {
                const auto& sr = *sr_it;
                params.telemetry_soft_restart_cooldown_steps = sr.value("cooldown_steps", params.telemetry_soft_restart_cooldown_steps);
            }

            if (auto base_it = tc.find("baseline"); base_it != tc.end() && base_it->is_object()) {
                const auto& base = *base_it;
                // telemetry_warmup_steps: derived as warmup_steps (see deriveComputedHyperparameters)
                params.telemetry_baseline_stabilization_steps = base.value("stabilization_steps", params.telemetry_baseline_stabilization_steps);
            }

            if (auto log_it = tc.find("logging"); log_it != tc.end() && log_it->is_object()) {
                const auto& logging = *log_it;
                params.telemetry_verbose_logging = logging.value("verbose", params.telemetry_verbose_logging);
                params.telemetry_fail_loud_on_accumulation_bug = logging.value("fail_loud_on_accumulation_bug", params.telemetry_fail_loud_on_accumulation_bug);
            }
            
            // Plateau noise sub-config
            if (auto pn_it = tc.find("plateau_noise"); pn_it != tc.end() && pn_it->is_object()) {
                const auto& pn = *pn_it;
                params.telemetry_plateau_noise_enabled = pn.value("enabled", params.telemetry_plateau_noise_enabled);
                params.telemetry_plateau_noise_patience = pn.value("patience", params.telemetry_plateau_noise_patience);
                params.telemetry_plateau_noise_variance_threshold = pn.value("variance_threshold", params.telemetry_plateau_noise_variance_threshold);
                params.telemetry_plateau_noise_std = pn.value("noise_std", params.telemetry_plateau_noise_std);
                params.telemetry_plateau_noise_proportional = pn.value("proportional", params.telemetry_plateau_noise_proportional);
                params.telemetry_plateau_noise_cooldown = pn.value("cooldown", params.telemetry_plateau_noise_cooldown);
                params.telemetry_plateau_noise_max_per_epoch = pn.value("max_per_epoch", params.telemetry_plateau_noise_max_per_epoch);
            }

            // Lattice construction sub-config (TelemetryLattice EMA hyperparams)
            if (auto lat_it = tc.find("lattice"); lat_it != tc.end() && lat_it->is_object()) {
                const auto& lat = *lat_it;
                params.telemetry_lattice_num_levels  = lat.value("num_levels",  params.telemetry_lattice_num_levels);
                params.telemetry_lattice_num_streams = lat.value("num_streams", params.telemetry_lattice_num_streams);
                params.telemetry_lattice_beta_mu     = lat.value("beta_mu",     params.telemetry_lattice_beta_mu);
                params.telemetry_lattice_beta_a      = lat.value("beta_a",      params.telemetry_lattice_beta_a);
                params.telemetry_lattice_beta_delta  = lat.value("beta_delta",  params.telemetry_lattice_beta_delta);
                params.telemetry_lattice_beta_r      = lat.value("beta_r",      params.telemetry_lattice_beta_r);
                params.telemetry_lattice_beta_run    = lat.value("beta_run",    params.telemetry_lattice_beta_run);
                params.telemetry_lattice_beta_v      = lat.value("beta_v",      params.telemetry_lattice_beta_v);
                params.telemetry_lattice_k_out0      = lat.value("k_out0",      params.telemetry_lattice_k_out0);
                params.telemetry_lattice_alpha_v     = lat.value("alpha_v",     params.telemetry_lattice_alpha_v);
                params.telemetry_lattice_epsilon     = lat.value("epsilon",     params.telemetry_lattice_epsilon);
                params.telemetry_lattice_strict_mode = lat.value("strict_mode", params.telemetry_lattice_strict_mode);
            }
        }
    }
    // Legacy: simple bool telemetry_control_enabled (backwards compat removed per Rule 20)
    // If you had "telemetry_control_enabled": true, migrate to "telemetry_control": { "enabled": true }

    // Load unified tape logging configuration
    if (auto it = trainConfig.find("logging"); it != trainConfig.end() && it->is_object()) {
        const auto& logCfg = *it;
        params.tape_logging.default_level = logCfg.value("default_level", params.tape_logging.default_level);
        params.tape_logging.equation_csv_enabled = logCfg.value("equation_csv_enabled", params.tape_logging.equation_csv_enabled);
        params.tape_logging.stderr_enabled = logCfg.value("stderr_enabled", params.tape_logging.stderr_enabled);
        params.tape_logging.initial_capacity = logCfg.value("initial_capacity", params.tape_logging.initial_capacity);
        if (logCfg.contains("group_overrides") && logCfg["group_overrides"].is_object()) {
            for (auto& [key, val] : logCfg["group_overrides"].items()) {
                if (val.is_string()) {
                    params.tape_logging.group_overrides[key] = val.get<std::string>();
                }
            }
        }
    }

    // Load Log Recorder configuration
    if (auto it = trainConfig.find("log_recorder"); it != trainConfig.end()) {
        const auto& logRec = *it;
        if (logRec.is_object()) {
            params.log_recorder.enabled = logRec.value("enabled", params.log_recorder.enabled);
            params.log_recorder.default_level = logRec.value("default_level", params.log_recorder.default_level);
            
            if (logRec.contains("modules") && logRec["modules"].is_object()) {
                for (auto& [key, val] : logRec["modules"].items()) {
                    if (val.is_string()) {
                        params.log_recorder.modules[key] = val.get<std::string>();
                    }
                }
            }
            
            // Parse layer logging enables
            if (logRec.contains("layers") && logRec["layers"].is_object()) {
                const auto& layers = logRec["layers"];
                params.log_recorder.layers.embedding = layers.value("embedding", params.log_recorder.layers.embedding);
                params.log_recorder.layers.rms_norm = layers.value("rms_norm", params.log_recorder.layers.rms_norm);
                params.log_recorder.layers.attention = layers.value("attention", params.log_recorder.layers.attention);
                params.log_recorder.layers.feed_forward = layers.value("feed_forward", params.log_recorder.layers.feed_forward);
                params.log_recorder.layers.residual = layers.value("residual", params.log_recorder.layers.residual);
                params.log_recorder.layers.encoding = layers.value("encoding", params.log_recorder.layers.encoding);
                params.log_recorder.layers.serialization = layers.value("serialization", params.log_recorder.layers.serialization);
                params.log_recorder.layers.execution_block = layers.value("execution_block", params.log_recorder.layers.execution_block);
            }
        }
    }
    
    // Load loss options
    if (auto it = trainConfig.find("loss"); it != trainConfig.end() && it->is_object()) {
        const auto& loss_cfg = *it;
        
        if (auto ls_it = loss_cfg.find("label_smoothing"); ls_it != loss_cfg.end()) {
            const auto& ls = *ls_it;
            if (ls.is_boolean()) {
                params.loss_label_smoothing_enabled = ls.get<bool>();
            } else if (ls.is_object()) {
                params.loss_label_smoothing_enabled = ls.value("enabled", params.loss_label_smoothing_enabled);
                params.loss_label_smoothing_epsilon = ls.value("epsilon", params.loss_label_smoothing_epsilon);
            }
        }
        
        if (auto fc_it = loss_cfg.find("focal"); fc_it != loss_cfg.end()) {
            const auto& fc = *fc_it;
            if (fc.is_boolean()) {
                params.loss_focal_enabled = fc.get<bool>();
            } else if (fc.is_object()) {
                params.loss_focal_enabled = fc.value("enabled", params.loss_focal_enabled);
                params.loss_focal_gamma = fc.value("gamma", params.loss_focal_gamma);
                params.loss_focal_alpha = fc.value("alpha", params.loss_focal_alpha);
            }
        }
        
        // Issue #44 FIX: Entropy regularization to prevent mode collapse
        if (auto er_it = loss_cfg.find("entropy_reg"); er_it != loss_cfg.end()) {
            const auto& er = *er_it;
            if (er.is_boolean()) {
                params.loss_entropy_reg_enabled = er.get<bool>();
            } else if (er.is_object()) {
                params.loss_entropy_reg_enabled = er.value("enabled", params.loss_entropy_reg_enabled);
                params.loss_entropy_reg_lambda = er.value("lambda", params.loss_entropy_reg_lambda);
            }
        }

        // Class-balanced loss: reweights per-token loss by 1/freq^β
        if (auto cb_it = loss_cfg.find("class_balanced"); cb_it != loss_cfg.end()) {
            const auto& cb = *cb_it;
            if (cb.is_boolean()) {
                params.loss_class_balanced_enabled = cb.get<bool>();
            } else if (cb.is_object()) {
                params.loss_class_balanced_enabled = cb.value("enabled", params.loss_class_balanced_enabled);
                params.loss_class_balanced_beta = cb.value("beta", params.loss_class_balanced_beta);
            }
        }

        
        if (auto pref_it = loss_cfg.find("preference"); pref_it != loss_cfg.end()) {
            const auto& pref = *pref_it;
            if (pref.is_boolean()) {
                params.loss_preference_enabled = pref.get<bool>();
            } else if (pref.is_object()) {
                params.loss_preference_enabled = pref.value("enabled", params.loss_preference_enabled);
                params.loss_preference_beta = pref.value("beta", params.loss_preference_beta);
            }
        }
        
        if (auto dist_it = loss_cfg.find("distillation"); dist_it != loss_cfg.end()) {
            const auto& dist = *dist_it;
            if (dist.is_boolean()) {
                params.loss_distillation_enabled = dist.get<bool>();
            } else if (dist.is_object()) {
                params.loss_distillation_enabled = dist.value("enabled", params.loss_distillation_enabled);
                params.loss_distillation_temperature = dist.value("temperature", params.loss_distillation_temperature);
                params.loss_distillation_lambda = dist.value("lambda", params.loss_distillation_lambda);
            }
        }
        
        if (auto mask_it = loss_cfg.find("masking"); mask_it != loss_cfg.end()) {
            const auto& mask = *mask_it;
            if (mask.is_boolean()) {
                params.loss_masking_enabled = mask.get<bool>();
            } else if (mask.is_object()) {
                params.loss_masking_enabled = mask.value("enabled", params.loss_masking_enabled);
                if (mask.contains("tag") && mask["tag"].is_string()) {
                    params.loss_masking_tag = mask["tag"].get<std::string>();
                }
            }
        }
    }
    
    // LM Head centering configuration (Issue #37 / #40)
    // Master toggle is training-only (lm_head_centering_enabled). Per-knob fields
    // (center_hidden_states / freeze_final_rms_gamma / center_logits /
    // center_encoder_residuals / project_out_pc1 / pc1_power_iters) live on architecture.
    params.lm_head_centering_enabled = false;  // Default to disabled (standard implementation)
    params.architecture.lm_head_center_hidden_states = false;
    params.architecture.lm_head_freeze_final_rms_gamma = false;  // Default: γ_final is trainable
    params.architecture.center_logits = false;  // Default to disabled (standard implementation)
    params.architecture.center_encoder_residuals = false;  // Default: disabled. Enable to prevent ρ buildup across layers (mode collapse fix).
                                               // Gradient cost: (1-1/n_tokens)^24 ≈ 0.996 for n≈6000 — negligible.
    params.architecture.project_out_pc1 = false;  // Default: disabled (Issue #149)
    params.architecture.pc1_power_iters = 5;
    if (auto it = trainConfig.find("lm_head_centering"); it != trainConfig.end() && it->is_object()) {
        const auto& lmc = *it;
        params.lm_head_centering_enabled = lmc.value("enabled", false);
        params.architecture.lm_head_center_hidden_states = lmc.value("center_hidden_states", false);
        params.architecture.lm_head_freeze_final_rms_gamma = lmc.value("freeze_final_rms_gamma", false);
        params.architecture.center_logits = lmc.value("center_logits", false);
        params.architecture.center_encoder_residuals = lmc.value("center_encoder_residuals", false);
        params.architecture.project_out_pc1 = lmc.value("project_out_pc1", false);
        params.architecture.pc1_power_iters = lmc.value("pc1_power_iters", 5);
    }
    
    // Issue #109: LayerScale (learnable residual scaling from CaiT paper)
    // Reduces correlation buildup between layers by gating sublayer outputs
    params.architecture.use_layer_scale = false;   // Default: disabled (standard residual connections)
    params.architecture.layer_scale_init = 0.1f;   // CaiT paper recommends 0.1 for deeper networks
    if (auto it = trainConfig.find("layer_scale"); it != trainConfig.end() && it->is_object()) {
        const auto& ls = *it;
        params.architecture.use_layer_scale = ls.value("enabled", false);
        params.architecture.layer_scale_init = ls.value("init_value", 0.1f);
    }
    
    // QK-norm: Per-head RMSNorm applied to Q and K after QKV projection, before RoPE.
    // Bounds attention logit magnitudes, prevents entropy collapse in deeper models.
    params.architecture.qk_norm_enabled = false;   // Default: disabled (standard unscaled Q/K)
    if (auto it = trainConfig.find("qk_norm"); it != trainConfig.end() && it->is_object()) {
        const auto& qkn = *it;
        params.architecture.qk_norm_enabled = qkn.value("enabled", false);
    }
    
    // Hardcoded Hidden States Diagnostic (Issue #42)
    using HCP = ::GRIM::HyperParameters::LanguageModelConfig::HardcodedPattern;
    params.architecture.hardcoded_hidden_pattern = HCP::DISABLED;
    params.architecture.hardcoded_log_every_n_batches = 1;
    if (auto it = trainConfig.find("hardcoded_hidden_states"); it != trainConfig.end() && it->is_object()) {
        const auto& hcs = *it;
        if (hcs.value("enabled", false)) {
            std::string pattern_str = hcs.value("pattern", "random_centered");
            if (pattern_str == "random_centered") {
                params.architecture.hardcoded_hidden_pattern = HCP::RANDOM_CENTERED;
            } else if (pattern_str == "orthogonal_w277") {
                params.architecture.hardcoded_hidden_pattern = HCP::ORTHOGONAL_W277;
            } else if (pattern_str == "aligned_w277") {
                params.architecture.hardcoded_hidden_pattern = HCP::ALIGNED_W277;
            } else if (pattern_str == "constant_uniform") {
                params.architecture.hardcoded_hidden_pattern = HCP::CONSTANT_UNIFORM;
            } else if (pattern_str == "zero_mean_sine") {
                params.architecture.hardcoded_hidden_pattern = HCP::ZERO_MEAN_SINE;
            }
            params.architecture.hardcoded_log_every_n_batches = hcs.value("log_every_n_batches", 1);
        }
    }
    
    // Load embedding freeze guard
    if (auto it = trainConfig.find("embedding_freeze"); it != trainConfig.end() && it->is_object()) {
        const auto& ef = *it;
        params.embedding_freeze_enabled = ef.value("enabled", params.embedding_freeze_enabled);
        params.embedding_freeze_after_step = ef.value("freeze_after_step", params.embedding_freeze_after_step);
    }

    // Load optimizer selector.
    // JSON layout (all fields optional; struct defaults from HyperParameters apply):
    //   "optimizer": {
    //     "kind": "adamw" | "radamw",
    //     "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8
    //   }
    if (auto it = trainConfig.find("optimizer"); it != trainConfig.end() && it->is_object()) {
        const auto& opt = *it;
        params.optimizer_kind   = opt.value("kind",    params.optimizer_kind);
        params.optimizer_beta1  = opt.value("beta1",   params.optimizer_beta1);
        params.optimizer_beta2  = opt.value("beta2",   params.optimizer_beta2);
        params.optimizer_epsilon = opt.value("epsilon", params.optimizer_epsilon);
        // Rule 20: validate kind explicitly — fail loud on typo.
        if (params.optimizer_kind != "adamw" && params.optimizer_kind != "radamw") {
            throw std::runtime_error(
                "[ai_config] training.config.optimizer.kind must be \"adamw\" or \"radamw\", got \""
                + params.optimizer_kind + "\"");
        }
    }

    // Load stability overrides - ALWAYS parse values even if disabled
    // (Phase1_Startup copies them unconditionally, so they must be initialized)
    params.stability_overrides_enabled = trainConfig.value("stability_overrides_enabled", params.stability_overrides_enabled);
    if (auto it = trainConfig.find("stability_overrides"); it != trainConfig.end() && it->is_object()) {
        const auto& stab = *it;
        params.stability_override_batch_size = stab.value("batch_size", params.stability_override_batch_size);
        params.stability_override_max_seq_len = stab.value("max_seq_len", params.stability_override_max_seq_len);
        params.stability_override_clip_per_token = stab.value("clip_per_token", params.stability_override_clip_per_token);
        params.stability_override_lr_min = stab.value("lr_min", params.stability_override_lr_min);
    }
    
    // Load scratch blocks configuration
    if (auto it = trainConfig.find("scratch_blocks"); it != trainConfig.end()) {
        const auto& scratch = *it;
        if (scratch.is_boolean()) {
            params.scratch_blocks_enabled = scratch.get<bool>();
        } else if (scratch.is_object()) {
            params.scratch_blocks_enabled = scratch.value("enabled", params.scratch_blocks_enabled);
            // scratch_max_tokens_per_block: derived as max_seq_len (see deriveComputedHyperparameters)
            params.scratch_num_blocks = scratch.value("num_blocks", params.scratch_num_blocks);
            params.scratch_write_combined = scratch.value("use_write_combined", params.scratch_write_combined);
        }
    }
    
    // Load scratch_block_reasoning configuration (model fields, live on architecture)
    if (auto it = trainConfig.find("scratch_block_reasoning"); it != trainConfig.end() && it->is_object()) {
        const auto& sbr = *it;
        params.architecture.use_scratch_block = sbr.value("enabled", params.architecture.use_scratch_block);
        params.architecture.scratch_block_atom_embedding_dim = sbr.value("atom_embedding_dim", params.architecture.scratch_block_atom_embedding_dim);
        params.architecture.scratch_block_max_atoms = sbr.value("max_atoms", params.architecture.scratch_block_max_atoms);
        params.architecture.scratch_block_atom_scale = sbr.value("atom_scale", params.architecture.scratch_block_atom_scale);
    }

    if (auto it = trainConfig.find("execution_block"); it != trainConfig.end() && it->is_object()) {
        const auto& eb = *it;
        assignTrainingField(params.architecture.execution_block_enabled, eb, "enabled");
        assignTrainingField(params.architecture.scratch_block_execution_first_type_only, eb, "execution_first_type_only");
        assignTrainingField(params.architecture.execution_block_layer, eb, "layer");
        assignTrainingField(params.architecture.execution_block_num_ops, eb, "num_ops");
        assignTrainingField(params.architecture.execution_block_num_slots, eb, "num_slots");
        assignTrainingField(params.architecture.execution_block_num_steps, eb, "num_steps");
        // execution_block_d_key: derived as d_model / num_heads (see deriveComputedHyperparameters)
        assignTrainingField(params.architecture.execution_block_d_type, eb, "d_type");
        // execution_block_cross_attn_head_dim: derived as d_model / num_heads (see deriveComputedHyperparameters)
        assignTrainingField(params.architecture.execution_block_cross_attn_topk, eb, "cross_attn_topk");
        assignTrainingField(params.architecture.execution_block_usage_decay, eb, "usage_decay");
        assignTrainingField(params.architecture.execution_block_diversity_kappa, eb, "diversity_kappa");
        assignTrainingField(params.architecture.execution_block_temp_start, eb, "temp_start");
        assignTrainingField(params.architecture.execution_block_temp_end, eb, "temp_end");
        assignTrainingField(params.architecture.execution_block_temp_schedule, eb, "temp_schedule");
        assignTrainingField(params.architecture.execution_block_entropy_weight, eb, "entropy_weight");
        assignTrainingField(params.architecture.step_x_multiplier, eb, "step_x_multiplier");
        assignTrainingField(params.architecture.step_y_multiplier, eb, "step_y_multiplier");
        assignTrainingField(params.architecture.step_y_overrides_x, eb, "step_y_overrides_x");
        assignTrainingField(params.architecture.entropy_aux_weight, eb, "entropy_aux_weight");
        assignTrainingField(params.architecture.value_match_epsilon, eb, "value_match_epsilon");
        assignTrainingField(params.architecture.final_slot_consistency_weight, eb, "final_slot_consistency_weight");
        assignTrainingField(params.architecture.execution_block_transition_hard_threshold, eb, "transition_hard_threshold");
        // execution_block_gate_warmup_steps: derived as warmup_steps (see deriveComputedHyperparameters)
        assignTrainingField(params.architecture.execution_block_causal_w1_transition, eb, "causal_w1_transition");
        assignTrainingField(params.architecture.div_invalid_penalty_weight, eb, "div_invalid_penalty_weight");
        assignTrainingField(params.architecture.div_magnitude_penalty_weight, eb, "div_magnitude_penalty_weight");
        assignTrainingField(params.architecture.arg_reinforce_weight, eb, "arg_reinforce_weight");
        assignTrainingField(params.architecture.arg_reinforce_baseline_decay, eb, "arg_reinforce_baseline_decay");
        assignTrainingField(params.architecture.structured_ce_enabled, eb, "structured_ce_enabled");
        assignTrainingField(params.architecture.structured_ce_weight, eb, "structured_ce_weight");

        // Decode-time slot selector (nested under execution_block)
        if (auto sit = eb.find("selector"); sit != eb.end() && sit->is_object()) {
            const auto& sel = *sit;
            assignTrainingField(params.architecture.selector_enabled, sel, "enabled");
            assignTrainingField(params.architecture.selector_d_selector, sel, "d_selector");
            assignTrainingField(params.architecture.selector_selection_margin, sel, "selection_margin");
            assignTrainingField(params.architecture.selector_supervision_weight, sel, "supervision_weight");
        }
    }
    
    // Load CUDA execution mode configuration
    if (auto it = trainConfig.find("cuda_execution"); it != trainConfig.end() && it->is_object()) {
        const auto& cuda_exec = *it;
        params.single_stream_mode = cuda_exec.value("single_stream_mode", params.single_stream_mode);
        params.disable_async_frees = cuda_exec.value("disable_async_frees", params.disable_async_frees);
        params.synchronize_after_kernels = cuda_exec.value("synchronize_after_kernels", params.synchronize_after_kernels);
    }
    
    // Load multi_token_prediction (MTP) configuration
    if (auto it = trainConfig.find("multi_token_prediction"); it != trainConfig.end() && it->is_object()) {
        const auto& mtp = *it;
        params.architecture.mtp_enabled = mtp.value("enabled", params.architecture.mtp_enabled);
        params.architecture.mtp_k = mtp.value("k", params.architecture.mtp_k);
        params.architecture.mtp_alpha = mtp.value("alpha", params.architecture.mtp_alpha);
        // mtp_alpha_warmup_steps: derived as warmup_steps (see deriveComputedHyperparameters)
        params.mtp_log_ratio_monitor = mtp.value("log_ratio_monitor", params.mtp_log_ratio_monitor);
    }

    // Load prediction comparison configuration
    if (auto it = trainConfig.find("prediction_comparison"); it != trainConfig.end() && it->is_object()) {
        const auto& pred_cmp = *it;
        params.prediction_comparison_enabled = pred_cmp.value("enabled", params.prediction_comparison_enabled);
        params.prediction_comparison_interval = pred_cmp.value("interval", params.prediction_comparison_interval);
        params.prediction_comparison_top_k = pred_cmp.value("top_k", params.prediction_comparison_top_k);
        params.prediction_comparison_max_positions = pred_cmp.value("max_positions", params.prediction_comparison_max_positions);
        params.prediction_comparison_log_path = pred_cmp.value("log_path", params.prediction_comparison_log_path);
    }

    // Logit update trace configuration
    if (auto it = trainConfig.find("logit_update_trace"); it != trainConfig.end()) {
        const auto& trace = *it;
        if (trace.is_boolean()) {
            params.logit_update_trace_enabled = trace.get<bool>();
        } else if (trace.is_object()) {
            params.logit_update_trace_enabled = trace.value("enabled", params.logit_update_trace_enabled);
            params.logit_update_trace_interval = trace.value("interval", params.logit_update_trace_interval);
        }
    }
    
    // Load attention diagnostics configuration
    // Use this to diagnose training plateau (saturated attention, gradient collapse)
    if (auto it = trainConfig.find("attention_diagnostics"); it != trainConfig.end() && it->is_object()) {
        const auto& attn_diag = *it;
        params.attention_diag_enabled = attn_diag.value("enabled", params.attention_diag_enabled);
        params.attention_diag_layer = attn_diag.value("layer", params.attention_diag_layer);
        params.attention_diag_head = attn_diag.value("head", params.attention_diag_head);
    }
}

// Compute formula-derived hyperparameters from base fields.
// Called AFTER applyTrainingConfigObject() so base fields are populated from JSON.
// These ALWAYS overwrite — they are mathematical derivations, not defaults.
// Rule 20: throws if any base field needed for derivation is invalid.
inline void deriveComputedHyperparameters(TrainingHyperparameters& params, const nlohmann::json& trainConfig) {
    // ── Sequence length derivations (max_seq_len lives on architecture, Phase 3b) ──
    if (params.architecture.max_seq_len <= 0)
        throw std::runtime_error("deriveComputedHyperparameters: architecture.max_seq_len must be > 0, got " + std::to_string(params.architecture.max_seq_len));
    params.min_seq_valid_tokens = params.architecture.max_seq_len / 4;
    params.architecture.min_seq_len_for_flash = params.architecture.max_seq_len / 4;
    params.scratch_max_tokens_per_block = static_cast<size_t>(params.architecture.max_seq_len);

    // ── LR floor ──
    if (params.cosine_decay_enabled) {
        if (params.learning_rate <= 0.0f)
            throw std::runtime_error("deriveComputedHyperparameters: learning_rate must be > 0 when cosine_decay is enabled, got " + std::to_string(params.learning_rate));
        params.cosine_decay_min_lr = params.learning_rate * 0.1f;
    }

    // ── Head dimension propagation (model fields live on architecture) ──
    if (trainConfig.contains("d_model") && trainConfig.contains("num_heads")) {
        int d_model = trainConfig["d_model"].get<int>();
        int num_heads = trainConfig["num_heads"].get<int>();
        if (d_model > 0 && num_heads > 0) {
            int head_dim = d_model / num_heads;
            params.architecture.execution_block_d_key = head_dim;
            params.architecture.execution_block_cross_attn_head_dim = head_dim;
            params.architecture.scratch_block_atom_embedding_dim = d_model / 8;
        }
    }

    // ── Warmup fraction validation (warmup_steps derived in Phase2 from warmup_fraction * total_steps) ──
    if (params.warmup_fraction <= 0.0f || params.warmup_fraction >= 1.0f)
        throw std::runtime_error("deriveComputedHyperparameters: warmup_fraction must be in (0, 1), got " + std::to_string(params.warmup_fraction));
    // warmup_steps, mtp_alpha_warmup_steps, telemetry_warmup_steps,
    // execution_block_gate_warmup_steps are all
    // derived in Phase2 via deriveWarmupSteps() once estimated_total_steps is known.

    // ── Stability overrides derived from base values ──
    params.stability_override_batch_size = params.batch_size * 2 / 3;
    if (params.stability_override_batch_size < 1) params.stability_override_batch_size = 1;
    params.stability_override_max_seq_len = params.architecture.max_seq_len;
    params.stability_override_clip_per_token = 0.02f;
    params.stability_override_lr_min = params.learning_rate * 0.83f;
}

/// Derive warmup_steps and dependent fields once estimated_total_steps is known (Phase2).
/// Must be called after deriveComputedHyperparameters() and before the training loop.
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

inline bool populateTrainingHyperparametersFromConfig(const nlohmann::json& config, TrainingHyperparameters& params) {
    if (config.contains("training") && config["training"].contains("config")) {
        const auto& training = config["training"];
        // Parse training-level selectors (sibling to "config")
        assignTrainingField(params.current_model_training, training, "current_model_training");
        assignTrainingField(params.current_curriculum, training, "current_curriculum");

        const auto& trainConfig = training["config"];
        // Phase 1: Set sensible defaults for all non-base params
        setDefaultHyperparameters(params);
        // Phase 2: Validate base+enable fields exist
        validateTrainingConfigJson(trainConfig);
        // Phase 3: Parse JSON (overwrites defaults only for keys present)
        applyTrainingConfigObject(trainConfig, params);
        // Phase 4: Compute formula-derived values (always overwrites)
        deriveComputedHyperparameters(params, trainConfig);
        return true;
    }
    return false;
}

inline bool populateDataCollectionConfigFromConfig(const nlohmann::json& config, DataCollectionConfig& dc_config) {
    if (!config.contains("data_collection")) {
        return false;
    }

    const auto& dc = config["data_collection"];
    if (!dc.is_object()) {
        return false;
    }

    assignTrainingField(dc_config.clear_merged_cache_on_merge, dc, "clear_merged_cache_on_merge");
    assignTrainingField(dc_config.max_new_entries_per_run, dc, "max_new_entries_per_run");

    return true;
}

// Populate SubprocessConfig from ai_config.json. Reads two semantically-related
// flags from different top-level sections so the coordinator (training/Subprocess/)
// has a single, validated, type-checked view. Throws std::runtime_error on a
// type mismatch (Rule 20: fail loud — wrong type is NEVER silently coerced).
// Missing fields default to false. Returns true if either field was found.
inline bool populateSubprocessConfigFromConfig(const nlohmann::json& config, SubprocessConfig& sc) {
    bool found_any = false;

    // subprocess.tokenizer.only_mode
    if (config.contains("subprocess")) {
        const auto& subp = config["subprocess"];
        if (!subp.is_object()) {
            throw std::runtime_error(
                "ai_config.json: 'subprocess' must be an object");
        }
        if (subp.contains("tokenizer")) {
            const auto& tok = subp["tokenizer"];
            if (!tok.is_object()) {
                throw std::runtime_error(
                    "ai_config.json: 'subprocess.tokenizer' must be an object");
            }
            if (tok.contains("only_mode")) {
                if (!tok["only_mode"].is_boolean()) {
                    throw std::runtime_error(
                        "ai_config.json: 'subprocess.tokenizer.only_mode' must be a boolean");
                }
                sc.tokenizer_only_mode = tok["only_mode"].get<bool>();
                found_any = true;
            }
        }
    }

    // training.config.force_rebuild_vocab
    if (config.contains("training") && config["training"].is_object()) {
        const auto& training = config["training"];
        if (training.contains("config") && training["config"].is_object()) {
            const auto& tcfg = training["config"];
            if (tcfg.contains("force_rebuild_vocab")) {
                if (!tcfg["force_rebuild_vocab"].is_boolean()) {
                    throw std::runtime_error(
                        "ai_config.json: 'training.config.force_rebuild_vocab' must be a boolean");
                }
                sc.tokenizer_force_rebuild_vocab = tcfg["force_rebuild_vocab"].get<bool>();
                found_any = true;
            }
        }
    }

    return found_any;
}

inline bool populateTokenizerConfigFromConfig(const nlohmann::json& config, TokenizerConfig& tokenizer_config, TrainingHyperparameters& hyperparameters) {
    if (!config.contains("tokenizer")) {
        return false;
    }

    const auto& tok = config["tokenizer"];
    if (!tok.is_object()) {
        return false;
    }

    assignTrainingField(tokenizer_config.vocab_size, tok, "vocab_size");
    assignTrainingField(tokenizer_config.max_vocab_size, tok, "max_vocab_size");
    assignTrainingField(tokenizer_config.max_length, tok, "max_length");
    assignTrainingField(tokenizer_config.min_subword_freq, tok, "min_subword_freq");
    assignTrainingField(tokenizer_config.prune_during_mining, tok, "prune_during_mining");
    assignTrainingField(tokenizer_config.enable_parallel_subword_mining, tok, "enable_parallel_subword_mining");
    assignTrainingField(tokenizer_config.subword_mining_workers, tok, "subword_mining_workers");
    assignTrainingField(tokenizer_config.subword_mining_max_bytes, tok, "subword_mining_max_bytes");
    assignTrainingField(tokenizer_config.model_type, tok, "model_type");
    assignTrainingField(tokenizer_config.add_bos, tok, "add_bos");
    assignTrainingField(tokenizer_config.add_eos, tok, "add_eos");
    assignTrainingField(tokenizer_config.unk_token, tok, "unk_token");
    assignTrainingField(tokenizer_config.pad_token, tok, "pad_token");
    assignTrainingField(tokenizer_config.bos_token, tok, "bos_token");
    assignTrainingField(tokenizer_config.eos_token, tok, "eos_token");
    assignTrainingField(tokenizer_config.enable_nfkc_normalization, tok, "enable_nfkc_normalization");
    assignTrainingField(tokenizer_config.enable_lowercasing, tok, "enable_lowercasing");
    assignTrainingField(tokenizer_config.enable_parallel_tokenization, tok, "enable_parallel_tokenization");
    assignTrainingField(tokenizer_config.parallel_threshold, tok, "parallel_threshold");
    assignTrainingField(tokenizer_config.enable_byte_fallback, tok, "enable_byte_fallback");
    assignTrainingField(tokenizer_config.expected_checksum, tok, "expected_checksum");
    assignTrainingField(tokenizer_config.save_text_vocab, tok, "save_text_vocab");
    assignTrainingField(tokenizer_config.vocab_score_multiplier, tok, "vocab_score_multiplier");

    // Backward compatibility with configs that only set max_vocab_size:
    // treat it as the target vocab size when vocab_size is omitted.
    if (!tok.contains("vocab_size") &&
        tok.contains("max_vocab_size") &&
        tokenizer_config.max_vocab_size > 0) {
        tokenizer_config.vocab_size = tokenizer_config.max_vocab_size;
    }

    // Scratch block reasoning configuration - load into hyperparameters (single source of truth)
    if (tok.contains("scratch_block_reasoning") && tok["scratch_block_reasoning"].is_object()) {
        const auto& sbr = tok["scratch_block_reasoning"];
        assignTrainingField(hyperparameters.tokenizer_enable_scratch_block_reasoning, sbr, "enabled");
        assignTrainingField(hyperparameters.tokenizer_detect_numbers, sbr, "detect_numbers");
    }

    if (tok.contains("special_tokens") && tok["special_tokens"].is_array()) {
        tokenizer_config.special_tokens.clear();
        for (const auto& token : tok["special_tokens"]) {
            if (token.is_string()) {
                tokenizer_config.special_tokens.push_back(token.get<std::string>());
            }
        }
    }

    return true;
}

} // namespace detail

/// Derive warmup_steps and dependent fields once estimated_total_steps is known (Phase2).
/// Must be called after populateTrainingHyperparametersFromConfig() and before the training loop.
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
    try {
        auto resolved_path = detail::resolveAiConfigPath(configPath);
        if (!resolved_path) {
            std::cerr << "[Config] ERROR: ai_config.json not found at: " << configPath << std::endl;
            std::cerr << "[Config]        Also searched current directory and parent directories" << std::endl;
            return std::nullopt;
        }

        std::ifstream configFile(*resolved_path);
        if (!configFile.is_open()) {
            std::cerr << "[Config] ERROR: Could not open ai_config.json at: " << *resolved_path << std::endl;
            return std::nullopt;
        }

        nlohmann::json config;
        configFile >> config;

        AiConfigSnapshot snapshot;
        snapshot.config_path = *resolved_path;
        snapshot.document = std::move(config);
        snapshot.has_grim_paths = detail::populateGrimTextPathsFromConfig(snapshot.document, snapshot.grim_paths);
        snapshot.has_training = detail::populateTrainingHyperparametersFromConfig(snapshot.document, snapshot.hyperparameters);
        snapshot.has_tokenizer = detail::populateTokenizerConfigFromConfig(snapshot.document, snapshot.tokenizer_config, snapshot.hyperparameters);
        snapshot.has_data_collection = detail::populateDataCollectionConfigFromConfig(snapshot.document, snapshot.data_collection_config);
        snapshot.has_subprocess = detail::populateSubprocessConfigFromConfig(snapshot.document, snapshot.subprocess_config);
        return snapshot;
    } catch (const std::exception& e) {
        std::cerr << "[Config] ERROR: Exception loading ai_config.json: " << e.what() << std::endl;
        return std::nullopt;
    }
}

inline bool loadGrimTextPaths(GrimTextPaths& paths, const std::string& configPath = "ai_config.json") {
    auto snapshot = loadAiConfigSnapshot(configPath);
    if (!snapshot) {
        return false;
    }

    if (!snapshot->has_grim_paths) {
        return false;
    }

    paths = snapshot->grim_paths;

    // Avoid spamming logs when polled frequently (e.g., from UI timers)
    static std::string lastPrintedSignature;
    const std::string signature =
        snapshot->config_path.string() + "|" +
        paths.vocab + "|" + paths.model + "|" + paths.training_data + "|" +
        paths.checkpoints + "|" + paths.collected + "|" + paths.directory_collection + "|" + paths.verified + "|" +
        paths.logs + "|" + paths.training_status + "|" + paths.merge_checkpoints_exe +
        "|" + paths.collector_log + "|" + paths.source_config +
        "|" + paths.model_store;

    if (signature != lastPrintedSignature) {
        std::cout << "[Config] Loading GRIM-text paths from: " << snapshot->config_path << std::endl;
        paths.printPaths();
        lastPrintedSignature = signature;
    }

    return paths.isValid();
}

inline bool loadTrainingHyperparameters(TrainingHyperparameters& params, const std::string& configPath = "ai_config.json") {
    auto snapshot = loadAiConfigSnapshot(configPath);
    if (!snapshot) {
        return false;
    }

    if (!snapshot->has_training) {
        return false;
    }

    params = snapshot->hyperparameters;
    return true;
}

inline bool loadTokenizerConfig(TokenizerConfig& config, const std::string& configPath = "ai_config.json") {
    auto snapshot = loadAiConfigSnapshot(configPath);
    if (!snapshot) {
        return false;
    }

    if (!snapshot->has_tokenizer) {
        return false;
    }

    config = snapshot->tokenizer_config;
    std::cout << "[Config] Loaded tokenizer config from: " << snapshot->config_path << std::endl;
    std::cout << "  vocab_size: " << config.vocab_size << std::endl;
    std::cout << "  max_vocab_size: " << config.max_vocab_size << std::endl;
    std::cout << "  max_length: " << config.max_length << std::endl;
    std::cout << "  enable_parallel_subword_mining: "
              << (config.enable_parallel_subword_mining ? "true" : "false") << std::endl;
    std::cout << "  subword_mining_workers: " << config.subword_mining_workers << std::endl;
    std::cout << "  subword_mining_max_bytes: " << config.subword_mining_max_bytes << std::endl;
    std::cout << "  model_type: " << config.model_type << std::endl;
    std::cout << "  add_bos: " << (config.add_bos ? "true" : "false") << std::endl;
    std::cout << "  add_eos: " << (config.add_eos ? "true" : "false") << std::endl;
    std::cout << "  enable_lowercasing: " << (config.enable_lowercasing ? "true" : "false") << std::endl;
    std::cout << "  enable_nfkc_normalization: " << (config.enable_nfkc_normalization ? "true" : "false") << std::endl;
    std::cout << "  enable_parallel_tokenization: " << (config.enable_parallel_tokenization ? "true" : "false") << std::endl;
    std::cout << "  parallel_threshold: " << config.parallel_threshold << std::endl;
    std::cout << "  special_tokens: [";
    for (size_t i = 0; i < config.special_tokens.size(); ++i) {
        std::cout << "\"" << config.special_tokens[i] << "\"";
        if (i + 1 < config.special_tokens.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    return true;
}

inline bool loadDataCollectionConfig(DataCollectionConfig& config, const std::string& configPath = "ai_config.json") {
    auto snapshot = loadAiConfigSnapshot(configPath);
    if (!snapshot) {
        return false;
    }

    if (!snapshot->has_data_collection) {
        return false;
    }

    config = snapshot->data_collection_config;
    return true;
}

/**
 * @brief Load the SubprocessConfig view (subprocess.tokenizer.* and
 *        training.config.force_rebuild_vocab) from ai_config.json.
 *
 * Returns true on success, false if the config file is missing or unreadable.
 * Throws std::runtime_error (via the populator) on a present-but-malformed
 * field — the coordinator MUST fail loud rather than spawn a subprocess with
 * silently-wrong flags.
 *
 * Missing fields default to false (the field defaults on SubprocessConfig).
 */
inline bool loadSubprocessConfig(SubprocessConfig& config, const std::string& configPath = "ai_config.json") {
    auto snapshot = loadAiConfigSnapshot(configPath);
    if (!snapshot) {
        return false;
    }
    config = snapshot->subprocess_config;
    return true;
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
    // First, try to load from ai_config.json
    GrimTextPaths paths;
    if (loadGrimTextPaths(paths) && !paths.training_status.empty()) {
        return paths.training_status;
    }
    
    // Fallback: Status file lives in the training directory
    // This ensures train_gpu.exe and training_control_server.exe can both find it
    std::filesystem::path grimRoot = std::filesystem::current_path();
    
    // Walk up to find GRIM root (has both 'control' and 'resources' directories)
    for (int i = 0; i < 5 && grimRoot.has_parent_path(); ++i) {
        if (std::filesystem::exists(grimRoot / "control") && 
            std::filesystem::exists(grimRoot / "resources")) {
            break;
        }
        grimRoot = grimRoot.parent_path();
    }
    
    std::filesystem::path statusPath = grimRoot / "resources" / "models" / "GRIM-text" / "training" / "training_status.fb";
    return statusPath.string();
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
    // First, try to load from ai_config.json
    GrimTextPaths paths;
    if (loadGrimTextPaths(paths) && !paths.checkpoints.empty()) {
        return paths.checkpoints;
    }
    
    // Fallback: Checkpoint directory
    std::filesystem::path grimRoot = std::filesystem::current_path();
    
    // Walk up to find GRIM root (has both 'control' and 'resources' directories)
    for (int i = 0; i < 5 && grimRoot.has_parent_path(); ++i) {
        if (std::filesystem::exists(grimRoot / "control") && 
            std::filesystem::exists(grimRoot / "resources")) {
            break;
        }
        grimRoot = grimRoot.parent_path();
    }
    
    std::filesystem::path checkpointPath = grimRoot / "resources" / "models" / "GRIM-text" / "checkpoints";
    return checkpointPath.string();
}

/**
 * Get the collector log file path from ai_config.json.
 * Falls back to automatic discovery if not in config.
 */
inline std::string getCollectorLogPath() {
    // First, try to load from ai_config.json
    GrimTextPaths paths;
    if (loadGrimTextPaths(paths) && !paths.collector_log.empty()) {
        return paths.collector_log;
    }
    
    // Fallback: Collector log path
    std::filesystem::path grimRoot = std::filesystem::current_path();
    
    // Walk up to find GRIM root (has both 'control' and 'resources' directories)
    for (int i = 0; i < 5 && grimRoot.has_parent_path(); ++i) {
        if (std::filesystem::exists(grimRoot / "control") && 
            std::filesystem::exists(grimRoot / "resources")) {
            break;
        }
        grimRoot = grimRoot.parent_path();
    }
    
    std::filesystem::path logPath = grimRoot / "resources" / "models" / "GRIM-text" / "training" / "logs" / "collector.log";
    return logPath.string();
}

/**
 * Get the collected data directory from ai_config.json.
 * Falls back to automatic discovery if not in config.
 */
inline std::string getCollectedDir() {
    // First, try to load from ai_config.json
    GrimTextPaths paths;
    if (loadGrimTextPaths(paths) && !paths.collected.empty()) {
        return paths.collected;
    }
    
    // Fallback: Collected data directory
    std::filesystem::path grimRoot = std::filesystem::current_path();
    
    // Walk up to find GRIM root (has both 'control' and 'resources' directories)
    for (int i = 0; i < 5 && grimRoot.has_parent_path(); ++i) {
        if (std::filesystem::exists(grimRoot / "control") && 
            std::filesystem::exists(grimRoot / "resources")) {
            break;
        }
        grimRoot = grimRoot.parent_path();
    }
    
    std::filesystem::path collectedPath = grimRoot / "resources" / "models" / "GRIM-text" / "data" / "collected";
    return collectedPath.string();
}

/**
 * Get the verified data directory from ai_config.json.
 * Falls back to automatic discovery if not in config.
 */
inline std::string getVerifiedDir() {
    // First, try to load from ai_config.json
    GrimTextPaths paths;
    if (loadGrimTextPaths(paths) && !paths.verified.empty()) {
        return paths.verified;
    }
    
    // Fallback: Verified data directory
    std::filesystem::path grimRoot = std::filesystem::current_path();
    
    // Walk up to find GRIM root (has both 'control' and 'resources' directories)
    for (int i = 0; i < 5 && grimRoot.has_parent_path(); ++i) {
        if (std::filesystem::exists(grimRoot / "control") && 
            std::filesystem::exists(grimRoot / "resources")) {
            break;
        }
        grimRoot = grimRoot.parent_path();
    }
    
    std::filesystem::path verifiedPath = grimRoot / "resources" / "models" / "GRIM-text" / "data" / "verified";
    return verifiedPath.string();
}

/**
 * @brief Get actual vocab size from vocab.bin file
 * 
 * Loads the tokenizer and returns its actual vocabulary size.
 * This is the authoritative source for vocab size, not ai_config.json.
 * 
 * @param vocabPath Path to vocab.bin file (optional, auto-detects if not provided)
 * @return Vocab size from tokenizer, or 0 on error
 */
inline uint32_t getActualVocabSize(const std::string& vocabPath = "") {
    try {
        std::string path = vocabPath;
        
        // Auto-detect vocab path if not provided
        if (path.empty()) {
            GrimTextPaths paths;
            if (loadGrimTextPaths(paths) && !paths.vocab.empty()) {
                path = paths.vocab;
            } else {
                // Fallback
                std::filesystem::path grimRoot = std::filesystem::current_path();
                for (int i = 0; i < 5 && grimRoot.has_parent_path(); ++i) {
                    if (std::filesystem::exists(grimRoot / "control") && 
                        std::filesystem::exists(grimRoot / "resources")) {
                        break;
                    }
                    grimRoot = grimRoot.parent_path();
                }
                path = (grimRoot / "resources" / "models" / "GRIM-text" / "training" / "data" / "vocab.bin").string();
            }
        }
        
        // Load tokenizer to get actual vocab size
        // Note: This requires linking against the tokenizer library
        // For a header-only solution, we could parse the binary format directly
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            std::cerr << "[Config] Failed to open vocab file: " << path << std::endl;
            return 0;
        }
        
        // Read GMKT header format (version 2)
        char magic[4];
        file.read(magic, 4);
        if (magic[0] != 'K' || magic[1] != 'T' || magic[2] != 'M' || magic[3] != 'G') {
            std::cerr << "[Config] Invalid vocab file magic" << std::endl;
            return 0;
        }
        
        uint16_t version;
        file.read(reinterpret_cast<char*>(&version), 2);
        
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
        
        return vocab_size;
        
    } catch (const std::exception& e) {
        std::cerr << "[Config] Error reading vocab size: " << e.what() << std::endl;
        return 0;
    }
}

} // namespace Config
} // namespace GRIM
