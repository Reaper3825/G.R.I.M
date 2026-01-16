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
// training.config            → TrainingHyperparameters
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
    std::string verified;
    std::string logs;
    std::string training_status;
    std::string merge_checkpoints_exe;
    std::string collector_log;
    std::string source_config;  // Path to source_data.json for data collection
    
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
        std::cout << "  verified: " << (verified.empty() ? "(not set)" : verified) << std::endl;
        std::cout << "  logs: " << (logs.empty() ? "(not set)" : logs) << std::endl;
        std::cout << "  training_status: " << (training_status.empty() ? "(not set)" : training_status) << std::endl;
        std::cout << "  merge_checkpoints_exe: " << (merge_checkpoints_exe.empty() ? "(not set)" : merge_checkpoints_exe) << std::endl;
        std::cout << "  collector_log: " << (collector_log.empty() ? "(not set)" : collector_log) << std::endl;
        std::cout << "  source_config: " << (source_config.empty() ? "(not set)" : source_config) << std::endl;
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
    } layers;
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
    
    // Log Recorder configuration
    LogRecorderConfig log_recorder;

    // Core training parameters - NO DEFAULTS
    int epochs;
    int64_t seed;
    int batch_size;
    int gradient_accumulation_steps;
    bool single_batch_overfit_enabled;
    int single_batch_overfit_max_steps;
    std::string batch_strategy;
    float learning_rate;
    float weight_decay;
    float grad_clip_norm;
    bool per_token_grad_scale;
    int warmup_steps;
    int max_seq_len;
    int min_seq_valid_tokens;  // Minimum valid tokens required (after masking first/last positions)
    int log_interval;
    int atom_stats_interval;
    int atom_stats_max_seqs;
    int validation_interval;
    int checkpoint_interval;
    bool use_gpu;
    bool use_flash_attention;
    
    // Dynamic LR - NO DEFAULTS
    bool dynamic_lr_enabled;
    bool dynamic_lr_autogenerate;
    float dynamic_lr_min;
    float dynamic_lr_max;
    float dynamic_lr_increase_factor;
    float dynamic_lr_decrease_factor;
    float dynamic_lr_upper_grad_norm;
    float dynamic_lr_lower_grad_norm;
    float dynamic_lr_max_loss_jump;
    float dynamic_lr_smoothing;
    int dynamic_lr_cooldown_steps;
    int dynamic_lr_warmup_steps;
    float dynamic_lr_max_step_up_ratio;
    float dynamic_lr_max_step_down_ratio;
    bool dynamic_lr_auto_band;
    float dynamic_lr_band_sigma;
    float dynamic_lr_band_floor;
    float dynamic_lr_band_ceiling;
    int dynamic_lr_band_min_samples;
    float dynamic_lr_band_min_span;
    bool dynamic_lr_adaptive_smoothing;
    float dynamic_lr_smoothing_min;
    float dynamic_lr_smoothing_max;
    float dynamic_lr_variance_reference;
    bool dynamic_lr_adaptive_cooldown;
    int dynamic_lr_cooldown_min;
    int dynamic_lr_cooldown_max;
    bool dynamic_lr_adaptive_loss;
    float dynamic_lr_loss_sigma;
    int dynamic_lr_loss_min_samples;
    float dynamic_lr_loss_floor;
    bool dynamic_lr_guard_logging;
    int dynamic_lr_guard_floor_steps;
    float dynamic_lr_guard_grad_multiplier;
    int dynamic_lr_guard_loss_patience;
    float dynamic_lr_guard_loss_multiplier;
    int dynamic_lr_baseline_capture_steps;
    float dynamic_lr_baseline_drift;
    int dynamic_lr_momentum_interval;
    float dynamic_lr_momentum_gain;
    float dynamic_lr_momentum_decay;
    int dynamic_lr_safety_interval;
    float dynamic_lr_safety_gain;
    float dynamic_lr_safety_scale;
    
    // Soft restart - NO DEFAULTS
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
    
    // Cache limits - NO DEFAULTS
    int cache_max_batch;
    int cache_max_seq_len;
    
    // Micro validation - NO DEFAULTS
    bool micro_validation_enabled;
    int micro_validation_interval;
    int micro_validation_batch_limit;
    int micro_validation_min_step;
    bool micro_validation_prefer_short;
    
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
    bool loss_numeric_head_enabled;
    float loss_numeric_head_weight;
    float loss_numeric_head_huber_delta;
    bool loss_numeric_head_log_scale;
    
    // Issue #44 FIX: Entropy regularization to prevent mode collapse
    // reg = λ * Σ_v p_v² (penalizes logit concentration)
    bool loss_entropy_reg_enabled;
    float loss_entropy_reg_lambda;
    
    // LM Head centering (Issue #37 / #40) - NO DEFAULTS
    // When enabled, centers hidden states and recenters gradients.
    // Set to false for standard PyTorch-style implementation.
    bool lm_head_centering_enabled;
    bool lm_head_center_hidden_states;
    bool lm_head_recenter_gradients;
    
    // Stability overrides - NO DEFAULTS
    bool stability_overrides_enabled;
    int stability_override_batch_size;
    int stability_override_max_seq_len;
    int stability_override_max_tokens_per_batch;
    float stability_override_clip_abs;
    float stability_override_clip_norm;
    float stability_override_lr_min;
    
    // Scratch blocks - NO DEFAULTS
    bool scratch_blocks_enabled;
    size_t scratch_max_tokens_per_block;
    size_t scratch_num_blocks;
    bool scratch_write_combined;
    
    // ScratchBlock reasoning - NO DEFAULTS
    bool scratch_block_reasoning_enabled;
    int scratch_block_reasoning_atom_embedding_dim;
    int scratch_block_reasoning_max_atoms;
    float scratch_block_reasoning_atom_scale;
    
    // Activation quantization - NO DEFAULTS
    bool activation_quantization_enabled;
    bool activation_quantization_apply_to_embeddings;
    bool activation_quantization_apply_to_encoder_outputs;
    bool activation_quantization_apply_to_layer_caches;
    bool activation_quantization_apply_to_qkv_cache;
    bool activation_quantization_apply_to_logits;
    float activation_quantization_scale;
    float activation_quantization_clip_min;
    float activation_quantization_clip_max;
    int activation_quantization_zero_point;
    bool activation_quantization_symmetric;
    
    // CUDA execution mode - NO DEFAULTS
    bool single_stream_mode;
    bool disable_async_frees;
    bool synchronize_after_kernels;
    
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
    std::string model_type = "unibytes";
    std::vector<std::string> special_tokens = {"<pad>", "<unk>", "<s>", "</s>", "<mask>"};
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
    bool enable_char_fallback = true;
    bool enable_byte_fallback = true;
    uint32_t expected_checksum = 0;
    bool save_text_vocab = true;  // Also save human-readable .txt alongside .bin
    
    // Scratch block reasoning configuration
    bool enable_scratch_block_reasoning = true;
    bool detect_numbers = true;
    bool detect_urls = true;
    bool detect_emails = true;
    bool detect_paths = true;
    bool detect_dates = true;
    bool detect_code_literals = true;
};

struct AiConfigSnapshot {
    std::filesystem::path config_path;
    nlohmann::json document;
    GrimTextPaths grim_paths;
    TrainingHyperparameters hyperparameters;
    TokenizerConfig tokenizer_config;
    DataCollectionConfig data_collection_config;
    bool has_grim_paths = false;
    bool has_training = false;
    bool has_tokenizer = false;
    bool has_data_collection = false;
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

    auto assignIfPresent = [&](const char* key, std::string& field) {
        if (grimTextPaths.contains(key) && grimTextPaths[key].is_string()) {
            field = grimTextPaths[key].get<std::string>();
        }
    };

    assignIfPresent("vocab", paths.vocab);
    assignIfPresent("model", paths.model);
    assignIfPresent("training_data", paths.training_data);
    assignIfPresent("checkpoints", paths.checkpoints);
    assignIfPresent("collected", paths.collected);
    assignIfPresent("verified", paths.verified);
    assignIfPresent("logs", paths.logs);
    assignIfPresent("training_status", paths.training_status);
    assignIfPresent("merge_checkpoints_exe", paths.merge_checkpoints_exe);
    assignIfPresent("collector_log", paths.collector_log);
    assignIfPresent("source_config", paths.source_config);

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
 * @brief Validate all required fields exist in training.config JSON
 * @param trainConfig The training.config JSON object
 * @throws std::runtime_error listing all missing required fields
 * 
 * Rule 20: NO SILENT DEFAULTS. EVERY field MUST be explicitly set in ai_config.json.
 * TrainingHyperparameters has NO default values - all must come from JSON.
 */
inline void validateTrainingConfigJson(const nlohmann::json& trainConfig) {
    // ALL fields required - NO DEFAULTS ANYWHERE
    static const std::vector<std::string> REQUIRED = {
        // Core training
        "epochs", "seed", "batch_size", "gradient_accumulation_steps",
        "batch_strategy", "learning_rate", "weight_decay",
        "per_token_grad_scale", "warmup_steps", "max_seq_len", "min_seq_valid_tokens", "log_interval",
        "atom_stats_interval", "atom_stats_max_seqs",
        "validation_interval", "checkpoint_interval", "use_gpu", "use_flash_attention",
        
        // Single batch overfit
        "single_batch.enabled", "single_batch.max_steps",
        
        // Dynamic LR
        "dynamic_lr.enabled", "dynamic_lr.auto_generate",
        "dynamic_lr.min", "dynamic_lr.max",
        "dynamic_lr.increase_factor", "dynamic_lr.decrease_factor",
        "dynamic_lr.upper_grad_norm", "dynamic_lr.lower_grad_norm",
        "dynamic_lr.max_loss_jump", "dynamic_lr.smoothing",
        "dynamic_lr.cooldown_steps", "dynamic_lr.warmup_steps",
        "dynamic_lr.max_step_up_ratio", "dynamic_lr.max_step_down_ratio",
        "dynamic_lr.auto_band", "dynamic_lr.band_sigma",
        "dynamic_lr.band_floor", "dynamic_lr.band_ceiling",
        "dynamic_lr.band_min_samples", "dynamic_lr.band_min_span",
        "dynamic_lr.adaptive_smoothing", "dynamic_lr.smoothing_min", "dynamic_lr.smoothing_max",
        "dynamic_lr.variance_reference", "dynamic_lr.adaptive_cooldown",
        "dynamic_lr.cooldown_min", "dynamic_lr.cooldown_max",
        "dynamic_lr.adaptive_loss", "dynamic_lr.loss_sigma",
        "dynamic_lr.loss_min_samples", "dynamic_lr.loss_floor",
        "dynamic_lr.guard_logging", "dynamic_lr.guard_floor_steps",
        "dynamic_lr.guard_grad_multiplier", "dynamic_lr.guard_loss_patience",
        "dynamic_lr.guard_loss_multiplier", "dynamic_lr.baseline_capture_steps",
        "dynamic_lr.baseline_drift", "dynamic_lr.momentum_interval",
        "dynamic_lr.momentum_gain", "dynamic_lr.momentum_decay",
        "dynamic_lr.safety_interval", "dynamic_lr.safety_gain", "dynamic_lr.safety_scale",
        
        // Soft restart
        "soft_restart.enabled", "soft_restart.loss_increase_threshold",
        "soft_restart.max_step_window", "soft_restart.cooldown_steps",
        
        // Auto stop
        "auto_stop.enabled", "auto_stop.plateau_patience", "auto_stop.plateau_min_delta",
        "auto_stop.high_loss_threshold", "auto_stop.high_loss_patience",
        
        // Cache limits
        "cache_limits.max_cached_batch", "cache_limits.max_cached_seq_len",
        
        // Micro validation
        "micro_validation.enabled", "micro_validation.interval",
        "micro_validation.batch_limit", "micro_validation.min_step", "micro_validation.prefer_short",
        
        // Guess aux
        "guess_aux.enabled", "guess_aux.lambda", "guess_aux.min_confidence",
        
        // Shuffle
        "shuffle.enabled", "shuffle.epochs",
        
        // Telemetry control
        "telemetry_control.enabled",
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
        "telemetry_control.baseline.warmup_steps",
        "telemetry_control.baseline.stabilization_steps",
        "telemetry_control.logging.verbose",
        "telemetry_control.logging.fail_loud_on_accumulation_bug",
        "telemetry_control.plateau_noise.enabled",
        "telemetry_control.plateau_noise.patience",
        "telemetry_control.plateau_noise.variance_threshold",
        "telemetry_control.plateau_noise.noise_std",
        "telemetry_control.plateau_noise.proportional",
        "telemetry_control.plateau_noise.cooldown",
        "telemetry_control.plateau_noise.max_per_epoch",
        
        // Loss
        "loss.label_smoothing.enabled", "loss.label_smoothing.epsilon",
        "loss.focal.enabled", "loss.focal.gamma", "loss.focal.alpha",
        "loss.preference.enabled", "loss.preference.beta",
        "loss.distillation.enabled", "loss.distillation.temperature", "loss.distillation.lambda",
        "loss.masking.enabled", "loss.masking.tag",
        "loss.numeric_head.enabled", "loss.numeric_head.weight",
        "loss.numeric_head.huber_delta", "loss.numeric_head.log_scale",
        
        // Stability overrides
        "stability_overrides_enabled",
        "stability_overrides.batch_size", "stability_overrides.max_seq_len",
        "stability_overrides.max_tokens_per_batch", "stability_overrides.clip_abs",
        "stability_overrides.clip_per_token", "stability_overrides.lr_min",
        
        // Scratch blocks
        "scratch_blocks.enabled", "scratch_blocks.max_tokens_per_block",
        "scratch_blocks.num_blocks", "scratch_blocks.use_write_combined",
        
        // Scratch block reasoning
        "scratch_block_reasoning.enabled", "scratch_block_reasoning.atom_embedding_dim",
        "scratch_block_reasoning.max_atoms", "scratch_block_reasoning.atom_scale",
        
        // Activation quantization
        "activation_quantization.enabled",
        "activation_quantization.apply_to_embeddings",
        "activation_quantization.apply_to_encoder_outputs",
        "activation_quantization.apply_to_layer_caches",
        "activation_quantization.apply_to_qkv_cache",
        "activation_quantization.apply_to_logits",
        "activation_quantization.scale",
        "activation_quantization.clip_min", "activation_quantization.clip_max",
        "activation_quantization.zero_point", "activation_quantization.symmetric",
        
        // CUDA execution
        "cuda_execution.single_stream_mode",
        "cuda_execution.disable_async_frees",
        "cuda_execution.synchronize_after_kernels",
        
        // Prediction comparison
        "prediction_comparison.enabled", "prediction_comparison.interval",
        "prediction_comparison.top_k", "prediction_comparison.max_positions",
        "prediction_comparison.log_path",
        
        // Logit update trace
        "logit_update_trace.enabled", "logit_update_trace.interval",
        
        // Attention diagnostics
        "attention_diagnostics.enabled", "attention_diagnostics.layer", "attention_diagnostics.head",
        
        // Log recorder
        "log_recorder.enabled", "log_recorder.default_level"
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
        oss << "\nRule 20: NO SILENT DEFAULTS. ALL training config fields MUST be explicitly set.\n";
        oss << "TrainingHyperparameters has NO default values - every field must come from JSON.";
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
    assignTrainingField(params.max_seq_len, trainConfig, "max_seq_len");
    assignTrainingField(params.min_seq_valid_tokens, trainConfig, "min_seq_valid_tokens");
    assignTrainingField(params.warmup_steps, trainConfig, "warmup_steps");
    assignTrainingField(params.log_interval, trainConfig, "log_interval");
    assignTrainingField(params.atom_stats_interval, trainConfig, "atom_stats_interval");
    assignTrainingField(params.atom_stats_max_seqs, trainConfig, "atom_stats_max_seqs");
    assignTrainingField(params.validation_interval, trainConfig, "validation_interval");
    assignTrainingField(params.checkpoint_interval, trainConfig, "checkpoint_interval");
    assignTrainingField(params.use_gpu, trainConfig, "use_gpu");
    assignTrainingField(params.use_flash_attention, trainConfig, "use_flash_attention");

    if (auto it = trainConfig.find("dynamic_lr"); it != trainConfig.end()) {
        const auto& dlr = *it;
        if (dlr.is_boolean()) {
            params.dynamic_lr_enabled = dlr.get<bool>();

        } else if (dlr.is_object()) {
            params.dynamic_lr_enabled = dlr.value("enabled", params.dynamic_lr_enabled);
            params.dynamic_lr_autogenerate = dlr.value("auto_generate", params.dynamic_lr_autogenerate);
            if (!params.dynamic_lr_autogenerate) {
                params.dynamic_lr_min = dlr.value("min", params.dynamic_lr_min);
                params.dynamic_lr_max = dlr.value("max", params.dynamic_lr_max);
                params.dynamic_lr_increase_factor = dlr.value("increase_factor", params.dynamic_lr_increase_factor);
                params.dynamic_lr_decrease_factor = dlr.value("decrease_factor", params.dynamic_lr_decrease_factor);
                params.dynamic_lr_upper_grad_norm = dlr.value("upper_grad_norm", params.dynamic_lr_upper_grad_norm);
                params.dynamic_lr_lower_grad_norm = dlr.value("lower_grad_norm", params.dynamic_lr_lower_grad_norm);
                params.dynamic_lr_max_loss_jump = dlr.value("max_loss_jump", params.dynamic_lr_max_loss_jump);
                params.dynamic_lr_smoothing = dlr.value("smoothing", params.dynamic_lr_smoothing);
                params.dynamic_lr_cooldown_steps = dlr.value("cooldown_steps", params.dynamic_lr_cooldown_steps);
                params.dynamic_lr_warmup_steps = dlr.value("warmup_steps", params.dynamic_lr_warmup_steps);
                params.dynamic_lr_max_step_up_ratio = dlr.value("max_step_up_ratio", params.dynamic_lr_max_step_up_ratio);
                params.dynamic_lr_max_step_down_ratio = dlr.value("max_step_down_ratio", params.dynamic_lr_max_step_down_ratio);
                params.dynamic_lr_auto_band = dlr.value("auto_band", params.dynamic_lr_auto_band);
                params.dynamic_lr_band_sigma = dlr.value("band_sigma", params.dynamic_lr_band_sigma);
                params.dynamic_lr_band_floor = dlr.value("band_floor", params.dynamic_lr_band_floor);
                params.dynamic_lr_band_ceiling = dlr.value("band_ceiling", params.dynamic_lr_band_ceiling);
                params.dynamic_lr_band_min_samples = dlr.value("band_min_samples", params.dynamic_lr_band_min_samples);
                params.dynamic_lr_band_min_span = dlr.value("band_min_span", params.dynamic_lr_band_min_span);
                params.dynamic_lr_adaptive_smoothing = dlr.value("adaptive_smoothing", params.dynamic_lr_adaptive_smoothing);
                params.dynamic_lr_smoothing_min = dlr.value("smoothing_min", params.dynamic_lr_smoothing_min);
                params.dynamic_lr_smoothing_max = dlr.value("smoothing_max", params.dynamic_lr_smoothing_max);
                params.dynamic_lr_variance_reference = dlr.value("variance_reference", params.dynamic_lr_variance_reference);
                params.dynamic_lr_adaptive_cooldown = dlr.value("adaptive_cooldown", params.dynamic_lr_adaptive_cooldown);
                params.dynamic_lr_cooldown_min = dlr.value("cooldown_min", params.dynamic_lr_cooldown_min);
                params.dynamic_lr_cooldown_max = dlr.value("cooldown_max", params.dynamic_lr_cooldown_max);
                params.dynamic_lr_adaptive_loss = dlr.value("adaptive_loss", params.dynamic_lr_adaptive_loss);
                params.dynamic_lr_loss_sigma = dlr.value("loss_sigma", params.dynamic_lr_loss_sigma);
                params.dynamic_lr_loss_min_samples = dlr.value("loss_min_samples", params.dynamic_lr_loss_min_samples);
                params.dynamic_lr_loss_floor = dlr.value("loss_floor", params.dynamic_lr_loss_floor);
                params.dynamic_lr_guard_logging = dlr.value("guard_logging", params.dynamic_lr_guard_logging);
                params.dynamic_lr_guard_floor_steps = dlr.value("guard_floor_steps", params.dynamic_lr_guard_floor_steps);
                params.dynamic_lr_guard_grad_multiplier = dlr.value("guard_grad_multiplier", params.dynamic_lr_guard_grad_multiplier);
                params.dynamic_lr_guard_loss_patience = dlr.value("guard_loss_patience", params.dynamic_lr_guard_loss_patience);
                params.dynamic_lr_guard_loss_multiplier = dlr.value("guard_loss_multiplier", params.dynamic_lr_guard_loss_multiplier);
                params.dynamic_lr_baseline_capture_steps = dlr.value("baseline_capture_steps", params.dynamic_lr_baseline_capture_steps);
                params.dynamic_lr_baseline_drift = dlr.value("baseline_drift", params.dynamic_lr_baseline_drift);
                params.dynamic_lr_momentum_interval = dlr.value("momentum_interval", params.dynamic_lr_momentum_interval);
                params.dynamic_lr_momentum_gain = dlr.value("momentum_gain", params.dynamic_lr_momentum_gain);
                params.dynamic_lr_momentum_decay = dlr.value("momentum_decay", params.dynamic_lr_momentum_decay);
                params.dynamic_lr_safety_interval = dlr.value("safety_interval", params.dynamic_lr_safety_interval);
                params.dynamic_lr_safety_gain = dlr.value("safety_gain", params.dynamic_lr_safety_gain);
                params.dynamic_lr_safety_scale = dlr.value("safety_scale", params.dynamic_lr_safety_scale);
            }
        }
    }

    if (auto it = trainConfig.find("cache_limits"); it != trainConfig.end() && it->is_object()) {
        params.cache_max_batch = it->value("max_cached_batch", params.cache_max_batch);
        params.cache_max_seq_len = it->value("max_cached_seq_len", params.cache_max_seq_len);
    }

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

    if (auto it = trainConfig.find("micro_validation"); it != trainConfig.end()) {
        const auto& micro = *it;
        if (micro.is_boolean()) {
            params.micro_validation_enabled = micro.get<bool>();
        } else if (micro.is_object()) {
            params.micro_validation_enabled = micro.value("enabled", params.micro_validation_enabled);
            params.micro_validation_interval = micro.value("interval", params.micro_validation_interval);
            params.micro_validation_batch_limit = micro.value(
                "batch_limit",
                micro.value("batches", params.micro_validation_batch_limit));
            params.micro_validation_min_step = micro.value("min_step", params.micro_validation_min_step);
            params.micro_validation_prefer_short = micro.value("prefer_short", params.micro_validation_prefer_short);
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
                params.telemetry_warmup_steps = base.value("warmup_steps", params.telemetry_warmup_steps);
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
        }
    }
    // Legacy: simple bool telemetry_control_enabled (backwards compat removed per Rule 20)
    // If you had "telemetry_control_enabled": true, migrate to "telemetry_control": { "enabled": true }

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

        if (auto num_it = loss_cfg.find("numeric_head"); num_it != loss_cfg.end()) {
            const auto& num = *num_it;
            if (num.is_boolean()) {
                params.loss_numeric_head_enabled = num.get<bool>();
            } else if (num.is_object()) {
                params.loss_numeric_head_enabled = num.value("enabled", params.loss_numeric_head_enabled);
                params.loss_numeric_head_weight = num.value("loss_weight",
                    num.value("weight", params.loss_numeric_head_weight));
                params.loss_numeric_head_huber_delta = num.value("huber_delta", params.loss_numeric_head_huber_delta);
                params.loss_numeric_head_log_scale = num.value("log_scale", params.loss_numeric_head_log_scale);
            }
        }
    }
    
    // LM Head centering configuration (Issue #37 / #40)
    // When enabled, centers hidden states before LM head projection.
    // Set to false for standard PyTorch-style implementation.
    params.lm_head_centering_enabled = false;  // Default to disabled (standard implementation)
    params.lm_head_center_hidden_states = false;
    params.lm_head_recenter_gradients = false;
    if (auto it = trainConfig.find("lm_head_centering"); it != trainConfig.end() && it->is_object()) {
        const auto& lmc = *it;
        params.lm_head_centering_enabled = lmc.value("enabled", false);
        params.lm_head_center_hidden_states = lmc.value("center_hidden_states", false);
        params.lm_head_recenter_gradients = lmc.value("recenter_gradients", false);
    }
    
    // Load stability overrides - ALWAYS parse values even if disabled
    // (Phase1_Startup copies them unconditionally, so they must be initialized)
    params.stability_overrides_enabled = trainConfig.value("stability_overrides_enabled", params.stability_overrides_enabled);
    if (auto it = trainConfig.find("stability_overrides"); it != trainConfig.end() && it->is_object()) {
        const auto& stab = *it;
        params.stability_override_batch_size = stab.value("batch_size", params.stability_override_batch_size);
        params.stability_override_max_seq_len = stab.value("max_seq_len", params.stability_override_max_seq_len);
        params.stability_override_max_tokens_per_batch = stab.value("max_tokens_per_batch", params.stability_override_max_tokens_per_batch);
        params.stability_override_clip_abs = stab.value("clip_abs", params.stability_override_clip_abs);
        params.stability_override_clip_norm = stab.value("clip_per_token", params.stability_override_clip_norm);
        params.stability_override_lr_min = stab.value("lr_min", params.stability_override_lr_min);
    }
    
    // Load scratch blocks configuration
    if (auto it = trainConfig.find("scratch_blocks"); it != trainConfig.end()) {
        const auto& scratch = *it;
        if (scratch.is_boolean()) {
            params.scratch_blocks_enabled = scratch.get<bool>();
        } else if (scratch.is_object()) {
            params.scratch_blocks_enabled = scratch.value("enabled", params.scratch_blocks_enabled);
            params.scratch_max_tokens_per_block = scratch.value("max_tokens_per_block", params.scratch_max_tokens_per_block);
            params.scratch_num_blocks = scratch.value("num_blocks", params.scratch_num_blocks);
            params.scratch_write_combined = scratch.value("use_write_combined", params.scratch_write_combined);
        }
    }
    
    // Load scratch_block_reasoning configuration (model config)
    if (auto it = trainConfig.find("scratch_block_reasoning"); it != trainConfig.end() && it->is_object()) {
        const auto& sbr = *it;
        params.scratch_block_reasoning_enabled = sbr.value("enabled", params.scratch_block_reasoning_enabled);
        params.scratch_block_reasoning_atom_embedding_dim = sbr.value("atom_embedding_dim", params.scratch_block_reasoning_atom_embedding_dim);
        params.scratch_block_reasoning_max_atoms = sbr.value("max_atoms", params.scratch_block_reasoning_max_atoms);
        params.scratch_block_reasoning_atom_scale = sbr.value("atom_scale", params.scratch_block_reasoning_atom_scale);
    }
    
    // Load activation quantization configuration
    if (auto it = trainConfig.find("activation_quantization"); it != trainConfig.end() && it->is_object()) {
        const auto& quant = *it;
        params.activation_quantization_enabled = quant.value("enabled", params.activation_quantization_enabled);
        params.activation_quantization_apply_to_embeddings = quant.value("apply_to_embeddings", params.activation_quantization_apply_to_embeddings);
        params.activation_quantization_apply_to_encoder_outputs = quant.value("apply_to_encoder_outputs", params.activation_quantization_apply_to_encoder_outputs);
        params.activation_quantization_apply_to_layer_caches = quant.value("apply_to_layer_caches", params.activation_quantization_apply_to_layer_caches);
        params.activation_quantization_apply_to_qkv_cache = quant.value("apply_to_qkv_cache", params.activation_quantization_apply_to_qkv_cache);
        params.activation_quantization_apply_to_logits = quant.value("apply_to_logits", params.activation_quantization_apply_to_logits);
        params.activation_quantization_scale = quant.value("scale", params.activation_quantization_scale);
        params.activation_quantization_clip_min = quant.value("clip_min", params.activation_quantization_clip_min);
        params.activation_quantization_clip_max = quant.value("clip_max", params.activation_quantization_clip_max);
        params.activation_quantization_zero_point = quant.value("zero_point", params.activation_quantization_zero_point);
        params.activation_quantization_symmetric = quant.value("symmetric", params.activation_quantization_symmetric);
    }
    
    // Load CUDA execution mode configuration
    if (auto it = trainConfig.find("cuda_execution"); it != trainConfig.end() && it->is_object()) {
        const auto& cuda_exec = *it;
        params.single_stream_mode = cuda_exec.value("single_stream_mode", params.single_stream_mode);
        params.disable_async_frees = cuda_exec.value("disable_async_frees", params.disable_async_frees);
        params.synchronize_after_kernels = cuda_exec.value("synchronize_after_kernels", params.synchronize_after_kernels);
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

inline bool populateTrainingHyperparametersFromConfig(const nlohmann::json& config, TrainingHyperparameters& params) {
    if (config.contains("training") && config["training"].contains("config")) {
        const auto& trainConfig = config["training"]["config"];
        // Rule 20: Validate required fields BEFORE parsing (fail fast)
        validateTrainingConfigJson(trainConfig);
        applyTrainingConfigObject(trainConfig, params);
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

inline bool populateTokenizerConfigFromConfig(const nlohmann::json& config, TokenizerConfig& tokenizer_config) {
    if (!config.contains("tokenizer")) {
        return false;
    }

    const auto& tok = config["tokenizer"];
    if (!tok.is_object()) {
        return false;
    }

    assignTrainingField(tokenizer_config.vocab_size, tok, "vocab_size");
    assignTrainingField(tokenizer_config.max_length, tok, "max_length");
    assignTrainingField(tokenizer_config.min_subword_freq, tok, "min_subword_freq");
    assignTrainingField(tokenizer_config.prune_during_mining, tok, "prune_during_mining");
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
    assignTrainingField(tokenizer_config.enable_char_fallback, tok, "enable_char_fallback");
    assignTrainingField(tokenizer_config.enable_byte_fallback, tok, "enable_byte_fallback");
    assignTrainingField(tokenizer_config.expected_checksum, tok, "expected_checksum");
    assignTrainingField(tokenizer_config.save_text_vocab, tok, "save_text_vocab");

    // Scratch block reasoning configuration
    if (tok.contains("scratch_block_reasoning") && tok["scratch_block_reasoning"].is_object()) {
        const auto& sbr = tok["scratch_block_reasoning"];
        assignTrainingField(tokenizer_config.enable_scratch_block_reasoning, sbr, "enabled");
        assignTrainingField(tokenizer_config.detect_numbers, sbr, "detect_numbers");
        assignTrainingField(tokenizer_config.detect_urls, sbr, "detect_urls");
        assignTrainingField(tokenizer_config.detect_emails, sbr, "detect_emails");
        assignTrainingField(tokenizer_config.detect_paths, sbr, "detect_paths");
        assignTrainingField(tokenizer_config.detect_dates, sbr, "detect_dates");
        assignTrainingField(tokenizer_config.detect_code_literals, sbr, "detect_code_literals");
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
        snapshot.has_tokenizer = detail::populateTokenizerConfigFromConfig(snapshot.document, snapshot.tokenizer_config);
        snapshot.has_data_collection = detail::populateDataCollectionConfigFromConfig(snapshot.document, snapshot.data_collection_config);
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
        paths.checkpoints + "|" + paths.collected + "|" + paths.verified + "|" +
        paths.logs + "|" + paths.training_status + "|" + paths.merge_checkpoints_exe +
        "|" + paths.collector_log + "|" + paths.source_config;

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
    std::cout << "  max_length: " << config.max_length << std::endl;
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
