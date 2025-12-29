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
 */
struct TrainingHyperparameters {
    // Log Recorder configuration
    LogRecorderConfig log_recorder;

    int epochs = 15;
    int64_t seed = -1;  // -1 = random (timestamp), >= 0 = deterministic reproducible seed
    int batch_size = 24;
    int gradient_accumulation_steps = 1;  // Accumulate gradients over N batches before stepping optimizer
    bool single_batch_overfit_enabled = false;  // Repeat first batch for overfit/debug
    int single_batch_overfit_max_steps = 0;     // 0 = default to 1 step when enabled
    std::string batch_strategy = "SIMILARITY_GROUPED";  // RANDOM, GREEDY, SIMILARITY_GROUPED, BEST_FIT_DECREASING
    float learning_rate = 0.00003f;
    float weight_decay = 0.01f;
    float grad_clip_norm = 10.0f;
    bool per_token_grad_scale = false;  // Match HyperParameters::DEFAULT_GRAD_SCALE_PER_TOKEN
    int warmup_steps = 10;
    int max_seq_len = 8192;
    int log_interval = 1;
    int validation_interval = 100;
    int checkpoint_interval = 1000;
    bool use_gpu = true;
    bool use_flash_attention = true;
    bool dynamic_lr_enabled = false;
    bool dynamic_lr_autogenerate = true;
    float dynamic_lr_min = 1.0e-6f;
    float dynamic_lr_max = 3.0e-4f;
    float dynamic_lr_increase_factor = 1.05f;
    float dynamic_lr_decrease_factor = 0.5f;
    float dynamic_lr_upper_grad_norm = 12.0f;
    float dynamic_lr_lower_grad_norm = 4.0f;
    float dynamic_lr_max_loss_jump = 1.5f;
    float dynamic_lr_smoothing = 0.2f;
    int dynamic_lr_cooldown_steps = 5;
    int dynamic_lr_warmup_steps = 25;
    float dynamic_lr_max_step_up_ratio = 1.12f;
    float dynamic_lr_max_step_down_ratio = 0.72f;
    bool dynamic_lr_auto_band = true;
    float dynamic_lr_band_sigma = 1.5f;
    float dynamic_lr_band_floor = 3.0f;
    float dynamic_lr_band_ceiling = 250.0f;
    int dynamic_lr_band_min_samples = 12;
    float dynamic_lr_band_min_span = 1.0f;
    bool dynamic_lr_adaptive_smoothing = true;
    float dynamic_lr_smoothing_min = 0.1f;
    float dynamic_lr_smoothing_max = 0.6f;
    float dynamic_lr_variance_reference = 25.0f;
    bool dynamic_lr_adaptive_cooldown = true;
    int dynamic_lr_cooldown_min = 1;
    int dynamic_lr_cooldown_max = 8;
    bool dynamic_lr_adaptive_loss = true;
    float dynamic_lr_loss_sigma = 3.0f;
    int dynamic_lr_loss_min_samples = 8;
    float dynamic_lr_loss_floor = 0.25f;
    bool dynamic_lr_guard_logging = true;
    int dynamic_lr_guard_floor_steps = 200;
    float dynamic_lr_guard_grad_multiplier = 1.6f;
    int dynamic_lr_guard_loss_patience = 12;
    float dynamic_lr_guard_loss_multiplier = 1.05f;
    int dynamic_lr_baseline_capture_steps = 32;
    float dynamic_lr_baseline_drift = 0.05f;
    int dynamic_lr_momentum_interval = 12;
    float dynamic_lr_momentum_gain = 0.35f;
    float dynamic_lr_momentum_decay = 0.7f;
    int dynamic_lr_safety_interval = 4;
    float dynamic_lr_safety_gain = 0.1f;
    float dynamic_lr_safety_scale = 2.4f;
    bool soft_restart_enabled = true;
    float soft_restart_loss_increase_threshold = 0.2f;
    int soft_restart_max_step_window = 50;
    int soft_restart_cooldown_steps = 200;
    bool auto_stop_enabled = true;
    int auto_stop_plateau_patience = 3;
    float auto_stop_plateau_min_delta = 0.01f;
    float auto_stop_high_loss_threshold = 9.0f;
    int auto_stop_high_loss_patience = 2;
    int cache_max_batch = 24;
    int cache_max_seq_len = 2048;
    bool micro_validation_enabled = false;
    int micro_validation_interval = 200;
    int micro_validation_batch_limit = 3;
    int micro_validation_min_step = 128;
    bool micro_validation_prefer_short = true;
    bool guess_aux_enabled = true;
    float guess_aux_lambda = 0.25f;
    float guess_aux_min_confidence = 0.7f;
    bool shuffle_train_enabled = true;
    int shuffle_train_epochs = 0;  // 0 == all epochs
    
    // Telemetry control (includes plateau noise injection)
    bool telemetry_control_enabled = true;  // Enable gradient spike detection and interventions
    float telemetry_spike_mild_threshold = 3.0f;
    float telemetry_spike_moderate_threshold = 5.0f;
    float telemetry_spike_severe_threshold = 10.0f;
    float telemetry_moderate_grad_scale = 0.5f;
    int telemetry_moderate_cooldown_extension = 3;
    float telemetry_min_grad_for_nonzero_loss = 1e-10f;
    float telemetry_loss_threshold_for_grad_check = 0.01f;
    int telemetry_max_consecutive_zero_grad_steps = 3;
    float telemetry_seq_len_regime_change_threshold = 0.3f;
    int telemetry_regime_change_suppression_steps = 2;
    float telemetry_volatility_damping_threshold = 100.0f;
    float telemetry_max_volatility_damping = 1.0f;
    float telemetry_gradient_decay_threshold = 0.0f;
    float telemetry_max_decay_boost = 1.0f;
    float telemetry_progress_boost_threshold = 100.0f;
    float telemetry_max_progress_boost = 1.0f;
    float telemetry_outlier_frequency_trigger = 0.95f;
    float telemetry_outlier_persistence_trigger = 0.90f;
    float telemetry_anchor_drift_sigma_multiplier = 5.0f;
    int telemetry_soft_restart_cooldown_steps = 10;
    int telemetry_warmup_steps = 100;
    int telemetry_baseline_stabilization_steps = 50;
    bool telemetry_verbose_logging = true;
    bool telemetry_fail_loud_on_accumulation_bug = true;
    bool telemetry_plateau_noise_enabled = true;
    int telemetry_plateau_noise_patience = 50;  // Batches of low variance before triggering
    float telemetry_plateau_noise_variance_threshold = 0.001f;  // Loss variance threshold
    float telemetry_plateau_noise_std = 0.001f;  // Noise standard deviation
    bool telemetry_plateau_noise_proportional = true;  // Scale noise by weight magnitude
    int telemetry_plateau_noise_cooldown = 500;  // Batches between injections
    int telemetry_plateau_noise_max_per_epoch = 3;  // Max injections per epoch
    
    // Loss options
    // NOTE: These are runtime defaults loaded from JSON. The actual Loss structs
    // (GRIM::Loss::FocalLossConfig, etc.) use HyperParameters constants as their
    // compile-time defaults. When training starts, these JSON values override those
    // defaults. Keep these in sync with HyperParameters::DEFAULT_LOSS_* values for
    // consistency, but the authoritative defaults are in HyperParameters_GPU.hpp.
    bool loss_label_smoothing_enabled = true;
    float loss_label_smoothing_epsilon = 0.1f;  // Match HyperParameters::DEFAULT_LOSS_LABEL_SMOOTHING_EPSILON
    bool loss_focal_enabled = false;
    float loss_focal_gamma = 2.0f;   // Match HyperParameters::DEFAULT_LOSS_FOCAL_GAMMA
    float loss_focal_alpha = 1.0f;   // Match HyperParameters::DEFAULT_LOSS_FOCAL_ALPHA
    bool loss_preference_enabled = false;
    float loss_preference_beta = 0.1f;  // Match HyperParameters::DEFAULT_LOSS_PREFERENCE_BETA
    bool loss_distillation_enabled = false;
    float loss_distillation_temperature = 1.0f;  // Match HyperParameters::DEFAULT_LOSS_DISTILLATION_TEMPERATURE
    float loss_distillation_lambda = 0.5f;  // Match HyperParameters::DEFAULT_LOSS_DISTILLATION_LAMBDA
    bool loss_masking_enabled = false;
    std::string loss_masking_tag = "";
    
    // Stability overrides
    bool stability_overrides_enabled = false;
    int stability_override_batch_size = 0;
    int stability_override_max_seq_len = 0;
    int stability_override_max_tokens_per_batch = 0;
    float stability_override_clip_abs = 0.0f;
    float stability_override_clip_norm = 0.0f;
    float stability_override_lr_min = 0.0f;
    
    // Scratch blocks
    // NOTE: Authoritative defaults in HyperParameters::DEFAULT_SCRATCH_* constants
    bool scratch_blocks_enabled = true;       // Match DEFAULT_SCRATCH_BLOCKS_ENABLED
    size_t scratch_max_tokens_per_block = 16384;  // Match DEFAULT_SCRATCH_MAX_TOKENS_PER_BLOCK
    size_t scratch_num_blocks = 4;            // Match DEFAULT_SCRATCH_NUM_BLOCKS
    bool scratch_write_combined = false;      // Match DEFAULT_SCRATCH_WRITE_COMBINED
    
    // ScratchBlock reasoning (model config)
    bool scratch_block_reasoning_enabled = true;
    int scratch_block_reasoning_atom_embedding_dim = 64;
    int scratch_block_reasoning_max_atoms = 256;
    float scratch_block_reasoning_atom_scale = 0.1f;
    
    // Activation quantization
    bool activation_quantization_enabled = false;
    bool activation_quantization_apply_to_embeddings = true;
    bool activation_quantization_apply_to_encoder_outputs = false;
    bool activation_quantization_apply_to_layer_caches = false;
    bool activation_quantization_apply_to_qkv_cache = false;
    bool activation_quantization_apply_to_logits = false;
    float activation_quantization_scale = 1.0f;
    float activation_quantization_clip_min = -127.0f;
    float activation_quantization_clip_max = 127.0f;
    int activation_quantization_zero_point = 0;
    bool activation_quantization_symmetric = false;
    
    // CUDA execution mode
    bool single_stream_mode = false;         // Force single CUDA stream (disable overlap)
    bool disable_async_frees = false;        // Use cudaFree instead of cudaFreeAsync
    bool synchronize_after_kernels = false;  // Add cudaDeviceSynchronize after each kernel
    
    // Prediction comparison logging (for debugging spike vs good batches)
    bool prediction_comparison_enabled = false;  // Log model predictions for comparison
    int prediction_comparison_interval = 100;    // Log every N good batches
    int prediction_comparison_top_k = 5;         // Show top-K predictions per position
    int prediction_comparison_max_positions = 10; // Max positions to log per sequence
    std::string prediction_comparison_log_path = "prediction_comparison.log";
    
    // Attention diagnostics - dump attention stats every step during training
    // Use this to diagnose training plateau (saturated attention, gradient collapse)
    bool attention_diag_enabled = true;   // Master switch - WARNING: adds ~10ms per batch
    int attention_diag_layer = -1;         // Which layer to dump (-1 = all, 0 = first, etc)
    int attention_diag_head = 0;           // Which head to dump (-1 = all, 0 = first only)
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
    int max_length = 8192;
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
    uint32_t expected_checksum = 0;
    bool save_text_vocab = false;  // Also save human-readable .txt alongside .bin
    
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
    assignTrainingField(params.warmup_steps, trainConfig, "warmup_steps");
    assignTrainingField(params.log_interval, trainConfig, "log_interval");
    assignTrainingField(params.validation_interval, trainConfig, "validation_interval");
    assignTrainingField(params.checkpoint_interval, trainConfig, "checkpoint_interval");
    assignTrainingField(params.use_gpu, trainConfig, "use_gpu");
    assignTrainingField(params.use_flash_attention, trainConfig, "use_flash_attention");

    if (auto it = trainConfig.find("dynamic_lr"); it != trainConfig.end()) {
        const auto& dlr = *it;
        if (dlr.is_boolean()) {
            params.dynamic_lr_enabled = dlr.get<bool>();
            params.dynamic_lr_autogenerate = true;
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
    
    // Load stability overrides
    params.stability_overrides_enabled = trainConfig.value("stability_overrides_enabled", params.stability_overrides_enabled);
    if (params.stability_overrides_enabled) {
        if (auto it = trainConfig.find("stability_overrides"); it != trainConfig.end() && it->is_object()) {
            const auto& stab = *it;
            params.stability_override_batch_size = stab.value("batch_size", params.stability_override_batch_size);
            params.stability_override_max_seq_len = stab.value("max_seq_len", params.stability_override_max_seq_len);
            params.stability_override_max_tokens_per_batch = stab.value("max_tokens_per_batch", params.stability_override_max_tokens_per_batch);
            params.stability_override_clip_abs = stab.value("clip_abs", params.stability_override_clip_abs);
            params.stability_override_clip_norm = stab.value("clip_per_token", params.stability_override_clip_norm);
            params.stability_override_lr_min = stab.value("lr_min", params.stability_override_lr_min);
        }
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
        applyTrainingConfigObject(config["training"]["config"], params);
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
