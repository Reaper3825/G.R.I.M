#pragma once

#include <string>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <optional>
#include <type_traits>

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
 * @brief Load training configuration from ai_config.json
 * 
 * This loads the training hyperparameters (epochs, batch_size, learning_rate, etc.)
 * from the training.config section of ai_config.json.
 */
struct TrainingHyperparameters {
    int epochs = 15;
    int batch_size = 24;
    float learning_rate = 0.00003f;
    float weight_decay = 0.01f;
    float grad_clip_norm = 10.0f;
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
};

struct AiConfigSnapshot {
    std::filesystem::path config_path;
    nlohmann::json document;
    GrimTextPaths grim_paths;
    TrainingHyperparameters hyperparameters;
    bool has_grim_paths = false;
    bool has_training = false;
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
    assignTrainingField(params.batch_size, trainConfig, "batch_size");
    assignTrainingField(params.learning_rate, trainConfig, "learning_rate");
    assignTrainingField(params.weight_decay, trainConfig, "weight_decay");
    assignTrainingField(params.grad_clip_norm, trainConfig, "gradient_clip");
    assignTrainingField(params.grad_clip_norm, trainConfig, "grad_clip_norm");
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
            }
        }
    }

    if (auto it = trainConfig.find("cache_limits"); it != trainConfig.end() && it->is_object()) {
        params.cache_max_batch = it->value("max_cached_batch", params.cache_max_batch);
        params.cache_max_seq_len = it->value("max_cached_seq_len", params.cache_max_seq_len);
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
}

inline bool populateTrainingHyperparametersFromConfig(const nlohmann::json& config, TrainingHyperparameters& params) {
    bool applied = false;
    if (config.contains("training") && config["training"].contains("config")) {
        applyTrainingConfigObject(config["training"]["config"], params);
        applied = true;
    }

    if (!applied && config.contains("training_hyperparameters")) {
        applyTrainingConfigObject(config["training_hyperparameters"], params);
        applied = true;
    }

    return applied;
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
    std::cout << "[Config] Loading GRIM-text paths from: " << snapshot->config_path << std::endl;
    paths.printPaths();
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

} // namespace Config
} // namespace GRIM
