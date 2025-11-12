#pragma once

#include <string>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>

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

inline bool loadGrimTextPaths(GrimTextPaths& paths, const std::string& configPath = "ai_config.json") {
    try {
        std::filesystem::path configFilePath(configPath);
        
        // Try to find ai_config.json if not found at specified path
        if (!std::filesystem::exists(configFilePath)) {
            // Try current directory
            configFilePath = std::filesystem::current_path() / "ai_config.json";
            
            if (!std::filesystem::exists(configFilePath)) {
                // Try parent directories (up to 5 levels)
                std::filesystem::path searchPath = std::filesystem::current_path();
                bool found = false;
                for (int i = 0; i < 5 && searchPath.has_parent_path(); ++i) {
                    searchPath = searchPath.parent_path();
                    auto candidatePath = searchPath / "ai_config.json";
                    if (std::filesystem::exists(candidatePath)) {
                        configFilePath = candidatePath;
                        found = true;
                        break;
                    }
                }
                
                if (!found) {
                    std::cerr << "[Config] ERROR: ai_config.json not found at: " << configPath << std::endl;
                    std::cerr << "[Config]        Also searched current directory and parent directories" << std::endl;
                    return false;
                }
            }
        }
        
        std::cout << "[Config] Loading GRIM-text paths from: " << configFilePath << std::endl;
        
        std::ifstream configFile(configFilePath);
        if (!configFile.is_open()) {
            std::cerr << "[Config] ERROR: Could not open ai_config.json at: " << configFilePath << std::endl;
            return false;
        }
        
        nlohmann::json config;
        configFile >> config;
        configFile.close();
        
        // Navigate to paths.grim_text
        if (!config.contains("paths")) {
            std::cerr << "[Config] WARNING: ai_config.json does not contain 'paths' section" << std::endl;
            return false;
        }
        
        if (!config["paths"].contains("grim_text")) {
            std::cerr << "[Config] WARNING: ai_config.json does not contain 'paths.grim_text' section" << std::endl;
            return false;
        }
        
        auto& grimTextPaths = config["paths"]["grim_text"];
        
        // Load each path if present
        if (grimTextPaths.contains("vocab")) {
            paths.vocab = grimTextPaths["vocab"].get<std::string>();
        }
        
        if (grimTextPaths.contains("model")) {
            paths.model = grimTextPaths["model"].get<std::string>();
        }
        
        if (grimTextPaths.contains("training_data")) {
            paths.training_data = grimTextPaths["training_data"].get<std::string>();
        }
        
        if (grimTextPaths.contains("checkpoints")) {
            paths.checkpoints = grimTextPaths["checkpoints"].get<std::string>();
        }
        
        if (grimTextPaths.contains("collected")) {
            paths.collected = grimTextPaths["collected"].get<std::string>();
        }
        
        if (grimTextPaths.contains("verified")) {
            paths.verified = grimTextPaths["verified"].get<std::string>();
        }
        
        if (grimTextPaths.contains("logs")) {
            paths.logs = grimTextPaths["logs"].get<std::string>();
        }
        
        if (grimTextPaths.contains("training_status")) {
            paths.training_status = grimTextPaths["training_status"].get<std::string>();
        }
        
        if (grimTextPaths.contains("merge_checkpoints_exe")) {
            paths.merge_checkpoints_exe = grimTextPaths["merge_checkpoints_exe"].get<std::string>();
        }
        
        if (grimTextPaths.contains("collector_log")) {
            paths.collector_log = grimTextPaths["collector_log"].get<std::string>();
        }
        
        if (grimTextPaths.contains("source_config")) {
            paths.source_config = grimTextPaths["source_config"].get<std::string>();
        }
        
        // Print loaded paths for debugging
        paths.printPaths();
        
        return paths.isValid();
        
    } catch (const std::exception& e) {
        std::cerr << "[Config] ERROR: Exception loading paths from ai_config.json: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Load training configuration from ai_config.json
 * 
 * This loads the training hyperparameters (epochs, batch_size, learning_rate, etc.)
 * from the training.config section of ai_config.json.
 */
struct TrainingHyperparameters {
    int epochs = 10;
    int batch_size = 8;
    float learning_rate = 0.0001f;
    int max_seq_len = 8192;
    int warmup_steps = 1000;
    bool use_gpu = true;
    bool use_flash_attention = true;
};

inline bool loadTrainingHyperparameters(TrainingHyperparameters& params, const std::string& configPath = "ai_config.json") {
    try {
        std::filesystem::path configFilePath(configPath);
        
        // Find ai_config.json (same logic as above)
        if (!std::filesystem::exists(configFilePath)) {
            configFilePath = std::filesystem::current_path() / "ai_config.json";
            
            if (!std::filesystem::exists(configFilePath)) {
                std::filesystem::path searchPath = std::filesystem::current_path();
                bool found = false;
                for (int i = 0; i < 5 && searchPath.has_parent_path(); ++i) {
                    searchPath = searchPath.parent_path();
                    auto candidatePath = searchPath / "ai_config.json";
                    if (std::filesystem::exists(candidatePath)) {
                        configFilePath = candidatePath;
                        found = true;
                        break;
                    }
                }
                
                if (!found) {
                    std::cerr << "[Config] ERROR: ai_config.json not found" << std::endl;
                    return false;
                }
            }
        }
        
        std::ifstream configFile(configFilePath);
        if (!configFile.is_open()) {
            return false;
        }
        
        nlohmann::json config;
        configFile >> config;
        configFile.close();
        
        // Load training config if present
        if (config.contains("training") && config["training"].contains("config")) {
            auto& trainConfig = config["training"]["config"];
            
            if (trainConfig.contains("epochs")) {
                params.epochs = trainConfig["epochs"].get<int>();
            }
            if (trainConfig.contains("batch_size")) {
                params.batch_size = trainConfig["batch_size"].get<int>();
            }
            if (trainConfig.contains("learning_rate")) {
                params.learning_rate = trainConfig["learning_rate"].get<float>();
            }
            if (trainConfig.contains("max_seq_len")) {
                params.max_seq_len = trainConfig["max_seq_len"].get<int>();
            }
            if (trainConfig.contains("warmup_steps")) {
                params.warmup_steps = trainConfig["warmup_steps"].get<int>();
            }
            if (trainConfig.contains("use_gpu")) {
                params.use_gpu = trainConfig["use_gpu"].get<bool>();
            }
            if (trainConfig.contains("use_flash_attention")) {
                params.use_flash_attention = trainConfig["use_flash_attention"].get<bool>();
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[Config] ERROR: Exception loading training config: " << e.what() << std::endl;
        return false;
    }
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
