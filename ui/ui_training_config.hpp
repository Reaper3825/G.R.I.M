#pragma once
#include <nlohmann/json.hpp>
#include "../control/training_control_client.hpp"
#include "../resources.hpp"
#include <string>
#include <fstream>
#include <iomanip>

namespace GRIMText {

class TrainingConfigManager {
public:
    static bool usesCanonicalRuntimeConfigPath(const std::string& configPath) {
        return configPath.empty() || configPath == AI_CONFIG_FILE;
    }

    static TrainingConfig loadFromJSON(const std::string& configPath = "ai_config.json") {
        TrainingConfig config;
        
        try {
            nlohmann::json j;
            if (usesCanonicalRuntimeConfigPath(configPath)) {
                j = loadGrimRuntimeAiConfig();
            } else {
                std::ifstream file(configPath);
                if (!file.is_open()) {
                    return config; // Return defaults
                }
                file >> j;
            }
            
            if (j.contains("training") && j["training"].contains("config")) {
                auto& tc = j["training"]["config"];
                
                // Load only editable hyperparameters here - snapshot path leaves are staged under training.config
                config.epochs = tc.value("epochs", 3);
                config.batchSize = tc.value("batch_size", 8);
                config.learningRate = tc.value("learning_rate", 0.0001f);
                config.maxSeqLen = tc.value("max_seq_len", 8192);
                config.warmupSteps = tc.value("warmup_steps", 1000);
                config.useGPU = tc.value("use_gpu", true);
                config.useFlashAttention = tc.value("use_flash_attention", true);
            }
        } catch (const std::exception& e) {
            // Return defaults on error
        }
        
        return config;
    }
    
    static bool saveToJSON(const TrainingConfig& config, const std::string& configPath = "ai_config.json") {
        try {
            nlohmann::json pending;
            pending["training"]["config"] = {
                {"epochs", config.epochs},
                {"batch_size", config.batchSize},
                {"learning_rate", config.learningRate},
                {"max_seq_len", config.maxSeqLen},
                {"warmup_steps", config.warmupSteps},
                {"use_gpu", config.useGPU},
                {"use_flash_attention", config.useFlashAttention}
            };

            if (usesCanonicalRuntimeConfigPath(configPath)) {
                saveGrimRuntimeAiConfig(pending);
                return true;
            }

            std::ifstream fileIn(configPath);
            nlohmann::json j;
            if (fileIn.is_open()) {
                fileIn >> j;
                fileIn.close();
            }

            if (!j.contains("training") || !j["training"].is_object()) {
                j["training"] = nlohmann::json::object();
            }
            if (!j["training"].contains("config") || !j["training"]["config"].is_object()) {
                j["training"]["config"] = nlohmann::json::object();
            }

            for (const auto& [key, value] : pending["training"]["config"].items()) {
                j["training"]["config"][key] = value;
            }
            
            std::ofstream fileOut(configPath);
            if (!fileOut.is_open()) {
                return false;
            }
            
            fileOut << std::setw(4) << j << std::endl;
            fileOut.close();
            
            return true;
        } catch (const std::exception& e) {
            return false;
        }
    }
    
    static std::string getServerHost(const std::string& configPath = "ai_config.json") {
        try {
            nlohmann::json j;
            if (usesCanonicalRuntimeConfigPath(configPath)) {
                j = loadGrimRuntimeAiConfig();
            } else {
                std::ifstream file(configPath);
                if (!file.is_open()) {
                    return "127.0.0.1";
                }
                file >> j;
            }
            
            if (j.contains("training") && j["training"].contains("server_host")) {
                return j["training"]["server_host"].get<std::string>();
            }
        } catch (...) {}
        
        return "127.0.0.1";
    }
    
    static int getServerPort(const std::string& configPath = "ai_config.json") {
        try {
            nlohmann::json j;
            if (usesCanonicalRuntimeConfigPath(configPath)) {
                j = loadGrimRuntimeAiConfig();
            } else {
                std::ifstream file(configPath);
                if (!file.is_open()) {
                    return 11436;
                }
                file >> j;
            }
            
            if (j.contains("training") && j["training"].contains("server_port")) {
                return j["training"]["server_port"].get<int>();
            }
        } catch (...) {}
        
        return 11436;
    }
};

} // namespace GRIMText
