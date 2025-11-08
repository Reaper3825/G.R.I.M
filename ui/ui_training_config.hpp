#pragma once
#include <nlohmann/json.hpp>
#include "../control/training_control_client.hpp"
#include <string>
#include <fstream>

namespace GRIMText {

class TrainingConfigManager {
public:
    static TrainingConfig loadFromJSON(const std::string& configPath = "ai_config.json") {
        TrainingConfig config;
        
        try {
            std::ifstream file(configPath);
            if (!file.is_open()) {
                return config; // Return defaults
            }
            
            nlohmann::json j;
            file >> j;
            
            if (j.contains("training") && j["training"].contains("config")) {
                auto& tc = j["training"]["config"];
                
                config.epochs = tc.value("epochs", 3);
                config.batchSize = tc.value("batch_size", 8);
                config.learningRate = tc.value("learning_rate", 0.0001f);
                config.maxSeqLen = tc.value("max_seq_len", 8192);
                config.warmupSteps = tc.value("warmup_steps", 1000);
                config.useGPU = tc.value("use_gpu", true);
                config.useFlashAttention = tc.value("use_flash_attention", true);
                config.dataPath = tc.value("data_path", "data/training_data.grmt");
                config.vocabPath = tc.value("vocab_path", "models/vocab.bin");
                config.outputPath = tc.value("output_path", "models/grim_text_trained.bin");
            }
        } catch (const std::exception& e) {
            // Return defaults on error
        }
        
        return config;
    }
    
    static bool saveToJSON(const TrainingConfig& config, const std::string& configPath = "ai_config.json") {
        try {
            // Read existing config
            std::ifstream fileIn(configPath);
            nlohmann::json j;
            
            if (fileIn.is_open()) {
                fileIn >> j;
                fileIn.close();
            }
            
            // Create/update training section
            if (!j.contains("training")) {
                j["training"] = nlohmann::json::object();
            }
            
            j["training"]["config"] = {
                {"epochs", config.epochs},
                {"batch_size", config.batchSize},
                {"learning_rate", config.learningRate},
                {"max_seq_len", config.maxSeqLen},
                {"warmup_steps", config.warmupSteps},
                {"use_gpu", config.useGPU},
                {"use_flash_attention", config.useFlashAttention},
                {"data_path", config.dataPath},
                {"vocab_path", config.vocabPath},
                {"output_path", config.outputPath}
            };
            
            // Write back to file
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
            std::ifstream file(configPath);
            if (!file.is_open()) {
                return "127.0.0.1";
            }
            
            nlohmann::json j;
            file >> j;
            
            if (j.contains("training") && j["training"].contains("server_host")) {
                return j["training"]["server_host"].get<std::string>();
            }
        } catch (...) {}
        
        return "127.0.0.1";
    }
    
    static int getServerPort(const std::string& configPath = "ai_config.json") {
        try {
            std::ifstream file(configPath);
            if (!file.is_open()) {
                return 11436;
            }
            
            nlohmann::json j;
            file >> j;
            
            if (j.contains("training") && j["training"].contains("server_port")) {
                return j["training"]["server_port"].get<int>();
            }
        } catch (...) {}
        
        return 11436;
    }
};

} // namespace GRIMText
