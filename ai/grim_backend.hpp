//======================================================//
//  GRIM Native Language Model Backend
//  Simple wrapper for external model calls
//  
//  Author: GRIM Development Team
//  Date: November 5, 2025
//  Version: 1.0.0 - External Model Reference
//======================================================//

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>

namespace GRIM {

//======================================================//
//  Simple Backend (External Model Reference)
//======================================================//

struct GRIMBackendConfig {
    std::string model_path;
    std::string vocab_path;
    int max_tokens = 256;
    float temperature = 0.7f;
};

class GRIMBackend {
public:
    GRIMBackend() = default;
    ~GRIMBackend() = default;
    
    bool initialize(const std::string& model_path, const std::string& vocab_path);
    
    std::string generate(const std::string& prompt, int max_tokens = 256);
    
    std::string generateWithHistory(
        const std::string& prompt,
        const std::vector<nlohmann::json>& history,
        int max_tokens = 256
    );
    
    bool isInitialized() const { return initialized_; }
    
private:
    bool initialized_ = false;
    std::string model_path_;
    std::string vocab_path_;
    mutable std::mutex mutex_;
};

// Global backend instance
extern std::unique_ptr<GRIMBackend> g_grimBackend;

// Initialize global backend
bool initGRIMBackend(const std::string& model_path, const std::string& vocab_path);

} // namespace GRIM
