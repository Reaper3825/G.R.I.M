//======================================================//
//  GRIM Native Backend - HTTP Client for GRIM-text Server
//======================================================//

#include "grim_backend.hpp"
#include "grim_text_server_manager.hpp"  // ✅ NEW: For server management
#include "../logger.hpp"
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <thread>
#include <chrono>

using json = nlohmann::json;

// External config
extern nlohmann::json aiConfig;

namespace GRIM {

std::unique_ptr<GRIMBackend> g_grimBackend;

bool initGRIMBackend(const std::string& model_path, const std::string& vocab_path) {
    g_grimBackend = std::make_unique<GRIMBackend>();
    return g_grimBackend->initialize(model_path, vocab_path);
}

bool GRIMBackend::initialize(const std::string& model_path, const std::string& vocab_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    LOG_DEBUG("GRIM", "Initializing GRIM-text backend (HTTP client)...");
    
    model_path_ = model_path;
    vocab_path_ = vocab_path;
    
    // GRIM-text runs as separate HTTP server (like Ollama)
    // Server: resources/models/GRIM-text/grim_text_server.exe
    // URL: http://127.0.0.1:11435
    
    initialized_ = true;
    LOG_DEBUG("GRIM", "GRIM-text backend initialized (offline HTTP mode)");
    return true;
}

std::string GRIMBackend::generate(const std::string& prompt, int max_tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        return "Error: Backend not initialized";
    }
    
    try {
        // Get GRIM-text server URL from config (default: http://127.0.0.1:11435)
        std::string server_url = aiConfig.value("grim_text_url", "http://127.0.0.1:11435");
        
        // ✅ NEW: Check if server is running, try to start if not
        if (!GRIMTextServerManager::getInstance().isRunning()) {
            LOG_DEBUG("GRIM", "GRIM-text server not running, attempting to start...");
            if (!startGRIMTextServer()) {
                return "Error: GRIM-text server offline and failed to start";
            }
        }
        
        // Build request payload (Ollama-compatible format)
        json request = {
            {"model", "grim-text"},
            {"prompt", prompt},
            {"max_tokens", max_tokens},
            {"temperature", 0.8f},
            {"stream", false}
        };
        
        LOG_DEBUG("GRIM", "Calling GRIM-text server: " + server_url + "/api/generate");
        
        // ✅ NEW: Retry logic for transient failures
        const int maxRetries = 3;
        int retryDelayMs = 1000;
        
        for (int attempt = 1; attempt <= maxRetries; ++attempt) {
            // Call GRIM-text HTTP server
            auto resp = cpr::Post(
                cpr::Url{server_url + "/api/generate"},
                cpr::Header{{"Content-Type", "application/json"}},
                cpr::Body{request.dump()},
                cpr::Timeout{30000}  // 30 second timeout
            );
            
            if (resp.status_code == 200) {
                auto response = json::parse(resp.text);
                
                if (response.contains("response")) {
                    return response["response"].get<std::string>();
                } else {
                    LOG_ERROR("GRIM", "GRIM-text response missing 'response' field");
                    return "Error: Invalid response format";
                }
            } else if (resp.status_code == 0) {
                LOG_ERROR("GRIM", "GRIM-text server not reachable (attempt " + 
                         std::to_string(attempt) + "/" + std::to_string(maxRetries) + ")");
                
                if (attempt < maxRetries) {
                    LOG_DEBUG("GRIM", "Retrying in " + std::to_string(retryDelayMs) + "ms...");
                    std::this_thread::sleep_for(std::chrono::milliseconds(retryDelayMs));
                    retryDelayMs *= 2;  // Exponential backoff
                    continue;
                }
                
                return "Error: GRIM-text server offline. Check server logs.";
            } else {
                LOG_ERROR("GRIM", "GRIM-text HTTP " + std::to_string(resp.status_code) + ": " + resp.text);
                return "Error: Server returned " + std::to_string(resp.status_code);
            }
        }
        
        return "Error: Max retries exceeded";
        
    } catch (const std::exception& e) {
        LOG_ERROR("GRIM", std::string("GRIM-text error: ") + e.what());
        return "Error: " + std::string(e.what());
    }
}

std::string GRIMBackend::generateWithHistory(
    const std::string& prompt,
    const std::vector<nlohmann::json>& history,
    int max_tokens)
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        return "Error: Backend not initialized";
    }
    
    try {
        std::string server_url = aiConfig.value("grim_text_url", "http://127.0.0.1:11435");
        
        // Build messages array from history
        json messages = json::array();
        for (const auto& msg : history) {
            messages.push_back(msg);
        }
        
        // Add current prompt
        messages.push_back({
            {"role", "user"},
            {"content", prompt}
        });
        
        json request = {
            {"model", "grim-text"},
            {"messages", messages},
            {"max_tokens", max_tokens},
            {"temperature", 0.8f},
            {"stream", false}
        };
        
        LOG_DEBUG("GRIM", "Calling GRIM-text chat endpoint with " + std::to_string(messages.size()) + " messages");
        
        auto resp = cpr::Post(
            cpr::Url{server_url + "/api/chat"},
            cpr::Header{{"Content-Type", "application/json"}},
            cpr::Body{request.dump()},
            cpr::Timeout{30000}
        );
        
        if (resp.status_code == 200) {
            auto response = json::parse(resp.text);
            
            if (response.contains("message") && response["message"].contains("content")) {
                return response["message"]["content"].get<std::string>();
            } else {
                LOG_ERROR("GRIM", "GRIM-text chat response invalid");
                return "Error: Invalid chat response";
            }
        } else {
            LOG_ERROR("GRIM", "GRIM-text chat HTTP " + std::to_string(resp.status_code));
            return "Error: Server returned " + std::to_string(resp.status_code);
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("GRIM", std::string("GRIM-text chat error: ") + e.what());
        return "Error: " + std::string(e.what());
    }
}

} // namespace GRIM
