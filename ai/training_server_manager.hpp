//======================================================//
//  Training Control Server Manager
//  Manages lifecycle of training_control_server.exe
//  
//  Separate from grim_text_server_manager.hpp which
//  handles the inference server (grim_text_server.exe)
//  
//  Author: GRIM Development Team
//  Date: November 7, 2025
//  Version: 1.0.0
//======================================================//

#pragma once

#include <string>
#include <memory>
#include <atomic>

#ifdef _WIN32
#include <windows.h>
#endif

namespace GRIM {

class TrainingServerManager {
public:
    static TrainingServerManager& getInstance();
    
    // Lifecycle management
    bool start();
    void shutdown();
    bool isRunning() const;
    
    // Health check
    bool checkHealth(int timeoutMs = 5000);
    
    // Check if server is already running on port (even if not managed by us)
    bool isServerAlreadyRunningOnPort();
    
    // Configuration
    void setServerPath(const std::string& path);
    void setServerURL(const std::string& url);
    
    std::string getServerURL() const { return serverURL_; }
    
    // Destructor must be public for unique_ptr
    ~TrainingServerManager();
    
private:
    TrainingServerManager();
    
    // Prevent copying
    TrainingServerManager(const TrainingServerManager&) = delete;
    TrainingServerManager& operator=(const TrainingServerManager&) = delete;
    
    std::string serverPath_;
    std::string serverURL_;
    std::atomic<bool> running_;
    
#ifdef _WIN32
    PROCESS_INFORMATION processInfo_;
    HANDLE hProcess_;
#endif
};

// Global instance
extern std::unique_ptr<TrainingServerManager> g_trainingServerManager;

// Helper functions
bool startTrainingServer();
void stopTrainingServer();
bool isTrainingServerRunning();

} // namespace GRIM
