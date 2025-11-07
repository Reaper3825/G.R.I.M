//======================================================//
//  GRIM-text Server Manager
//  Manages lifecycle of grim_text_server.exe
//  
//  Author: GRIM Development Team
//  Date: November 6, 2025
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

class GRIMTextServerManager {
public:
    static GRIMTextServerManager& getInstance();
    
    // Lifecycle management
    bool start();
    void shutdown();
    bool isRunning() const;
    
    // Health check
    bool checkHealth(int timeoutMs = 5000);
    
    // Training control server
    bool startTrainingControlServer();
    void stopTrainingControlServer();
    bool isTrainingControlServerRunning() const;
    
    // Configuration
    void setServerPath(const std::string& path);
    void setServerURL(const std::string& url);
    
    std::string getServerURL() const { return serverURL_; }
    std::string getTrainingControlURL() const { return trainingControlURL_; }
    
    // Destructor must be public for unique_ptr
    ~GRIMTextServerManager();
    
private:
    GRIMTextServerManager();
    
    // Prevent copying
    GRIMTextServerManager(const GRIMTextServerManager&) = delete;
    GRIMTextServerManager& operator=(const GRIMTextServerManager&) = delete;
    
    std::string serverPath_;
    std::string serverURL_;
    std::string trainingControlURL_;
    std::atomic<bool> running_;
    std::atomic<bool> trainingControlRunning_;
    
#ifdef _WIN32
    PROCESS_INFORMATION processInfo_;
    HANDLE hProcess_;
    PROCESS_INFORMATION trainingControlProcessInfo_;
    HANDLE hTrainingControlProcess_;
#endif
};

// Global instance
extern std::unique_ptr<GRIMTextServerManager> g_grimTextServerManager;

// Helper functions
bool startGRIMTextServer();
void stopGRIMTextServer();
bool isGRIMTextServerRunning();

// Training control server helpers
bool startTrainingControlServer();
void stopTrainingControlServer();
bool isTrainingControlServerRunning();

} // namespace GRIM
