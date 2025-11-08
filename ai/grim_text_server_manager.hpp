//======================================================//
//  GRIM-text Server Manager
//  Manages lifecycle of grim_text_server.exe (INFERENCE ONLY)
//  
//  For training control, use training_server_manager.hpp
//  
//  Author: GRIM Development Team
//  Date: November 7, 2025
//  Version: 2.0.0 - Separated from training control
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
    
    // Configuration
    void setServerPath(const std::string& path);
    void setServerURL(const std::string& url);
    
    std::string getServerURL() const { return serverURL_; }
    
    // Destructor must be public for unique_ptr
    ~GRIMTextServerManager();
    
private:
    GRIMTextServerManager();
    
    // Prevent copying
    GRIMTextServerManager(const GRIMTextServerManager&) = delete;
    GRIMTextServerManager& operator=(const GRIMTextServerManager&) = delete;
    
    std::string serverPath_;
    std::string serverURL_;
    std::atomic<bool> running_;
    
#ifdef _WIN32
    PROCESS_INFORMATION processInfo_;
    HANDLE hProcess_;
#endif
};

// Global instance
extern std::unique_ptr<GRIMTextServerManager> g_grimTextServerManager;

// Helper functions
bool startGRIMTextServer();
void stopGRIMTextServer();
bool isGRIMTextServerRunning();

} // namespace GRIM
