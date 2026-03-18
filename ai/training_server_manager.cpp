//======================================================//
//  Training Control Server Manager Implementation
//======================================================//

#include "training_server_manager.hpp"
#include "../logger.hpp"
#include <httplib.h>
#include <filesystem>
#include <thread>
#include <chrono>

#ifdef _WIN32
#include "core/grim_platform.h"
#endif

namespace fs = std::filesystem;

namespace GRIM {

std::unique_ptr<TrainingServerManager> g_trainingServerManager;

TrainingServerManager& TrainingServerManager::getInstance() {
    if (!g_trainingServerManager) {
        g_trainingServerManager = std::unique_ptr<TrainingServerManager>(new TrainingServerManager());
    }
    return *g_trainingServerManager;
}

TrainingServerManager::TrainingServerManager() 
    : serverPath_("control/build/Release/training_control_server.exe"),
      serverURL_("http://127.0.0.1:11436"),
      running_(false)
#ifdef _WIN32
      , hProcess_(nullptr)
#endif
{
#ifdef _WIN32
    ZeroMemory(&processInfo_, sizeof(processInfo_));
#endif
}

TrainingServerManager::~TrainingServerManager() {
    shutdown();
}

void TrainingServerManager::setServerPath(const std::string& path) {
    serverPath_ = path;
}

void TrainingServerManager::setServerURL(const std::string& url) {
    serverURL_ = url;
}

bool TrainingServerManager::checkHealth(int timeoutMs) {
    try {
        httplib::Client client("127.0.0.1", 11436);
        client.set_connection_timeout(0, timeoutMs * 1000);  // seconds, microseconds
        client.set_read_timeout(1, 0);
        
        auto res = client.Get("/health");
        return res && res->status == 200;
    } catch (...) {
        return false;
    }
}

bool TrainingServerManager::isServerAlreadyRunningOnPort() {
    // Check if port 11436 is in use by making a health check
    try {
        httplib::Client client("127.0.0.1", 11436);
        client.set_connection_timeout(0, 500000);  // 500ms timeout
        client.set_read_timeout(0, 500000);
        
        auto res = client.Get("/health");
        if (res && res->status == 200) {
            LOG_DEBUG("TrainingServer", "Detected existing server responding on port 11436");
            return true;
        }
    } catch (...) {
        // Port not responding
    }
    return false;
}

bool TrainingServerManager::start() {
    // ROBUST GUARD 1: Check if we already have a managed instance
    if (running_) {
        LOG_DEBUG("TrainingServer", "Manager already tracking a running server - verifying health");
        
        // Verify the server is actually responding
        if (checkHealth(1000)) {
            LOG_DEBUG("TrainingServer", "Existing managed server verified healthy on port 11436");
            return true;
        } else {
            LOG_ERROR("TrainingServer", "Managed server marked running but not responding - will restart");
            shutdown();
            // Continue to start new instance
        }
    }
    
    // ROBUST GUARD 2: Check if ANY server is running on the port (even if not managed by us)
    if (isServerAlreadyRunningOnPort()) {
        LOG_DEBUG("TrainingServer", "External training control server detected on port 11436");
        LOG_DEBUG("TrainingServer", "Adopting existing server instead of starting new instance");
        
        running_ = true;
#ifdef _WIN32
        hProcess_ = nullptr;
#endif
        
        return true;
    }
    
    LOG_DEBUG("TrainingServer", "No existing server detected - starting new instance...");
    
#ifdef _WIN32
    fs::path serverExe = fs::absolute(serverPath_);
    
    if (!fs::exists(serverExe)) {
        LOG_ERROR("TrainingServer", "Server executable not found: " + serverExe.string());
        LOG_ERROR("TrainingServer", "Expected at: control/build/Release/training_control_server.exe");
        return false;
    }
    
    LOG_DEBUG("TrainingServer", "Server path: " + serverExe.string());
    
    // Setup startup info
    STARTUPINFOA si{};
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;  // Hidden console
    
    ZeroMemory(&processInfo_, sizeof(processInfo_));
    
    // Build command line
    std::string cmdLine = "\"" + serverExe.string() + "\" --port 11436";
    
    std::vector<char> mutableCmd(cmdLine.begin(), cmdLine.end());
    mutableCmd.push_back('\0');
    
    // Get GRIM root directory (current working directory of GRIM.exe)
    fs::path grimRoot = fs::current_path();
    
    LOG_DEBUG("TrainingServer", "Command: " + cmdLine);
    LOG_DEBUG("TrainingServer", "Working directory: " + grimRoot.string());
    
    BOOL success = CreateProcessA(
        nullptr,
        mutableCmd.data(),
        nullptr,
        nullptr,
        FALSE,
        CREATE_NEW_CONSOLE | CREATE_NO_WINDOW,
        nullptr,
        grimRoot.string().c_str(),  // Use GRIM root as working directory
        &si,
        &processInfo_
    );
    
    if (!success) {
        DWORD error = GetLastError();
        LOG_ERROR("TrainingServer", "Failed to launch server. Error: " + std::to_string(error));
        return false;
    }
    
    hProcess_ = processInfo_.hProcess;
    running_ = true;
    
    LOG_DEBUG("TrainingServer", "Server started (PID: " + std::to_string(processInfo_.dwProcessId) + ")");
    LOG_DEBUG("TrainingServer", "Server will be ready in a few seconds...");
    
    // Return immediately - caller will poll for health
    return true;
    
#else
    LOG_ERROR("TrainingServer", "Not implemented for this platform");
    return false;
#endif
}

void TrainingServerManager::shutdown() {
    if (!running_) {
        LOG_DEBUG("TrainingServer", "Shutdown called but server not running");
        return;
    }
    
    LOG_DEBUG("TrainingServer", "Stopping training control server...");
    
#ifdef _WIN32
    // Only terminate if we own the process handle
    if (hProcess_ != nullptr) {
        LOG_DEBUG("TrainingServer", "Terminating managed server process (PID: " + std::to_string(processInfo_.dwProcessId) + ")");
        TerminateProcess(hProcess_, 0);
        WaitForSingleObject(hProcess_, 5000);
        CloseHandle(hProcess_);
        CloseHandle(processInfo_.hThread);
        hProcess_ = nullptr;
        ZeroMemory(&processInfo_, sizeof(processInfo_));
        LOG_DEBUG("TrainingServer", "Managed server process terminated successfully");
    } else {
        LOG_DEBUG("TrainingServer", "Server not managed by this instance (external process)");
        LOG_DEBUG("TrainingServer", "Leaving external server running");
    }
#endif
    
    running_ = false;
    LOG_DEBUG("TrainingServer", "Training control server shutdown complete");
}

bool TrainingServerManager::isRunning() const {
#ifdef _WIN32
    if (!running_) {
        return false;
    }
    
    // If we don't own the process, just check HTTP health
    if (hProcess_ == nullptr) {
        return const_cast<TrainingServerManager*>(this)->checkHealth(1000);
    }
    
    // We own the process - check if it's still alive
    DWORD exitCode = 0;
    if (GetExitCodeProcess(hProcess_, &exitCode)) {
        if (exitCode != STILL_ACTIVE) {
            LOG_ERROR("TrainingServer", "Managed process exited unexpectedly");
            return false;
        }
    }
    
    // Process is running, verify it's responding to HTTP
    return const_cast<TrainingServerManager*>(this)->checkHealth(1000);
#endif
    
    return running_;
}

// Helper functions
bool startTrainingServer() {
    return TrainingServerManager::getInstance().start();
}

void stopTrainingServer() {
    TrainingServerManager::getInstance().shutdown();
}

bool isTrainingServerRunning() {
    return TrainingServerManager::getInstance().isRunning();
}

} // namespace GRIM
