//======================================================//
//  UI Training Controller (Header-Only)
//  High-level training control interface for UI panels
//  
//  Wraps TrainingControlClient with UI-friendly methods
//  and manages training session state, progress tracking,
//  and server lifecycle.
//  
//  Use this from ui_training_panel.cpp instead of directly
//  calling TrainingControlClient.
//  
//  Author: GRIM Development Team
//  Date: November 12, 2025
//======================================================//

#pragma once
#include "training_control_client.hpp"
#include <memory>
#include <string>
#include <functional>
#include <chrono>
#include <atomic>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#endif

namespace GRIM {
namespace UI {

//======================================================//
//  Training Session State
//======================================================//

struct TrainingSessionState {
    bool isActive = false;
    bool isPaused = false;
    bool isServerRunning = false;
    bool isProcessRunning = false;
    
    GRIMText::Control::TrainingState currentState = GRIMText::Control::TrainingState_Idle;
    GRIMText::TrainingStats stats;
    GRIMText::TrainingConfig config;
    
    // Session metadata
    std::chrono::steady_clock::time_point sessionStartTime;
    std::chrono::steady_clock::time_point lastStatusUpdate;
    int consecutiveErrors = 0;
    std::string lastError;
};

//======================================================//
//  Training Progress Callback Types
//======================================================//

using ProgressCallback = std::function<void(const GRIMText::TrainingStats&)>;
using StateChangeCallback = std::function<void(GRIMText::Control::TrainingState, GRIMText::Control::TrainingState)>;
using ErrorCallback = std::function<void(const std::string&)>;

//======================================================//
//  UI Training Controller
//======================================================//

class UITrainingController {
public:
    UITrainingController(const std::string& serverHost = "127.0.0.1", int serverPort = 11436)
        : client_(serverHost, serverPort)
        , serverHost_(serverHost)
        , serverPort_(serverPort)
        , serverProcessHandle_(nullptr) {
    }
    
    ~UITrainingController() {
        // Ensure cleanup
        stopServer();
    }
    
    // ============================================================
    // Server Lifecycle Management
    // ============================================================
    
    // Start training control server process
    bool startServer(const std::string& serverExecutablePath = "control/training_control_server.exe") {
        std::lock_guard<std::mutex> lock(stateMutex_);
        
        // Check if already running
        if (isServerRunning()) {
            return true;
        }
        
#ifdef _WIN32
        STARTUPINFOA si = { sizeof(si) };
        PROCESS_INFORMATION pi = {};
        
        std::string cmdLine = serverExecutablePath + " --port " + std::to_string(serverPort_);
        
        if (!CreateProcessA(
            nullptr,
            const_cast<char*>(cmdLine.c_str()),
            nullptr, nullptr, FALSE,
            CREATE_NO_WINDOW,
            nullptr, nullptr,
            &si, &pi)) {
            lastError_ = "Failed to start training control server process";
            return false;
        }
        
        serverProcessHandle_ = pi.hProcess;
        CloseHandle(pi.hThread);
        
        // Wait for server to be ready (max 5 seconds)
        for (int i = 0; i < 50; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (client_.isServerRunning()) {
                sessionState_.isServerRunning = true;
                return true;
            }
        }
        
        lastError_ = "Server process started but not responding";
        return false;
#else
        lastError_ = "Server start not implemented for this platform";
        return false;
#endif
    }
    
    // Stop training control server gracefully
    bool stopServer() {
        std::lock_guard<std::mutex> lock(stateMutex_);
        
        if (!sessionState_.isServerRunning) {
            return true;
        }
        
        // Try graceful shutdown first
        if (client_.shutdownServer()) {
            sessionState_.isServerRunning = false;
            
#ifdef _WIN32
            if (serverProcessHandle_) {
                // Wait for process to exit (max 2 seconds)
                WaitForSingleObject(serverProcessHandle_, 2000);
                CloseHandle(serverProcessHandle_);
                serverProcessHandle_ = nullptr;
            }
#endif
            return true;
        }
        
        // Force kill if graceful shutdown failed
#ifdef _WIN32
        if (serverProcessHandle_) {
            TerminateProcess(serverProcessHandle_, 1);
            CloseHandle(serverProcessHandle_);
            serverProcessHandle_ = nullptr;
        }
#endif
        
        sessionState_.isServerRunning = false;
        return true;
    }
    
    // Check if server is running
    bool isServerRunning() {
        bool running = client_.isServerRunning();
        sessionState_.isServerRunning = running;
        return running;
    }
    
    // ============================================================
    // Training Session Control
    // ============================================================
    
    // Start a new training session
    bool startTraining(const GRIMText::TrainingConfig& config) {
        std::lock_guard<std::mutex> lock(stateMutex_);
        
        // Ensure server is running
        if (!sessionState_.isServerRunning && !isServerRunning()) {
            lastError_ = "Training control server is not running";
            if (errorCallback_) {
                errorCallback_(lastError_);
            }
            return false;
        }
        
        // Start training via client
        if (!client_.startTraining(&config)) {
            lastError_ = "Failed to start training: " + client_.getLastError();
            if (errorCallback_) {
                errorCallback_(lastError_);
            }
            return false;
        }
        
        // Update session state
        sessionState_.isActive = true;
        sessionState_.isPaused = false;
        sessionState_.sessionStartTime = std::chrono::steady_clock::now();
        sessionState_.config = config;
        sessionState_.consecutiveErrors = 0;
        
        return true;
    }
    
    // Stop current training session
    bool stopTraining() {
        std::lock_guard<std::mutex> lock(stateMutex_);
        
        if (!sessionState_.isActive) {
            return true;
        }
        
        if (!client_.stopTraining()) {
            lastError_ = "Failed to stop training: " + client_.getLastError();
            return false;
        }
        
        sessionState_.isActive = false;
        sessionState_.isPaused = false;
        
        return true;
    }
    
    // ============================================================
    // Status and Progress Polling
    // ============================================================
    
    // Poll server for current status
    bool pollStatus() {
        std::lock_guard<std::mutex> lock(stateMutex_);
        
        GRIMText::Control::TrainingState newState;
        GRIMText::TrainingStats newStats;
        GRIMText::TrainingConfig newConfig;
        
        if (!client_.getStatus(newState, newStats, newConfig)) {
            sessionState_.consecutiveErrors++;
            
            if (sessionState_.consecutiveErrors > 5) {
                lastError_ = "Lost connection to training control server";
                sessionState_.isServerRunning = false;
                
                if (errorCallback_) {
                    errorCallback_(lastError_);
                }
            }
            return false;
        }
        
        // Reset error counter on success
        sessionState_.consecutiveErrors = 0;
        sessionState_.lastStatusUpdate = std::chrono::steady_clock::now();
        
        // Detect state changes
        if (newState != sessionState_.currentState) {
            GRIMText::Control::TrainingState oldState = sessionState_.currentState;
            sessionState_.currentState = newState;
            
            if (stateChangeCallback_) {
                stateChangeCallback_(oldState, newState);
            }
        }
        
        // Update stats
        sessionState_.stats = newStats;
        
        // Notify progress callback
        if (progressCallback_) {
            progressCallback_(newStats);
        }
        
        // Update active state based on training state
        sessionState_.isActive = (newState == GRIMText::Control::TrainingState_Training ||
                                  newState == GRIMText::Control::TrainingState_Collecting ||
                                  newState == GRIMText::Control::TrainingState_Verifying);
        
        return true;
    }
    
    // ============================================================
    // Configuration Management
    // ============================================================
    
    // Update training configuration
    bool updateConfig(const GRIMText::TrainingConfig& config) {
        std::lock_guard<std::mutex> lock(stateMutex_);
        
        if (sessionState_.isActive) {
            lastError_ = "Cannot update configuration while training is active";
            return false;
        }
        
        if (!client_.updateConfig(config)) {
            lastError_ = "Failed to update config: " + client_.getLastError();
            return false;
        }
        
        sessionState_.config = config;
        return true;
    }
    
    // ============================================================
    // Data Collection Operations
    // ============================================================
    
    // Start data collection pipeline
    bool startDataCollection(const std::string& mode = "full") {
        std::lock_guard<std::mutex> lock(stateMutex_);
        
        auto result = client_.startDataCollection(mode);
        
        if (!result.success) {
            lastError_ = "Data collection failed: " + result.error;
            if (errorCallback_) {
                errorCallback_(lastError_);
            }
            return false;
        }
        
        return true;
    }
    
    // ============================================================
    // Checkpoint Operations
    // ============================================================
    
    // Detect available checkpoints
    GRIMText::TrainingControlClient::CheckpointDetectResult detectCheckpoints(
        const std::string& checkpointDir = "data") {
        return client_.detectCheckpoints(checkpointDir);
    }
    
    // Merge checkpoint files to training data
    GRIMText::TrainingControlClient::CheckpointMergeResult mergeCheckpoints(
        const std::string& checkpointDir = "data",
        const std::string& verifiedDir = "data/verified",
        const std::string& outputDir = "data",
        bool skipVerification = false) {
        return client_.mergeCheckpoints(checkpointDir, verifiedDir, outputDir, skipVerification);
    }
    
    // ============================================================
    // Callbacks Registration
    // ============================================================
    
    void setProgressCallback(ProgressCallback callback) {
        progressCallback_ = callback;
    }
    
    void setStateChangeCallback(StateChangeCallback callback) {
        stateChangeCallback_ = callback;
    }
    
    void setErrorCallback(ErrorCallback callback) {
        errorCallback_ = callback;
    }
    
    // ============================================================
    // State Accessors
    // ============================================================
    
    const TrainingSessionState& getSessionState() const {
        return sessionState_;
    }
    
    bool isTrainingActive() const {
        return sessionState_.isActive;
    }
    
    bool isTrainingPaused() const {
        return sessionState_.isPaused;
    }
    
    GRIMText::Control::TrainingState getCurrentState() const {
        return sessionState_.currentState;
    }
    
    const GRIMText::TrainingStats& getCurrentStats() const {
        return sessionState_.stats;
    }
    
    const GRIMText::TrainingConfig& getCurrentConfig() const {
        return sessionState_.config;
    }
    
    std::string getLastError() const {
        return lastError_;
    }
    
    // ============================================================
    // Utility Methods
    // ============================================================
    
    // Get human-readable state name
    static std::string getStateName(GRIMText::Control::TrainingState state) {
        switch (state) {
            case GRIMText::Control::TrainingState_Idle: return "Idle";
            case GRIMText::Control::TrainingState_Collecting: return "Collecting Data";
            case GRIMText::Control::TrainingState_Verifying: return "Verifying Data";
            case GRIMText::Control::TrainingState_Training: return "Training";
            case GRIMText::Control::TrainingState_Paused: return "Paused";
            case GRIMText::Control::TrainingState_Completed: return "Completed";
            case GRIMText::Control::TrainingState_Error: return "Error";
            default: return "Unknown";
        }
    }
    
    // Calculate estimated time remaining
    std::chrono::seconds getEstimatedTimeRemaining() const {
        if (!sessionState_.isActive || sessionState_.stats.trainingProgress <= 0.0f) {
            return std::chrono::seconds(0);
        }
        
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - sessionState_.sessionStartTime
        );
        
        float progressRatio = sessionState_.stats.trainingProgress;
        if (progressRatio >= 0.99f) {
            return std::chrono::seconds(0);
        }
        
        auto totalEstimated = elapsed.count() / progressRatio;
        auto remaining = totalEstimated - elapsed.count();
        
        return std::chrono::seconds(static_cast<int64_t>(std::max(0.0f, remaining)));
    }

private:
    GRIMText::TrainingControlClient client_;
    TrainingSessionState sessionState_;
    
    std::string serverHost_;
    int serverPort_;
    std::string lastError_;
    
#ifdef _WIN32
    HANDLE serverProcessHandle_;
#else
    void* serverProcessHandle_;
#endif
    
    // Callbacks
    ProgressCallback progressCallback_;
    StateChangeCallback stateChangeCallback_;
    ErrorCallback errorCallback_;
    
    // Thread safety
    mutable std::mutex stateMutex_;
};

} // namespace UI
} // namespace GRIM
