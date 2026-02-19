//======================================================//
//  GRIM-text Training Control Server
//  HTTP server for controlling training operations
//  
//  Allows external applications (like GRIM.exe UI) to:
//  - Start/stop/pause training
//  - Query training status
//  - Update training parameters
//  - Monitor progress in real-time
//  
//  Communication: HTTP + FlatBuffers on localhost:11436
//  No linking required - pure HTTP communication
//  
//  Author: GRIM Development Team
//  Date: November 6, 2025
//======================================================//

// Windows socket includes MUST come first and in correct order
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#endif

#include <httplib.h>
#include <flatbuffers/flatbuffers.h>
#include "training_control_generated.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <cstdlib>
#include <sstream>
#include <nlohmann/json.hpp>
#include "training_paths.hpp"
#include "ai_config_paths.hpp"

#ifdef _WIN32
#include <windows.h>
#include <processthreadsapi.h>
#endif

namespace fs = std::filesystem;
using namespace GRIMText::Control;

//======================================================//
//  Training State Management
//======================================================//

struct InternalTrainingStats {
    int currentEpoch = 0;
    int totalEpochs = 0;
    int currentBatch = 0;
    int totalBatches = 0;
    float currentLoss = 0.0f;
    float avgLoss = 0.0f;
    float perplexity = 0.0f;
    float tokensPerSec = 0.0f;
    float gpuMemoryUsed = 0.0f;
    float gpuMemoryTotal = 0.0f;
    float trainingProgress = 0.0f;
    float collectionProgress = 0.0f;
    std::string currentPhase = "Idle";
    std::string lastError = "";
    int64_t startTime = 0;
    int64_t elapsedTime = 0;
};

struct InternalTrainingConfig {
    int epochs = 3;
    int batchSize = 8;
    float learningRate = 0.0001f;
    int maxSeqLen = 8192;
    int warmupSteps = 1000;
    bool useGPU = true;
    bool useFlashAttention = true;
    std::string dataPath = "data/training_data.grmt";
    std::string vocabPath = "models/vocab.bin";
    std::string outputPath = "models/grim_text_trained.bin";
};

//======================================================//
//  Global State (Thread-Safe)
//======================================================//

class ControllerState {
public:
    std::atomic<TrainingState> state{TrainingState_Idle};
    std::mutex statsMutex;
    InternalTrainingStats stats;
    InternalTrainingConfig config;
    
    // Stuck state detection
    std::chrono::steady_clock::time_point lastStateChange;
    std::chrono::seconds stuckStateTimeout{300};  // 5 minutes
    
#ifdef _WIN32
    HANDLE trainingProcess = nullptr;
    DWORD trainingPID = 0;
#endif
    
    ControllerState() {
        lastStateChange = std::chrono::steady_clock::now();
    }
    
    void setState(TrainingState newState) {
        state = newState;
        lastStateChange = std::chrono::steady_clock::now();
    }
    
    bool isStateStuckForTooLong() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastStateChange);
        return elapsed > stuckStateTimeout;
    }
    
    void updateStats(const InternalTrainingStats& newStats) {
        std::lock_guard<std::mutex> lock(statsMutex);
        stats = newStats;
    }
    
    InternalTrainingStats getStats() {
        std::lock_guard<std::mutex> lock(statsMutex);
        return stats;
    }
    
    void updateConfig(const InternalTrainingConfig& newConfig) {
        std::lock_guard<std::mutex> lock(statsMutex);
        config = newConfig;
    }
    
    InternalTrainingConfig getConfig() {
        std::lock_guard<std::mutex> lock(statsMutex);
        return config;
    }
};

ControllerState g_state;

//======================================================//
//  Status File Monitor
//======================================================//

class StatusFileMonitor {
public:
    StatusFileMonitor(const std::string& statusFile) 
        : statusFile_(statusFile), running_(false) {}
    
    void start() {
        running_ = true;
        monitorThread_ = std::thread([this]() { monitorLoop(); });
    }
    
    void stop() {
        running_ = false;
        if (monitorThread_.joinable()) {
            monitorThread_.join();
        }
    }
    
private:
    void monitorLoop() {
        while (running_) {
            try {
                if (fs::exists(statusFile_)) {
                    std::ifstream file(statusFile_, std::ios::binary);
                    if (file.is_open()) {
                        // Read FlatBuffer from file
                        file.seekg(0, std::ios::end);
                        size_t size = file.tellg();
                        file.seekg(0, std::ios::beg);
                        
                        std::vector<uint8_t> buffer(size);
                        file.read(reinterpret_cast<char*>(buffer.data()), size);
                        file.close();
                        
                        // Parse FlatBuffer
                        auto statusResponse = GetStatusResponse(buffer.data());
                        
                        // Update global stats from FlatBuffer
                        InternalTrainingStats stats;
                        if (statusResponse->stats()) {
                            auto fbStats = statusResponse->stats();
                            stats.currentEpoch = fbStats->current_epoch();
                            stats.totalEpochs = fbStats->total_epochs();
                            stats.currentBatch = fbStats->current_batch();
                            stats.totalBatches = fbStats->total_batches();
                            stats.currentLoss = fbStats->current_loss();
                            stats.avgLoss = fbStats->avg_loss();
                            stats.perplexity = fbStats->perplexity();
                            stats.tokensPerSec = fbStats->tokens_per_sec();
                            stats.gpuMemoryUsed = fbStats->gpu_memory_used();
                            stats.gpuMemoryTotal = fbStats->gpu_memory_total();
                            stats.trainingProgress = fbStats->training_progress();
                            stats.currentPhase = fbStats->current_phase() ? fbStats->current_phase()->str() : "";
                            stats.lastError = fbStats->last_error() ? fbStats->last_error()->str() : "";
                            stats.startTime = fbStats->start_time();
                            stats.elapsedTime = fbStats->elapsed_time();
                        }
                        
                        g_state.updateStats(stats);
                        
                        // Update state based on FlatBuffer state
                        g_state.setState(statusResponse->state());
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "[Monitor] Error reading status file: " << e.what() << std::endl;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
    
    std::string statusFile_;
    std::atomic<bool> running_;
    std::thread monitorThread_;
};

//======================================================//
//  Training Process Control
//======================================================//

class TrainingProcessController {
public:
    bool startTraining(const InternalTrainingConfig& config) {
#ifdef _WIN32
        // Open debug log file
        std::ofstream debugLog("training_control_debug.log", std::ios::app);
        auto now = std::chrono::system_clock::now();
        auto timeT = std::chrono::system_clock::to_time_t(now);
        std::tm timeTm;
        localtime_s(&timeTm, &timeT);
        char timeBuf[64];
        std::strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%d %H:%M:%S", &timeTm);
        
        debugLog << "\n===== START TRAINING ATTEMPT =====" << std::endl;
        debugLog << "Time: " << timeBuf << std::endl;
        
        // Kill existing process if running
        stopTraining();
        debugLog << "Stopped existing process (if any)" << std::endl;
        
        // Use safe path resolution to find train_gpu.exe
        std::string exePathStr = GRIM::Training::getSafeResourcePath(
            "resources/models/GRIM-text/training/TrainingLoop/build/Release/train_gpu.exe",
            GRIM::Training::PathResolutionMode::Relative
        );
        
        debugLog << "First getSafeResourcePath returned: '" << exePathStr << "'" << std::endl;
        
        if (exePathStr.empty()) {
            // Try searching for it
            debugLog << "First path resolution failed, searching..." << std::endl;
            exePathStr = GRIM::Training::getSafeResourcePath(
                "train_gpu.exe",
                GRIM::Training::PathResolutionMode::Search
            );
            debugLog << "Search getSafeResourcePath returned: '" << exePathStr << "'" << std::endl;
        }
        
        if (exePathStr.empty()) {
            debugLog << "ERROR: train_gpu.exe not found!" << std::endl;
            debugLog.close();
            std::cerr << "[Controller] ERROR: train_gpu.exe not found in GRIM directory tree" << std::endl;
            std::cerr << "[Controller] Please ensure the training executable is built" << std::endl;
            return false;
        }
        
        fs::path exePath(exePathStr);
        
        debugLog << "exePath: " << exePath << std::endl;
        debugLog << "exePath.parent_path(): " << exePath.parent_path() << std::endl;
        
        // Get GRIM root for resolving other paths
        fs::path grimRoot;
#ifdef GRIM_ROOT_DIR
        grimRoot = fs::path(GRIM_ROOT_DIR);
        debugLog << "Using GRIM_ROOT_DIR: " << grimRoot << std::endl;
#else
        // getSafeResourcePath returns paths starting from the canonical resource root
        // So we need to find where that resource root is and go up one level
        grimRoot = exePath.parent_path();
        debugLog << "Starting grimRoot walk from: " << grimRoot << std::endl;
        
        for (int i = 0; i < 10; ++i) {
            debugLog << "  Checking iteration " << i << ": " << grimRoot << std::endl;
            debugLog << "    Has 'control'? " << fs::exists(grimRoot / "control") << std::endl;
            debugLog << "    Has 'resources'? " << fs::exists(grimRoot / "resources") << std::endl;
            
            // We want to find the folder that CONTAINS both "control" AND "resources"
            if (fs::exists(grimRoot / "control") && fs::exists(grimRoot / "resources")) {
                debugLog << "  Found GRIM root!" << std::endl;
                break;
            }
            if (!grimRoot.has_parent_path()) {
                debugLog << "  No more parents, using current_path()" << std::endl;
                grimRoot = fs::current_path();
                break;
            }
            grimRoot = grimRoot.parent_path();
        }
        debugLog << "Final grimRoot: " << grimRoot << std::endl;
#endif
        
        // Set working directory to GRIM root so train_gpu.exe can find ai_config.json
        fs::path workingDir = fs::absolute(grimRoot);
        
        // Use paths from config (which are now loaded from ai_config.json and resolved to absolute paths)
        // The paths should already be absolute after populateGrimTextPathsFromConfig() resolves them
        // But if they're still relative (backward compatibility), resolve them from GRIM root
        fs::path dataPath = config.dataPath;
        fs::path vocabPath = config.vocabPath;
        fs::path outputPath = config.outputPath;
        
        // If paths are relative, resolve them from GRIM root (shouldn't happen with new code, but handle for safety)
        if (dataPath.is_relative()) {
            dataPath = grimRoot / dataPath;
        }
        if (vocabPath.is_relative()) {
            vocabPath = grimRoot / vocabPath;
        }
        if (outputPath.is_relative()) {
            outputPath = grimRoot / outputPath;
        }
        
        debugLog << "GRIM root: " << grimRoot << std::endl;
        debugLog << "train_gpu.exe: " << exePath << std::endl;
        debugLog << "Working dir: " << workingDir << std::endl;
        debugLog << "Data path: " << dataPath << std::endl;
        debugLog << "Vocab path: " << vocabPath << std::endl;
        debugLog << "Output path: " << outputPath << std::endl;
        
        std::cout << "[Controller] GRIM root: " << grimRoot << std::endl;
        std::cout << "[Controller] Found train_gpu.exe at: " << exePath << std::endl;
        std::cout << "[Controller] Data path: " << dataPath << std::endl;
        std::cout << "[Controller] Vocab path: " << vocabPath << std::endl;
        std::cout << "[Controller] Output path: " << outputPath << std::endl;
        
        // Verify paths exist
        if (!fs::exists(vocabPath)) {
            debugLog << "ERROR: Vocab file not found at: " << vocabPath << std::endl;
            debugLog.close();
            std::cerr << "[Controller] ERROR: Vocab file not found at: " << vocabPath << std::endl;
            std::cerr << "[Controller] Expected: " << vocabPath << std::endl;
            return false;
        }
        if (!fs::exists(dataPath)) {
            debugLog << "ERROR: Data file not found at: " << dataPath << std::endl;
            debugLog.close();
            std::cerr << "[Controller] ERROR: Data file not found at: " << dataPath << std::endl;
            std::cerr << "[Controller] Expected: " << dataPath << std::endl;
            return false;
        }
        
        debugLog << "Path validation passed" << std::endl;
        
        std::ostringstream cmdLine;
        cmdLine << "\"" << exePath.string() << "\""
                << " --data \"" << dataPath.string() << "\""
                << " --vocab \"" << vocabPath.string() << "\""
                << " --output \"" << outputPath.string() << "\""
                << " --epochs " << config.epochs
                << " --batch-size " << config.batchSize
                << " --lr " << config.learningRate
                << " --max-seq-len " << config.maxSeqLen
                << " --warmup-steps " << config.warmupSteps;
        
        if (!config.useGPU) {
            cmdLine << " --cpu";
        }
        
        debugLog << "Command line length: " << cmdLine.str().length() << " chars" << std::endl;
        debugLog << "Full command: " << cmdLine.str() << std::endl;
        
        std::cout << "[Controller] Starting training: " << cmdLine.str() << std::endl;
        
        // Create process
        STARTUPINFOA si = {sizeof(si)};
        PROCESS_INFORMATION pi = {0};
        
        std::string cmd = cmdLine.str();
        debugLog << "About to call CreateProcessA..." << std::endl;
        
        if (!CreateProcessA(
            nullptr,
            const_cast<char*>(cmd.c_str()),
            nullptr,
            nullptr,
            FALSE,
            CREATE_NO_WINDOW,
            nullptr,
            workingDir.string().c_str(),  // Set working directory
            &si,
            &pi
        )) {
            DWORD errorCode = GetLastError();
            
            debugLog << "CreateProcessA FAILED!" << std::endl;
            debugLog << "Windows Error Code: " << errorCode << std::endl;
            debugLog.close();
            
            // Log to console (may be lost if CREATE_NO_WINDOW)
            std::cout << "[Controller] *** FAILED TO START TRAINING PROCESS ***" << std::endl;
            std::cout << "[Controller] Windows Error Code: " << errorCode << std::endl;
            std::cout << "[Controller] Working Directory: " << workingDir.string() << std::endl;
            std::cout << "[Controller] Command Line Length: " << cmd.length() << " chars" << std::endl;
            std::cout << "[Controller] Command: " << cmd << std::endl;
            std::cerr << "[Controller] Failed to start process: " << errorCode << std::endl;
            
            // Also write to file for debugging
            std::ofstream errorLog("training_control_error.log", std::ios::app);
            if (errorLog.is_open()) {
                auto now = std::chrono::system_clock::now();
                auto timeT = std::chrono::system_clock::to_time_t(now);
                std::tm timeTm;
                localtime_s(&timeTm, &timeT);
                char timeBuf[64];
                std::strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%d %H:%M:%S", &timeTm);
                
                errorLog << "===== TRAINING START FAILURE =====" << std::endl;
                errorLog << "Time: " << timeBuf << std::endl;
                errorLog << "Windows Error Code: " << errorCode << std::endl;
                errorLog << "Working Directory: " << workingDir.string() << std::endl;
                errorLog << "Command Length: " << cmd.length() << " chars" << std::endl;
                errorLog << "Full Command: " << cmd << std::endl;
                errorLog << "===================================\n" << std::endl;
                errorLog.close();
            }
            
            return false;
        }
        
        debugLog << "CreateProcessA SUCCESS!" << std::endl;
        debugLog << "Process ID: " << pi.dwProcessId << std::endl;
        debugLog << "===================================\n" << std::endl;
        debugLog.close();
        
        g_state.trainingProcess = pi.hProcess;
        g_state.trainingPID = pi.dwProcessId;
        CloseHandle(pi.hThread);
        
        g_state.setState(TrainingState_Training);
        std::cout << "[Controller] Training started (PID: " << pi.dwProcessId << ")" << std::endl;
        return true;
#else
        std::cerr << "[Controller] Training control not implemented for this platform" << std::endl;
        return false;
#endif
    }
    
    bool stopTraining() {
#ifdef _WIN32
        if (g_state.trainingProcess != nullptr) {
            std::cout << "[Controller] Stopping training (PID: " << g_state.trainingPID << ")" << std::endl;
            TerminateProcess(g_state.trainingProcess, 0);
            CloseHandle(g_state.trainingProcess);
            g_state.trainingProcess = nullptr;
            g_state.trainingPID = 0;
            g_state.setState(TrainingState_Idle);
            return true;
        }
#endif
        return false;
    }
    
    bool isTrainingRunning() {
#ifdef _WIN32
        if (g_state.trainingProcess != nullptr) {
            DWORD exitCode;
            if (GetExitCodeProcess(g_state.trainingProcess, &exitCode)) {
                if (exitCode == STILL_ACTIVE) {
                    return true;
                } else {
                    // Process has exited - check exit code to determine success vs error
                    CloseHandle(g_state.trainingProcess);
                    g_state.trainingProcess = nullptr;
                    g_state.trainingPID = 0;
                    
                    // Only mark as completed if exit code is 0 (success)
                    // Otherwise it's an error or premature exit
                    if (exitCode == 0) {
                        g_state.setState(TrainingState_Completed);
                    } else {
                        g_state.setState(TrainingState_Error);
                        std::ostringstream errorMsg;
                        errorMsg << "Training process exited with code " << exitCode;
                        g_state.stats.lastError = errorMsg.str();
                    }
                }
            }
        }
#endif
        return false;
    }
};

TrainingProcessController g_processController;

//======================================================//
//  HTTP API Handlers
//======================================================//

void setupAPI(httplib::Server& server) {
    // Health check
    server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        flatbuffers::FlatBufferBuilder builder(256);
        
        auto status = builder.CreateString("ok");
        auto service = builder.CreateString("grim-text-training-control");
        auto version = builder.CreateString("1.0.0");
        
        auto healthCheck = CreateHealthCheckResponse(builder, status, service, version);
        builder.Finish(healthCheck);
        
        res.set_content(reinterpret_cast<const char*>(builder.GetBufferPointer()), 
                       builder.GetSize(), 
                       "application/octet-stream");
    });
    
    // Get current status
    server.Get("/api/status", [](const httplib::Request&, httplib::Response& res) {
        flatbuffers::FlatBufferBuilder builder(1024);
        
        // Smart guard: Auto-cleanup stuck states on every status check
        bool processActuallyRunning = g_processController.isTrainingRunning();
        auto currentState = g_state.state.load();
        
        // Guard 1: Process died but state wasn't updated
        if ((currentState == TrainingState_Training || currentState == TrainingState_Collecting) 
            && !processActuallyRunning) {
            std::cout << "[Controller] Auto-cleanup: Process died (state=" << currentState 
                     << " but no process running)" << std::endl;
            g_state.setState(TrainingState_Idle);
            
            InternalTrainingStats stats = g_state.getStats();
            if (stats.lastError.empty()) {
                stats.lastError = "Process terminated unexpectedly";
            }
            stats.currentPhase = "Ready";
            g_state.updateStats(stats);
        }
        // Guard 2: State stuck for too long (5+ minutes without change)
        else if (currentState != TrainingState_Idle && g_state.isStateStuckForTooLong()) {
            std::cout << "[Controller] Auto-cleanup: State stuck for >5 minutes (state=" << currentState << ")" << std::endl;
            g_state.setState(TrainingState_Idle);
            
            InternalTrainingStats stats = g_state.getStats();
            stats.lastError = "Training timed out or became unresponsive";
            stats.currentPhase = "Ready";
            g_state.updateStats(stats);
        }
        
        auto stats = g_state.getStats();
        auto config = g_state.getConfig();
        
        // Build TrainingStats
        auto currentPhase = builder.CreateString(stats.currentPhase);
        auto lastError = builder.CreateString(stats.lastError);
        
        auto fbStats = CreateTrainingStats(builder,
            stats.currentEpoch,
            stats.totalEpochs,
            stats.currentBatch,
            stats.totalBatches,
            stats.currentLoss,
            stats.avgLoss,
            stats.perplexity,
            stats.tokensPerSec,
            stats.gpuMemoryUsed,
            stats.gpuMemoryTotal,
            stats.trainingProgress,
            stats.collectionProgress,
            currentPhase,
            lastError,
            stats.startTime,
            stats.elapsedTime
        );
        
        // Build TrainingConfig
        auto dataPath = builder.CreateString(config.dataPath);
        auto vocabPath = builder.CreateString(config.vocabPath);
        auto outputPath = builder.CreateString(config.outputPath);
        
        auto fbConfig = CreateTrainingConfig(builder,
            config.epochs,
            config.batchSize,
            config.learningRate,
            config.maxSeqLen,
            config.warmupSteps,
            config.useGPU,
            config.useFlashAttention,
            dataPath,
            vocabPath,
            outputPath
        );
        
        // Build StatusResponse
        auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();
        auto statusResponse = CreateStatusResponse(builder,
            g_state.state.load(),
            fbStats,
            fbConfig,
            g_processController.isTrainingRunning(),
            timestamp
        );
        
        builder.Finish(statusResponse);
        
        res.set_content(reinterpret_cast<const char*>(builder.GetBufferPointer()), 
                       builder.GetSize(), 
                       "application/octet-stream");
    });
    
    // Start training
    server.Post("/api/training/start", [](const httplib::Request& req, httplib::Response& res) {
        flatbuffers::FlatBufferBuilder builder(256);
        
        // Smart guard: Check if state is stuck (process dead but state not idle)
        bool processActuallyRunning = g_processController.isTrainingRunning();
        bool stateClaimsRunning = (g_state.state != TrainingState_Idle && g_state.state != TrainingState_Completed);
        
        if (stateClaimsRunning && !processActuallyRunning) {
            // Stuck state detected: process is dead but state wasn't cleared
            std::cout << "[Controller] WARNING: Stuck state detected (state=" << g_state.state 
                     << " but no process running)" << std::endl;
            std::cout << "[Controller] Auto-clearing stuck state..." << std::endl;
            
            g_state.setState(TrainingState_Idle);
            InternalTrainingStats stats = g_state.getStats();
            stats.currentPhase = "Ready";
            stats.lastError = "";
            g_state.updateStats(stats);
            
            stateClaimsRunning = false;  // Allow training to proceed
        }
        
        if (stateClaimsRunning) {
            auto error = builder.CreateString("Training already in progress");
            auto message = builder.CreateString("");
            auto response = CreateStartTrainingResponse(builder, false, message, error);
            builder.Finish(response);
            
            res.status = 400;
            res.set_content(reinterpret_cast<const char*>(builder.GetBufferPointer()), 
                           builder.GetSize(), 
                           "application/octet-stream");
            return;
        }
        
        // Parse optional config updates from FlatBuffer
        auto config = g_state.getConfig();
        if (!req.body.empty()) {
            try {
                auto request = flatbuffers::GetRoot<StartTrainingRequest>(req.body.data());
                if (request->config()) {
                    auto fbConfig = request->config();
                    config.epochs = fbConfig->epochs();
                    config.batchSize = fbConfig->batch_size();
                    config.learningRate = fbConfig->learning_rate();
                    config.maxSeqLen = fbConfig->max_seq_len();
                    config.useGPU = fbConfig->use_gpu();
                    // Also read path config if provided
                    if (fbConfig->data_path() && fbConfig->data_path()->size() > 0) {
                        config.dataPath = fbConfig->data_path()->str();
                    }
                    if (fbConfig->vocab_path() && fbConfig->vocab_path()->size() > 0) {
                        config.vocabPath = fbConfig->vocab_path()->str();
                    }
                    if (fbConfig->output_path() && fbConfig->output_path()->size() > 0) {
                        config.outputPath = fbConfig->output_path()->str();
                    }
                    g_state.updateConfig(config);
                }
            } catch (const std::exception& e) {
                auto error = builder.CreateString(std::string("Invalid FlatBuffer: ") + e.what());
                auto message = builder.CreateString("");
                auto response = CreateStartTrainingResponse(builder, false, message, error);
                builder.Finish(response);
                
                res.status = 400;
                res.set_content(reinterpret_cast<const char*>(builder.GetBufferPointer()), 
                               builder.GetSize(), 
                               "application/octet-stream");
                return;
            }
        }
        
        if (g_processController.startTraining(config)) {
            auto message = builder.CreateString("Training started");
            auto error = builder.CreateString("");
            auto response = CreateStartTrainingResponse(builder, true, message, error);
            builder.Finish(response);
            
            res.set_content(reinterpret_cast<const char*>(builder.GetBufferPointer()), 
                           builder.GetSize(), 
                           "application/octet-stream");
        } else {
            auto error = builder.CreateString("Failed to start training process");
            auto message = builder.CreateString("");
            auto response = CreateStartTrainingResponse(builder, false, message, error);
            builder.Finish(response);
            
            res.status = 500;
            res.set_content(reinterpret_cast<const char*>(builder.GetBufferPointer()), 
                           builder.GetSize(), 
                           "application/octet-stream");
        }
    });
    
    // Stop training
    server.Post("/api/training/stop", [](const httplib::Request&, httplib::Response& res) {
        flatbuffers::FlatBufferBuilder builder(256);
        
        if (g_processController.stopTraining()) {
            auto message = builder.CreateString("Training stopped");
            auto response = CreateStopTrainingResponse(builder, true, message);
            builder.Finish(response);
            
            res.set_content(reinterpret_cast<const char*>(builder.GetBufferPointer()), 
                           builder.GetSize(), 
                           "application/octet-stream");
        } else {
            auto message = builder.CreateString("No training process running");
            auto response = CreateStopTrainingResponse(builder, false, message);
            builder.Finish(response);
            
            res.status = 400;
            res.set_content(reinterpret_cast<const char*>(builder.GetBufferPointer()), 
                           builder.GetSize(), 
                           "application/octet-stream");
        }
    });
    
    // Update configuration
    server.Post("/api/config", [](const httplib::Request& req, httplib::Response& res) {
        flatbuffers::FlatBufferBuilder builder(512);
        
        if (g_state.state == TrainingState_Training) {
            auto error = builder.CreateString("Cannot update config while training");
            auto message = builder.CreateString("");
            auto response = CreateUpdateConfigResponse(builder, false, message, error);
            builder.Finish(response);
            
            res.status = 400;
            res.set_content(reinterpret_cast<const char*>(builder.GetBufferPointer()), 
                           builder.GetSize(), 
                           "application/octet-stream");
            return;
        }
        
        try {
            auto request = flatbuffers::GetRoot<UpdateConfigRequest>(req.body.data());
            auto config = g_state.getConfig();
            
            if (request->config()) {
                auto fbConfig = request->config();
                config.epochs = fbConfig->epochs();
                config.batchSize = fbConfig->batch_size();
                config.learningRate = fbConfig->learning_rate();
                config.maxSeqLen = fbConfig->max_seq_len();
                config.warmupSteps = fbConfig->warmup_steps();
                config.useGPU = fbConfig->use_gpu();
                config.useFlashAttention = fbConfig->use_flash_attention();
                if (fbConfig->data_path()) config.dataPath = fbConfig->data_path()->str();
                if (fbConfig->vocab_path()) config.vocabPath = fbConfig->vocab_path()->str();
                if (fbConfig->output_path()) config.outputPath = fbConfig->output_path()->str();
            }
            
            g_state.updateConfig(config);
            
            auto message = builder.CreateString("Configuration updated");
            auto error = builder.CreateString("");
            auto response = CreateUpdateConfigResponse(builder, true, message, error);
            builder.Finish(response);
            
            res.set_content(reinterpret_cast<const char*>(builder.GetBufferPointer()), 
                           builder.GetSize(), 
                           "application/octet-stream");
        } catch (const std::exception& e) {
            auto error = builder.CreateString(std::string("Invalid FlatBuffer: ") + e.what());
            auto message = builder.CreateString("");
            auto response = CreateUpdateConfigResponse(builder, false, message, error);
            builder.Finish(response);
            
            res.status = 400;
            res.set_content(reinterpret_cast<const char*>(builder.GetBufferPointer()), 
                           builder.GetSize(), 
                           "application/octet-stream");
        }
    });
    
    // Shutdown server
    server.Post("/api/server/shutdown", [&server](const httplib::Request&, httplib::Response& res) {
        flatbuffers::FlatBufferBuilder builder(256);
        
        std::cout << "[Server] Shutdown request received" << std::endl;
        
        // Stop any running training first
        if (g_processController.stopTraining()) {
            std::cout << "[Server] Training process stopped" << std::endl;
        }
        
        // Send success response before shutting down
        auto message = builder.CreateString("Server shutting down");
        auto response = CreateServerShutdownResponse(builder, true, message);
        builder.Finish(response);
        
        res.set_content(reinterpret_cast<const char*>(builder.GetBufferPointer()), 
                       builder.GetSize(), 
                       "application/octet-stream");
        
        // Stop server in separate thread to allow response to be sent
        std::thread([&server]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::cout << "[Server] Stopping HTTP server..." << std::endl;
            server.stop();
        }).detach();
    });
    
}

//======================================================//
//  Main
//======================================================//

int main(int argc, char** argv) {
    // Initialize Windows Sockets
#ifdef _WIN32
    WSADATA wsaData;
    int wsaResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (wsaResult != 0) {
        std::cerr << "[ERROR] WSAStartup failed: " << wsaResult << std::endl;
        return 1;
    }
    std::cout << "[Server] Windows Sockets initialized" << std::endl;
#endif

    int port = 11436;  // Different from grim_text_server (11435)
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "GRIM-text Training Control Server" << std::endl;
            std::cout << "Usage: training_control_server [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --port <port>    Server port (default: 11436)" << std::endl;
            std::cout << "  --help           Show this help" << std::endl;
#ifdef _WIN32
            WSACleanup();
#endif
            return 0;
        }
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "  GRIM-text Training Control Server" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Resolve status file path from canonical resource root
    fs::path statusFilePath = GRIM::Training::getTrainingStatusFilePath();
    std::cout << "[Server] Monitoring status file: " << statusFilePath << std::endl;
    
    // ✅ FIX: Delete stale status file on server startup to prevent false "training in progress" state
    if (fs::exists(statusFilePath)) {
        try {
            fs::remove(statusFilePath);
            std::cout << "[Server] Deleted stale status file from previous session" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Server] WARNING: Failed to delete stale status file: " << e.what() << std::endl;
        }
    }
    
    // Ensure state starts clean
    g_state.setState(TrainingState_Idle);
    InternalTrainingStats cleanStats = {};
    cleanStats.currentPhase = "Ready";
    cleanStats.lastError = "";
    g_state.updateStats(cleanStats);
    std::cout << "[Server] State initialized to Idle" << std::endl;
    
    // Load paths from ai_config.json (the centralized source of truth)
    GRIM::Config::GrimTextPaths grimPaths;
    if (GRIM::Config::loadGrimTextPaths(grimPaths)) {
        // Update config with loaded paths
        InternalTrainingConfig loadedConfig = g_state.getConfig();
        
        if (!grimPaths.training_data.empty()) {
            loadedConfig.dataPath = grimPaths.training_data;
            std::cout << "[Server] ✓ Loaded training_data path from ai_config.json: " << grimPaths.training_data << std::endl;
        }
        
        if (!grimPaths.vocab.empty()) {
            loadedConfig.vocabPath = grimPaths.vocab;
            std::cout << "[Server] ✓ Loaded vocab path from ai_config.json: " << grimPaths.vocab << std::endl;
        }
        
        if (!grimPaths.model.empty()) {
            loadedConfig.outputPath = grimPaths.model;
            std::cout << "[Server] ✓ Loaded model path from ai_config.json: " << grimPaths.model << std::endl;
        }
        
        g_state.updateConfig(loadedConfig);
    } else {
        std::cout << "[Server] WARNING: Could not load paths from ai_config.json, using defaults" << std::endl;
    }
    
    // Also load training hyperparameters
    GRIM::Config::TrainingHyperparameters hyperparams;
    if (GRIM::Config::loadTrainingHyperparameters(hyperparams)) {
        InternalTrainingConfig config = g_state.getConfig();
        config.epochs = hyperparams.epochs;
        config.batchSize = hyperparams.batch_size;
        config.learningRate = hyperparams.learning_rate;
        config.maxSeqLen = hyperparams.max_seq_len;
        config.warmupSteps = hyperparams.warmup_steps;
        config.useGPU = hyperparams.use_gpu;
        config.useFlashAttention = hyperparams.use_flash_attention;
        g_state.updateConfig(config);
        std::cout << "[Server] ✓ Loaded training hyperparameters from ai_config.json" << std::endl;
    }
    
    // Start status file monitor
    StatusFileMonitor monitor(statusFilePath.string());
    std::cout << "[Server] Starting status monitor..." << std::endl;
    std::cout.flush();
    monitor.start();
    std::cout << "[Server] Monitor started" << std::endl;
    std::cout.flush();
    
    // Create HTTP server
    httplib::Server server;
    std::cout << "[Server] HTTP server object created" << std::endl;
    std::cout.flush();
    setupAPI(server);
    std::cout << "[Server] API routes configured" << std::endl;
    std::cout.flush();
    
    std::cout << "[Server] Starting on http://127.0.0.1:" << port << std::endl;
    std::cout << std::endl;
    std::cout << "API Endpoints:" << std::endl;
    std::cout << "  GET  /health                - Health check" << std::endl;
    std::cout << "  GET  /api/status            - Get training status" << std::endl;
    std::cout << "  POST /api/training/start    - Start training" << std::endl;
    std::cout << "  POST /api/training/stop     - Stop training" << std::endl;
    std::cout << "  POST /api/config            - Update configuration" << std::endl;
    std::cout << "  POST /api/server/shutdown   - Shutdown server" << std::endl;
    std::cout << std::endl;
    std::cout << "Press Ctrl+C to stop" << std::endl;
    std::cout << std::endl;
    
    // Run server
    std::cout << "[Server] Binding to port..." << std::endl;
    std::cout.flush();  // Force flush before potentially blocking call
    
    server.set_error_handler([](const httplib::Request&, httplib::Response& res) {
        std::cerr << "[HTTP] Error handler called with status: " << res.status << std::endl;
    });
    
    std::cout << "[Server] Calling listen()..." << std::endl;
    std::cout.flush();
    
    bool result = false;
    try {
        result = server.listen("127.0.0.1", port);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in listen(): " << e.what() << std::endl;
        monitor.stop();
#ifdef _WIN32
        WSACleanup();
#endif
        return 1;
    } catch (...) {
        std::cerr << "[ERROR] Unknown exception in listen()" << std::endl;
        monitor.stop();
#ifdef _WIN32
        WSACleanup();
#endif
        return 1;
    }
    
    if (!result) {
        std::cerr << "[ERROR] Failed to start server on port " << port << std::endl;
        std::cerr << "[ERROR] Port may be in use or permission denied" << std::endl;
        std::cerr << "[ERROR] listen() returned: " << result << std::endl;
        monitor.stop();
#ifdef _WIN32
        WSACleanup();
#endif
        return 1;
    }
    
    std::cout << "[Server] Server stopped" << std::endl;
    monitor.stop();
#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}
