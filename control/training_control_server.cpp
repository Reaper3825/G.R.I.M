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
#include <filesystem>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include "training_paths.hpp"

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
    int totalEpochs = 3;
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
    
#ifdef _WIN32
    HANDLE trainingProcess = nullptr;
    DWORD trainingPID = 0;
#endif
    
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
                        g_state.state = statusResponse->state();
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
        // Kill existing process if running
        stopTraining();
        
        // Use safe path resolution to find train_gpu.exe
        std::string exePathStr = GRIM::Training::getSafeResourcePath(
            "resources/models/GRIM-text/training/TrainingLoop/build/Release/train_gpu.exe",
            GRIM::Training::PathResolutionMode::Relative
        );
        
        if (exePathStr.empty()) {
            // Try searching for it
            exePathStr = GRIM::Training::getSafeResourcePath(
                "train_gpu.exe",
                GRIM::Training::PathResolutionMode::Search
            );
        }
        
        if (exePathStr.empty()) {
            std::cerr << "[Controller] ERROR: train_gpu.exe not found in GRIM directory tree" << std::endl;
            std::cerr << "[Controller] Please ensure the training executable is built" << std::endl;
            return false;
        }
        
        fs::path exePath(exePathStr);
        
        // Get GRIM root for resolving other paths
        fs::path grimRoot;
#ifdef GRIM_ROOT_DIR
        grimRoot = fs::path(GRIM_ROOT_DIR);
#else
        // Walk up from exe path to find GRIM root
        grimRoot = exePath.parent_path();
        for (int i = 0; i < 10; ++i) {
            if (fs::exists(grimRoot / "control") || fs::exists(grimRoot / "resources")) {
                break;
            }
            if (!grimRoot.has_parent_path()) {
                grimRoot = fs::current_path();
                break;
            }
            grimRoot = grimRoot.parent_path();
        }
#endif
        
        fs::path workingDir = grimRoot / "resources/models/GRIM-text/training";
        
        // Convert config paths to absolute paths from GRIM root
        fs::path dataPath = grimRoot / "resources/models/GRIM-text/training" / config.dataPath;
        fs::path vocabPath = grimRoot / "resources/models/GRIM-text/training" / config.vocabPath;
        fs::path outputPath = grimRoot / "resources/models/GRIM-text/training" / config.outputPath;
        
        std::cout << "[Controller] GRIM root: " << grimRoot << std::endl;
        std::cout << "[Controller] Found train_gpu.exe at: " << exePath << std::endl;
        std::cout << "[Controller] Data path: " << dataPath << std::endl;
        std::cout << "[Controller] Vocab path: " << vocabPath << std::endl;
        std::cout << "[Controller] Output path: " << outputPath << std::endl;
        
        // Verify paths exist
        if (!fs::exists(vocabPath)) {
            std::cerr << "[Controller] ERROR: Vocab file not found at: " << vocabPath << std::endl;
            std::cerr << "[Controller] Expected: " << vocabPath << std::endl;
            return false;
        }
        if (!fs::exists(dataPath)) {
            std::cerr << "[Controller] ERROR: Data file not found at: " << dataPath << std::endl;
            std::cerr << "[Controller] Expected: " << dataPath << std::endl;
            return false;
        }
        
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
        
        std::cout << "[Controller] Starting training: " << cmdLine.str() << std::endl;
        
        // Create process
        STARTUPINFOA si = {sizeof(si)};
        PROCESS_INFORMATION pi = {0};
        
        std::string cmd = cmdLine.str();
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
            std::cerr << "[Controller] Failed to start process: " << GetLastError() << std::endl;
            return false;
        }
        
        g_state.trainingProcess = pi.hProcess;
        g_state.trainingPID = pi.dwProcessId;
        CloseHandle(pi.hThread);
        
        g_state.state = TrainingState_Training;
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
            g_state.state = TrainingState_Idle;
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
                    // Process has exited
                    CloseHandle(g_state.trainingProcess);
                    g_state.trainingProcess = nullptr;
                    g_state.trainingPID = 0;
                    g_state.state = TrainingState_Completed;
                }
            }
        }
#endif
        return false;
    }
};

TrainingProcessController g_processController;

//======================================================//
//  Checkpoint Detection & Merge Functions
//======================================================//

struct CheckpointDetectResult {
    bool success = false;
    int checkpointCount = 0;
    int totalEntries = 0;
    int64_t totalSize = 0;
    std::vector<std::tuple<std::string, int64_t, int>> checkpoints; // path, size, entries
    std::string error;
};

CheckpointDetectResult detectCheckpoints(const std::string& checkpoint_dir) {
    CheckpointDetectResult result;
    
    try {
        if (!fs::exists(checkpoint_dir)) {
            result.error = "Checkpoint directory does not exist: " + checkpoint_dir;
            return result;
        }
        
        // Find all checkpoint_*.json files
        for (const auto& entry : fs::directory_iterator(checkpoint_dir)) {
            if (!entry.is_regular_file()) continue;
            
            std::string filename = entry.path().filename().string();
            if (filename.substr(0, 11) == "checkpoint_" && 
                filename.size() > 5 && 
                filename.substr(filename.size() - 5) == ".json") {
                
                // Get file size
                int64_t fileSize = fs::file_size(entry.path());
                
                // Count entries in checkpoint
                int entryCount = 0;
                try {
                    std::ifstream file(entry.path());
                    nlohmann::json j;
                    file >> j;
                    if (j.contains("data") && j["data"].is_array()) {
                        entryCount = j["data"].size();
                    }
                } catch (...) {
                    // Skip malformed checkpoints
                    continue;
                }
                
                result.checkpoints.push_back({entry.path().string(), fileSize, entryCount});
                result.totalSize += fileSize;
                result.totalEntries += entryCount;
            }
        }
        
        result.checkpointCount = result.checkpoints.size();
        result.success = true;
        
    } catch (const std::exception& e) {
        result.error = std::string("Exception during checkpoint detection: ") + e.what();
        result.success = false;
    }
    
    return result;
}

struct CheckpointMergeResult {
    bool success = false;
    int checkpointEntries = 0;
    int verifiedEntries = 0;
    int finalEntries = 0;
    std::string outputGrmtPath;
    std::string error;
    std::string message;
};

CheckpointMergeResult mergeCheckpoints(const std::string& checkpoint_dir, 
                                       const std::string& verified_dir,
                                       const std::string& output_dir,
                                       bool skip_verification) {
    CheckpointMergeResult result;
    
    try {
        // Build command to run merge_checkpoints executable
        // Try multiple possible locations
        std::vector<fs::path> possible_paths = {
            fs::absolute(".") / "resources/models/GRIM-text/training/build_vs_cuda/Release/merge_checkpoints.exe",
            fs::absolute(".") / "resources/models/GRIM-text/training/build_ninja/merge_checkpoints.exe",
            fs::absolute(".") / "resources/models/GRIM-text/training/merge_checkpoints.exe"
        };
        
        fs::path exe_path;
        bool found = false;
        for (const auto& path : possible_paths) {
            if (fs::exists(path)) {
                exe_path = path;
                found = true;
                break;
            }
        }
        
        if (!found) {
            result.error = "merge_checkpoints.exe not found. Tried:\n";
            for (const auto& path : possible_paths) {
                result.error += "  - " + path.string() + "\n";
            }
            return result;
        }
        
        std::cout << "[Checkpoint Merge] Using executable: " << exe_path.string() << std::endl;
        
        std::stringstream cmd;
        cmd << "\"" << exe_path.string() << "\""
            << " --checkpoint-dir \"" << checkpoint_dir << "\""
            << " --verified-dir \"" << verified_dir << "\""
            << " --output-dir \"" << output_dir << "\"";
        
        if (skip_verification) {
            cmd << " --skip-verification";
        }
        
        std::cout << "[Checkpoint Merge] Running: " << cmd.str() << std::endl;
        
        // Execute the merge process
#ifdef _WIN32
        STARTUPINFOA si = {sizeof(si)};
        PROCESS_INFORMATION pi = {0};
        
        std::string cmdStr = cmd.str();
        if (!CreateProcessA(
            nullptr,
            const_cast<char*>(cmdStr.c_str()),
            nullptr,
            nullptr,
            FALSE,
            0, // Show window for feedback
            nullptr,
            nullptr,
            &si,
            &pi
        )) {
            result.error = "Failed to start merge process: " + std::to_string(GetLastError());
            return result;
        }
        
        // Wait for process to complete
        WaitForSingleObject(pi.hProcess, INFINITE);
        
        DWORD exitCode;
        GetExitCodeProcess(pi.hProcess, &exitCode);
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
        
        if (exitCode != 0) {
            result.error = "Merge process failed with exit code: " + std::to_string(exitCode);
            return result;
        }
#else
        int exitCode = system(cmd.str().c_str());
        if (exitCode != 0) {
            result.error = "Merge process failed with exit code: " + std::to_string(exitCode);
            return result;
        }
#endif
        
        // After successful merge, look for output .grmt file
        fs::path tokenizedDir = fs::path(output_dir) / "tokenized";
        fs::path trainBin = tokenizedDir / "train.bin";
        
        if (fs::exists(trainBin)) {
            result.outputGrmtPath = trainBin.string();
            result.success = true;
            result.message = "Checkpoint merge completed successfully";
            
            // Try to get statistics from the merge output
            // (In a real implementation, you'd parse the output or read a status file)
            result.checkpointEntries = 0; // Would be populated from merge output
            result.verifiedEntries = 0;
            result.finalEntries = 0;
        } else {
            result.error = "Merge completed but output file not found: " + trainBin.string();
        }
        
    } catch (const std::exception& e) {
        result.error = std::string("Exception during checkpoint merge: ") + e.what();
        result.success = false;
    }
    
    return result;
}

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
        
        if (g_state.state != TrainingState_Idle && g_state.state != TrainingState_Completed) {
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
    
    // Detect checkpoints
    server.Post("/api/checkpoints/detect", [](const httplib::Request& req, httplib::Response& res) {
        flatbuffers::FlatBufferBuilder builder(2048);
        
        std::string checkpoint_dir = "data"; // Default
        
        if (!req.body.empty()) {
            try {
                auto request = flatbuffers::GetRoot<CheckpointDetectRequest>(req.body.data());
                if (request->checkpoint_dir()) {
                    checkpoint_dir = request->checkpoint_dir()->str();
                }
            } catch (const std::exception& e) {
                auto error = builder.CreateString(std::string("Invalid request: ") + e.what());
                auto message = builder.CreateString("");
                auto response = CreateCheckpointDetectResponse(builder, false, 0, 0, 0, 0, message, error);
                builder.Finish(response);
                
                res.status = 400;
                res.set_content(reinterpret_cast<const char*>(builder.GetBufferPointer()), 
                               builder.GetSize(), 
                               "application/octet-stream");
                return;
            }
        }
        
        auto detectResult = detectCheckpoints(checkpoint_dir);
        
        // Build CheckpointInfo vector
        std::vector<flatbuffers::Offset<CheckpointInfo>> checkpointInfos;
        for (const auto& [path, size, count] : detectResult.checkpoints) {
            auto pathStr = builder.CreateString(path);
            checkpointInfos.push_back(CreateCheckpointInfo(builder, pathStr, size, count));
        }
        auto checkpointsVec = builder.CreateVector(checkpointInfos);
        
        auto message = builder.CreateString(detectResult.success ? "Checkpoints detected" : "");
        auto error = builder.CreateString(detectResult.error);
        
        auto response = CreateCheckpointDetectResponse(builder,
            detectResult.success,
            detectResult.checkpointCount,
            detectResult.totalEntries,
            detectResult.totalSize,
            checkpointsVec,
            message,
            error
        );
        builder.Finish(response);
        
        res.set_content(reinterpret_cast<const char*>(builder.GetBufferPointer()), 
                       builder.GetSize(), 
                       "application/octet-stream");
    });
    
    // Merge checkpoints
    server.Post("/api/checkpoints/merge", [](const httplib::Request& req, httplib::Response& res) {
        flatbuffers::FlatBufferBuilder builder(1024);
        
        std::string checkpoint_dir = "data";
        std::string verified_dir = "data/verified";
        std::string output_dir = "data";
        bool skip_verification = false;
        
        if (!req.body.empty()) {
            try {
                auto request = flatbuffers::GetRoot<CheckpointMergeRequest>(req.body.data());
                if (request->checkpoint_dir()) checkpoint_dir = request->checkpoint_dir()->str();
                if (request->verified_dir()) verified_dir = request->verified_dir()->str();
                if (request->output_dir()) output_dir = request->output_dir()->str();
                skip_verification = request->skip_verification();
            } catch (const std::exception& e) {
                auto error = builder.CreateString(std::string("Invalid request: ") + e.what());
                auto message = builder.CreateString("");
                auto outputPath = builder.CreateString("");
                auto response = CreateCheckpointMergeResponse(builder, false, message, error, 0, 0, 0, outputPath);
                builder.Finish(response);
                
                res.status = 400;
                res.set_content(reinterpret_cast<const char*>(builder.GetBufferPointer()), 
                               builder.GetSize(), 
                               "application/octet-stream");
                return;
            }
        }
        
        auto mergeResult = mergeCheckpoints(checkpoint_dir, verified_dir, output_dir, skip_verification);
        
        auto message = builder.CreateString(mergeResult.message);
        auto error = builder.CreateString(mergeResult.error);
        auto outputPath = builder.CreateString(mergeResult.outputGrmtPath);
        
        auto response = CreateCheckpointMergeResponse(builder,
            mergeResult.success,
            message,
            error,
            mergeResult.checkpointEntries,
            mergeResult.verifiedEntries,
            mergeResult.finalEntries,
            outputPath
        );
        builder.Finish(response);
        
        if (!mergeResult.success) {
            res.status = 500;
        }
        
        res.set_content(reinterpret_cast<const char*>(builder.GetBufferPointer()), 
                       builder.GetSize(), 
                       "application/octet-stream");
    });
    
    // Data collection endpoint
    server.Post("/api/collection/start", [](const httplib::Request& req, httplib::Response& res) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "=== DATA COLLECTION REQUEST RECEIVED ===" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "[SERVER] >>> HTTP POST to /api/collection/start" << std::endl;
        std::cout << "[SERVER]     FROM: " << req.remote_addr << ":" << req.remote_port << std::endl;
        std::cout << "[SERVER]     CONTENT-TYPE: " << req.get_header_value("Content-Type") << std::endl;
        std::cout << "[SERVER]     BODY SIZE: " << req.body.size() << " bytes" << std::endl;
        std::cout << "[Data Collection] Got request from UI" << std::endl;
        
        flatbuffers::FlatBufferBuilder builder(512);
        
        std::string mode = "full";
        
        if (!req.body.empty()) {
            try {
                auto request = flatbuffers::GetRoot<DataCollectionRequest>(req.body.data());
                if (request->mode()) {
                    mode = request->mode()->str();
                    std::cout << "[Data Collection] Mode: " << mode << std::endl;
                }
            } catch (const std::exception& e) {
                std::cout << "[Data Collection] ERROR parsing request: " << e.what() << std::endl;
                auto error = builder.CreateString(std::string("Invalid request: ") + e.what());
                auto message = builder.CreateString("");
                auto response = CreateDataCollectionResponse(builder, false, message, error);
                builder.Finish(response);
                
                res.status = 400;
                res.set_content(reinterpret_cast<const char*>(builder.GetBufferPointer()), 
                               builder.GetSize(), 
                               "application/octet-stream");
                return;
            }
        }
        
        // Update state to Collecting
        {
            g_state.state = TrainingState_Collecting;
            InternalTrainingStats stats = g_state.getStats();
            stats.collectionProgress = 0.0f;
            stats.currentPhase = std::string("Starting data collection: ") + mode;
            g_state.updateStats(stats);
        }
        
        std::string result_message;
        std::string result_error;
        bool success = false;
        
        std::cout << "[Data Collection] Starting pipeline in mode: " << mode << std::endl;
        
        try {
            // Build path to grim_data_pipeline.exe using safe path resolution
            std::cout << "[Data Collection] Searching for grim_data_pipeline.exe..." << std::endl;
            std::string exePathStr = GRIM::Training::getSafeResourcePath(
                "resources/models/GRIM-text/DataCollection/build/Release/grim_data_pipeline.exe",
                GRIM::Training::PathResolutionMode::Relative
            );
            
            if (exePathStr.empty()) {
                std::cout << "[Data Collection] Relative path failed, trying search..." << std::endl;
                // Try searching for it
                exePathStr = GRIM::Training::getSafeResourcePath(
                    "grim_data_pipeline.exe",
                    GRIM::Training::PathResolutionMode::Search
                );
            }
            
            if (exePathStr.empty()) {
                result_error = "grim_data_pipeline.exe not found in GRIM directory tree";
                std::cerr << "[Data Collection] ERROR: " << result_error << std::endl;
            } else {
                std::cout << "[Data Collection] Found executable at: " << exePathStr << std::endl;
                std::filesystem::path exePath(exePathStr);
                // Run the pipeline
                std::string cmdLine = "\"" + exePath.string() + "\" " + mode;
                std::cout << "[Data Collection] Command line: " << cmdLine << std::endl;
                
#ifdef _WIN32
                STARTUPINFOA si = {};
                si.cb = sizeof(si);
                si.dwFlags = STARTF_USESHOWWINDOW;
                si.wShowWindow = SW_HIDE;
                
                PROCESS_INFORMATION pi = {};
                
                std::vector<char> cmdBuf(cmdLine.begin(), cmdLine.end());
                cmdBuf.push_back('\0');
                
                std::filesystem::path workingDir = exePath.parent_path().parent_path().parent_path();
                std::cout << "[Data Collection] Working directory: " << workingDir.string() << std::endl;
                
                BOOL createResult = CreateProcessA(
                    nullptr,
                    cmdBuf.data(),
                    nullptr,
                    nullptr,
                    FALSE,
                    CREATE_NO_WINDOW,
                    nullptr,
                    workingDir.string().c_str(),
                    &si,
                    &pi
                );
                
                if (!createResult) {
                    DWORD lastError = GetLastError();
                    result_error = "Failed to start pipeline process. Error code: " + std::to_string(lastError);
                    std::cerr << "[Data Collection] " << result_error << std::endl;
                } else {
                    std::cout << "[Data Collection] Process started (PID: " << pi.dwProcessId << ")" << std::endl;
                    
                    // Update initial phase
                    {
                        InternalTrainingStats stats = g_state.getStats();
                        stats.currentPhase = "Processing data sources...";
                        g_state.updateStats(stats);
                    }
                    
                    // Update progress periodically while waiting
                    DWORD waitResult;
                    int progressStep = 0;
                    int phaseCounter = 0;
                    do {
                        waitResult = WaitForSingleObject(pi.hProcess, 1000); // Wait 1 second
                        
                        if (waitResult == WAIT_TIMEOUT) {
                            // Still running, update progress (cap at 99, will set to 100 on success)
                            if (progressStep < 99) {
                                progressStep++;
                                InternalTrainingStats stats = g_state.getStats();
                                stats.collectionProgress = static_cast<float>(progressStep);
                                
                                // Update phase description every 10 seconds to show activity
                                phaseCounter++;
                                if (phaseCounter % 10 == 0) {
                                    if (progressStep < 30) {
                                        stats.currentPhase = "Collecting from data sources...";
                                    } else if (progressStep < 60) {
                                        stats.currentPhase = "Verifying collected data...";
                                    } else if (progressStep < 90) {
                                        stats.currentPhase = "Merging data entries...";
                                    } else {
                                        stats.currentPhase = "Finalizing dataset...";
                                    }
                                }
                                
                                g_state.updateStats(stats);
                                std::cout << "[Data Collection] Progress: " << progressStep << "% - " << stats.currentPhase << std::endl;
                            } else {
                                // Stay at 99% until process completes
                                InternalTrainingStats stats = g_state.getStats();
                                stats.currentPhase = "Completing final steps...";
                                g_state.updateStats(stats);
                                std::cout << "[Data Collection] Still running (99%)..." << std::endl;
                            }
                        }
                    } while (waitResult == WAIT_TIMEOUT);
                    
                    DWORD exitCode = 0;
                    GetExitCodeProcess(pi.hProcess, &exitCode);
                    
                    std::cout << "[Data Collection] Process finished with exit code: " << exitCode << std::endl;
                    
                    CloseHandle(pi.hProcess);
                    CloseHandle(pi.hThread);
                    
                    if (exitCode == 0) {
                        success = true;
                        result_message = "Data collection completed successfully";
                        std::cout << "[Data Collection] SUCCESS" << std::endl;
                        
                        InternalTrainingStats stats = g_state.getStats();
                        stats.collectionProgress = 100.0f;
                        stats.currentPhase = "Data collection complete!";
                        g_state.updateStats(stats);
                    } else {
                        result_error = "Pipeline failed with exit code: " + std::to_string(exitCode);
                        std::cerr << "[Data Collection] FAILED: " << result_error << std::endl;
                    }
                }
#else
                // Unix-like systems
                int exitCode = system(cmdLine.c_str());
                if (exitCode == 0) {
                    success = true;
                    result_message = "Data collection completed successfully";
                } else {
                    result_error = "Pipeline failed with exit code: " + std::to_string(exitCode);
                }
#endif
            }
        } catch (const std::exception& e) {
            result_error = std::string("Exception during collection: ") + e.what();
            std::cerr << "[Data Collection] EXCEPTION: " << result_error << std::endl;
        }
        
        std::cout << "[Data Collection] Resetting state to Idle" << std::endl;
        
        // Reset state
        {
            g_state.state = TrainingState_Idle;
            InternalTrainingStats stats = g_state.getStats();
            if (!success) {
                stats.collectionProgress = 0.0f;
            }
            stats.currentPhase = "";
            g_state.updateStats(stats);
        }
        
        auto message = builder.CreateString(result_message);
        auto error = builder.CreateString(result_error);
        auto response = CreateDataCollectionResponse(builder, success, message, error);
        builder.Finish(response);
        
        if (!success) {
            res.status = 500;
        }
        
        res.set_content(reinterpret_cast<const char*>(builder.GetBufferPointer()), 
                       builder.GetSize(), 
                       "application/octet-stream");
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
    std::cout << "  POST /api/checkpoints/detect - Detect checkpoint files" << std::endl;
    std::cout << "  POST /api/checkpoints/merge  - Merge checkpoints to .grmt" << std::endl;
    std::cout << "  POST /api/collection/start   - Start data collection pipeline" << std::endl;
    std::cout << "  GET  /api/logs              - Get training logs" << std::endl;
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
