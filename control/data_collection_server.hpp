#pragma once

#include <string>
#include <thread>
#include <atomic>
#include <filesystem>
#include <iostream>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#endif

#include "httplib.h"
#include "flatbuffers/flatbuffers.h"
#include "data_collection_protocol_generated.h"

namespace GRIM {
namespace DataCollection {

// Server state and progress tracking
struct DataCollectionState {
    std::atomic<bool> isCollecting{false};
    std::atomic<float> progress{0.0f};
    std::atomic<bool> shouldStop{false};
    std::string currentPhase;
    std::string lastError;
    mutable std::mutex stateMutex;  // mutable so we can lock in const methods
};

class DataCollectionServer {
public:
    DataCollectionServer(const std::string& host = "localhost", int port = 11437)
        : host_(host), port_(port), running_(false) {
        std::cout << "[DataCollectionServer] Initializing on " << host << ":" << port << std::endl;
    }

    ~DataCollectionServer() {
        stop();
    }

    bool start() {
        if (running_) {
            std::cout << "[DataCollectionServer] Already running" << std::endl;
            return true;
        }

        running_ = true;
        serverThread_ = std::thread([this]() { this->run(); });
        
        std::cout << "[DataCollectionServer] Started successfully on port " << port_ << std::endl;
        return true;
    }

    void stop() {
        if (!running_) return;

        std::cout << "[DataCollectionServer] Stopping server..." << std::endl;
        running_ = false;
        state_.shouldStop = true;

        if (serverThread_.joinable()) {
            serverThread_.join();
        }

        std::cout << "[DataCollectionServer] Stopped" << std::endl;
    }

    bool isRunning() const { return running_; }
    bool isCollecting() const { return state_.isCollecting; }
    float getProgress() const { return state_.progress; }
    std::string getCurrentPhase() const {
        std::lock_guard<std::mutex> lock(state_.stateMutex);
        return state_.currentPhase;
    }

private:
    void run() {
        auto serverPtr = std::make_shared<httplib::Server>();
        httplib::Server& server = *serverPtr;

        // Health check endpoint
        server.Get("/health", [](const httplib::Request& req, httplib::Response& res) {
            res.set_content("{\"status\":\"ok\"}", "application/json");
        });

        // Start data collection endpoint
        server.Post("/api/collection/start", [this](const httplib::Request& req, httplib::Response& res) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "=== DATA COLLECTION REQUEST RECEIVED ===" << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << "[DataCollectionServer] >>> HTTP POST to /api/collection/start" << std::endl;
            std::cout << "[DataCollectionServer]     FROM: " << req.remote_addr << ":" << req.remote_port << std::endl;
            std::cout << "[DataCollectionServer]     CONTENT-TYPE: " << req.get_header_value("Content-Type") << std::endl;
            std::cout << "[DataCollectionServer]     BODY SIZE: " << req.body.size() << " bytes" << std::endl;

            handleCollectionStart(req, res);
        });

        // Get status endpoint
        server.Get("/api/collection/status", [this](const httplib::Request& req, httplib::Response& res) {
            handleStatusRequest(req, res);
        });

        // Stop collection endpoint
        server.Post("/api/collection/stop", [this](const httplib::Request& req, httplib::Response& res) {
            std::cout << "[DataCollectionServer] Stop request received" << std::endl;
            state_.shouldStop = true;
            state_.isCollecting = false;  // Force reset state
            state_.progress = 0.0f;
            {
                std::lock_guard<std::mutex> lock(state_.stateMutex);
                state_.currentPhase = "Stopped";
            }
            res.set_content("{\"status\":\"stopped\"}", "application/json");
        });

        // Shutdown server endpoint
        server.Post("/shutdown", [this, serverPtr](const httplib::Request& req, httplib::Response& res) {
            std::cout << "[DataCollectionServer] Shutdown request received" << std::endl;
            res.set_content("{\"status\":\"shutting_down\"}", "application/json");
            running_ = false;
            serverPtr->stop();
        });

        std::cout << "[DataCollectionServer] Listening on " << host_ << ":" << port_ << std::endl;
        server.listen(host_.c_str(), port_);
    }

    void handleCollectionStart(const httplib::Request& req, httplib::Response& res) {
        flatbuffers::FlatBufferBuilder builder(512);

        // Check if already collecting
        if (state_.isCollecting) {
            std::cout << "[DataCollectionServer] ERROR: Collection already in progress" << std::endl;
            auto error = builder.CreateString("Collection already in progress");
            auto message = builder.CreateString("");
            auto response = CreateDataCollectionResponse(builder, false, message, error);
            builder.Finish(response);

            res.status = 409; // Conflict
            res.set_content(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                           builder.GetSize(),
                           "application/octet-stream");
            return;
        }

        // Parse request
        std::string mode = "full";
        if (!req.body.empty()) {
            try {
                auto request = flatbuffers::GetRoot<DataCollectionRequest>(req.body.data());
                if (request->mode()) {
                    mode = request->mode()->str();
                    std::cout << "[DataCollectionServer] Mode: " << mode << std::endl;
                }
            } catch (const std::exception& e) {
                std::cout << "[DataCollectionServer] ERROR parsing request: " << e.what() << std::endl;
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

        // ✅ FIX: Set isCollecting flag BEFORE launching thread to avoid race condition
        state_.isCollecting.store(true, std::memory_order_release);
        
        // ✅ SMART PROGRESS RESET: If progress is -1 (completion signal), reset to 0
        // Otherwise start from current progress (allows resuming)
        float currentProgress = state_.progress.load(std::memory_order_acquire);
        if (currentProgress < 0.0f) {
            std::cout << "[DataCollectionServer] Resetting progress from -1 to 0 (new collection cycle)" << std::endl;
            state_.progress.store(0.0f, std::memory_order_release);
        } else {
            std::cout << "[DataCollectionServer] Starting collection from progress: " << currentProgress << "%" << std::endl;
        }
        
        state_.shouldStop.store(false, std::memory_order_release);
        state_.lastError = "";
        {
            std::lock_guard<std::mutex> lock(state_.stateMutex);
            state_.currentPhase = "Starting collection...";
        }

        // Start collection in background thread
        std::thread([this, mode]() {
            runDataCollection(mode);
        }).detach();

        // Return success immediately (collection runs in background)
        std::cout << "[DataCollectionServer] Collection started in background" << std::endl;
        auto message = builder.CreateString("Data collection started");
        auto error = builder.CreateString("");
        auto response = CreateDataCollectionResponse(builder, true, message, error);
        builder.Finish(response);

        res.status = 202; // Accepted
        res.set_content(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                       builder.GetSize(),
                       "application/octet-stream");
    }

    void handleStatusRequest(const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(state_.stateMutex);
        
        bool collecting = state_.isCollecting.load(std::memory_order_acquire);  // Explicit atomic load with acquire semantics
        std::string status = collecting ? "collecting" : "idle";
        std::string json = "{"
            "\"status\":\"" + status + "\","
            "\"progress\":" + std::to_string(state_.progress.load(std::memory_order_acquire)) + ","
            "\"phase\":\"" + state_.currentPhase + "\","
            "\"error\":\"" + state_.lastError + "\""
            "}";

        res.set_content(json, "application/json");
    }

    void runDataCollection(const std::string& mode) {
        std::cout << "[DataCollectionServer] >>> STARTING DATA COLLECTION PROCESS <<<" << std::endl;
        
        // ✅ NOTE: isCollecting, progress, shouldStop, and lastError are already set by handleCollectionStart
        // No need to reset them here (avoids race condition)

        {
            std::lock_guard<std::mutex> lock(state_.stateMutex);
            state_.currentPhase = "Initializing pipeline...";
        }

        try {
            // Find the data pipeline executable
            std::string exePath = findDataPipelineExecutable();
            
            if (exePath.empty()) {
                state_.lastError = "grim_data_pipeline.exe not found";
                std::cerr << "[DataCollectionServer] ERROR: " << state_.lastError << std::endl;
                {
                    std::lock_guard<std::mutex> lock(state_.stateMutex);
                    state_.currentPhase = "Error: Executable not found";
                }
                // Keep isCollecting = true briefly so UI can see the error
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                state_.isCollecting = false;
                return;
            }

            std::cout << "[DataCollectionServer] Found executable: " << exePath << std::endl;
            
            // Run the pipeline
            bool success = executeDataPipeline(exePath, mode);
            
            if (success) {
                std::cout << "[DataCollectionServer] Collection completed successfully!" << std::endl;
                {
                    std::lock_guard<std::mutex> lock(state_.stateMutex);
                    state_.currentPhase = "Complete";
                }
                // Set progress to -1 to signal completion (will be reset to 0 on next start)
                state_.progress.store(-1.0f, std::memory_order_release);
                std::cout << "[DataCollectionServer] Progress set to -1 (completion signal)" << std::endl;
            } else {
                std::cerr << "[DataCollectionServer] Collection failed!" << std::endl;
                {
                    std::lock_guard<std::mutex> lock(state_.stateMutex);
                    state_.currentPhase = "Failed";
                }
                // Set progress to -1 on failure too (clean restart)
                state_.progress.store(-1.0f, std::memory_order_release);
            }

        } catch (const std::exception& e) {
            state_.lastError = e.what();
            std::cerr << "[DataCollectionServer] EXCEPTION: " << e.what() << std::endl;
            // Set progress to -1 on exception too
            state_.progress.store(-1.0f, std::memory_order_release);
        }

        state_.isCollecting = false;
        std::cout << "[DataCollectionServer] >>> DATA COLLECTION PROCESS FINISHED <<<" << std::endl;
    }

    std::string findDataPipelineExecutable() {
        std::cout << "[DataCollectionServer] Searching for grim_data_pipeline.exe..." << std::endl;

        // Try multiple possible locations
        std::vector<std::filesystem::path> searchPaths = {
            "resources/models/GRIM-text/DataCollection/build/Release/grim_data_pipeline.exe",
            "resources/models/GRIM-text/DataCollection/build/Debug/grim_data_pipeline.exe",
            "DataCollection/build/Release/grim_data_pipeline.exe",
            "../resources/models/GRIM-text/DataCollection/build/Release/grim_data_pipeline.exe"
        };

        // Get GRIM root directory
        std::filesystem::path grimRoot = std::filesystem::current_path();
        
#ifdef _WIN32
        char exeBuffer[MAX_PATH];
        DWORD len = GetModuleFileNameA(nullptr, exeBuffer, MAX_PATH);
        if (len > 0 && len < MAX_PATH) {
            std::filesystem::path exePath(exeBuffer);
            grimRoot = exePath.parent_path();
            
            // Walk up to find GRIM root
            for (int i = 0; i < 5; ++i) {
                if (std::filesystem::exists(grimRoot / "resources") && 
                    std::filesystem::exists(grimRoot / "control")) {
                    break;
                }
                grimRoot = grimRoot.parent_path();
            }
        }
#endif

        std::cout << "[DataCollectionServer] GRIM root: " << grimRoot.string() << std::endl;

        for (const auto& relPath : searchPaths) {
            auto fullPath = grimRoot / relPath;
            std::cout << "[DataCollectionServer] Checking: " << fullPath.string() << std::endl;
            
            if (std::filesystem::exists(fullPath)) {
                std::cout << "[DataCollectionServer] ✓ Found!" << std::endl;
                return fullPath.string();
            }
        }

        std::cout << "[DataCollectionServer] ✗ Not found in any standard location" << std::endl;
        return "";
    }

    bool executeDataPipeline(const std::string& exePath, const std::string& mode) {
        std::cout << "[DataCollectionServer] >>> LAUNCHING EXECUTABLE <<<" << std::endl;
        std::cout << "[DataCollectionServer]     EXE: " << exePath << std::endl;
        std::cout << "[DataCollectionServer]     MODE: " << mode << std::endl;

#ifdef _WIN32
        std::filesystem::path exePathObj(exePath);
        std::filesystem::path workingDir = exePathObj.parent_path().parent_path().parent_path();
        
        std::string cmdLine = "\"" + exePath + "\" " + mode;
        std::cout << "[DataCollectionServer]     CMD: " << cmdLine << std::endl;
        std::cout << "[DataCollectionServer]     CWD: " << workingDir.string() << std::endl;

        STARTUPINFOA si = {};
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESHOWWINDOW;
        si.wShowWindow = SW_HIDE;

        PROCESS_INFORMATION pi = {};

        std::vector<char> cmdBuf(cmdLine.begin(), cmdLine.end());
        cmdBuf.push_back('\0');

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
            state_.lastError = "CreateProcess failed with error code: " + std::to_string(lastError);
            std::cerr << "[DataCollectionServer] " << state_.lastError << std::endl;
            return false;
        }

        std::cout << "[DataCollectionServer] Process created (PID: " << pi.dwProcessId << ")" << std::endl;

        {
            std::lock_guard<std::mutex> lock(state_.stateMutex);
            state_.currentPhase = "Processing data sources...";
        }

        // Monitor process with progress updates
        DWORD waitResult;
        int progressStep = 0;
        int phaseCounter = 0;

        do {
            waitResult = WaitForSingleObject(pi.hProcess, 1000); // Wait 1 second

            if (waitResult == WAIT_TIMEOUT) {
                // Still running, update progress
                if (progressStep < 99) {
                    progressStep++;
                    state_.progress = static_cast<float>(progressStep);

                    // Update phase every 10 seconds
                    phaseCounter++;
                    if (phaseCounter % 10 == 0) {
                        std::lock_guard<std::mutex> lock(state_.stateMutex);
                        if (progressStep < 30) {
                            state_.currentPhase = "Collecting from data sources...";
                        } else if (progressStep < 60) {
                            state_.currentPhase = "Verifying collected data...";
                        } else if (progressStep < 90) {
                            state_.currentPhase = "Merging data entries...";
                        } else {
                            state_.currentPhase = "Finalizing dataset...";
                        }
                    }

                    std::cout << "[DataCollectionServer] Progress: " << progressStep << "%" << std::endl;
                }

                // Check if stop requested
                if (state_.shouldStop) {
                    std::cout << "[DataCollectionServer] Stop requested, terminating process..." << std::endl;
                    TerminateProcess(pi.hProcess, 1);
                    CloseHandle(pi.hProcess);
                    CloseHandle(pi.hThread);
                    return false;
                }
            }
        } while (waitResult == WAIT_TIMEOUT);

        DWORD exitCode = 0;
        GetExitCodeProcess(pi.hProcess, &exitCode);

        std::cout << "[DataCollectionServer] Process finished with exit code: " << exitCode << std::endl;

        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);

        if (exitCode == 0) {
            return true;
        } else {
            state_.lastError = "Pipeline failed with exit code: " + std::to_string(exitCode);
            return false;
        }
#else
        // Unix-like systems
        std::string cmdLine = "\"" + exePath + "\" " + mode;
        int exitCode = system(cmdLine.c_str());
        return (exitCode == 0);
#endif
    }

private:
    std::string host_;
    int port_;
    std::atomic<bool> running_;
    std::thread serverThread_;
    DataCollectionState state_;
};

// Global instance
inline std::unique_ptr<DataCollectionServer> g_dataCollectionServer;

inline bool startDataCollectionServer() {
    if (!g_dataCollectionServer) {
        g_dataCollectionServer = std::make_unique<DataCollectionServer>("localhost", 11437);
    }
    return g_dataCollectionServer->start();
}

inline void stopDataCollectionServer() {
    if (g_dataCollectionServer) {
        g_dataCollectionServer->stop();
        g_dataCollectionServer.reset();
    }
}

inline bool isDataCollectionServerRunning() {
    return g_dataCollectionServer && g_dataCollectionServer->isRunning();
}

} // namespace DataCollection
} // namespace GRIM
