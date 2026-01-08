#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <filesystem>
#include <flatbuffers/flatbuffers.h>
#include "training_logs_generated.h"

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace fs = std::filesystem;
using namespace GRIM::Training;

//======================================================//
//  Training Logger with FlatBuffer
//======================================================//

class TrainingLogger {
public:
    TrainingLogger(const std::string& log_dir, const std::string& session_id)
 : log_dir_(log_dir), session_id_(session_id) {
        fs::create_directories(log_dir);
        
     // Open text log file
        // Append so anything emitted before logger construction (stdout redirect) is preserved.
        log_file_.open(log_dir_ + "/training_" + session_id_ + ".log", std::ios::app);
        log_file_ << std::fixed << std::setprecision(6);
        
 start_time_ = std::chrono::steady_clock::now();
    
log("========================================");
        log("  GRIM-text Training Session");
   log("  Session ID: " + session_id_);
      log("========================================");
    }
    
    ~TrainingLogger() {
if (log_file_.is_open()) {
        log_file_.close();
        }
    }
    
    void log(const std::string& message) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        std::stringstream ss;
        ss << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "] " << message;
    
        if (log_file_.is_open()) {
            log_file_ << ss.str() << std::endl;
            log_file_.flush();
        }
        // Silence stdout for non-fatal logs; fatal errors continue to use stderr elsewhere.
    }
    
    void logStep(flatbuffers::Offset<TrainingStepLog> step_log) {
      step_logs_.push_back(step_log);
    }
    
    void logEpoch(flatbuffers::Offset<EpochSummary> epoch_summary) {
        epoch_summaries_.push_back(epoch_summary);
    }
    
    void logValidation(flatbuffers::Offset<ValidationMetrics> val_metrics) {
        validation_logs_.push_back(val_metrics);
    }
  
 void logCheckpoint(flatbuffers::Offset<CheckpointMetadata> checkpoint) {
        checkpoints_.push_back(checkpoint);
    }
    
    bool saveFlatBuffer(flatbuffers::Offset<TrainingConfig> config) {
      flatbuffers::FlatBufferBuilder builder(1024 * 1024);  // 1MB initial
        
      auto session_id_str = builder.CreateString(session_id_);
        auto gpu_name_str = builder.CreateString(getGPUName());
        auto cuda_ver_str = builder.CreateString(getCUDAVersion());
        auto cudnn_ver_str = builder.CreateString("8.x");
        auto driver_ver_str = builder.CreateString(getDriverVersion());
        auto cpu_str = builder.CreateString(getCPUModel());
        auto git_commit_str = builder.CreateString("unknown");
     auto git_branch_str = builder.CreateString("master");
  auto early_stop_str = builder.CreateString("");
    
        auto end_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
        
        auto start_time_ms = end_time_ms - static_cast<uint64_t>(
 std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time_).count() * 1000);
        
auto session = CreateTrainingSession(builder,
            session_id_str,
         start_time_ms,
          end_time_ms,
            config,
        builder.CreateVector(step_logs_),
            builder.CreateVector(epoch_summaries_),
   builder.CreateVector(validation_logs_),
            builder.CreateVector(checkpoints_),
     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0, 0.0f, 0, 0, 0.0f,
          gpu_name_str, 1, cuda_ver_str, cudnn_ver_str, driver_ver_str,
       cpu_str, getTotalRAMGB(), 0, 0.0f, 0, 0.0f,
            0, 0, 0, 0, false, early_stop_str,
git_commit_str, git_branch_str, false
        );
        
        builder.Finish(session);
  
      // Save to file
        std::string fb_path = log_dir_ + "/training_" + session_id_ + ".fb";
        std::ofstream file(fb_path, std::ios::binary);
        if (!file) {
            log("ERROR: Failed to save FlatBuffer: " + fb_path);
    return false;
}
   
        file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
      log("Saved training log: " + fb_path);
        return true;
    }
    
private:
    std::string log_dir_;
    std::string session_id_;
    std::ofstream log_file_;
    std::chrono::steady_clock::time_point start_time_;
    
    std::vector<flatbuffers::Offset<TrainingStepLog>> step_logs_;
    std::vector<flatbuffers::Offset<EpochSummary>> epoch_summaries_;
    std::vector<flatbuffers::Offset<ValidationMetrics>> validation_logs_;
    std::vector<flatbuffers::Offset<CheckpointMetadata>> checkpoints_;
    
    std::string getGPUName() {
#ifdef USE_CUDA
cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
   return std::string(prop.name);
#else
        return "CPU";
#endif
    }
    
    std::string getCUDAVersion() {
#ifdef USE_CUDA
        int runtime_ver;
   cudaRuntimeGetVersion(&runtime_ver);
        return std::to_string(runtime_ver / 1000) + "." + std::to_string((runtime_ver % 100) / 10);
#else
     return "N/A";
#endif
    }
    
    std::string getDriverVersion() {
#ifdef USE_CUDA
        int driver_ver;
     cudaDriverGetVersion(&driver_ver);
        return std::to_string(driver_ver / 1000) + "." + std::to_string((driver_ver % 100) / 10);
#else
        return "N/A";
#endif
}
    
    std::string getCPUModel() {
#ifdef _WIN32
        return "Windows CPU";
#else
        return "Linux/Apple CPU";
#endif
    }
    
    float getTotalRAMGB() {
#ifdef _WIN32
      MEMORYSTATUSEX mem_info;
        mem_info.dwLength = sizeof(MEMORYSTATUSEX);
   GlobalMemoryStatusEx(&mem_info);
        return static_cast<float>(mem_info.ullTotalPhys) / (1024.0f * 1024.0f * 1024.0f);
#else
        return 0.0f;
#endif
    }
};
