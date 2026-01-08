#pragma once

#include <string>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <flatbuffers/flatbuffers.h>
#include "../../../control/training_control_generated.h"

namespace fs = std::filesystem;

//======================================================//
//  Status File Writer for Training Control Server
//======================================================//

class StatusFileWriter {
public:
    StatusFileWriter(const std::string& filepath = "training_status.fb")
        : filepath_(filepath), start_time_(std::chrono::steady_clock::now()) {}
    
    void writeStatus(
        GRIMText::Control::TrainingState state,
        int current_epoch, int total_epochs,
        int current_batch, int total_batches,
        float current_loss, float avg_loss,
        float perplexity, float tokens_per_sec,
        float gpu_memory_used, float gpu_memory_total,
        const std::string& current_phase,
        const std::string& last_error = ""
    ) {
 flatbuffers::FlatBufferBuilder builder(1024);
     
    // Calculate elapsed time
      auto now = std::chrono::steady_clock::now();
        int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_).count();
        int64_t start_time_unix = std::chrono::duration_cast<std::chrono::milliseconds>(
   std::chrono::system_clock::now().time_since_epoch()).count() - elapsed_ms;
        
      // Calculate progress (epochs are 1-indexed, so subtract 1)
 float training_progress = 0.0f;
    if (total_epochs > 0 && total_batches > 0) {
            training_progress = (float)((current_epoch - 1) * total_batches + current_batch) / 
     (float)(total_epochs * total_batches) * 100.0f;
        }
        
        // Build TrainingStats
        auto phase_str = builder.CreateString(current_phase);
        auto error_str = builder.CreateString(last_error);
        
        auto stats = GRIMText::Control::CreateTrainingStats(builder,
            current_epoch, total_epochs,
    current_batch, total_batches,
            current_loss, avg_loss,
            perplexity, tokens_per_sec,
            gpu_memory_used, gpu_memory_total,
     training_progress,
      0.0f,  // collection_progress - not used during training
   phase_str, error_str,
    start_time_unix, elapsed_ms
        );
        
        // Build empty config (server has the config)
     auto config = GRIMText::Control::CreateTrainingConfig(builder);
        
        // Build StatusResponse
        int64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
      auto status_response = GRIMText::Control::CreateStatusResponse(builder,
state, stats, config, true, timestamp
        );
        
        builder.Finish(status_response);
        
   // Write to file atomically (write to temp, then rename)
     try {
            std::string temp_path = filepath_ + ".tmp";
        std::ofstream file(temp_path, std::ios::binary);
   if (file) {
         file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
     file.close();
 
           // Atomic rename
      std::error_code ec;
         fs::rename(temp_path, filepath_, ec);
             // Silently ignore errors - status file updates are non-critical
        }
        } catch (...) {
    // Silently ignore exceptions - status file updates are non-critical
        }
    }
    
private:
    std::string filepath_;
    std::chrono::steady_clock::time_point start_time_;
};
