//======================================================//
//  GRIM-text Training Control Client (Header-Only)
//  Simple HTTP client for controlling training
//  
//  Use this from GRIM.exe to control training without
//  linking to CUDA code.
//  
//  Uses FlatBuffers for efficient, schema-validated
//  binary communication.
//  
//  Author: GRIM Development Team
//  Date: November 6, 2025
//======================================================//

#pragma once
#include <string>
#include <functional>
#include <httplib.h>
#include <flatbuffers/flatbuffers.h>
#include "training_control_generated.h"

namespace GRIMText {

using TrainingState = Control::TrainingState;

struct TrainingStats {
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
    std::string currentPhase;
    std::string lastError;
    int64_t startTime = 0;
    int64_t elapsedTime = 0;
};

struct TrainingConfig {
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

class TrainingControlClient {
public:
    TrainingControlClient(const std::string& host = "127.0.0.1", int port = 11436)
        : host_(host), port_(port) {}
    
    // Check if server is running
    bool isServerRunning() {
        try {
            httplib::Client client(host_, port_);
            client.set_connection_timeout(1);
            auto res = client.Get("/health");
            return res && res->status == 200;
        } catch (...) {
            return false;
        }
    }
    
    // Get current training status
    bool getStatus(TrainingState& state, TrainingStats& stats, TrainingConfig& config) {
        try {
            httplib::Client client(host_, port_);
            auto res = client.Get("/api/status");
            
            if (!res || res->status != 200) {
                return false;
            }
            
            // Parse FlatBuffer response
            auto statusResponse = Control::GetStatusResponse(
                reinterpret_cast<const uint8_t*>(res->body.data())
            );
            
            // Parse state
            state = statusResponse->state();
            
            // Parse stats
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
                stats.collectionProgress = fbStats->collection_progress();
                stats.currentPhase = fbStats->current_phase() ? fbStats->current_phase()->str() : "";
                stats.lastError = fbStats->last_error() ? fbStats->last_error()->str() : "";
                stats.startTime = fbStats->start_time();
                stats.elapsedTime = fbStats->elapsed_time();
            }
            
            // Parse config
            if (statusResponse->config()) {
                auto fbConfig = statusResponse->config();
                config.epochs = fbConfig->epochs();
                config.batchSize = fbConfig->batch_size();
                config.learningRate = fbConfig->learning_rate();
                config.maxSeqLen = fbConfig->max_seq_len();
                config.warmupSteps = fbConfig->warmup_steps();
                config.useGPU = fbConfig->use_gpu();
                config.useFlashAttention = fbConfig->use_flash_attention();
                config.dataPath = fbConfig->data_path() ? fbConfig->data_path()->str() : "";
                config.vocabPath = fbConfig->vocab_path() ? fbConfig->vocab_path()->str() : "";
                config.outputPath = fbConfig->output_path() ? fbConfig->output_path()->str() : "";
            }
            
            return true;
        } catch (const std::exception& e) {
            lastError_ = e.what();
            return false;
        }
    }
    
    // Start training
    bool startTraining(const TrainingConfig* customConfig = nullptr) {
        try {
            httplib::Client client(host_, port_);
            
            flatbuffers::FlatBufferBuilder builder(512);
            flatbuffers::Offset<Control::TrainingConfig> configOffset = 0;
            
            if (customConfig) {
                auto dataPath = builder.CreateString(customConfig->dataPath);
                auto vocabPath = builder.CreateString(customConfig->vocabPath);
                auto outputPath = builder.CreateString(customConfig->outputPath);
                
                configOffset = Control::CreateTrainingConfig(builder,
                    customConfig->epochs,
                    customConfig->batchSize,
                    customConfig->learningRate,
                    customConfig->maxSeqLen,
                    customConfig->warmupSteps,
                    customConfig->useGPU,
                    customConfig->useFlashAttention,
                    dataPath,
                    vocabPath,
                    outputPath
                );
            }
            
            auto request = Control::CreateStartTrainingRequest(builder, configOffset);
            builder.Finish(request);
            
            auto res = client.Post("/api/training/start", 
                                  reinterpret_cast<const char*>(builder.GetBufferPointer()),
                                  builder.GetSize(),
                                  "application/octet-stream");
            
            if (!res || res->status != 200) {
                if (res) {
                    auto response = flatbuffers::GetRoot<Control::StartTrainingResponse>(
                        reinterpret_cast<const uint8_t*>(res->body.data())
                    );
                    lastError_ = response->error() ? response->error()->str() : "Unknown error";
                }
                return false;
            }
            
            return true;
        } catch (const std::exception& e) {
            lastError_ = e.what();
            return false;
        }
    }
    
    // Stop training
    bool stopTraining() {
        try {
            httplib::Client client(host_, port_);
            
            flatbuffers::FlatBufferBuilder builder(64);
            auto request = Control::CreateStopTrainingRequest(builder);
            builder.Finish(request);
            
            auto res = client.Post("/api/training/stop",
                                  reinterpret_cast<const char*>(builder.GetBufferPointer()),
                                  builder.GetSize(),
                                  "application/octet-stream");
            
            if (!res || res->status != 200) {
                if (res) {
                    auto response = flatbuffers::GetRoot<Control::StopTrainingResponse>(
                        reinterpret_cast<const uint8_t*>(res->body.data())
                    );
                    lastError_ = response->message() ? response->message()->str() : "Unknown error";
                }
                return false;
            }
            
            return true;
        } catch (const std::exception& e) {
            lastError_ = e.what();
            return false;
        }
    }
    
    // Update configuration
    bool updateConfig(const TrainingConfig& config) {
        try {
            httplib::Client client(host_, port_);
            
            flatbuffers::FlatBufferBuilder builder(512);
            
            auto dataPath = builder.CreateString(config.dataPath);
            auto vocabPath = builder.CreateString(config.vocabPath);
            auto outputPath = builder.CreateString(config.outputPath);
            
            auto fbConfig = Control::CreateTrainingConfig(builder,
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
            
            auto request = Control::CreateUpdateConfigRequest(builder, fbConfig);
            builder.Finish(request);
            
            auto res = client.Post("/api/config",
                                  reinterpret_cast<const char*>(builder.GetBufferPointer()),
                                  builder.GetSize(),
                                  "application/octet-stream");
            
            if (!res || res->status != 200) {
                if (res) {
                    auto response = flatbuffers::GetRoot<Control::UpdateConfigResponse>(
                        reinterpret_cast<const uint8_t*>(res->body.data())
                    );
                    lastError_ = response->error() ? response->error()->str() : "Unknown error";
                }
                return false;
            }
            
            return true;
        } catch (const std::exception& e) {
            lastError_ = e.what();
            return false;
        }
    }
    
    // Get last error message
    std::string getLastError() const {
        return lastError_;
    }
    
    // Checkpoint operations
    struct CheckpointInfo {
        std::string path;
        int64_t size = 0;
        int entryCount = 0;
    };
    
    struct CheckpointDetectResult {
        bool success = false;
        int checkpointCount = 0;
        int totalEntries = 0;
        int64_t totalSize = 0;
        std::vector<CheckpointInfo> checkpoints;
        std::string message;
        std::string error;
    };
    
    struct CheckpointMergeResult {
        bool success = false;
        std::string message;
        std::string error;
        int checkpointEntries = 0;
        int verifiedEntries = 0;
        int finalEntries = 0;
        std::string outputGrmtPath;
    };
    
    // Detect checkpoint files
    CheckpointDetectResult detectCheckpoints(const std::string& checkpoint_dir = "data") {
        CheckpointDetectResult result;
        
        try {
            httplib::Client client(host_, port_);
            
            flatbuffers::FlatBufferBuilder builder(256);
            auto dirPath = builder.CreateString(checkpoint_dir);
            auto request = Control::CreateCheckpointDetectRequest(builder, dirPath);
            builder.Finish(request);
            
            auto res = client.Post("/api/checkpoints/detect",
                                  reinterpret_cast<const char*>(builder.GetBufferPointer()),
                                  builder.GetSize(),
                                  "application/octet-stream");
            
            if (!res || res->status != 200) {
                result.error = "Failed to contact server";
                return result;
            }
            
            auto response = flatbuffers::GetRoot<Control::CheckpointDetectResponse>(
                reinterpret_cast<const uint8_t*>(res->body.data())
            );
            
            result.success = response->success();
            result.checkpointCount = response->checkpoint_count();
            result.totalEntries = response->total_entries();
            result.totalSize = response->total_size();
            result.message = response->message() ? response->message()->str() : "";
            result.error = response->error() ? response->error()->str() : "";
            
            if (response->checkpoints()) {
                for (const auto* checkpoint : *response->checkpoints()) {
                    CheckpointInfo info;
                    info.path = checkpoint->path() ? checkpoint->path()->str() : "";
                    info.size = checkpoint->size();
                    info.entryCount = checkpoint->entry_count();
                    result.checkpoints.push_back(info);
                }
            }
            
            return result;
        } catch (const std::exception& e) {
            result.error = std::string("Exception: ") + e.what();
            result.success = false;
            return result;
        }
    }
    
    // Merge checkpoint files to .grmt
    CheckpointMergeResult mergeCheckpoints(const std::string& checkpoint_dir = "data",
                                           const std::string& verified_dir = "data/verified",
                                           const std::string& output_dir = "data",
                                           bool skip_verification = false) {
        CheckpointMergeResult result;
        
        try {
            httplib::Client client(host_, port_);
            client.set_read_timeout(300); // 5 minutes for merge operation
            
            flatbuffers::FlatBufferBuilder builder(512);
            auto checkpointDirStr = builder.CreateString(checkpoint_dir);
            auto verifiedDirStr = builder.CreateString(verified_dir);
            auto outputDirStr = builder.CreateString(output_dir);
            
            auto request = Control::CreateCheckpointMergeRequest(builder,
                checkpointDirStr,
                verifiedDirStr,
                outputDirStr,
                skip_verification
            );
            builder.Finish(request);
            
            auto res = client.Post("/api/checkpoints/merge",
                                  reinterpret_cast<const char*>(builder.GetBufferPointer()),
                                  builder.GetSize(),
                                  "application/octet-stream");
            
            if (!res) {
                result.error = "Failed to contact server or timeout";
                return result;
            }
            
            auto response = flatbuffers::GetRoot<Control::CheckpointMergeResponse>(
                reinterpret_cast<const uint8_t*>(res->body.data())
            );
            
            result.success = response->success();
            result.message = response->message() ? response->message()->str() : "";
            result.error = response->error() ? response->error()->str() : "";
            result.checkpointEntries = response->checkpoint_entries();
            result.verifiedEntries = response->verified_entries();
            result.finalEntries = response->final_entries();
            result.outputGrmtPath = response->output_grmt_path() ? response->output_grmt_path()->str() : "";
            
            return result;
        } catch (const std::exception& e) {
            result.error = std::string("Exception: ") + e.what();
            result.success = false;
            return result;
        }
    }
    
    // Start data collection pipeline
    struct DataCollectionResult {
        bool success = false;
        std::string message;
        std::string error;
    };
    
    DataCollectionResult startDataCollection(const std::string& mode = "full") {
        DataCollectionResult result;
        
        std::cout << "\n[CLIENT] >>> HTTP POST REQUEST INITIATED <<<" << std::endl;
        std::cout << "[CLIENT]     FROM: TrainingControlClient::startDataCollection()" << std::endl;
        std::cout << "[CLIENT]     TO: " << host_ << ":" << port_ << "/api/collection/start" << std::endl;
        std::cout << "[CLIENT]     MODE: " << mode << std::endl;
        std::cout << "[CLIENT]     TIMEOUT: 600 seconds" << std::endl;
        
        try {
            httplib::Client client(host_, port_);
            client.set_read_timeout(600); // 10 minutes for full pipeline
            
            std::cout << "[CLIENT] Building FlatBuffer request..." << std::endl;
            flatbuffers::FlatBufferBuilder builder(256);
            auto modeStr = builder.CreateString(mode);
            auto request = Control::CreateDataCollectionRequest(builder, modeStr);
            builder.Finish(request);
            
            std::cout << "[CLIENT] Sending HTTP POST (size: " << builder.GetSize() << " bytes)..." << std::endl;
            auto res = client.Post("/api/collection/start",
                                  reinterpret_cast<const char*>(builder.GetBufferPointer()),
                                  builder.GetSize(),
                                  "application/octet-stream");
            
            if (!res) {
                std::cout << "[CLIENT] ERROR: No response from server (connection failed or timeout)" << std::endl;
                result.error = "Failed to contact server or timeout";
                return result;
            }
            
            std::cout << "[CLIENT] <<< HTTP RESPONSE RECEIVED <<<" << std::endl;
            std::cout << "[CLIENT]     STATUS: " << res->status << std::endl;
            std::cout << "[CLIENT]     BODY SIZE: " << res->body.size() << " bytes" << std::endl;
            
            auto response = flatbuffers::GetRoot<Control::DataCollectionResponse>(
                reinterpret_cast<const uint8_t*>(res->body.data())
            );
            
            result.success = response->success();
            result.message = response->message() ? response->message()->str() : "";
            result.error = response->error() ? response->error()->str() : "";
            
            std::cout << "[CLIENT]     SUCCESS: " << (result.success ? "true" : "false") << std::endl;
            std::cout << "[CLIENT]     MESSAGE: " << result.message << std::endl;
            std::cout << "[CLIENT]     ERROR: " << result.error << std::endl;
            
            return result;
        } catch (const std::exception& e) {
            std::cout << "[CLIENT] EXCEPTION: " << e.what() << std::endl;
            result.error = std::string("Exception: ") + e.what();
            result.success = false;
            return result;
        }
    }

private:
    std::string host_;
    int port_;
    std::string lastError_;
};

} // namespace GRIMText
