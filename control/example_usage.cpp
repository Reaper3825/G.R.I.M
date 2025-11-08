//======================================================//
//  Example: Using Training Control Client in GRIM.exe
//  
//  This file demonstrates how to integrate the training
//  control system into your GRIM.exe UI panel.
//  
//  NO CUDA LINKING REQUIRED!
//======================================================//

#include "training_control_client.hpp"
#include "logger.hpp"
#include <thread>
#include <chrono>

//======================================================//
//  Example 1: Simple Status Query
//======================================================//

void exampleGetStatus() {
    GRIMText::TrainingControlClient client("127.0.0.1", 11436);
    
    // Check if server is running
    if (!client.isServerRunning()) {
        LOG_ERROR("Training", "Control server not running. Start it first!");
        return;
    }
    
    // Query current status
    GRIMText::TrainingState state;
    GRIMText::TrainingStats stats;
    GRIMText::TrainingConfig config;
    
    if (client.getStatus(state, stats, config)) {
        LOG_DEBUG("Training", "Current state: " + std::to_string(static_cast<int>(state)));
        LOG_DEBUG("Training", "Epoch: " + std::to_string(stats.currentEpoch) + "/" + std::to_string(stats.totalEpochs));
        LOG_DEBUG("Training", "Batch: " + std::to_string(stats.currentBatch) + "/" + std::to_string(stats.totalBatches));
        LOG_DEBUG("Training", "Loss: " + std::to_string(stats.currentLoss));
        LOG_DEBUG("Training", "Perplexity: " + std::to_string(stats.perplexity));
        LOG_DEBUG("Training", "Tokens/sec: " + std::to_string(stats.tokensPerSec));
        LOG_DEBUG("Training", "GPU Memory: " + std::to_string(stats.gpuMemoryUsed) + " / " + std::to_string(stats.gpuMemoryTotal) + " MB");
        LOG_DEBUG("Training", "Progress: " + std::to_string(stats.trainingProgress * 100.0f) + "%");
    } else {
        LOG_ERROR("Training", "Failed to get status: " + client.getLastError());
    }
}

//======================================================//
//  Example 2: Start Training with Custom Config
//======================================================//

void exampleStartTraining() {
    GRIMText::TrainingControlClient client("127.0.0.1", 11436);
    
    if (!client.isServerRunning()) {
        LOG_ERROR("Training", "Control server not running");
        return;
    }
    
    // Create custom configuration
    GRIMText::TrainingConfig config;
    config.epochs = 5;
    config.batchSize = 16;
    config.learningRate = 0.0002f;
    config.maxSeqLen = 4096;
    config.warmupSteps = 500;
    config.useGPU = true;
    config.useFlashAttention = true;
    config.dataPath = "data/training_data.grmt";
    config.vocabPath = "models/vocab.bin";
    config.outputPath = "models/grim_text_custom.bin";
    
    // Start training
    if (client.startTraining(&config)) {
        LOG_DEBUG("Training", "Training started successfully");
    } else {
        LOG_ERROR("Training", "Failed to start training: " + client.getLastError());
    }
}

//======================================================//
//  Example 3: Monitor Training Progress
//======================================================//

void exampleMonitorTraining() {
    GRIMText::TrainingControlClient client("127.0.0.1", 11436);
    
    if (!client.isServerRunning()) {
        LOG_ERROR("Training", "Control server not running");
        return;
    }
    
    LOG_DEBUG("Training", "Monitoring training progress...");
    
    while (true) {
        GRIMText::TrainingState state;
        GRIMText::TrainingStats stats;
        GRIMText::TrainingConfig config;
        
        if (client.getStatus(state, stats, config)) {
            // Check state
            if (state == GRIMText::TrainingState::Idle) {
                LOG_DEBUG("Training", "Training not started");
                break;
            } else if (state == GRIMText::TrainingState::Training) {
                // Training in progress
                float progress = stats.trainingProgress * 100.0f;
                LOG_DEBUG("Training", 
                    "Epoch " + std::to_string(stats.currentEpoch) + "/" + std::to_string(stats.totalEpochs) +
                    " | Loss: " + std::to_string(stats.currentLoss) +
                    " | Progress: " + std::to_string(progress) + "%"
                );
            } else if (state == GRIMText::TrainingState::Completed) {
                LOG_DEBUG("Training", "Training completed!");
                LOG_DEBUG("Training", "Final loss: " + std::to_string(stats.avgLoss));
                LOG_DEBUG("Training", "Final perplexity: " + std::to_string(stats.perplexity));
                break;
            } else if (state == GRIMText::TrainingState::Error) {
                LOG_ERROR("Training", "Training error: " + stats.lastError);
                break;
            }
        } else {
            LOG_ERROR("Training", "Failed to get status: " + client.getLastError());
            break;
        }
        
        // Poll every 2 seconds
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}

//======================================================//
//  Example 4: Stop Training
//======================================================//

void exampleStopTraining() {
    GRIMText::TrainingControlClient client("127.0.0.1", 11436);
    
    if (!client.isServerRunning()) {
        LOG_ERROR("Training", "Control server not running");
        return;
    }
    
    if (client.stopTraining()) {
        LOG_DEBUG("Training", "Training stopped successfully");
    } else {
        LOG_ERROR("Training", "Failed to stop training: " + client.getLastError());
    }
}

//======================================================//
//  Example 5: Integration with UI Panel
//======================================================//

class UITrainingPanel_Example {
private:
    GRIMText::TrainingControlClient client{"127.0.0.1", 11436};
    float pollTimer = 0.0f;
    float pollInterval = 1.0f;  // Poll every second
    
    GRIMText::TrainingState currentState = GRIMText::TrainingState::Idle;
    GRIMText::TrainingStats currentStats;
    GRIMText::TrainingConfig currentConfig;
    
public:
    void update(float dt) {
        // Check if server is running
        if (!client.isServerRunning()) {
            // Could auto-start server here
            return;
        }
        
        // Poll status periodically
        pollTimer += dt;
        if (pollTimer >= pollInterval) {
            pollTimer = 0.0f;
            
            if (client.getStatus(currentState, currentStats, currentConfig)) {
                // Status updated successfully
                // UI can now use currentStats to update displays
            }
        }
    }
    
    void onStartButtonClicked() {
        // Get config from UI widgets
        GRIMText::TrainingConfig config;
        config.epochs = getEpochsFromUI();
        config.batchSize = getBatchSizeFromUI();
        config.learningRate = getLearningRateFromUI();
        // ... etc
        
        if (client.startTraining(&config)) {
            LOG_DEBUG("TrainingPanel", "Training started");
        } else {
            LOG_ERROR("TrainingPanel", "Failed to start: " + client.getLastError());
        }
    }
    
    void onStopButtonClicked() {
        if (client.stopTraining()) {
            LOG_DEBUG("TrainingPanel", "Training stopped");
        } else {
            LOG_ERROR("TrainingPanel", "Failed to stop: " + client.getLastError());
        }
    }
    
    void drawProgressBar() {
        float progress = currentStats.trainingProgress;
        // Draw progress bar with 'progress' value (0.0 to 1.0)
    }
    
    void drawStats() {
        // Display current training statistics
        // drawText("Epoch: " + std::to_string(currentStats.currentEpoch));
        // drawText("Loss: " + std::to_string(currentStats.currentLoss));
        // etc...
    }
    
private:
    int getEpochsFromUI() { return 3; /* Get from slider */ }
    int getBatchSizeFromUI() { return 8; /* Get from slider */ }
    float getLearningRateFromUI() { return 0.0001f; /* Get from slider */ }
};

//======================================================//
//  Main (for testing)
//======================================================//

int main() {
    // Example 1: Get status
    exampleGetStatus();
    
    // Example 2: Start training
    // exampleStartTraining();
    
    // Example 3: Monitor progress
    // exampleMonitorTraining();
    
    // Example 4: Stop training
    // exampleStopTraining();
    
    return 0;
}
