#include "data_collection_manager.hpp"
#include "control/ai_config_paths.hpp"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <functional>

// Forward declare the renamed main function from grim_data_pipeline.cpp
extern int StartDataCollection(int argc, char** argv, std::function<void(float)> progressCallback);

namespace fs = std::filesystem;

namespace GRIM {
namespace DataCollection {

DataCollectionManager::DataCollectionManager()
    : running_(false)
    , shouldStop_(false) {
}

DataCollectionManager::~DataCollectionManager() {
    shutdown();
}

bool DataCollectionManager::startCollection(const std::string& mode) {
    std::lock_guard<std::mutex> lock(statusMutex_);
    
    if (running_) {
        currentStatus_.message = "Collection already in progress";
        return false;
    }
    
    // Reset from completion state if needed
    if (currentStatus_.progress == -1.0f) {
        currentStatus_.progress = 0.0f;
    }
    
    running_ = true;
    shouldStop_ = false;
    currentStatus_.phase = "starting";
    currentStatus_.progress = 0.0f;
    currentStatus_.message = "Initializing collection pipeline...";
    currentStatus_.isRunning = true;
    
    // Start collection thread that calls the pipeline
    collectionThread_ = std::make_unique<std::thread>(&DataCollectionManager::collectionThreadFunc, this);
    
    return true;
}

void DataCollectionManager::stopCollection() {
    shouldStop_ = true;
    
    if (collectionThread_ && collectionThread_->joinable()) {
        collectionThread_->join();
    }
    
    std::lock_guard<std::mutex> lock(statusMutex_);
    running_ = false;
    currentStatus_.phase = "stopped";
    currentStatus_.progress = -1.0f;
    currentStatus_.message = "Collection stopped by user";
    currentStatus_.isRunning = false;
}

CollectionStatus DataCollectionManager::getStatus() const {
    std::lock_guard<std::mutex> lock(statusMutex_);
    return currentStatus_;
}

void DataCollectionManager::shutdown() {
    stopCollection();
}

void DataCollectionManager::collectionThreadFunc() {
    try {
        {
            std::lock_guard<std::mutex> lock(statusMutex_);
            currentStatus_.phase = "collecting";
            currentStatus_.progress = 0.0f;
            currentStatus_.message = "Running data collection pipeline...";
        }
        
        // Call the pipeline directly with "full" mode and progress callback
        const char* argv[] = {"grim_data_pipeline", "full"};
        int result = StartDataCollection(2, const_cast<char**>(argv), 
            [this](float progress) {
                std::lock_guard<std::mutex> lock(statusMutex_);
                currentStatus_.progress = progress;
            });
        
        if (result != 0) {
            throw std::runtime_error("Data collection pipeline failed with code " + std::to_string(result));
        }
        
        // Complete
        {
            std::lock_guard<std::mutex> lock(statusMutex_);
            currentStatus_.phase = "complete";
            currentStatus_.progress = 100.0f;
            currentStatus_.message = "Collection pipeline completed successfully";
            currentStatus_.isRunning = false;
            running_ = false;
        }
        
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(statusMutex_);
        currentStatus_.phase = "error";
        currentStatus_.progress = -1.0f;
        currentStatus_.message = std::string("Collection failed: ") + e.what();
        currentStatus_.isRunning = false;
        running_ = false;
    }
}

} // namespace DataCollection
} // namespace GRIM
