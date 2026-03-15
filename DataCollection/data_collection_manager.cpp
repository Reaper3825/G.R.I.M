#include "data_collection_manager.hpp"
#include "collection_state.hpp"
#include "control/ai_config_paths.hpp"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <functional>
#include <chrono>

// Forward declare the renamed main function from grim_data_pipeline.cpp
extern int StartDataCollection(int argc, char** argv, std::function<void(float)> progressCallback);

namespace fs = std::filesystem;

namespace GRIM {
namespace DataCollection {

DataCollectionManager::DataCollectionManager()
    : running_(false)
    , shouldStop_(false)
    , initialized_(false) {
}

DataCollectionManager::~DataCollectionManager() {
    shutdown();
}

bool DataCollectionManager::initialize(const std::string& stateDir) {
    std::lock_guard<std::mutex> lock(statusMutex_);
    
    // Determine state directory
    if (stateDir.empty()) {
        // Use default from ai_config or fallback
        GRIM::Config::GrimTextPaths paths;
        if (GRIM::Config::loadGrimTextPaths(paths) && !paths.collected.empty()) {
            stateDir_ = (fs::path(paths.collected).parent_path() / "collection_state").string();
        } else {
            stateDir_ = "data/collection_state";
        }
    } else {
        stateDir_ = stateDir;
    }
    
    // Create state manager
    stateManager_ = std::make_unique<CollectionStateManager>();
    if (!stateManager_->initialize(stateDir_)) {
        std::cerr << "[DataCollectionManager] Failed to initialize state manager at: " << stateDir_ << std::endl;
        return false;
    }
    
    std::cout << "[DataCollectionManager] Initialized with state dir: " << stateDir_ << std::endl;
    std::cout << "[DataCollectionManager] Loaded " << stateManager_->getTotalUniqueUrls() << " tracked URLs" << std::endl;
    std::cout << "[DataCollectionManager] Loaded " << stateManager_->getTotalUniqueContent() << " content hashes" << std::endl;
    
    initialized_ = true;
    return true;
}

bool DataCollectionManager::startCollection(const CollectionConfig& config) {
    std::unique_lock<std::mutex> lock(statusMutex_);
    
    if (running_) {
        currentStatus_.message = "Collection already in progress";
        return false;
    }
    
    // Initialize if not done
    if (!initialized_) {
        lock.unlock();
        if (!initialize()) {
            return false;
        }
        lock.lock();
    }
    
    // Join previous thread before starting a new one (destroying a joinable
    // std::thread calls std::terminate → crash with STATUS_STACK_BUFFER_OVERRUN)
    if (collectionThread_ && collectionThread_->joinable()) {
        lock.unlock();
        collectionThread_->join();
        lock.lock();
    }
    
    // Reset from completion state if needed
    if (currentStatus_.progress == -1.0f) {
        currentStatus_.progress = 0.0f;
    }
    
    currentConfig_ = config;
    running_ = true;
    shouldStop_ = false;
    currentStatus_.phase = "starting";
    currentStatus_.progress = 0.0f;
    currentStatus_.message = "Initializing collection pipeline...";
    currentStatus_.isRunning = true;
    currentStatus_.startTimestamp = getCurrentTimestamp();
    currentStatus_.entriesCollected = 0;
    currentStatus_.duplicatesSkipped = 0;
    
    // Start collection thread
    collectionThread_ = std::make_unique<std::thread>(&DataCollectionManager::collectionThreadFunc, this);
    
    return true;
}

bool DataCollectionManager::startCollection(const std::string& mode) {
    CollectionConfig config;
    config.mode = mode;
    return startCollection(config);
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
    
    // Save state on stop
    if (stateManager_) {
        stateManager_->saveState();
    }
}

CollectionStatus DataCollectionManager::getStatus() const {
    std::lock_guard<std::mutex> lock(statusMutex_);
    
    // Update elapsed time if running
    CollectionStatus status = currentStatus_;
    if (status.isRunning && status.startTimestamp > 0) {
        status.elapsedSeconds = getCurrentTimestamp() - status.startTimestamp;
    }
    
    return status;
}

void DataCollectionManager::shutdown() {
    stopCollection();
    
    if (stateManager_) {
        stateManager_->saveState();
        stateManager_.reset();
    }
    
    initialized_ = false;
}

void DataCollectionManager::updateStatus(const std::string& phase, float progress, const std::string& message) {
    std::lock_guard<std::mutex> lock(statusMutex_);
    currentStatus_.phase = phase;
    currentStatus_.progress = progress;
    currentStatus_.message = message;
    
    if (progressCallback_) {
        progressCallback_(progress, message);
    }
}

int64_t DataCollectionManager::getCurrentTimestamp() const {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

size_t DataCollectionManager::getTotalUniqueUrls() const {
    if (stateManager_) {
        return stateManager_->getTotalUniqueUrls();
    }
    return 0;
}

size_t DataCollectionManager::getTotalUniqueContent() const {
    if (stateManager_) {
        return stateManager_->getTotalUniqueContent();
    }
    return 0;
}

size_t DataCollectionManager::getDuplicatesSkipped() const {
    if (stateManager_) {
        return stateManager_->getDuplicateCount();
    }
    return 0;
}

void DataCollectionManager::setProgressCallback(ProgressCallback callback) {
    progressCallback_ = std::move(callback);
}

void DataCollectionManager::collectionThreadFunc() {
    try {
        updateStatus("collecting", 0.0f, "Running data collection pipeline...");
        
        // Call the pipeline directly with mode and progress callback
        std::string mode = currentConfig_.mode.empty() ? "full" : currentConfig_.mode;
        const char* argv[] = {"grim_data_pipeline", mode.c_str()};
        
        int result = StartDataCollection(2, const_cast<char**>(argv), 
            [this](float progress) {
                std::lock_guard<std::mutex> lock(statusMutex_);
                currentStatus_.progress = progress;
                
                if (progressCallback_) {
                    progressCallback_(progress, currentStatus_.message);
                }
            });
        
        if (result != 0) {
            throw std::runtime_error("Data collection pipeline failed with code " + std::to_string(result));
        }
        
        // Save state after successful collection
        if (stateManager_) {
            stateManager_->saveState();
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
        
        // Still save state on error
        if (stateManager_) {
            stateManager_->saveState();
        }
    }
}

} // namespace DataCollection
} // namespace GRIM
