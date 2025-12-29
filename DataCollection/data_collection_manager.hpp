#pragma once
#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <vector>
#include <functional>

// Forward declarations to avoid including heavy headers
namespace GRIM {
namespace DataCollection {
    class WebDataCollector;
    struct Config;
    class Verifier;
    class CollectionStateManager;
}
}

namespace GRIM {
namespace DataCollection {

struct CollectionStatus {
    std::string phase;        // "idle", "collecting", "verifying", "merging", "complete", "error"
    float progress;           // 0-100, or -1 for completion signal
    std::string message;
    int sourcesProcessed;
    int totalSources;
    int checkpointsCollected;
    int entriesCollected;
    int duplicatesSkipped;
    bool isRunning;
    
    // Extended status for UI
    std::string currentSource;
    int64_t startTimestamp;
    int64_t elapsedSeconds;
    
    CollectionStatus() 
        : phase("idle")
        , progress(0.0f)
        , sourcesProcessed(0)
        , totalSources(0)
        , checkpointsCollected(0)
        , entriesCollected(0)
        , duplicatesSkipped(0)
        , isRunning(false)
        , startTimestamp(0)
        , elapsedSeconds(0) {}
};

// Collection configuration from UI
struct CollectionConfig {
    std::string mode = "full";  // "full", "collect", "verify", "merge"
    std::string sourceConfigPath;  // Path to source_data.json
    int fetchLimit = 100;
    int vocabSize = 50000;
    float verificationThreshold = 0.7f;
    bool skipVerification = false;
    bool retrainVocab = false;
    bool deduplicateContent = true;
    bool deduplicateUrls = true;
    int refreshIntervalHours = 24;  // Re-collect sources after this many hours
};

class DataCollectionManager {
public:
    DataCollectionManager();
    ~DataCollectionManager();
    
    // Initialize with state directory for persistence
    bool initialize(const std::string& stateDir = "");
    
    // Start collection with config
    bool startCollection(const CollectionConfig& config);
    
    // Start full collection pipeline: collect → verify → merge
    bool startCollection(const std::string& mode = "full");
    
    // Stop current collection operation
    void stopCollection();
    
    // Get current status
    CollectionStatus getStatus() const;
    
    // Check if manager is ready
    bool isReady() const { return initialized_; }
    
    // Cleanup resources
    void shutdown();
    
    // State management
    CollectionStateManager* getStateManager() { return stateManager_.get(); }
    const CollectionStateManager* getStateManager() const { return stateManager_.get(); }
    
    // Statistics
    size_t getTotalUniqueUrls() const;
    size_t getTotalUniqueContent() const;
    size_t getDuplicatesSkipped() const;
    
    // Progress callback for external monitoring
    using ProgressCallback = std::function<void(float progress, const std::string& message)>;
    void setProgressCallback(ProgressCallback callback);

private:
    void collectionThreadFunc();
    void updateStatus(const std::string& phase, float progress, const std::string& message);
    int64_t getCurrentTimestamp() const;
    
    std::atomic<bool> running_;
    std::atomic<bool> shouldStop_;
    std::atomic<bool> initialized_;
    std::unique_ptr<std::thread> collectionThread_;
    
    mutable std::mutex statusMutex_;
    CollectionStatus currentStatus_;
    CollectionConfig currentConfig_;
    
    std::unique_ptr<CollectionStateManager> stateManager_;
    ProgressCallback progressCallback_;
    
    std::string stateDir_;
};

} // namespace DataCollection
} // namespace GRIM
