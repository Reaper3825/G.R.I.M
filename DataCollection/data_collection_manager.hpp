#pragma once
#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <vector>

// Forward declarations to avoid including heavy headers
namespace GRIM {
namespace DataCollection {
    class WebDataCollector;
    struct Config;
    class Verifier;
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
    bool isRunning;
    
    CollectionStatus() 
        : phase("idle")
        , progress(0.0f)
        , sourcesProcessed(0)
        , totalSources(0)
        , checkpointsCollected(0)
        , isRunning(false) {}
};

class DataCollectionManager {
public:
    DataCollectionManager();
    ~DataCollectionManager();
    
    // Start full collection pipeline: collect → verify → merge
    bool startCollection(const std::string& mode = "full");
    
    // Stop current collection operation
    void stopCollection();
    
    // Get current status
    CollectionStatus getStatus() const;
    
    // Check if manager is ready
    bool isReady() const { return true; }  // Always ready in-process
    
    // Cleanup resources
    void shutdown();

private:
    void collectionThreadFunc();
    
    std::atomic<bool> running_;
    std::atomic<bool> shouldStop_;
    std::unique_ptr<std::thread> collectionThread_;
    
    mutable std::mutex statusMutex_;
    CollectionStatus currentStatus_;
};

} // namespace DataCollection
} // namespace GRIM
