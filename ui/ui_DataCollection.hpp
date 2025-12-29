#pragma once
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <optional>
#include <atomic>
#include "ui_panel.hpp"
#include "ui_button.hpp"
#include "ui_slider.hpp"
#include "ui_progress_bar.hpp"
#include "ui_layout_box.hpp"
#include "ui_inputbox.hpp"
#include "ui_scrollbox.hpp"
#include "ui_dropdown.hpp"
#include "DataCollection/data_collection_manager.hpp"
#include "DataCollection/huggingface_webhook.hpp"

class OverlayRenderer;
struct InputState;

class UIDataCollectionPanel : public UIPanel {
public:
    UIDataCollectionPanel();
    ~UIDataCollectionPanel() override;

    void update(const InputState& input, float dt) override;
    void drawOverlay(OverlayRenderer& renderer) override;
    
    // Public control functions
    void startFullCollection();
    void startCollectOnly();
    void startVerifyOnly();
    void startMergeOnly();
    void startForceRebuild();  // Rebuild ignoring deduplication state
    void stopCollection();
    void addDataSource(const std::string& url);
    
    // Hugging Face specific functions
    void searchHuggingFaceDatasets();
    void searchHuggingFaceByCategory(const std::string& category);
    void downloadHuggingFaceDataset(const std::string& datasetId);
    void browseHuggingFaceDatasets();
    
    // Filter functions
    void setSourceFilter(const std::string& filter);
    void setStatusFilter(const std::string& filter);
    void clearFilters();
    
    // Queue functions
    void addToDownloadQueue(const std::string& datasetId, const std::string& name);
    void processDownloadQueue();
    void clearDownloadQueue();
    void removeFromQueue(int index);
    void retryQueueItem(int index);
    void clearCompletedFromQueue();
    void pauseQueueProcessing();
    bool isQueueProcessing() const { return queueProcessing.load(); }
    
    // Config persistence
    void loadUIConfig();
    void saveUIConfig();
    void loadDownloadQueue();
    void saveDownloadQueue();
    
private:
    // Collection methods
    void loadSourcesFromJSON();
    void loadHFTokenFromConfig();
    void saveSourceToJSON(const std::string& url);
    void updateCollectionStatus();
    void pollCollectionManager();
    void addLog(const std::string& message, int level = 0);
    void updateDatasetStats();
    void updateSourceList();
    void rebuildLeftPanel();  // Rebuild scrollbox contents
    void populateHFResults(float containerWidth);  // Populate HF results scrollbox with widgets
    void populateDownloadQueue();  // Populate download queue with widgets
    
    struct LogEntry {
        std::string timestamp;
        std::string message;
        int level = 0;
    };
    
    // Data collection manager (in-process)
    std::unique_ptr<GRIM::DataCollection::DataCollectionManager> collectionManager;
    
    // Hugging Face integration
    std::unique_ptr<GRIM::DataCollection::HuggingFaceWebhook> hfWebhook;
    std::vector<GRIM::DataCollection::HFDatasetInfo> hfSearchResults;
    int selectedHFDataset;
    
    // Status tracking
    bool collectionActive;
    bool collectionCompleted;
    float pollTimer;
    float pollInterval;
    float statsUpdateTimer;
    float statsUpdateInterval;
    
    // UI Components - Buttons
    std::shared_ptr<UIButton> startFullButton;
    std::shared_ptr<UIButton> startCollectButton;
    std::shared_ptr<UIButton> startVerifyButton;
    std::shared_ptr<UIButton> startMergeButton;
    std::shared_ptr<UIButton> forceRebuildButton;  // Rebuild ignoring dedup state
    std::shared_ptr<UIButton> stopButton;
    std::shared_ptr<UIButton> addSourceButton;
    std::shared_ptr<UIButton> refreshStatsButton;
    
    // Hugging Face UI buttons
    std::shared_ptr<UIButton> searchHFButton;
    std::shared_ptr<UIButton> searchHFCategoryButton;
    std::shared_ptr<UIButton> browseHFButton;
    
    // Filter UI buttons
    std::shared_ptr<UIButton> filterWebButton;
    std::shared_ptr<UIButton> filterHFButton;
    std::shared_ptr<UIButton> filterAllButton;
    std::shared_ptr<UIButton> clearFiltersButton;
    
    // Queue UI buttons
    std::shared_ptr<UIButton> processQueueButton;
    std::shared_ptr<UIButton> clearQueueButton;
    
    // UI Components - Input
    std::shared_ptr<UIInputBox> sourceUrlInput;
    std::string sourceUrlBuffer;
    
    std::shared_ptr<UIInputBox> hfSearchInput;
    std::string hfSearchBuffer;
    
    std::shared_ptr<UIDropdown> hfCategoryDropdown;
    
    std::shared_ptr<UIInputBox> hfTokenInput;
    std::string hfTokenBuffer;
    
    // UI Components - Sliders
    std::shared_ptr<UISlider> fetchLimitSlider;
    std::shared_ptr<UISlider> vocabSizeSlider;
    std::shared_ptr<UISlider> verificationThresholdSlider;
    std::shared_ptr<UISlider> maxHFResultsSlider;
    
    // UI Components - Progress
    std::shared_ptr<UIProgressBar> collectionProgressBar;
    float collectionAnimTime;
    
    // Layout boxes
    std::shared_ptr<UIVBox> buttonVBox;
    std::shared_ptr<UIScrollBox> leftPanelScrollBox;
    std::shared_ptr<UIScrollBox> hfResultsScrollBox;
    std::atomic<bool> hfResultsNeedsPopulate{false};

    
    // Status and stats
    std::string currentPhase;
    float currentProgress;
    std::string collectionMessage;
    int sourcesProcessed;
    int totalSources;
    int checkpointsCollected;
    
    // Dataset statistics
    std::string datasetSizeInfo;
    std::string checkpointStatsInfo;
    std::string verificationStatsInfo;
    std::string sourceListInfo;
    std::string hfSearchResultsInfo;
    std::string hfDownloadStatus;
    
    // Filter state
    std::string activeSourceFilter;  // "all", "web", "huggingface"
    std::string activeStatusFilter;  // "all", "pending", "completed", "failed"
    
    // Download queue
    struct QueuedDownload {
        std::string datasetId;
        std::string displayName;
        std::string status;  // "pending", "downloading", "completed", "failed"
        float progress;
        int retryCount = 0;
        std::string errorMessage;
    };
    std::vector<QueuedDownload> downloadQueue;
    std::mutex queueMutex;
    std::atomic<bool> queueProcessing{false};
    std::atomic<int> currentQueueIndex{-1};  // Currently processing item
    std::string queueStatusInfo;
    int hoveredQueueItem = -1;
    
    // Loading states
    std::atomic<bool> hfSearching{false};
    std::atomic<bool> hfDownloading{false};
    float searchAnimTime = 0.0f;
    std::string lastSearchError;
    
    // Configuration values
    int fetchLimit;
    int vocabSize;
    float verificationThreshold;
    int maxHFResults;
    
    // Logs
    std::vector<LogEntry> logEntries;
    std::mutex logMutex;
    size_t maxLogEntries;
    float logScrollPosition;
    bool autoScrollLogs;
    
    // Left panel scrolling (keeping manual implementation alongside scrollbox for now)
    float leftPanelScrollPosition;
    float leftPanelContentHeight;
};