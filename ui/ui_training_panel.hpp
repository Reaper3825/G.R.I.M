#pragma once
#include "ui_panel.hpp"
#include "ui_button.hpp"
#include "ui_slider.hpp"
#include "ui_progress_bar.hpp"
#include "ui_layout_box.hpp"
#include "ui_inputbox.hpp"
#include "ui_training_config.hpp"
#include "../control/training_control_client.hpp"
#include "../control/data_collection_client.hpp"
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <chrono>

class OverlayRenderer;
struct InputState;

class UITrainingPanel : public UIPanel {
public:
    UITrainingPanel();
    ~UITrainingPanel() override;

    void update(const InputState& input, float dt) override;
    void drawOverlay(OverlayRenderer& renderer) override;
    
    // Public control functions for commands
    void startTrainingSession();
    void stopTrainingSession();
    void pauseTrainingSession();
    void resumeTrainingSession();
    void shutdownTrainingServer();
    
    // Check if any slider is currently editing text
    bool isAnySliderEditing() const;

private:
    void initializeClient();
    void resetState();
    void pollServer();
    void pollDataCollectionServerAsync();  // Async polling to avoid blocking main thread
    void handleStartTraining();
    void handleStopTraining();
    void handlePauseResume();
    void loadConfigFromJSON();
    void saveConfigToJSON();
    void updateConfigFromSliders();
    void updateSlidersFromConfig();
    
    struct LogEntry {
        std::string timestamp;
        std::string message;
        int level = 0;
    };
    
    void addLog(const std::string& message, int level);
    std::string getStateString(GRIMText::Control::TrainingState state) const;
    uint32_t getStateColor(GRIMText::Control::TrainingState state) const;
    
    std::unique_ptr<GRIMText::TrainingControlClient> client;
    std::unique_ptr<GRIM::DataCollection::DataCollectionClient> dataCollectionClient;
    GRIMText::Control::TrainingState currentState;
    GRIMText::TrainingStats currentStats;
    GRIMText::TrainingConfig currentConfig;
    
    bool serverConnected;
    bool serverStarting;  // Flag to prevent duplicate server starts
    bool dataCollectionServerConnected;  // Connection status for data collection server
    bool dataCollectionActive;  // Flag to track if data collection is in progress
    bool dataCollectionCompleted;  // Flag to track if current collection has completed (prevents re-logging)
    bool pipelineRequestPending;  // Flag to track if a pipeline request is in flight
    bool firstPollDone;  // Flag to detect first poll after connection for stale state detection
    float collectionStuckTimer;  // Timer to detect stuck collection operations
    float lastCollectionProgress;  // Last recorded collection progress
    std::string lastError;
    std::string checkpointMergeStatus;  // Status message for checkpoint merge operations
    float pollTimer;
    float pollInterval;
    float dataCollectionPollTimer;  // Separate poll timer for data collection server
    float dataCollectionPollInterval;  // Poll interval for data collection (500ms to reduce load)
    std::future<void> dataCollectionPollFuture;  // Async polling to avoid blocking main thread
    std::atomic<bool> dataCollectionPollInProgress;  // Flag to prevent overlapping polls
    
    std::shared_ptr<UIButton> startButton;
    std::shared_ptr<UIButton> stopButton;
    std::shared_ptr<UIButton> pauseResumeButton;
    std::shared_ptr<UIButton> saveConfigButton;
    std::shared_ptr<UIButton> shutdownServerButton;
    std::shared_ptr<UIButton> resetStatusButton;
    std::shared_ptr<UIButton> addSourceButton;
    
    // Layout boxes for organizing buttons
    std::shared_ptr<UIVBox> buttonVBox;  // Vertical box for stacking 4 buttons
    
    // Source entry
    std::shared_ptr<UIInputBox> sourceUrlInput;
    std::string sourceUrlBuffer;
    void addDataSource(const std::string& url);
    void loadSourcesFromJSON();
    void saveSourceToJSON(const std::string& url);
    
    // Unified data pipeline system (collect → verify → merge)
    std::shared_ptr<UIButton> collectDataButton;
    void startDataCollection();
    std::string verificationStats;
    void updateVerificationStats();
    
    // Data size tracking
    std::string datasetSizeInfo;
    void updateDatasetSize();
    
    // Progress bars
    std::shared_ptr<UIProgressBar> trainingProgressBar;
    std::shared_ptr<UIProgressBar> collectionProgressBar;
    float collectionAnimTime;  // For animated progress during collection
    
    // Hardware info and estimation
    std::string hardwareInfo;
    std::string estimatedTimeStr;
    float estimatedTrainingTimeSeconds;
    
    void updateHardwareInfo();
    void calculateTrainingEstimate();
    
    // Configuration sliders
    std::shared_ptr<UISlider> epochsSlider;
    std::shared_ptr<UISlider> batchSizeSlider;
    std::shared_ptr<UISlider> learningRateSlider;
    std::shared_ptr<UISlider> maxSeqLenSlider;
    std::shared_ptr<UISlider> warmupStepsSlider;
    
    std::vector<LogEntry> logEntries;
    std::mutex logMutex;
    size_t maxLogEntries;
    float logScrollPosition;
    bool autoScrollLogs;
    
    // Left panel scrolling
    float leftPanelScrollPosition;
    float leftPanelContentHeight;
    bool leftPanelScrolling;
    Vec2 leftPanelScrollStartPos;
    
    struct LossPoint {
        int step;
        float loss;
    };
    std::vector<LossPoint> lossHistory;
    size_t maxLossHistory;
};