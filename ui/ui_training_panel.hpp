#pragma once
#include "ui_panel.hpp"
#include "ui_button.hpp"
#include "ui_slider.hpp"
#include "ui_progress_bar.hpp"
#include "ui_layout_box.hpp"
#include "ui_inputbox.hpp"
#include "ui_graph.hpp"
#include "ui_training_config.hpp"
#include "control/training_controller.hpp"
#include "DataCollection/data_collection_manager.hpp"
#include "hardware/resource_values.hpp"
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <optional>
#include <atomic>

class OverlayRenderer;
struct InputState;

class UITrainingPanel : public UIPanel {
public:
    UITrainingPanel();
    ~UITrainingPanel() override;

    void update(const InputState& input, float dt) override;
    bool drawOverlay(OverlayRenderer& renderer) override;
    
    // Public control functions for commands
    void startTrainingSession();
    void stopTrainingSession();
    void pauseTrainingSession();
    void resumeTrainingSession();
    void shutdownTrainingServer();
    
    // Check if any slider is currently editing text
    bool isAnySliderEditing() const;

private:
    void initializeController();
    void setupCallbacks();
    void resetState();
    void pollServer();
    void handleStartTraining();
    void handleStopTraining();
    void handlePauseResume();
    void loadConfigFromJSON();
    void saveConfigToJSON();
    void updateConfigFromSliders();
    void updateSlidersFromConfig();
    
    // Callbacks for training controller
    void onProgressUpdate(const GRIMText::TrainingStats& stats);
    void onStateChange(GRIMText::TrainingState oldState, GRIMText::TrainingState newState);
    void onError(const std::string& error);
    
    struct LogEntry {
        std::string timestamp;
        std::string message;
        int level = 0;
    };
    
    void addLog(const std::string& message, int level);
    std::string getStateString(GRIMText::TrainingState state) const;
    uint32_t getStateColor(GRIMText::TrainingState state) const;
    
    // Training controller (replaces direct client usage)
    std::unique_ptr<GRIM::UI::UITrainingController> trainingController;
    
    GRIMText::TrainingState currentState;
    GRIMText::TrainingStats currentStats;
    GRIMText::TrainingConfig currentConfig;
    
    // Data collection manager (in-process)
    std::unique_ptr<GRIM::DataCollection::DataCollectionManager> collectionManager;
    
    bool serverConnected;
    bool serverStarting;  // Flag to prevent duplicate server starts
    bool dataCollectionActive;  // Flag to track if data collection is in progress
    bool dataCollectionCompleted;  // Flag to track if current collection has completed (prevents re-logging)
    bool pipelineRequestPending;  // Flag to track if a pipeline request is in flight
    std::string lastError;
    std::string checkpointMergeStatus;  // Status message for checkpoint merge operations
    float pollTimer;
    float pollInterval;
    
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
    
    // GRIM-text path configuration - only training data path shown in UI
    std::shared_ptr<UIInputBox> trainingDataPathInput;
    std::string vocabPathBuffer;
    std::string modelPathBuffer;
    std::string trainingDataPathBuffer;
    std::string checkpointsPathBuffer;
    std::string logsPathBuffer;
    void loadPathsFromConfig();
    void savePathsToConfig();
    
    // Data size tracking
    std::string datasetSizeInfo;
    std::string checkpointStatsInfo;  // Info about collected checkpoints
    float datasetUpdateTimer;  // Timer to throttle expensive file operations
    float datasetUpdateInterval;  // Update interval (e.g., 2.0 seconds)
    void updateDatasetSize();
    void updateCheckpointStats();
    std::string readDatasetSizeSnapshot();
    std::string readCheckpointStatsSnapshot();
    void requestDatasetSnapshot();
    void applyPendingDatasetSnapshot();
    
    struct DatasetSnapshotResult {
        std::string datasetInfo;
        std::string checkpointInfo;
    };
    std::atomic<bool> datasetSnapshotInFlight{false};
    std::mutex datasetSnapshotMutex;
    std::optional<DatasetSnapshotResult> pendingDatasetSnapshot;
    
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
    
    // System resource monitoring
    std::shared_ptr<UIGraph> resourceMonitorGraph;
    float resourceSampleTimer;
    float resourceSampleInterval;  // Sample every 0.5 seconds
    std::vector<DataPoint> cpuHistory;
    std::vector<DataPoint> memoryHistory;
    std::vector<DataPoint> gpuHistory;
    int resourceSampleCount;
    int maxResourceSamples;
    
    void updateResourceMonitoring(float dt);
};
