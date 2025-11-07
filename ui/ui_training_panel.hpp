#pragma once
#include "ui_panel.hpp"
#include "ui_button.hpp"
#include "ui_slider.hpp"
#include "ui_progress_bar.hpp"
#include "ui_layout_box.hpp"
#include "ui_inputbox.hpp"
#include "ui_training_config.hpp"
#include "../resources/models/GRIM-text/training/control/training_control_client.hpp"
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
    GRIMText::Control::TrainingState currentState;
    GRIMText::TrainingStats currentStats;
    GRIMText::TrainingConfig currentConfig;
    
    bool serverConnected;
    std::string lastError;
    float pollTimer;
    float pollInterval;
    
    std::shared_ptr<UIButton> startButton;
    std::shared_ptr<UIButton> stopButton;
    std::shared_ptr<UIButton> pauseResumeButton;
    std::shared_ptr<UIButton> saveConfigButton;
    std::shared_ptr<UIButton> shutdownServerButton;
    std::shared_ptr<UIButton> addSourceButton;
    
    // Layout boxes for organizing buttons
    std::shared_ptr<UIVBox> buttonVBox;  // Vertical box for stacking 4 buttons
    
    // Source entry
    std::shared_ptr<UIInputBox> sourceUrlInput;
    std::string sourceUrlBuffer;
    void addDataSource(const std::string& url);
    void loadSourcesFromJSON();
    void saveSourceToJSON(const std::string& url);
    
    // Verification system
    std::shared_ptr<UIButton> runVerificationButton;
    std::string verificationStats;
    void runDataVerification();
    void updateVerificationStats();
    
    // Progress bar
    std::shared_ptr<UIProgressBar> trainingProgressBar;
    
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