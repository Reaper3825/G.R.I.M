#pragma once
//======================================================//
//  UITrainingPanel — Unified Model + Training Hub
//
//  Five tabs: Home | Knowledge Gaps | Tool Gaps | Training | Tokenizer
//
//  Merges the former ModelRegistry panel and Training panel
//  into a single DataHub-style tabbed interface.
//
//  - Home:           model browser (registry status, resource bars)
//  - Knowledge Gaps: gap queue (router misses → create model)
//  - Tool Gaps:      tool-gap proposals (ToolGapPlanner)
//  - Training:       hyperparameters, training controls, monitoring
//  - Tokenizer:      standalone tokenizer validation & encode
//======================================================//

#include "primitives/ui_panel.hpp"
#include "primitives/ui_button.hpp"
#include "primitives/ui_slider.hpp"
#include "primitives/ui_progress_bar.hpp"
#include "primitives/ui_layout_box.hpp"
#include "primitives/ui_inputbox.hpp"
#include "primitives/ui_dropdown.hpp"
#include "primitives/ui_label.hpp"
#include "primitives/ui_scrollbox.hpp"
#include "primitives/ui_graph.hpp"
#include "ui_training_config.hpp"
#include "control/training_controller.hpp"
#include "control/hyperparameter_registry.hpp"
#include "../MMO/Shared/MMD.hpp"
#include "../MMO/Core/ToolGapPlanner.hpp"
#include "../hardware/resource_values.hpp"
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <optional>
#include <atomic>
#include "DataCollection/dataset_target.hpp"

class OverlayRenderer;
struct InputState;

// ─────────────────────────────────────────────────────────
//  Knowledge gap entry (from router → gap queue)
// ─────────────────────────────────────────────────────────
struct KnowledgeGapEntry {
    std::string subject;
    std::vector<std::string> tags;
    std::string original_context;
    std::string request_id;
    std::chrono::steady_clock::time_point timestamp;
};

// ─────────────────────────────────────────────────────────
//  View enum — DataHub-style tabs
// ─────────────────────────────────────────────────────────
enum class TrainingPanelTab : uint8_t {
    Home          = 0,
    KnowledgeGaps = 1,
    ToolGaps      = 2,
    Training      = 3,
    Tokenizer     = 4
};

// ─────────────────────────────────────────────────────────
//  UITrainingPanel — Unified Model + Training Hub
// ─────────────────────────────────────────────────────────
class UITrainingPanel : public UIPanel {
public:
    UITrainingPanel();
    ~UITrainingPanel() override;

    void update(const InputState& input, float dt) override;
    bool drawOverlay(OverlayRenderer& renderer) override;

    // --- View control (DataHub pattern) -------------------------
    void setView(TrainingPanelTab tab);
    TrainingPanelTab currentView() const { return activeTab_; }

    // --- Public training control (used by commands_training.cpp) ─
    void startTrainingSession();
    void stopTrainingSession();
    void pauseTrainingSession();
    void resumeTrainingSession();
    void shutdownTrainingServer();

    // --- Knowledge gap intake -----------------------------------
    void pushKnowledgeGap(KnowledgeGapEntry entry);
    size_t pendingGapCount() const;

    // --- Tool gap intake ----------------------------------------
    void pushToolGap(GRIM::MMO::ToolGapProposal proposal);
    size_t pendingToolGapCount() const;

    // --- Model browser ------------------------------------------
    void refreshModelList();

    // --- Model creator (pre-fill from gap) ----------------------
    void prefillCreatorFromGap(const KnowledgeGapEntry& gap);

    // --- Focus query --------------------------------------------
    bool isAnySliderEditing() const;
    bool isAnyInputEditing() const;

private:
    // ═════════════════════════════════════════════════════
    //  Tab state
    // ═════════════════════════════════════════════════════
    TrainingPanelTab activeTab_ = TrainingPanelTab::Home;

    std::shared_ptr<UIButton> tabHomeBtn_;
    std::shared_ptr<UIButton> tabKnowledgeGapsBtn_;
    std::shared_ptr<UIButton> tabToolGapsBtn_;
    std::shared_ptr<UIButton> tabTrainingBtn_;
    std::shared_ptr<UIButton> tabTokenizerBtn_;

    // ═════════════════════════════════════════════════════
    //  Home tab — Model Browser
    // ═════════════════════════════════════════════════════
    struct ModelListEntry {
        std::string id;
        std::string name;
        std::string subject;
        std::string backend;
        std::string status;
        uint32_t    statusColor = 0xFFFFFFFF;
        long        ram_mb  = 0;
        long        vram_mb = 0;
        bool        is_router = false;
    };

    std::vector<ModelListEntry> modelEntries_;
    std::shared_ptr<UIScrollBox> browserScrollBox_;
    std::shared_ptr<UIProgressBar> vramBar_;
    std::shared_ptr<UIProgressBar> ramBar_;
    float browserRefreshTimer_ = 0.0f;
    float browserRefreshInterval_ = 2.0f;
    int   hoveredBrowserRow_ = -1;

    static constexpr float kRowHeight  = 20.0f;

    // Model Creator (sub-view of Home tab)
    bool showCreatorForm_ = false;
    std::shared_ptr<UIButton> createModelBtn_;

    std::shared_ptr<UIInputBox> creatorNameInput_;
    std::shared_ptr<UIInputBox> creatorParamInput_;
    std::shared_ptr<UIInputBox> creatorSubjectInput_;
    std::shared_ptr<UIInputBox> creatorTagsInput_;
    std::shared_ptr<UIInputBox> creatorDescInput_;
    std::shared_ptr<UIButton>   creatorRegisterBtn_;
    std::shared_ptr<UIButton>   creatorCancelBtn_;
    std::shared_ptr<UILabel>    creatorStatusLabel_;

    std::string bufId_;
    std::string bufName_;
    std::string bufParamStr_;
    std::string bufSubject_;
    std::string bufTags_;
    std::string bufDesc_;

    float totalVramMb_ = 0.0f;
    float usedVramMb_  = 0.0f;
    float totalRamMb_  = 0.0f;
    float usedRamMb_   = 0.0f;

    void drawHomeTab(OverlayRenderer& renderer, const PanelRect& content);
    void drawBrowserView(OverlayRenderer& renderer, const PanelRect& content);
    void processBrowserClicks(const InputState& input, const PanelRect& content);
    void handleModelAction(const std::string& model_id, const std::string& action);
    void drawCreatorForm(OverlayRenderer& renderer, const PanelRect& content);
    void clearCreatorFields();
    bool validateCreatorFields(std::string& out_error) const;
    void submitNewModel();
    void regenerateId();
    static std::string formatParamCount(int64_t params);
    static int64_t parseParamStr(const std::string& s);
    static std::string slugifyName(const std::string& name);
    static int nextVersionForBase(const std::string& base);
    void updateResourceBars();
    bool persistSubModel(const GRIM::MMO::ModelInfo& model);
    bool removeSubModelFromConfig(const std::string& model_id);

    // ═════════════════════════════════════════════════════
    //  Knowledge Gaps tab
    // ═════════════════════════════════════════════════════
    std::vector<KnowledgeGapEntry> gapQueue_;
    mutable std::mutex gapMutex_;
    std::shared_ptr<UIScrollBox> gapScrollBox_;
    int hoveredGapRow_ = -1;

    void drawKnowledgeGapsTab(OverlayRenderer& renderer, const PanelRect& content);
    void processGapClicks(const InputState& input, const PanelRect& content);
    void dismissGap(size_t index);
    void createFromGap(size_t index);

    // ═════════════════════════════════════════════════════
    //  Tool Gaps tab
    // ═════════════════════════════════════════════════════
    std::vector<GRIM::MMO::ToolGapProposal> toolGapQueue_;
    mutable std::mutex toolGapMutex_;
    std::shared_ptr<UIScrollBox> toolGapScrollBox_;
    int hoveredToolGapRow_ = -1;

    void drawToolGapsTab(OverlayRenderer& renderer, const PanelRect& content);
    void processToolGapClicks(const InputState& input, const PanelRect& content);

    // ═════════════════════════════════════════════════════
    //  Training tab
    // ═════════════════════════════════════════════════════

    // Training controller
    std::unique_ptr<GRIM::UI::UITrainingController> trainingController;
    GRIMText::TrainingState currentState;
    GRIMText::TrainingStats currentStats;
    GRIMText::TrainingConfig currentConfig;

    bool serverConnected;
    bool serverStarting;
    std::string lastError;
    std::string checkpointMergeStatus;
    float pollTimer;
    float pollInterval;

    // Configuration sliders
    std::shared_ptr<UISlider> epochsSlider;
    std::shared_ptr<UISlider> batchSizeSlider;
    std::shared_ptr<UISlider> learningRateSlider;
    std::shared_ptr<UISlider> maxSeqLenSlider;
    std::shared_ptr<UISlider> warmupStepsSlider;

    // Config buttons
    std::shared_ptr<UIButton> saveConfigButton;

    // Action buttons (bottom bar)
    std::shared_ptr<UIButton> startButton;
    std::shared_ptr<UIButton> stopButton;
    std::shared_ptr<UIButton> pauseResumeButton;
    std::shared_ptr<UIButton> resetStatusButton;
    std::shared_ptr<UIButton> closeButton;
    std::shared_ptr<UIButton> runTokenizerButton;

    // Curriculum + Model selection
    std::shared_ptr<UIDropdown> curriculumDropdown_;
    std::shared_ptr<UIDropdown> trainModelDropdown_;
    std::string selectedCurriculumId_;
    std::string selectedTrainModelId_;
    std::unique_ptr<DatasetTarget> trainingDatasetTarget_;

    std::string vocabPathBuffer;
    std::string modelPathBuffer;
    std::string checkpointsPathBuffer;
    std::string logsPathBuffer;

    // Progress bars
    std::shared_ptr<UIProgressBar> trainingProgressBar;

    // System resource monitoring
    std::shared_ptr<UIGraph> resourceMonitorGraph;
    float resourceSampleTimer;
    float resourceSampleInterval;
    std::vector<DataPoint> cpuHistory;
    std::vector<DataPoint> memoryHistory;
    std::vector<DataPoint> gpuHistory;
    int resourceSampleCount;
    int maxResourceSamples;

    // Loss tracking
    struct LossPoint {
        int step;
        float loss;
    };
    std::vector<LossPoint> lossHistory;
    size_t maxLossHistory;

    // Dataset tracking
    std::string datasetSizeInfo;
    std::string checkpointStatsInfo;
    float datasetUpdateTimer;
    float datasetUpdateInterval;

    struct DatasetSnapshotResult {
        std::string datasetInfo;
        std::string checkpointInfo;
    };
    std::atomic<bool> datasetSnapshotInFlight{false};
    std::mutex datasetSnapshotMutex;
    std::optional<DatasetSnapshotResult> pendingDatasetSnapshot;

    // Hardware info
    std::string hardwareInfo;
    std::string estimatedTimeStr;
    float estimatedTrainingTimeSeconds = 0.0f;

    // Logs
    struct LogEntry {
        std::string timestamp;
        std::string message;
        int level = 0;
    };
    std::vector<LogEntry> logEntries;
    std::mutex logMutex;
    size_t maxLogEntries;
    float logScrollPosition;
    bool autoScrollLogs;

    // Training tab draw methods
    void drawTrainingTab(OverlayRenderer& renderer, const PanelRect& content);
    void drawBottomBar(OverlayRenderer& renderer, float barY, float barWidth, float barX);
    void drawStatCard(OverlayRenderer& renderer, const Vec2& pos, const Vec2& sz,
                      const std::string& label, const std::string& value, uint32_t accentColor);

    // Training internals
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
    void loadPathsFromConfig();
    void savePathsToConfig();
    void refreshCurriculumDropdown();
    void refreshModelDropdown();
    void refreshTrainingDropdowns();
    void updateResourceMonitoring(float dt);
    void updateHardwareInfo();
    void calculateTrainingEstimate();
    void updateDatasetSize();
    void updateCheckpointStats();
    std::string readDatasetSizeSnapshot();
    std::string readCheckpointStatsSnapshot();
    void requestDatasetSnapshot();
    void applyPendingDatasetSnapshot();

    // Callbacks
    void onProgressUpdate(const GRIMText::TrainingStats& stats);
    void onStateChange(GRIMText::TrainingState oldState, GRIMText::TrainingState newState);
    void onError(const std::string& error);

    // Logging
    void addLog(const std::string& message, int level);
    std::string getStateString(GRIMText::TrainingState state) const;
    uint32_t getStateColor(GRIMText::TrainingState state) const;

    // Tokenizer runner state
    bool tokenizerRunning_ = false;
    bool tokenizerComplete_ = false;
    bool tokenizerSuccess_ = false;
    std::string tokenizerStatusMessage_;
    GRIMText::TrainingControlClient::TokenizerResult lastTokenizerResult_;
    void handleRunTokenizer();
    void drawTokenizerStatus(OverlayRenderer& renderer, float x, float y, float width);

    // ═════════════════════════════════════════════════════
    //  Tokenizer tab
    // ═════════════════════════════════════════════════════

    // Encode state
    std::string encodeInputBuffer_;
    std::shared_ptr<UIInputBox> encodeInputBox_;
    std::shared_ptr<UIButton> encodeButton_;
    std::shared_ptr<UIButton> clearEncodeButton_;
    std::shared_ptr<UIButton> tokenizerRunValidationBtn_;
    std::shared_ptr<UIButton> tokenizerCloseBtn_;
    std::shared_ptr<UIScrollBox> tokenizerScrollBox_;
    std::atomic<bool> encodeRunning_{false};
    bool encodeComplete_ = false;
    bool encodeSuccess_ = false;
    std::string encodeErrorMessage_;
    GRIMText::TrainingControlClient::EncodeResult lastEncodeResult_;
    std::mutex encodeMutex_;

    // Tokenizer tab draw methods
    void drawTokenizerTab(OverlayRenderer& renderer, const PanelRect& content);
    void drawTokenizerBottomBar(OverlayRenderer& renderer, float barY, float barWidth, float barX);
    void drawEncodeResults(OverlayRenderer& renderer, float x, float y, float width, float maxHeight);
    void handleEncodeText();

    // Config scroll
    float configScrollOffset = 0.0f;
    float configContentHeight = 0.0f;

    // ═════════════════════════════════════════════════════
    //  Hyperparameter Registry (filterable param browser)
    // ═════════════════════════════════════════════════════
    GRIM::Config::HyperparameterRegistry hyperparamRegistry_;
    GRIM::Config::TrainingHyperparameters hyperparamSnapshot_;
    bool hyperparamsLoaded_ = false;

    std::shared_ptr<UIDropdown> paramCategoryFilter_;
    std::shared_ptr<UIScrollBox> paramScrollBox_;
    float paramScrollOffset_ = 0.0f;
    int selectedParamCategory_ = 0;        // 0 = "All"
    int hoveredParamRow_ = -1;
    bool hoveredParamScrollbar_ = false;
    bool draggingParamScrollbar_ = false;
    float paramScrollbarDragStartY_ = 0.0f;
    float paramScrollbarDragStartOffset_ = 0.0f;

    // Inline editing state
    int editingParamIndex_ = -1;              // index into current filtered list (-1 = none)
    std::string editParamBuffer_;             // backing string for the input box
    std::shared_ptr<UIInputBox> paramEditInput_;
    bool paramEditDirty_ = false;             // true when a value was changed (pending save)

    void loadHyperparamSnapshot();
    void drawParamBrowser(OverlayRenderer& renderer, const Vec2& origin, const Vec2& sz);
    void processParamBrowserClicks(const InputState& input);
    void commitParamEdit();
    void cancelParamEdit();
    bool persistHyperparamToJSON(const GRIM::Config::HyperparamEntry& entry);
};
