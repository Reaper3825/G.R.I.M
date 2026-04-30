// ============================================================
// UITrainingPanel — Unified Model + Training Hub
//
// Four tabs (DataHub-style):
//   Home:           model browser, resource bars, model creator
//   Knowledge Gaps: gap queue from router misses
//   Tool Gaps:      tool-gap proposals from ToolGapPlanner
//   Training:       hyperparameters, monitoring, logs, controls
//
// Bottom action bar with training session controls.
// ============================================================
#include <sstream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <ctime>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
#include "core/grim_platform.h"
#include "core/input_parser.hpp"
#include <filesystem>
#include "ui_training_panel.hpp"
#include "ui_slider.hpp"
#include "ui_graph.hpp"
#include "overlay_renderer.hpp"
#include "ui_theme.hpp"
#include "ui_draw_helpers.hpp"
#include "logger.hpp"
#include "../MMO/Core/HardwareInventory.hpp"
#include "../MMO/Core/ModelRegistry.hpp"
#include "../MMO/Core/ModelLoader.hpp"
#include "../MMO/Core/ResourceSignal.hpp"
#include "ai/training_server_manager.hpp"
#include "resources.hpp"
#include "resources/models/GRIM-text/Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "DataCollection/dataset_target.hpp"

using namespace GRIMText;
using namespace UITheme;

extern GRIM::MMO::HardwareInventory g_hardwareInventory;
extern GRIM::MMO::ModelLoader*    g_modelLoader;
extern GRIM::MMO::ResourceSignal* g_resourceSignal;

// Layout constants
static constexpr float kTabBarY       = 35.0f;
static constexpr float kContentTopY   = 68.0f;
static constexpr float kBottomBarH    = 50.0f;
static constexpr float kStatCardH     = 60.0f;
static constexpr float kStatCardGap   = 10.0f;
static constexpr float kLogLineH      = 18.0f;

namespace {

constexpr float kParamBrowserWheelPixelsPerStep = 22.0f;
constexpr float kParamBrowserScrollbarWidth = 12.0f;
constexpr float kParamBrowserScrollbarInset = 2.0f;
constexpr float kParamBrowserScrollbarMinThumbH = 24.0f;

struct ParamScrollbarMetrics {
    bool visible = false;
    float trackX = 0.0f;
    float trackY = 0.0f;
    float trackW = 0.0f;
    float trackH = 0.0f;
    float thumbY = 0.0f;
    float thumbH = 0.0f;
    float maxScroll = 0.0f;
    float maxThumbTravel = 0.0f;
};

static float normalizeMouseWheelDelta(float rawDelta) {
    if (std::fabs(rawDelta) >= 120.0f) {
        return rawDelta / 120.0f;
    }
    return rawDelta;
}

static bool pointInRect(const Vec2& point, float x, float y, float w, float h) {
    return point.x >= x && point.x <= x + w && point.y >= y && point.y <= y + h;
}

static size_t countParamCategoryHeaderRows(
    const std::vector<const GRIM::Config::HyperparamEntry*>& params,
    const std::string& activeCategory)
{
    if (!activeCategory.empty()) {
        return 0;
    }

    size_t headerCount = 0;
    std::string lastCategory;
    for (const auto* entry : params) {
        if (!entry) continue;
        if (entry->category != lastCategory) {
            lastCategory = entry->category;
            ++headerCount;
        }
    }
    return headerCount;
}

static float computeParamBrowserContentHeight(
    const std::vector<const GRIM::Config::HyperparamEntry*>& params,
    const std::string& activeCategory,
    float rowH)
{
    const size_t headerRows = countParamCategoryHeaderRows(params, activeCategory);
    return static_cast<float>(params.size() + headerRows) * rowH;
}

static ParamScrollbarMetrics computeParamScrollbarMetrics(
    float x,
    float y,
    float w,
    float listH,
    float totalContentH,
    float scrollOffset)
{
    ParamScrollbarMetrics metrics;
    metrics.maxScroll = std::max(0.0f, totalContentH - listH);
    metrics.visible = metrics.maxScroll > 0.0f;
    metrics.trackW = kParamBrowserScrollbarWidth;
    metrics.trackX = x + w - metrics.trackW - kParamBrowserScrollbarInset;
    metrics.trackY = y + kParamBrowserScrollbarInset;
    metrics.trackH = std::max(0.0f, listH - 2.0f * kParamBrowserScrollbarInset);

    if (!metrics.visible || metrics.trackH <= 0.0f) {
        metrics.thumbY = metrics.trackY;
        metrics.thumbH = metrics.trackH;
        return metrics;
    }

    const float visibleRatio = listH / totalContentH;
    metrics.thumbH = std::clamp(metrics.trackH * visibleRatio,
                                kParamBrowserScrollbarMinThumbH,
                                metrics.trackH);
    metrics.maxThumbTravel = std::max(0.0f, metrics.trackH - metrics.thumbH);

    const float scrollRatio = (metrics.maxScroll > 0.0f)
        ? std::clamp(scrollOffset / metrics.maxScroll, 0.0f, 1.0f)
        : 0.0f;
    metrics.thumbY = metrics.trackY + scrollRatio * metrics.maxThumbTravel;
    return metrics;
}

} // namespace

// =========================================================
// Helpers
// =========================================================

static std::string backendDisplayName(GRIM::MMO::BackendType bt) {
    return GRIM::MMO::ModelRegistry::backendTypeToString(bt);
}

static uint32_t residencyStatusColor(GRIM::MMO::ResidencyState state) {
    switch (state) {
        case GRIM::MMO::ResidencyState::Unloaded:      return Colors::TextMuted;
        case GRIM::MMO::ResidencyState::Loading:       return Colors::Warning;
        case GRIM::MMO::ResidencyState::Loaded:        return Colors::Success;
        case GRIM::MMO::ResidencyState::InUse:         return Colors::AccentBlue;
        case GRIM::MMO::ResidencyState::Idle:          return Colors::Info;
        case GRIM::MMO::ResidencyState::EvictEligible: return Colors::Warning;
        case GRIM::MMO::ResidencyState::Unloading:     return Colors::Danger;
    }
    return Colors::TextPrimary;
}

static std::vector<std::string> splitCommaTags(const std::string& input) {
    std::vector<std::string> tags;
    std::istringstream ss(input);
    std::string tag;
    while (std::getline(ss, tag, ',')) {
        size_t start = tag.find_first_not_of(" \t");
        size_t end   = tag.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos)
            tags.push_back(tag.substr(start, end - start + 1));
    }
    return tags;
}

static const std::vector<std::string> kBackendOptions = {
    "grim_text_server", "llama_cpp", "ollama", "external"
};

// ============================================================
// Constructor
// ============================================================

UITrainingPanel::UITrainingPanel()
    : UIPanel("GRIM-text Training Control", true),
      currentState(Control::TrainingState_Idle),
      serverConnected(false),
      serverStarting(false),
      pollTimer(0.0f),
      pollInterval(0.2f),
      maxLogEntries(1000),
      logScrollPosition(0.0f),
      autoScrollLogs(true),
      maxLossHistory(500),
      datasetUpdateTimer(0.0f),
      datasetUpdateInterval(2.0f)
{
    position = { 200, 80 };
    size     = { 1100, 800 };
    setVisible(false);
    setBackground(Colors::PanelBg);

    if (position.y < 50.0f) position.y = 50.0f;

    resetState();
    loadConfigFromJSON();

    // ── Training controller ──
    std::string host = TrainingConfigManager::getServerHost();
    int port = TrainingConfigManager::getServerPort();
    try {
        trainingController = std::make_unique<GRIM::UI::UITrainingController>(host, port);
        setupCallbacks();
    } catch (const std::exception& e) {
        LOG_ERROR("UITrainingPanel", std::string("Controller init failed: ") + e.what());
        lastError = std::string("Controller init failed: ") + e.what();
    }

    // ══════════════════════════════════════════════════════
    //  Tab buttons (DataHub pattern)
    // ══════════════════════════════════════════════════════

    tabHomeBtn_ = std::make_shared<UIButton>("Home", [this]() {
        setView(TrainingPanelTab::Home);
    });
    tabHomeBtn_->setSize(90.0f, 28.0f);

    tabKnowledgeGapsBtn_ = std::make_shared<UIButton>("Knowledge Gaps", [this]() {
        setView(TrainingPanelTab::KnowledgeGaps);
    });
    tabKnowledgeGapsBtn_->setSize(120.0f, 28.0f);

    tabToolGapsBtn_ = std::make_shared<UIButton>("Tool Gaps", [this]() {
        setView(TrainingPanelTab::ToolGaps);
    });
    tabToolGapsBtn_->setSize(90.0f, 28.0f);

    tabTrainingBtn_ = std::make_shared<UIButton>("Training", [this]() {
        setView(TrainingPanelTab::Training);
    });
    tabTrainingBtn_->setSize(90.0f, 28.0f);

    tabTokenizerBtn_ = std::make_shared<UIButton>("Tokenizer", [this]() {
        setView(TrainingPanelTab::Tokenizer);
    });
    tabTokenizerBtn_->setSize(90.0f, 28.0f);

    // ══════════════════════════════════════════════════════
    //  Home tab — Model Browser widgets
    // ══════════════════════════════════════════════════════

    browserScrollBox_ = std::make_shared<UIScrollBox>();
    browserScrollBox_->setSize(1060.0f, 500.0f);

    vramBar_ = std::make_shared<UIProgressBar>("VRAM", 1.0f);
    vramBar_->setSize(500.0f, 20.0f);
    vramBar_->setShowPercentage(true);
    vramBar_->setFillColor(Colors::AccentBlue);
    vramBar_->setBackgroundColor(Colors::ContentAreaBg);

    ramBar_ = std::make_shared<UIProgressBar>("RAM", 1.0f);
    ramBar_->setSize(500.0f, 20.0f);
    ramBar_->setShowPercentage(true);
    ramBar_->setFillColor(Colors::Success);
    ramBar_->setBackgroundColor(Colors::ContentAreaBg);

    createModelBtn_ = std::make_shared<UIButton>("+ New Model", [this]() {
        showCreatorForm_ = !showCreatorForm_;
    });
    createModelBtn_->setSize(120.0f, 30.0f);

    // Model Creator form widgets
    creatorNameInput_ = std::make_shared<UIInputBox>(&bufName_);
    creatorNameInput_->setPlaceholder("Display name");
    creatorNameInput_->setSize(400.0f, 26.0f);

    bufParamStr_ = "0M";
    creatorParamInput_ = std::make_shared<UIInputBox>(&bufParamStr_);
    creatorParamInput_->setPlaceholder("Parameters (e.g. 7B, 405M)");
    creatorParamInput_->setSize(200.0f, 26.0f);

    creatorSubjectInput_ = std::make_shared<UIInputBox>(&bufSubject_);
    creatorSubjectInput_->setPlaceholder("Subject (e.g. medical)");
    creatorSubjectInput_->setSize(400.0f, 26.0f);

    creatorTagsInput_ = std::make_shared<UIInputBox>(&bufTags_);
    creatorTagsInput_->setPlaceholder("Tags (comma-separated)");
    creatorTagsInput_->setSize(400.0f, 26.0f);

    creatorDescInput_ = std::make_shared<UIInputBox>(&bufDesc_);
    creatorDescInput_->setPlaceholder("Description");
    creatorDescInput_->setSize(400.0f, 26.0f);

    creatorStatusLabel_ = std::make_shared<UILabel>("", Colors::TextPrimary);
    creatorStatusLabel_->setSize(400.0f, 20.0f);

    creatorRegisterBtn_ = std::make_shared<UIButton>("Register", [this]() {
        submitNewModel();
    });
    creatorRegisterBtn_->setSize(100.0f, 30.0f);

    creatorCancelBtn_ = std::make_shared<UIButton>("Cancel", [this]() {
        clearCreatorFields();
        showCreatorForm_ = false;
    });
    creatorCancelBtn_->setSize(100.0f, 30.0f);

    // ══════════════════════════════════════════════════════
    //  Knowledge Gap queue scroll box
    // ══════════════════════════════════════════════════════

    gapScrollBox_ = std::make_shared<UIScrollBox>();
    gapScrollBox_->setSize(1060.0f, 600.0f);

    // ══════════════════════════════════════════════════════
    //  Tool Gap queue scroll box
    // ══════════════════════════════════════════════════════

    toolGapScrollBox_ = std::make_shared<UIScrollBox>();
    toolGapScrollBox_->setSize(1060.0f, 600.0f);

    // ══════════════════════════════════════════════════════
    //  Training tab widgets
    // ══════════════════════════════════════════════════════

    epochsSlider = std::make_shared<UISlider>("Epochs", 1.0f, 50.0f,
        static_cast<float>(currentConfig.epochs),
        [this](float v) { currentConfig.epochs = static_cast<int>(v); calculateTrainingEstimate(); }, 1.0f);

    batchSizeSlider = std::make_shared<UISlider>("Batch Size", 1.0f, 128.0f,
        static_cast<float>(currentConfig.batchSize),
        [this](float v) { currentConfig.batchSize = static_cast<int>(v); calculateTrainingEstimate(); }, 1.0f);

    learningRateSlider = std::make_shared<UISlider>("Learning Rate", 0.000001f, 0.01f,
        currentConfig.learningRate,
        [this](float v) { currentConfig.learningRate = v; calculateTrainingEstimate(); }, 0.000001f);

    maxSeqLenSlider = std::make_shared<UISlider>("Max Seq Length", 512.0f, 16384.0f,
        static_cast<float>(currentConfig.maxSeqLen),
        [this](float v) { currentConfig.maxSeqLen = static_cast<int>(v); calculateTrainingEstimate(); }, 128.0f);

    warmupStepsSlider = std::make_shared<UISlider>("Warmup Steps", 0.0f, 5000.0f,
        static_cast<float>(currentConfig.warmupSteps),
        [this](float v) { currentConfig.warmupSteps = static_cast<int>(v); calculateTrainingEstimate(); }, 10.0f);

    saveConfigButton = std::make_shared<UIButton>("Save Config", [this]() { saveConfigToJSON(); });

    trainingDatasetTarget_ = [&]() -> std::unique_ptr<DatasetTarget> {
        namespace fs = std::filesystem;
        GRIM::Config::GrimTextPaths paths;
        GRIM::Config::loadGrimTextPaths(paths);
        fs::path grimRoot = GRIM::Config::detail::resolveGrimRoot();
        fs::path modelStoreRoot = paths.model_store.empty()
            ? grimRoot / "resources" / "models" / "model_store"
            : fs::path(paths.model_store);
        fs::path massDatasetPath = paths.training_data.empty()
            ? grimRoot / "resources" / "models" / "GRIM-text" / "training" / "data" / "mass_dataset.jsonl"
            : fs::path(paths.training_data).parent_path() / "mass_dataset.jsonl";
        auto dt = std::make_unique<DatasetTarget>(modelStoreRoot, massDatasetPath);
        dt->loadCurriculumRegistry();
        return dt;
    }();

    curriculumDropdown_ = std::make_shared<UIDropdown>(
        "Curriculum", std::vector<std::string>{"(none)"}, 0,
        [this](int /*idx*/, const std::string& /*name*/) {
            if (!trainingDatasetTarget_) return;
            const auto& curricula = trainingDatasetTarget_->getCurriculums();
            int sel = curriculumDropdown_->getSelectedIndex() - 1; // -1 for (none)
            selectedCurriculumId_ = (sel >= 0 && sel < static_cast<int>(curricula.size()))
                ? curricula[sel].id : "";
        });
    curriculumDropdown_->setSize(300.0f, 26.0f);

    trainModelDropdown_ = std::make_shared<UIDropdown>(
        "Model", std::vector<std::string>{"(none)"}, 0,
        [this](int /*idx*/, const std::string& /*name*/) {
            auto models = GRIM::MMO::ModelRegistry::instance().getAllModels();
            int sel = trainModelDropdown_->getSelectedIndex() - 1; // -1 for (none)
            selectedTrainModelId_ = (sel >= 0 && sel < static_cast<int>(models.size()))
                ? models[sel]->id : "";
        });
    trainModelDropdown_->setSize(300.0f, 26.0f);

    refreshTrainingDropdowns();

    loadPathsFromConfig();

    // ── Bottom action buttons ──
    startButton = std::make_shared<UIButton>("Start", [this]() {
        if (serverStarting) { addLog("Server startup already in progress...", 1); return; }
        addLog("Starting training session...", 0);
        serverStarting = true;

        std::thread([this]() {
            addLog("Checking for training control server...", 0);
            bool ready = GRIM::isTrainingServerRunning();
            if (ready) {
                addLog("Training server already running", 0);
            } else {
                addLog("Starting training server...", 0);
                if (!GRIM::startTrainingServer()) {
                    addLog("Failed to start server, attempting connection...", 1);
                } else {
                    addLog("Server launch initiated", 0);
                }
                addLog("Waiting for server...", 0);
                for (int i = 0; i < 10; i++) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    if (GRIM::isTrainingServerRunning()) { ready = true; addLog("Server ready", 0); break; }
                    if ((i + 1) % 4 == 0) addLog("Still waiting... (" + std::to_string((i + 1) / 2) + "s)", 0);
                }
                if (!ready) {
                    addLog("Server not responding — check port 11436", 2);
                    serverStarting = false;
                    return;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            pollServer();
            addLog("Sending training start command...", 0);
            startTrainingSession();
            serverStarting = false;
        }).detach();
    });

    stopButton = std::make_shared<UIButton>("Stop", [this]() {
        if (currentState == Control::TrainingState_Training || currentState == Control::TrainingState_Paused) {
            stopTrainingSession();
        }
    });

    pauseResumeButton = std::make_shared<UIButton>("Pause", [this]() {
        if (currentState == Control::TrainingState_Training) { pauseTrainingSession(); }
        else if (currentState == Control::TrainingState_Paused) { resumeTrainingSession(); }
    });

    resetStatusButton = std::make_shared<UIButton>("Reset", [this]() {
        resetState();
        checkpointMergeStatus = "";
        addLog("Training status reset", 0);
        std::string statusFilePath = getResourcePath() + "/models/GRIM-text/training/training_status.fb";
        if (std::filesystem::exists(statusFilePath)) {
            std::filesystem::remove(statusFilePath);
            addLog("Cleared stale status file", 0);
        }
    });

    closeButton = std::make_shared<UIButton>("Close", [this]() {
        setVisible(false);
    });

    runTokenizerButton = std::make_shared<UIButton>("Run Tokenizer", [this]() {
        handleRunTokenizer();
    });

    // ── Progress bars ──
    trainingProgressBar = std::make_shared<UIProgressBar>("Training Progress", 1.0f);
    trainingProgressBar->setFillColor(Colors::Success);
    trainingProgressBar->setBackgroundColor(Colors::ContentAreaBg);

    // ── Resource monitor graph ──
    resourceMonitorGraph = std::make_shared<UIGraph>("System Resources", GraphType::Area);
    resourceSampleTimer = 0.0f;
    resourceSampleInterval = 1.0f;
    resourceSampleCount = 0;
    maxResourceSamples = 30;

    GraphConfig& gc = resourceMonitorGraph->getConfig();
    gc.showGrid = true;
    gc.showAxes = true;
    gc.showLegend = true;
    gc.showLabels = true;
    gc.autoScale = false;
    gc.minValue = 0.0f;
    gc.maxValue = 100.0f;
    gc.gridLines = 5;
    gc.lineThickness = 2.0f;
    gc.backgroundColor = Colors::ContentAreaBg;
    gc.gridColor = Colors::DividerFaint;
    gc.axisColor = Colors::BorderMedium;
    gc.textColor = Colors::TextSecondary;
    gc.paddingLeft = 50.0f;
    gc.paddingRight = 15.0f;
    gc.paddingTop = 30.0f;
    gc.paddingBottom = 50.0f;

    resourceMonitorGraph->addSeries("CPU", {}, Colors::AccentBlue);
    resourceMonitorGraph->addSeries("Memory", {}, Colors::Success);
    resourceMonitorGraph->addSeries("GPU", {}, Colors::Warning);

    ResourceMonitor::getInstance().initialize();

    updateHardwareInfo();
    calculateTrainingEstimate();

    // ══════════════════════════════════════════════════════
    //  Hyperparameter registry + param browser widgets
    // ══════════════════════════════════════════════════════

    paramScrollBox_ = std::make_shared<UIScrollBox>();
    paramScrollBox_->setSize(340.0f, 400.0f);

    loadHyperparamSnapshot();

    // Category filter dropdown — built after loading snapshot
    std::vector<std::string> filterItems = {"All"};
    if (hyperparamsLoaded_) {
        for (const auto& cat : hyperparamRegistry_.categories()) {
            filterItems.push_back(cat);
        }
    }
    paramCategoryFilter_ = std::make_shared<UIDropdown>(
        "Filter", filterItems, 0,
        [this](int idx, const std::string& /*name*/) {
            selectedParamCategory_ = idx;
            paramScrollOffset_ = 0.0f;
            hoveredParamRow_ = -1;
        });
    paramCategoryFilter_->setSize(200.0f, 26.0f);

    // Inline param editor — shared input box, hidden until a row is clicked
    paramEditInput_ = std::make_shared<UIInputBox>(&editParamBuffer_);
    paramEditInput_->setSize(120.0f, 20.0f);
    paramEditInput_->OnTextSubmitted.Bind([this](const std::string&) { commitParamEdit(); });

    // ══════════════════════════════════════════════════════
    //  Tokenizer tab widgets
    // ══════════════════════════════════════════════════════

    encodeInputBox_ = std::make_shared<UIInputBox>(&encodeInputBuffer_);
    encodeInputBox_->setSize(500.0f, 28.0f);
    encodeInputBox_->setPlaceholder("Enter text to encode...");

    encodeButton_ = std::make_shared<UIButton>("Encode", [this]() {
        handleEncodeText();
    });
    encodeButton_->setSize(90.0f, Sizes::ButtonHeight);

    clearEncodeButton_ = std::make_shared<UIButton>("Clear", [this]() {
        encodeInputBuffer_.clear();
        encodeComplete_ = false;
        encodeSuccess_ = false;
        encodeErrorMessage_.clear();
        lastEncodeResult_ = {};
    });
    clearEncodeButton_->setSize(70.0f, Sizes::ButtonHeight);

    tokenizerRunValidationBtn_ = std::make_shared<UIButton>("Run Validation", [this]() {
        handleRunTokenizer();
    });
    tokenizerRunValidationBtn_->setSize(120.0f, Sizes::ButtonHeight);

    tokenizerCloseBtn_ = std::make_shared<UIButton>("Close", [this]() {
        setVisible(false);
    });
    tokenizerCloseBtn_->setSize(90.0f, Sizes::ButtonHeight);

    tokenizerScrollBox_ = std::make_shared<UIScrollBox>();
    tokenizerScrollBox_->setSize(600.0f, 300.0f);
}

UITrainingPanel::~UITrainingPanel() = default;

// ============================================================
// View Control (DataHub pattern)
// ============================================================

void UITrainingPanel::setView(TrainingPanelTab tab) {
    activeTab_ = tab;
    if (tab == TrainingPanelTab::Home) {
        refreshModelList();
        updateResourceBars();
    } else if (tab == TrainingPanelTab::Training) {
        refreshTrainingDropdowns();
    }
}

// ============================================================
// Knowledge Gap Intake
// ============================================================

void UITrainingPanel::pushKnowledgeGap(KnowledgeGapEntry entry) {
    std::lock_guard<std::mutex> lock(gapMutex_);
    gapQueue_.push_back(std::move(entry));
}

size_t UITrainingPanel::pendingGapCount() const {
    std::lock_guard<std::mutex> lock(gapMutex_);
    return gapQueue_.size();
}

// ============================================================
// Tool Gap Intake
// ============================================================

void UITrainingPanel::pushToolGap(GRIM::MMO::ToolGapProposal proposal) {
    std::lock_guard<std::mutex> lock(toolGapMutex_);
    toolGapQueue_.push_back(std::move(proposal));
}

size_t UITrainingPanel::pendingToolGapCount() const {
    std::lock_guard<std::mutex> lock(toolGapMutex_);
    return toolGapQueue_.size();
}

// ============================================================
// Callbacks
// ============================================================

void UITrainingPanel::setupCallbacks() {
    if (!trainingController) return;

    trainingController->setProgressCallback([this](const GRIMText::TrainingStats& stats) {
        currentStats = stats;
    });
    trainingController->setStateChangeCallback([this](Control::TrainingState oldS, Control::TrainingState newS) {
        addLog("State: " + getStateString(oldS) + " -> " + getStateString(newS), 1);
    });
    trainingController->setErrorCallback([this](const std::string& error) {
        lastError = error;
        addLog("Error: " + error, 2);
    });
}

void UITrainingPanel::resetState() {
    currentState = Control::TrainingState_Idle;
    serverStarting = false;
    memset(&currentStats, 0, sizeof(currentStats));
    memset(&currentConfig, 0, sizeof(currentConfig));

    currentConfig.epochs = 5;
    currentConfig.batchSize = 16;
    currentConfig.learningRate = 0.000001f;
}

// ============================================================
// Update
// ============================================================

void UITrainingPanel::update(const InputState& input, float dt) {
    if (!isVisible()) return;
    UIPanel::update(input, dt);

    // Tab buttons — always update
    tabHomeBtn_->update(input, dt);
    tabKnowledgeGapsBtn_->update(input, dt);
    tabToolGapsBtn_->update(input, dt);
    tabTrainingBtn_->update(input, dt);
    tabTokenizerBtn_->update(input, dt);

    // Bottom bar buttons — always active
    startButton->update(input, dt);
    stopButton->update(input, dt);
    pauseResumeButton->update(input, dt);
    resetStatusButton->update(input, dt);
    closeButton->update(input, dt);

    // Tab-specific updates
    switch (activeTab_) {
        case TrainingPanelTab::Home: {
            browserScrollBox_->update(input, dt);
            vramBar_->update(input, dt);
            ramBar_->update(input, dt);
            createModelBtn_->update(input, dt);

            processBrowserClicks(input, getContentRect());

            browserRefreshTimer_ += dt;
            if (browserRefreshTimer_ >= browserRefreshInterval_) {
                browserRefreshTimer_ = 0.0f;
                refreshModelList();
                updateResourceBars();
            }

            if (showCreatorForm_) {
                creatorNameInput_->update(input, dt);
                creatorParamInput_->update(input, dt);
                creatorSubjectInput_->update(input, dt);
                creatorTagsInput_->update(input, dt);
                creatorDescInput_->update(input, dt);
                creatorRegisterBtn_->update(input, dt);
                creatorCancelBtn_->update(input, dt);

                // Regenerate ID whenever subject or param count changes
                static std::string lastBufName;
                static std::string lastBufParam;
                if (bufSubject_ != lastBufName || bufParamStr_ != lastBufParam) {
                    lastBufName  = bufSubject_;
                    lastBufParam = bufParamStr_;
                    regenerateId();
                }
            }
            break;
        }
        case TrainingPanelTab::KnowledgeGaps: {
            gapScrollBox_->update(input, dt);
            processGapClicks(input, getContentRect());
            break;
        }
        case TrainingPanelTab::ToolGaps: {
            toolGapScrollBox_->update(input, dt);
            processToolGapClicks(input, getContentRect());
            break;
        }
        case TrainingPanelTab::Training: {
            if (curriculumDropdown_) curriculumDropdown_->update(input, dt);
            if (trainModelDropdown_) trainModelDropdown_->update(input, dt);
            if (paramCategoryFilter_) paramCategoryFilter_->update(input, dt);

            // Param browser scroll + hover — check if mouse is over left panel area
            {
                auto content = getContentRect();
                content.origin.y += (kContentTopY - kTabBarY);
                content.size.y -= (kContentTopY - kTabBarY);
                content.size.y -= kBottomBarH;
                constexpr float kParamBrowserW = 350.0f;
                constexpr float rowH = 22.0f;
                Vec2 m = input.mousePos;

                float bx = content.origin.x;
                float by = content.origin.y;
                // Skip same layout offsets as drawParamBrowser
                by += Sizes::HeaderHeight + Spacing::Small;   // section header
                by += 26.0f + Spacing::Small;                 // category dropdown
                float listH = content.size.y - (by - content.origin.y) - Spacing::Small;
                if (listH < 60.0f) listH = 60.0f;

                std::string catFilter;
                if (hyperparamsLoaded_ && paramCategoryFilter_ && selectedParamCategory_ > 0) {
                    const auto& cats = hyperparamRegistry_.categories();
                    if (selectedParamCategory_ - 1 < static_cast<int>(cats.size())) {
                        catFilter = cats[static_cast<size_t>(selectedParamCategory_ - 1)];
                    }
                }
                auto params = hyperparamRegistry_.filtered(catFilter);
                const float totalContentH = computeParamBrowserContentHeight(params, catFilter, rowH);
                const float maxScroll = std::max(0.0f, totalContentH - listH);
                paramScrollOffset_ = std::clamp(paramScrollOffset_, 0.0f, maxScroll);

                const ParamScrollbarMetrics scrollbar = computeParamScrollbarMetrics(
                    bx, by, kParamBrowserW, listH, totalContentH, paramScrollOffset_);
                const bool overPanel = pointInRect(m, bx, content.origin.y, kParamBrowserW, content.size.y);
                const bool overList = pointInRect(m, bx, by, kParamBrowserW, listH);
                const bool overScrollbarTrack = scrollbar.visible &&
                    pointInRect(m, scrollbar.trackX, scrollbar.trackY, scrollbar.trackW, scrollbar.trackH);
                const bool overScrollbarThumb = scrollbar.visible &&
                    pointInRect(m, scrollbar.trackX, scrollbar.thumbY, scrollbar.trackW, scrollbar.thumbH);

                hoveredParamScrollbar_ = overScrollbarThumb;

                if (scrollbar.visible && input.mousePressed[0] && overScrollbarTrack) {
                    draggingParamScrollbar_ = true;
                    if (!overScrollbarThumb && scrollbar.maxThumbTravel > 0.0f) {
                        const float targetThumbY = std::clamp(
                            m.y - scrollbar.thumbH * 0.5f,
                            scrollbar.trackY,
                            scrollbar.trackY + scrollbar.maxThumbTravel);
                        const float thumbRatio = (targetThumbY - scrollbar.trackY) / scrollbar.maxThumbTravel;
                        paramScrollOffset_ = thumbRatio * scrollbar.maxScroll;
                    }
                    paramScrollbarDragStartY_ = m.y;
                    paramScrollbarDragStartOffset_ = paramScrollOffset_;
                }

                if (draggingParamScrollbar_) {
                    if (input.mouseDown[0]) {
                        if (scrollbar.maxThumbTravel > 0.0f && scrollbar.maxScroll > 0.0f) {
                            const float deltaY = m.y - paramScrollbarDragStartY_;
                            const float scrollDelta = (deltaY / scrollbar.maxThumbTravel) * scrollbar.maxScroll;
                            paramScrollOffset_ = std::clamp(
                                paramScrollbarDragStartOffset_ + scrollDelta,
                                0.0f,
                                scrollbar.maxScroll);
                        }
                    } else {
                        draggingParamScrollbar_ = false;
                    }
                }

                if (overList && !draggingParamScrollbar_ && input.mouseWheelDelta != 0.0f) {
                    const float wheelSteps = normalizeMouseWheelDelta(input.mouseWheelDelta);
                    paramScrollOffset_ = std::clamp(
                        paramScrollOffset_ - wheelSteps * kParamBrowserWheelPixelsPerStep,
                        0.0f,
                        maxScroll);
                }

                if (!overPanel && !draggingParamScrollbar_) {
                    hoveredParamScrollbar_ = false;
                }

                // Hover detection — must match drawParamBrowser geometry exactly
                hoveredParamRow_ = -1;
                if (hyperparamsLoaded_ && overList && !draggingParamScrollbar_ &&
                    (!scrollbar.visible || m.x < scrollbar.trackX)) {
                    float drawY = by + 2.0f - paramScrollOffset_;
                    std::string lastCat;
                    for (size_t i = 0; i < params.size(); ++i) {
                        const auto* entry = params[i];
                        if (catFilter.empty() && entry->category != lastCat) {
                            lastCat = entry->category;
                            drawY += rowH; // category sub-header row
                        }
                        if (drawY + rowH >= by && drawY <= by + listH &&
                            m.y >= drawY && m.y < drawY + rowH) {
                            hoveredParamRow_ = static_cast<int>(i);
                            break;
                        }
                        drawY += rowH;
                    }
                }
            }

            // Inline param edit input
            if (editingParamIndex_ >= 0 && paramEditInput_) {
                paramEditInput_->update(input, dt);
            }

            // Click handling for param browser rows
            processParamBrowserClicks(input);
            break;
        }
        case TrainingPanelTab::Tokenizer: {
            if (encodeInputBox_) encodeInputBox_->update(input, dt);
            if (encodeButton_) encodeButton_->update(input, dt);
            if (clearEncodeButton_) clearEncodeButton_->update(input, dt);
            if (tokenizerRunValidationBtn_) tokenizerRunValidationBtn_->update(input, dt);
            if (tokenizerCloseBtn_) tokenizerCloseBtn_->update(input, dt);
            if (tokenizerScrollBox_) tokenizerScrollBox_->update(input, dt);
            break;
        }
    }

    // Poll server
    pollTimer += dt;
    if (pollTimer >= pollInterval) {
        pollTimer = 0.0f;
        pollServer();
    }

    // Dataset snapshot I/O
    datasetUpdateTimer += dt;
    if (datasetUpdateTimer >= datasetUpdateInterval) {
        datasetUpdateTimer = 0.0f;
        requestDatasetSnapshot();
    }
    if (datasetSizeInfo.empty() && !datasetSnapshotInFlight.load()) {
        requestDatasetSnapshot();
    }
    applyPendingDatasetSnapshot();
}

// ============================================================
// Server Polling
// ============================================================

void UITrainingPanel::pollServer() {
    if (!trainingController) return;

    bool previouslyConnected = serverConnected;
    serverConnected = trainingController->isServerRunning();

    if (previouslyConnected && !serverConnected) {
        if (currentState == Control::TrainingState_Training ||
            currentState == Control::TrainingState_Collecting ||
            currentState == Control::TrainingState_Verifying) {
            addLog("Server disconnected during operation", 2);
            currentState = Control::TrainingState_Error;
            currentStats.lastError = "Server disconnected unexpectedly";
            checkpointMergeStatus = "";
        }
    }

    if (serverConnected) {
        if (trainingController->pollStatus()) {
            Control::TrainingState prev = currentState;
            currentState = trainingController->getCurrentState();
            currentStats = trainingController->getCurrentStats();
            currentConfig = trainingController->getCurrentConfig();

            if (prev != currentState) {
                if (prev == Control::TrainingState_Training && currentState == Control::TrainingState_Completed) {
                    addLog("Training completed!", 0); checkpointMergeStatus = "";
                } else if (prev == Control::TrainingState_Training && currentState == Control::TrainingState_Error) {
                    addLog("Training error: " + currentStats.lastError, 2); checkpointMergeStatus = "";
                } else if (prev == Control::TrainingState_Training && currentState == Control::TrainingState_Paused) {
                    addLog("Training paused", 1);
                } else if (prev == Control::TrainingState_Paused && currentState == Control::TrainingState_Training) {
                    addLog("Training resumed", 0);
                } else if (prev == Control::TrainingState_Training && currentState == Control::TrainingState_Idle) {
                    addLog("Training stopped unexpectedly", 2);
                    currentStats.lastError = "Training process terminated"; checkpointMergeStatus = "";
                }
            }

            static std::string lastPhase;
            if (currentState == Control::TrainingState_Collecting &&
                !currentStats.currentPhase.empty() && currentStats.currentPhase != lastPhase) {
                addLog("  Phase: " + currentStats.currentPhase, 0);
                lastPhase = currentStats.currentPhase;
            } else if (currentState != Control::TrainingState_Collecting) {
                lastPhase = "";
            }
        } else if (currentState == Control::TrainingState_Training) {
            addLog("Failed to communicate with training server", 2);
        }
    }
}

// ============================================================
// Draw — Main entry + tab dispatch (DataHub pattern)
// ============================================================

bool UITrainingPanel::drawOverlay(OverlayRenderer& renderer) {
    if (!UIPanel::drawOverlay(renderer)) return false;

    // ── Tab buttons ──
    float tabX = position.x + 10.0f;
    tabHomeBtn_->setPosition(tabX, position.y + kTabBarY);
    tabKnowledgeGapsBtn_->setPosition(tabX + 95.0f, position.y + kTabBarY);
    tabToolGapsBtn_->setPosition(tabX + 220.0f, position.y + kTabBarY);
    tabTrainingBtn_->setPosition(tabX + 315.0f, position.y + kTabBarY);
    tabTokenizerBtn_->setPosition(tabX + 410.0f, position.y + kTabBarY);

    tabHomeBtn_->drawOverlay(renderer, position);
    tabKnowledgeGapsBtn_->drawOverlay(renderer, position);
    tabToolGapsBtn_->drawOverlay(renderer, position);
    tabTrainingBtn_->drawOverlay(renderer, position);
    tabTokenizerBtn_->drawOverlay(renderer, position);

    // Active tab indicator (2px underline)
    float indicatorX = tabX;
    float indicatorW = 90.0f;
    switch (activeTab_) {
        case TrainingPanelTab::Home:          indicatorX = tabX;           indicatorW = 90.0f;  break;
        case TrainingPanelTab::KnowledgeGaps: indicatorX = tabX + 95.0f;  indicatorW = 120.0f; break;
        case TrainingPanelTab::ToolGaps:      indicatorX = tabX + 220.0f; indicatorW = 90.0f;  break;
        case TrainingPanelTab::Training:      indicatorX = tabX + 315.0f; indicatorW = 90.0f;  break;
        case TrainingPanelTab::Tokenizer:     indicatorX = tabX + 410.0f; indicatorW = 90.0f;  break;
    }
    renderer.drawRect({indicatorX, position.y + kTabBarY + 28.0f}, {indicatorW, 2.0f},
                      Colors::Primary);

    // ── Status pill (top-right) ──
    {
        std::string stateText = getStateString(currentState);
        uint32_t sc = getStateColor(currentState);
        float tw = UIDrawHelpers::getTextWidth(stateText);
        float px = position.x + size.x - tw - Spacing::Large - 20.0f;
        renderer.drawText({px, position.y + kTabBarY + 4.0f}, stateText, sc);

        // Connection dot
        float dotX = px - 14.0f;
        float dotY = position.y + kTabBarY + 8.0f;
        uint32_t dotColor = serverConnected ? Colors::Success : Colors::Danger;
        renderer.drawRoundedRect({dotX, dotY}, {8.0f, 8.0f}, dotColor, 4.0f);
    }

    // ── Tab content area ──
    PanelRect content = getContentRect();
    content.origin.y += (kContentTopY - kTabBarY);
    content.size.y   -= (kContentTopY - kTabBarY);
    // Reserve space for bottom bar on Training and Tokenizer tabs
    if (activeTab_ == TrainingPanelTab::Training || activeTab_ == TrainingPanelTab::Tokenizer) {
        content.size.y -= kBottomBarH;
    }

    switch (activeTab_) {
        case TrainingPanelTab::Home:          drawHomeTab(renderer, content);          break;
        case TrainingPanelTab::KnowledgeGaps: drawKnowledgeGapsTab(renderer, content); break;
        case TrainingPanelTab::ToolGaps:      drawToolGapsTab(renderer, content);      break;
        case TrainingPanelTab::Training:      drawTrainingTab(renderer, content);      break;
        case TrainingPanelTab::Tokenizer:     drawTokenizerTab(renderer, content);     break;
    }

    // ── Bottom action bar (Training tab) ──
    if (activeTab_ == TrainingPanelTab::Training) {
        float barY = position.y + size.y - kBottomBarH - 10.0f;
        float barW = size.x - 20.0f;
        float barX = position.x + 10.0f;
        UIDrawHelpers::drawDivider(renderer, {barX, barY}, barW);
        drawBottomBar(renderer, barY + 8.0f, barW, barX);
    }

    // ── Bottom action bar (Tokenizer tab) ──
    if (activeTab_ == TrainingPanelTab::Tokenizer) {
        float barY = position.y + size.y - kBottomBarH - 10.0f;
        float barW = size.x - 20.0f;
        float barX = position.x + 10.0f;
        UIDrawHelpers::drawDivider(renderer, {barX, barY}, barW);
        drawTokenizerBottomBar(renderer, barY + 8.0f, barW, barX);
    }

    // ── Dropdowns on top ──
    if (paramCategoryFilter_ && paramCategoryFilter_->isExpanded())
        paramCategoryFilter_->drawExpandedList(renderer, position);
    if (curriculumDropdown_ && curriculumDropdown_->isExpanded())
        curriculumDropdown_->drawExpandedList(renderer, position);
    if (trainModelDropdown_ && trainModelDropdown_->isExpanded())
        trainModelDropdown_->drawExpandedList(renderer, position);

    renderer.popClipRect();
    return true;
}

// ============================================================
// Home Tab — Model Browser
// ============================================================

void UITrainingPanel::drawHomeTab(OverlayRenderer& renderer, const PanelRect& content) {
    float x = content.origin.x + Spacing::PaddingX;
    float y = content.origin.y + Spacing::Small;
    float w = content.size.x - 2.0f * Spacing::PaddingX;

    // Section header + create button
    UIDrawHelpers::drawSectionHeader(renderer, {x - Spacing::PaddingX, y}, content.size.x,
                                     "Registered Models", Colors::SectionAI);
    y += Sizes::HeaderHeight + Spacing::Small;

    // Create model button (top-right of section)
    createModelBtn_->setPosition(x + w - 130.0f, y - Sizes::HeaderHeight - 2.0f);
    createModelBtn_->drawOverlay(renderer, position);

    if (showCreatorForm_) {
        drawCreatorForm(renderer, content);
        return;  // Creator form replaces browser when visible
    }

    drawBrowserView(renderer, content);
}

void UITrainingPanel::drawBrowserView(OverlayRenderer& renderer, const PanelRect& content) {
    float x = content.origin.x + Spacing::PaddingX;
    float y = content.origin.y + Sizes::HeaderHeight + Spacing::Medium + Spacing::Small;
    float w = content.size.x - 2.0f * Spacing::PaddingX;

    // Column headers
    renderer.drawText({x, y}, "Name", Colors::TextSecondary);
    renderer.drawText({x + 200.0f, y}, "Status", Colors::TextSecondary);
    renderer.drawText({x + 340.0f, y}, "Backend", Colors::TextSecondary);
    renderer.drawText({x + 480.0f, y}, "RAM", Colors::TextSecondary);
    renderer.drawText({x + 560.0f, y}, "VRAM", Colors::TextSecondary);
    renderer.drawText({x + 640.0f, y}, "Actions", Colors::TextSecondary);
    y += 22.0f;

    UIDrawHelpers::drawDivider(renderer, {x, y}, w);
    y += 4.0f;

    if (modelEntries_.empty()) {
        renderer.drawText({x, y}, "No models registered. Use '+ New Model' to add one.", Colors::TextMuted);
    }

    for (size_t i = 0; i < modelEntries_.size(); ++i) {
        const auto& entry = modelEntries_[i];

        // Hover highlight
        if (static_cast<int>(i) == hoveredBrowserRow_) {
            renderer.drawRoundedRect({x, y - 1.0f}, {w, kRowHeight}, Colors::ContentAreaBg, 4.0f);
        }

        std::string label = entry.name;
        if (entry.is_router) label += " [R]";

        renderer.drawText({x, y}, label, Colors::TextPrimary);
        renderer.drawText({x + 200.0f, y}, entry.status, entry.statusColor);
        renderer.drawText({x + 340.0f, y}, entry.backend, Colors::TextSecondary);
        renderer.drawText({x + 480.0f, y},
            std::to_string(entry.ram_mb) + " MB", Colors::TextSecondary);
        renderer.drawText({x + 560.0f, y},
            std::to_string(entry.vram_mb) + " MB", Colors::TextSecondary);

        // Action buttons
        if (!entry.is_router) {
            bool isLoaded = (entry.status != "Unloaded" && entry.status != "N/A");
            if (isLoaded) {
                renderer.drawText({x + 640.0f, y}, "[Unload]", Colors::Warning);
            } else {
                renderer.drawText({x + 640.0f, y}, "[Load]", Colors::Success);
            }
            renderer.drawText({x + 710.0f, y}, "[x]", Colors::Danger);
        }

        y += kRowHeight;
    }

    // Resource bars at bottom
    float barY = content.origin.y + content.size.y - 30.0f;
    vramBar_->setPosition(content.origin.x + Spacing::PaddingX, barY);
    ramBar_->setPosition(content.origin.x + content.size.x / 2.0f + 10.0f, barY);
    vramBar_->setSize((content.size.x / 2.0f) - Spacing::PaddingX - 10.0f, 20.0f);
    ramBar_->setSize((content.size.x / 2.0f) - Spacing::PaddingX - 10.0f, 20.0f);
    vramBar_->drawOverlay(renderer, position);
    ramBar_->drawOverlay(renderer, position);
}

void UITrainingPanel::processBrowserClicks(const InputState& input, const PanelRect& content) {
    Vec2 m = input.mousePos;
    float x = content.origin.x + Spacing::PaddingX;
    float dataY = content.origin.y + Sizes::HeaderHeight + Spacing::Medium + Spacing::Small + 26.0f;
    float w = content.size.x - 2.0f * Spacing::PaddingX;

    hoveredBrowserRow_ = -1;

    if (m.x < x || m.x > x + w || m.y < dataY) return;

    int row = static_cast<int>((m.y - dataY) / kRowHeight);
    if (row < 0 || row >= static_cast<int>(modelEntries_.size())) return;

    hoveredBrowserRow_ = row;
    const auto& entry = modelEntries_[static_cast<size_t>(row)];

    if (!input.mousePressed[0]) return;
    if (entry.is_router) return;

    float actionLoadX = x + 640.0f;
    float actionRemoveX = x + 710.0f;

    if (m.x >= actionRemoveX && m.x <= actionRemoveX + 30.0f) {
        handleModelAction(entry.id, "remove");
    } else if (m.x >= actionLoadX && m.x <= actionLoadX + 60.0f) {
        bool isLoaded = (entry.status != "Unloaded" && entry.status != "N/A");
        handleModelAction(entry.id, isLoaded ? "unload" : "load");
    }
}

void UITrainingPanel::handleModelAction(const std::string& model_id,
                                        const std::string& action) {
    auto& registry = GRIM::MMO::ModelRegistry::instance();

    if (action == "load") {
        if (g_modelLoader) {
            auto result = g_modelLoader->ensureLoaded(model_id);
            LOG_DEBUG("UITrainingPanel", "ensureLoaded('" + model_id + "') → "
                + std::string(GRIM::MMO::loadResultToString(result)));
        }
    } else if (action == "unload") {
        if (g_modelLoader) {
            g_modelLoader->unload(model_id);
            LOG_DEBUG("UITrainingPanel", "unloaded '" + model_id + "'");
        }
    } else if (action == "remove") {
        if (g_modelLoader) {
            auto state = g_modelLoader->getState(model_id);
            if (state != GRIM::MMO::ResidencyState::Unloaded) {
                g_modelLoader->unload(model_id);
            }
        }
        registry.removeModel(model_id);
        removeSubModelFromConfig(model_id);
        LOG_DEBUG("UITrainingPanel", "removed model '" + model_id + "'");
        refreshModelList();
    }
}

// ============================================================
// Model Creator Form (overlay on Home tab)
// ============================================================

void UITrainingPanel::drawCreatorForm(OverlayRenderer& renderer, const PanelRect& content) {
    float x = content.origin.x + Spacing::PaddingX;
    float y = content.origin.y + Sizes::HeaderHeight + Spacing::Medium + Spacing::Small;
    float inputX = x + 100.0f;

    UIDrawHelpers::drawSectionHeader(renderer, {x - Spacing::PaddingX, y}, content.size.x,
                                     "Register New Model", Colors::SectionPersonality);
    y += Sizes::HeaderHeight + Spacing::Small;

    auto drawFormRow = [&](const std::string& label, std::shared_ptr<UIInputBox>& input) {
        renderer.drawText({x, y + 4.0f}, label, Colors::TextSecondary);
        input->setPosition(inputX, y);
        input->setSize(400.0f, 26.0f);
        input->drawOverlay(renderer, position);
        y += 32.0f;
    };

    // Generated ID display row
    renderer.drawText({x, y + 4.0f}, "ID:", Colors::TextSecondary);
    renderer.drawText({inputX, y + 4.0f},
        bufId_.empty() ? "(enter name to generate)" : bufId_,
        bufId_.empty() ? Colors::TextMuted : Colors::AccentBlue);
    y += 32.0f;

    drawFormRow("Name:", creatorNameInput_);

    // Params row
    renderer.drawText({x, y + 4.0f}, "Params:", Colors::TextSecondary);
    creatorParamInput_->setPosition(inputX, y);
    creatorParamInput_->setSize(200.0f, 26.0f);
    creatorParamInput_->drawOverlay(renderer, position);
    y += 32.0f;
    drawFormRow("Subject:", creatorSubjectInput_);
    drawFormRow("Tags:", creatorTagsInput_);
    drawFormRow("Desc:", creatorDescInput_);

    // Status label
    creatorStatusLabel_->setPosition(inputX, y);
    creatorStatusLabel_->drawOverlay(renderer, position);
    y += 26.0f;

    // Register / Cancel buttons
    creatorRegisterBtn_->setPosition(inputX, y);
    creatorRegisterBtn_->drawOverlay(renderer, position);
    creatorCancelBtn_->setPosition(inputX + 110.0f, y);
    creatorCancelBtn_->drawOverlay(renderer, position);
}

void UITrainingPanel::clearCreatorFields() {
    bufId_.clear(); bufName_.clear(); bufParamStr_ = "0M";
    bufSubject_.clear();
    bufTags_.clear(); bufDesc_.clear();

    creatorNameInput_->clear();
    creatorParamInput_->setText("0M");
    creatorSubjectInput_->clear();
    creatorTagsInput_->clear();
    creatorDescInput_->clear();
    creatorStatusLabel_->setText("");
}

bool UITrainingPanel::validateCreatorFields(std::string& out_error) const {
    if (bufName_.empty())    { out_error = "Name is required"; return false; }
    if (bufSubject_.empty()) { out_error = "Subject is required (used for ID)"; return false; }
    if (bufId_.empty())      { out_error = "Subject is required to generate ID"; return false; }

    auto& registry = GRIM::MMO::ModelRegistry::instance();
    if (registry.getModelById(bufId_)) {
        out_error = "Model ID '" + bufId_ + "' already exists";
        return false;
    }
    return true;
}

void UITrainingPanel::submitNewModel() {
    std::string error;
    if (!validateCreatorFields(error)) {
        creatorStatusLabel_->setText(error);
        creatorStatusLabel_->setColor(Colors::Danger);
        return;
    }

    GRIM::MMO::ModelInfo model;
    model.id          = bufId_;
    model.name        = bufName_;
    model.subject     = bufSubject_;
    model.description = bufDesc_;
    model.subject_tags = splitCommaTags(bufTags_);

    // Auto-create model directory under model_store
    {
        namespace fs = std::filesystem;
        GRIM::Config::GrimTextPaths paths;
        GRIM::Config::loadGrimTextPaths(paths);
        fs::path grimRoot = GRIM::Config::detail::resolveGrimRoot();
        fs::path modelStoreRoot = paths.model_store.empty()
            ? grimRoot / "resources" / "models" / "model_store"
            : fs::path(paths.model_store);
        fs::path modelDir = modelStoreRoot / bufId_;
        std::error_code ec;
        fs::create_directories(modelDir, ec);
        if (ec) {
            creatorStatusLabel_->setText("Failed to create model dir: " + ec.message());
            creatorStatusLabel_->setColor(Colors::Danger);
            return;
        }
        model.model_path = modelDir.string();
    }

    model.backend_type = GRIM::MMO::BackendType::GrimTextServer;

    auto& registry = GRIM::MMO::ModelRegistry::instance();
    try {
        registry.registerModel(std::move(model));
    } catch (const std::runtime_error& e) {
        creatorStatusLabel_->setText(e.what());
        creatorStatusLabel_->setColor(Colors::Danger);
        return;
    }

    const auto* registered = registry.getModelById(bufId_);
    if (registered && !persistSubModel(*registered)) {
        creatorStatusLabel_->setText("Registered but failed to save config");
        creatorStatusLabel_->setColor(Colors::Warning);
        return;
    }

    LOG_DEBUG("UITrainingPanel", "Registered new sub-model '" + bufId_ + "'");
    clearCreatorFields();
    showCreatorForm_ = false;
    refreshModelList();
    creatorStatusLabel_->setText("Registered successfully");
    creatorStatusLabel_->setColor(Colors::Success);
}

// ============================================================
// ID generation helpers
// ============================================================

std::string UITrainingPanel::slugifyName(const std::string& name) {
    std::string slug;
    slug.reserve(name.size());
    bool lastWasDash = true; // suppress leading dash
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            slug += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            lastWasDash = false;
        } else if (!lastWasDash && !slug.empty()) {
            slug += '-';
            lastWasDash = true;
        }
    }
    // Strip trailing dash
    while (!slug.empty() && slug.back() == '-')
        slug.pop_back();
    return slug;
}

int64_t UITrainingPanel::parseParamStr(const std::string& s) {
    if (s.empty()) return 0;
    double val = 0.0;
    size_t i = 0;
    while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.'))
        ++i;
    try { val = std::stod(s.substr(0, i)); } catch (...) { return 0; }

    // Optional suffix: K / M / B / T (case-insensitive)
    if (i < s.size()) {
        char suffix = static_cast<char>(std::toupper(static_cast<unsigned char>(s[i])));
        if      (suffix == 'K') val *= 1e3;
        else if (suffix == 'M') val *= 1e6;
        else if (suffix == 'B') val *= 1e9;
        else if (suffix == 'T') val *= 1e12;
    }
    return static_cast<int64_t>(val);
}

std::string UITrainingPanel::formatParamCount(int64_t params) {
    // Format to 3 significant figures with unit suffix
    struct Tier { double divisor; const char* suffix; };
    static const Tier tiers[] = {
        { 1e12, "T" }, { 1e9, "B" }, { 1e6, "M" }, { 1e3, "K" }
    };
    for (const auto& t : tiers) {
        if (params >= static_cast<int64_t>(t.divisor * 0.5)) {
            double v = params / t.divisor;
            char buf[32];
            // 3 significant figures
            if      (v >= 100.0) std::snprintf(buf, sizeof(buf), "%.0f%s", v, t.suffix);
            else if (v >= 10.0)  std::snprintf(buf, sizeof(buf), "%.1f%s", v, t.suffix);
            else                 std::snprintf(buf, sizeof(buf), "%.2f%s", v, t.suffix);
            return buf;
        }
    }
    // Sub-kilo: just raw count
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%lldP", static_cast<long long>(params));
    return buf;
}

int UITrainingPanel::nextVersionForBase(const std::string& base) {
    auto& registry = GRIM::MMO::ModelRegistry::instance();
    for (int v = 1; v <= 9999; ++v) {
        std::string candidate = base + "-v" + std::to_string(v);
        if (!registry.getModelById(candidate))
            return v;
    }
    return 1;
}

void UITrainingPanel::regenerateId() {
    std::string slug = slugifyName(bufSubject_);
    if (slug.empty()) { bufId_.clear(); return; }

    int64_t params = parseParamStr(bufParamStr_.empty() ? "0M" : bufParamStr_);
    std::string paramFmt = formatParamCount(params);

    std::string base = slug + "-" + paramFmt;
    int ver = nextVersionForBase(base);
    bufId_ = base + "-v" + std::to_string(ver);
}

void UITrainingPanel::prefillCreatorFromGap(const KnowledgeGapEntry& gap) {
    setView(TrainingPanelTab::Home);
    showCreatorForm_ = true;
    bufSubject_ = gap.subject;
    creatorSubjectInput_->setText(gap.subject);

    std::string tagStr;
    for (size_t i = 0; i < gap.tags.size(); ++i) {
        if (i > 0) tagStr += ", ";
        tagStr += gap.tags[i];
    }
    bufTags_ = tagStr;
    creatorTagsInput_->setText(tagStr);

    creatorStatusLabel_->setText("Pre-filled from knowledge gap");
    creatorStatusLabel_->setColor(Colors::Warning);
}

// ============================================================
// Model Browser — Data
// ============================================================

void UITrainingPanel::refreshModelList() {
    auto& registry = GRIM::MMO::ModelRegistry::instance();
    auto allModels = registry.getAllModels();
    const auto* router = registry.getRouter();

    modelEntries_.clear();
    modelEntries_.reserve(allModels.size());

    for (const auto* model : allModels) {
        ModelListEntry entry;
        entry.id       = model->id;
        entry.name     = model->name;
        entry.subject  = model->subject;
        entry.backend  = backendDisplayName(model->backend_type);
        entry.ram_mb   = model->estimated_ram_mb;
        entry.vram_mb  = model->estimated_vram_mb;
        entry.is_router = (router && model->id == router->id);

        if (g_modelLoader) {
            auto state = g_modelLoader->getState(model->id);
            entry.status      = GRIM::MMO::residencyStateToString(state);
            entry.statusColor = residencyStatusColor(state);
        } else {
            entry.status      = "N/A";
            entry.statusColor = Colors::TextMuted;
        }
        modelEntries_.push_back(std::move(entry));
    }
}

// ============================================================
// Config persistence (ai_config.json → mmo.sub_models)
// ============================================================

bool UITrainingPanel::persistSubModel(const GRIM::MMO::ModelInfo& model) {
    namespace fs = std::filesystem;

    try {
        nlohmann::json config;
        {
            std::ifstream f(AI_CONFIG_FILE);
            if (!f.is_open()) {
                LOG_ERROR("UITrainingPanel", "Cannot open ai_config.json for reading");
                return false;
            }
            f >> config;
        }

        if (!config.contains("mmo") || !config["mmo"].is_object()) {
            LOG_ERROR("UITrainingPanel", "ai_config.json missing 'mmo' section");
            return false;
        }

        if (!config["mmo"].contains("sub_models"))
            config["mmo"]["sub_models"] = nlohmann::json::array();

        config["mmo"]["sub_models"].push_back(
            GRIM::MMO::ModelRegistry::serializeModelToJson(model));

        const std::string tmpPath = std::string(AI_CONFIG_FILE) + ".tmp";
        {
            std::ofstream out(tmpPath);
            if (!out.is_open()) {
                LOG_ERROR("UITrainingPanel", "Cannot write temp config file");
                return false;
            }
            out << config.dump(4);
        }
        fs::rename(tmpPath, AI_CONFIG_FILE);

        aiConfig = config;
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("UITrainingPanel", std::string("persistSubModel failed: ") + e.what());
        return false;
    }
}

bool UITrainingPanel::removeSubModelFromConfig(const std::string& model_id) {
    namespace fs = std::filesystem;

    try {
        nlohmann::json config;
        {
            std::ifstream f(AI_CONFIG_FILE);
            if (!f.is_open()) return false;
            f >> config;
        }

        if (!config.contains("mmo") || !config["mmo"].contains("sub_models"))
            return false;

        auto& subs = config["mmo"]["sub_models"];
        bool found = false;
        for (auto it = subs.begin(); it != subs.end(); ++it) {
            if (it->value("id", "") == model_id) {
                subs.erase(it);
                found = true;
                break;
            }
        }
        if (!found) return false;

        const std::string tmpPath = std::string(AI_CONFIG_FILE) + ".tmp";
        {
            std::ofstream out(tmpPath);
            if (!out.is_open()) return false;
            out << config.dump(4);
        }
        fs::rename(tmpPath, AI_CONFIG_FILE);

        aiConfig = config;
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("UITrainingPanel", std::string("removeSubModelFromConfig failed: ") + e.what());
        return false;
    }
}

void UITrainingPanel::updateResourceBars() {
    if (!g_resourceSignal) return;

    auto snap = g_resourceSignal->latest();

    totalRamMb_ = static_cast<float>(snap.ram_used_mb + snap.ram_available_mb);
    usedRamMb_  = static_cast<float>(snap.ram_used_mb);
    ramBar_->setMaxValue(totalRamMb_);
    ramBar_->setValue(usedRamMb_);
    ramBar_->setLabel("RAM: " + std::to_string(snap.ram_used_mb) + " / "
        + std::to_string(snap.ram_used_mb + snap.ram_available_mb) + " MB");

    long totalVram = 0, usedVram = 0;
    for (const auto& gpu : snap.gpus) {
        totalVram += gpu.vram_used_mb + gpu.vram_free_mb;
        usedVram  += gpu.vram_used_mb;
    }
    totalVramMb_ = static_cast<float>(totalVram);
    usedVramMb_  = static_cast<float>(usedVram);
    vramBar_->setMaxValue(totalVramMb_);
    vramBar_->setValue(usedVramMb_);
    vramBar_->setLabel("VRAM: " + std::to_string(usedVram) + " / "
        + std::to_string(totalVram) + " MB");
}

// ============================================================
// Knowledge Gaps Tab
// ============================================================

void UITrainingPanel::drawKnowledgeGapsTab(OverlayRenderer& renderer, const PanelRect& content) {
    std::lock_guard<std::mutex> lock(gapMutex_);

    float x = content.origin.x + Spacing::PaddingX;
    float y = content.origin.y + Spacing::Small;
    float w = content.size.x - 2.0f * Spacing::PaddingX;

    UIDrawHelpers::drawSectionHeader(renderer, {x - Spacing::PaddingX, y}, content.size.x,
                                     "Knowledge Gap Queue", Colors::SectionWhisper);
    y += Sizes::HeaderHeight + Spacing::Small;

    if (gapQueue_.empty()) {
        renderer.drawText({x, y}, "No knowledge gaps queued.", Colors::TextMuted);
        renderer.drawText({x, y + 20.0f},
            "Gaps appear when the router cannot find a matching sub-model.", Colors::TextMuted);
        return;
    }

    // Column headers
    renderer.drawText({x, y}, "Subject", Colors::TextSecondary);
    renderer.drawText({x + 250.0f, y}, "Tags", Colors::TextSecondary);
    renderer.drawText({x + 600.0f, y}, "Actions", Colors::TextSecondary);
    y += 22.0f;

    UIDrawHelpers::drawDivider(renderer, {x, y}, w);
    y += 4.0f;

    for (size_t i = 0; i < gapQueue_.size(); ++i) {
        const auto& gap = gapQueue_[i];

        if (static_cast<int>(i) == hoveredGapRow_) {
            renderer.drawRoundedRect({x, y - 1.0f}, {w, kRowHeight}, Colors::ContentAreaBg, 4.0f);
        }

        renderer.drawText({x, y}, gap.subject, Colors::TextPrimary);

        std::string tagStr;
        for (size_t t = 0; t < gap.tags.size(); ++t) {
            if (t > 0) tagStr += ", ";
            tagStr += gap.tags[t];
        }
        renderer.drawText({x + 250.0f, y}, tagStr, Colors::TextSecondary);
        renderer.drawText({x + 600.0f, y}, "[Create]", Colors::AccentBlue);
        renderer.drawText({x + 670.0f, y}, "[Dismiss]", Colors::Danger);

        y += kRowHeight;
    }
}

void UITrainingPanel::processGapClicks(const InputState& input, const PanelRect& content) {
    Vec2 m = input.mousePos;
    float x = content.origin.x + Spacing::PaddingX;
    float w = content.size.x - 2.0f * Spacing::PaddingX;
    float dataY = content.origin.y + Sizes::HeaderHeight + Spacing::Medium + Spacing::Small + 26.0f;

    hoveredGapRow_ = -1;

    if (m.x < x || m.x > x + w || m.y < dataY) return;

    KnowledgeGapEntry gapCopy;
    int action = 0;
    {
        std::lock_guard<std::mutex> lock(gapMutex_);
        if (gapQueue_.empty()) return;

        int row = static_cast<int>((m.y - dataY) / kRowHeight);
        if (row < 0 || row >= static_cast<int>(gapQueue_.size())) return;

        hoveredGapRow_ = row;
        if (!input.mousePressed[0]) return;

        float createX  = x + 600.0f;
        float dismissX = x + 670.0f;

        if (m.x >= dismissX && m.x <= dismissX + 70.0f) {
            gapQueue_.erase(gapQueue_.begin() + static_cast<ptrdiff_t>(row));
            hoveredGapRow_ = -1;
            action = 1;
        } else if (m.x >= createX && m.x <= createX + 60.0f) {
            gapCopy = gapQueue_[static_cast<size_t>(row)];
            gapQueue_.erase(gapQueue_.begin() + static_cast<ptrdiff_t>(row));
            hoveredGapRow_ = -1;
            action = 2;
        }
    }

    if (action == 2) {
        prefillCreatorFromGap(gapCopy);
    }
}

void UITrainingPanel::dismissGap(size_t index) {
    std::lock_guard<std::mutex> lock(gapMutex_);
    if (index < gapQueue_.size())
        gapQueue_.erase(gapQueue_.begin() + static_cast<ptrdiff_t>(index));
}

void UITrainingPanel::createFromGap(size_t index) {
    KnowledgeGapEntry gap;
    {
        std::lock_guard<std::mutex> lock(gapMutex_);
        if (index >= gapQueue_.size()) return;
        gap = gapQueue_[index];
        gapQueue_.erase(gapQueue_.begin() + static_cast<ptrdiff_t>(index));
    }
    prefillCreatorFromGap(gap);
}

// ============================================================
// Tool Gaps Tab
// ============================================================

void UITrainingPanel::drawToolGapsTab(OverlayRenderer& renderer, const PanelRect& content) {
    std::lock_guard<std::mutex> lock(toolGapMutex_);

    float x = content.origin.x + Spacing::PaddingX;
    float y = content.origin.y + Spacing::Small;
    float w = content.size.x - 2.0f * Spacing::PaddingX;

    UIDrawHelpers::drawSectionHeader(renderer, {x - Spacing::PaddingX, y}, content.size.x,
                                     "Tool Gap Proposals", Colors::SectionNeutral);
    y += Sizes::HeaderHeight + Spacing::Small;

    if (toolGapQueue_.empty()) {
        renderer.drawText({x, y}, "No tool gaps queued.", Colors::TextMuted);
        renderer.drawText({x, y + 20.0f},
            "Tool gaps appear when the model needs a capability not in the ToolRegistry.", Colors::TextMuted);
        return;
    }

    // Column headers
    renderer.drawText({x, y}, "Missing Capability", Colors::TextSecondary);
    renderer.drawText({x + 300.0f, y}, "Reason", Colors::TextSecondary);
    renderer.drawText({x + 500.0f, y}, "Proposed Tool", Colors::TextSecondary);
    renderer.drawText({x + 700.0f, y}, "Actions", Colors::TextSecondary);
    y += 22.0f;

    UIDrawHelpers::drawDivider(renderer, {x, y}, w);
    y += 4.0f;

    for (size_t i = 0; i < toolGapQueue_.size(); ++i) {
        const auto& proposal = toolGapQueue_[i];

        if (static_cast<int>(i) == hoveredToolGapRow_) {
            renderer.drawRoundedRect({x, y - 1.0f}, {w, kRowHeight * 2.0f}, Colors::ContentAreaBg, 4.0f);
        }

        renderer.drawText({x, y}, proposal.missing_capability, Colors::TextPrimary);

        std::string reasonStr;
        switch (proposal.reason) {
            case GRIM::MMO::ToolGapReason::NoMatchingCapability:   reasonStr = "No match"; break;
            case GRIM::MMO::ToolGapReason::CapabilityMismatch:     reasonStr = "Mismatch"; break;
            case GRIM::MMO::ToolGapReason::PermissionInsufficient: reasonStr = "Permissions"; break;
            case GRIM::MMO::ToolGapReason::PolicyBlocked:          reasonStr = "Policy"; break;
        }
        renderer.drawText({x + 300.0f, y}, reasonStr, Colors::Warning);
        renderer.drawText({x + 500.0f, y}, proposal.proposed_spec.display_name, Colors::TextSecondary);
        renderer.drawText({x + 700.0f, y}, "[Approve]", Colors::Success);
        renderer.drawText({x + 780.0f, y}, "[Dismiss]", Colors::Danger);

        // Second row: rationale
        if (!proposal.rationale.empty()) {
            std::string rationale = proposal.rationale;
            if (rationale.size() > 100) rationale = rationale.substr(0, 97) + "...";
            renderer.drawText({x + 20.0f, y + kRowHeight}, rationale, Colors::TextMuted);
        }

        y += kRowHeight * 2.0f;
    }
}

void UITrainingPanel::processToolGapClicks(const InputState& input, const PanelRect& content) {
    Vec2 m = input.mousePos;
    float x = content.origin.x + Spacing::PaddingX;
    float w = content.size.x - 2.0f * Spacing::PaddingX;
    float dataY = content.origin.y + Sizes::HeaderHeight + Spacing::Medium + Spacing::Small + 26.0f;

    hoveredToolGapRow_ = -1;

    if (m.x < x || m.x > x + w || m.y < dataY) return;

    std::lock_guard<std::mutex> lock(toolGapMutex_);
    if (toolGapQueue_.empty()) return;

    int row = static_cast<int>((m.y - dataY) / (kRowHeight * 2.0f));
    if (row < 0 || row >= static_cast<int>(toolGapQueue_.size())) return;

    hoveredToolGapRow_ = row;
    if (!input.mousePressed[0]) return;

    float approveX = x + 700.0f;
    float dismissX = x + 780.0f;

    if (m.x >= dismissX && m.x <= dismissX + 70.0f) {
        toolGapQueue_.erase(toolGapQueue_.begin() + static_cast<ptrdiff_t>(row));
        hoveredToolGapRow_ = -1;
    } else if (m.x >= approveX && m.x <= approveX + 70.0f) {
        // For now, just log approval — actual tool scaffolding is a separate pipeline
        LOG_DEBUG("UITrainingPanel", "Tool gap approved: " + toolGapQueue_[static_cast<size_t>(row)].missing_capability);
        toolGapQueue_.erase(toolGapQueue_.begin() + static_cast<ptrdiff_t>(row));
        hoveredToolGapRow_ = -1;
    }
}

// ============================================================
// Hyperparameter Snapshot Loader
// ============================================================

void UITrainingPanel::loadHyperparamSnapshot() {
    auto snapshot = GRIM::Config::loadAiConfigSnapshot("ai_config.json");
    if (!snapshot || !snapshot->has_training) {
        hyperparamsLoaded_ = false;
        return;
    }
    hyperparamSnapshot_ = snapshot->hyperparameters;
    hyperparamRegistry_.populate(hyperparamSnapshot_);
    hyperparamsLoaded_ = true;
}

// ============================================================
// Parameter Browser (left-side scroll panel)
// ============================================================

void UITrainingPanel::drawParamBrowser(OverlayRenderer& renderer, const Vec2& origin, const Vec2& sz) {
    float x = origin.x;
    float y = origin.y;
    float w = sz.x;
    float h = sz.y;

    // Header
    UIDrawHelpers::drawSectionHeader(renderer, {x, y}, w,
                                     "Model Parameters", Colors::SectionAI);
    y += Sizes::HeaderHeight + Spacing::Small;

    // Category filter dropdown
    if (paramCategoryFilter_) {
        paramCategoryFilter_->setPosition(x + Spacing::PaddingX, y);
        paramCategoryFilter_->setSize(w - 2.0f * Spacing::PaddingX, 26.0f);
        paramCategoryFilter_->drawOverlay(renderer, position);
        y += 26.0f + Spacing::Small;
    }

    if (!hyperparamsLoaded_) {
        renderer.drawText({x + Spacing::PaddingX, y + 10.0f},
                          "No hyperparameters loaded", Colors::TextMuted);
        return;
    }

    // Determine active category filter
    std::string activeCategory;
    if (selectedParamCategory_ > 0) {
        const auto& cats = hyperparamRegistry_.categories();
        int catIdx = selectedParamCategory_ - 1; // 0 = "All"
        if (catIdx >= 0 && catIdx < static_cast<int>(cats.size())) {
            activeCategory = cats[static_cast<size_t>(catIdx)];
        }
    }

    auto filteredParams = hyperparamRegistry_.filtered(activeCategory);
    if (filteredParams.empty()) {
        renderer.drawText({x + Spacing::PaddingX, y + 10.0f},
                          "No parameters in this category", Colors::TextMuted);
        return;
    }

    // Scroll area background
    float listH = h - (y - origin.y) - Spacing::Small;
    if (listH < 60.0f) listH = 60.0f;

    renderer.drawRoundedRect({x, y}, {w, listH}, Colors::ContentAreaBg, Sizes::WidgetRadius);
    renderer.drawRoundedBorder({x, y}, {w, listH}, Colors::BorderSubtle, Sizes::WidgetRadius);

    // Row dimensions
    constexpr float rowH = 22.0f;
    constexpr float padX = 8.0f;
    float nameColW = w * 0.55f;
    // Scroll content
    float totalContentH = computeParamBrowserContentHeight(filteredParams, activeCategory, rowH);
    float maxScroll = std::max(0.0f, totalContentH - listH);
    if (paramScrollOffset_ > maxScroll) paramScrollOffset_ = maxScroll;
    if (paramScrollOffset_ < 0.0f) paramScrollOffset_ = 0.0f;

    const ParamScrollbarMetrics scrollbar = computeParamScrollbarMetrics(
        x, y, w, listH, totalContentH, paramScrollOffset_);

    float drawY = y + 2.0f - paramScrollOffset_;
    std::string lastCategory;

    for (size_t i = 0; i < filteredParams.size(); ++i) {
        const auto* entry = filteredParams[i];

        // Category sub-header (when showing "All")
        if (activeCategory.empty() && entry->category != lastCategory) {
            lastCategory = entry->category;
            // Draw category label row
            if (drawY + rowH >= y && drawY <= y + listH) {
                renderer.drawRect({x + 2.0f, drawY}, {w - 4.0f, rowH}, Colors::TableHeaderBg);
                renderer.drawText({x + padX, drawY + 3.0f}, entry->category, Colors::TextHeader);
            }
            drawY += rowH;
            if (drawY > y + listH) break;
        }

        // Skip rows above visible area
        if (drawY + rowH < y) { drawY += rowH; continue; }
        // Stop below visible area
        if (drawY > y + listH) break;

        // Only draw visible rows
        if (drawY >= y && drawY + rowH <= y + listH) {
            // Alternating row background
            uint32_t rowBg = (i % 2 == 0) ? Colors::RowEven : Colors::RowOdd;
            if (static_cast<int>(i) == hoveredParamRow_) {
                rowBg = Colors::RowHover;
            }
            renderer.drawRect({x + 2.0f, drawY}, {w - 4.0f, rowH}, rowBg);

            // Parameter name
            renderer.drawText({x + padX, drawY + 3.0f}, entry->display_name, Colors::TextLight);

            // Value column — inline edit or read-only display
            if (static_cast<int>(i) == editingParamIndex_ && paramEditInput_ &&
                entry->type != GRIM::Config::HyperparamType::Bool) {
                // Draw the input box for the editing row
                float editX = x + nameColW - 2.0f;
                float editRightPad = scrollbar.visible ? (kParamBrowserScrollbarWidth + 8.0f) : 8.0f;
                float editW = w - nameColW - padX - editRightPad;
                paramEditInput_->setPosition(editX, drawY);
                paramEditInput_->setSize(editW, rowH);
                paramEditInput_->drawOverlay(renderer, position);
            } else {
                std::string valStr = entry->valueAsString();

                // Color-code booleans
                uint32_t valColor = Colors::TextValue;
                if (entry->type == GRIM::Config::HyperparamType::Bool) {
                    valColor = (entry->ptr_bool && *entry->ptr_bool) ? Colors::Success : Colors::TextMuted;
                }

                renderer.drawText({x + nameColW, drawY + 3.0f}, valStr, valColor);
            }
        }

        drawY += rowH;
    }

    // Scrollbar indicator if content overflows
    if (scrollbar.visible) {
        renderer.drawRoundedRect({scrollbar.trackX, scrollbar.trackY},
                                 {scrollbar.trackW, scrollbar.trackH},
                                 Colors::TableHeaderBg,
                                 3.0f);
        const uint32_t thumbColor = draggingParamScrollbar_
            ? Colors::ScrollThumbDrag
            : (hoveredParamScrollbar_ ? Colors::ScrollThumbHover : Colors::ScrollThumb);
        renderer.drawRoundedRect({scrollbar.trackX, scrollbar.thumbY},
                                 {scrollbar.trackW, scrollbar.thumbH},
                                 thumbColor,
                                 3.0f);
        renderer.drawRoundedBorder({scrollbar.trackX, scrollbar.thumbY},
                                   {scrollbar.trackW, scrollbar.thumbH},
                                   Colors::BorderPrimary,
                                   3.0f);
    }
}

// ============================================================
// Param Browser Click Handling
// ============================================================

void UITrainingPanel::processParamBrowserClicks(const InputState& input) {
    using namespace UITheme;

    if (!hyperparamsLoaded_) return;
    if (!input.mousePressed[0]) return;  // left click only

    // Reconstruct the same geometry as drawParamBrowser
    auto content = getContentRect();
    content.origin.y += (kContentTopY - kTabBarY);
    content.size.y -= (kContentTopY - kTabBarY);
    content.size.y -= kBottomBarH;
    constexpr float kParamBrowserW = 350.0f;

    float x = content.origin.x;
    float w = kParamBrowserW;
    float y = content.origin.y;

    // Skip header + dropdown (must match drawParamBrowser layout)
    y += Sizes::HeaderHeight + Spacing::Small;   // "Model Parameters" header
    y += 26.0f + Spacing::Small;                 // category dropdown

    float listH = content.size.y - (y - content.origin.y) - Spacing::Small;
    if (listH < 60.0f) listH = 60.0f;
    float nameColW = w * 0.55f;
    constexpr float rowH = 22.0f;

    Vec2 m = input.mousePos;

    // Check if click is within the list area at all
    if (m.x < x || m.x > x + w || m.y < y || m.y > y + listH) {
        // Clicked outside the param browser — commit any active edit
        if (editingParamIndex_ >= 0) commitParamEdit();
        return;
    }

    // Get the current filtered list
    std::string catFilter;
    if (paramCategoryFilter_ && selectedParamCategory_ > 0) {
        const auto& cats = hyperparamRegistry_.categories();
        if (selectedParamCategory_ - 1 < static_cast<int>(cats.size())) {
            catFilter = cats[selectedParamCategory_ - 1];
        }
    }
    auto params = hyperparamRegistry_.filtered(catFilter);
    const float totalContentH = computeParamBrowserContentHeight(params, catFilter, rowH);
    const ParamScrollbarMetrics scrollbar = computeParamScrollbarMetrics(
        x, y, w, listH, totalContentH, std::clamp(paramScrollOffset_, 0.0f, std::max(0.0f, totalContentH - listH)));

    if (draggingParamScrollbar_ ||
        (scrollbar.visible && pointInRect(input.mousePos, scrollbar.trackX, scrollbar.trackY, scrollbar.trackW, scrollbar.trackH))) {
        return;
    }

    // Find which row was clicked — must match drawParamBrowser geometry exactly
    float drawY = y + 2.0f - paramScrollOffset_;
    std::string lastCategory;
    for (size_t i = 0; i < params.size(); ++i) {
        const auto* entry = params[i];

        // Account for category sub-header rows (when showing "All")
        if (catFilter.empty() && entry->category != lastCategory) {
            lastCategory = entry->category;
            drawY += rowH; // skip category header row
        }

        if (drawY + rowH < y || drawY > y + listH) {
            drawY += rowH;
            continue;  // clipped
        }

        if (m.y >= drawY && m.y < drawY + rowH) {
            bool clickedValue = (m.x >= x + nameColW);

            if (!clickedValue) {
                // Clicked on name column — just commit any open edit
                if (editingParamIndex_ >= 0) commitParamEdit();
                return;
            }

            // Clicked on value column
            if (entry->type == GRIM::Config::HyperparamType::Bool) {
                // Toggle booleans immediately
                if (editingParamIndex_ >= 0) commitParamEdit();
                if (entry->ptr_bool) {
                    *entry->ptr_bool = !(*entry->ptr_bool);
                    persistHyperparamToJSON(*entry);
                }
            } else {
                // Open inline editor for non-bool types
                if (editingParamIndex_ >= 0) commitParamEdit();
                editingParamIndex_ = static_cast<int>(i);
                editParamBuffer_ = entry->valueAsString();
                if (paramEditInput_) {
                    paramEditInput_->setText(editParamBuffer_);
                }
            }
            return;
        }
        drawY += rowH;
    }

    // Click was in list area but not on any row — commit
    if (editingParamIndex_ >= 0) commitParamEdit();
}

// ============================================================
// Commit / Cancel inline param edit
// ============================================================

void UITrainingPanel::commitParamEdit() {
    if (editingParamIndex_ < 0) return;

    // Get filtered list to find the entry
    std::string catFilter;
    if (paramCategoryFilter_ && selectedParamCategory_ > 0) {
        const auto& cats = hyperparamRegistry_.categories();
        if (selectedParamCategory_ - 1 < static_cast<int>(cats.size())) {
            catFilter = cats[selectedParamCategory_ - 1];
        }
    }
    auto params = hyperparamRegistry_.filtered(catFilter);

    if (editingParamIndex_ >= 0 && editingParamIndex_ < static_cast<int>(params.size())) {
        const auto* entry = params[editingParamIndex_];
        const std::string& val = editParamBuffer_;

        try {
            switch (entry->type) {
                case GRIM::Config::HyperparamType::Int:
                    if (entry->ptr_int) *entry->ptr_int = std::stoi(val);
                    break;
                case GRIM::Config::HyperparamType::Int64:
                    if (entry->ptr_int64) *entry->ptr_int64 = std::stoll(val);
                    break;
                case GRIM::Config::HyperparamType::Float:
                    if (entry->ptr_float) *entry->ptr_float = std::stof(val);
                    break;
                case GRIM::Config::HyperparamType::String:
                    if (entry->ptr_string) *entry->ptr_string = val;
                    break;
                case GRIM::Config::HyperparamType::SizeT:
                    if (entry->ptr_sizet) *entry->ptr_sizet = static_cast<size_t>(std::stoull(val));
                    break;
                case GRIM::Config::HyperparamType::Bool:
                    // Bools are toggled on click, not edited via text
                    break;
            }
            persistHyperparamToJSON(*entry);
        } catch (const std::exception&) {
            // Invalid input — discard change silently (value stays as-is)
        }
    }

    editingParamIndex_ = -1;
    editParamBuffer_.clear();
}

void UITrainingPanel::cancelParamEdit() {
    editingParamIndex_ = -1;
    editParamBuffer_.clear();
}

// ============================================================
// Persist single hyperparam value to ai_config.json
// ============================================================

bool UITrainingPanel::persistHyperparamToJSON(const GRIM::Config::HyperparamEntry& entry) {
    try {
        std::ifstream fileIn("ai_config.json");
        nlohmann::json j;
        if (fileIn.is_open()) { fileIn >> j; fileIn.close(); }

        if (!j.contains("training")) j["training"] = nlohmann::json::object();
        if (!j["training"].contains("config")) j["training"]["config"] = nlohmann::json::object();
        auto& tc = j["training"]["config"];

        // Get the value as a JSON type
        nlohmann::json val;
        switch (entry.type) {
            case GRIM::Config::HyperparamType::Bool:   val = entry.ptr_bool   ? *entry.ptr_bool   : false; break;
            case GRIM::Config::HyperparamType::Int:    val = entry.ptr_int    ? *entry.ptr_int    : 0;     break;
            case GRIM::Config::HyperparamType::Int64:  val = entry.ptr_int64  ? *entry.ptr_int64  : 0;     break;
            case GRIM::Config::HyperparamType::Float:  val = entry.ptr_float  ? *entry.ptr_float  : 0.0f;  break;
            case GRIM::Config::HyperparamType::String: val = entry.ptr_string ? *entry.ptr_string : "";    break;
            case GRIM::Config::HyperparamType::SizeT:  val = entry.ptr_sizet  ? static_cast<int64_t>(*entry.ptr_sizet) : 0; break;
        }

        // ── Nested JSON path mapping ──
        // The loader (applyTrainingConfigObject) reads many fields from nested objects.
        // We must write to the same nested path so changes round-trip correctly.
        // Keys not matched here are written as flat keys to training.config.
        const std::string& k = entry.key;

        auto setNested = [&](const std::string& section, const std::string& field) {
            if (!tc.contains(section) || !tc[section].is_object())
                tc[section] = nlohmann::json::object();
            tc[section][field] = val;
        };
        auto setDoubleNested = [&](const std::string& s1, const std::string& s2, const std::string& field) {
            if (!tc.contains(s1) || !tc[s1].is_object())
                tc[s1] = nlohmann::json::object();
            if (!tc[s1].contains(s2) || !tc[s1][s2].is_object())
                tc[s1][s2] = nlohmann::json::object();
            tc[s1][s2][field] = val;
        };

        bool handled = false;

        // Cosine decay
        if (k == "cosine_decay_enabled")  { setNested("cosine_decay", "enabled"); handled = true; }
        if (k == "cosine_decay_min_lr")   { setNested("cosine_decay", "min_lr");  handled = true; }

        // Soft restart
        if (!handled && k.rfind("soft_restart_", 0) == 0) {
            setNested("soft_restart", k.substr(13)); handled = true;
        }
        // Auto stop
        if (!handled && k.rfind("auto_stop_", 0) == 0) {
            setNested("auto_stop", k.substr(10)); handled = true;
        }

        // Shuffle
        if (!handled && k == "shuffle_train_enabled") { setNested("shuffle", "enabled"); handled = true; }
        if (!handled && k == "shuffle_train_epochs")  { setNested("shuffle", "epochs");  handled = true; }
        // Guess aux
        if (!handled && k.rfind("guess_aux_", 0) == 0) {
            setNested("guess_aux", k.substr(10)); handled = true;
        }
        // Loss (double-nested under loss.subsection)
        if (!handled && k.rfind("loss_label_smoothing_", 0) == 0) {
            setDoubleNested("loss", "label_smoothing", k.substr(21)); handled = true;
        }
        if (!handled && k.rfind("loss_focal_", 0) == 0) {
            setDoubleNested("loss", "focal", k.substr(11)); handled = true;
        }
        if (!handled && k.rfind("loss_preference_", 0) == 0) {
            setDoubleNested("loss", "preference", k.substr(16)); handled = true;
        }
        if (!handled && k.rfind("loss_distillation_", 0) == 0) {
            setDoubleNested("loss", "distillation", k.substr(18)); handled = true;
        }
        if (!handled && k.rfind("loss_masking_", 0) == 0) {
            setDoubleNested("loss", "masking", k.substr(13)); handled = true;
        }
        if (!handled && k.rfind("loss_entropy_reg_", 0) == 0) {
            setDoubleNested("loss", "entropy_reg", k.substr(17)); handled = true;
        }
        if (!handled && k.rfind("loss_class_balanced_", 0) == 0) {
            setDoubleNested("loss", "class_balanced", k.substr(20)); handled = true;
        }
        // LM head centering
        if (!handled && k == "lm_head_centering_enabled")       { setNested("lm_head_centering", "enabled"); handled = true; }
        if (!handled && k == "lm_head_center_hidden_states")    { setNested("lm_head_centering", "center_hidden_states"); handled = true; }
        if (!handled && k == "lm_head_freeze_final_rms_gamma")  { setNested("lm_head_centering", "freeze_final_rms_gamma"); handled = true; }
        if (!handled && k == "center_logits")                 { setNested("lm_head_centering", "center_logits"); handled = true; }
        if (!handled && k == "center_encoder_residuals")      { setNested("lm_head_centering", "center_encoder_residuals"); handled = true; }
        if (!handled && k == "project_out_pc1")               { setNested("lm_head_centering", "project_out_pc1"); handled = true; }
        if (!handled && k == "pc1_power_iters")               { setNested("lm_head_centering", "pc1_power_iters"); handled = true; }
        // Layer scale
        if (!handled && k == "use_layer_scale")  { setNested("layer_scale", "enabled");    handled = true; }
        if (!handled && k == "layer_scale_init") { setNested("layer_scale", "init_value"); handled = true; }
        // QK-norm
        if (!handled && k == "qk_norm_enabled") { setNested("qk_norm", "enabled"); handled = true; }
        // Attention diagnostics
        if (!handled && k == "attention_diag_enabled") { setNested("attention_diagnostics", "enabled"); handled = true; }
        if (!handled && k == "attention_diag_layer")   { setNested("attention_diagnostics", "layer");   handled = true; }
        if (!handled && k == "attention_diag_head")    { setNested("attention_diagnostics", "head");    handled = true; }
        // Scratch blocks
        if (!handled && k == "scratch_blocks_enabled")       { setNested("scratch_blocks", "enabled"); handled = true; }
        if (!handled && k == "scratch_max_tokens_per_block") { setNested("scratch_blocks", "max_tokens_per_block"); handled = true; }
        if (!handled && k == "scratch_num_blocks")           { setNested("scratch_blocks", "num_blocks"); handled = true; }
        if (!handled && k == "scratch_write_combined")       { setNested("scratch_blocks", "use_write_combined"); handled = true; }
        // Scratch block reasoning
        if (!handled && k.rfind("scratch_block_reasoning_", 0) == 0) {
            setNested("scratch_block_reasoning", k.substr(24)); handled = true;
        }
        // Execution block
        if (!handled && k.rfind("execution_block_", 0) == 0) {
            setNested("execution_block", k.substr(16)); handled = true;
        }
        if (!handled && k.rfind("execution_step_", 0) == 0) {
            setNested("execution_block", k.substr(10)); handled = true;
        }
        if (!handled && k.rfind("execution_entropy_", 0) == 0) {
            setNested("execution_block", k.substr(10)); handled = true;
        }
        if (!handled && k.rfind("execution_value_", 0) == 0) {
            setNested("execution_block", k.substr(10)); handled = true;
        }
        if (!handled && k.rfind("execution_final_", 0) == 0) {
            setNested("execution_block", k.substr(10)); handled = true;
        }
        // Selector (nested under execution_block.selector)
        if (!handled && k.rfind("selector_", 0) == 0) {
            setDoubleNested("execution_block", "selector", k.substr(9)); handled = true;
        }
        // Embedding freeze
        if (!handled && k == "embedding_freeze_enabled")    { setNested("embedding_freeze", "enabled"); handled = true; }
        if (!handled && k == "embedding_freeze_after_step") { setNested("embedding_freeze", "freeze_after_step"); handled = true; }
        // Stability overrides
        if (!handled && k.rfind("stability_override_", 0) == 0 && k != "stability_overrides_enabled") {
            setNested("stability_overrides", k.substr(19)); handled = true;
        }
        // CUDA execution
        if (!handled && (k == "single_stream_mode" || k == "disable_async_frees" || k == "synchronize_after_kernels")) {
            setNested("cuda_execution", k); handled = true;
        }
        // MTP
        if (!handled && k.rfind("mtp_", 0) == 0) {
            setNested("multi_token_prediction", k.substr(4)); handled = true;
        }
        // Telemetry control (top-level enable; sub-sections like spike_thresholds
        // are deeply nested and written as flat keys as a best-effort fallback)
        if (!handled && k == "telemetry_control_enabled") {
            setNested("telemetry_control", "enabled"); handled = true;
        }
        if (!handled && k == "telemetry_verbose_logging") {
            setDoubleNested("telemetry_control", "logging", "verbose"); handled = true;
        }
        if (!handled && k == "telemetry_fail_loud_on_accumulation_bug") {
            setDoubleNested("telemetry_control", "logging", "fail_loud_on_accumulation_bug"); handled = true;
        }
        if (!handled && k == "telemetry_plateau_noise_enabled") {
            setDoubleNested("telemetry_control", "plateau_noise", "enabled"); handled = true;
        }

        // Fallback: write as flat key (works for core/optimizer fields
        // like learning_rate, epochs, batch_size, warmup_steps, etc.)
        if (!handled) {
            tc[k] = val;
        }

        // Write back
        std::ofstream fileOut("ai_config.json");
        if (!fileOut.is_open()) return false;
        fileOut << std::setw(4) << j << std::endl;
        return true;

    } catch (const std::exception&) {
        return false;
    }
}

// ============================================================
// Training Tab
// ============================================================

void UITrainingPanel::drawTrainingTab(OverlayRenderer& renderer, const PanelRect& content) {
    // ── Split layout: left param browser | right controls ──
    constexpr float kParamBrowserW = 350.0f;
    constexpr float kGutterW = 10.0f;

    float fullW = content.size.x;
    float fullH = content.size.y;

    // Left: parameter browser
    drawParamBrowser(renderer,
                     {content.origin.x, content.origin.y},
                     {kParamBrowserW, fullH});

    // Right: existing training controls
    float rightX = content.origin.x + kParamBrowserW + kGutterW;
    float rightW = fullW - kParamBrowserW - kGutterW - 2.0f * Spacing::PaddingX;

    float x = rightX + Spacing::PaddingX;
    float y = content.origin.y + Spacing::Small;
    float w = rightW;

    // Curriculum selection
    UIDrawHelpers::drawLabeledValue(renderer, {x, y}, "Curriculum:", "");
    y += 20.0f;
    if (curriculumDropdown_) {
        curriculumDropdown_->setPosition(x, y);
        curriculumDropdown_->setSize(w, 26.0f);
        curriculumDropdown_->drawOverlay(renderer, position);
        y += 26.0f + Spacing::Small;
    }

    // Model selection
    UIDrawHelpers::drawLabeledValue(renderer, {x, y}, "Model:", "");
    y += 20.0f;
    if (trainModelDropdown_) {
        trainModelDropdown_->setPosition(x, y);
        trainModelDropdown_->setSize(w, 26.0f);
        trainModelDropdown_->drawOverlay(renderer, position);
        y += 26.0f + Spacing::Medium;
    }

    // Dataset info
    if (!datasetSizeInfo.empty()) {
        renderer.drawText({x, y}, datasetSizeInfo, Colors::Success);
        y += 20.0f;
    }
    if (!checkpointStatsInfo.empty()) {
        renderer.drawText({x, y}, checkpointStatsInfo, Colors::TextSecondary);
        y += 20.0f;
    }
    
    // Tokenizer status
    drawTokenizerStatus(renderer, x, y, w);
    if (tokenizerComplete_ || tokenizerRunning_) {
        y += tokenizerComplete_ && tokenizerSuccess_ ? 50.0f : 28.0f;
    }

    y += Spacing::Medium;

    // ── Section: Monitoring ──
    UIDrawHelpers::drawSectionHeader(renderer, {x - Spacing::PaddingX, y}, w + 2.0f * Spacing::PaddingX,
                                     "Monitoring", Colors::SectionNeutral);
    y += Sizes::HeaderHeight + Spacing::Small;

    // Stat cards
    if (serverConnected) {
        float cardW = (w - 3.0f * kStatCardGap) / 4.0f;
        char buf[64];

        snprintf(buf, sizeof(buf), "%d / %d", currentStats.currentEpoch, currentStats.totalEpochs);
        drawStatCard(renderer, {x, y}, {cardW, kStatCardH}, "Epoch", buf, Colors::AccentBlue);

        snprintf(buf, sizeof(buf), "%d / %d", currentStats.currentBatch, currentStats.totalBatches);
        drawStatCard(renderer, {x + cardW + kStatCardGap, y}, {cardW, kStatCardH}, "Batch", buf, Colors::Primary);

        snprintf(buf, sizeof(buf), "%.4f", currentStats.currentLoss);
        drawStatCard(renderer, {x + 2.0f * (cardW + kStatCardGap), y}, {cardW, kStatCardH}, "Loss", buf, Colors::Warning);

        snprintf(buf, sizeof(buf), "%.2f", currentStats.perplexity);
        drawStatCard(renderer, {x + 3.0f * (cardW + kStatCardGap), y}, {cardW, kStatCardH}, "Perplexity", buf, Colors::Success);

        y += kStatCardH + Spacing::Medium;
    } else {
        renderer.drawText({x, y + 10.0f}, "Server offline — connect to see metrics", Colors::TextMuted);
        y += 40.0f;
    }

    // Checkpoint merge status
    if (!checkpointMergeStatus.empty()) {
        renderer.drawText({x, y}, checkpointMergeStatus, Colors::Warning);
        y += 22.0f;
    }

    // Progress bar
    float barH = Sizes::ProgressBarHeight;
    if (trainingProgressBar) {
        float progress = currentStats.trainingProgress / 100.0f;
        trainingProgressBar->setValue(progress);
        trainingProgressBar->setPosition({x, y});
        trainingProgressBar->setSize({w, barH});
        trainingProgressBar->drawOverlay(renderer, position);
        y += barH + Spacing::Medium;
    }

    // Training Logs
    UIDrawHelpers::drawSectionHeader(renderer, {x - Spacing::PaddingX, y}, w + 2.0f * Spacing::PaddingX,
                                     "Training Logs", Colors::SectionNeutral);
    y += Sizes::HeaderHeight;

    float logAreaH = content.origin.y + content.size.y - y - Spacing::Small;
    if (logAreaH < 40.0f) logAreaH = 40.0f;

    renderer.drawRoundedRect({x, y}, {w, logAreaH}, Colors::ContentAreaBg, Sizes::WidgetRadius);
    renderer.drawRoundedBorder({x, y}, {w, logAreaH}, Colors::BorderSubtle, Sizes::WidgetRadius);

    {
        std::lock_guard<std::mutex> lock(logMutex);
        float logY = y + 4.0f;
        int visibleLines = static_cast<int>(logAreaH / kLogLineH);
        int startIdx = std::max(0, static_cast<int>(logEntries.size()) - visibleLines);

        for (size_t i = startIdx; i < logEntries.size(); i++) {
            const auto& entry = logEntries[i];
            uint32_t color = Colors::Success;
            if (entry.level == 1) color = Colors::Warning;
            else if (entry.level == 2) color = Colors::Danger;

            renderer.drawText({x + Spacing::Small, logY}, entry.timestamp + " " + entry.message, color);
            logY += kLogLineH;
            if (logY > y + logAreaH - 4.0f) break;
        }
    }

    configContentHeight = y - content.origin.y;
}

// ============================================================
// Bottom Action Bar
// ============================================================

void UITrainingPanel::drawBottomBar(OverlayRenderer& renderer, float barY, float barWidth, float barX) {
    float btnW = 90.0f;
    float btnH = Sizes::ButtonHeight;
    float gap = Spacing::Small;
    float totalW = btnW * 5.0f + gap * 4.0f;
    float startX = barX + (barWidth - totalW) / 2.0f;

    auto placeBtn = [&](std::shared_ptr<UIButton>& btn, int idx) {
        if (!btn) return;
        float bx = startX + idx * (btnW + gap);
        btn->setPosition(bx, barY);
        btn->setSize(btnW, btnH);
        btn->drawOverlay(renderer, position);
    };

    placeBtn(startButton, 0);
    placeBtn(stopButton, 1);
    placeBtn(pauseResumeButton, 2);
    placeBtn(resetStatusButton, 3);
    placeBtn(closeButton, 4);
}

// ============================================================
// Stat Card Helper
// ============================================================

void UITrainingPanel::drawStatCard(OverlayRenderer& renderer, const Vec2& pos, const Vec2& sz,
                                    const std::string& label, const std::string& value, uint32_t accentColor) {
    renderer.drawRoundedRect(pos, sz, Colors::CardSurface, Sizes::WidgetRadius);
    renderer.drawRoundedBorder(pos, sz, Colors::BorderSubtle, Sizes::WidgetRadius);
    renderer.drawRect({pos.x, pos.y + 6.0f}, {3.0f, sz.y - 12.0f}, accentColor);
    renderer.drawText({pos.x + 12.0f, pos.y + 8.0f}, label, Colors::TextSecondary);
    renderer.drawText({pos.x + 12.0f, pos.y + 30.0f}, value, Colors::TextPrimary);
}

// ============================================================
// Public Control Functions
// ============================================================

void UITrainingPanel::startTrainingSession() {
    if (!trainingController) { addLog("Cannot start: controller not initialized", 2); return; }
    if (!serverConnected) {
        addLog("Server not connected, checking...", 0);
        pollServer();
        if (!serverConnected) { addLog("Cannot start: server not connected", 2); return; }
    }
    if (currentState == Control::TrainingState_Training) {
        addLog("Training already in progress", 1); return;
    }

    updateConfigFromSliders();
    currentStats = TrainingStats();
    if (trainingProgressBar) trainingProgressBar->setValue(0.0f);

    std::string grmtPath = "resources/models/GRIM-text/training/data/training_data.grmt";
    std::ifstream checkGrmt(grmtPath, std::ios::binary | std::ios::ate);
    bool hasGrmt = checkGrmt.is_open() && checkGrmt.tellg() > 0;
    checkGrmt.close();

    if (!hasGrmt) {
        addLog("No training data found — run data pipeline first", 2);
        addLog("Expected: " + grmtPath, 1);
        currentState = Control::TrainingState_Idle;
        return;
    }

    addLog("Training data found", 0);
    currentConfig.dataPath = "data/training_data.grmt";

    GRIM::Config::GrimTextPaths paths;
    if (GRIM::Config::loadGrimTextPaths(paths)) {
        currentConfig.vocabPath = paths.vocab;
        currentConfig.outputPath = paths.model;
        addLog("Vocab: " + currentConfig.vocabPath, 0);
        addLog("Output: " + currentConfig.outputPath, 0);
    } else {
        addLog("Failed to load paths from config, using defaults", 1);
    }

    checkpointMergeStatus = "";
    if (trainingController && trainingController->startTraining(currentConfig)) {
        addLog("Training launched — monitoring progress", 0);
        currentState = Control::TrainingState_Training;
    } else {
        std::string error = "Failed to start training";
        if (trainingController) error += ": " + trainingController->getLastError();
        addLog(error, 2);
        currentState = Control::TrainingState_Idle;
    }
}

void UITrainingPanel::stopTrainingSession() {
    if (!trainingController || !serverConnected) { addLog("Cannot stop: not connected", 2); return; }
    if (currentState != Control::TrainingState_Training && currentState != Control::TrainingState_Paused) {
        addLog("No active session to stop", 1); return;
    }
    addLog("Stopping training...", 0);
    if (trainingController->stopTraining()) {
        addLog("Training stopped", 0);
        currentState = Control::TrainingState_Idle;
        if (trainingProgressBar) trainingProgressBar->setValue(0.0f);
    } else {
        addLog("Failed to stop: " + trainingController->getLastError(), 2);
    }
}

void UITrainingPanel::pauseTrainingSession() {
    if (!trainingController || !serverConnected) { addLog("Cannot pause: not connected", 2); return; }
    if (currentState != Control::TrainingState_Training) { addLog("Cannot pause: not training", 1); return; }
    addLog("Pausing training...", 1);
    currentState = Control::TrainingState_Paused;
}

void UITrainingPanel::resumeTrainingSession() {
    if (!trainingController || !serverConnected) { addLog("Cannot resume: not connected", 2); return; }
    if (currentState != Control::TrainingState_Paused) { addLog("Cannot resume: not paused", 1); return; }
    addLog("Resuming training...", 0);
    currentState = Control::TrainingState_Training;
}

void UITrainingPanel::shutdownTrainingServer() {
    if (!trainingController) { addLog("Cannot shutdown: controller not initialized", 2); return; }
    if (serverConnected && currentState == Control::TrainingState_Training) stopTrainingSession();
    addLog("Shutting down server...", 1);
    if (trainingController->stopServer()) {
        addLog("Server shutdown successful", 0);
        serverConnected = false;
        currentState = Control::TrainingState_Idle;
    } else {
        addLog("Failed to shutdown: " + trainingController->getLastError(), 2);
    }
}

void UITrainingPanel::handleStartTraining() { startTrainingSession(); }
void UITrainingPanel::handleStopTraining() { stopTrainingSession(); }
void UITrainingPanel::handlePauseResume() {
    if (currentState == Control::TrainingState_Training) pauseTrainingSession();
    else if (currentState == Control::TrainingState_Paused) resumeTrainingSession();
}

// ============================================================
// Logging
// ============================================================

void UITrainingPanel::addLog(const std::string& message, int level) {
    std::lock_guard<std::mutex> lock(logMutex);

    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);

    char timeBuf[32];
    std::strftime(timeBuf, sizeof(timeBuf), "%H:%M:%S", &tm);

    LogEntry entry;
    entry.timestamp = timeBuf;
    entry.message = message;
    entry.level = level;
    logEntries.push_back(entry);

    if (logEntries.size() > maxLogEntries) logEntries.erase(logEntries.begin());

    if (level == 1) LOG_TRACE("TrainingPanel", message);
    else if (level == 2) LOG_ERROR("TrainingPanel", message);
}

// ============================================================
// State Helpers
// ============================================================

std::string UITrainingPanel::getStateString(Control::TrainingState state) const {
    switch (state) {
        case Control::TrainingState_Idle:       return "Idle";
        case Control::TrainingState_Collecting:  return "Collecting";
        case Control::TrainingState_Verifying:   return "Verifying";
        case Control::TrainingState_Training:    return "Training";
        case Control::TrainingState_Paused:      return "Paused";
        case Control::TrainingState_Completed:   return "Completed";
        case Control::TrainingState_Error:       return "Error";
        default: return "Unknown";
    }
}

uint32_t UITrainingPanel::getStateColor(Control::TrainingState state) const {
    switch (state) {
        case Control::TrainingState_Idle:       return Colors::TextMuted;
        case Control::TrainingState_Collecting:  return Colors::AccentBlue;
        case Control::TrainingState_Verifying:   return Colors::Info;
        case Control::TrainingState_Training:    return Colors::Success;
        case Control::TrainingState_Paused:      return Colors::Warning;
        case Control::TrainingState_Completed:   return Colors::Primary;
        case Control::TrainingState_Error:       return Colors::Danger;
        default: return Colors::TextPrimary;
    }
}

// ============================================================
// Config Persistence
// ============================================================

void UITrainingPanel::loadConfigFromJSON() {
    currentConfig = TrainingConfigManager::loadFromJSON();
    updateSlidersFromConfig();
    addLog("Configuration loaded from ai_config.json", 0);
}

void UITrainingPanel::saveConfigToJSON() {
    updateConfigFromSliders();
    if (TrainingConfigManager::saveToJSON(currentConfig))
        addLog("Configuration saved", 0);
    else
        addLog("Failed to save configuration", 2);
}

void UITrainingPanel::updateConfigFromSliders() {
    if (epochsSlider) currentConfig.epochs = static_cast<int>(epochsSlider->getValue());
    if (batchSizeSlider) currentConfig.batchSize = static_cast<int>(batchSizeSlider->getValue());
    if (learningRateSlider) currentConfig.learningRate = learningRateSlider->getValue();
    if (maxSeqLenSlider) currentConfig.maxSeqLen = static_cast<int>(maxSeqLenSlider->getValue());
    if (warmupStepsSlider) currentConfig.warmupSteps = static_cast<int>(warmupStepsSlider->getValue());
}

void UITrainingPanel::updateSlidersFromConfig() {
    if (epochsSlider) epochsSlider->setValue(static_cast<float>(currentConfig.epochs));
    if (batchSizeSlider) batchSizeSlider->setValue(static_cast<float>(currentConfig.batchSize));
    if (learningRateSlider) learningRateSlider->setValue(currentConfig.learningRate);
    if (maxSeqLenSlider) maxSeqLenSlider->setValue(static_cast<float>(currentConfig.maxSeqLen));
    if (warmupStepsSlider) warmupStepsSlider->setValue(static_cast<float>(currentConfig.warmupSteps));
}

// ============================================================
// Hardware & Estimation
// ============================================================

void UITrainingPanel::updateHardwareInfo() {
    std::stringstream ss;
    if (g_hardwareInventory.hasGPU() && g_hardwareInventory.hasCUDA()) {
        std::string gpuName = g_hardwareInventory.gpus.empty() ? "Unknown" : g_hardwareInventory.gpus[0].name;
        long vramMB = g_hardwareInventory.gpus.empty() ? 0 : g_hardwareInventory.gpus[0].vram_mb;
        ss << "GPU: " << gpuName << " (" << vramMB << " MB VRAM)\nCUDA: Available\n";
    } else if (g_hardwareInventory.hasGPU()) {
        std::string gpuName = g_hardwareInventory.gpus.empty() ? "Unknown" : g_hardwareInventory.gpus[0].name;
        ss << "GPU: " << gpuName << " (No CUDA)\n";
    } else {
        ss << "GPU: None (CPU training only)\n";
    }
    ss << "CPU: " << g_hardwareInventory.cpu_cores << " cores\n";
    ss << "RAM: " << g_hardwareInventory.ram_total_mb << " MB";
    hardwareInfo = ss.str();
}

void UITrainingPanel::calculateTrainingEstimate() {
    float timePerBatch = 1.0f;

    if (g_hardwareInventory.hasGPU() && g_hardwareInventory.hasCUDA()) {
        timePerBatch = 0.15f;
        long vramMB = g_hardwareInventory.gpus.empty() ? 0 : g_hardwareInventory.gpus[0].vram_mb;
        if (vramMB >= 12000) timePerBatch *= 0.8f;
        else if (vramMB >= 8000) timePerBatch *= 0.9f;
        else if (vramMB >= 4000) timePerBatch *= 1.1f;
    } else {
        timePerBatch = 8.0f;
        if (g_hardwareInventory.cpu_cores >= 16) timePerBatch *= 0.6f;
        else if (g_hardwareInventory.cpu_cores >= 8) timePerBatch *= 0.75f;
    }

    timePerBatch *= (currentConfig.maxSeqLen / 512.0f);

    if (currentConfig.batchSize >= 32) timePerBatch *= 1.5f;
    else if (currentConfig.batchSize >= 16) timePerBatch *= 1.2f;
    else if (currentConfig.batchSize >= 4) timePerBatch *= 0.7f;
    else timePerBatch *= 0.5f;

    int estimatedDatasetSize = 500;
    std::string sourcePath = getResourcePath() + "/models/GRIM-text/training/source_data.json";
    try {
        if (std::filesystem::exists(sourcePath)) {
            std::ifstream sourceFile(sourcePath);
            nlohmann::json sourceData;
            sourceFile >> sourceData;
            sourceFile.close();

            int enabledSourceCount = 0;
            int totalFetchLimit = 0;
            if (sourceData.contains("data_sources")) {
                for (const auto& source : sourceData["data_sources"]) {
                    if (source.value("enabled", false)) {
                        enabledSourceCount++;
                        totalFetchLimit += source.value("fetch_limit", 100);
                    }
                }
            }
            if (totalFetchLimit > 0)
                estimatedDatasetSize = static_cast<int>(totalFetchLimit * 0.6f);
            else if (enabledSourceCount > 0)
                estimatedDatasetSize = enabledSourceCount * 200;
        }
    } catch (...) {}

    int batchesPerEpoch = estimatedDatasetSize / std::max(1, currentConfig.batchSize);
    int totalBatches = batchesPerEpoch * currentConfig.epochs;
    estimatedTrainingTimeSeconds = totalBatches * timePerBatch;
    if (currentConfig.warmupSteps > 0)
        estimatedTrainingTimeSeconds += currentConfig.warmupSteps * timePerBatch;

    std::stringstream ss;
    int hours = static_cast<int>(estimatedTrainingTimeSeconds / 3600);
    int minutes = static_cast<int>((estimatedTrainingTimeSeconds - hours * 3600) / 60);
    int seconds = static_cast<int>(estimatedTrainingTimeSeconds) % 60;
    if (hours > 0) ss << hours << "h " << minutes << "m " << seconds << "s";
    else if (minutes > 0) ss << minutes << "m " << seconds << "s";
    else ss << seconds << "s";
    estimatedTimeStr = ss.str();
}

// ============================================================
// Dataset Size & Checkpoint I/O
// ============================================================

void UITrainingPanel::updateDatasetSize() {
    datasetSizeInfo = readDatasetSizeSnapshot();
}

void UITrainingPanel::updateCheckpointStats() {
    checkpointStatsInfo = readCheckpointStatsSnapshot();
}

std::string UITrainingPanel::readDatasetSizeSnapshot() {
    GRIM::Config::GrimTextPaths paths;
    if (!GRIM::Config::loadGrimTextPaths(paths))
        return "Dataset: Config error";

    const std::string grmtPath = paths.training_data;
    try {
        if (std::filesystem::exists(grmtPath)) {
            auto fileSize = std::filesystem::file_size(grmtPath);
            std::stringstream ss;
            if (fileSize >= 1024ull * 1024ull * 1024ull)
                ss << std::fixed << std::setprecision(2) << (fileSize / (1024.0 * 1024.0 * 1024.0)) << " GB";
            else if (fileSize >= 1024ull * 1024ull)
                ss << std::fixed << std::setprecision(2) << (fileSize / (1024.0 * 1024.0)) << " MB";
            else if (fileSize >= 1024ull)
                ss << std::fixed << std::setprecision(2) << (fileSize / 1024.0) << " KB";
            else
                ss << fileSize << " bytes";

            int estimatedSamples = static_cast<int>(fileSize / 2048);
            return "Dataset: " + ss.str() + " (~" + std::to_string(estimatedSamples) + " samples)";
        }
        return "Dataset: Not found";
    } catch (const std::exception& e) {
        return "Dataset: Error reading size";
    }
}

std::string UITrainingPanel::readCheckpointStatsSnapshot() {
    std::string checkpointDir = getResourcePath() + "/models/GRIM-text/DataCollection/data";
    try {
        if (!std::filesystem::exists(checkpointDir))
            return "Raw Data: No checkpoints";

        int totalEntries = 0;
        uintmax_t totalSize = 0;
        std::filesystem::path latestCheckpoint;
        std::filesystem::file_time_type latestTime;
        bool hasCheckpoints = false;

        for (const auto& entry : std::filesystem::directory_iterator(checkpointDir)) {
            if (entry.is_regular_file() && entry.path().filename().string().rfind("checkpoint_", 0) == 0) {
                totalSize += entry.file_size();
                auto modTime = entry.last_write_time();
                if (!hasCheckpoints || modTime > latestTime) {
                    latestTime = modTime;
                    latestCheckpoint = entry.path();
                    hasCheckpoints = true;
                }
            }
        }
        if (!hasCheckpoints) return "Raw Data: No checkpoints";

        std::ifstream latestFile(latestCheckpoint);
        if (latestFile.is_open()) {
            totalEntries = std::count(std::istreambuf_iterator<char>(latestFile),
                                      std::istreambuf_iterator<char>(), '\n');
            latestFile.close();
        }

        std::stringstream ss;
        ss << "Raw Data: " << totalEntries << " entries";
        if (totalSize >= 1024ull * 1024ull)
            ss << " (" << std::fixed << std::setprecision(1) << (totalSize / (1024.0 * 1024.0)) << " MB)";
        else if (totalSize >= 1024ull)
            ss << " (" << std::fixed << std::setprecision(1) << (totalSize / 1024.0) << " KB)";
        return ss.str();
    } catch (const std::exception& e) {
        return "Raw Data: Error reading checkpoints";
    }
}

void UITrainingPanel::requestDatasetSnapshot() {
    if (datasetSnapshotInFlight.exchange(true)) return;

    std::thread([this]() {
        DatasetSnapshotResult snapshot;
        snapshot.datasetInfo = readDatasetSizeSnapshot();
        snapshot.checkpointInfo = readCheckpointStatsSnapshot();
        {
            std::lock_guard<std::mutex> lock(datasetSnapshotMutex);
            pendingDatasetSnapshot = std::move(snapshot);
        }
        datasetSnapshotInFlight.store(false);
    }).detach();
}

void UITrainingPanel::applyPendingDatasetSnapshot() {
    std::optional<DatasetSnapshotResult> snapshot;
    {
        std::lock_guard<std::mutex> lock(datasetSnapshotMutex);
        if (!pendingDatasetSnapshot.has_value()) return;
        snapshot = std::move(pendingDatasetSnapshot);
        pendingDatasetSnapshot.reset();
    }
    if (snapshot) {
        datasetSizeInfo = snapshot->datasetInfo;
        checkpointStatsInfo = snapshot->checkpointInfo;
    }
}

// ============================================================
// Focus Queries
// ============================================================

bool UITrainingPanel::isAnySliderEditing() const {
    if (epochsSlider && epochsSlider->isEditing()) return true;
    if (batchSizeSlider && batchSizeSlider->isEditing()) return true;
    if (learningRateSlider && learningRateSlider->isEditing()) return true;
    if (maxSeqLenSlider && maxSeqLenSlider->isEditing()) return true;
    if (warmupStepsSlider && warmupStepsSlider->isEditing()) return true;
    return false;
}

bool UITrainingPanel::isAnyInputEditing() const {
    if (activeTab_ == TrainingPanelTab::Tokenizer) {
        return encodeInputBox_ && encodeInputBox_->isFocused();
    }
    if (activeTab_ != TrainingPanelTab::Home || !showCreatorForm_) return false;
    return creatorNameInput_->isFocused()
        || creatorParamInput_->isFocused()
        || creatorSubjectInput_->isFocused()
        || creatorTagsInput_->isFocused()
        || creatorDescInput_->isFocused();
}

// ============================================================
// Resource Monitoring
// ============================================================

void UITrainingPanel::updateResourceMonitoring(float dt) {
    resourceSampleTimer += dt;
    if (resourceSampleTimer < resourceSampleInterval) return;
    resourceSampleTimer = 0.0f;

    ResourceMonitor::getInstance().update();
    ResourceUsage usage = ResourceMonitor::getInstance().getCurrentUsage();
    std::string label = std::to_string(resourceSampleCount);

    cpuHistory.push_back(DataPoint(usage.cpuUsage, label));
    if (cpuHistory.size() > static_cast<size_t>(maxResourceSamples)) cpuHistory.erase(cpuHistory.begin());

    memoryHistory.push_back(DataPoint(usage.memoryUsage, label));
    if (memoryHistory.size() > static_cast<size_t>(maxResourceSamples)) memoryHistory.erase(memoryHistory.begin());

    gpuHistory.push_back(DataPoint(usage.gpuUsage, label));
    if (gpuHistory.size() > static_cast<size_t>(maxResourceSamples)) gpuHistory.erase(gpuHistory.begin());

    if (resourceMonitorGraph) {
        resourceMonitorGraph->clearSeries();
        resourceMonitorGraph->addSeries("CPU", cpuHistory, Colors::AccentBlue);
        resourceMonitorGraph->addSeries("Memory", memoryHistory, Colors::Success);
        resourceMonitorGraph->addSeries("GPU", gpuHistory, Colors::Warning);
    }
    resourceSampleCount++;
}

// ============================================================
// Path Configuration I/O
// ============================================================

void UITrainingPanel::loadPathsFromConfig() {
    try {
        std::ifstream configFile("ai_config.json");
        if (!configFile.is_open()) return;

        nlohmann::json config;
        configFile >> config;

        if (config.contains("paths") && config["paths"].contains("grim_text")) {
            auto& paths = config["paths"]["grim_text"];
            if (paths.contains("vocab")) vocabPathBuffer = paths["vocab"].get<std::string>();
            if (paths.contains("model")) modelPathBuffer = paths["model"].get<std::string>();
            if (paths.contains("checkpoints")) checkpointsPathBuffer = paths["checkpoints"].get<std::string>();
            if (paths.contains("logs")) logsPathBuffer = paths["logs"].get<std::string>();
        }
    } catch (const std::exception& e) {
        LOG_ERROR("UITrainingPanel", "Error loading paths: " + std::string(e.what()));
    }
}

static std::string makeRelativeToGrimRoot(const std::string& pathStr) {
    namespace fs = std::filesystem;
    if (pathStr.empty()) return pathStr;
    fs::path path(pathStr);
    if (!path.is_absolute()) return pathStr;

    fs::path grimRoot = fs::current_path();
    for (int i = 0; i < 10 && grimRoot.has_parent_path(); ++i) {
        if (fs::exists(grimRoot / "control") && fs::exists(grimRoot / "resources")) break;
        grimRoot = grimRoot.parent_path();
    }
    try {
        fs::path relativePath = fs::relative(path, grimRoot);
        if (!relativePath.empty()) {
            const std::string relative = relativePath.string();
            if (relative.size() < 2 || relative.compare(0, 2, "..") != 0) {
                return relative;
            }
        }
    } catch (...) {}
    return pathStr;
}

void UITrainingPanel::savePathsToConfig() {
    try {
        std::ifstream configFileIn("ai_config.json");
        if (!configFileIn.is_open()) return;

        nlohmann::json config;
        configFileIn >> config;
        configFileIn.close();

        if (!config.contains("paths")) config["paths"] = nlohmann::json::object();
        if (!config["paths"].contains("grim_text")) config["paths"]["grim_text"] = nlohmann::json::object();

        config["paths"]["grim_text"]["vocab"] = makeRelativeToGrimRoot(vocabPathBuffer);
        config["paths"]["grim_text"]["model"] = makeRelativeToGrimRoot(modelPathBuffer);
        config["paths"]["grim_text"]["checkpoints"] = makeRelativeToGrimRoot(checkpointsPathBuffer);
        config["paths"]["grim_text"]["logs"] = makeRelativeToGrimRoot(logsPathBuffer);

        auto preserveRelative = [&](const std::string& key) {
            if (config["paths"]["grim_text"].contains(key))
                config["paths"]["grim_text"][key] = makeRelativeToGrimRoot(config["paths"]["grim_text"][key].get<std::string>());
        };
        preserveRelative("collected");
        preserveRelative("verified");
        preserveRelative("collector_log");
        preserveRelative("merge_checkpoints_exe");
        preserveRelative("source_config");

        std::ofstream configFileOut("ai_config.json");
        if (!configFileOut.is_open()) return;
        configFileOut << config.dump(4);
        configFileOut.close();
    } catch (const std::exception& e) {
        LOG_ERROR("UITrainingPanel", "Error saving paths: " + std::string(e.what()));
    }
}

// ============================================================
// Dropdown Refresh
// ============================================================

void UITrainingPanel::refreshCurriculumDropdown() {
    if (!curriculumDropdown_) return;
    if (trainingDatasetTarget_) trainingDatasetTarget_->loadCurriculumRegistry();

    std::vector<std::string> names;
    names.push_back("(none)");
    int selectedIdx = 0;
    if (trainingDatasetTarget_) {
        const auto& curricula = trainingDatasetTarget_->getCurriculums();
        for (size_t i = 0; i < curricula.size(); ++i) {
            names.push_back(curricula[i].name);
            if (curricula[i].id == selectedCurriculumId_)
                selectedIdx = static_cast<int>(i) + 1;
        }
    }
    curriculumDropdown_->setItems(names);
    curriculumDropdown_->setSelectedIndex(selectedIdx);
}

void UITrainingPanel::refreshModelDropdown() {
    if (!trainModelDropdown_) return;

    std::vector<std::string> names;
    names.push_back("(none)");
    int selectedIdx = 0;
    const auto models = GRIM::MMO::ModelRegistry::instance().getAllModels();
    for (size_t i = 0; i < models.size(); ++i) {
        names.push_back(models[i]->name);
        if (models[i]->id == selectedTrainModelId_)
            selectedIdx = static_cast<int>(i) + 1;
    }
    trainModelDropdown_->setItems(names);
    trainModelDropdown_->setSelectedIndex(selectedIdx);
}

void UITrainingPanel::refreshTrainingDropdowns() {
    refreshCurriculumDropdown();
    refreshModelDropdown();
}

// ============================================================
// Tokenizer Runner
// ============================================================

void UITrainingPanel::handleRunTokenizer() {
    if (tokenizerRunning_) {
        addLog("Tokenizer already running", 1);
        return;
    }
    if (!trainingController) {
        addLog("Cannot run tokenizer: controller not initialized", 2);
        return;
    }
    if (!serverConnected) {
        pollServer();
        if (!serverConnected) {
            addLog("Cannot run tokenizer: server not connected", 2);
            return;
        }
    }
    
    tokenizerRunning_ = true;
    tokenizerComplete_ = false;
    tokenizerSuccess_ = false;
    tokenizerStatusMessage_ = "Running tokenizer validation...";
    addLog("Starting tokenizer validation...", 0);
    
    // Run async to avoid blocking UI
    std::thread([this]() {
        auto result = trainingController->runTokenizer();
        
        lastTokenizerResult_ = result;
        tokenizerSuccess_ = result.success;
        tokenizerComplete_ = true;
        tokenizerRunning_ = false;
        
        if (result.success) {
            tokenizerStatusMessage_ = "Tokenizer OK: " + 
                std::to_string(result.total_vocab_size) + " tokens (" +
                std::to_string(result.validation_tests_passed) + "/" +
                std::to_string(result.validation_tests_total) + " tests passed)";
            addLog(tokenizerStatusMessage_, 0);
        } else {
            tokenizerStatusMessage_ = "Tokenizer FAILED: " + result.error;
            addLog(tokenizerStatusMessage_, 2);
            for (const auto& f : result.failures) {
                addLog("  - " + f, 2);
            }
        }
    }).detach();
}

void UITrainingPanel::drawTokenizerStatus(OverlayRenderer& renderer, float x, float y, float width) {
    if (!tokenizerComplete_ && !tokenizerRunning_) return;
    
    uint32_t bgColor = tokenizerRunning_ ? Colors::ContentAreaBg :
                        (tokenizerSuccess_ ? 0xFF1A3A1A : 0xFF3A1A1A);
    uint32_t textColor = tokenizerRunning_ ? Colors::TextSecondary :
                          (tokenizerSuccess_ ? Colors::Success : Colors::Danger);
    
    float h = 24.0f;
    renderer.drawRoundedRect({x, y}, {width, h}, bgColor, Sizes::WidgetRadius);
    renderer.drawText({x + Spacing::Small, y + 4.0f}, tokenizerStatusMessage_, textColor);
    
    if (tokenizerComplete_ && tokenizerSuccess_) {
        float detailY = y + h + 2.0f;
        auto& r = lastTokenizerResult_;
        std::string details = "Vocab: " + std::to_string(r.unigram_vocab_size) + " unigram + " +
            std::to_string(r.byte_vocab_size) + " byte + " +
            std::to_string(r.atom_vocab_size) + " atom | " +
            "Load: " + std::to_string(static_cast<int>(r.load_time_ms)) + "ms | " +
            "Val: " + std::to_string(static_cast<int>(r.validation_time_ms)) + "ms";
        renderer.drawText({x + Spacing::Small, detailY}, details, Colors::TextSecondary);
    }
}

// ============================================================
// Tokenizer Tab
// ============================================================

void UITrainingPanel::drawTokenizerTab(OverlayRenderer& renderer, const PanelRect& content) {
    float x = content.origin.x + Spacing::PaddingX;
    float y = content.origin.y + Spacing::Small;
    float w = content.size.x - 2.0f * Spacing::PaddingX;

    // ── Section 1: Validation ──
    UIDrawHelpers::drawSectionHeader(renderer, {x - Spacing::PaddingX, y}, content.size.x,
                                     "Tokenizer Validation", Colors::SectionAI);
    y += Sizes::HeaderHeight + Spacing::Small;

    // Run Validation button
    tokenizerRunValidationBtn_->setPosition(x, y);
    tokenizerRunValidationBtn_->drawOverlay(renderer, position);

    // Running indicator
    if (tokenizerRunning_) {
        renderer.drawText({x + 130.0f, y + 8.0f}, "Running...", Colors::Warning);
    }

    y += Sizes::ButtonHeight + Spacing::Medium;

    // Validation status
    drawTokenizerStatus(renderer, x, y, w);
    if (tokenizerComplete_ || tokenizerRunning_) {
        y += 24.0f + Spacing::Small;
        if (tokenizerComplete_ && tokenizerSuccess_) {
            y += 18.0f + Spacing::Small; // detail line
        }
    }

    // Validation detail: test failures
    if (tokenizerComplete_ && !tokenizerSuccess_ && !lastTokenizerResult_.failures.empty()) {
        float failY = y;
        renderer.drawText({x, failY}, "Failed tests:", Colors::Danger);
        failY += 18.0f;
        for (const auto& f : lastTokenizerResult_.failures) {
            if (failY > content.origin.y + content.size.y - 40.0f) break;
            renderer.drawText({x + Spacing::Medium, failY}, "- " + f, Colors::TextSecondary);
            failY += 16.0f;
        }
        y = failY + Spacing::Small;
    }

    // Vocab info cards (show after successful validation)
    if (tokenizerComplete_ && tokenizerSuccess_) {
        auto& r = lastTokenizerResult_;
        float cardW = (w - 2.0f * kStatCardGap) / 3.0f;

        drawStatCard(renderer, {x, y}, {cardW, kStatCardH},
                     "Total Vocab", std::to_string(r.total_vocab_size), Colors::Primary);
        drawStatCard(renderer, {x + cardW + kStatCardGap, y}, {cardW, kStatCardH},
                     "Unigram", std::to_string(r.unigram_vocab_size), Colors::AccentBlue);
        drawStatCard(renderer, {x + 2.0f * (cardW + kStatCardGap), y}, {cardW, kStatCardH},
                     "Byte + Atom", std::to_string(r.byte_vocab_size + r.atom_vocab_size), Colors::Warning);
        y += kStatCardH + Spacing::Large;

        // Special token IDs
        std::string specialStr = "PAD=" + std::to_string(r.pad_id) +
            "  UNK=" + std::to_string(r.unk_id) +
            "  BOS=" + std::to_string(r.bos_id) +
            "  EOS=" + std::to_string(r.eos_id);
        renderer.drawText({x, y}, specialStr, Colors::TextSecondary);
        y += 18.0f + Spacing::Small;
    }

    y += Spacing::Medium;

    // ── Section 2: Encode Text ──
    UIDrawHelpers::drawSectionHeader(renderer, {x - Spacing::PaddingX, y}, content.size.x,
                                     "Encode Text", Colors::SectionNeutral);
    y += Sizes::HeaderHeight + Spacing::Small;

    // Input box + buttons row
    float inputW = w - 90.0f - 70.0f - 2.0f * Spacing::Small;
    encodeInputBox_->setPosition(x, y);
    encodeInputBox_->setSize(inputW, 28.0f);
    encodeInputBox_->drawOverlay(renderer, position);

    encodeButton_->setPosition(x + inputW + Spacing::Small, y - 2.0f);
    encodeButton_->drawOverlay(renderer, position);

    clearEncodeButton_->setPosition(x + inputW + Spacing::Small + 90.0f + Spacing::Small, y - 2.0f);
    clearEncodeButton_->drawOverlay(renderer, position);

    if (encodeRunning_.load()) {
        renderer.drawText({x, y + 32.0f}, "Encoding...", Colors::Warning);
    }

    y += 28.0f + Spacing::Medium;

    // ── Encode results ──
    float remainingH = (content.origin.y + content.size.y) - y - Spacing::Small;
    if (remainingH > 40.0f) {
        drawEncodeResults(renderer, x, y, w, remainingH);
    }
}

void UITrainingPanel::drawEncodeResults(OverlayRenderer& renderer, float x, float y, float width, float maxHeight) {
    if (!encodeComplete_ && !encodeRunning_.load()) return;

    if (!encodeSuccess_ && encodeComplete_) {
        // Error display
        renderer.drawRoundedRect({x, y}, {width, 24.0f}, 0xFF3A1A1A, Sizes::WidgetRadius);
        renderer.drawText({x + Spacing::Small, y + 4.0f},
                          "Encode failed: " + encodeErrorMessage_, Colors::Danger);
        return;
    }

    if (!encodeComplete_) return;

    std::lock_guard<std::mutex> lock(encodeMutex_);
    auto& r = lastEncodeResult_;

    // Summary bar
    std::string summary = std::to_string(r.token_count) + " tokens | " +
        std::to_string(static_cast<int>(r.encode_time_ms * 1000.0)) + "us encode";
    if (r.total_vocab_size > 0) {
        summary += " | vocab " + std::to_string(r.total_vocab_size);
    }
    renderer.drawRoundedRect({x, y}, {width, 22.0f}, Colors::CardSurface, Sizes::SmallRadius);
    renderer.drawText({x + Spacing::Small, y + 3.0f}, summary, Colors::TextPrimary);
    y += 22.0f + Spacing::Small;

    // Round-trip check
    if (r.decoded_text != r.input_text) {
        renderer.drawText({x, y}, "Round-trip mismatch!", Colors::Danger);
        y += 16.0f;
    }

    // Token flow — colored chips showing the tokenization
    float chipX = x;
    float chipY = y;
    float chipH = 26.0f;
    float chipGap = 3.0f;
    float chipPadX = 6.0f;

    // Alternating colors for visual token boundary separation
    static const uint32_t kTokenColors[] = {
        0xD0283050,  // blue-ish
        0xD0304028,  // green-ish
        0xD0403028,  // amber-ish
        0xD0382850,  // purple-ish
        0xD0284040,  // teal-ish
        0xD0402840,  // magenta-ish
    };
    static constexpr int kNumTokenColors = 6;

    for (size_t i = 0; i < r.tokens.size(); ++i) {
        const auto& tok = r.tokens[i];

        // Determine display text
        std::string displayText = tok.piece;
        if (displayText.empty()) {
            displayText = "<" + std::to_string(tok.id) + ">";
        }
        // Replace control chars for display
        for (char& c : displayText) {
            if (c == '\n') c = '\xAC';  // ¬ for newline
            else if (c == '\t') c = '\xBB'; // » for tab
            else if (c < 0x20 && c >= 0) c = '\xB7'; // · for other control
        }

        float textW = UIDrawHelpers::getTextWidth(displayText);
        float chipW = textW + 2.0f * chipPadX;

        // Wrap to next line
        if (chipX + chipW > x + width && chipX > x) {
            chipX = x;
            chipY += chipH + chipGap;
            if (chipY + chipH > y + maxHeight - 40.0f) break; // out of space
        }

        // Chip background
        uint32_t bgColor = kTokenColors[i % kNumTokenColors];
        renderer.drawRoundedRect({chipX, chipY}, {chipW, chipH}, bgColor, Sizes::SmallRadius);

        // Token text
        renderer.drawText({chipX + chipPadX, chipY + 5.0f}, displayText, Colors::TextPrimary);

        // Token ID subscript
        std::string idStr = std::to_string(tok.id);
        renderer.drawText({chipX + chipPadX, chipY + chipH - 10.0f}, idStr, Colors::TextSecondary);

        chipX += chipW + chipGap;
    }

    // Token type legend at bottom
    chipY += chipH + Spacing::Medium;
    if (chipY < y + maxHeight - 20.0f) {
        renderer.drawText({x, chipY},
                          "Types: special | byte | atom | unigram", Colors::TextSecondary);
    }
}

void UITrainingPanel::drawTokenizerBottomBar(OverlayRenderer& renderer, float barY, float barWidth, float barX) {
    float btnW = 120.0f;
    float btnH = Sizes::ButtonHeight;
    float gap = Spacing::Small;
    float totalW = btnW + 90.0f + gap;
    float startX = barX + (barWidth - totalW) / 2.0f;

    tokenizerRunValidationBtn_->setPosition(startX, barY);
    tokenizerRunValidationBtn_->setSize(btnW, btnH);
    tokenizerRunValidationBtn_->drawOverlay(renderer, position);

    tokenizerCloseBtn_->setPosition(startX + btnW + gap, barY);
    tokenizerCloseBtn_->setSize(90.0f, btnH);
    tokenizerCloseBtn_->drawOverlay(renderer, position);
}

void UITrainingPanel::handleEncodeText() {
    if (encodeRunning_.load()) return;
    if (encodeInputBuffer_.empty()) return;
    if (!trainingController) {
        encodeErrorMessage_ = "Controller not initialized";
        encodeComplete_ = true;
        encodeSuccess_ = false;
        return;
    }
    if (!serverConnected) {
        pollServer();
        if (!serverConnected) {
            encodeErrorMessage_ = "Server not connected";
            encodeComplete_ = true;
            encodeSuccess_ = false;
            return;
        }
    }

    encodeRunning_.store(true);
    encodeComplete_ = false;
    encodeSuccess_ = false;
    encodeErrorMessage_.clear();

    std::string textCopy = encodeInputBuffer_;
    std::thread([this, textCopy]() {
        auto result = trainingController->encodeText(textCopy);

        {
            std::lock_guard<std::mutex> lock(encodeMutex_);
            lastEncodeResult_ = result;
        }
        encodeSuccess_ = result.success;
        encodeComplete_ = true;
        encodeRunning_.store(false);

        if (!result.success) {
            encodeErrorMessage_ = result.error;
            addLog("Encode failed: " + result.error, 2);
        } else {
            addLog("Encoded " + std::to_string(result.token_count) + " tokens", 0);
        }
    }).detach();
}
