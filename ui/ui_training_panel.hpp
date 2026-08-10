#pragma once
//======================================================//
//  UITrainingPanel — Unified Model + Training Hub
//
//  Five tabs: Home | Knowledge Gaps | Tool Gaps | Model Config | Tokenizer
//
//  Merges the former ModelRegistry panel and Training panel
//  into a single DataHub-style tabbed interface.
//
//  - Home:           model browser (registry status, resource bars)
//  - Knowledge Gaps: gap queue (router misses → create model)
//  - Tool Gaps:      tool-gap proposals (ToolGapPlanner)
//  - Model Config:   edit a snapshot and compile a per-model .grimcfg
//  - Tokenizer:      standalone tokenizer validation & encode
//======================================================//

#include "primitives/ui_panel.hpp"
#include "primitives/ui_button.hpp"
#include "primitives/ui_progress_bar.hpp"
#include "primitives/ui_inputbox.hpp"
#include "primitives/ui_dropdown.hpp"
#include "primitives/ui_label.hpp"
#include "primitives/ui_scrollbox.hpp"
#include "control/training_controller.hpp"
#include "control/hyperparameter_registry.hpp"
#include "../MMO/Shared/MMD.hpp"
#include "../MMO/Core/ToolGapPlanner.hpp"
#include <future>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <atomic>

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
    ModelConfig   = 3,
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
private:
    // ═════════════════════════════════════════════════════
    //  Tab state
    // ═════════════════════════════════════════════════════
    TrainingPanelTab activeTab_ = TrainingPanelTab::Home;

    std::shared_ptr<UIButton> tabHomeBtn_;
    std::shared_ptr<UIButton> tabKnowledgeGapsBtn_;
    std::shared_ptr<UIButton> tabToolGapsBtn_;
    std::shared_ptr<UIButton> tabModelConfigBtn_;
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

    // Tokenizer service connectivity. Config compilation is local and never
    // starts a training process.
    std::unique_ptr<GRIM::UI::UITrainingController> trainingController;
    bool serverConnected = false;
    float pollTimer = 0.0f;
    float pollInterval = 0.2f;
    void pollServer();

    // Model configuration preset creator.
    nlohmann::json configPresetDocument_;
    bool configPresetDirty_ = false;
    std::vector<std::string> configModelIds_;
    std::string configModelIdBuffer_;
    std::string configVocabPathBuffer_;
    std::string configCompilerPathBuffer_;
    std::string configCompileStatus_;
    bool configCompileSuccess_ = false;

    std::shared_ptr<UIDropdown> configModelDropdown_;
    std::shared_ptr<UIInputBox> configModelIdInput_;
    std::shared_ptr<UIInputBox> configVocabPathInput_;
    std::shared_ptr<UIInputBox> configCompilerPathInput_;
    std::shared_ptr<UIButton> configCompileButton_;
    std::shared_ptr<UIButton> configReloadButton_;

    struct ConfigCompileResult {
        bool success = false;
        std::string message;
    };
    std::future<ConfigCompileResult> configCompileFuture_;

    void drawModelConfigTab(OverlayRenderer& renderer, const PanelRect& content);
    void updateModelConfigTab(const InputState& input, float dt);
    void beginConfigCompile();
    void applyConfigCompileResult();
    void reloadConfigPresetTemplate();
    void refreshConfigModelDropdown();
    std::string findConfigCompilerExecutable() const;
    std::string configOutputPath() const;
    static ConfigCompileResult compileConfigPreset(
        const nlohmann::json& source,
        const std::string& compilerPath,
        const std::string& vocabPath,
        const std::string& modelStorePath,
        const std::string& modelId);

    // Tokenizer runner state
    bool tokenizerRunning_ = false;
    bool tokenizerComplete_ = false;
    bool tokenizerSuccess_ = false;
    std::string tokenizerStatusMessage_;
    GRIMText::TrainingControlClient::TokenizerResult lastTokenizerResult_;
    void handleRunTokenizer();
    void drawTokenizerStatus(OverlayRenderer& renderer, float x, float y, float width);
    void drawStatCard(OverlayRenderer& renderer, const Vec2& pos, const Vec2& size,
                      const std::string& label, const std::string& value,
                      uint32_t accentColor);

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

    // ═════════════════════════════════════════════════════
    //  FlatBuffer model-schema registry (filterable authored-field browser)
    // ═════════════════════════════════════════════════════
    GRIM::Config::HyperparameterRegistry hyperparamRegistry_;
    bool hyperparamsLoaded_ = false;

    std::shared_ptr<UIDropdown> paramCategoryFilter_;
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
    void loadHyperparamSnapshot();
    void drawParamBrowser(OverlayRenderer& renderer, const Vec2& origin, const Vec2& sz);
    void processParamBrowserClicks(const InputState& input);
    void commitParamEdit();
    void cancelParamEdit();
    bool setPresetHyperparamValue(const GRIM::Config::HyperparamEntry& entry,
                                  const nlohmann::json& value);
};
