#pragma once
#include "ui_panel.hpp"
#include "ui_button.hpp"
#include "ui_slider.hpp"
#include "ui_progress_bar.hpp"
#include "ui_layout_box.hpp"
#include "ui_inputbox.hpp"
#include "ui_dropdown.hpp"
#include "ui_label.hpp"
#include "ui_scrollbox.hpp"
#include "../MMO/Shared/MMD.hpp"
#include "../hardware/resource_values.hpp"
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <atomic>

class OverlayRenderer;
struct InputState;

// =========================================================
// Knowledge gap entry — queued when the router finds no
// sub-model matching a subject. Feeds into the Model Creator
// with pre-filled subject/tags so the user can source or
// configure a new frozen brick.
// =========================================================
struct KnowledgeGapEntry {
    std::string subject;
    std::vector<std::string> tags;
    std::string original_context;
    std::string request_id;
    std::chrono::steady_clock::time_point timestamp;
};

// =========================================================
// Active view within the model panel
// =========================================================
enum class ModelPanelView : uint8_t {
    Browser  = 0,   // browse registered models + status
    Creator  = 1,   // register a new sub-model
    GapQueue = 2    // knowledge gap proposals
};

// =========================================================
// UIModelPanel — Model Creator Suite
//
// Three views:
//   Browser:  all models from ModelRegistry, residency state
//             from ModelLoader, resource footprint, load/unload.
//   Creator:  form to register a new sub-model. Pre-flight
//             validates (ID uniqueness, path, resource claim)
//             then writes to ModelRegistry + ai_config.json.
//   GapQueue: incoming KnowledgeGapEntry items when the router
//             finds no subject match. "Create from this" pre-fills
//             the Creator.
//
// Integration:
//   ModelRegistry  — read / register / remove models
//   ModelLoader    — residency state, ensureLoaded / unload
//   ResourceSignal — live VRAM/RAM for resource bars
//   ai_config.json — persist mmo.sub_models entries
// =========================================================
class UIModelPanel : public UIPanel {
public:
    UIModelPanel();
    ~UIModelPanel() override;

    void update(const InputState& input, float dt) override;
    void drawOverlay(OverlayRenderer& renderer) override;

    // --- View control ---------------------------------------------------
    void setView(ModelPanelView view);
    ModelPanelView currentView() const { return activeView_; }

    // --- Knowledge gap intake -------------------------------------------
    // Called by the orchestrator when routing finds no subject match.
    void pushKnowledgeGap(KnowledgeGapEntry entry);
    size_t pendingGapCount() const;

    // --- Model browser --------------------------------------------------
    void refreshModelList();

    // --- Model creator (pre-fill from gap) ------------------------------
    void prefillCreatorFromGap(const KnowledgeGapEntry& gap);

    // Check if any input is currently being edited
    bool isAnyInputEditing() const;

private:
    // --- View state -----------------------------------------------------
    ModelPanelView activeView_ = ModelPanelView::Browser;

    // --- Tab buttons ----------------------------------------------------
    std::shared_ptr<UIButton> tabBrowserBtn_;
    std::shared_ptr<UIButton> tabCreatorBtn_;
    std::shared_ptr<UIButton> tabGapQueueBtn_;

    // ====================================================================
    // Browser view
    // ====================================================================
    struct ModelListEntry {
        std::string id;
        std::string name;
        std::string subject;
        std::string backend;       // display string
        std::string status;        // display string from ResidencyState
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
    int   hoveredBrowserRow_ = -1;   // -1 = none

    static constexpr float kRowHeight  = 20.0f;
    static constexpr float kHeaderY    = 70.0f;   // relative to panel top
    static constexpr float kDataStartY = 96.0f;   // header + gap

    void drawBrowserView(OverlayRenderer& renderer);
    void handleModelAction(const std::string& model_id, const std::string& action);
    void processBrowserClicks(const InputState& input);

    // ====================================================================
    // Creator view
    // ====================================================================
    std::shared_ptr<UIInputBox> creatorIdInput_;
    std::shared_ptr<UIInputBox> creatorNameInput_;
    std::shared_ptr<UIInputBox> creatorSubjectInput_;
    std::shared_ptr<UIInputBox> creatorTagsInput_;       // comma-separated
    std::shared_ptr<UIInputBox> creatorDescInput_;
    std::shared_ptr<UIInputBox> creatorPathInput_;
    std::shared_ptr<UIInputBox> creatorUrlInput_;
    std::shared_ptr<UIDropdown> creatorBackendDropdown_;
    std::shared_ptr<UISlider>   creatorRamSlider_;
    std::shared_ptr<UISlider>   creatorVramSlider_;
    std::shared_ptr<UIButton>   creatorRegisterBtn_;
    std::shared_ptr<UIButton>   creatorCancelBtn_;
    std::shared_ptr<UILabel>    creatorStatusLabel_;

    // Buffers mirroring input box contents
    std::string bufId_;
    std::string bufName_;
    std::string bufSubject_;
    std::string bufTags_;
    std::string bufDesc_;
    std::string bufPath_;
    std::string bufUrl_;

    void drawCreatorView(OverlayRenderer& renderer);
    void clearCreatorFields();
    bool validateCreatorFields(std::string& out_error) const;
    void submitNewModel();

    // ====================================================================
    // Knowledge Gap Queue view
    // ====================================================================
    std::vector<KnowledgeGapEntry> gapQueue_;
    mutable std::mutex gapMutex_;
    std::shared_ptr<UIScrollBox> gapScrollBox_;
    int hoveredGapRow_ = -1;   // -1 = none

    void drawGapQueueView(OverlayRenderer& renderer);
    void dismissGap(size_t index);
    void createFromGap(size_t index);
    void processGapClicks(const InputState& input);

    // ====================================================================
    // Config persistence (ai_config.json → mmo.sub_models)
    // ====================================================================
    bool persistSubModel(const GRIM::MMO::ModelInfo& model);
    bool removeSubModelFromConfig(const std::string& model_id);

    // ====================================================================
    // Resource display
    // ====================================================================
    void updateResourceBars();
    float totalVramMb_  = 0.0f;
    float usedVramMb_   = 0.0f;
    float totalRamMb_   = 0.0f;
    float usedRamMb_    = 0.0f;
};
