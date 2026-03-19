//======================================================//
//  UIDataHubPanel — Tabbed data collection + structuring
//
//  Four tabs: Home | Sources | HuggingFace | Structurer
//
//  Each tab is a widget group.  setView() hides the
//  current group and shows the next.  No sub-panel
//  subclasses or per-tab files.
//
//  UI owns layout + events only.  All logic lives in
//  DataCollectionManager, HuggingFaceWebhook,
//  DataStructurer, and DatasetTarget.
//======================================================//

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ui_panel.hpp"
#include "ui_button.hpp"
#include "ui_dropdown.hpp"
#include "ui_inputbox.hpp"
#include "ui_layout_box.hpp"
#include "ui_progress_bar.hpp"
#include "ui_scrollbox.hpp"
#include "ui_slider.hpp"
#include "ui_textarea.hpp"
#include "ui_toggle.hpp"

#include "DataCollection/data_collection_manager.hpp"
#include "DataCollection/huggingface_webhook.hpp"

class OverlayRenderer;
struct InputState;
class DatasetTarget;

namespace GRIM { namespace DataCollection {
    class DataStructurer;
    struct DataStructuringConfig;
}}

// ─────────────────────────────────────────────────────────
//  View enum
// ─────────────────────────────────────────────────────────

enum class DataHubView : uint8_t {
    Home        = 0,
    Sources     = 1,
    HuggingFace = 2,
    Structurer  = 3
};

// ─────────────────────────────────────────────────────────
//  Panel class
// ─────────────────────────────────────────────────────────

class UIDataHubPanel : public UIPanel {
public:
    UIDataHubPanel();
    ~UIDataHubPanel() override;

    void update(const InputState& input, float dt) override;
    bool drawOverlay(OverlayRenderer& renderer) override;

    void        setView(DataHubView view);
    DataHubView currentView() const { return activeView_; }

private:
    // ═════════════════════════════════════════════════════
    //  View management
    // ═════════════════════════════════════════════════════

    DataHubView activeView_ = DataHubView::Home;

    std::vector<std::shared_ptr<Widget>> homeWidgets_;
    std::vector<std::shared_ptr<Widget>> sourcesWidgets_;
    std::vector<std::shared_ptr<Widget>> hfWidgets_;
    std::vector<std::shared_ptr<Widget>> structWidgets_;

    // ═════════════════════════════════════════════════════
    //  Tab buttons (always visible)
    // ═════════════════════════════════════════════════════

    std::shared_ptr<UIButton> tabHomeBtn_;
    std::shared_ptr<UIButton> tabSourcesBtn_;
    std::shared_ptr<UIButton> tabHFBtn_;
    std::shared_ptr<UIButton> tabStructBtn_;

    // ═════════════════════════════════════════════════════
    //  Home tab widgets
    // ═════════════════════════════════════════════════════

    std::shared_ptr<UIProgressBar> progressBar_;
    std::shared_ptr<UIButton>      btnFull_;
    std::shared_ptr<UIButton>      btnCollect_;
    std::shared_ptr<UIButton>      btnVerify_;
    std::shared_ptr<UIButton>      btnMerge_;
    std::shared_ptr<UIButton>      btnRebuild_;
    std::shared_ptr<UIButton>      btnStop_;
    std::shared_ptr<UIButton>      btnRefreshStats_;

    // ═════════════════════════════════════════════════════
    //  Sources tab widgets
    // ═════════════════════════════════════════════════════

    std::shared_ptr<UIButton>    btnAddCard_;

    // ═════════════════════════════════════════════════════
    //  HuggingFace tab widgets
    // ═════════════════════════════════════════════════════

    std::shared_ptr<UIInputBox>  hfSearchInput_;
    std::shared_ptr<UIButton>    btnSearchHF_;
    std::shared_ptr<UIButton>    btnBrowseHF_;
    std::shared_ptr<UIDropdown>  hfCategoryDropdown_;
    std::shared_ptr<UIInputBox>  hfTokenInput_;
    std::shared_ptr<UIScrollBox> hfResultsScrollBox_;
    std::shared_ptr<UISlider>    sliderMaxHFResults_;
    std::shared_ptr<UIButton>    btnProcessQueue_;
    std::shared_ptr<UIButton>    btnClearQueue_;
    std::atomic<bool>            hfResultsNeedsPopulate_{false};

    // ═════════════════════════════════════════════════════
    //  Structurer tab widgets
    // ═════════════════════════════════════════════════════

    std::shared_ptr<UIDropdown>  modelDropdown_;
    std::shared_ptr<UIDropdown>  formatDropdown_;
    std::shared_ptr<UIDropdown>  viewModeDropdown_;
    std::shared_ptr<UIInputBox>  structSearchInput_;
    std::shared_ptr<UIScrollBox> searchPreviewScrollBox_;
    std::shared_ptr<UITextArea>  rawTextArea_;
    std::shared_ptr<UITextArea>  structuredTextArea_;
    std::shared_ptr<UIButton>    btnStructure_;
    std::shared_ptr<UIButton>    btnStructureAll_;
    std::shared_ptr<UIButton>    btnSave_;
    std::shared_ptr<UIButton>    btnPrevSeq_;
    std::shared_ptr<UIButton>    btnNextSeq_;
    std::shared_ptr<UIButton>    btnAssign_;
    std::shared_ptr<UIButton>    btnRemoveAssign_;
    std::shared_ptr<UISlider>    sliderMaxEntries_;
    std::shared_ptr<UISlider>    sliderParallel_;

    // ═════════════════════════════════════════════════════
    //  Backend services
    // ═════════════════════════════════════════════════════

    std::unique_ptr<GRIM::DataCollection::DataCollectionManager> collectionManager_;
    std::unique_ptr<GRIM::DataCollection::HuggingFaceWebhook>    hfWebhook_;

    // ═════════════════════════════════════════════════════
    //  Tab draw methods
    // ═════════════════════════════════════════════════════

    void drawHomeTab(OverlayRenderer& renderer, const PanelRect& content);
    void drawSourcesTab(OverlayRenderer& renderer, const PanelRect& content);
    void drawHuggingFaceTab(OverlayRenderer& renderer, const PanelRect& content);
    void drawStructurerTab(OverlayRenderer& renderer, const PanelRect& content);
    void drawQueueSection(OverlayRenderer& renderer, float x, float& y, float width);

    // ═════════════════════════════════════════════════════
    //  Home tab state
    // ═════════════════════════════════════════════════════

    bool  collectionActive_    = false;
    bool  collectionCompleted_ = false;
    float pollTimer_           = 0.0f;
    float statsUpdateTimer_    = 0.0f;
    float collectionAnimTime_  = 0.0f;

    std::string currentPhase_;
    float       currentProgress_      = 0.0f;
    std::string collectionMessage_;
    int         sourcesProcessed_     = 0;
    int         totalSources_         = 0;
    int         checkpointsCollected_ = 0;

    std::string datasetSizeInfo_;
    std::string checkpointStatsInfo_;
    std::string verificationStatsInfo_;

    int   fetchLimit_             = 100;
    int   vocabSize_              = 50000;
    float verificationThreshold_  = 0.7f;
    int   maxHFResults_           = 4;

    struct LogEntry {
        std::string timestamp;
        std::string message;
        int         level = 0;
    };
    std::vector<LogEntry> logEntries_;
    std::mutex            logMutex_;
    size_t                maxLogEntries_ = 1000;

    // ═════════════════════════════════════════════════════
    //  Sources tab state
    // ═════════════════════════════════════════════════════

    struct SourceCard {
        size_t      cardId       = 0;
        std::string name         = "New Source";
        std::string url;
        int         priority     = 5;
        bool        enabled      = true;
        int         crawlDepth   = 2;
        int         fetchLimit   = 100;
        bool        requiresAuth = false;

        std::shared_ptr<UIInputBox>  nameInput;
        std::shared_ptr<UIInputBox>  urlInput;
        std::shared_ptr<UISlider>    prioritySlider;
        std::shared_ptr<UISlider>    depthSlider;
        std::shared_ptr<UISlider>    limitSlider;
        std::shared_ptr<UIToggle>    enabledToggle;
        std::shared_ptr<UIButton>    deleteBtn;
    };
    std::vector<SourceCard> sourceCards_;
    size_t nextSourceCardId_ = 0;
    int    cardToDelete_     = -1;
    bool   sourcesDirty_     = false;
    float  sourcesScrollOffset_ = 0.0f;

    // ═════════════════════════════════════════════════════
    //  HuggingFace tab state
    // ═════════════════════════════════════════════════════

    std::vector<GRIM::DataCollection::HFDatasetInfo> hfSearchResults_;
    std::string        hfSearchBuffer_;
    std::string        hfTokenBuffer_;
    std::string        hfDownloadStatus_;
    std::string        lastSearchError_;
    std::atomic<bool>  hfSearching_{false};
    float              searchAnimTime_ = 0.0f;

    struct QueuedDownload {
        std::string datasetId;
        std::string displayName;
        std::string status;
        float       progress   = 0.0f;
        int         retryCount = 0;
        std::string errorMessage;
    };
    std::vector<QueuedDownload> downloadQueue_;
    std::mutex                  queueMutex_;
    std::atomic<bool>           queueProcessing_{false};
    std::atomic<int>            currentQueueIndex_{-1};
    int                         hoveredQueueItem_ = -1;

    // ═════════════════════════════════════════════════════
    //  Structurer tab state
    // ═════════════════════════════════════════════════════

    size_t currentSequenceIndex_ = 0;
    size_t totalSequences_       = 0;
    size_t assignedSequences_    = 0;
    size_t structuredCount_      = 0;
    size_t failedCount_          = 0;
    bool   datasetViewMode_      = true;

    // ═════════════════════════════════════════════════════
    //  Internal methods — Home
    // ═════════════════════════════════════════════════════

    void startCollection(const std::string& mode);
    void stopCollection();
    void pollCollectionManager();
    void updateDatasetStats();
    void addLog(const std::string& message, int level = 0);

    // ═════════════════════════════════════════════════════
    //  Internal methods — Sources
    // ═════════════════════════════════════════════════════

    void loadSourceCards();
    void saveSourceCards();
    SourceCard buildSourceCard(const std::string& name = "New Source",
                               const std::string& url = "",
                               int priority = 5, int depth = 2,
                               int limit = 100, bool enabled = true);
    void removeSourceCard(size_t cardId);

    // ═════════════════════════════════════════════════════
    //  Internal methods — HuggingFace
    // ═════════════════════════════════════════════════════

    void searchHuggingFaceDatasets();
    void searchHuggingFaceByCategory(const std::string& category);
    void browseHuggingFaceDatasets();
    void populateHFResults(float containerWidth);
    void addToDownloadQueue(const std::string& datasetId, const std::string& name);
    void processDownloadQueue();
    void clearDownloadQueue();
    void removeFromQueue(int index);
    void retryQueueItem(int index);
    void clearCompletedFromQueue();
    void updateQueueInteraction(const InputState& input, float queueBoxX,
                                float queueBoxY, float queueBoxW, float queueBoxH);

    // ═════════════════════════════════════════════════════
    //  Config persistence
    // ═════════════════════════════════════════════════════

    void loadUIConfig();
    void saveUIConfig();
    void loadDownloadQueue();
    void saveDownloadQueue();
    void loadHFTokenFromConfig();
};
