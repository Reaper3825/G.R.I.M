//======================================================//
//  UIDataHubPanel — Tabbed data collection + structuring
//
//  Five tabs: Home | Sources | HuggingFace | Structurer | Curriculum
//
//  Each tab is a widget group. setView() hides the
//  current group and shows the next. Implementation
//  is split across ui/data_hub/*.cpp for sanity.
//
//  UI owns layout + events only.  All logic lives in
//  PipelineOrchestrator, HuggingFaceWebhook,
//  DataStructurer, and DatasetTarget.
//======================================================//

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "primitives/ui_panel.hpp"
#include "primitives/ui_button.hpp"
#include "primitives/ui_dropdown.hpp"
#include "primitives/ui_inputbox.hpp"
#include "primitives/ui_layout_box.hpp"
#include "primitives/ui_progress_bar.hpp"
#include "primitives/ui_scrollbox.hpp"
#include "primitives/ui_slider.hpp"
#include "primitives/ui_textarea.hpp"
#include "primitives/ui_toggle.hpp"
#include "primitives/ui_action_menu.hpp"

#include "DataCollection/pipeline/pipeline_orchestrator.hpp"
#include "DataCollection/huggingface_webhook.hpp"

class OverlayRenderer;
struct InputState;

#include "DataCollection/dataset_target.hpp"
#include "DataCollection/data_structurer.hpp"

// ─────────────────────────────────────────────────────────
//  View enum
// ─────────────────────────────────────────────────────────

enum class DataHubView : uint8_t {
    Home        = 0,
    Sources     = 1,
    HuggingFace = 2,
    Structurer  = 3,
    Curriculum  = 4
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
    std::vector<std::shared_ptr<Widget>> curriculumWidgets_;

    // ═════════════════════════════════════════════════════
    //  Tab buttons (always visible)
    // ═════════════════════════════════════════════════════

    std::shared_ptr<UIButton> tabHomeBtn_;
    std::shared_ptr<UIButton> tabSourcesBtn_;
    std::shared_ptr<UIButton> tabHFBtn_;
    std::shared_ptr<UIButton> tabStructBtn_;
    std::shared_ptr<UIButton> tabCurriculumBtn_;

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
    std::shared_ptr<UIButton>      btnCollectDir_;

    // ── Directory collection widgets ────────────────────

    std::shared_ptr<UIInputBox>    dirPathInput_;

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
    std::shared_ptr<UITextArea>  hfPreviewArea_;
    std::shared_ptr<UIButton>    btnHFQueuePreview_;
    std::shared_ptr<UIActionMenu> queueActionMenu_;      // Process / Clear
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
    std::shared_ptr<UIActionMenu> structureActionMenu_;  // Structure / Structure All
    std::shared_ptr<UIActionMenu> datasetActionMenu_;    // Save / Assign / Remove
    std::shared_ptr<UIButton>    btnPrevSeq_;
    std::shared_ptr<UIButton>    btnNextSeq_;
    std::shared_ptr<UIButton>    btnGenerate_;
    std::shared_ptr<UIButton>    btnAddSequence_;
    std::shared_ptr<UITextArea>  customPromptArea_;
    std::shared_ptr<UIButton>    btnAppendEntry_;
    std::shared_ptr<UISlider>    sliderMaxEntries_;
    std::shared_ptr<UISlider>    sliderParallel_;

    // ═════════════════════════════════════════════════════
    //  Backend services
    // ═════════════════════════════════════════════════════

    std::unique_ptr<GRIM::Pipeline::PipelineOrchestrator> pipelineOrchestrator_;
    std::unique_ptr<GRIM::DataCollection::HuggingFaceWebhook>    hfWebhook_;
    std::unique_ptr<DatasetTarget>                               datasetTarget_;
    std::unique_ptr<GRIM::DataCollection::DataStructurer>        structurer_;

    // ═════════════════════════════════════════════════════
    //  Tab draw methods
    // ═════════════════════════════════════════════════════

    void drawHomeTab(OverlayRenderer& renderer, const PanelRect& content);
    void drawSourcesTab(OverlayRenderer& renderer, const PanelRect& content);
    void drawHuggingFaceTab(OverlayRenderer& renderer, const PanelRect& content);
    void drawStructurerTab(OverlayRenderer& renderer, const PanelRect& content);
    void drawCurriculumTab(OverlayRenderer& renderer, const PanelRect& content);
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
    int         entriesCollected_     = 0;
    int         duplicatesSkipped_    = 0;
    int64_t     elapsedSeconds_       = 0;

    std::string datasetSizeInfo_;
    std::string hudFileSize_          = "--";
    int         hudCheckpoints_       = 0;
    std::string checkpointStatsInfo_;
    std::string verificationStatsInfo_;

    int   fetchLimit_             = 100;
    float verificationThreshold_  = 0.7f;
    int   maxHFResults_           = 4;

    // ── Directory collection state ──────────────────────

    struct DirFileEntry {
        std::string path;
        std::string filename;
        bool        collect     = true;
        bool        deleteAfter = false;
    };
    std::vector<DirFileEntry> dirFileEntries_;
    float  dirScrollOffset_       = 0.0f;
    bool   dirNeedsScan_          = true;
    std::string dirScanPath_;

    struct { float x=0, y=0, w=0, h=0; } dirScrollAreaRect_;
    bool  dirClickPending_ = false;
    Vec2  dirClickPos_{};

    void loadDirectoryCollectionPathFromConfig();
    void scanDirectory();
    void collectFromDirectory();

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

    int         hfSelectedResultIndex_ = -1;
    std::string hfPreviewDatasetId_;
    std::string hfPreviewDisplayName_;
    std::string hfPreviewPendingText_;
    std::atomic<bool> hfPreviewApply_{false};
    std::atomic<int>  hfPreviewGen_{0};

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
    int    structViewMode_       = 0;   // 0=Dataset, 1=Sequence, 2=Curriculum

    struct SequenceCard {
        size_t cardId       = 0;
        int    formatIndex  = 0;

        struct EntryField {
            std::string                prefix;
            std::shared_ptr<UITextArea> textArea;
        };
        std::vector<EntryField> entries;

        std::shared_ptr<UIDropdown> formatDropdown;
        std::shared_ptr<UIButton>   generateBtn;
        std::shared_ptr<UIButton>   addEntryBtn;
        std::shared_ptr<UIButton>   deleteBtn;
        std::shared_ptr<UIButton>   saveBtn;
    };
    std::vector<SequenceCard> sequenceCards_;
    size_t nextSeqCardId_    = 0;
    int    seqCardToDelete_  = -1;
    float  seqScrollOffset_  = 0.0f;
    float  seqCardAreaTop_   = 0.0f;
    float  seqCardAreaH_     = 0.0f;

    // ═════════════════════════════════════════════════════
    //  Curriculum view state
    // ═════════════════════════════════════════════════════

    float  poolScrollOffset_    = 0.0f;
    float  currScrollOffset_    = 0.0f;
    float  detailPanelHeight_   = 200.0f;
    bool   detailPanelOpen_     = true;
    int    selectedPoolRow_     = -1;
    int    hoveredPoolRow_      = -1;
    int    selectedCurrRow_     = -1;
    int    hoveredCurrRow_      = -1;
    std::vector<size_t> filteredPoolIndices_;
    std::vector<size_t> selectedPoolRows_;
    bool   poolFilterDirty_     = true;

    int         filterSubjectIdx_  = 0;
    int         filterQualityIdx_  = 0;
    std::string filterSearchQuery_;

    std::shared_ptr<UIDropdown> subjectFilterDropdown_;
    std::shared_ptr<UIDropdown> qualityFilterDropdown_;
    std::shared_ptr<UIInputBox> poolSearchInput_;

    std::shared_ptr<UIButton>   btnAssignSelected_;
    std::shared_ptr<UIActionMenu> currListActionMenu_;   // + Phase / << Remove

    std::shared_ptr<UITextArea> detailContentArea_;
    std::shared_ptr<UITextArea> detailStructuredArea_;
    std::shared_ptr<UIButton>   btnDetailSave_;
    int    detailSourceRow_     = -1;   // -1 = pool, >=0 = curriculum idx
    size_t detailSeqIndex_      = SIZE_MAX;

    // ═════════════════════════════════════════════════════
    //  Curriculum tab widgets
    // ═════════════════════════════════════════════════════

    std::shared_ptr<UIDropdown>  cbModelDropdown_;
    std::shared_ptr<UIDropdown>  cbCurriculumDropdown_;
    std::shared_ptr<UIDropdown>  cbTrainingStageDropdown_;
    /// Shown in the ConceptBlock list on the selected row (format / type).
    std::shared_ptr<UIDropdown>  cbListTypeDropdown_;
    /// Toolbar filter-by-type dropdown ("All", "Q/A", "Chain of Thought", etc.).
    std::shared_ptr<UIDropdown>  cbTypeFilterDropdown_;
    std::shared_ptr<UIToggle>    cbCurriculumFilterToggle_;
    std::shared_ptr<UIInputBox>  cbSearchInput_;
    std::shared_ptr<UIInputBox>  cbNameInput_;
    std::shared_ptr<UITextArea>  cbPromptArea_;
    std::shared_ptr<UITextArea>  cbTargetStateArea_;
    struct CBSuccessCriterionRow {
        std::shared_ptr<UITextArea> criterionArea;
        std::shared_ptr<UITextArea> evidenceArea;
    };
    std::vector<CBSuccessCriterionRow> cbSuccessCriterionRows_;
    std::vector<std::shared_ptr<UITextArea>> cbConstraintAreas_;
    std::shared_ptr<UITextArea>  cbAnswerArea_;
    std::shared_ptr<UITextArea>  cbCustomPromptArea_;
    std::vector<std::shared_ptr<UITextArea>> cbIntermediateAreas_;

    struct CBExecStepRow {
        std::shared_ptr<UIDropdown>  opDropdown;    // add/sub/mul/div
        std::shared_ptr<UIInputBox>  argSlotsInput; // e.g. "0,1"
        std::shared_ptr<UIInputBox>  argsInput;     // e.g. "2.0,3.0"
        std::shared_ptr<UIInputBox>  resultInput;   // e.g. "5.0"
    };
    std::vector<CBExecStepRow> cbExecStepRows_;

    float  cbEditorScrollOffset_ = 0.0f;

    std::shared_ptr<UIButton>    btnCBGenerate_;
    std::shared_ptr<UIActionMenu> successCriteriaActionMenu_; // + / - criterion
    std::shared_ptr<UIActionMenu> constraintsActionMenu_; // + / - constraint
    std::shared_ptr<UIActionMenu> stepActionMenu_;       // + Step / - Step
    std::shared_ptr<UIActionMenu> execStepActionMenu_;   // + Exec Step / - Exec Step
    std::shared_ptr<UIActionMenu> blockActionMenu_;      // New / Save / Delete
    std::shared_ptr<UIActionMenu> curriculumActionMenu_;   // New / Delete / Assign / Rename
    std::shared_ptr<UIActionMenu> blockCurriculumMenu_;     // + Curr / - Curr
    std::shared_ptr<UIInputBox>  cbCurriculumRenameInput_;  // inline rename field
    bool renamingCurriculum_ = false;
    bool renameJustActivated_ = false;  // one-frame guard to prevent same-click cancel

    // ═════════════════════════════════════════════════════
    //  Curriculum tab state
    // ═════════════════════════════════════════════════════

    float  cbListScrollOffset_   = 0.0f;
    int    selectedCBRow_        = -1;
    int    hoveredCBRow_         = -1;
    std::vector<size_t> filteredCBIndices_;
    bool   cbFilterDirty_        = true;
    std::string cbFilterSearch_;
    int    cbFormatFilterIdx_    = 0;
    bool   cbCurriculumFilterActive_ = false;
    size_t cbTotalCount_         = 0;
    size_t cbInCurrCount_        = 0;
    /// ID of the currently selected curriculum in the dropdown.
    std::string activeCurriculumId_;
    /// Unsaved row pinned at list index 0; shows live name / question preview.
    bool   cbDraftPreviewActive_ = false;

    int  cbCurriculumListRowCount() const;
    bool cbCurriculumRowIsDraft(int listRow) const;
    bool cbCurriculumRowToBlockIndex(int listRow, size_t& outBlockIndex) const;
    void syncCBListTypeDropdownFromToolbar();
    void layoutCBListTypeDropdownInList(float listX, float listY, float listW);

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
    void selectHuggingFaceResult(size_t index);
    void addToDownloadQueue(const std::string& datasetId, const std::string& name);
    void processDownloadQueue();
    void clearDownloadQueue();
    void removeFromQueue(int index);
    void retryQueueItem(int index);
    void clearCompletedFromQueue();
    void updateQueueInteraction(const InputState& input, float queueBoxX,
                                float queueBoxY, float queueBoxW, float queueBoxH);

    // ═════════════════════════════════════════════════════
    //  Internal methods — Structurer
    // ═════════════════════════════════════════════════════

    void refreshStructurerState();
    void loadCurrentSequence();
    void populateModelDropdown();

    SequenceCard buildSequenceCard(int formatIndex = 0);
    void applyFormatTemplate(SequenceCard& card, int formatIndex);
    void removeSequenceCard(size_t cardId);
    void generateForCard(size_t cardId);
    void saveSequenceCard(size_t cardId);
    float sequenceCardHeight(const SequenceCard& card) const;

    // ── Curriculum view methods (Structurer sub-view) ───
    void drawCurriculumView(OverlayRenderer& renderer, const PanelRect& content,
                            float x, float y, float fullW, float contentEndY);
    void drawPoolTable(OverlayRenderer& renderer, float x, float y, float w, float h);
    void drawCurriculumList(OverlayRenderer& renderer, float x, float y, float w, float h);
    void drawDetailEditor(OverlayRenderer& renderer, float x, float y, float w, float h);
    void rebuildFilteredPool();
    void selectPoolRow(int row);
    void selectCurriculumRow(int row);
    void loadDetailForSequence(size_t seqIndex);

    // ── Curriculum tab methods ───────────────────────────
    void refreshCurriculumTabState();
    void rebuildFilteredCBList();
    void loadConceptBlockIntoEditor(size_t cbIndex);
    void clearCBEditor();
    void syncSuccessCriterionRows(int count);
    void syncConstraintAreas(int count);
    void syncIntermediateAreas(int count);
    void syncExecStepRows(int count);
    bool buildConceptBlockFromEditor(GRIM::ConceptBlock& out,
                                     std::string& validation_error) const;
    std::string buildTrainingPreview(const GRIM::ConceptBlock& cb, bool conceptMode) const;
    void generateConceptBlock();
    void populateCBModelDropdown();
    void populateCBCurriculumDropdown();
    void syncCurriculumTrainingStageDropdown();
    void selectActiveCurriculum(int dropdownIndex);

    // ═════════════════════════════════════════════════════
    //  Config persistence
    // ═════════════════════════════════════════════════════

    void loadUIConfig();
    void saveUIConfig();
    void loadDownloadQueue();
    void saveDownloadQueue();
    void loadHFTokenFromConfig();
};
