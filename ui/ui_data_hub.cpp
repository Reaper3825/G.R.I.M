// ui_data_hub.cpp — DataHub: tabbed data collection + structuring panel
// See ui_data_hub.hpp for class documentation.
//======================================================//

#include "ui_data_hub.hpp"
#include "overlay_renderer.hpp"
#include "ui_theme.hpp"
#include "ui_draw_helpers.hpp"
#include "logger.hpp"
#include "resources.hpp"
#include "resources/models/GRIM-text/Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "core/input_parser.hpp"
#include "MMO/Core/ModelRegistry.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <unordered_set>

static constexpr float kPollInterval  = 0.2f;
static constexpr float kStatsInterval = 2.0f;
static constexpr float kTabBarY       = 35.0f;
static constexpr float kContentTopY   = 68.0f;

static constexpr float kCardTopBarH  = 28.0f;
static constexpr float kCardFieldH   = 28.0f;
static constexpr float kCardLabelGap = 3.0f;
static constexpr float kCardLabelH   = 13.0f;
static constexpr float kCardRowGap   = 6.0f;
static constexpr float kCardRowH     = kCardFieldH + kCardLabelGap + kCardLabelH + kCardRowGap;
static constexpr float kCardPadBot   = 8.0f;
static constexpr float kCardHeight   = kCardTopBarH + 2.0f * kCardRowH + kCardPadBot;
static constexpr float kCardGap      = 14.0f;
static constexpr float kCardInnerPad = 12.0f;

// Structurer tab spacing (labels above text areas need ~18px + padding)
static constexpr float kStructRowGap     = 12.0f;   // padding between toolbar rows
static constexpr float kStructSectionGap = 16.0f;   // padding between sections
static constexpr float kStructLabelSpace = 24.0f;   // space for labels above input areas

// Sequence card dimensions (used in drawStructurerTab and helpers)
static constexpr float kSeqCardTopBarH     = 32.0f;
static constexpr float kSeqCardEntryLabelH = 16.0f;
static constexpr float kSeqCardEntryTextH  = 55.0f;
static constexpr float kSeqCardEntryGapH   = 6.0f;
static constexpr float kSeqCardEntryH      = kSeqCardEntryLabelH + kSeqCardEntryTextH + kSeqCardEntryGapH;
static constexpr float kSeqCardBtnRowH     = 30.0f;
static constexpr float kSeqCardPadBot      = 8.0f;
static constexpr float kSeqCardInnerPad    = 12.0f;

// Curriculum view dimensions
static constexpr float kPoolRowH       = 24.0f;
static constexpr float kCBListRowH     = 44.0f; // ConceptBlock registry (name + question preview)
static constexpr float kPoolHeaderH    = 28.0f;
static constexpr float kCurrRowH       = 28.0f;
static constexpr float kPhaseRowH      = 22.0f;
static constexpr float kFilterBarH     = 34.0f;
static constexpr float kDetailDividerH = 18.0f;

static constexpr float kColNum         = 0.06f;
static constexpr float kColSubject     = 0.12f;
static constexpr float kColQuality     = 0.10f;
static constexpr float kColSource      = 0.10f;
static constexpr float kColStructured  = 0.08f;

// System-only: auto-detect fetcher from URL (matches web_collector logic)
static std::string cbSingleLinePreview(const std::string& s, size_t maxLen) {
    std::string t;
    t.reserve(s.size());
    for (char c : s) {
        if (c == '\n' || c == '\r' || c == '\t')
            t.push_back(' ');
        else
            t.push_back(c);
    }
    while (!t.empty() && t.back() == ' ')
        t.pop_back();
    if (t.size() > maxLen)
        return t.substr(0, maxLen - 2) + "..";
    return t;
}

static std::string detectFetcherFromUrl(const std::string& url) {
    if (url.find("api.github.com") != std::string::npos || url.find("github.com") != std::string::npos)
        return "github_api";
    if (url.find("arxiv.org") != std::string::npos) return "arxiv_api";
    if (url.find("wikipedia.org") != std::string::npos) return "wikipedia_api";
    if (url.find("stackexchange.com") != std::string::npos || url.find("stackoverflow.com") != std::string::npos)
        return "stackoverflow_api";
    if (url.find("reddit.com") != std::string::npos) return "reddit_api";
    if (url.find("newsapi.org") != std::string::npos) return "news_api";
    if (url.find("huggingface.co") != std::string::npos || url.find("huggingface://") == 0)
        return "huggingface";
    return "html_crawl";
}

static std::string getSourceConfigPath() {
    auto snapshot = GRIM::Config::loadAiConfigSnapshot();
    if (snapshot && snapshot->has_grim_paths && !snapshot->grim_text_source_config.empty()) {
        return snapshot->grim_text_source_config;
    }
    return (std::filesystem::path(getGrimRootDir()) / "DataCollection" / "source_data.json").string();
}

namespace {
struct CurriculumTabLayout {
    float listX   = 0.0f;
    float listY   = 0.0f;
    float listW   = 0.0f;
    float listH   = 0.0f;
    float editorX = 0.0f;
    float editorW = 0.0f;
    float bottomBarH = 36.0f;
    float statusBarH = 25.0f;
};

CurriculumTabLayout computeCurriculumTabLayout(const PanelRect& content) {
    CurriculumTabLayout layout;
    layout.listX = content.origin.x + 15.0f;

    float y = content.origin.y + 10.0f;
    constexpr float kToolbarRowH = 36.0f;
    y += kToolbarRowH + 6.0f;
    y += kToolbarRowH + 12.0f;

    const float fullW = content.size.x - 30.0f;

    layout.listY = y;
    layout.listW = fullW * 0.38f;
    layout.listH = (content.origin.y + content.size.y) - y
                 - layout.bottomBarH - layout.statusBarH - 10.0f;
    if (layout.listH < 100.0f)
        layout.listH = 100.0f;

    layout.editorX = layout.listX + layout.listW + 10.0f;
    layout.editorW = fullW - layout.listW - 10.0f;
    return layout;
}
}

// =========================================================
// Constructor
// =========================================================

UIDataHubPanel::UIDataHubPanel()
    : UIPanel("DataHub", true)
{
    position = {200, 100};
    size     = {1400, 750};
    setVisible(false);
    setBackground(UITheme::Colors::PanelBg);

    // ── Tab buttons ─────────────────────────────────────

    tabHomeBtn_ = std::make_shared<UIButton>("Home", [this]() {
        setView(DataHubView::Home);
    });
    tabHomeBtn_->setSize(90.0f, 28.0f);

    tabSourcesBtn_ = std::make_shared<UIButton>("Sources", [this]() {
        setView(DataHubView::Sources);
    });
    tabSourcesBtn_->setSize(90.0f, 28.0f);

    tabHFBtn_ = std::make_shared<UIButton>("HuggingFace", [this]() {
        setView(DataHubView::HuggingFace);
    });
    tabHFBtn_->setSize(110.0f, 28.0f);

    tabStructBtn_ = std::make_shared<UIButton>("Structurer", [this]() {
        setView(DataHubView::Structurer);
    });
    tabStructBtn_->setSize(100.0f, 28.0f);

    tabCurriculumBtn_ = std::make_shared<UIButton>("Curriculum", [this]() {
        setView(DataHubView::Curriculum);
    });
    tabCurriculumBtn_->setSize(100.0f, 28.0f);

    // ── Home tab widgets ────────────────────────────────

    progressBar_ = std::make_shared<UIProgressBar>("Pipeline Progress", 1.0f);
    progressBar_->setFillColor(UITheme::Colors::Info);
    progressBar_->setBackgroundColor(UITheme::Colors::Background);

    btnFull_    = std::make_shared<UIButton>("Full Pipeline",  [this]() { startCollection("full"); });
    btnCollect_ = std::make_shared<UIButton>("Collect Only",   [this]() { startCollection("collect"); });
    btnVerify_  = std::make_shared<UIButton>("Verify Only",    [this]() { startCollection("verify"); });
    btnMerge_   = std::make_shared<UIButton>("Merge Only",     [this]() { startCollection("merge"); });
    btnRebuild_ = std::make_shared<UIButton>("Force Rebuild",  [this]() { startCollection("merge-rebuild"); });
    btnStop_    = std::make_shared<UIButton>("Stop",           [this]() { stopCollection(); });
    btnRefreshStats_ = std::make_shared<UIButton>("Refresh Stats", [this]() { updateDatasetStats(); });
    btnCollectDir_ = std::make_shared<UIButton>("Collect Dir", [this]() { collectFromDirectory(); });

    dirPathInput_ = std::make_shared<UIInputBox>();
    dirPathInput_->setPlaceholder("Directory path for file collection...");

    homeWidgets_ = {
        btnFull_, btnCollect_, btnVerify_, btnMerge_,
        btnRebuild_, btnStop_, btnRefreshStats_, btnCollectDir_,
        dirPathInput_
    };

    // ── Sources tab widgets ─────────────────────────────

    btnAddCard_ = std::make_shared<UIButton>("+ Add Source", [this]() {
        sourceCards_.push_back(buildSourceCard());
        sourcesDirty_ = true;
        addLog("Added new source card", 0);
    });

    sourcesWidgets_ = { btnAddCard_ };

    // ── HuggingFace tab widgets ─────────────────────────

    hfSearchInput_ = std::make_shared<UIInputBox>();
    hfSearchInput_->setPlaceholder("Search Hugging Face datasets...");

    btnSearchHF_ = std::make_shared<UIButton>("Search", [this]() {
        searchHuggingFaceDatasets();
    });
    btnBrowseHF_ = std::make_shared<UIButton>("Browse Popular", [this]() {
        browseHuggingFaceDatasets();
    });

    std::vector<std::string> hfCategories = {
        "All Categories", "text-generation", "question-answering",
        "summarization", "translation", "text-classification",
        "conversational", "token-classification", "fill-mask",
        "text2text-generation", "sentence-similarity"
    };
    hfCategoryDropdown_ = std::make_shared<UIDropdown>(
        "Category", hfCategories, 0,
        [this](int idx, const std::string& cat) {
            if (idx > 0) searchHuggingFaceByCategory(cat);
        });
    hfCategoryDropdown_->setMaxVisibleItems(8);

    hfTokenInput_ = std::make_shared<UIInputBox>();
    hfTokenInput_->setPlaceholder("HF API Token (optional)...");

    hfResultsScrollBox_ = std::make_shared<UIScrollBox>();
    hfResultsScrollBox_->setChildSpacing(5.0f);

    sliderMaxHFResults_ = std::make_shared<UISlider>(
        "Max Results", 1.0f, 20.0f, static_cast<float>(maxHFResults_),
        [this](float v) { maxHFResults_ = static_cast<int>(v); }, 1.0f);

    hfPreviewArea_ = std::make_shared<UITextArea>(
        "Sample rows", "Click a dataset in the list above to load a preview (Hugging Face datasets-server API).",
        [](const std::string&) {});
    btnHFQueuePreview_ = std::make_shared<UIButton>("Add previewed dataset to queue", [this]() {
        if (hfPreviewDatasetId_.empty()) {
            addLog("Select a dataset in the list first", 1);
            return;
        }
        addToDownloadQueue(hfPreviewDatasetId_, hfPreviewDisplayName_);
    });

    queueActionMenu_ = std::make_shared<UIActionMenu>("Queue");
    queueActionMenu_->addItem("Process Queue", [this]() {
        processDownloadQueue();
    });
    queueActionMenu_->addSeparator();
    queueActionMenu_->addItem("Clear Queue", [this]() {
        clearDownloadQueue();
    }, UITheme::Colors::Danger);

    hfWidgets_ = {
        hfSearchInput_, btnSearchHF_, btnBrowseHF_, hfCategoryDropdown_,
        hfTokenInput_, hfResultsScrollBox_, sliderMaxHFResults_,
        hfPreviewArea_, btnHFQueuePreview_,
        queueActionMenu_
    };

    // ── Structurer tab widgets ──────────────────────────

    modelDropdown_ = std::make_shared<UIDropdown>(
        "Model", std::vector<std::string>{"(none)"}, 0,
        [this](int idx, const std::string& name) {
            if (!datasetTarget_ || idx < 0) return;
            auto models = GRIM::MMO::ModelRegistry::instance().getAllModels();
            if (idx < static_cast<int>(models.size())) {
                datasetTarget_->setActiveModel(models[idx]->id, models[idx]->name);
                datasetTarget_->loadAssignments();
                assignedSequences_ = datasetTarget_->assignedCount();
            }
        });

    formatDropdown_ = std::make_shared<UIDropdown>(
        "Format", std::vector<std::string>{"Q/A", "Thought", "Conversation", "Instruct", "Raw"}, 0,
        [this](int, const std::string&) {});

    viewModeDropdown_ = std::make_shared<UIDropdown>(
        "View", std::vector<std::string>{"Dataset View", "Sequence View", "Curriculum View"}, 0,
        [this](int idx, const std::string&) {
            structViewMode_ = idx;
            if (idx == 2) {
                poolFilterDirty_ = true;
                rebuildFilteredPool();
            }
        });

    structSearchInput_ = std::make_shared<UIInputBox>();
    structSearchInput_->setPlaceholder("Search sequences...");

    searchPreviewScrollBox_ = std::make_shared<UIScrollBox>();
    searchPreviewScrollBox_->setChildSpacing(3.0f);

    rawTextArea_ = std::make_shared<UITextArea>("Raw Source", "",
        [](const std::string&) {});
    structuredTextArea_ = std::make_shared<UITextArea>("Structured Output", "",
        [](const std::string&) {});

    structureActionMenu_ = std::make_shared<UIActionMenu>("Structure");
    structureActionMenu_->addItem("Structure", [this]() {
        if (!structurer_ || !rawTextArea_) return;
        std::string raw = rawTextArea_->getText();
        if (raw.empty()) { addLog("No raw text to structure", 1); return; }
        auto results = structurer_->structureEntry(raw);
        if (results.empty()) {
            addLog("Structuring failed — LLM returned no output", 2);
            return;
        }
        std::string combined;
        for (size_t i = 0; i < results.size(); ++i) {
            if (i > 0) combined += "\n\n---\n\n";
            combined += results[i];
        }
        if (structuredTextArea_) structuredTextArea_->setText(combined);
        addLog("Structured into " + std::to_string(results.size()) + " pair(s)", 0);
    });
    structureActionMenu_->addItem("Structure All", [this]() {
        if (!structurer_ || !datasetTarget_) return;
        addLog("Structure All — running in background...", 0);
        std::thread([this]() {
            std::vector<std::string> texts;
            size_t count = datasetTarget_->massDatasetSize();
            for (size_t i = 0; i < count; ++i) {
                auto seq = datasetTarget_->getSequence(i);
                if (!seq.is_structured) texts.push_back(seq.content);
            }
            if (texts.empty()) { addLog("No unstructured sequences found", 1); return; }
            auto result = structurer_->structureBatch(texts,
                [this](size_t done, size_t total) {
                    addLog("Structuring: " + std::to_string(done) + "/" + std::to_string(total), 0);
                });
            size_t writeIdx = 0;
            for (size_t i = 0; i < datasetTarget_->massDatasetSize() && writeIdx < result.structured.size(); ++i) {
                auto seq = datasetTarget_->getSequence(i);
                if (seq.is_structured) continue;
                const auto& pairs = result.structured[writeIdx++];
                if (!pairs.empty()) {
                    std::string combined;
                    for (size_t p = 0; p < pairs.size(); ++p) {
                        if (p > 0) combined += "\n\n---\n\n";
                        combined += pairs[p];
                    }
                    datasetTarget_->writeStructuredOutput(i, combined);
                }
            }
            structuredCount_ = 0;
            for (size_t i = 0; i < datasetTarget_->massDatasetSize(); ++i)
                if (datasetTarget_->getSequence(i).is_structured) structuredCount_++;
            failedCount_ = result.failed;
            addLog("Structure All complete: " + std::to_string(result.succeeded) + " ok, "
                 + std::to_string(result.failed) + " failed", 0);
        }).detach();
    });

    datasetActionMenu_ = std::make_shared<UIActionMenu>("Data");
    datasetActionMenu_->addItem("Save", [this]() {
        if (!datasetTarget_ || !structuredTextArea_) return;
        std::string text = structuredTextArea_->getText();
        if (text.empty()) { addLog("Nothing to save", 1); return; }
        if (currentSequenceIndex_ < datasetTarget_->massDatasetSize()) {
            if (datasetTarget_->writeStructuredOutput(currentSequenceIndex_, text)) {
                structuredCount_ = 0;
                for (size_t i = 0; i < datasetTarget_->massDatasetSize(); ++i)
                    if (datasetTarget_->getSequence(i).is_structured) structuredCount_++;
                addLog("Saved structured output for sequence " + std::to_string(currentSequenceIndex_), 0);
                loadCurrentSequence();
            } else {
                addLog("Failed to save structured output", 2);
            }
        } else {
            std::string raw = rawTextArea_ ? rawTextArea_->getText() : "";
            if (datasetTarget_->appendStructuredEntry(raw, text)) {
                totalSequences_ = datasetTarget_->massDatasetSize();
                structuredCount_++;
                addLog("Appended new structured entry to mass dataset", 0);
            } else {
                addLog("Failed to append new entry", 2);
            }
        }
    });
    datasetActionMenu_->addSeparator();
    datasetActionMenu_->addItem("Assign to Model", [this]() {
        if (!datasetTarget_) return;
        if (datasetTarget_->activeModelId().empty()) {
            addLog("Select a model first", 1); return;
        }
        if (datasetTarget_->assignSequenceToModel(currentSequenceIndex_)) {
            assignedSequences_ = datasetTarget_->assignedCount();
            addLog("Assigned sequence to " + datasetTarget_->activeModelName(), 0);
        } else {
            addLog("Failed to assign sequence", 2);
        }
    });
    datasetActionMenu_->addItem("Remove Assignment", [this]() {
        if (!datasetTarget_) return;
        if (datasetTarget_->removeSequenceFromModel(currentSequenceIndex_)) {
            assignedSequences_ = datasetTarget_->assignedCount();
            addLog("Removed assignment from " + datasetTarget_->activeModelName(), 0);
        } else {
            addLog("Failed to remove assignment", 2);
        }
    }, UITheme::Colors::Danger);

    btnPrevSeq_ = std::make_shared<UIButton>("<", [this]() {
        if (currentSequenceIndex_ > 0) {
            currentSequenceIndex_--;
            loadCurrentSequence();
        }
    });
    btnNextSeq_ = std::make_shared<UIButton>(">", [this]() {
        if (currentSequenceIndex_ + 1 < totalSequences_) {
            currentSequenceIndex_++;
            loadCurrentSequence();
        }
    });

    btnGenerate_ = std::make_shared<UIButton>("Generate", [this]() {
        if (!structurer_ || !rawTextArea_) return;
        std::string raw = rawTextArea_->getText();
        if (raw.empty()) { addLog("No raw text to generate from", 1); return; }

        static const char* modes[] = {"qa", "thought", "conversation", "instruct", "raw"};
        int fmtIdx = formatDropdown_ ? formatDropdown_->getSelectedIndex() : 0;
        std::string mode = (fmtIdx >= 0 && fmtIdx < 5) ? modes[fmtIdx] : "qa";
        std::string prompt = customPromptArea_ ? customPromptArea_->getText() : "";

        auto results = structurer_->structureEntry(raw, mode, prompt);
        if (results.empty()) {
            std::string err = structurer_->lastError();
            addLog("Generation failed: " + (err.empty() ? "LLM returned no output" : err), 2);
            return;
        }
        std::string combined;
        for (size_t i = 0; i < results.size(); ++i) {
            if (i > 0) combined += "\n\n---\n\n";
            combined += results[i];
        }
        if (structuredTextArea_) structuredTextArea_->setText(combined);
        addLog("Generated " + std::to_string(results.size()) + " pair(s) in " + mode + " format", 0);
    });

    btnAddSequence_ = std::make_shared<UIButton>("+ Add Sequence", [this]() {
        int defaultFormat = formatDropdown_ ? formatDropdown_->getSelectedIndex() : 0;
        sequenceCards_.push_back(buildSequenceCard(defaultFormat));
        addLog("Added new sequence card", 0);
    });

    customPromptArea_ = std::make_shared<UITextArea>("", "",
        [](const std::string&) {});

    btnAppendEntry_ = std::make_shared<UIButton>("Append", [this]() {
        if (!datasetTarget_ || !structuredTextArea_) return;
        std::string structured = structuredTextArea_->getText();
        if (structured.empty()) {
            addLog("Nothing to append — structured text area is empty", 1);
            return;
        }
        std::string raw = rawTextArea_ ? rawTextArea_->getText() : "";
        if (datasetTarget_->appendStructuredEntry(raw, structured)) {
            totalSequences_ = datasetTarget_->massDatasetSize();
            structuredCount_++;
            addLog("Appended new entry to dataset (" + std::to_string(totalSequences_) + " total)", 0);
        } else {
            addLog("Failed to append entry to dataset", 2);
        }
    });

    sliderMaxEntries_ = std::make_shared<UISlider>("Max Entries", 0.0f, 1000.0f, 0.0f, [](float) {}, 10.0f);
    sliderParallel_   = std::make_shared<UISlider>("Parallel",    1.0f, 16.0f,   4.0f, [](float) {}, 1.0f);

    // ── Curriculum view widgets ──────────────────────────

    subjectFilterDropdown_ = std::make_shared<UIDropdown>(
        "Subject",
        std::vector<std::string>{"All", "general", "code", "math", "science",
                                  "history", "medical", "legal"},
        0, [this](int idx, const std::string&) {
            filterSubjectIdx_ = idx;
            poolFilterDirty_ = true;
            rebuildFilteredPool();
        });

    qualityFilterDropdown_ = std::make_shared<UIDropdown>(
        "Quality",
        std::vector<std::string>{"All", "high", "medium", "low"},
        0, [this](int idx, const std::string&) {
            filterQualityIdx_ = idx;
            poolFilterDirty_ = true;
            rebuildFilteredPool();
        });

    poolSearchInput_ = std::make_shared<UIInputBox>();
    poolSearchInput_->setPlaceholder("Filter pool...");

    btnAssignSelected_ = std::make_shared<UIButton>("Assign >>", [this]() {
        if (!datasetTarget_ || datasetTarget_->activeModelId().empty()) {
            addLog("Select a model first", 1); return;
        }
        if (selectedPoolRows_.empty() && selectedPoolRow_ >= 0)
            selectedPoolRows_.push_back(static_cast<size_t>(selectedPoolRow_));
        if (selectedPoolRows_.empty()) return;
        std::vector<size_t> seqIndices;
        for (size_t r : selectedPoolRows_) {
            if (r < filteredPoolIndices_.size())
                seqIndices.push_back(filteredPoolIndices_[r]);
        }
        datasetTarget_->assignMultiple(seqIndices);
        assignedSequences_ = datasetTarget_->assignedCount();
        selectedPoolRows_.clear();
        selectedPoolRow_ = -1;
        poolFilterDirty_ = true;
        rebuildFilteredPool();
        addLog("Assigned " + std::to_string(seqIndices.size()) + " sequence(s)", 0);
    });

    currListActionMenu_ = std::make_shared<UIActionMenu>("Actions");
    currListActionMenu_->addItem("+ Phase", [this]() {
        if (!datasetTarget_) return;
        size_t pos = (selectedCurrRow_ >= 0) ? static_cast<size_t>(selectedCurrRow_) : datasetTarget_->assignedCount();
        datasetTarget_->insertPhaseMarker(pos, "New Phase");
        addLog("Added phase marker", 0);
    });
    currListActionMenu_->addSeparator();
    currListActionMenu_->addItem("<< Remove", [this]() {
        if (!datasetTarget_ || selectedCurrRow_ < 0) return;
        size_t ci = static_cast<size_t>(selectedCurrRow_);
        const auto& order = datasetTarget_->curriculumOrder();
        if (ci >= order.size()) return;
        datasetTarget_->removeSequenceFromModel(order[ci]);
        assignedSequences_ = datasetTarget_->assignedCount();
        selectedCurrRow_ = -1;
        poolFilterDirty_ = true;
        rebuildFilteredPool();
        addLog("Removed sequence from curriculum", 0);
    }, UITheme::Colors::Danger);

    detailContentArea_ = std::make_shared<UITextArea>("Content", "",
        [](const std::string&) {});
    detailStructuredArea_ = std::make_shared<UITextArea>("Structured Output", "",
        [](const std::string&) {});

    btnDetailSave_ = std::make_shared<UIButton>("Save", [this]() {
        if (!datasetTarget_ || detailSeqIndex_ == SIZE_MAX) return;
        std::string text = detailStructuredArea_ ? detailStructuredArea_->getText() : "";
        if (datasetTarget_->writeStructuredOutput(detailSeqIndex_, text)) {
            addLog("Saved structured output for sequence", 0);
            refreshStructurerState();
        } else {
            addLog("Failed to save structured output", 2);
        }
    });

    structWidgets_ = {
        modelDropdown_, formatDropdown_, viewModeDropdown_,
        structSearchInput_, searchPreviewScrollBox_,
        rawTextArea_, structuredTextArea_, customPromptArea_,
        structureActionMenu_, datasetActionMenu_, btnGenerate_,
        btnPrevSeq_, btnNextSeq_,
        sliderMaxEntries_, sliderParallel_, btnAddSequence_, btnAppendEntry_,
        subjectFilterDropdown_, qualityFilterDropdown_, poolSearchInput_,
        btnAssignSelected_, currListActionMenu_,
        detailContentArea_, detailStructuredArea_, btnDetailSave_
    };

    // ── Curriculum tab widgets ────────────────────────────

    cbModelDropdown_ = std::make_shared<UIDropdown>(
        "Model", std::vector<std::string>{"(none)"}, 0,
        [this](int, const std::string&) {});

    cbCurriculumDropdown_ = std::make_shared<UIDropdown>(
        "Curriculum", std::vector<std::string>{"(none)"}, 0,
        [this](int idx, const std::string&) {
            selectActiveCurriculum(idx);
        });

    cbListTypeDropdown_ = std::make_shared<UIDropdown>(
        "", GRIM::presetLabels(), 1,
        [this](int idx, const std::string&) {
            const bool draftRow = (cbDraftPreviewActive_ && selectedCBRow_ == 0);
            if (draftRow || selectedCBRow_ < 0) {
                if (idx >= 0 && idx < GRIM::kConceptPresetCount)
                    syncIntermediateAreas(GRIM::kConceptPresets[idx].defaultIntermediateCount);
                return;
            }
            size_t dsIdx = 0;
            if (!datasetTarget_ || !cbCurriculumRowToBlockIndex(selectedCBRow_, dsIdx))
                return;
            if (idx < 0 || idx >= GRIM::kConceptPresetCount)
                return;
            auto cb = datasetTarget_->getConceptBlock(dsIdx);
            const char* newKey = GRIM::kConceptPresets[idx].key;
            if (cb.format_type == newKey)
                return;
            cb.format_type = newKey;
            cb.recomputeDerived();
            if (datasetTarget_->updateConceptBlock(cb.id, cb))
                addLog("Updated block type", 0);
        });
    cbListTypeDropdown_->setMaxVisibleItems(6);

    {
        std::vector<std::string> typeFilterItems;
        typeFilterItems.push_back("All Types");
        for (int i = 0; i < GRIM::kConceptPresetCount; ++i)
            typeFilterItems.push_back(GRIM::kConceptPresets[i].label);
        cbTypeFilterDropdown_ = std::make_shared<UIDropdown>(
            "", typeFilterItems, 0,
            [this](int idx, const std::string&) {
                cbFormatFilterIdx_ = idx;
                cbFilterDirty_ = true;
            });
        cbTypeFilterDropdown_->setMaxVisibleItems(8);
    }

    cbCurriculumFilterToggle_ = std::make_shared<UIToggle>(
        "In Curriculum", false,
        [this](bool state) {
            cbCurriculumFilterActive_ = state;
            cbFilterDirty_ = true;
        });

    cbSearchInput_ = std::make_shared<UIInputBox>();
    cbSearchInput_->setPlaceholder("Search concept blocks...");

    cbNameInput_ = std::make_shared<UIInputBox>();
    cbNameInput_->setPlaceholder("Block name...");

    cbQuestionArea_ = std::make_shared<UITextArea>("Question", "",
        [](const std::string&) {});
    cbAnswerArea_ = std::make_shared<UITextArea>("Answer", "",
        [](const std::string&) {});
    cbCustomPromptArea_ = std::make_shared<UITextArea>("Custom Prompt", "",
        [](const std::string&) {});

    // ── State 0 / Execution / State 1 widgets ───────────
    cbState0TypeInput_ = std::make_shared<UIInputBox>();
    cbState0TypeInput_->setPlaceholder("e.g. arithmetic");
    cbState0AtomsInput_ = std::make_shared<UIInputBox>();
    cbState0AtomsInput_->setPlaceholder("e.g. 2.0, 3.0");

    btnCBGenerate_ = std::make_shared<UIButton>("Generate", [this]() {
        generateConceptBlock();
    });

    stepActionMenu_ = std::make_shared<UIActionMenu>("Steps");
    stepActionMenu_->addItem("+ Step", [this]() {
        auto area = std::make_shared<UITextArea>(
            "Step " + std::to_string(cbIntermediateAreas_.size() + 1), "",
            [](const std::string&) {});
        cbIntermediateAreas_.push_back(area);
    }, UITheme::Colors::Success);
    stepActionMenu_->addItem("- Step", [this]() {
        if (!cbIntermediateAreas_.empty())
            cbIntermediateAreas_.pop_back();
    }, UITheme::Colors::Danger);

    execStepActionMenu_ = std::make_shared<UIActionMenu>("Exec Steps");
    execStepActionMenu_->addItem("+ Exec Step", [this]() {
        syncExecStepRows(static_cast<int>(cbExecStepRows_.size()) + 1);
    }, UITheme::Colors::Success);
    execStepActionMenu_->addItem("- Exec Step", [this]() {
        if (!cbExecStepRows_.empty())
            cbExecStepRows_.pop_back();
    }, UITheme::Colors::Danger);

    blockActionMenu_ = std::make_shared<UIActionMenu>("Block");
    blockActionMenu_->addItem("New Block", [this]() {
        clearCBEditor();
        cbDraftPreviewActive_ = true;
        selectedCBRow_        = 0;
        cbListScrollOffset_   = 0.0f;
        int presetIdx = cbListTypeDropdown_ ? cbListTypeDropdown_->getSelectedIndex() : 1;
        if (presetIdx >= 0 && presetIdx < GRIM::kConceptPresetCount)
            syncIntermediateAreas(GRIM::kConceptPresets[presetIdx].defaultIntermediateCount);
    }, UITheme::Colors::Success);
    blockActionMenu_->addItem("Save", [this]() {
        if (!datasetTarget_) return;
        std::string name = cbNameInput_ ? cbNameInput_->getText() : "";
        std::string question = cbQuestionArea_ ? cbQuestionArea_->getText() : "";
        if (name.empty() && question.empty()) {
            addLog("ConceptBlock needs a name or question", 1);
            return;
        }

        int presetIdx = cbListTypeDropdown_ ? cbListTypeDropdown_->getSelectedIndex() : 1;
        std::string formatKey = (presetIdx >= 0 && presetIdx < GRIM::kConceptPresetCount)
            ? GRIM::kConceptPresets[presetIdx].key : "chain_of_thought";

        GRIM::ConceptBlock cb;
        cb.name = name;
        cb.question = question;
        cb.answer = cbAnswerArea_ ? cbAnswerArea_->getText() : "";
        cb.format_type = formatKey;
        cb.intermediates.clear();
        for (const auto& area : cbIntermediateAreas_) {
            cb.intermediates.push_back(area ? area->getText() : "");
        }

        // State 0
        cb.state_0.type = cbState0TypeInput_ ? cbState0TypeInput_->getText() : "";
        if (cbState0AtomsInput_) {
            std::string atomsStr = cbState0AtomsInput_->getText();
            cb.state_0.atoms.clear();
            if (!atomsStr.empty()) {
                std::istringstream iss(atomsStr);
                std::string tok;
                while (std::getline(iss, tok, ',')) {
                    try { cb.state_0.atoms.push_back(std::stod(tok)); }
                    catch (...) {}
                }
            }
        }

        // Execution steps
        cb.execution.clear();
        for (const auto& row : cbExecStepRows_) {
            GRIM::ConceptExecutionStep step;
            static const char* opNames[] = {"add", "sub", "mul", "div"};
            int opIdx = row.opDropdown ? row.opDropdown->getSelectedIndex() : 0;
            step.op = (opIdx >= 0 && opIdx < 4) ? opNames[opIdx] : "add";

            if (row.argSlotsInput) {
                std::istringstream iss(row.argSlotsInput->getText());
                std::string tok;
                while (std::getline(iss, tok, ',')) {
                    try { step.arg_slots.push_back(std::stoi(tok)); }
                    catch (...) {}
                }
            }
            if (row.argsInput) {
                std::istringstream iss(row.argsInput->getText());
                std::string tok;
                while (std::getline(iss, tok, ',')) {
                    try { step.args.push_back(std::stod(tok)); }
                    catch (...) {}
                }
            }
            if (row.resultInput) {
                try { step.result = std::stod(row.resultInput->getText()); }
                catch (...) { step.result = 0.0; }
            }
            cb.execution.push_back(std::move(step));
        }

        // State 1 — derived from last execution step result
        if (!cb.execution.empty()) {
            cb.state_1.result = cb.execution.back().result;
            cb.state_1.has_result = true;
        } else {
            cb.state_1.has_result = false;
        }

        cb.recomputeDerived();
        cb.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        size_t existingIdx = 0;
        if (cbCurriculumRowToBlockIndex(selectedCBRow_, existingIdx)) {
            auto existing = datasetTarget_->getConceptBlock(existingIdx);
            cb.id = existing.id;
            cb.source_sequence_id = existing.source_sequence_id;
            if (datasetTarget_->updateConceptBlock(cb.id, cb)) {
                addLog("Updated ConceptBlock: " + cb.name, 0);
            } else {
                addLog("Failed to update ConceptBlock", 2);
            }
        } else {
            std::string seed = cb.name + cb.question + std::to_string(cb.timestamp);
            cb.id = "";
            uint64_t h1 = 14695981039346656037ULL;
            uint64_t h2 = 14695981039346656037ULL;
            for (size_t i = 0; i < seed.size(); ++i) {
                uint8_t c = static_cast<uint8_t>(seed[i]);
                h1 ^= c; h1 *= 1099511628211ULL;
                if (i + 1 < seed.size()) { h2 ^= static_cast<uint8_t>(seed[i+1]); h2 *= 1099511628211ULL; }
            }
            std::ostringstream oss;
            oss << std::hex << std::setfill('0') << std::setw(16) << h1 << std::setw(16) << h2;
            cb.id = oss.str();

            if (datasetTarget_->addConceptBlock(cb)) {
                addLog("Created ConceptBlock: " + cb.name, 0);
                cbDraftPreviewActive_ = false;
                cbFilterDirty_         = true;
                rebuildFilteredCBList();
                for (int i = 0; i < cbCurriculumListRowCount(); ++i) {
                    size_t idx = 0;
                    if (!cbCurriculumRowToBlockIndex(i, idx))
                        continue;
                    if (datasetTarget_->getConceptBlock(idx).id == cb.id) {
                        selectedCBRow_ = i;
                        break;
                    }
                }
                syncCBListTypeDropdownFromToolbar();
            } else {
                addLog("Failed to create ConceptBlock", 2);
            }
        }
        refreshCurriculumTabState();
    });
    blockActionMenu_->addSeparator();
    blockActionMenu_->addItem("Delete", [this]() {
        if (cbCurriculumRowIsDraft(selectedCBRow_)) {
            cbDraftPreviewActive_ = false;
            selectedCBRow_        = -1;
            clearCBEditor();
            return;
        }
        if (!datasetTarget_ || selectedCBRow_ < 0) return;
        size_t realIdx = 0;
        if (!cbCurriculumRowToBlockIndex(selectedCBRow_, realIdx))
            return;
        auto cb = datasetTarget_->getConceptBlock(realIdx);
        if (datasetTarget_->removeConceptBlock(cb.id)) {
            addLog("Deleted ConceptBlock: " + cb.name, 0);
            selectedCBRow_ = -1;
            clearCBEditor();
            cbFilterDirty_ = true;
            refreshCurriculumTabState();
        }
    }, UITheme::Colors::Danger);

    // ── Curriculum action menus ────────────────────────

    blockCurriculumMenu_ = std::make_shared<UIActionMenu>("Curriculum");
    blockCurriculumMenu_->addItem("Add to Curriculum", [this]() {
        if (activeCurriculumId_.empty()) {
            addLog("Select a curriculum first", 1);
            return;
        }
        if (cbCurriculumRowIsDraft(selectedCBRow_)) {
            addLog("Save the block before adding to curriculum", 1);
            return;
        }
        if (!datasetTarget_ || selectedCBRow_ < 0) return;
        size_t idx = 0;
        if (!cbCurriculumRowToBlockIndex(selectedCBRow_, idx))
            return;
        auto cb = datasetTarget_->getConceptBlock(idx);
        if (datasetTarget_->addConceptBlockToCurriculum(cb.id, activeCurriculumId_)) {
            addLog("Added to curriculum: " + cb.name, 0);
            refreshCurriculumTabState();
        }
    });
    blockCurriculumMenu_->addItem("Remove from Curriculum", [this]() {
        if (activeCurriculumId_.empty()) return;
        if (cbCurriculumRowIsDraft(selectedCBRow_)) return;
        if (!datasetTarget_ || selectedCBRow_ < 0) return;
        size_t idx = 0;
        if (!cbCurriculumRowToBlockIndex(selectedCBRow_, idx))
            return;
        auto cb = datasetTarget_->getConceptBlock(idx);
        if (datasetTarget_->removeConceptBlockFromCurriculum(cb.id, activeCurriculumId_)) {
            addLog("Removed from curriculum: " + cb.name, 0);
            refreshCurriculumTabState();
        }
    }, UITheme::Colors::Danger);

    curriculumActionMenu_ = std::make_shared<UIActionMenu>("Manage");
    curriculumActionMenu_->addItem("New Curriculum", [this]() {
        if (!datasetTarget_) return;
        GRIM::Curriculum curr;
        curr.name = "Untitled Curriculum";
        curr.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::string seed = curr.name + std::to_string(curr.timestamp)
                         + std::to_string(datasetTarget_->curriculumCount());
        uint64_t h = 14695981039346656037ULL;
        for (auto c : seed) { h ^= static_cast<uint8_t>(c); h *= 1099511628211ULL; }
        std::ostringstream oss;
        oss << std::hex << std::setfill('0') << std::setw(16) << h;
        curr.id = "curr_" + oss.str();

        if (datasetTarget_->addCurriculum(curr)) {
            addLog("Created curriculum: " + curr.name, 0);
            activeCurriculumId_ = curr.id;
            populateCBCurriculumDropdown();
            refreshCurriculumTabState();
        }
    }, UITheme::Colors::Success);
    curriculumActionMenu_->addItem("Rename Curriculum", [this]() {
        if (activeCurriculumId_.empty() || !datasetTarget_) {
            addLog("Select a curriculum first", 1);
            return;
        }
        auto curr = datasetTarget_->getCurriculumById(activeCurriculumId_);
        if (cbCurriculumRenameInput_) {
            cbCurriculumRenameInput_->setText(curr.name);
        }
        renamingCurriculum_ = true;
        renameJustActivated_ = true;
    });
    curriculumActionMenu_->addSeparator();
    curriculumActionMenu_->addItem("Delete Curriculum", [this]() {
        if (activeCurriculumId_.empty() || !datasetTarget_) return;
        auto curr = datasetTarget_->getCurriculumById(activeCurriculumId_);
        if (datasetTarget_->removeCurriculum(activeCurriculumId_)) {
            addLog("Deleted curriculum: " + curr.name, 0);
            activeCurriculumId_.clear();
            populateCBCurriculumDropdown();
            refreshCurriculumTabState();
        }
    }, UITheme::Colors::Danger);
    curriculumActionMenu_->addSeparator();
    curriculumActionMenu_->addItem("Assign to Model", [this]() {
        if (activeCurriculumId_.empty()) {
            addLog("Select a curriculum first", 1);
            return;
        }
        if (!datasetTarget_ || datasetTarget_->activeModelId().empty()) {
            addLog("Select a model first", 1);
            return;
        }
        if (datasetTarget_->isCurriculumAssigned(activeCurriculumId_)) {
            if (datasetTarget_->removeCurriculumFromModel(activeCurriculumId_)) {
                addLog("Unassigned curriculum from model", 0);
                refreshCurriculumTabState();
            }
        } else {
            if (datasetTarget_->assignCurriculumToModel(activeCurriculumId_)) {
                auto curr = datasetTarget_->getCurriculumById(activeCurriculumId_);
                addLog("Assigned curriculum to model: " + curr.name, 0);
                refreshCurriculumTabState();
            }
        }
    });
    curriculumActionMenu_->addSeparator();
    curriculumActionMenu_->addItem("Toggle Format Mode", [this]() {
        if (activeCurriculumId_.empty() || !datasetTarget_) {
            addLog("Select a curriculum first", 1);
            return;
        }
        auto curr = datasetTarget_->getCurriculumById(activeCurriculumId_);
        curr.format_as_concept = !curr.format_as_concept;
        if (datasetTarget_->updateCurriculum(activeCurriculumId_, curr)) {
            std::string mode = curr.format_as_concept ? "Concept Block" : "Plain Text (PT)";
            addLog("Curriculum format set to: " + mode, 0);
            refreshCurriculumTabState();
        }
    });

    // ── Curriculum rename input ──────────────────────
    cbCurriculumRenameInput_ = std::make_shared<UIInputBox>();
    cbCurriculumRenameInput_->setPlaceholder("Curriculum name...");
    cbCurriculumRenameInput_->OnTextSubmitted.Bind([this](const std::string& newName) {
        if (newName.empty() || activeCurriculumId_.empty() || !datasetTarget_) {
            renamingCurriculum_ = false;
            return;
        }
        auto curr = datasetTarget_->getCurriculumById(activeCurriculumId_);
        curr.name = newName;
        if (datasetTarget_->updateCurriculum(activeCurriculumId_, curr)) {
            addLog("Renamed curriculum to: " + newName, 0);
            populateCBCurriculumDropdown();
        }
        renamingCurriculum_ = false;
    });

    curriculumWidgets_ = {
        cbModelDropdown_, cbCurriculumDropdown_, cbCurriculumRenameInput_,
        cbListTypeDropdown_, cbTypeFilterDropdown_, cbCurriculumFilterToggle_, cbSearchInput_,
        cbNameInput_, cbQuestionArea_, cbAnswerArea_, cbCustomPromptArea_,
        cbState0TypeInput_, cbState0AtomsInput_,
        btnCBGenerate_, stepActionMenu_, execStepActionMenu_, blockActionMenu_,
        curriculumActionMenu_, blockCurriculumMenu_
    };

    // ── Hide non-active groups ──────────────────────────

    for (auto& w : sourcesWidgets_)    w->setVisible(false);
    for (auto& w : hfWidgets_)         w->setVisible(false);
    for (auto& w : structWidgets_)     w->setVisible(false);
    for (auto& w : curriculumWidgets_) w->setVisible(false);

    // ── Backend services ────────────────────────────────

    pipelineOrchestrator_ = std::make_unique<GRIM::Pipeline::PipelineOrchestrator>();
    hfWebhook_            = std::make_unique<GRIM::DataCollection::HuggingFaceWebhook>();

    {
        namespace fs = std::filesystem;
        auto snapshot = GRIM::Config::loadAiConfigSnapshot();

        fs::path grimRoot = GRIM::Config::detail::resolveGrimRoot();
        fs::path modelStoreRoot;
        if (snapshot && snapshot->has_grim_paths && !snapshot->grim_text_model_store.empty())
            modelStoreRoot = fs::path(snapshot->grim_text_model_store);
        else
            modelStoreRoot = grimRoot / "resources" / "models" / "model_store";

        fs::path massDatasetPath;
        if (snapshot && snapshot->has_grim_paths && !snapshot->grim_text_training_data.empty())
            massDatasetPath = fs::path(snapshot->grim_text_training_data).parent_path() / "mass_dataset.jsonl";
        else
            massDatasetPath = grimRoot / "resources" / "models" / "GRIM-text" / "training" / "data" / "mass_dataset.jsonl";

        datasetTarget_ = std::make_unique<DatasetTarget>(modelStoreRoot, massDatasetPath);

        GRIM::DataCollection::DataStructuringConfig structCfg;
        try {
            auto snapshot = GRIM::Config::loadAiConfigSnapshot();
            if (snapshot && snapshot->document.contains("data_structuring")) {
                structCfg = GRIM::DataCollection::DataStructuringConfig::fromJson(
                    snapshot->document["data_structuring"]);
            }
        } catch (...) {}
        if (structCfg.ollama_model.empty()) structCfg.ollama_model = "llama3.1:8b";
        structurer_ = std::make_unique<GRIM::DataCollection::DataStructurer>(structCfg);
    }

    // ── Load persisted state ────────────────────────────

    loadUIConfig();
    loadDirectoryCollectionPathFromConfig();
    loadSourceCards();
    loadDownloadQueue();
    loadHFTokenFromConfig();
    updateDatasetStats();
    populateModelDropdown();
    refreshStructurerState();

    if (datasetTarget_) datasetTarget_->loadConceptBlocks();
    if (datasetTarget_) datasetTarget_->loadCurriculumRegistry();

    addLog("DataHub initialized", 0);
    LOG_DEBUG("DataHub", "Panel initialized — 5 tabs ready");
}

// =========================================================
// Destructor
// =========================================================

UIDataHubPanel::~UIDataHubPanel() {
    saveUIConfig();
    if (sourcesDirty_) saveSourceCards();
    saveDownloadQueue();
    if (pipelineOrchestrator_) pipelineOrchestrator_->stopPipeline();
    LOG_DEBUG("DataHub", "Panel destroyed — config saved");
}

// =========================================================
// View management
// =========================================================

void UIDataHubPanel::setView(DataHubView view) {
    if (view == activeView_) return;

    auto hideGroup = [](std::vector<std::shared_ptr<Widget>>& g) {
        for (auto& w : g) w->setVisible(false);
    };
    auto showGroup = [](std::vector<std::shared_ptr<Widget>>& g) {
        for (auto& w : g) w->setVisible(true);
    };

    if (activeView_ == DataHubView::Sources && sourcesDirty_)
        saveSourceCards();

    switch (activeView_) {
        case DataHubView::Home:        hideGroup(homeWidgets_);       break;
        case DataHubView::Sources:     hideGroup(sourcesWidgets_);    break;
        case DataHubView::HuggingFace: hideGroup(hfWidgets_);         break;
        case DataHubView::Structurer:  hideGroup(structWidgets_);     break;
        case DataHubView::Curriculum:  hideGroup(curriculumWidgets_); break;
    }

    activeView_ = view;

    switch (activeView_) {
        case DataHubView::Home:        showGroup(homeWidgets_);       break;
        case DataHubView::Sources:     showGroup(sourcesWidgets_);    break;
        case DataHubView::HuggingFace: showGroup(hfWidgets_);         break;
        case DataHubView::Structurer:  showGroup(structWidgets_);     break;
        case DataHubView::Curriculum:  showGroup(curriculumWidgets_); break;
    }

    if (activeView_ == DataHubView::Structurer)
        refreshStructurerState();
    if (activeView_ == DataHubView::Curriculum)
        refreshCurriculumTabState();

    LOG_DEBUG("DataHub", "Switched to tab " + std::to_string(static_cast<int>(view)));
}

// =========================================================
// Update
// =========================================================

void UIDataHubPanel::update(const InputState& input, float dt) {
    if (!isVisible()) return;
    UIPanel::update(input, dt);
    // Cache edge-triggered click for directory file list (consumed in draw)
    if (input.mousePressed[0]) {
        Vec2 m = input.mousePos;
        if (!dirFileEntries_.empty() &&
            m.x >= dirScrollAreaRect_.x && m.x <= dirScrollAreaRect_.x + dirScrollAreaRect_.w &&
            m.y >= dirScrollAreaRect_.y && m.y <= dirScrollAreaRect_.y + dirScrollAreaRect_.h) {
            dirClickPending_ = true;
            dirClickPos_     = m;
        }
    }

    // Tab buttons (always active)
    float tabX = position.x + 10.0f;
    tabHomeBtn_->setPosition(tabX, position.y + kTabBarY);
    tabSourcesBtn_->setPosition(tabX + 95.0f, position.y + kTabBarY);
    tabHFBtn_->setPosition(tabX + 190.0f, position.y + kTabBarY);
    tabStructBtn_->setPosition(tabX + 305.0f, position.y + kTabBarY);
    tabCurriculumBtn_->setPosition(tabX + 410.0f, position.y + kTabBarY);

    tabHomeBtn_->update(input, dt);
    tabSourcesBtn_->update(input, dt);
    tabHFBtn_->update(input, dt);
    tabStructBtn_->update(input, dt);
    tabCurriculumBtn_->update(input, dt);

    // Background timers (run regardless of active tab)
    pollTimer_ += dt;
    if (pollTimer_ >= kPollInterval) { pollTimer_ = 0.0f; pollCollectionManager(); }

    statsUpdateTimer_ += dt;
    if (statsUpdateTimer_ >= kStatsInterval) { statsUpdateTimer_ = 0.0f; updateDatasetStats(); }

    if (collectionActive_) collectionAnimTime_ += dt;
    if (hfSearching_.load()) searchAnimTime_ += dt;

    // Active tab widget updates
    switch (activeView_) {
        case DataHubView::Home:
            for (auto& w : homeWidgets_) w->update(input, dt);

            // Directory file list scroll wheel
            if (!dirFileEntries_.empty()) {
                Vec2 m = input.mousePos;
                if (m.x >= dirScrollAreaRect_.x && m.x <= dirScrollAreaRect_.x + dirScrollAreaRect_.w &&
                    m.y >= dirScrollAreaRect_.y && m.y <= dirScrollAreaRect_.y + dirScrollAreaRect_.h) {
                    dirScrollOffset_ -= input.mouseWheelDelta;
                    static constexpr float kDirRowH = 28.0f;
                    float totalH  = dirFileEntries_.size() * kDirRowH;
                    float maxScr  = std::max(0.0f, totalH - (dirScrollAreaRect_.h - 18.0f));
                    dirScrollOffset_ = std::clamp(dirScrollOffset_, 0.0f, maxScr);
                }
            }
            break;

        case DataHubView::Sources: {
            btnAddCard_->update(input, dt);

            for (auto& card : sourceCards_) {
                card.nameInput->update(input, dt);
                card.urlInput->update(input, dt);
                card.prioritySlider->update(input, dt);
                card.depthSlider->update(input, dt);
                card.limitSlider->update(input, dt);
                card.enabledToggle->update(input, dt);
                card.deleteBtn->update(input, dt);

                card.name       = card.nameInput->getText();
                card.url        = card.urlInput->getText();
                card.priority   = static_cast<int>(card.prioritySlider->getValue());
                card.crawlDepth = static_cast<int>(card.depthSlider->getValue());
                card.fetchLimit = static_cast<int>(card.limitSlider->getValue());
                card.enabled    = card.enabledToggle->getState();
            }

            if (cardToDelete_ >= 0) {
                size_t delId = static_cast<size_t>(cardToDelete_);
                sourceCards_.erase(
                    std::remove_if(sourceCards_.begin(), sourceCards_.end(),
                        [delId](const SourceCard& c) { return c.cardId == delId; }),
                    sourceCards_.end());
                cardToDelete_ = -1;
                sourcesDirty_ = true;
                addLog("Removed source card", 0);
            }

            // Mouse wheel scrolling over the card area
            PanelRect pContent = getContentRect();
            float scrollAreaTop = pContent.origin.y + (kContentTopY - kTabBarY) + 5.0f;
            float scrollAreaH   = pContent.size.y - (kContentTopY - kTabBarY) - 60.0f;
            Vec2 m = input.mousePos;
            if (m.x >= pContent.origin.x && m.x <= pContent.origin.x + pContent.size.x &&
                m.y >= scrollAreaTop && m.y <= scrollAreaTop + scrollAreaH) {
                sourcesScrollOffset_ -= input.mouseWheelDelta;
                float totalH = sourceCards_.size() * (kCardHeight + kCardGap) + 10.0f;
                float maxScroll = std::max(0.0f, totalH - scrollAreaH);
                sourcesScrollOffset_ = std::clamp(sourcesScrollOffset_, 0.0f, maxScroll);
            }
            break;
        }

        case DataHubView::HuggingFace:
            if (hfPreviewApply_.exchange(false) && hfPreviewArea_) {
                hfPreviewArea_->setText(hfPreviewPendingText_);
            }
            hfSearchInput_->update(input, dt);
            hfSearchBuffer_ = hfSearchInput_->getText();
            hfTokenInput_->update(input, dt);
            {
                std::string newToken = hfTokenInput_->getText();
                if (newToken != hfTokenBuffer_) {
                    hfTokenBuffer_ = newToken;
                    if (hfWebhook_) hfWebhook_->setApiToken(hfTokenBuffer_);
                }
            }
            btnSearchHF_->update(input, dt);
            btnBrowseHF_->update(input, dt);
            hfCategoryDropdown_->update(input, dt);
            hfResultsScrollBox_->update(input, dt);
            sliderMaxHFResults_->update(input, dt);
            if (hfPreviewArea_) hfPreviewArea_->update(input, dt);
            btnHFQueuePreview_->update(input, dt);
            queueActionMenu_->update(input, dt);
            break;

        case DataHubView::Structurer:
            for (auto& w : structWidgets_) w->update(input, dt);

            if (structViewMode_ == 1) {
                btnAddSequence_->update(input, dt);
                for (auto& card : sequenceCards_) {
                    card.formatDropdown->update(input, dt);
                    card.generateBtn->update(input, dt);
                    if (card.formatIndex == 1) card.addEntryBtn->update(input, dt);
                    card.deleteBtn->update(input, dt);
                    card.saveBtn->update(input, dt);
                    for (auto& entry : card.entries)
                        entry.textArea->update(input, dt);
                }

                if (seqCardToDelete_ >= 0) {
                    size_t delId = static_cast<size_t>(seqCardToDelete_);
                    removeSequenceCard(delId);
                    seqCardToDelete_ = -1;
                    addLog("Removed sequence card", 0);
                }

                Vec2 m = input.mousePos;
                if (seqCardAreaH_ > 0.0f &&
                    m.x >= position.x && m.x <= position.x + size.x &&
                    m.y >= seqCardAreaTop_ && m.y <= seqCardAreaTop_ + seqCardAreaH_) {
                    seqScrollOffset_ -= input.mouseWheelDelta;
                    float totalH = 10.0f;
                    for (const auto& c : sequenceCards_)
                        totalH += sequenceCardHeight(c) + kCardGap;
                    float maxScroll = std::max(0.0f, totalH - seqCardAreaH_);
                    seqScrollOffset_ = std::clamp(seqScrollOffset_, 0.0f, maxScroll);
                }
            }

            if (structViewMode_ == 2) {
                if (subjectFilterDropdown_) subjectFilterDropdown_->update(input, dt);
                if (qualityFilterDropdown_) qualityFilterDropdown_->update(input, dt);
                if (poolSearchInput_)       poolSearchInput_->update(input, dt);
                if (btnAssignSelected_)     btnAssignSelected_->update(input, dt);
                if (currListActionMenu_)    currListActionMenu_->update(input, dt);
                if (detailPanelOpen_) {
                    if (detailContentArea_)     detailContentArea_->update(input, dt);
                    if (detailStructuredArea_)  detailStructuredArea_->update(input, dt);
                    if (btnDetailSave_)         btnDetailSave_->update(input, dt);
                }

                // Compute layout geometry (mirrors drawStructurerTab + drawCurriculumView)
                PanelRect cRect = getContentRect();
                float cvX = cRect.origin.x + 15.0f;
                float cvFullW = cRect.size.x - 30.0f;
                // toolbar row1 + gap + toolbar row2 + gap + divider label + filter bar + gap
                float cvY = cRect.origin.y + 10.0f
                          + 36.0f + kStructRowGap
                          + 36.0f + kStructRowGap
                          + kStructLabelSpace
                          + kFilterBarH + 6.0f;
                float cvStatusY     = cRect.origin.y + cRect.size.y - 30.0f;
                float cvPromptH     = 50.0f + 6.0f + 16.0f + 2.0f + 8.0f;
                float cvContentEndY = cvStatusY - cvPromptH;
                float detailH = detailPanelOpen_ ? detailPanelHeight_ : kDetailDividerH;
                float splitH = (cvContentEndY - cvY) - detailH - 6.0f;
                if (splitH < 80.0f) splitH = 80.0f;

                float leftW  = cvFullW * 0.58f;
                float rightW = cvFullW - leftW - 10.0f;

                Vec2 m = input.mousePos;

                // Pool table interaction
                float poolBodyY = cvY + kPoolHeaderH;
                float poolBodyH = splitH - kPoolHeaderH;
                if (m.x >= cvX && m.x <= cvX + leftW &&
                    m.y >= poolBodyY && m.y <= poolBodyY + poolBodyH) {
                    // Hover
                    int startRow = static_cast<int>(poolScrollOffset_ / kPoolRowH);
                    float relY = m.y - poolBodyY + (poolScrollOffset_ - startRow * kPoolRowH);
                    int hovRow = startRow + static_cast<int>(relY / kPoolRowH);
                    if (hovRow >= 0 && hovRow < static_cast<int>(filteredPoolIndices_.size()))
                        hoveredPoolRow_ = hovRow;
                    else
                        hoveredPoolRow_ = -1;

                    // Click to select
                    if (input.mousePressed[0] && hoveredPoolRow_ >= 0) {
                        selectPoolRow(hoveredPoolRow_);
                    }

                    // Scroll
                    poolScrollOffset_ -= input.mouseWheelDelta;
                    float maxScroll = std::max(0.0f,
                        static_cast<float>(filteredPoolIndices_.size()) * kPoolRowH - poolBodyH);
                    poolScrollOffset_ = std::clamp(poolScrollOffset_, 0.0f, maxScroll);
                } else {
                    hoveredPoolRow_ = -1;
                }

                // Curriculum list interaction
                float currX = cvX + leftW + 10.0f;
                float currBodyY = cvY + kPoolHeaderH;
                float currBodyH = splitH - kPoolHeaderH;
                if (m.x >= currX && m.x <= currX + rightW &&
                    m.y >= currBodyY && m.y <= currBodyY + currBodyH && datasetTarget_) {

                    const auto& order = datasetTarget_->curriculumOrder();
                    const auto& phases = datasetTarget_->phaseMarkers();

                    // Build flat list heights to find hovered row
                    float itemY = currBodyY - currScrollOffset_;
                    int hovCurr = -1;
                    size_t phI = 0;
                    for (size_t i = 0; i < order.size(); ++i) {
                        while (phI < phases.size() && phases[phI].position <= i) {
                            itemY += kPhaseRowH;
                            phI++;
                        }
                        if (m.y >= itemY && m.y < itemY + kCurrRowH)
                            hovCurr = static_cast<int>(i);
                        itemY += kCurrRowH;
                    }
                    hoveredCurrRow_ = hovCurr;

                    if (input.mousePressed[0] && hoveredCurrRow_ >= 0) {
                        selectCurriculumRow(hoveredCurrRow_);
                    }

                    // Check for move up/down arrow clicks
                    if (input.mousePressed[0] && hoveredCurrRow_ >= 0) {
                        float arrowX = currX + rightW - 50.0f;
                        if (m.x >= arrowX && m.x < arrowX + 16.0f) {
                            size_t ci = static_cast<size_t>(hoveredCurrRow_);
                            if (ci > 0) {
                                datasetTarget_->moveSequenceUp(ci);
                                selectedCurrRow_ = static_cast<int>(ci - 1);
                                selectCurriculumRow(selectedCurrRow_);
                            }
                        } else if (m.x >= arrowX + 18.0f && m.x < arrowX + 34.0f) {
                            size_t ci = static_cast<size_t>(hoveredCurrRow_);
                            if (ci + 1 < order.size()) {
                                datasetTarget_->moveSequenceDown(ci);
                                selectedCurrRow_ = static_cast<int>(ci + 1);
                                selectCurriculumRow(selectedCurrRow_);
                            }
                        }
                    }

                    // Scroll
                    currScrollOffset_ -= input.mouseWheelDelta;
                    float totalCurrH = order.size() * kCurrRowH + phases.size() * kPhaseRowH;
                    float maxCurrScroll = std::max(0.0f, totalCurrH - currBodyH);
                    currScrollOffset_ = std::clamp(currScrollOffset_, 0.0f, maxCurrScroll);
                } else {
                    hoveredCurrRow_ = -1;
                }

                // Detail panel toggle
                float detailDivY = cvY + splitH + 6.0f;
                if (input.mousePressed[0] &&
                    m.x >= cvX && m.x <= cvX + cvFullW &&
                    m.y >= detailDivY && m.y <= detailDivY + kDetailDividerH) {
                    detailPanelOpen_ = !detailPanelOpen_;
                }
            }

            if (structViewMode_ == 0 && structSearchInput_ && datasetTarget_) {
                std::string query = structSearchInput_->getText();
                if (!query.empty() && searchPreviewScrollBox_) {
                    auto results = datasetTarget_->searchSequences(query, 12);
                    searchPreviewScrollBox_->clearChildren();
                    for (const auto& r : results) {
                        size_t idx = r.index;
                        auto btn = std::make_shared<UIButton>(r.preview,
                            [this, idx]() {
                                currentSequenceIndex_ = idx;
                                loadCurrentSequence();
                            });
                        btn->setSize(400.0f, 30.0f);
                        searchPreviewScrollBox_->addChild(btn);
                    }
                    searchPreviewScrollBox_->autoLayoutChildren();
                }
            }
            break;

        case DataHubView::Curriculum: {
                PanelRect content = getContentRect();
                content.origin.y += (kContentTopY - kTabBarY);
                content.size.y   -= (kContentTopY - kTabBarY);

                const CurriculumTabLayout layout = computeCurriculumTabLayout(content);
                const float cbListX = layout.listX;
                const float cbListY = layout.listY;
                const float cbListW = layout.listW;
                const float cbListH = layout.listH;
                const bool cbListTypeDropdownWasExpanded =
                    cbListTypeDropdown_ && cbListTypeDropdown_->isExpanded();

            layoutCBListTypeDropdownInList(cbListX, cbListY, cbListW);

            for (auto& w : curriculumWidgets_) w->update(input, dt);
            for (auto& area : cbIntermediateAreas_)
                if (area) area->update(input, dt);
            for (auto& row : cbExecStepRows_) {
                if (row.opDropdown)    row.opDropdown->update(input, dt);
                if (row.argSlotsInput) row.argSlotsInput->update(input, dt);
                if (row.argsInput)     row.argsInput->update(input, dt);
                if (row.resultInput)   row.resultInput->update(input, dt);
            }

            {
                std::string curSearch = cbSearchInput_ ? cbSearchInput_->getText() : "";
                if (curSearch != cbFilterSearch_) {
                    cbFilterSearch_ = curSearch;
                    cbFilterDirty_ = true;
                }
                int curTypeIdx = cbTypeFilterDropdown_ ? cbTypeFilterDropdown_->getSelectedIndex() : 0;
                if (curTypeIdx != cbFormatFilterIdx_) {
                    cbFormatFilterIdx_ = curTypeIdx;
                    cbFilterDirty_ = true;
                }
                // Track curriculum dropdown selection change
                if (renamingCurriculum_ && cbCurriculumRenameInput_) {
                    if (renameJustActivated_) {
                        // Skip cancel check on the frame the rename was activated
                        renameJustActivated_ = false;
                    } else if (!cbCurriculumRenameInput_->isFocused() && input.mousePressed[0]) {
                        renamingCurriculum_ = false;
                    }
                } else if (cbCurriculumDropdown_) {
                    int curCurrIdx = cbCurriculumDropdown_->getSelectedIndex();
                    const auto& curricula = datasetTarget_ ? datasetTarget_->getCurriculums() : std::vector<GRIM::Curriculum>{};
                    std::string selectedId;
                    if (curCurrIdx > 0 && curCurrIdx <= static_cast<int>(curricula.size()))
                        selectedId = curricula[curCurrIdx - 1].id;
                    if (selectedId != activeCurriculumId_)
                        selectActiveCurriculum(curCurrIdx);
                }
                if (cbFilterDirty_) rebuildFilteredCBList();

                Vec2 m = input.mousePos;
                const int rowCount = cbCurriculumListRowCount();
                const bool cbListTypeDropdownOwnsInput =
                    cbListTypeDropdownWasExpanded ||
                    (cbListTypeDropdown_ && cbListTypeDropdown_->isExpanded());
                if (!cbListTypeDropdownOwnsInput &&
                    m.x >= cbListX && m.x <= cbListX + cbListW &&
                    m.y >= cbListY + kPoolHeaderH && m.y <= cbListY + cbListH) {
                    float bodyY = cbListY + kPoolHeaderH;
                    int startRow = static_cast<int>(cbListScrollOffset_ / kCBListRowH);
                    float relY = m.y - bodyY + (cbListScrollOffset_ - startRow * kCBListRowH);
                    int hovRow = startRow + static_cast<int>(relY / kCBListRowH);
                    if (hovRow >= 0 && hovRow < rowCount)
                        hoveredCBRow_ = hovRow;
                    else
                        hoveredCBRow_ = -1;

                    const float typeColStart = cbListX + cbListW - 135.0f;
                    const bool inTypeBand =
                        m.x >= typeColStart && m.x <= cbListX + cbListW - 4.0f;
                    if (input.mousePressed[0] && hoveredCBRow_ >= 0) {
                        if (!(inTypeBand && hoveredCBRow_ == selectedCBRow_)) {
                            selectedCBRow_ = hoveredCBRow_;
                            if (!cbCurriculumRowIsDraft(selectedCBRow_)) {
                                cbDraftPreviewActive_ = false;
                                size_t idx = 0;
                                if (cbCurriculumRowToBlockIndex(selectedCBRow_, idx))
                                    loadConceptBlockIntoEditor(idx);
                            } else {
                                syncCBListTypeDropdownFromToolbar();
                            }
                        }
                    }

                    cbListScrollOffset_ -= input.mouseWheelDelta;
                    float maxScroll = std::max(0.0f,
                        static_cast<float>(rowCount) * kCBListRowH - (cbListH - kPoolHeaderH));
                    cbListScrollOffset_ = std::clamp(cbListScrollOffset_, 0.0f, maxScroll);
                } else if (m.x >= layout.editorX && m.x <= layout.editorX + layout.editorW &&
                           m.y >= cbListY && m.y <= cbListY + cbListH) {
                    // Editor panel scroll
                    cbEditorScrollOffset_ -= input.mouseWheelDelta;
                    hoveredCBRow_ = -1;
                } else {
                    hoveredCBRow_ = -1;
                }
            }
            break;
        }
    }
}

// =========================================================
// Draw
// =========================================================

bool UIDataHubPanel::drawOverlay(OverlayRenderer& renderer) {
    if (!UIPanel::drawOverlay(renderer)) return false;

    // Tab buttons
    float tabX = position.x + 10.0f;
    tabHomeBtn_->setPosition(tabX, position.y + kTabBarY);
    tabSourcesBtn_->setPosition(tabX + 95.0f, position.y + kTabBarY);
    tabHFBtn_->setPosition(tabX + 190.0f, position.y + kTabBarY);
    tabStructBtn_->setPosition(tabX + 305.0f, position.y + kTabBarY);
    tabCurriculumBtn_->setPosition(tabX + 410.0f, position.y + kTabBarY);

    tabHomeBtn_->drawOverlay(renderer, position);
    tabSourcesBtn_->drawOverlay(renderer, position);
    tabHFBtn_->drawOverlay(renderer, position);
    tabStructBtn_->drawOverlay(renderer, position);
    tabCurriculumBtn_->drawOverlay(renderer, position);

    // Active tab indicator (2px underline)
    float indicatorX = tabX;
    float indicatorW = 90.0f;
    switch (activeView_) {
        case DataHubView::Home:        indicatorX = tabX;           indicatorW = 90.0f;  break;
        case DataHubView::Sources:     indicatorX = tabX + 95.0f;   indicatorW = 90.0f;  break;
        case DataHubView::HuggingFace: indicatorX = tabX + 190.0f;  indicatorW = 110.0f; break;
        case DataHubView::Structurer:  indicatorX = tabX + 305.0f;  indicatorW = 100.0f; break;
        case DataHubView::Curriculum:  indicatorX = tabX + 410.0f;  indicatorW = 100.0f; break;
    }
    renderer.drawRect({indicatorX, position.y + kTabBarY + 28.0f}, {indicatorW, 2.0f},
                      UITheme::Colors::Primary);

    // Tab content
    PanelRect content = getContentRect();
    content.origin.y += (kContentTopY - kTabBarY);
    content.size.y   -= (kContentTopY - kTabBarY);

    switch (activeView_) {
        case DataHubView::Home:        drawHomeTab(renderer, content);        break;
        case DataHubView::Sources:     drawSourcesTab(renderer, content);     break;
        case DataHubView::HuggingFace: drawHuggingFaceTab(renderer, content); break;
        case DataHubView::Structurer:  drawStructurerTab(renderer, content);  break;
        case DataHubView::Curriculum:  drawCurriculumTab(renderer, content);  break;
    }

    // Dropdowns draw on top of everything
    if (hfCategoryDropdown_ && hfCategoryDropdown_->isExpanded())
        hfCategoryDropdown_->drawExpandedList(renderer, position);
    if (modelDropdown_ && modelDropdown_->isExpanded())
        modelDropdown_->drawExpandedList(renderer, position);
    if (formatDropdown_ && formatDropdown_->isExpanded())
        formatDropdown_->drawExpandedList(renderer, position);
    if (viewModeDropdown_ && viewModeDropdown_->isExpanded())
        viewModeDropdown_->drawExpandedList(renderer, position);
    for (const auto& card : sequenceCards_) {
        if (card.formatDropdown && card.formatDropdown->isExpanded())
            card.formatDropdown->drawExpandedList(renderer, position);
    }
    if (cbModelDropdown_ && cbModelDropdown_->isExpanded())
        cbModelDropdown_->drawExpandedList(renderer, position);
    if (cbListTypeDropdown_ && cbListTypeDropdown_->isExpanded())
        cbListTypeDropdown_->drawExpandedList(renderer, position);
    if (cbTypeFilterDropdown_ && cbTypeFilterDropdown_->isExpanded())
        cbTypeFilterDropdown_->drawExpandedList(renderer, position);
    if (cbCurriculumDropdown_ && cbCurriculumDropdown_->isExpanded())
        cbCurriculumDropdown_->drawExpandedList(renderer, position);
    if (curriculumActionMenu_ && curriculumActionMenu_->isExpanded())
        curriculumActionMenu_->drawExpandedList(renderer, position);
    if (blockCurriculumMenu_ && blockCurriculumMenu_->isExpanded())
        blockCurriculumMenu_->drawExpandedList(renderer, position);
    if (structureActionMenu_ && structureActionMenu_->isExpanded())
        structureActionMenu_->drawExpandedList(renderer, position);
    if (datasetActionMenu_ && datasetActionMenu_->isExpanded())
        datasetActionMenu_->drawExpandedList(renderer, position);
    if (queueActionMenu_ && queueActionMenu_->isExpanded())
        queueActionMenu_->drawExpandedList(renderer, position);
    if (blockActionMenu_ && blockActionMenu_->isExpanded())
        blockActionMenu_->drawExpandedList(renderer, position);
    if (execStepActionMenu_ && execStepActionMenu_->isExpanded())
        execStepActionMenu_->drawExpandedList(renderer, position);
    if (stepActionMenu_ && stepActionMenu_->isExpanded())
        stepActionMenu_->drawExpandedList(renderer, position);
    for (const auto& row : cbExecStepRows_) {
        if (row.opDropdown && row.opDropdown->isExpanded())
            row.opDropdown->drawExpandedList(renderer, position);
    }
    if (currListActionMenu_ && currListActionMenu_->isExpanded())
        currListActionMenu_->drawExpandedList(renderer, position);

    renderer.popClipRect();
    return true;
}

// =========================================================
// Home tab layout
// =========================================================

void UIDataHubPanel::drawHomeTab(OverlayRenderer& renderer, const PanelRect& content) {
    static constexpr float kPi          = 3.14159265f;
    static constexpr float kGaugeRadius = 88.0f;
    static constexpr float kArcThick    = 10.0f;
    static constexpr int   kArcSegs     = 64;
    static constexpr int   kTileCount   = 6;
    static constexpr float kTileH       = 58.0f;
    static constexpr float kTileGap     = 10.0f;
    static constexpr float kBtnW        = 130.0f;
    static constexpr float kBtnH        = 32.0f;
    static constexpr float kBtnGap      = 8.0f;
    static constexpr int   kBtnCount    = 8;

    float x     = content.origin.x + 15.0f;
    float y     = content.origin.y + 10.0f;
    float fullW = content.size.x - 30.0f;

    // ── Status banner ───────────────────────────────────

    {
        uint32_t dotColor = collectionActive_ ? UITheme::Colors::Success
                          : collectionCompleted_ ? UITheme::Colors::Info
                          : UITheme::Colors::TextDisabled;
        renderer.drawRoundedRect({x, y + 3.0f}, {8.0f, 8.0f}, dotColor, 4.0f);

        std::string statusText = collectionActive_    ? "ACTIVE"
                               : collectionCompleted_ ? "COMPLETE"
                               : "IDLE";
        uint32_t statusColor = collectionActive_    ? UITheme::Colors::Success
                             : collectionCompleted_ ? UITheme::Colors::Info
                             : UITheme::Colors::TextSecondary;
        renderer.drawText({x + 14.0f, y}, statusText, statusColor);

        if (!currentPhase_.empty() && currentPhase_ != "idle")
            renderer.drawText({x + 90.0f, y}, currentPhase_, UITheme::Colors::TextMuted);

        if (!collectionMessage_.empty()) {
            float msgW = UIDrawHelpers::getTextWidth(collectionMessage_);
            renderer.drawText({x + fullW - msgW, y}, collectionMessage_,
                              UITheme::Colors::TextSecondary);
        }
    }
    y += 22.0f;
    renderer.drawRect({x, y}, {fullW, 1.0f}, UITheme::Colors::DividerLine);
    y += 8.0f;

    // ── Half-circle progress gauge ──────────────────────

    {
        float gaugeCX = x + fullW / 2.0f;
        float gaugeCY = y + kGaugeRadius + 5.0f;
        float progress = std::clamp(currentProgress_ / 100.0f, 0.0f, 1.0f);

        uint32_t arcFillColor = collectionActive_    ? UITheme::Colors::Info
                              : collectionCompleted_ ? UITheme::Colors::Success
                              : UITheme::Colors::Primary;

        for (int i = 0; i < kArcSegs; i++) {
            float a1 = kPi - (kPi * static_cast<float>(i)     / kArcSegs);
            float a2 = kPi - (kPi * static_cast<float>(i + 1) / kArcSegs);
            Vec2 p1 = {gaugeCX + kGaugeRadius * std::cos(a1),
                       gaugeCY - kGaugeRadius * std::sin(a1)};
            Vec2 p2 = {gaugeCX + kGaugeRadius * std::cos(a2),
                       gaugeCY - kGaugeRadius * std::sin(a2)};
            renderer.drawLine(p1, p2, UITheme::Colors::SliderTrack, kArcThick);
        }

        if (progress > 0.005f) {
            int filledSegs = static_cast<int>(kArcSegs * progress);

            uint32_t glowColor = (arcFillColor & 0x00FFFFFF) | 0x28000000;
            for (int i = 0; i < filledSegs; i++) {
                float a1 = kPi - (kPi * static_cast<float>(i)     / kArcSegs);
                float a2 = kPi - (kPi * static_cast<float>(i + 1) / kArcSegs);
                Vec2 p1 = {gaugeCX + kGaugeRadius * std::cos(a1),
                           gaugeCY - kGaugeRadius * std::sin(a1)};
                Vec2 p2 = {gaugeCX + kGaugeRadius * std::cos(a2),
                           gaugeCY - kGaugeRadius * std::sin(a2)};
                renderer.drawLine(p1, p2, glowColor, kArcThick + 8.0f);
            }

            for (int i = 0; i < filledSegs; i++) {
                float a1 = kPi - (kPi * static_cast<float>(i)     / kArcSegs);
                float a2 = kPi - (kPi * static_cast<float>(i + 1) / kArcSegs);
                Vec2 p1 = {gaugeCX + kGaugeRadius * std::cos(a1),
                           gaugeCY - kGaugeRadius * std::sin(a1)};
                Vec2 p2 = {gaugeCX + kGaugeRadius * std::cos(a2),
                           gaugeCY - kGaugeRadius * std::sin(a2)};
                renderer.drawLine(p1, p2, arcFillColor, kArcThick);
            }
        }

        for (int tick = 0; tick <= 4; tick++) {
            float frac  = tick / 4.0f;
            float angle = kPi - (kPi * frac);
            float rIn   = kGaugeRadius - kArcThick / 2.0f - 2.0f;
            float rOut  = kGaugeRadius - kArcThick / 2.0f - 8.0f;
            Vec2 tp1 = {gaugeCX + rIn  * std::cos(angle), gaugeCY - rIn  * std::sin(angle)};
            Vec2 tp2 = {gaugeCX + rOut * std::cos(angle), gaugeCY - rOut * std::sin(angle)};
            renderer.drawLine(tp1, tp2, UITheme::Colors::BorderSubtle, 1.0f);
        }

        int pctVal = static_cast<int>(currentProgress_);
        std::string pctText = std::to_string(pctVal) + "%";
        float pctW = UIDrawHelpers::getTextWidth(pctText, 14.0f);
        renderer.drawText({gaugeCX - pctW / 2.0f, gaugeCY - kGaugeRadius * 0.48f - 8.0f},
                          pctText, UITheme::Colors::TextPrimary);

        std::string phaseLabel = collectionActive_ ? currentPhase_ : "Pipeline Progress";
        float labelW = UIDrawHelpers::getTextWidth(phaseLabel, 14.0f);
        renderer.drawText({gaugeCX - labelW / 2.0f, gaugeCY - kGaugeRadius * 0.48f + 10.0f},
                          phaseLabel, UITheme::Colors::TextSecondary);

        y = gaugeCY + 15.0f;
    }

    // ── HUD stat tiles ──────────────────────────────────

    {
        float tileW = (fullW - (kTileCount - 1) * kTileGap) / kTileCount;

        std::string elapsedStr = "--";
        if (elapsedSeconds_ > 0) {
            int hrs  = static_cast<int>(elapsedSeconds_ / 3600);
            int mins = static_cast<int>((elapsedSeconds_ % 3600) / 60);
            int secs = static_cast<int>(elapsedSeconds_ % 60);
            std::ostringstream oss;
            if (hrs > 0)
                oss << hrs << ":"
                    << std::setfill('0') << std::setw(2) << mins << ":"
                    << std::setw(2) << secs;
            else
                oss << std::setfill('0') << std::setw(2) << mins << ":"
                    << std::setw(2) << secs;
            elapsedStr = oss.str();
        }

        struct Tile { const char* label; std::string value; uint32_t accent; };
        Tile tiles[kTileCount] = {
            {"DATABASE",    hudFileSize_,
             UITheme::Colors::Info},
            {"SEQUENCES",   std::to_string(totalSequences_),
             UITheme::Colors::Primary},
            {"SOURCES",     std::to_string(totalSources_) + " / "
                            + std::to_string(sourceCards_.size()),
             UITheme::Colors::Success},
            {"CHECKPOINTS", std::to_string(hudCheckpoints_),
             UITheme::Colors::Warning},
            {"ENTRIES",     std::to_string(entriesCollected_),
             UITheme::Colors::PrimaryLight},
            {"ELAPSED",     elapsedStr,
             UITheme::Colors::AccentBlue},
        };

        for (int i = 0; i < kTileCount; i++) {
            float tx = x + i * (tileW + kTileGap);

            renderer.drawRoundedRect({tx, y}, {tileW, kTileH},
                                     UITheme::Colors::CardSurface,
                                     UITheme::Sizes::SmallRadius);
            renderer.drawRoundedBorder({tx, y}, {tileW, kTileH},
                                       UITheme::Colors::BorderSubtle,
                                       UITheme::Sizes::SmallRadius);

            renderer.drawRect({tx + 8.0f, y + 4.0f},
                              {tileW - 16.0f, 2.0f}, tiles[i].accent);

            renderer.drawText({tx + 10.0f, y + 12.0f},
                              tiles[i].label, UITheme::Colors::TextSecondary);
            renderer.drawText({tx + 10.0f, y + 32.0f},
                              tiles[i].value, UITheme::Colors::TextPrimary);
        }

        y += kTileH + 15.0f;
    }

    // ── Action buttons (centered horizontal row) ────────

    {
        float totalBtnW = kBtnCount * kBtnW + (kBtnCount - 1) * kBtnGap;
        float startX    = x + (fullW - totalBtnW) / 2.0f;

        std::shared_ptr<UIButton>* btns[kBtnCount] = {
            &btnFull_, &btnCollect_, &btnVerify_, &btnMerge_,
            &btnRebuild_, &btnStop_, &btnRefreshStats_, &btnCollectDir_
        };

        for (int i = 0; i < kBtnCount; i++) {
            float bx = startX + i * (kBtnW + kBtnGap);
            (*btns[i])->setPosition(bx, y);
            (*btns[i])->setSize(kBtnW, kBtnH);
            (*btns[i])->drawOverlay(renderer, position);
        }

        y += kBtnH + 12.0f;
    }

    // ── Directory collection ────────────────────────────

    {
        UIDrawHelpers::drawSectionHeader(renderer, {x, y}, fullW, "Directory Collection",
                                         UITheme::Colors::SectionNeutral);
        y += UITheme::Sizes::HeaderHeight;

        // Directory path input
        dirPathInput_->setPosition(x, y);
        dirPathInput_->setSize({fullW, 28.0f});
        dirPathInput_->drawOverlay(renderer, position);
        y += 34.0f;

        // Rescan when path changes
        std::string currentPath = dirPathInput_->getText();
        if (currentPath != dirScanPath_) {
            dirScanPath_ = currentPath;
            dirNeedsScan_ = true;
        }
        if (dirNeedsScan_ && !dirScanPath_.empty()) {
            dirNeedsScan_ = false;
            scanDirectory();
        }

        // File scrollbox
        static constexpr float kDirRowH = 28.0f;
        size_t fileCount = dirFileEntries_.size();
        float scrollH = std::min(160.0f, content.origin.y + content.size.y - y - 200.0f);
        if (scrollH < 60.0f) scrollH = 60.0f;

        renderer.drawRoundedRect({x, y}, {fullW, scrollH},
                                 UITheme::Colors::ContentAreaBg, UITheme::Sizes::WidgetRadius);
        renderer.drawRoundedBorder({x, y}, {fullW, scrollH},
                                   UITheme::Colors::BorderSubtle, UITheme::Sizes::WidgetRadius);

        if (fileCount == 0) {
            std::string emptyMsg = dirScanPath_.empty()
                ? "Enter a directory path above"
                : "No files found";
            float msgW = UIDrawHelpers::getTextWidth(emptyMsg);
            renderer.drawText({x + (fullW - msgW) / 2.0f, y + scrollH / 2.0f - 6.0f},
                              emptyMsg, UITheme::Colors::TextDisabled);
        } else {
            renderer.pushClipRect({x, y}, {fullW, scrollH});

            float colCheckX   = x + 8.0f;
            float colNameX    = x + 38.0f;
            float colToggleX  = x + fullW - 90.0f;
            float headerRowY  = y + 2.0f;
            renderer.drawText({colCheckX, headerRowY}, "Collect", UITheme::Colors::TextMuted);
            renderer.drawText({colNameX,  headerRowY}, "File", UITheme::Colors::TextMuted);
            renderer.drawText({colToggleX, headerRowY}, "Del After", UITheme::Colors::TextMuted);

            float listTop = y + 18.0f;
            float totalH  = fileCount * kDirRowH;
            float maxScroll = std::max(0.0f, totalH - (scrollH - 18.0f));
            dirScrollOffset_ = std::clamp(dirScrollOffset_, 0.0f, maxScroll);

            bool clicked = dirClickPending_;
            Vec2 clickM  = dirClickPos_;
            if (clicked) dirClickPending_ = false;  // consume once

            for (size_t i = 0; i < fileCount; ++i) {
                float rowY = listTop + i * kDirRowH - dirScrollOffset_;
                if (rowY + kDirRowH < listTop || rowY > y + scrollH) continue;

                auto& entry = dirFileEntries_[i];

                // Alternating row background
                if (i % 2 == 0) {
                    renderer.drawRect({x + 2.0f, rowY}, {fullW - 4.0f, kDirRowH},
                                      (UITheme::Colors::CardSurface & 0x00FFFFFF) | 0x18000000);
                }

                // Collect checkbox
                float cbX = colCheckX;
                float cbY = rowY + 4.0f;
                float cbSz = 18.0f;
                renderer.drawRoundedRect({cbX, cbY}, {cbSz, cbSz},
                                         UITheme::Colors::Background, 3.0f);
                renderer.drawRoundedBorder({cbX, cbY}, {cbSz, cbSz},
                                           UITheme::Colors::BorderSubtle, 3.0f);
                if (entry.collect) {
                    renderer.drawRoundedRect({cbX + 3.0f, cbY + 3.0f},
                                             {cbSz - 6.0f, cbSz - 6.0f},
                                             UITheme::Colors::Primary, 2.0f);
                }

                // Click on checkbox
                if (clicked &&
                    clickM.x >= cbX && clickM.x <= cbX + cbSz &&
                    clickM.y >= cbY && clickM.y <= cbY + cbSz) {
                    entry.collect = !entry.collect;
                }

                // File name
                renderer.drawText({colNameX, rowY + 6.0f}, entry.filename,
                                  UITheme::Colors::TextPrimary);

                // Delete-after toggle (pill shape)
                float tgX = colToggleX + 10.0f;
                float tgY = rowY + 5.0f;
                float tgW = 36.0f;
                float tgH = 18.0f;
                uint32_t tgBg = entry.deleteAfter ? UITheme::Colors::Danger
                                                  : UITheme::Colors::SliderTrack;
                renderer.drawRoundedRect({tgX, tgY}, {tgW, tgH}, tgBg, tgH / 2.0f);
                float knobX = entry.deleteAfter ? tgX + tgW - tgH + 2.0f : tgX + 2.0f;
                renderer.drawRoundedRect({knobX, tgY + 2.0f},
                                         {tgH - 4.0f, tgH - 4.0f},
                                         UITheme::Colors::TextPrimary,
                                         (tgH - 4.0f) / 2.0f);

                // Click on toggle
                if (clicked &&
                    clickM.x >= tgX && clickM.x <= tgX + tgW &&
                    clickM.y >= tgY && clickM.y <= tgY + tgH) {
                    entry.deleteAfter = !entry.deleteAfter;
                }
            }

            renderer.popClipRect();
        }

        // Cache scroll area rect for update() scroll handling
        dirScrollAreaRect_ = {x, y, fullW, scrollH};

        y += scrollH + 8.0f;
    }

    // ── Collection log viewer ───────────────────────────

    {
        UIDrawHelpers::drawSectionHeader(renderer, {x, y}, fullW, "Collection Logs",
                                         UITheme::Colors::SectionNeutral);
        y += UITheme::Sizes::HeaderHeight;

        float logH = content.origin.y + content.size.y - y - 10.0f;
        if (logH < 40.0f) logH = 40.0f;

        renderer.drawRoundedRect({x, y}, {fullW, logH},
                                 UITheme::Colors::Background, UITheme::Sizes::WidgetRadius);

        std::lock_guard<std::mutex> lock(logMutex_);
        float entryY = y + 5.0f;
        int visible  = static_cast<int>(logH / 18.0f);
        int start    = std::max(0, static_cast<int>(logEntries_.size()) - visible);
        for (size_t i = static_cast<size_t>(start); i < logEntries_.size(); ++i) {
            const auto& e = logEntries_[i];
            uint32_t color = UITheme::Colors::Success;
            if (e.level == 1) color = UITheme::Colors::Warning;
            else if (e.level == 2) color = UITheme::Colors::Danger;
            renderer.drawText({x + 5.0f, entryY}, e.timestamp + " " + e.message, color);
            entryY += 18.0f;
            if (entryY > y + logH) break;
        }
    }
}

// =========================================================
// Sources tab layout
// =========================================================

void UIDataHubPanel::drawSourcesTab(OverlayRenderer& renderer, const PanelRect& content) {
    float x = content.origin.x + 15.0f;
    float fullW = content.size.x - 30.0f;
    float scrollAreaTop = content.origin.y + 5.0f;
    float scrollAreaH   = content.size.y - 60.0f;

    // Count enabled sources for header
    size_t enabledCount = 0;
    for (const auto& c : sourceCards_) { if (c.enabled) enabledCount++; }
    std::string header = "Data Sources (" + std::to_string(enabledCount)
                       + "/" + std::to_string(sourceCards_.size()) + ")";
    renderer.drawText({x + 2.0f, scrollAreaTop - 1.0f}, header,
                      UITheme::Colors::TextSecondary);

    // Scroll area background
    float cardAreaTop = scrollAreaTop + 16.0f;
    float cardAreaH   = scrollAreaH - 16.0f;
    renderer.drawRoundedRect({x, cardAreaTop}, {fullW, cardAreaH},
                             UITheme::Colors::ContentAreaBg, UITheme::Sizes::WidgetRadius);
    renderer.drawRoundedBorder({x, cardAreaTop}, {fullW, cardAreaH},
                               UITheme::Colors::BorderSubtle, UITheme::Sizes::WidgetRadius);

    // Clip to card area
    renderer.pushClipRect({x, cardAreaTop}, {fullW, cardAreaH});

    float cardY = cardAreaTop + 8.0f - sourcesScrollOffset_;

    for (size_t i = 0; i < sourceCards_.size(); i++) {
        auto& card = sourceCards_[i];
        float cardX = x + 8.0f;
        float cardW = fullW - 16.0f;

        if (cardY + kCardHeight < cardAreaTop || cardY > cardAreaTop + cardAreaH) {
            cardY += kCardHeight + kCardGap;
            continue;
        }

        // ── Card background ─────────────────────────
        renderer.drawRoundedRect({cardX, cardY}, {cardW, kCardHeight},
                                 UITheme::Colors::CardSurface, UITheme::Sizes::WidgetRadius);
        renderer.drawRoundedBorder({cardX, cardY}, {cardW, kCardHeight},
                                   UITheme::Colors::BorderSubtle, UITheme::Sizes::WidgetRadius);

        float innerX = cardX + kCardInnerPad;
        float innerW = cardW - 2.0f * kCardInnerPad;
        float rowY   = cardY;

        // ── Top bar: number + enabled toggle + delete ──
        renderer.drawText({innerX, rowY + 7.0f},
                          std::to_string(i + 1),
                          UITheme::Colors::TextMuted);

        float toggleW = 50.0f;
        float delW = 28.0f;
        float topRightX = cardX + cardW - kCardInnerPad;

        card.deleteBtn->setPosition(topRightX - delW, rowY + 2.0f);
        card.deleteBtn->setSize(delW, 24.0f);
        card.deleteBtn->drawOverlay(renderer, position);

        card.enabledToggle->setPosition(topRightX - delW - toggleW - 8.0f, rowY + 2.0f);
        card.enabledToggle->setSize(toggleW, 24.0f);
        card.enabledToggle->drawOverlay(renderer, position);

        // Divider
        rowY += kCardTopBarH;
        renderer.drawRect({innerX, rowY - 1.0f}, {innerW, 1.0f}, UITheme::Colors::DividerLine);

        // ── Row 1: Name | URL ───────────────────────
        float halfW = (innerW - 10.0f) / 2.0f;

        card.nameInput->setPosition(innerX, rowY + 2.0f);
        card.nameInput->setSize(halfW, kCardFieldH);
        card.nameInput->drawOverlay(renderer, position);

        card.urlInput->setPosition(innerX + halfW + 10.0f, rowY + 2.0f);
        card.urlInput->setSize(halfW, kCardFieldH);
        card.urlInput->drawOverlay(renderer, position);

        float labelY = rowY + kCardFieldH + kCardLabelGap;
        renderer.drawText({innerX + 2.0f, labelY}, "NAME",
                          UITheme::Colors::TextSecondary);
        renderer.drawText({innerX + halfW + 12.0f, labelY}, "URL",
                          UITheme::Colors::TextSecondary);

        rowY += kCardRowH;

        // ── Row 2: Priority | Depth | Limit ──
        float col3W = (innerW - 20.0f) / 3.0f;
        float qx = innerX;

        card.prioritySlider->setPosition(qx, rowY + 2.0f);
        card.prioritySlider->setSize(col3W, kCardFieldH);
        card.prioritySlider->drawOverlay(renderer, position);

        qx += col3W + 10.0f;
        card.depthSlider->setPosition(qx, rowY + 2.0f);
        card.depthSlider->setSize(col3W, kCardFieldH);
        card.depthSlider->drawOverlay(renderer, position);

        qx += col3W + 10.0f;
        card.limitSlider->setPosition(qx, rowY + 2.0f);
        card.limitSlider->setSize(col3W, kCardFieldH);
        card.limitSlider->drawOverlay(renderer, position);

        labelY = rowY + kCardFieldH + kCardLabelGap;
        qx = innerX;
        renderer.drawText({qx + 2.0f, labelY}, "PRIORITY",
                          UITheme::Colors::TextSecondary);
        qx += col3W + 10.0f;
        renderer.drawText({qx + 2.0f, labelY}, "CRAWL DEPTH",
                          UITheme::Colors::TextSecondary);
        qx += col3W + 10.0f;
        renderer.drawText({qx + 2.0f, labelY}, "FETCH LIMIT",
                          UITheme::Colors::TextSecondary);

        cardY += kCardHeight + kCardGap;
    }

    if (sourceCards_.empty()) {
        float cx = x + fullW / 2.0f - 120.0f;
        float cy = cardAreaTop + cardAreaH / 2.0f - 10.0f;
        renderer.drawText({cx, cy},
                          "No sources configured. Click '+ Add Source' below.",
                          UITheme::Colors::TextDisabled);
    }

    renderer.popClipRect();

    // Scroll indicator (thin bar on right side)
    float totalH = sourceCards_.size() * (kCardHeight + kCardGap) + 10.0f;
    if (totalH > cardAreaH) {
        float scrollRatio  = sourcesScrollOffset_ / (totalH - cardAreaH);
        float thumbRatio   = cardAreaH / totalH;
        float thumbH       = std::max(20.0f, cardAreaH * thumbRatio);
        float thumbY       = cardAreaTop + scrollRatio * (cardAreaH - thumbH);
        float barX         = x + fullW - 6.0f;
        renderer.drawRoundedRect({barX, thumbY}, {4.0f, thumbH},
                                 UITheme::Colors::ScrollThumb, 2.0f);
    }

    // ── Add Source button (centered below card area) ──
    float addBtnW = 220.0f;
    float addBtnH = 36.0f;
    float addBtnX = x + (fullW - addBtnW) / 2.0f;
    float addBtnY = cardAreaTop + cardAreaH + 10.0f;
    btnAddCard_->setPosition(addBtnX, addBtnY);
    btnAddCard_->setSize(addBtnW, addBtnH);
    btnAddCard_->drawOverlay(renderer, position);
}

// =========================================================
// HuggingFace tab layout
// =========================================================

void UIDataHubPanel::drawHuggingFaceTab(OverlayRenderer& renderer, const PanelRect& content) {
    float x = content.origin.x + 15.0f;
    float y = content.origin.y + 10.0f;
    float fullW = content.size.x - 30.0f;

    // ── Search section ──────────────────────────────────

    UIDrawHelpers::drawSectionHeader(renderer, {x, y}, fullW, "HuggingFace Datasets",
                                     UITheme::Colors::SectionNeutral);
    y += UITheme::Sizes::HeaderHeight + 5.0f;

    // Token input
    hfTokenInput_->setPosition(x, y);
    hfTokenInput_->setSize(fullW, 25.0f);
    hfTokenInput_->drawOverlay(renderer, position);
    y += 30.0f;

    // Search bar + buttons
    float searchW = fullW * 0.55f;
    hfSearchInput_->setPosition(x, y);
    hfSearchInput_->setSize(searchW, 28.0f);
    hfSearchInput_->drawOverlay(renderer, position);

    float btnX = x + searchW + 10.0f;
    float sbtnW = 90.0f;
    btnSearchHF_->setPosition(btnX, y);
    btnSearchHF_->setSize(sbtnW, 28.0f);
    btnSearchHF_->drawOverlay(renderer, position);

    btnBrowseHF_->setPosition(btnX + sbtnW + 5.0f, y);
    btnBrowseHF_->setSize(120.0f, 28.0f);
    btnBrowseHF_->drawOverlay(renderer, position);
    y += 35.0f;

    // Category dropdown + max results slider
    float halfW = (fullW - 10.0f) / 2.0f;
    hfCategoryDropdown_->setPosition(x, y);
    hfCategoryDropdown_->setSize(halfW, 30.0f);
    hfCategoryDropdown_->drawOverlay(renderer, position);

    sliderMaxHFResults_->setPosition(x + halfW + 10.0f, y);
    sliderMaxHFResults_->setSize(halfW, 30.0f);
    sliderMaxHFResults_->drawOverlay(renderer, position);
    y += 40.0f;

    // ── Results section ─────────────────────────────────

    float resultsH = 0.0f;
    if (!hfSearchResults_.empty() && hfResultsScrollBox_) {
        UIDrawHelpers::drawSectionHeader(renderer, {x, y}, fullW, "Search Results",
                                         UITheme::Colors::SectionNeutral);
        y += UITheme::Sizes::HeaderHeight;

        if (hfResultsNeedsPopulate_.load() || hfResultsScrollBox_->getChildren().empty()) {
            populateHFResults(fullW);
            hfResultsNeedsPopulate_.store(false);
        }
        resultsH = std::min(140.0f, static_cast<float>(maxHFResults_) * 68.0f);
        hfResultsScrollBox_->setPosition(x, y);
        hfResultsScrollBox_->setSize(fullW, resultsH);
        hfResultsScrollBox_->drawOverlay(renderer, position);
        y += resultsH + 10.0f;
    } else if (hfSearching_.load()) {
        int dots = static_cast<int>(searchAnimTime_ * 3.0f) % 4;
        std::string loadingText = "Searching HuggingFace" + std::string(static_cast<size_t>(dots), '.');
        renderer.drawText({x, y}, loadingText, UITheme::Colors::Warning);
        y += 25.0f;
    } else if (!lastSearchError_.empty()) {
        renderer.drawText({x, y}, lastSearchError_, UITheme::Colors::Danger);
        y += 25.0f;
    }

    // ── Dataset preview (datasets-server /first-rows, same source as huggingface.co viewer) ──

    UIDrawHelpers::drawSectionHeader(renderer, {x, y}, fullW, "Dataset preview",
                                     UITheme::Colors::SectionNeutral);
    y += UITheme::Sizes::HeaderHeight;

    constexpr float kHfPreviewH = 128.0f;
    if (hfPreviewArea_) {
        hfPreviewArea_->setPosition(x, y);
        hfPreviewArea_->setSize(fullW, kHfPreviewH);
        hfPreviewArea_->drawOverlay(renderer, position);
    }
    y += kHfPreviewH + 8.0f;

    if (btnHFQueuePreview_) {
        btnHFQueuePreview_->setPosition(x, y);
        btnHFQueuePreview_->setSize(280.0f, 28.0f);
        btnHFQueuePreview_->drawOverlay(renderer, position);
    }
    y += 34.0f;

    renderer.drawText({x, y},
                      "Tip: click a result to preview; use the button above to queue (not one-click add).",
                      UITheme::Colors::TextDisabled);
    y += 18.0f;

    // ── Download queue ──────────────────────────────────

    drawQueueSection(renderer, x, y, fullW);
}

// =========================================================
// Queue section (drawn inside HF tab)
// =========================================================

void UIDataHubPanel::drawQueueSection(OverlayRenderer& renderer,
                                      float x, float& y, float width) {
    size_t queueCount = 0, pendingCount = 0, completedCount = 0, failedCount = 0;
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        queueCount = downloadQueue_.size();
        for (const auto& item : downloadQueue_) {
            if (item.status == "pending")    pendingCount++;
            else if (item.status == "completed") completedCount++;
            else if (item.status == "failed")    failedCount++;
        }
    }

    std::string qHeader = "Download Queue";
    if (queueCount > 0) qHeader += " (" + std::to_string(queueCount) + ")";
    if (queueProcessing_.load()) qHeader += " [PROCESSING]";

    UIDrawHelpers::drawSectionHeader(renderer, {x, y}, width, qHeader,
                                     UITheme::Colors::SectionNeutral);
    y += UITheme::Sizes::HeaderHeight;

    if (queueCount > 0) {
        std::string stats;
        if (pendingCount > 0)   stats += std::to_string(pendingCount) + " pending  ";
        if (completedCount > 0) stats += std::to_string(completedCount) + " done  ";
        if (failedCount > 0)    stats += std::to_string(failedCount) + " failed";
        if (!stats.empty())
            renderer.drawText({x + width - 200.0f, y}, stats, UITheme::Colors::TextDisabled);
    }

    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        if (!downloadQueue_.empty()) {
            float itemH = UITheme::Sizes::WidgetHeight;
            float boxH  = std::min(200.0f, downloadQueue_.size() * itemH + 10.0f);
            renderer.drawRoundedRect({x, y}, {width, boxH},
                                     UITheme::Colors::ScrollboxBg,
                                     UITheme::Sizes::WidgetRadius);

            float itemY = y + 5.0f;
            for (size_t i = 0; i < downloadQueue_.size() && itemY + itemH <= y + boxH; ++i) {
                const auto& item = downloadQueue_[i];
                bool isHov = (hoveredQueueItem_ == static_cast<int>(i));
                bool isProc = (currentQueueIndex_.load() == static_cast<int>(i));

                UIDrawHelpers::drawWidgetBackground(renderer,
                    {x + 2.0f, itemY}, {width - 4.0f, itemH - 4.0f}, isHov, isProc, false);

                uint32_t sColor = UITheme::Colors::TextDisabled;
                if      (item.status == "downloading") sColor = UITheme::Colors::Info;
                else if (item.status == "completed")   sColor = UITheme::Colors::Success;
                else if (item.status == "failed")      sColor = UITheme::Colors::Danger;
                else if (item.status == "pending")     sColor = UITheme::Colors::Warning;

                UIDrawHelpers::drawCategoryIndicator(renderer, {x + 2.0f, itemY}, itemH - 4.0f, sColor);

                std::string name = item.displayName;
                if (name.length() > 40) name = name.substr(0, 37) + "...";
                renderer.drawText({x + 12.0f, itemY + 3.0f}, name, UITheme::Colors::TextPrimary);

                std::string statusLine;
                if (item.status == "downloading") {
                    int pct = static_cast<int>(item.progress * 100.0f);
                    statusLine = "Downloading... " + std::to_string(pct) + "%";
                    float barW = width - 80.0f;
                    float barH = 4.0f;
                    float barY = itemY + itemH - 10.0f;
                    renderer.drawRoundedRect({x + 12.0f, barY}, {barW, barH},
                                             UITheme::Colors::SliderTrack,
                                             UITheme::Sizes::SmallRadius);
                    renderer.drawRoundedRect({x + 12.0f, barY}, {barW * item.progress, barH},
                                             UITheme::Colors::Info,
                                             UITheme::Sizes::SmallRadius);
                } else if (item.status == "completed") {
                    statusLine = "Download complete";
                } else if (item.status == "failed") {
                    statusLine = item.errorMessage.empty() ? "Failed" : item.errorMessage;
                    if (statusLine.length() > 40) statusLine = statusLine.substr(0, 37) + "...";
                } else if (item.status == "pending") {
                    statusLine = item.retryCount > 0
                        ? "Pending (retry #" + std::to_string(item.retryCount) + ")"
                        : "Waiting in queue...";
                }
                renderer.drawText({x + 12.0f, itemY + 18.0f}, statusLine, sColor);

                if (isHov && item.status != "downloading") {
                    float removeX = x + width - 25.0f;
                    renderer.drawRoundedRect({removeX, itemY + 8.0f}, {18.0f, 18.0f},
                                             UITheme::Colors::Danger & 0x44FFFFFF,
                                             UITheme::Sizes::SmallRadius);
                    renderer.drawText({removeX + 4.0f, itemY + 8.0f}, "X", UITheme::Colors::Danger);
                }

                itemY += itemH;
            }
            y += boxH + 5.0f;
        } else {
            renderer.drawText({x + 5.0f, y}, "Queue empty — search above to add datasets",
                              UITheme::Colors::TextDisabled);
            y += 20.0f;
        }
    }
    y += 10.0f;

    // Queue control menu
    float qBtnW = 120.0f;
    float qBtnH = UITheme::Sizes::ButtonHeight;

    queueActionMenu_->setItemLabel(0, queueProcessing_.load() ? "Processing..." : "Process Queue");
    queueActionMenu_->setTitle(queueProcessing_.load() ? "Processing..." : "Queue");
    queueActionMenu_->setPosition(x, y);
    queueActionMenu_->setSize(qBtnW, qBtnH);
    queueActionMenu_->drawOverlay(renderer, position);

    y += qBtnH + 10.0f;
}

// =========================================================
// Structurer tab layout
// =========================================================

void UIDataHubPanel::drawStructurerTab(OverlayRenderer& renderer, const PanelRect& content) {
    float x = content.origin.x + 15.0f;
    float y = content.origin.y + 10.0f;
    float fullW = content.size.x - 30.0f;

    // ── Toolbar row 1 ───────────────────────────────────
    float ddW = 260.0f;
    float sliderW = 280.0f;
    float gap = 14.0f;
    float rowH = 36.0f;
    float genW = 90.0f;

    float cx = x;
    modelDropdown_->setPosition(cx, y);
    modelDropdown_->setSize(ddW, rowH);
    modelDropdown_->drawOverlay(renderer, position);
    cx += ddW + gap;

    formatDropdown_->setPosition(cx, y);
    formatDropdown_->setSize(ddW, rowH);
    formatDropdown_->drawOverlay(renderer, position);
    cx += ddW + 8.0f;

    btnGenerate_->setPosition(cx, y);
    btnGenerate_->setSize(genW, rowH);
    btnGenerate_->drawOverlay(renderer, position);
    cx += genW + gap;

    viewModeDropdown_->setPosition(cx, y);
    viewModeDropdown_->setSize(ddW, rowH);
    viewModeDropdown_->drawOverlay(renderer, position);

    float rightX = x + fullW;
    float menuW = 110.0f;
    float btnGap = 8.0f;

    structureActionMenu_->setPosition(rightX - menuW, y);
    structureActionMenu_->setSize(menuW, 28.0f);
    structureActionMenu_->drawOverlay(renderer, position);
    y += rowH + kStructRowGap;

    // ── Toolbar row 2 ───────────────────────────────────

    sliderMaxEntries_->setPosition(x, y);
    sliderMaxEntries_->setSize(sliderW, rowH);
    sliderMaxEntries_->drawOverlay(renderer, position);

    sliderParallel_->setPosition(x + sliderW + gap, y);
    sliderParallel_->setSize(sliderW, rowH);
    sliderParallel_->drawOverlay(renderer, position);

    datasetActionMenu_->setPosition(rightX - menuW, y);
    datasetActionMenu_->setSize(menuW, 28.0f);
    datasetActionMenu_->drawOverlay(renderer, position);
    y += rowH + kStructRowGap;

    // ── Search bar (Dataset View) or header (Sequence View) ──

    float searchH = 28.0f;

    if (structViewMode_ == 0) {
        structSearchInput_->setPosition(x, y);
        structSearchInput_->setSize(fullW * 0.4f, searchH);
        structSearchInput_->drawOverlay(renderer, position);

        float navX = x + fullW * 0.4f + 20.0f;
        btnPrevSeq_->setPosition(navX, y);
        btnPrevSeq_->setSize(35.0f, searchH);
        btnPrevSeq_->drawOverlay(renderer, position);

        std::string seqLabel = std::to_string(currentSequenceIndex_ + 1)
                             + " / " + std::to_string(totalSequences_);
        renderer.drawText({navX + 42.0f, y + 6.0f}, seqLabel, UITheme::Colors::TextPrimary);

        btnNextSeq_->setPosition(navX + 130.0f, y);
        btnNextSeq_->setSize(35.0f, searchH);
        btnNextSeq_->drawOverlay(renderer, position);
        y += searchH + kStructSectionGap;
    } else if (structViewMode_ == 1) {
        size_t enabledCards = sequenceCards_.size();
        std::string header = "Sequences (" + std::to_string(enabledCards) + ")";
        renderer.drawText({x + 2.0f, y + 6.0f}, header, UITheme::Colors::TextSecondary);
        y += searchH + kStructSectionGap;
    }

    UIDrawHelpers::drawDivider(renderer, {x, y}, fullW);
    y += kStructLabelSpace;

    // ── Bottom-anchored layout ──────────────────────────
    // Status bar + custom prompt section are anchored to bottom

    float statusY     = content.origin.y + content.size.y - 30.0f;
    float promptTextH = 50.0f;
    float promptLabelH = 16.0f;
    float promptGap    = 6.0f;
    float promptTopY   = statusY - promptTextH - promptGap;
    float promptLabelY = promptTopY - promptLabelH - 2.0f;
    float contentEndY  = promptLabelY - 8.0f;

    // ── Content split: Dataset View / Sequence View / Curriculum View ──

    if (structViewMode_ == 2) {
        drawCurriculumView(renderer, content, x, y, fullW, contentEndY);
        return;
    }

    if (structViewMode_ == 0) {
        // ── Dual text areas ─────────────────────────────
        float areaH = contentEndY - y;
        if (areaH < 100.0f) areaH = 100.0f;
        float halfW = (fullW - 10.0f) / 2.0f;

        rawTextArea_->setPosition(x, y);
        rawTextArea_->setSize(halfW, areaH);
        rawTextArea_->drawOverlay(renderer, position);

        structuredTextArea_->setPosition(x + halfW + 10.0f, y);
        structuredTextArea_->setSize(halfW, areaH);
        structuredTextArea_->drawOverlay(renderer, position);
        y += areaH + 5.0f;
    } else if (structViewMode_ == 1) {
        // ── Sequence cards (scrollable) ─────────────────
        float cardAreaH = contentEndY - y - 46.0f;
        if (cardAreaH < 120.0f) cardAreaH = 120.0f;

        seqCardAreaTop_ = y;
        seqCardAreaH_   = cardAreaH;

        renderer.drawRoundedRect({x, y}, {fullW, cardAreaH},
                                 UITheme::Colors::ContentAreaBg, UITheme::Sizes::WidgetRadius);
        renderer.drawRoundedBorder({x, y}, {fullW, cardAreaH},
                                   UITheme::Colors::BorderSubtle, UITheme::Sizes::WidgetRadius);

        renderer.pushClipRect({x, y}, {fullW, cardAreaH});

        float cardY = y + 8.0f - seqScrollOffset_;

        for (size_t ci = 0; ci < sequenceCards_.size(); ++ci) {
            auto& card = sequenceCards_[ci];
            float cardH = sequenceCardHeight(card);
            float cardX = x + 8.0f;
            float cardW = fullW - 16.0f;

            if (cardY + cardH < y || cardY > y + cardAreaH) {
                cardY += cardH + kCardGap;
                continue;
            }

            // Card background
            renderer.drawRoundedRect({cardX, cardY}, {cardW, cardH},
                                     UITheme::Colors::CardSurface, UITheme::Sizes::WidgetRadius);
            renderer.drawRoundedBorder({cardX, cardY}, {cardW, cardH},
                                       UITheme::Colors::BorderSubtle, UITheme::Sizes::WidgetRadius);

            float innerX = cardX + kSeqCardInnerPad;
            float innerW = cardW - 2.0f * kSeqCardInnerPad;
            float rowY = cardY;

            // Top bar: number + format dropdown + generate + delete
            renderer.drawText({innerX, rowY + 9.0f},
                              "#" + std::to_string(ci + 1),
                              UITheme::Colors::TextMuted);

            float fmtDdW = 180.0f;
            float genBtnW = 80.0f;
            float delBtnW = 28.0f;

            card.formatDropdown->setPosition(innerX + 30.0f, rowY + 2.0f);
            card.formatDropdown->setSize(fmtDdW, 28.0f);
            card.formatDropdown->drawOverlay(renderer, position);

            card.generateBtn->setPosition(innerX + 30.0f + fmtDdW + 8.0f, rowY + 2.0f);
            card.generateBtn->setSize(genBtnW, 28.0f);
            card.generateBtn->drawOverlay(renderer, position);

            float topRightX = cardX + cardW - kSeqCardInnerPad;
            card.deleteBtn->setPosition(topRightX - delBtnW, rowY + 2.0f);
            card.deleteBtn->setSize(delBtnW, 28.0f);
            card.deleteBtn->drawOverlay(renderer, position);

            rowY += kSeqCardTopBarH;
            renderer.drawRect({innerX, rowY - 1.0f}, {innerW, 1.0f}, UITheme::Colors::DividerLine);

            // Entry fields
            for (auto& entry : card.entries) {
                renderer.drawText({innerX + 2.0f, rowY + 2.0f}, entry.prefix,
                                  UITheme::Colors::TextSecondary);
                rowY += kSeqCardEntryLabelH;

                entry.textArea->setPosition(innerX, rowY);
                entry.textArea->setSize(innerW, kSeqCardEntryTextH);
                entry.textArea->drawOverlay(renderer, position);
                rowY += kSeqCardEntryTextH + kSeqCardEntryGapH;
            }

            // Add Entry button (Thought format only)
            if (card.formatIndex == 1) {
                float addW = 90.0f;
                card.addEntryBtn->setPosition(innerX, rowY);
                card.addEntryBtn->setSize(addW, 24.0f);
                card.addEntryBtn->drawOverlay(renderer, position);
                rowY += kSeqCardBtnRowH;
            }

            // Save button row
            float saveBtnW = 80.0f;
            card.saveBtn->setPosition(innerX + innerW - saveBtnW, rowY);
            card.saveBtn->setSize(saveBtnW, 24.0f);
            card.saveBtn->drawOverlay(renderer, position);

            cardY += cardH + kCardGap;
        }

        if (sequenceCards_.empty()) {
            float emptyX = x + fullW / 2.0f - 160.0f;
            float emptyY = y + cardAreaH / 2.0f - 10.0f;
            renderer.drawText({emptyX, emptyY},
                              "No sequences. Click '+ Add Sequence' below.",
                              UITheme::Colors::TextDisabled);
        }

        renderer.popClipRect();

        // Scroll indicator
        float totalCardH = 10.0f;
        for (const auto& c : sequenceCards_)
            totalCardH += sequenceCardHeight(c) + kCardGap;
        if (totalCardH > cardAreaH) {
            float scrollRatio = seqScrollOffset_ / (totalCardH - cardAreaH);
            float thumbRatio  = cardAreaH / totalCardH;
            float thumbH      = std::max(20.0f, cardAreaH * thumbRatio);
            float thumbY      = y + scrollRatio * (cardAreaH - thumbH);
            float barX        = x + fullW - 6.0f;
            renderer.drawRoundedRect({barX, thumbY}, {4.0f, thumbH},
                                     UITheme::Colors::ScrollThumb, 2.0f);
        }

        y += cardAreaH + 10.0f;

        // Add Sequence button (centered below card area)
        float addBtnW = 220.0f;
        float addBtnH = 36.0f;
        float addBtnX = x + (fullW - addBtnW) / 2.0f;
        btnAddSequence_->setPosition(addBtnX, y);
        btnAddSequence_->setSize(addBtnW, addBtnH);
        btnAddSequence_->drawOverlay(renderer, position);

        y += addBtnH + 5.0f;
    }

    // ── Custom prompt + type ────────────────────────────

    renderer.drawText({x + 2.0f, promptLabelY}, "Custom Prompt",
                      UITheme::Colors::TextSecondary);

    std::string currentFmt = formatDropdown_ ? formatDropdown_->getSelectedItem() : "Q/A";
    float fmtLabelW = UIDrawHelpers::getTextWidth("Type: " + currentFmt);
    renderer.drawText({x + fullW - fmtLabelW - 2.0f, promptLabelY},
                      "Type: " + currentFmt, UITheme::Colors::TextMuted);

    float appendBtnW = 80.0f;
    float promptAreaW = fullW - appendBtnW - 8.0f;

    customPromptArea_->setPosition(x, promptTopY);
    customPromptArea_->setSize(promptAreaW, promptTextH);
    customPromptArea_->drawOverlay(renderer, position);

    btnAppendEntry_->setPosition(x + promptAreaW + 8.0f, promptTopY);
    btnAppendEntry_->setSize(appendBtnW, promptTextH);
    btnAppendEntry_->drawOverlay(renderer, position);

    // ── Status bar ──────────────────────────────────────

    renderer.drawRoundedRect({x, statusY}, {fullW, 25.0f},
                             UITheme::Colors::Background, UITheme::Sizes::SmallRadius);

    std::string status = "Total: " + std::to_string(totalSequences_)
                       + "  |  Assigned: " + std::to_string(assignedSequences_)
                       + "  |  Structured: " + std::to_string(structuredCount_)
                       + "  |  Failed: " + std::to_string(failedCount_);
    renderer.drawText({x + 10.0f, statusY + 5.0f}, status, UITheme::Colors::TextSecondary);
}

// =========================================================
// Curriculum view
// =========================================================

static uint32_t subjectBadgeColor(const std::string& subj) {
    if (subj == "code")    return 0xFF5090FF;
    if (subj == "math")    return 0xFFF0B040;
    if (subj == "science") return 0xFF50E080;
    if (subj == "history") return 0xFFA090FF;
    if (subj == "medical") return 0xFFE84060;
    if (subj == "legal")   return 0xFF6B90F0;
    return 0xFF8888A0;
}

static uint32_t qualityBadgeColor(const std::string& q) {
    if (q == "high")   return 0xFF50E080;
    if (q == "medium") return 0xFFF0B040;
    if (q == "low")    return 0xFFE84060;
    return 0xFF8888A0;
}

void UIDataHubPanel::rebuildFilteredPool() {
    filteredPoolIndices_.clear();
    if (!datasetTarget_) return;

    std::string subj;
    if (subjectFilterDropdown_ && filterSubjectIdx_ > 0)
        subj = subjectFilterDropdown_->getSelectedItem();

    std::string qual;
    if (qualityFilterDropdown_ && filterQualityIdx_ > 0)
        qual = qualityFilterDropdown_->getSelectedItem();

    std::string query;
    if (poolSearchInput_)
        query = poolSearchInput_->getText();

    filteredPoolIndices_ = datasetTarget_->filterSequences(subj, qual, query);
    poolFilterDirty_ = false;
}

void UIDataHubPanel::selectPoolRow(int row) {
    selectedPoolRow_ = row;
    selectedCurrRow_ = -1;
    if (row >= 0 && static_cast<size_t>(row) < filteredPoolIndices_.size()) {
        loadDetailForSequence(filteredPoolIndices_[row]);
    }
}

void UIDataHubPanel::selectCurriculumRow(int row) {
    selectedCurrRow_ = row;
    selectedPoolRow_ = -1;
    if (!datasetTarget_ || row < 0) return;
    const auto& order = datasetTarget_->curriculumOrder();
    if (static_cast<size_t>(row) >= order.size()) return;
    const std::string& seqId = order[row];
    for (size_t i = 0; i < datasetTarget_->massDatasetSize(); ++i) {
        auto seq = datasetTarget_->getSequence(i);
        if (seq.id == seqId) {
            loadDetailForSequence(i);
            return;
        }
    }
}

void UIDataHubPanel::loadDetailForSequence(size_t seqIndex) {
    detailSeqIndex_ = seqIndex;
    if (!datasetTarget_ || seqIndex >= datasetTarget_->massDatasetSize()) return;
    auto seq = datasetTarget_->getSequence(seqIndex);
    if (detailContentArea_)    detailContentArea_->setText(seq.content);
    if (detailStructuredArea_) detailStructuredArea_->setText(seq.structured);
}

void UIDataHubPanel::drawPoolTable(OverlayRenderer& renderer,
                                    float x, float y, float w, float h) {
    renderer.drawRoundedRect({x, y}, {w, h},
                             UITheme::Colors::ContentAreaBg, UITheme::Sizes::SmallRadius);
    renderer.drawRoundedBorder({x, y}, {w, h},
                               UITheme::Colors::BorderSubtle, UITheme::Sizes::SmallRadius);

    // Header row
    renderer.drawRect({x, y}, {w, kPoolHeaderH}, UITheme::Colors::TableHeaderBg);
    float colX = x + 4.0f;
    float hdrY = y + 6.0f;
    renderer.drawText({colX, hdrY}, "#", UITheme::Colors::TextWhite);
    colX += w * kColNum;
    renderer.drawText({colX, hdrY}, "Subject", UITheme::Colors::TextWhite);
    colX += w * kColSubject;
    renderer.drawText({colX, hdrY}, "Quality", UITheme::Colors::TextWhite);
    colX += w * kColQuality;
    renderer.drawText({colX, hdrY}, "Source", UITheme::Colors::TextWhite);
    colX += w * kColSource;
    renderer.drawText({colX, hdrY}, "Str?", UITheme::Colors::TextWhite);
    colX += w * kColStructured;
    renderer.drawText({colX, hdrY}, "Preview", UITheme::Colors::TextWhite);

    float bodyY = y + kPoolHeaderH;
    float bodyH = h - kPoolHeaderH;
    if (bodyH <= 0) return;

    renderer.pushClipRect({x, bodyY}, {w, bodyH});

    int visibleRows = static_cast<int>(bodyH / kPoolRowH) + 1;
    int startRow = static_cast<int>(poolScrollOffset_ / kPoolRowH);
    startRow = std::max(0, std::min(startRow, static_cast<int>(filteredPoolIndices_.size()) - 1));

    int totalRows = static_cast<int>(filteredPoolIndices_.size());

    for (int i = startRow; i < std::min(startRow + visibleRows, totalRows); ++i) {
        size_t seqIdx = filteredPoolIndices_[i];
        auto seq = datasetTarget_->getSequence(seqIdx);

        float rowY = bodyY + (i - startRow) * kPoolRowH
                   - (poolScrollOffset_ - startRow * kPoolRowH);

        bool isAssigned = datasetTarget_->isAssigned(seqIdx);
        uint32_t rowColor = (i == selectedPoolRow_) ? UITheme::Colors::RowSelected :
                            (i == hoveredPoolRow_)  ? UITheme::Colors::RowHover :
                            (i % 2 == 0)            ? UITheme::Colors::RowEven
                                                    : UITheme::Colors::RowOdd;
        renderer.drawRect({x, rowY}, {w, kPoolRowH}, rowColor);

        if (isAssigned) {
            renderer.drawRect({x, rowY}, {3.0f, kPoolRowH}, UITheme::Colors::Success);
        }

        float textY = rowY + 4.0f;
        colX = x + 4.0f;
        renderer.drawText({colX, textY}, std::to_string(seqIdx + 1), UITheme::Colors::TextMuted);
        colX += w * kColNum;

        renderer.drawText({colX, textY}, seq.subject, subjectBadgeColor(seq.subject));
        colX += w * kColSubject;

        renderer.drawText({colX, textY}, seq.quality_tier, qualityBadgeColor(seq.quality_tier));
        colX += w * kColQuality;

        std::string srcShort = seq.source_type;
        if (srcShort.size() > 10) srcShort = srcShort.substr(0, 10);
        renderer.drawText({colX, textY}, srcShort, UITheme::Colors::TextLight);
        colX += w * kColSource;

        renderer.drawText({colX, textY}, seq.is_structured ? "Y" : "-",
                          seq.is_structured ? UITheme::Colors::Success : UITheme::Colors::TextDisabled);
        colX += w * kColStructured;

        float previewW = w - (colX - x) - 6.0f;
        int maxChars = static_cast<int>(previewW / 7.5f);
        std::string preview = seq.content.substr(0, std::min(seq.content.size(), static_cast<size_t>(maxChars)));
        std::replace(preview.begin(), preview.end(), '\n', ' ');
        if (static_cast<int>(seq.content.size()) > maxChars) preview += "...";
        renderer.drawText({colX, textY}, preview, UITheme::Colors::TextPrimary);
    }

    renderer.popClipRect();

    // Scrollbar
    float totalH = totalRows * kPoolRowH;
    if (totalH > bodyH) {
        float ratio = poolScrollOffset_ / (totalH - bodyH);
        float thumbRatio = bodyH / totalH;
        float thumbH = std::max(20.0f, bodyH * thumbRatio);
        float thumbY = bodyY + ratio * (bodyH - thumbH);
        float barX = x + w - 6.0f;
        renderer.drawRoundedRect({barX, thumbY}, {4.0f, thumbH},
                                 UITheme::Colors::ScrollThumb, 2.0f);
    }
}

void UIDataHubPanel::drawCurriculumList(OverlayRenderer& renderer,
                                         float x, float y, float w, float h) {
    renderer.drawRoundedRect({x, y}, {w, h},
                             UITheme::Colors::ContentAreaBg, UITheme::Sizes::SmallRadius);
    renderer.drawRoundedBorder({x, y}, {w, h},
                               UITheme::Colors::BorderSubtle, UITheme::Sizes::SmallRadius);

    if (!datasetTarget_) return;
    const auto& order = datasetTarget_->curriculumOrder();
    const auto& phases = datasetTarget_->phaseMarkers();

    // Header
    std::string hdr = "Curriculum (" + std::to_string(order.size()) + ")";
    renderer.drawRect({x, y}, {w, kPoolHeaderH}, UITheme::Colors::TableHeaderBg);
    renderer.drawText({x + 8.0f, y + 6.0f}, hdr, UITheme::Colors::TextWhite);

    // Curriculum list action menu in header
    float menuW = 100.0f;
    float btnH = 22.0f;
    float btnY = y + 3.0f;
    if (currListActionMenu_) {
        currListActionMenu_->setPosition(x + w - menuW - 8.0f, btnY);
        currListActionMenu_->setSize(menuW, btnH);
        currListActionMenu_->drawOverlay(renderer, position);
    }

    float bodyY = y + kPoolHeaderH;
    float bodyH = h - kPoolHeaderH;
    if (bodyH <= 0) return;

    renderer.pushClipRect({x, bodyY}, {w, bodyH});

    // Build a flat list of items: phases interleaved with sequences
    struct CurrItem { bool isPhase; size_t index; size_t phaseIdx; };
    std::vector<CurrItem> items;
    items.reserve(order.size() + phases.size());

    size_t phaseI = 0;
    for (size_t i = 0; i < order.size(); ++i) {
        while (phaseI < phases.size() && phases[phaseI].position <= i) {
            items.push_back({true, i, phaseI});
            phaseI++;
        }
        items.push_back({false, i, 0});
    }
    while (phaseI < phases.size()) {
        items.push_back({true, order.size(), phaseI});
        phaseI++;
    }

    float itemY = bodyY - currScrollOffset_;
    for (size_t ii = 0; ii < items.size(); ++ii) {
        const auto& item = items[ii];
        float rowH = item.isPhase ? kPhaseRowH : kCurrRowH;

        if (itemY + rowH >= bodyY && itemY < bodyY + bodyH) {
            if (item.isPhase) {
                renderer.drawRect({x, itemY}, {w, rowH}, UITheme::Colors::SectionNeutral);
                renderer.drawRect({x, itemY + rowH - 1.0f}, {w, 1.0f}, UITheme::Colors::DividerLine);
                const std::string& label = phases[item.phaseIdx].label;
                renderer.drawText({x + 10.0f, itemY + 3.0f}, label, UITheme::Colors::TextSecondary);
            } else {
                size_t ci = item.index;
                int row = static_cast<int>(ci);
                uint32_t rowColor = (row == selectedCurrRow_) ? UITheme::Colors::RowSelected :
                                    (row == hoveredCurrRow_)  ? UITheme::Colors::RowHover :
                                    (ci % 2 == 0)             ? UITheme::Colors::RowEven
                                                              : UITheme::Colors::RowOdd;
                renderer.drawRect({x, itemY}, {w, rowH}, rowColor);

                const std::string& seqId = order[ci];
                SequenceHandle seq;
                for (size_t si = 0; si < datasetTarget_->massDatasetSize(); ++si) {
                    auto s = datasetTarget_->getSequence(si);
                    if (s.id == seqId) { seq = s; break; }
                }

                float textY = itemY + 5.0f;
                float cx = x + 6.0f;

                renderer.drawText({cx, textY}, std::to_string(ci + 1), UITheme::Colors::TextMuted);
                cx += 30.0f;

                renderer.drawText({cx, textY}, seq.subject, subjectBadgeColor(seq.subject));
                cx += 65.0f;

                renderer.drawText({cx, textY}, seq.quality_tier, qualityBadgeColor(seq.quality_tier));
                cx += 55.0f;

                float previewW = w - (cx - x) - 60.0f;
                int maxChars = static_cast<int>(previewW / 7.5f);
                if (maxChars < 0) maxChars = 0;
                std::string preview = seq.content.substr(
                    0, std::min(seq.content.size(), static_cast<size_t>(maxChars)));
                std::replace(preview.begin(), preview.end(), '\n', ' ');
                if (static_cast<int>(seq.content.size()) > maxChars) preview += "...";
                renderer.drawText({cx, textY}, preview, UITheme::Colors::TextPrimary);

                // Move up/down arrows on right side
                float arrowX = x + w - 50.0f;
                if (ci > 0) {
                    renderer.drawText({arrowX, textY}, "^", UITheme::Colors::TextLink);
                }
                if (ci + 1 < order.size()) {
                    renderer.drawText({arrowX + 18.0f, textY}, "v", UITheme::Colors::TextLink);
                }
            }
        }

        itemY += rowH;
    }

    renderer.popClipRect();

    // Scrollbar
    float totalH = 0;
    for (const auto& it : items)
        totalH += it.isPhase ? kPhaseRowH : kCurrRowH;
    if (totalH > bodyH) {
        float ratio = currScrollOffset_ / (totalH - bodyH);
        float thumbRatio = bodyH / totalH;
        float thumbH = std::max(20.0f, bodyH * thumbRatio);
        float thumbY = bodyY + ratio * (bodyH - thumbH);
        float barX = x + w - 6.0f;
        renderer.drawRoundedRect({barX, thumbY}, {4.0f, thumbH},
                                 UITheme::Colors::ScrollThumb, 2.0f);
    }
}

void UIDataHubPanel::drawDetailEditor(OverlayRenderer& renderer,
                                       float x, float y, float w, float h) {
    // Divider bar (clickable to toggle)
    renderer.drawRect({x, y}, {w, kDetailDividerH}, UITheme::Colors::SectionNeutral);
    renderer.drawRect({x, y + kDetailDividerH - 1.0f}, {w, 1.0f}, UITheme::Colors::DividerLine);
    std::string divLabel = detailPanelOpen_ ? "Detail  [click to collapse]" : "Detail  [click to expand]";
    renderer.drawText({x + 10.0f, y + 2.0f}, divLabel, UITheme::Colors::TextSecondary);

    if (!detailPanelOpen_) return;

    float panelY = y + kDetailDividerH;
    float panelH = h - kDetailDividerH;
    if (panelH < 40.0f) return;

    renderer.drawRoundedRect({x, panelY}, {w, panelH},
                             UITheme::Colors::ContentAreaBg, UITheme::Sizes::SmallRadius);

    if (detailSeqIndex_ == SIZE_MAX) {
        renderer.drawText({x + 20.0f, panelY + panelH / 2.0f - 8.0f},
                          "Select a sequence to view details",
                          UITheme::Colors::TextDisabled);
        return;
    }

    float halfW = (w - 18.0f) / 2.0f;
    float areaH = panelH - 32.0f;
    if (areaH < 30.0f) areaH = 30.0f;

    detailContentArea_->setPosition(x + 4.0f, panelY + 4.0f);
    detailContentArea_->setSize(halfW, areaH);
    detailContentArea_->drawOverlay(renderer, position);

    detailStructuredArea_->setPosition(x + halfW + 14.0f, panelY + 4.0f);
    detailStructuredArea_->setSize(halfW, areaH);
    detailStructuredArea_->drawOverlay(renderer, position);

    float saveBtnW = 80.0f;
    float saveBtnH = 24.0f;
    btnDetailSave_->setPosition(x + w - saveBtnW - 8.0f, panelY + panelH - saveBtnH - 4.0f);
    btnDetailSave_->setSize(saveBtnW, saveBtnH);
    btnDetailSave_->drawOverlay(renderer, position);
}

void UIDataHubPanel::drawCurriculumView(OverlayRenderer& renderer,
                                         const PanelRect& /*content*/,
                                         float x, float y, float fullW,
                                         float contentEndY) {
    if (poolFilterDirty_) rebuildFilteredPool();

    float availH = contentEndY - y;
    if (availH < 100.0f) return;

    // Filter bar
    float filterX = x;
    float ddW = 140.0f;
    float gap = 8.0f;
    float searchW = 200.0f;

    subjectFilterDropdown_->setPosition(filterX, y);
    subjectFilterDropdown_->setSize(ddW, kFilterBarH);
    subjectFilterDropdown_->drawOverlay(renderer, position);
    filterX += ddW + gap;

    qualityFilterDropdown_->setPosition(filterX, y);
    qualityFilterDropdown_->setSize(ddW, kFilterBarH);
    qualityFilterDropdown_->drawOverlay(renderer, position);
    filterX += ddW + gap;

    poolSearchInput_->setPosition(filterX, y + 2.0f);
    poolSearchInput_->setSize(searchW, kFilterBarH - 4.0f);
    poolSearchInput_->drawOverlay(renderer, position);
    filterX += searchW + gap;

    btnAssignSelected_->setPosition(filterX, y + 2.0f);
    btnAssignSelected_->setSize(100.0f, kFilterBarH - 4.0f);
    btnAssignSelected_->drawOverlay(renderer, position);

    y += kFilterBarH + 6.0f;

    // Rebuild filter if search text changed
    std::string curSearch = poolSearchInput_ ? poolSearchInput_->getText() : "";
    if (curSearch != filterSearchQuery_) {
        filterSearchQuery_ = curSearch;
        rebuildFilteredPool();
    }

    // Layout: left pool (60%), right curriculum (40%)
    float detailH = detailPanelOpen_ ? detailPanelHeight_ : kDetailDividerH;
    float splitH = (contentEndY - y) - detailH - 6.0f;
    if (splitH < 80.0f) splitH = 80.0f;

    float leftW  = fullW * 0.58f;
    float rightW = fullW - leftW - 10.0f;

    drawPoolTable(renderer, x, y, leftW, splitH);
    drawCurriculumList(renderer, x + leftW + 10.0f, y, rightW, splitH);

    float detailY = y + splitH + 6.0f;
    drawDetailEditor(renderer, x, detailY, fullW, detailH);

    // Status bar
    float statusY = contentEndY + 2.0f;
    renderer.drawRoundedRect({x, statusY}, {fullW, 25.0f},
                             UITheme::Colors::Background, UITheme::Sizes::SmallRadius);
    std::string status = "Pool: " + std::to_string(filteredPoolIndices_.size())
                       + " / " + std::to_string(totalSequences_)
                       + "  |  Curriculum: " + std::to_string(assignedSequences_)
                       + " sequences";
    if (datasetTarget_) {
        status += "  |  " + std::to_string(datasetTarget_->phaseMarkers().size()) + " phases";
    }
    renderer.drawText({x + 10.0f, statusY + 5.0f}, status, UITheme::Colors::TextSecondary);
}

// =========================================================
// Home tab actions
// =========================================================

void UIDataHubPanel::startCollection(const std::string& mode) {
    if (collectionActive_) {
        addLog("Collection already active", 1);
        return;
    }

    addLog("=== STARTING " + mode + " ===", 0);
    collectionActive_    = true;
    collectionCompleted_ = false;

    GRIM::Pipeline::PipelineMode pipelineMode = GRIM::Pipeline::PipelineMode::Full;
    if (mode == "collect")       pipelineMode = GRIM::Pipeline::PipelineMode::CollectOnly;
    else if (mode == "verify")   pipelineMode = GRIM::Pipeline::PipelineMode::VerifyOnly;
    else if (mode == "merge")    pipelineMode = GRIM::Pipeline::PipelineMode::MergeOnly;
    else if (mode == "merge-rebuild") pipelineMode = GRIM::Pipeline::PipelineMode::MergeRebuild;

    pipelineOrchestrator_->startPipeline(pipelineMode);
    addLog(mode + " started", 0);
    LOG_DEBUG("DataHub", "Pipeline started: " + mode);
}

void UIDataHubPanel::stopCollection() {
    if (!collectionActive_) {
        addLog("No active collection to stop", 1);
        return;
    }
    addLog("Stopping collection...", 0);
    pipelineOrchestrator_->stopPipeline();
    collectionActive_ = false;
    LOG_DEBUG("DataHub", "Collection stopped by user");
}

void UIDataHubPanel::pollCollectionManager() {
    if (!pipelineOrchestrator_) return;

    auto status = pipelineOrchestrator_->getStatus();
    currentPhase_          = status.phase;
    currentProgress_       = status.progress;
    collectionMessage_     = status.message;
    entriesCollected_      = static_cast<int>(status.stats.entriesIngested);
    duplicatesSkipped_     = static_cast<int>(status.stats.duplicatesRemoved);
    elapsedSeconds_        = status.elapsedSeconds;

    if (status.state == GRIM::Pipeline::PipelineState::Complete && collectionActive_) {
        addLog("Pipeline completed successfully", 0);
        collectionActive_    = false;
        collectionCompleted_ = true;
        updateDatasetStats();
    } else if (status.state == GRIM::Pipeline::PipelineState::Error && collectionActive_) {
        addLog("Pipeline error: " + status.message, 2);
        collectionActive_ = false;
    }
}

void UIDataHubPanel::updateDatasetStats() {
    auto snapshot = GRIM::Config::loadAiConfigSnapshot();
    if (!snapshot || !snapshot->hasRequiredGrimTextPaths()) {
        datasetSizeInfo_ = "Dataset: Config error";
        hudFileSize_ = "N/A";
        return;
    }

    try {
        if (std::filesystem::exists(snapshot->grim_text_training_data)) {
            auto sz = std::filesystem::file_size(snapshot->grim_text_training_data);
            std::stringstream ss;
            if (sz >= 1024ULL * 1024 * 1024)
                ss << std::fixed << std::setprecision(2) << (sz / (1024.0 * 1024.0 * 1024.0)) << " GB";
            else if (sz >= 1024ULL * 1024)
                ss << std::fixed << std::setprecision(2) << (sz / (1024.0 * 1024.0)) << " MB";
            else
                ss << (sz / 1024) << " KB";
            hudFileSize_ = ss.str();
            datasetSizeInfo_ = "Dataset: " + hudFileSize_;
        } else {
            datasetSizeInfo_ = "Dataset: Not found";
            hudFileSize_ = "N/A";
        }

        std::string ckptDir = GRIM::Config::getRequiredGrimTextPath(GRIM::Config::GrimTextPathKey::Checkpoints);
        if (std::filesystem::exists(ckptDir)) {
            int count = 0;
            for (const auto& e : std::filesystem::directory_iterator(ckptDir))
                if (e.path().extension() == ".ckpt") count++;
            hudCheckpoints_ = count;
            checkpointStatsInfo_ = "Checkpoints: " + std::to_string(count);
        }
    } catch (const std::exception&) {
        datasetSizeInfo_ = "Dataset: Error";
        hudFileSize_ = "Error";
    }
}

void UIDataHubPanel::addLog(const std::string& message, int level) {
    std::lock_guard<std::mutex> lock(logMutex_);

    auto now  = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);

    char buf[32];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", &tm);

    logEntries_.push_back({buf, message, level});
    if (logEntries_.size() > maxLogEntries_)
        logEntries_.erase(logEntries_.begin());

    if (level >= 1)
        LOG_DEBUG("DataHub", message);
}

// =========================================================
// Directory collection
// =========================================================

void UIDataHubPanel::loadDirectoryCollectionPathFromConfig() {
    if (!dirPathInput_) {
        throw std::runtime_error("dirPathInput_ is NULL - Home tab directory input must exist");
    }

    auto snapshot = GRIM::Config::loadAiConfigSnapshot();
    if (!snapshot || !snapshot->has_grim_paths) {
        addLog("Failed to load ai_config.json paths for directory collection", 2);
        return;
    }

    if (snapshot->grim_text_directory_collection.empty()) {
        addLog("ai_config.json missing paths.grim_text.directory_collection for directory collection", 2);
        return;
    }

    dirPathInput_->setText(snapshot->grim_text_directory_collection);
    dirScanPath_ = snapshot->grim_text_directory_collection;
    dirNeedsScan_ = true;
    addLog("Loaded directory collection path from ai_config.json", 0);
}

void UIDataHubPanel::scanDirectory() {
    dirFileEntries_.clear();
    dirScrollOffset_ = 0.0f;

    if (dirScanPath_.empty()) return;

    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::exists(dirScanPath_, ec)) {
        addLog("Directory collection path does not exist: " + dirScanPath_, 2);
        return;
    }
    if (!fs::is_directory(dirScanPath_, ec)) {
        addLog("Directory collection path is not a directory: " + dirScanPath_, 2);
        return;
    }

    for (const auto& entry : fs::directory_iterator(dirScanPath_, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        DirFileEntry fe;
        fe.filename    = entry.path().filename().string();
        fe.collect     = true;
        fe.deleteAfter = false;
        dirFileEntries_.push_back(std::move(fe));
    }

    std::sort(dirFileEntries_.begin(), dirFileEntries_.end(),
              [](const DirFileEntry& a, const DirFileEntry& b) {
                  return a.filename < b.filename;
              });

    addLog("Scanned directory: " + std::to_string(dirFileEntries_.size()) + " file(s)", 0);
}

void UIDataHubPanel::collectFromDirectory() {
    if (!datasetTarget_) {
        addLog("Dataset target not initialized", 2);
        return;
    }
    if (dirScanPath_.empty()) {
        addLog("No directory path specified", 1);
        return;
    }

    namespace fs = std::filesystem;
    int collected = 0;
    int failed    = 0;

    for (auto it = dirFileEntries_.begin(); it != dirFileEntries_.end(); ) {
        if (!it->collect) { ++it; continue; }

        fs::path filePath = fs::path(dirScanPath_) / it->filename;
        std::error_code ec;
        if (!fs::exists(filePath, ec) || !fs::is_regular_file(filePath, ec)) {
            addLog("File not found: " + it->filename, 2);
            ++failed;
            ++it;
            continue;
        }

        // Read file content
        std::ifstream ifs(filePath, std::ios::in | std::ios::binary);
        if (!ifs.is_open()) {
            addLog("Cannot open: " + it->filename, 2);
            ++failed;
            ++it;
            continue;
        }
        std::ostringstream oss;
        oss << ifs.rdbuf();
        ifs.close();
        std::string content = oss.str();

        if (content.empty()) {
            addLog("Empty file skipped: " + it->filename, 1);
            ++it;
            continue;
        }

        // Determine source type from extension
        std::string ext = filePath.extension().string();
        std::string sourceType = "file";
        if (ext == ".txt")       sourceType = "text_file";
        else if (ext == ".json") sourceType = "json_file";
        else if (ext == ".jsonl") sourceType = "jsonl_file";
        else if (ext == ".md")   sourceType = "markdown_file";
        else if (ext == ".csv")  sourceType = "csv_file";
        else if (ext == ".html" || ext == ".htm") sourceType = "html_file";

        if (datasetTarget_->appendStructuredEntry(content, "", sourceType, filePath.string())) {
            ++collected;

            // Delete the file if flagged
            if (it->deleteAfter) {
                std::error_code delEc;
                fs::remove(filePath, delEc);
                if (delEc)
                    addLog("Collected but failed to delete: " + it->filename, 1);
                it = dirFileEntries_.erase(it);
                continue;
            }
        } else {
            addLog("Failed to append: " + it->filename, 2);
            ++failed;
        }
        ++it;
    }

    totalSequences_ = datasetTarget_->massDatasetSize();
    updateDatasetStats();

    addLog("Directory collection: " + std::to_string(collected) + " collected, "
         + std::to_string(failed) + " failed", collected > 0 ? 0 : 1);

    // Rescan to refresh the file list
    dirNeedsScan_ = true;
}

// =========================================================
// Sources tab actions
// =========================================================

UIDataHubPanel::SourceCard UIDataHubPanel::buildSourceCard(
    const std::string& name, const std::string& url,
    int priority, int depth, int limit, bool enabled)
{
    SourceCard card;
    card.cardId     = nextSourceCardId_++;
    card.name       = name;
    card.url        = url;
    card.priority   = priority;
    card.enabled    = enabled;
    card.crawlDepth = depth;
    card.fetchLimit = limit;

    card.nameInput = std::make_shared<UIInputBox>();
    card.nameInput->setPlaceholder("Source name...");
    if (!name.empty() && name != "New Source") card.nameInput->setText(name);

    card.urlInput = std::make_shared<UIInputBox>();
    card.urlInput->setPlaceholder("https://...");
    if (!url.empty()) card.urlInput->setText(url);

    card.prioritySlider = std::make_shared<UISlider>(
        "Priority", 1.0f, 10.0f, static_cast<float>(priority),
        [this](float) { sourcesDirty_ = true; }, 1.0f);

    card.depthSlider = std::make_shared<UISlider>(
        "Crawl Depth", 1.0f, 5.0f, static_cast<float>(depth),
        [this](float) { sourcesDirty_ = true; }, 1.0f);

    card.limitSlider = std::make_shared<UISlider>(
        "Fetch Limit", 0.0f, 5000.0f, static_cast<float>(limit),
        [this](float) { sourcesDirty_ = true; }, 50.0f);

    card.enabledToggle = std::make_shared<UIToggle>(
        "", enabled, [this](bool) { sourcesDirty_ = true; });

    size_t id = card.cardId;
    card.deleteBtn = std::make_shared<UIButton>("x", [this, id]() {
        cardToDelete_ = static_cast<int>(id);
    });

    return card;
}

void UIDataHubPanel::removeSourceCard(size_t cardId) {
    sourceCards_.erase(
        std::remove_if(sourceCards_.begin(), sourceCards_.end(),
            [cardId](const SourceCard& c) { return c.cardId == cardId; }),
        sourceCards_.end());
    sourcesDirty_ = true;
}

void UIDataHubPanel::loadSourceCards() {
    std::string sourcePath = getSourceConfigPath();
    sourceCards_.clear();
    try {
        if (!std::filesystem::exists(sourcePath)) return;
        std::ifstream f(sourcePath);
        nlohmann::json data;
        f >> data;
        if (!data.contains("data_sources")) return;

        for (const auto& s : data["data_sources"]) {
            sourceCards_.push_back(buildSourceCard(
                s.value("name", "Unknown"),
                s.value("url", ""),
                s.value("priority", 5),
                s.value("crawl_depth", 2),
                s.value("fetch_limit", 100),
                s.value("enabled", true)
            ));
        }

        totalSources_ = 0;
        for (const auto& c : sourceCards_)
            if (c.enabled) totalSources_++;

        LOG_DEBUG("DataHub", "Loaded " + std::to_string(sourceCards_.size()) + " source cards");
    } catch (const std::exception& e) {
        addLog("Error loading sources: " + std::string(e.what()), 2);
        sourceCards_.clear();
    }
}

void UIDataHubPanel::saveSourceCards() {
    std::string sourcePath = getSourceConfigPath();
    try {
        nlohmann::json data;
        std::ifstream inFile(sourcePath);
        if (inFile.good()) {
            inFile >> data;
            inFile.close();
        } else {
            data["version"]     = "1.0.0";
            data["description"] = "GRIM Web Data Collection Configuration";
        }

        data["data_sources"] = nlohmann::json::array();
        for (const auto& card : sourceCards_) {
            data["data_sources"].push_back({
                {"name",         card.name},
                {"url",          card.url},
                {"source_type",  detectFetcherFromUrl(card.url)},
                {"priority",     card.priority},
                {"enabled",      card.enabled},
                {"crawl_depth",  card.crawlDepth},
                {"fetch_limit",  card.fetchLimit},
                {"requires_auth", card.requiresAuth}
            });
        }

        std::ofstream outFile(sourcePath);
        if (outFile.is_open()) {
            outFile << data.dump(2);
            addLog("Sources saved (" + std::to_string(sourceCards_.size()) + " entries)", 0);
        }

        totalSources_ = 0;
        for (const auto& c : sourceCards_)
            if (c.enabled) totalSources_++;

    } catch (const std::exception& e) {
        addLog("Error saving sources: " + std::string(e.what()), 2);
    }
    sourcesDirty_ = false;
}

// =========================================================
// HuggingFace tab actions
// =========================================================

void UIDataHubPanel::searchHuggingFaceDatasets() {
    if (hfSearchBuffer_.empty()) { addLog("Enter a search query", 1); return; }
    if (hfSearching_.load()) { addLog("Search already in progress", 1); return; }

    addLog("Searching HF: " + hfSearchBuffer_, 0);
    hfSearching_.store(true);
    lastSearchError_.clear();
    searchAnimTime_ = 0.0f;
    hfSelectedResultIndex_ = -1;
    hfPreviewDatasetId_.clear();
    hfPreviewDisplayName_.clear();
    ++hfPreviewGen_;
    if (hfPreviewArea_) hfPreviewArea_->setText("Searching...");

    std::string query = hfSearchBuffer_;
    std::string token = hfTokenBuffer_;

    std::thread([this, query, token]() {
        try {
            if (!token.empty() && hfWebhook_) hfWebhook_->setApiToken(token);
            auto results = hfWebhook_->searchDatasets(query, 20, "task_categories:text-generation");
            hfSearchResults_ = std::move(results);
            hfSelectedResultIndex_ = -1;
            hfPreviewDatasetId_.clear();
            hfPreviewDisplayName_.clear();
            if (hfSearchResults_.empty()) {
                lastSearchError_ = "No datasets found for: " + query;
                addLog("No results for: " + query, 1);
                hfPreviewPendingText_ = "No datasets match this search.";
                hfPreviewApply_.store(true);
            } else {
                lastSearchError_.clear();
                addLog("Found " + std::to_string(hfSearchResults_.size()) + " datasets", 0);
                hfPreviewPendingText_ =
                    "Click a dataset in the list to preview sample rows (Hugging Face datasets-server, same as the website viewer).";
                hfPreviewApply_.store(true);
            }
        } catch (const std::exception& e) {
            lastSearchError_ = "Search failed: " + std::string(e.what());
            addLog("HF search error: " + std::string(e.what()), 2);
            hfSearchResults_.clear();
            hfPreviewPendingText_ = std::string("Search failed: ") + e.what();
            hfPreviewApply_.store(true);
        }
        hfSearching_.store(false);
        hfResultsNeedsPopulate_.store(true);
    }).detach();
}

void UIDataHubPanel::searchHuggingFaceByCategory(const std::string& category) {
    if (category.empty() || category == "All Categories") return;
    if (hfSearching_.load()) { addLog("Search already in progress", 1); return; }

    addLog("Searching HF category: " + category, 0);
    hfSearching_.store(true);
    lastSearchError_.clear();
    searchAnimTime_ = 0.0f;
    hfSelectedResultIndex_ = -1;
    hfPreviewDatasetId_.clear();
    hfPreviewDisplayName_.clear();
    ++hfPreviewGen_;
    if (hfPreviewArea_) hfPreviewArea_->setText("Searching...");

    std::string token = hfTokenBuffer_;

    std::thread([this, category, token]() {
        try {
            if (!token.empty() && hfWebhook_) hfWebhook_->setApiToken(token);
            auto results = hfWebhook_->searchDatasets("", 20, "task_categories:" + category);
            hfSearchResults_ = std::move(results);
            hfSelectedResultIndex_ = -1;
            hfPreviewDatasetId_.clear();
            hfPreviewDisplayName_.clear();
            if (hfSearchResults_.empty()) {
                lastSearchError_ = "No datasets in category: " + category;
                hfPreviewPendingText_ = "No datasets in this category.";
                hfPreviewApply_.store(true);
            } else {
                lastSearchError_.clear();
                addLog("Found " + std::to_string(hfSearchResults_.size()) + " datasets", 0);
                hfPreviewPendingText_ =
                    "Click a dataset in the list to preview sample rows (Hugging Face datasets-server, same as the website viewer).";
                hfPreviewApply_.store(true);
            }
        } catch (const std::exception& e) {
            lastSearchError_ = "Category search failed: " + std::string(e.what());
            hfSearchResults_.clear();
            hfPreviewPendingText_ = std::string("Category search failed: ") + e.what();
            hfPreviewApply_.store(true);
        }
        hfSearching_.store(false);
        hfResultsNeedsPopulate_.store(true);
    }).detach();
}

void UIDataHubPanel::browseHuggingFaceDatasets() {
    if (hfSearching_.load()) { addLog("Search already in progress", 1); return; }

    addLog("Browsing popular HF datasets...", 0);
    hfSearching_.store(true);
    lastSearchError_.clear();
    searchAnimTime_ = 0.0f;
    hfSelectedResultIndex_ = -1;
    hfPreviewDatasetId_.clear();
    hfPreviewDisplayName_.clear();
    ++hfPreviewGen_;
    if (hfPreviewArea_) hfPreviewArea_->setText("Loading...");

    std::string token = hfTokenBuffer_;

    std::thread([this, token]() {
        try {
            if (!token.empty() && hfWebhook_) hfWebhook_->setApiToken(token);
            auto results = hfWebhook_->searchDatasets("", 20, "task_categories:text-generation");
            hfSearchResults_ = std::move(results);
            hfSelectedResultIndex_ = -1;
            hfPreviewDatasetId_.clear();
            hfPreviewDisplayName_.clear();
            if (hfSearchResults_.empty()) {
                lastSearchError_ = "Could not load popular datasets";
                hfPreviewPendingText_ = "Could not load popular datasets.";
                hfPreviewApply_.store(true);
            } else {
                lastSearchError_.clear();
                addLog("Loaded " + std::to_string(hfSearchResults_.size()) + " popular datasets", 0);
                hfPreviewPendingText_ =
                    "Click a dataset in the list to preview sample rows (Hugging Face datasets-server, same as the website viewer).";
                hfPreviewApply_.store(true);
            }
        } catch (const std::exception& e) {
            lastSearchError_ = "Browse failed: " + std::string(e.what());
            hfSearchResults_.clear();
            hfPreviewPendingText_ = std::string("Browse failed: ") + e.what();
            hfPreviewApply_.store(true);
        }
        hfSearching_.store(false);
        hfResultsNeedsPopulate_.store(true);
    }).detach();
}

void UIDataHubPanel::populateHFResults(float containerWidth) {
    if (!hfResultsScrollBox_) return;
    hfResultsScrollBox_->clearChildren();

    for (size_t i = 0; i < hfSearchResults_.size(); ++i) {
        const auto& ds = hfSearchResults_[i];
        std::string nameLine = ds.name.empty() ? ds.id : ds.name;
        std::string prefix = (static_cast<int>(i) == hfSelectedResultIndex_) ? "> " : "  ";
        std::string label = prefix + nameLine + "\n " + ds.author
            + "\n Downloads: " + std::to_string(ds.downloads / 1000) + "k | Likes: " + std::to_string(ds.likes);
        auto btn = std::make_shared<UIButton>(label, [this, i]() { selectHuggingFaceResult(i); });
        btn->setSize(containerWidth - 10.0f, 70.0f);
        hfResultsScrollBox_->addChild(btn);
    }
    hfResultsScrollBox_->autoLayoutChildren();
}

void UIDataHubPanel::selectHuggingFaceResult(size_t index) {
    if (index >= hfSearchResults_.size() || !hfWebhook_) return;

    int generation = ++hfPreviewGen_;
    hfSelectedResultIndex_ = static_cast<int>(index);
    hfResultsNeedsPopulate_.store(true);

    const auto& ds = hfSearchResults_[index];
    hfPreviewDatasetId_ = ds.id;
    hfPreviewDisplayName_ = ds.name.empty() ? ds.id : ds.name;

    if (hfPreviewArea_)
        hfPreviewArea_->setText("Loading sample rows from datasets.huggingface.co ...");

    std::string token = hfTokenBuffer_;
    std::string idCopy = ds.id;

    std::thread([this, idCopy, token, generation]() {
        if (!token.empty() && hfWebhook_) hfWebhook_->setApiToken(token);
        std::string sample = hfWebhook_->getDatasetPreviewSample(idCopy, 6, 480);
        if (hfPreviewGen_.load() != generation) return;
        if (sample.empty()) {
            std::string err = hfWebhook_->getLastError();
            sample = err.empty()
                ? "No preview available. The dataset may be private, gated, or not supported by the Hugging Face datasets viewer API."
                : err;
        }
        hfPreviewPendingText_ = std::move(sample);
        hfPreviewApply_.store(true);
    }).detach();
}

void UIDataHubPanel::addToDownloadQueue(const std::string& datasetId, const std::string& name) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    for (const auto& item : downloadQueue_)
        if (item.datasetId == datasetId) { addLog("Already in queue: " + name, 1); return; }
    if (hfWebhook_ && hfWebhook_->isDatasetDownloaded(datasetId)) {
        addLog("Already downloaded: " + name, 1); return;
    }
    downloadQueue_.push_back({datasetId, name.empty() ? datasetId : name, "pending", 0.0f, 0, ""});
    addLog("Queued: " + (name.empty() ? datasetId : name), 0);
    saveDownloadQueue();
}

void UIDataHubPanel::processDownloadQueue() {
    if (queueProcessing_.load()) { addLog("Queue already processing", 1); return; }
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        if (downloadQueue_.empty()) { addLog("Queue is empty", 1); return; }
        bool hasPending = false;
        for (const auto& i : downloadQueue_) if (i.status == "pending") { hasPending = true; break; }
        if (!hasPending) { addLog("No pending items", 1); return; }
    }

    queueProcessing_.store(true);
    currentQueueIndex_.store(-1);
    addLog("=== PROCESSING DOWNLOAD QUEUE ===", 0);

    std::thread([this]() {
        size_t success = 0, fail = 0, skipped = 0;
        size_t qSize;
        { std::lock_guard<std::mutex> lock(queueMutex_); qSize = downloadQueue_.size(); }

        for (size_t i = 0; i < qSize; ++i) {
            std::string datasetId, displayName;
            bool shouldProcess = false;
            {
                std::lock_guard<std::mutex> lock(queueMutex_);
                if (i >= downloadQueue_.size()) break;
                auto& item = downloadQueue_[i];
                if (item.status != "pending") continue;
                if (hfWebhook_ && hfWebhook_->isDatasetDownloaded(item.datasetId)) {
                    item.status = "completed"; item.progress = 1.0f;
                    skipped++; continue;
                }
                item.status = "downloading"; item.progress = 0.0f;
                datasetId = item.datasetId; displayName = item.displayName;
                shouldProcess = true;
            }
            if (!shouldProcess) continue;
            currentQueueIndex_.store(static_cast<int>(i));
            addLog("Downloading: " + displayName, 0);

            std::string datasetName = datasetId;
            size_t slash = datasetId.find('/');
            if (slash != std::string::npos) datasetName = datasetId.substr(slash + 1);
            std::string outDir = getResourcePath() + "/models/GRIM-text/training/data/huggingface/" + datasetName;

            try {
                bool ok = hfWebhook_->downloadDataset(datasetId, outDir, "train", "",
                    [this, i](size_t downloaded, size_t total, const std::string&) {
                        std::lock_guard<std::mutex> lock(queueMutex_);
                        if (i < downloadQueue_.size() && total > 0)
                            downloadQueue_[i].progress = static_cast<float>(downloaded) / static_cast<float>(total);
                    });
                {
                    std::lock_guard<std::mutex> lock(queueMutex_);
                    if (i < downloadQueue_.size()) {
                        if (ok) {
                            downloadQueue_[i].status = "completed";
                            downloadQueue_[i].progress = 1.0f;
                            success++;
                            addLog("Completed: " + displayName, 0);
                        } else {
                            downloadQueue_[i].status = "failed";
                            downloadQueue_[i].errorMessage = hfWebhook_->getLastError();
                            fail++;
                            addLog("Failed: " + displayName, 2);
                        }
                    }
                }
                if (ok) {
                    sourceCards_.push_back(buildSourceCard(
                        displayName, "huggingface://" + datasetId));
                    sourcesDirty_ = true;
                }
                saveDownloadQueue();
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(queueMutex_);
                if (i < downloadQueue_.size()) {
                    downloadQueue_[i].status = "failed";
                    downloadQueue_[i].errorMessage = e.what();
                    fail++;
                }
                saveDownloadQueue();
            }
        }

        currentQueueIndex_.store(-1);
        queueProcessing_.store(false);
        std::string summary = "Queue done: " + std::to_string(success) + " ok, "
                            + std::to_string(fail) + " failed";
        if (skipped > 0) summary += ", " + std::to_string(skipped) + " skipped";
        addLog(summary, 0);
        saveDownloadQueue();
        updateDatasetStats();
    }).detach();
}

void UIDataHubPanel::clearDownloadQueue() {
    if (queueProcessing_.load()) { addLog("Cannot clear while processing", 1); return; }
    std::lock_guard<std::mutex> lock(queueMutex_);
    downloadQueue_.clear();
    currentQueueIndex_.store(-1);
    addLog("Queue cleared", 0);
    saveDownloadQueue();
}

void UIDataHubPanel::removeFromQueue(int index) {
    if (queueProcessing_.load() && currentQueueIndex_.load() == index) return;
    std::lock_guard<std::mutex> lock(queueMutex_);
    if (index >= 0 && index < static_cast<int>(downloadQueue_.size())) {
        std::string name = downloadQueue_[static_cast<size_t>(index)].displayName;
        downloadQueue_.erase(downloadQueue_.begin() + index);
        addLog("Removed: " + name, 0);
        saveDownloadQueue();
    }
}

void UIDataHubPanel::retryQueueItem(int index) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    if (index >= 0 && index < static_cast<int>(downloadQueue_.size())) {
        auto& item = downloadQueue_[static_cast<size_t>(index)];
        if (item.status == "failed") {
            item.status = "pending";
            item.retryCount++;
            item.progress = 0.0f;
            item.errorMessage.clear();
            addLog("Retrying: " + item.displayName, 0);
            saveDownloadQueue();
        }
    }
}

void UIDataHubPanel::clearCompletedFromQueue() {
    std::lock_guard<std::mutex> lock(queueMutex_);
    size_t before = downloadQueue_.size();
    downloadQueue_.erase(
        std::remove_if(downloadQueue_.begin(), downloadQueue_.end(),
            [](const QueuedDownload& d) { return d.status == "completed"; }),
        downloadQueue_.end());
    size_t removed = before - downloadQueue_.size();
    if (removed > 0) addLog("Cleared " + std::to_string(removed) + " completed", 0);
}

void UIDataHubPanel::updateQueueInteraction(const InputState& input,
                                            float queueBoxX, float queueBoxY,
                                            float queueBoxW, float queueBoxH) {
    hoveredQueueItem_ = -1;
    std::lock_guard<std::mutex> lock(queueMutex_);
    if (downloadQueue_.empty()) return;

    float itemH = UITheme::Sizes::WidgetHeight;
    bool mouseOver = (input.mousePos.x >= queueBoxX && input.mousePos.x <= queueBoxX + queueBoxW &&
                      input.mousePos.y >= queueBoxY && input.mousePos.y <= queueBoxY + queueBoxH);
    if (!mouseOver) return;

    float itemY = queueBoxY + 5.0f;
    for (size_t i = 0; i < downloadQueue_.size(); ++i) {
        if (input.mousePos.y >= itemY && input.mousePos.y <= itemY + itemH) {
            hoveredQueueItem_ = static_cast<int>(i);
            if (input.mousePressed[0] && downloadQueue_[i].status != "downloading") {
                float removeX = queueBoxX + queueBoxW - 25.0f;
                if (input.mousePos.x >= removeX && input.mousePos.x <= removeX + 18.0f) {
                    downloadQueue_.erase(downloadQueue_.begin() + static_cast<ptrdiff_t>(i));
                    addLog("Removed from queue", 0);
                    break;
                }
            }
            break;
        }
        itemY += itemH;
    }

    if (input.mousePressed[1]) clearCompletedFromQueue();
}

// =========================================================
// Structurer helpers
// =========================================================

void UIDataHubPanel::populateModelDropdown() {
    auto models = GRIM::MMO::ModelRegistry::instance().getAllModels();
    std::vector<std::string> names;
    for (const auto* m : models) names.push_back(m->name);
    if (names.empty()) names.push_back("(no models)");
    if (modelDropdown_) modelDropdown_->setItems(names);
}

void UIDataHubPanel::refreshStructurerState() {
    if (!datasetTarget_) return;
    datasetTarget_->loadMassDataset();
    totalSequences_ = datasetTarget_->massDatasetSize();
    assignedSequences_ = datasetTarget_->assignedCount();
    structuredCount_ = 0;
    for (size_t i = 0; i < totalSequences_; ++i)
        if (datasetTarget_->getSequence(i).is_structured) structuredCount_++;
    if (currentSequenceIndex_ >= totalSequences_ && totalSequences_ > 0)
        currentSequenceIndex_ = totalSequences_ - 1;
    loadCurrentSequence();
}

void UIDataHubPanel::loadCurrentSequence() {
    if (!datasetTarget_ || datasetTarget_->massDatasetSize() == 0) {
        if (rawTextArea_) rawTextArea_->setText("");
        if (structuredTextArea_) structuredTextArea_->setText("");
        return;
    }
    auto seq = datasetTarget_->getSequence(currentSequenceIndex_);
    if (rawTextArea_) rawTextArea_->setText(seq.content);
    if (structuredTextArea_) structuredTextArea_->setText(seq.structured);
}

// =========================================================
// Sequence card helpers
// =========================================================

float UIDataHubPanel::sequenceCardHeight(const SequenceCard& card) const {
    float h = kSeqCardTopBarH;
    h += card.entries.size() * kSeqCardEntryH;
    if (card.formatIndex == 1) h += kSeqCardBtnRowH;
    h += kSeqCardBtnRowH;
    h += kSeqCardPadBot;
    return h;
}

void UIDataHubPanel::applyFormatTemplate(SequenceCard& card, int formatIndex) {
    card.formatIndex = formatIndex;
    card.entries.clear();

    auto makeField = [](const std::string& prefix) {
        UIDataHubPanel::SequenceCard::EntryField f;
        f.prefix = prefix;
        f.textArea = std::make_shared<UITextArea>(prefix, "", [](const std::string&) {});
        return f;
    };

    switch (formatIndex) {
        case 0: // Q/A
            card.entries.push_back(makeField("Q:"));
            card.entries.push_back(makeField("A:"));
            break;
        case 1: // Thought
            card.entries.push_back(makeField("T:"));
            break;
        case 2: // Conversation
            card.entries.push_back(makeField("Human:"));
            card.entries.push_back(makeField("Assistant:"));
            break;
        case 3: // Instruct
            card.entries.push_back(makeField("Instruction:"));
            card.entries.push_back(makeField("Response:"));
            break;
        case 4: // Raw
        default:
            card.entries.push_back(makeField("Content:"));
            break;
    }
}

UIDataHubPanel::SequenceCard UIDataHubPanel::buildSequenceCard(int formatIndex) {
    SequenceCard card;
    card.cardId      = nextSeqCardId_++;
    card.formatIndex = formatIndex;

    size_t id = card.cardId;

    std::vector<std::string> formats = {"Q/A", "Thought", "Conversation", "Instruct", "Raw"};
    card.formatDropdown = std::make_shared<UIDropdown>(
        "Format", formats, formatIndex,
        [this, id](int idx, const std::string&) {
            for (auto& c : sequenceCards_) {
                if (c.cardId == id) { applyFormatTemplate(c, idx); break; }
            }
        });
    card.formatDropdown->setMaxVisibleItems(5);

    card.generateBtn = std::make_shared<UIButton>("Generate", [this, id]() {
        generateForCard(id);
    });

    card.addEntryBtn = std::make_shared<UIButton>("+ Entry", [this, id]() {
        for (auto& c : sequenceCards_) {
            if (c.cardId == id) {
                SequenceCard::EntryField f;
                f.prefix = "T:";
                f.textArea = std::make_shared<UITextArea>("T:", "", [](const std::string&) {});
                c.entries.push_back(std::move(f));
                break;
            }
        }
    });

    card.deleteBtn = std::make_shared<UIButton>("x", [this, id]() {
        seqCardToDelete_ = static_cast<int>(id);
    });

    card.saveBtn = std::make_shared<UIButton>("Save", [this, id]() {
        saveSequenceCard(id);
    });

    applyFormatTemplate(card, formatIndex);
    return card;
}

void UIDataHubPanel::removeSequenceCard(size_t cardId) {
    sequenceCards_.erase(
        std::remove_if(sequenceCards_.begin(), sequenceCards_.end(),
            [cardId](const SequenceCard& c) { return c.cardId == cardId; }),
        sequenceCards_.end());
}

void UIDataHubPanel::generateForCard(size_t cardId) {
    SequenceCard* card = nullptr;
    for (auto& c : sequenceCards_) {
        if (c.cardId == cardId) { card = &c; break; }
    }
    if (!card || !structurer_) return;

    std::string raw = rawTextArea_ ? rawTextArea_->getText() : "";
    if (raw.empty()) { addLog("Put raw text in 'Raw Source' first", 1); return; }

    static const char* modes[] = {"qa", "thought", "conversation", "instruct", "raw"};
    std::string mode = (card->formatIndex >= 0 && card->formatIndex < 5)
                           ? modes[card->formatIndex] : "qa";

    std::string prompt = customPromptArea_ ? customPromptArea_->getText() : "";
    addLog("Generating " + mode + " format...", 0);

    auto results = structurer_->structureEntry(raw, mode, prompt);
    if (results.empty()) {
        std::string err = structurer_->lastError();
        addLog("Generation failed: " + (err.empty() ? "LLM returned no output" : err), 2);
        return;
    }

    auto makeField = [](const std::string& prefix, const std::string& text) {
        UIDataHubPanel::SequenceCard::EntryField f;
        f.prefix = prefix;
        f.textArea = std::make_shared<UITextArea>(prefix, text, [](const std::string&) {});
        return f;
    };

    if (mode == "qa" || mode == "instruct") {
        while (card->entries.size() < 2) {
            std::string pfx = card->entries.empty() ? "Q:" : "A:";
            card->entries.push_back(makeField(pfx, ""));
        }
        std::string result = results[0];
        size_t aPos = result.find("\n\nA: ");
        if (aPos != std::string::npos) {
            std::string q = result.substr(3, aPos - 3);
            std::string a = result.substr(aPos + 5);
            card->entries[0].textArea->setText(q);
            card->entries[1].textArea->setText(a);
        } else {
            card->entries[0].textArea->setText(result);
        }
    } else if (mode == "thought") {
        card->entries.clear();
        for (const auto& r : results) {
            std::string text = r;
            if (text.size() > 3 && text.substr(0, 3) == "T: ") text = text.substr(3);
            card->entries.push_back(makeField("T:", text));
        }
        if (card->entries.empty())
            card->entries.push_back(makeField("T:", ""));
    } else if (mode == "conversation") {
        card->entries.clear();
        if (!results.empty()) {
            std::istringstream ss(results[0]);
            std::string line;
            std::string accum;
            std::string curPrefix;
            while (std::getline(ss, line)) {
                if (line.substr(0, 7) == "Human: " || line.substr(0, 11) == "Assistant: ") {
                    if (!curPrefix.empty())
                        card->entries.push_back(makeField(curPrefix, accum));
                    if (line.substr(0, 7) == "Human: ") {
                        curPrefix = "Human:";
                        accum = line.substr(7);
                    } else {
                        curPrefix = "Assistant:";
                        accum = line.substr(11);
                    }
                } else if (!curPrefix.empty() && !line.empty()) {
                    accum += "\n" + line;
                }
            }
            if (!curPrefix.empty())
                card->entries.push_back(makeField(curPrefix, accum));
        }
        if (card->entries.empty()) {
            card->entries.push_back(makeField("Human:", ""));
            card->entries.push_back(makeField("Assistant:", ""));
        }
    } else {
        if (!card->entries.empty() && !results.empty()) {
            std::string combined;
            for (size_t i = 0; i < results.size(); ++i) {
                if (i > 0) combined += "\n\n";
                combined += results[i];
            }
            card->entries[0].textArea->setText(combined);
        }
    }

    addLog("Generated " + std::to_string(results.size()) + " entries in " + mode + " format", 0);
}

void UIDataHubPanel::saveSequenceCard(size_t cardId) {
    SequenceCard* card = nullptr;
    for (auto& c : sequenceCards_) {
        if (c.cardId == cardId) { card = &c; break; }
    }
    if (!card || !datasetTarget_) return;

    std::string structured;
    for (size_t i = 0; i < card->entries.size(); ++i) {
        if (i > 0) structured += "\n\n";
        structured += card->entries[i].prefix + " " + card->entries[i].textArea->getText();
    }

    if (structured.empty()) {
        addLog("Nothing to save — fill in the sequence fields", 1);
        return;
    }

    std::string rawContent;
    for (size_t i = 0; i < card->entries.size(); ++i) {
        if (i > 0) rawContent += "\n\n";
        rawContent += card->entries[i].textArea->getText();
    }

    if (datasetTarget_->appendStructuredEntry(rawContent, structured)) {
        totalSequences_ = datasetTarget_->massDatasetSize();
        structuredCount_++;
        addLog("Saved sequence card #" + std::to_string(card->cardId) + " to dataset", 0);
    } else {
        addLog("Failed to save sequence card", 2);
    }
}

// =========================================================
// Config persistence
// =========================================================

void UIDataHubPanel::loadUIConfig() {
    std::string path = GRIM::Config::getRequiredGrimTextPath(GRIM::Config::GrimTextPathKey::Checkpoints) + "/collection_state/ui_config.json";
    try {
        if (!std::filesystem::exists(path)) return;
        std::ifstream f(path);
        nlohmann::json cfg;
        f >> cfg;

        if (cfg.contains("fetchLimit"))             { fetchLimit_ = cfg["fetchLimit"]; }
        if (cfg.contains("verificationThreshold"))  { verificationThreshold_ = cfg["verificationThreshold"]; }
        if (cfg.contains("maxHFResults"))            { maxHFResults_ = cfg["maxHFResults"]; if (sliderMaxHFResults_) sliderMaxHFResults_->setValue(static_cast<float>(maxHFResults_)); }

        addLog("UI config loaded", 0);
    } catch (const std::exception& e) {
        addLog("Error loading config: " + std::string(e.what()), 1);
    }
}

void UIDataHubPanel::saveUIConfig() {
    std::string stateDir = GRIM::Config::getRequiredGrimTextPath(GRIM::Config::GrimTextPathKey::Checkpoints) + "/collection_state";
    std::string path = stateDir + "/ui_config.json";
    try {
        std::filesystem::create_directories(stateDir);
        nlohmann::json cfg;
        cfg["fetchLimit"]             = fetchLimit_;
        cfg["verificationThreshold"]  = verificationThreshold_;
        cfg["maxHFResults"]           = maxHFResults_;
        std::ofstream f(path);
        f << std::setw(2) << cfg << std::endl;
    } catch (const std::exception& e) {
        LOG_ERROR("DataHub", "Failed to save UI config: " + std::string(e.what()));
    }
}

void UIDataHubPanel::loadDownloadQueue() {
    std::string path = GRIM::Config::getRequiredGrimTextPath(GRIM::Config::GrimTextPathKey::Checkpoints) + "/collection_state/download_queue.json";
    try {
        if (!std::filesystem::exists(path)) return;
        std::ifstream f(path);
        nlohmann::json j;
        f >> j;
        std::lock_guard<std::mutex> lock(queueMutex_);
        downloadQueue_.clear();
        if (j.contains("queue") && j["queue"].is_array()) {
            for (const auto& item : j["queue"]) {
                QueuedDownload qd;
                qd.datasetId    = item.value("datasetId", "");
                qd.displayName  = item.value("displayName", "");
                qd.status       = item.value("status", "pending");
                qd.progress     = item.value("progress", 0.0f);
                qd.retryCount   = item.value("retryCount", 0);
                qd.errorMessage = item.value("errorMessage", "");
                if (qd.status == "downloading") { qd.status = "pending"; qd.progress = 0.0f; }
                if (qd.status == "pending" || qd.status == "failed")
                    downloadQueue_.push_back(qd);
            }
            addLog("Restored " + std::to_string(downloadQueue_.size()) + " queue items", 0);
        }
    } catch (const std::exception& e) {
        addLog("Error loading queue: " + std::string(e.what()), 1);
    }
}

void UIDataHubPanel::saveDownloadQueue() {
    std::string stateDir = GRIM::Config::getRequiredGrimTextPath(GRIM::Config::GrimTextPathKey::Checkpoints) + "/collection_state";
    std::string path = stateDir + "/download_queue.json";
    try {
        std::filesystem::create_directories(stateDir);
        nlohmann::json j;
        nlohmann::json arr = nlohmann::json::array();
        {
            std::lock_guard<std::mutex> lock(queueMutex_);
            for (const auto& item : downloadQueue_) {
                arr.push_back({
                    {"datasetId", item.datasetId}, {"displayName", item.displayName},
                    {"status", item.status}, {"progress", item.progress},
                    {"retryCount", item.retryCount}, {"errorMessage", item.errorMessage}
                });
            }
        }
        j["queue"] = arr;
        std::ofstream f(path);
        f << std::setw(2) << j << std::endl;
    } catch (const std::exception& e) {
        LOG_ERROR("DataHub", "Failed to save queue: " + std::string(e.what()));
    }
}

// =========================================================
// Curriculum tab layout
// =========================================================

void UIDataHubPanel::drawCurriculumTab(OverlayRenderer& renderer,
                                        const PanelRect& content) {
    float x     = content.origin.x + 15.0f;
    float y     = content.origin.y + 10.0f;
    float fullW = content.size.x - 30.0f;

    // ── Toolbar row 1: Model + Curriculum + Manage ────────
    float gap   = 8.0f;
    float rowH  = 36.0f;
    float btnW  = 80.0f;
    float menuW = 100.0f;

    // Proportional: two dropdowns share remaining space after menu
    float ddW = (fullW - menuW - 3.0f * gap) * 0.5f;
    if (ddW < 220.0f) ddW = 220.0f;

    float cx = x;
    cbModelDropdown_->setPosition(cx, y);
    cbModelDropdown_->setSize(ddW, rowH);
    cbModelDropdown_->drawOverlay(renderer, position);
    cx += ddW + gap;

    if (renamingCurriculum_ && cbCurriculumRenameInput_) {
        cbCurriculumRenameInput_->setPosition(cx, y);
        cbCurriculumRenameInput_->setSize(ddW, rowH);
        cbCurriculumRenameInput_->drawOverlay(renderer, position);
    } else {
        cbCurriculumDropdown_->setPosition(cx, y);
        cbCurriculumDropdown_->setSize(ddW, rowH);
        cbCurriculumDropdown_->drawOverlay(renderer, position);
    }
    cx += ddW + gap;

    curriculumActionMenu_->setPosition(cx, y + 4.0f);
    curriculumActionMenu_->setSize(menuW, rowH - 8.0f);
    curriculumActionMenu_->drawOverlay(renderer, position);
    cx += menuW + gap;

    // Format-mode badge: show whether active curriculum is Concept or PT
    if (!activeCurriculumId_.empty() && datasetTarget_) {
        auto activeCurr = datasetTarget_->getCurriculumById(activeCurriculumId_);
        if (!activeCurr.id.empty()) {
            const char* modeLabel = activeCurr.format_as_concept ? "Concept" : "PT";
            uint32_t badgeBg = activeCurr.format_as_concept
                ? UITheme::Colors::Success : 0xFF88AADD;
            float badgeW = activeCurr.format_as_concept ? 68.0f : 38.0f;
            float badgeH = rowH - 12.0f;
            float badgeY = y + 6.0f;
            renderer.drawRoundedRect({cx, badgeY}, {badgeW, badgeH}, badgeBg, 4.0f);
            renderer.drawText({cx + 6.0f, badgeY + 3.0f}, modeLabel, 0xFFFFFFFF);
        }
    }

    y += rowH + 6.0f;

    // ── Toolbar row 2: Curriculum filter + Format filter + Generate + Search ──
    float toggleW = 130.0f;
    cx = x;
    cbCurriculumFilterToggle_->setPosition(cx, y + 4.0f);
    cbCurriculumFilterToggle_->setSize(toggleW, rowH - 8.0f);
    cbCurriculumFilterToggle_->drawOverlay(renderer, position);
    cx += toggleW + gap;

    float filterDdW = 275.0f;
    cbTypeFilterDropdown_->setPosition(cx, y);
    cbTypeFilterDropdown_->setSize(filterDdW, rowH);
    cbTypeFilterDropdown_->drawOverlay(renderer, position);
    cx += filterDdW + gap;

    btnCBGenerate_->setPosition(cx, y + 4.0f);
    btnCBGenerate_->setSize(btnW, rowH - 8.0f);
    btnCBGenerate_->drawOverlay(renderer, position);
    cx += btnW + gap;

    float searchW = x + fullW - cx - 8.0f;
    if (searchW < 100.0f) searchW = 100.0f;
    cbSearchInput_->setPosition(cx, y + 4.0f);
    cbSearchInput_->setSize(searchW, rowH - 8.0f);
    cbSearchInput_->drawOverlay(renderer, position);

    y += rowH + 12.0f;

    // ── Split: list left (38%) | editor right (62%) ─────
    const CurriculumTabLayout layout = computeCurriculumTabLayout(content);
    y = layout.listY;
    float availH = layout.listH;
    float listW  = layout.listW;
    float editorX = layout.editorX;
    float editorW = layout.editorW;
    float bottomBarH = layout.bottomBarH;
    float statusBarH = layout.statusBarH;

    // Determine if active curriculum uses concept block formatting
    bool conceptMode = true; // default when no curriculum selected
    if (!activeCurriculumId_.empty() && datasetTarget_) {
        auto activeCurr = datasetTarget_->getCurriculumById(activeCurriculumId_);
        if (!activeCurr.id.empty())
            conceptMode = activeCurr.format_as_concept;
    }

    // ── ConceptBlock list ────────────────────────────────
    renderer.drawRoundedRect({x, y}, {listW, availH},
                             UITheme::Colors::Background, UITheme::Sizes::SmallRadius);

    // Header
    renderer.drawRoundedRect({x, y}, {listW, kPoolHeaderH},
                             UITheme::Colors::WidgetBg, UITheme::Sizes::SmallRadius);
    const float nameColW = listW * 0.34f;
    const float qColX    = x + 8.0f + nameColW;
    renderer.drawText({x + 8.0f, y + 6.0f}, "Name", UITheme::Colors::TextSecondary);
    renderer.drawText({qColX, y + 6.0f},
                      conceptMode ? "Question (preview)" : "Text (preview)",
                      UITheme::Colors::TextSecondary);
    renderer.drawText({x + listW - 118.0f, y + 6.0f}, "Type", UITheme::Colors::TextSecondary);

    float bodyY = y + kPoolHeaderH;
    float bodyH = availH - kPoolHeaderH;
    renderer.pushClipRect({x, bodyY}, {listW, bodyH});

    const int cbRows = cbCurriculumListRowCount();
    int startRow = static_cast<int>(cbListScrollOffset_ / kCBListRowH);
    float offsetY = bodyY - (cbListScrollOffset_ - startRow * kCBListRowH);
    for (int i = startRow; i < cbRows; ++i) {
        if (offsetY > bodyY + bodyH) break;
        if (offsetY + kCBListRowH < bodyY) {
            offsetY += kCBListRowH;
            continue;
        }

        std::string nameStr;
        std::string qRaw;
        std::string formatKey = "chain_of_thought";
        std::string blockId;
        bool isDraft = cbCurriculumRowIsDraft(i);

        if (isDraft) {
            nameStr = cbNameInput_ ? cbNameInput_->getText() : "";
            qRaw    = cbQuestionArea_ ? cbQuestionArea_->getText() : "";
            int pti = cbListTypeDropdown_ ? cbListTypeDropdown_->getSelectedIndex() : 1;
            if (pti >= 0 && pti < GRIM::kConceptPresetCount)
                formatKey = GRIM::kConceptPresets[pti].key;
            if (nameStr.empty())
                nameStr = "(new block — unsaved)";
        } else {
            size_t realIdx = 0;
            if (!cbCurriculumRowToBlockIndex(i, realIdx)) {
                offsetY += kCBListRowH;
                continue;
            }
            auto cb = datasetTarget_ ? datasetTarget_->getConceptBlock(realIdx) : GRIM::ConceptBlock{};
            nameStr   = cb.name;
            qRaw      = cb.question;
            formatKey = cb.format_type;
            blockId   = cb.id;
        }

        bool selected = (i == selectedCBRow_);
        bool hovered  = (i == hoveredCBRow_);
        uint32_t rowBg = selected ? UITheme::Colors::Primary
                       : hovered  ? UITheme::Colors::WidgetBgHover
                       : (i % 2 == 0) ? 0x00000000 : 0x08FFFFFF;

        if (rowBg != 0x00000000)
            renderer.drawRect({x, offsetY}, {listW, kCBListRowH}, rowBg);

        uint32_t textCol = selected ? 0xFFFFFFFF : UITheme::Colors::TextPrimary;
        std::string nameLine = cbSingleLinePreview(nameStr, 36);
        std::string qLine    = cbSingleLinePreview(qRaw, 48);
        if (qLine.empty())
            qLine = "—";

        renderer.drawText({x + 8.0f, offsetY + 4.0f}, nameLine, textCol);
        renderer.drawText({x + 8.0f, offsetY + 22.0f}, qLine,
                          selected ? 0xCCFFFFFF : UITheme::Colors::TextSecondary);

        if (!(selected && cbListTypeDropdown_)) {
            int pi = GRIM::presetIndexForKey(formatKey);
            std::string fmtLabel = (pi >= 0 && pi < GRIM::kConceptPresetCount)
                ? GRIM::kConceptPresets[pi].label : formatKey;
            if (fmtLabel.size() > 14)
                fmtLabel = fmtLabel.substr(0, 12) + "..";
            renderer.drawText({x + listW - 118.0f, offsetY + 14.0f}, fmtLabel,
                              selected ? 0xCCFFFFFF : UITheme::Colors::TextSecondary);
        }

        if (!isDraft && datasetTarget_ && !blockId.empty()
            && !activeCurriculumId_.empty()
            && datasetTarget_->isConceptBlockInCurriculum(blockId, activeCurriculumId_))
            renderer.drawRoundedRect({x + listW - 14.0f, offsetY + 10.0f}, {8.0f, 8.0f},
                                     UITheme::Colors::Success, 4.0f);

        offsetY += kCBListRowH;
    }
    renderer.popClipRect();

    layoutCBListTypeDropdownInList(x, y, listW);
    if (cbListTypeDropdown_)
        cbListTypeDropdown_->drawOverlay(renderer, position);

    // ── Editor panel ─────────────────────────────────────
    renderer.drawRoundedRect({editorX, y}, {editorW, availH},
                             UITheme::Colors::Background, 6.0f);

    float ePad = 16.0f;
    float eInnerW = editorW - 2.0f * ePad;
    float fieldH = 30.0f;
    float areaH  = 56.0f;
    float sectionGap = 6.0f;   // gap between section divider and next label
    float sectionPad = 6.0f;   // internal padding inside section boxes
    float sectionRad = 5.0f;   // corner radius for section backgrounds

    int presetIdx = cbListTypeDropdown_ ? cbListTypeDropdown_->getSelectedIndex() : 1;
    if (presetIdx < 0 || presetIdx >= GRIM::kConceptPresetCount) presetIdx = 1;
    const auto& preset = GRIM::kConceptPresets[presetIdx];

    // ── Scrollable editor content ───────────────────────
    renderer.pushClipRect({editorX, y}, {editorW, availH});

    float ey = y + 12.0f - cbEditorScrollOffset_;

    // ─── Name ───────────────────────────────────────────
    renderer.drawText({editorX + ePad, ey}, "Name", UITheme::Colors::TextSecondary);
    ey += 20.0f;
    cbNameInput_->setPosition(editorX + ePad, ey);
    cbNameInput_->setSize(eInnerW, fieldH);
    cbNameInput_->drawOverlay(renderer, position);
    ey += fieldH + 14.0f;

    // ─── Q: Question / Raw Text ───────────────────────────
    // Divider line
    renderer.drawRect({editorX + ePad, ey}, {eInnerW, 1.0f}, 0x18FFFFFF);
    ey += sectionGap;
    if (conceptMode) {
        std::string qLabel = std::string("Q: ") + preset.questionLabel;
        renderer.drawText({editorX + ePad, ey}, qLabel, UITheme::Colors::TextSecondary);
    } else {
        renderer.drawText({editorX + ePad, ey}, "Text", UITheme::Colors::TextSecondary);
    }
    ey += 20.0f;
    cbQuestionArea_->setPosition(editorX + ePad, ey);
    cbQuestionArea_->setSize(eInnerW, conceptMode ? areaH : areaH * 2.0f);
    cbQuestionArea_->drawOverlay(renderer, position);
    ey += (conceptMode ? areaH : areaH * 2.0f) + 16.0f;

    // ─── STATE0: Bootstrap Atoms (concept mode only) ────
    if (conceptMode) {
    renderer.drawRect({editorX + ePad, ey}, {eInnerW, 1.0f}, 0x18FFFFFF);
    ey += sectionGap;
    {
        float s0StartY = ey;
        ey += sectionPad;

        renderer.drawText({editorX + ePad + sectionPad, ey}, "STATE0  (bootstrap atoms)",
                          UITheme::Colors::TextSecondary);
        ey += 20.0f;
        float halfW = (eInnerW - 2.0f * sectionPad - 12.0f) * 0.5f;
        float innerLeft = editorX + ePad + sectionPad;
        renderer.drawText({innerLeft, ey}, "Type", UITheme::Colors::TextMuted);
        renderer.drawText({innerLeft + halfW + 12.0f, ey}, "Atoms (comma-separated)",
                          UITheme::Colors::TextMuted);
        ey += 16.0f;
        cbState0TypeInput_->setPosition(innerLeft, ey);
        cbState0TypeInput_->setSize(halfW, fieldH);
        cbState0TypeInput_->drawOverlay(renderer, position);
        cbState0AtomsInput_->setPosition(innerLeft + halfW + 12.0f, ey);
        cbState0AtomsInput_->setSize(halfW, fieldH);
        cbState0AtomsInput_->drawOverlay(renderer, position);
        ey += fieldH + sectionPad;

        // Section background (drawn behind)
        renderer.drawRoundedRect({editorX + ePad, s0StartY},
                                 {eInnerW, ey - s0StartY}, 0x0CFFFFFF, sectionRad);
    }
    ey += 16.0f;

    // ─── EXP: Explanation / Intermediates ────────────────
    if (preset.intermediatesLabel) {
        renderer.drawRect({editorX + ePad, ey}, {eInnerW, 1.0f}, 0x18FFFFFF);
        ey += sectionGap;
        {
            float expStartY = ey;
            ey += sectionPad;
            float innerLeft = editorX + ePad + sectionPad;

            std::string expLabel = std::string("EXP: ") + preset.intermediatesLabel;
            renderer.drawText({innerLeft, ey}, expLabel,
                              UITheme::Colors::TextSecondary);
            ey += 22.0f;

            float stepAreaH = 44.0f;
            float expInnerW = eInnerW - 2.0f * sectionPad;

            for (size_t si = 0; si < cbIntermediateAreas_.size(); ++si) {
                cbIntermediateAreas_[si]->setPosition(innerLeft, ey);
                cbIntermediateAreas_[si]->setSize(expInnerW, stepAreaH);
                cbIntermediateAreas_[si]->drawOverlay(renderer, position);
                ey += stepAreaH + 10.0f;
            }

            stepActionMenu_->setPosition(innerLeft, ey);
            stepActionMenu_->setSize(90.0f, 26.0f);
            stepActionMenu_->drawOverlay(renderer, position);
            ey += 32.0f + sectionPad;

            // Section background
            renderer.drawRoundedRect({editorX + ePad, expStartY},
                                     {eInnerW, ey - expStartY}, 0x0CFFFFFF, sectionRad);
        }
        ey += 16.0f;
    }

    // ─── EXEC: Execution Steps ──────────────────────────
    renderer.drawRect({editorX + ePad, ey}, {eInnerW, 1.0f}, 0x18FFFFFF);
    ey += sectionGap;
    {
        float execStartY = ey;
        ey += sectionPad;
        float innerLeft = editorX + ePad + sectionPad;
        float innerW = eInnerW - 2.0f * sectionPad;

        renderer.drawText({innerLeft, ey}, "EXEC Steps",
                          UITheme::Colors::TextSecondary);
        ey += 20.0f;

        float opW    = 80.0f;
        float slotsW = 90.0f;
        float colGap = 6.0f;
        float argsW  = (innerW - opW - slotsW - colGap * 3.0f) * 0.5f;
        float resW   = argsW;

        // Column sub-headers
        if (!cbExecStepRows_.empty()) {
            float hx = innerLeft;
            renderer.drawText({hx, ey}, "op", UITheme::Colors::TextMuted);
            hx += opW + colGap;
            renderer.drawText({hx, ey}, "arg_slots", UITheme::Colors::TextMuted);
            hx += slotsW + colGap;
            renderer.drawText({hx, ey}, "args", UITheme::Colors::TextMuted);
            hx += argsW + colGap;
            renderer.drawText({hx, ey}, "=> result", UITheme::Colors::TextMuted);
            ey += 16.0f;
        }

        for (size_t ei = 0; ei < cbExecStepRows_.size(); ++ei) {
            auto& row = cbExecStepRows_[ei];

            float cx = innerLeft;
            row.opDropdown->setPosition(cx, ey);
            row.opDropdown->setSize(opW, fieldH);
            row.opDropdown->drawOverlay(renderer, position);
            cx += opW + colGap;

            row.argSlotsInput->setPosition(cx, ey);
            row.argSlotsInput->setSize(slotsW, fieldH);
            row.argSlotsInput->drawOverlay(renderer, position);
            cx += slotsW + colGap;

            row.argsInput->setPosition(cx, ey);
            row.argsInput->setSize(argsW, fieldH);
            row.argsInput->drawOverlay(renderer, position);
            cx += argsW + colGap;

            row.resultInput->setPosition(cx, ey);
            row.resultInput->setSize(resW, fieldH);
            row.resultInput->drawOverlay(renderer, position);

            ey += fieldH + 8.0f;
        }

        // Exec step add/remove menu
        execStepActionMenu_->setPosition(innerLeft, ey);
        execStepActionMenu_->setSize(120.0f, 26.0f);
        execStepActionMenu_->drawOverlay(renderer, position);
        ey += 32.0f;

        // STATE1 derived result (inside the exec box)
        if (!cbExecStepRows_.empty()) {
            renderer.drawRect({innerLeft, ey}, {innerW, 1.0f}, 0x10FFFFFF);
            ey += 8.0f;
            std::string s1Label = "STATE1  result = ";
            auto& lastRow = cbExecStepRows_.back();
            std::string resText = lastRow.resultInput ? lastRow.resultInput->getText() : "?";
            s1Label += resText.empty() ? "?" : resText;
            renderer.drawText({innerLeft, ey}, s1Label, UITheme::Colors::Success);
            ey += 20.0f;
        }

        ey += sectionPad;

        // Section background
        renderer.drawRoundedRect({editorX + ePad, execStartY},
                                 {eInnerW, ey - execStartY}, 0x0CFFFFFF, sectionRad);
    }
    ey += 16.0f;

    } // end if (conceptMode) — STATE0 / EXP / EXEC hidden in PT mode

    // ─── A: Answer (concept mode only) ────────────────────
    if (conceptMode) {
    renderer.drawRect({editorX + ePad, ey}, {eInnerW, 1.0f}, 0x18FFFFFF);
    ey += sectionGap;
    std::string aLabel = std::string("A: ") + preset.answerLabel;
    renderer.drawText({editorX + ePad, ey}, aLabel,
                      UITheme::Colors::TextSecondary);
    ey += 20.0f;
    cbAnswerArea_->setPosition(editorX + ePad, ey);
    cbAnswerArea_->setSize(eInnerW, areaH);
    cbAnswerArea_->drawOverlay(renderer, position);
    ey += areaH + 18.0f;
    } // end if (conceptMode)

    // ─── Training Preview (read-only) ───────────────────
    renderer.drawRect({editorX + ePad, ey}, {eInnerW, 1.0f}, 0x18FFFFFF);
    ey += sectionGap;
    {
        renderer.drawText({editorX + ePad, ey},
                          conceptMode ? "Training Preview (canonical format)"
                                      : "Training Preview (raw)",
                          UITheme::Colors::TextSecondary);
        ey += 22.0f;

        // Build a ConceptBlock from current editor state for preview
        GRIM::ConceptBlock previewCB;
        previewCB.question = cbQuestionArea_ ? cbQuestionArea_->getText() : "";
        previewCB.answer   = (conceptMode && cbAnswerArea_) ? cbAnswerArea_->getText() : "";
        previewCB.state_0.type = cbState0TypeInput_ ? cbState0TypeInput_->getText() : "";
        if (conceptMode && cbState0AtomsInput_) {
            std::string atomsStr = cbState0AtomsInput_->getText();
            if (!atomsStr.empty()) {
                std::istringstream iss(atomsStr);
                std::string tok;
                while (std::getline(iss, tok, ',')) {
                    try { previewCB.state_0.atoms.push_back(std::stod(tok)); }
                    catch (...) {}
                }
            }
        }
        if (conceptMode) {
        for (const auto& row : cbExecStepRows_) {
            GRIM::ConceptExecutionStep step;
            static const char* opNames[] = {"add", "sub", "mul", "div"};
            int opIdx = row.opDropdown ? row.opDropdown->getSelectedIndex() : 0;
            step.op = (opIdx >= 0 && opIdx < 4) ? opNames[opIdx] : "add";
            if (row.argsInput) {
                std::istringstream iss(row.argsInput->getText());
                std::string tok;
                while (std::getline(iss, tok, ',')) {
                    try { step.args.push_back(std::stod(tok)); }
                    catch (...) {}
                }
            }
            if (row.resultInput) {
                try { step.result = std::stod(row.resultInput->getText()); }
                catch (...) { step.result = 0.0; }
            }
            previewCB.execution.push_back(std::move(step));
        }
        if (!previewCB.execution.empty()) {
            previewCB.state_1.result = previewCB.execution.back().result;
            previewCB.state_1.has_result = true;
        }
        for (const auto& area : cbIntermediateAreas_) {
            previewCB.explanation.push_back(area ? area->getText() : "");
        }
        } // end if (conceptMode)

        std::string preview = buildTrainingPreview(previewCB, conceptMode);

        // Pre-count lines so we can draw the background FIRST
        std::vector<std::pair<std::string, uint32_t>> previewLines;
        {
            std::istringstream pss(preview);
            std::string pline;
            while (std::getline(pss, pline)) {
                uint32_t lineCol = UITheme::Colors::TextMuted;
                if (conceptMode) {
                    if (pline.size() >= 2 && pline[0] == 'Q' && pline[1] == ':')
                        lineCol = UITheme::Colors::TextPrimary;
                    else if (pline.size() >= 2 && pline[0] == 'A' && pline[1] == ':')
                        lineCol = UITheme::Colors::TextPrimary;
                    else if (pline.size() >= 5 && pline.substr(0, 5) == "STATE")
                        lineCol = UITheme::Colors::Success;
                    else if (pline.size() >= 4 && pline.substr(0, 4) == "EXEC")
                        lineCol = 0xFF88CCFF;
                    else if (pline.size() >= 4 && pline.substr(0, 4) == "EXP:")
                        lineCol = UITheme::Colors::TextSecondary;
                }
                previewLines.push_back({pline, lineCol});
            }
        }

        // Draw background FIRST (correct z-order)
        float lineH = 18.0f;
        float previewH = static_cast<float>(previewLines.size()) * lineH + 16.0f;
        renderer.drawRoundedRect({editorX + ePad, ey - 4.0f},
                                 {eInnerW, previewH}, 0x14FFFFFF, 6.0f);

        // Then draw text on top
        float textY = ey + 4.0f;
        for (const auto& [text, col] : previewLines) {
            renderer.drawText({editorX + ePad + 10.0f, textY}, text, col);
            textY += lineH;
        }
        ey += previewH + 12.0f;
    }

    float totalEditorContentH = (ey + cbEditorScrollOffset_) - (y + 12.0f);
    renderer.popClipRect();

    // Handle editor scroll
    float maxEditorScroll = std::max(0.0f, totalEditorContentH - availH);
    cbEditorScrollOffset_ = std::clamp(cbEditorScrollOffset_, 0.0f, maxEditorScroll);

    // ── Bottom action bar ────────────────────────────────
    float barY = y + availH + 6.0f;
    float bx = x;
    float bGap = 8.0f;
    float bBtnW = 90.0f;
    float bBtnH = 28.0f;

    float blockMenuW = 90.0f;
    blockActionMenu_->setPosition(bx, barY + 4.0f);
    blockActionMenu_->setSize(blockMenuW, bBtnH);
    blockActionMenu_->drawOverlay(renderer, position);
    bx += blockMenuW + bGap;

    float menuW2 = 110.0f;
    blockCurriculumMenu_->setPosition(bx, barY + 4.0f);
    blockCurriculumMenu_->setSize(menuW2, bBtnH);
    blockCurriculumMenu_->drawOverlay(renderer, position);
    bx += menuW2;

    // Custom prompt (right side of action bar)
    float promptW = fullW - (bx + bGap - x) - 10.0f;
    if (promptW > 200.0f) {
        float promptX = bx + bBtnW + bGap + 10.0f;
        cbCustomPromptArea_->setPosition(promptX, barY);
        cbCustomPromptArea_->setSize(promptW, bottomBarH);
        cbCustomPromptArea_->drawOverlay(renderer, position);
    }

    // ── Status bar ───────────────────────────────────────
    float statusY = barY + bottomBarH + 4.0f;
    renderer.drawRoundedRect({x, statusY}, {fullW, statusBarH},
                             UITheme::Colors::Background, UITheme::Sizes::SmallRadius);
    std::string status = "Blocks: " + std::to_string(cbTotalCount_)
                       + "  |  Showing: " + std::to_string(filteredCBIndices_.size())
                       + "  |  In Curriculum: " + std::to_string(cbInCurrCount_);
    if (cbFormatFilterIdx_ > 0 && cbFormatFilterIdx_ <= GRIM::kConceptPresetCount)
        status += std::string("  |  Filter: ") + GRIM::kConceptPresets[cbFormatFilterIdx_ - 1].label;
    if (presetIdx >= 0 && presetIdx < GRIM::kConceptPresetCount)
        status += std::string("  |  Format: ") + GRIM::kConceptPresets[presetIdx].label;
    renderer.drawText({x + 10.0f, statusY + 5.0f}, status, UITheme::Colors::TextSecondary);
}

// =========================================================
// Curriculum tab helpers
// =========================================================

void UIDataHubPanel::refreshCurriculumTabState() {
    if (!datasetTarget_) return;
    datasetTarget_->loadConceptBlocks();
    cbTotalCount_ = datasetTarget_->conceptBlockCount();
    populateCBCurriculumDropdown();
    // Count blocks in active curriculum
    if (!activeCurriculumId_.empty()) {
        auto curr = datasetTarget_->getCurriculumById(activeCurriculumId_);
        cbInCurrCount_ = curr.id.empty() ? 0 : static_cast<int>(curr.concept_block_ids.size());
    } else {
        cbInCurrCount_ = 0;
    }
    cbFilterDirty_ = true;
    rebuildFilteredCBList();
    populateCBModelDropdown();
}

void UIDataHubPanel::rebuildFilteredCBList() {
    cbFilterDirty_ = false;
    filteredCBIndices_.clear();
    if (!datasetTarget_) return;

    std::string formatFilter;
    if (cbFormatFilterIdx_ > 0 && cbFormatFilterIdx_ <= GRIM::kConceptPresetCount) {
        // Index 0 = "All Types"; indices 1..N map to kConceptPresets[0..N-1]
        formatFilter = GRIM::kConceptPresets[cbFormatFilterIdx_ - 1].key;
    }
    filteredCBIndices_ = datasetTarget_->filterConceptBlocks(formatFilter, cbFilterSearch_);

    // Optional: restrict to blocks in the active curriculum
    if (cbCurriculumFilterActive_ && !activeCurriculumId_.empty()) {
        auto curr = datasetTarget_->getCurriculumById(activeCurriculumId_);
        if (!curr.id.empty()) {
            std::unordered_set<std::string> inCurr(
                curr.concept_block_ids.begin(), curr.concept_block_ids.end());
            std::vector<size_t> filtered;
            filtered.reserve(filteredCBIndices_.size());
            for (size_t idx : filteredCBIndices_) {
                auto cb = datasetTarget_->getConceptBlock(idx);
                if (inCurr.count(cb.id))
                    filtered.push_back(idx);
            }
            filteredCBIndices_ = std::move(filtered);
        }
    }

    cbTotalCount_ = datasetTarget_->conceptBlockCount();
}

void UIDataHubPanel::loadConceptBlockIntoEditor(size_t cbIndex) {
    if (!datasetTarget_) return;
    auto cb = datasetTarget_->getConceptBlock(cbIndex);
    if (cb.id.empty()) return;

    if (cbNameInput_)    cbNameInput_->setText(cb.name);
    if (cbQuestionArea_) cbQuestionArea_->setText(cb.question);
    if (cbAnswerArea_)   cbAnswerArea_->setText(cb.answer);

    int pi = GRIM::presetIndexForKey(cb.format_type);
    if (cbListTypeDropdown_ && pi >= 0) cbListTypeDropdown_->setSelectedIndex(pi);

    syncIntermediateAreas(static_cast<int>(cb.intermediates.size()));
    for (size_t i = 0; i < cb.intermediates.size() && i < cbIntermediateAreas_.size(); ++i) {
        cbIntermediateAreas_[i]->setText(cb.intermediates[i]);
    }

    // State 0
    if (cbState0TypeInput_)  cbState0TypeInput_->setText(cb.state_0.type);
    if (cbState0AtomsInput_) {
        std::ostringstream oss;
        for (size_t i = 0; i < cb.state_0.atoms.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << cb.state_0.atoms[i];
        }
        cbState0AtomsInput_->setText(oss.str());
    }

    // Execution steps
    syncExecStepRows(static_cast<int>(cb.execution.size()));
    for (size_t i = 0; i < cb.execution.size() && i < cbExecStepRows_.size(); ++i) {
        const auto& step = cb.execution[i];
        auto& row = cbExecStepRows_[i];

        // Map op string to dropdown index
        int opIdx = 0;
        if (step.op == "add") opIdx = 0;
        else if (step.op == "sub") opIdx = 1;
        else if (step.op == "mul") opIdx = 2;
        else if (step.op == "div") opIdx = 3;
        if (row.opDropdown) row.opDropdown->setSelectedIndex(opIdx);

        if (row.argSlotsInput) {
            std::ostringstream oss;
            for (size_t j = 0; j < step.arg_slots.size(); ++j) {
                if (j > 0) oss << ", ";
                oss << step.arg_slots[j];
            }
            row.argSlotsInput->setText(oss.str());
        }
        if (row.argsInput) {
            std::ostringstream oss;
            for (size_t j = 0; j < step.args.size(); ++j) {
                if (j > 0) oss << ", ";
                oss << step.args[j];
            }
            row.argsInput->setText(oss.str());
        }
        if (row.resultInput) row.resultInput->setText(std::to_string(step.result));
    }

    cbEditorScrollOffset_ = 0.0f;
}

void UIDataHubPanel::clearCBEditor() {
    if (cbNameInput_)    cbNameInput_->setText("");
    if (cbQuestionArea_) cbQuestionArea_->setText("");
    if (cbAnswerArea_)   cbAnswerArea_->setText("");
    cbIntermediateAreas_.clear();
    if (cbState0TypeInput_)   cbState0TypeInput_->setText("");
    if (cbState0AtomsInput_)  cbState0AtomsInput_->setText("");
    cbExecStepRows_.clear();
    cbEditorScrollOffset_ = 0.0f;
}

void UIDataHubPanel::syncIntermediateAreas(int count) {
    if (count < 0) count = 0;
    while (static_cast<int>(cbIntermediateAreas_.size()) < count) {
        auto area = std::make_shared<UITextArea>(
            "EXP " + std::to_string(cbIntermediateAreas_.size() + 1), "",
            [](const std::string&) {});
        cbIntermediateAreas_.push_back(area);
    }
    while (static_cast<int>(cbIntermediateAreas_.size()) > count) {
        cbIntermediateAreas_.pop_back();
    }
}

void UIDataHubPanel::syncExecStepRows(int count) {
    if (count < 0) count = 0;
    while (static_cast<int>(cbExecStepRows_.size()) < count) {
        CBExecStepRow row;
        row.opDropdown = std::make_shared<UIDropdown>(
            "", std::vector<std::string>{"add", "sub", "mul", "div"}, 0,
            [](int, const std::string&) {});
        row.opDropdown->setMaxVisibleItems(4);
        row.argSlotsInput = std::make_shared<UIInputBox>();
        row.argSlotsInput->setPlaceholder("slots: 0, 1");
        row.argsInput = std::make_shared<UIInputBox>();
        row.argsInput->setPlaceholder("args: 2.0, 3.0");
        row.resultInput = std::make_shared<UIInputBox>();
        row.resultInput->setPlaceholder("result");
        cbExecStepRows_.push_back(std::move(row));
    }
    while (static_cast<int>(cbExecStepRows_.size()) > count) {
        cbExecStepRows_.pop_back();
    }
}

std::string UIDataHubPanel::buildTrainingPreview(const GRIM::ConceptBlock& cb, bool conceptMode) const {
    std::ostringstream oss;

    if (conceptMode) {
        oss << "Q: " << cb.question << "\n";

        if (!cb.state_0.atoms.empty()) {
            oss << "STATE0 type=" << cb.state_0.type;
            for (double a : cb.state_0.atoms)
                oss << " " << a;
            oss << "\n";
        }

        for (const auto& exp : cb.explanation)
            oss << "EXP: " << exp << "\n";

        for (const auto& step : cb.execution) {
            oss << "EXEC " << step.op;
            for (double a : step.args)
                oss << " " << a;
            oss << " => " << step.result << "\n";
        }

        if (cb.state_1.has_result)
            oss << "STATE1 result=" << cb.state_1.result << "\n";

        oss << "A: " << cb.answer;
    } else {
        // Plain text / raw mode — no structural prefixes
        oss << cb.question;
        if (!cb.answer.empty())
            oss << "\n" << cb.answer;
    }

    return oss.str();
}

void UIDataHubPanel::generateConceptBlock() {
    if (!structurer_ || !cbQuestionArea_) return;
    std::string input = cbQuestionArea_->getText();
    if (input.empty()) {
        addLog("Enter a question/prompt to generate from", 1);
        return;
    }

    std::string customPrompt = cbCustomPromptArea_ ? cbCustomPromptArea_->getText() : "";
    auto results = structurer_->structureEntry(input, "concept_block", customPrompt);
    if (results.empty()) {
        std::string err = structurer_->lastError();
        addLog("Generation failed: " + (err.empty() ? "LLM returned no output" : err), 2);
        return;
    }

    std::vector<std::string> intermediates;
    std::string question;
    std::string answer;

    for (const auto& line : results) {
        if (line.size() > 3 && line.substr(0, 3) == "Q: ") {
            question = line.substr(3);
        } else if (line.size() > 3 && line.substr(0, 3) == "T: ") {
            intermediates.push_back(line.substr(3));
        } else if (line.size() > 3 && line.substr(0, 3) == "A: ") {
            answer = line.substr(3);
        }
    }

    if (!question.empty() && cbQuestionArea_)
        cbQuestionArea_->setText(question);

    syncIntermediateAreas(static_cast<int>(intermediates.size()));
    for (size_t i = 0; i < intermediates.size() && i < cbIntermediateAreas_.size(); ++i) {
        cbIntermediateAreas_[i]->setText(intermediates[i]);
    }

    if (!answer.empty() && cbAnswerArea_)
        cbAnswerArea_->setText(answer);

    addLog("Generated ConceptBlock with " + std::to_string(intermediates.size()) + " steps", 0);
}

int UIDataHubPanel::cbCurriculumListRowCount() const {
    return static_cast<int>(filteredCBIndices_.size()) + (cbDraftPreviewActive_ ? 1 : 0);
}

bool UIDataHubPanel::cbCurriculumRowIsDraft(int listRow) const {
    return cbDraftPreviewActive_ && listRow == 0;
}

bool UIDataHubPanel::cbCurriculumRowToBlockIndex(int listRow, size_t& outBlockIndex) const {
    if (listRow < 0) return false;
    if (cbCurriculumRowIsDraft(listRow)) return false;
    const int adj = cbDraftPreviewActive_ ? (listRow - 1) : listRow;
    if (adj < 0 || adj >= static_cast<int>(filteredCBIndices_.size())) return false;
    outBlockIndex = filteredCBIndices_[static_cast<size_t>(adj)];
    return true;
}

void UIDataHubPanel::syncCBListTypeDropdownFromToolbar() {
    // No-op: toolbar format dropdown removed; cbListTypeDropdown_ is now the single source of truth.
}

void UIDataHubPanel::layoutCBListTypeDropdownInList(float listX, float listY, float listW) {
    if (!cbListTypeDropdown_) return;
    if (selectedCBRow_ < 0 || selectedCBRow_ >= cbCurriculumListRowCount()) {
        cbListTypeDropdown_->setPosition(-2000.0f, -2000.0f);
        return;
    }
    const float bodyY  = listY + kPoolHeaderH;
    const float rowTop = bodyY + selectedCBRow_ * kCBListRowH - cbListScrollOffset_;
    const float typeColX = listX + listW - 128.0f;
    cbListTypeDropdown_->setPosition(typeColX - 150.0f, rowTop + 4.0f);
    cbListTypeDropdown_->setSize(290.0f, 36.0f);
}

void UIDataHubPanel::populateCBModelDropdown() {
    if (!cbModelDropdown_) return;
    namespace fs = std::filesystem;
    auto snapshot = GRIM::Config::loadAiConfigSnapshot();
    fs::path grimRoot = GRIM::Config::detail::resolveGrimRoot();
    fs::path modelStoreRoot;
    if (snapshot && snapshot->has_grim_paths && !snapshot->grim_text_model_store.empty())
        modelStoreRoot = fs::path(snapshot->grim_text_model_store);
    else
        modelStoreRoot = grimRoot / "resources" / "models" / "model_store";

    std::vector<std::string> names;
    try {
        if (fs::exists(modelStoreRoot)) {
            for (const auto& entry : fs::directory_iterator(modelStoreRoot)) {
                if (entry.is_directory())
                    names.push_back(entry.path().filename().string());
            }
        }
    } catch (...) {}

    if (names.empty()) names.push_back("(none)");
    cbModelDropdown_->setItems(names);

    if (datasetTarget_) {
        // Update curriculum count for active curriculum
        if (!activeCurriculumId_.empty()) {
            auto curr = datasetTarget_->getCurriculumById(activeCurriculumId_);
            cbInCurrCount_ = curr.id.empty() ? 0 : static_cast<int>(curr.concept_block_ids.size());
        } else {
            cbInCurrCount_ = 0;
        }
    }
}

// ── Curriculum dropdown helpers ─────────────────────────

void UIDataHubPanel::populateCBCurriculumDropdown() {
    if (!cbCurriculumDropdown_ || !datasetTarget_) return;
    const auto& curricula = datasetTarget_->getCurriculums();
    std::vector<std::string> names;
    names.push_back("(none)");
    int selectedIdx = 0;
    for (size_t i = 0; i < curricula.size(); ++i) {
        names.push_back(curricula[i].name);
        if (curricula[i].id == activeCurriculumId_)
            selectedIdx = static_cast<int>(i) + 1; // +1 for "(none)" entry
    }
    cbCurriculumDropdown_->setItems(names);
    cbCurriculumDropdown_->setSelectedIndex(selectedIdx);
}

void UIDataHubPanel::selectActiveCurriculum(int dropdownIndex) {
    if (!datasetTarget_) return;
    const auto& curricula = datasetTarget_->getCurriculums();
    if (dropdownIndex <= 0 || dropdownIndex > static_cast<int>(curricula.size())) {
        activeCurriculumId_.clear();
        cbInCurrCount_ = 0;
    } else {
        const auto& curr = curricula[dropdownIndex - 1]; // -1 for "(none)" entry
        activeCurriculumId_ = curr.id;
        cbInCurrCount_ = static_cast<int>(curr.concept_block_ids.size());
    }
    cbFilterDirty_ = true;
}

void UIDataHubPanel::loadHFTokenFromConfig() {
    try {
        std::string token = resolveHuggingFaceApiToken();
        if (token.empty()) return;
        hfTokenInput_->setText(token);
        hfTokenBuffer_ = token;
        if (hfWebhook_) hfWebhook_->setApiToken(token);
        addLog("HF token loaded (HF_TOKEN / HUGGINGFACE_HUB_TOKEN or ai_config)", 0);
    } catch (const std::exception& e) {
        addLog("Error loading HF token: " + std::string(e.what()), 1);
    }
}
