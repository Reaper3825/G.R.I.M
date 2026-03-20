// ui_data_hub.cpp — DataHub: tabbed data collection + structuring panel
// See ui_data_hub.hpp for class documentation.
//======================================================//

#include "ui_data_hub.hpp"
#include "overlay_renderer.hpp"
#include "ui_theme.hpp"
#include "ui_draw_helpers.hpp"
#include "logger.hpp"
#include "resources.hpp"
#include "control/ai_config_paths.hpp"
#include "core/input_parser.hpp"

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

// System-only: auto-detect fetcher from URL (matches web_collector logic)
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
    GRIM::Config::GrimTextPaths paths;
    if (GRIM::Config::loadGrimTextPaths(paths) && !paths.source_config.empty()) {
        std::filesystem::path p(paths.source_config);
        if (p.is_absolute())
            return paths.source_config;
        return (std::filesystem::path(getGrimRootDir()) / paths.source_config).string();
    }
    return (std::filesystem::path(getGrimRootDir()) / "DataCollection" / "source_data.json").string();
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

    homeWidgets_ = {
        btnFull_, btnCollect_, btnVerify_, btnMerge_,
        btnRebuild_, btnStop_, btnRefreshStats_
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
        [this](float v) { maxHFResults_ = static_cast<int>(v); });

    btnProcessQueue_ = std::make_shared<UIButton>("Process Queue", [this]() {
        processDownloadQueue();
    });
    btnClearQueue_ = std::make_shared<UIButton>("Clear Queue", [this]() {
        clearDownloadQueue();
    });

    hfWidgets_ = {
        hfSearchInput_, btnSearchHF_, btnBrowseHF_, hfCategoryDropdown_,
        hfTokenInput_, hfResultsScrollBox_, sliderMaxHFResults_,
        btnProcessQueue_, btnClearQueue_
    };

    // ── Structurer tab widgets ──────────────────────────

    modelDropdown_ = std::make_shared<UIDropdown>(
        "Model", std::vector<std::string>{"(none)"}, 0,
        [this](int, const std::string&) {
            LOG_DEBUG("DataHub", "Model selection changed");
        });

    formatDropdown_ = std::make_shared<UIDropdown>(
        "Format", std::vector<std::string>{"Q/A", "Conversation", "Instruct", "Raw"}, 0,
        [this](int, const std::string&) {});

    viewModeDropdown_ = std::make_shared<UIDropdown>(
        "View", std::vector<std::string>{"Dataset View", "Sequence View"}, 0,
        [this](int idx, const std::string&) { datasetViewMode_ = (idx == 0); });

    structSearchInput_ = std::make_shared<UIInputBox>();
    structSearchInput_->setPlaceholder("Search sequences...");

    searchPreviewScrollBox_ = std::make_shared<UIScrollBox>();
    searchPreviewScrollBox_->setChildSpacing(3.0f);

    rawTextArea_ = std::make_shared<UITextArea>("Raw Source", "",
        [](const std::string&) {});
    structuredTextArea_ = std::make_shared<UITextArea>("Structured Output", "",
        [](const std::string&) {});

    btnStructure_    = std::make_shared<UIButton>("Structure",     [this]() { LOG_DEBUG("DataHub", "Structure — not yet wired"); });
    btnStructureAll_ = std::make_shared<UIButton>("Structure All", [this]() { LOG_DEBUG("DataHub", "Structure All — not yet wired"); });
    btnSave_         = std::make_shared<UIButton>("Save",          [this]() { LOG_DEBUG("DataHub", "Save — not yet wired"); });
    btnPrevSeq_      = std::make_shared<UIButton>("<",  [this]() { if (currentSequenceIndex_ > 0) currentSequenceIndex_--; });
    btnNextSeq_      = std::make_shared<UIButton>(">",  [this]() { if (currentSequenceIndex_ + 1 < totalSequences_) currentSequenceIndex_++; });
    btnAssign_       = std::make_shared<UIButton>("Assign",  [this]() { LOG_DEBUG("DataHub", "Assign — not yet wired"); });
    btnRemoveAssign_ = std::make_shared<UIButton>("Remove",  [this]() { LOG_DEBUG("DataHub", "Remove — not yet wired"); });

    sliderMaxEntries_ = std::make_shared<UISlider>("Max Entries", 0.0f, 1000.0f, 0.0f, [](float) {});
    sliderParallel_   = std::make_shared<UISlider>("Parallel",    1.0f, 16.0f,   4.0f, [](float) {});

    structWidgets_ = {
        modelDropdown_, formatDropdown_, viewModeDropdown_,
        structSearchInput_, searchPreviewScrollBox_,
        rawTextArea_, structuredTextArea_,
        btnStructure_, btnStructureAll_, btnSave_,
        btnPrevSeq_, btnNextSeq_, btnAssign_, btnRemoveAssign_,
        sliderMaxEntries_, sliderParallel_
    };

    // ── Hide non-active groups ──────────────────────────

    for (auto& w : sourcesWidgets_) w->setVisible(false);
    for (auto& w : hfWidgets_)      w->setVisible(false);
    for (auto& w : structWidgets_)  w->setVisible(false);

    // ── Backend services ────────────────────────────────

    pipelineOrchestrator_ = std::make_unique<GRIM::Pipeline::PipelineOrchestrator>();
    hfWebhook_            = std::make_unique<GRIM::DataCollection::HuggingFaceWebhook>();

    // ── Load persisted state ────────────────────────────

    loadUIConfig();
    loadSourceCards();
    loadDownloadQueue();
    loadHFTokenFromConfig();
    updateDatasetStats();

    addLog("DataHub initialized", 0);
    LOG_DEBUG("DataHub", "Panel initialized — 4 tabs ready");
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
        case DataHubView::Home:        hideGroup(homeWidgets_);    break;
        case DataHubView::Sources:     hideGroup(sourcesWidgets_); break;
        case DataHubView::HuggingFace: hideGroup(hfWidgets_);      break;
        case DataHubView::Structurer:  hideGroup(structWidgets_);  break;
    }

    activeView_ = view;

    switch (activeView_) {
        case DataHubView::Home:        showGroup(homeWidgets_);    break;
        case DataHubView::Sources:     showGroup(sourcesWidgets_); break;
        case DataHubView::HuggingFace: showGroup(hfWidgets_);      break;
        case DataHubView::Structurer:  showGroup(structWidgets_);  break;
    }

    LOG_DEBUG("DataHub", "Switched to tab " + std::to_string(static_cast<int>(view)));
}

// =========================================================
// Update
// =========================================================

void UIDataHubPanel::update(const InputState& input, float dt) {
    if (!isVisible()) return;
    UIPanel::update(input, dt);

    // Tab buttons (always active)
    float tabX = position.x + 10.0f;
    tabHomeBtn_->setPosition(tabX, position.y + kTabBarY);
    tabSourcesBtn_->setPosition(tabX + 95.0f, position.y + kTabBarY);
    tabHFBtn_->setPosition(tabX + 190.0f, position.y + kTabBarY);
    tabStructBtn_->setPosition(tabX + 305.0f, position.y + kTabBarY);

    tabHomeBtn_->update(input, dt);
    tabSourcesBtn_->update(input, dt);
    tabHFBtn_->update(input, dt);
    tabStructBtn_->update(input, dt);

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
            btnProcessQueue_->update(input, dt);
            btnClearQueue_->update(input, dt);
            break;

        case DataHubView::Structurer:
            for (auto& w : structWidgets_) w->update(input, dt);
            break;
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

    tabHomeBtn_->drawOverlay(renderer, position);
    tabSourcesBtn_->drawOverlay(renderer, position);
    tabHFBtn_->drawOverlay(renderer, position);
    tabStructBtn_->drawOverlay(renderer, position);

    // Active tab indicator (2px underline)
    float indicatorX = tabX;
    float indicatorW = 90.0f;
    switch (activeView_) {
        case DataHubView::Home:        indicatorX = tabX;           indicatorW = 90.0f;  break;
        case DataHubView::Sources:     indicatorX = tabX + 95.0f;   indicatorW = 90.0f;  break;
        case DataHubView::HuggingFace: indicatorX = tabX + 190.0f;  indicatorW = 110.0f; break;
        case DataHubView::Structurer:  indicatorX = tabX + 305.0f;  indicatorW = 100.0f; break;
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
    static constexpr int   kBtnCount    = 7;

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
            &btnRebuild_, &btnStop_, &btnRefreshStats_
        };

        for (int i = 0; i < kBtnCount; i++) {
            float bx = startX + i * (kBtnW + kBtnGap);
            (*btns[i])->setPosition(bx, y);
            (*btns[i])->setSize(kBtnW, kBtnH);
            (*btns[i])->drawOverlay(renderer, position);
        }

        y += kBtnH + 12.0f;
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
        resultsH = std::min(180.0f, static_cast<float>(maxHFResults_) * 75.0f);
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

    // Queue control buttons
    float qBtnW = 140.0f;
    float qBtnH = UITheme::Sizes::ButtonHeight;

    btnProcessQueue_->setText(queueProcessing_.load() ? "Processing..." : "Process Queue");
    btnProcessQueue_->setPosition(x, y);
    btnProcessQueue_->setSize(qBtnW, qBtnH);
    btnProcessQueue_->drawOverlay(renderer, position);

    btnClearQueue_->setPosition(x + qBtnW + 10.0f, y);
    btnClearQueue_->setSize(qBtnW, qBtnH);
    btnClearQueue_->drawOverlay(renderer, position);

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
    // Dropdowns: UIDropdown uses 150px for label + (size.x - 160) for box. Need size.x > 160.
    float ddW = 260.0f;   // ~100px for dropdown box after 150px label
    float sliderW = 280.0f;  // ~120px for slider bar after 150px label (UISlider uses same layout)
    float gap = 14.0f;
    float rowH = 36.0f;   // Enough for dropdown box (30px) + padding

    modelDropdown_->setPosition(x, y);
    modelDropdown_->setSize(ddW, rowH);
    modelDropdown_->drawOverlay(renderer, position);

    formatDropdown_->setPosition(x + ddW + gap, y);
    formatDropdown_->setSize(ddW, rowH);
    formatDropdown_->drawOverlay(renderer, position);

    viewModeDropdown_->setPosition(x + (ddW + gap) * 2.0f, y);
    viewModeDropdown_->setSize(ddW, rowH);
    viewModeDropdown_->drawOverlay(renderer, position);

    // Action buttons right-aligned in top right
    float rightX = x + fullW;
    float structW = 100.0f, structAllW = 110.0f, saveW = 100.0f, assignW = 80.0f, removeW = 80.0f;
    float btnGap = 8.0f;

    btnStructureAll_->setPosition(rightX - structAllW, y);
    btnStructureAll_->setSize(structAllW, rowH);
    btnStructureAll_->drawOverlay(renderer, position);

    btnStructure_->setPosition(rightX - structAllW - btnGap - structW, y);
    btnStructure_->setSize(structW, rowH);
    btnStructure_->drawOverlay(renderer, position);
    y += rowH + kStructRowGap;

    // ── Toolbar row 2 ───────────────────────────────────
    // Sliders need more width for the bar; UISlider uses 150px label + (size.x - 160) for bar

    sliderMaxEntries_->setPosition(x, y);
    sliderMaxEntries_->setSize(sliderW, rowH);
    sliderMaxEntries_->drawOverlay(renderer, position);

    sliderParallel_->setPosition(x + sliderW + gap, y);
    sliderParallel_->setSize(sliderW, rowH);
    sliderParallel_->drawOverlay(renderer, position);

    // Save, Assign, Remove right-aligned
    btnRemoveAssign_->setPosition(rightX - removeW, y);
    btnRemoveAssign_->setSize(removeW, rowH);
    btnRemoveAssign_->drawOverlay(renderer, position);

    btnAssign_->setPosition(rightX - removeW - btnGap - assignW, y);
    btnAssign_->setSize(assignW, rowH);
    btnAssign_->drawOverlay(renderer, position);

    btnSave_->setPosition(rightX - removeW - btnGap - assignW - btnGap - saveW, y);
    btnSave_->setSize(saveW, rowH);
    btnSave_->drawOverlay(renderer, position);
    y += rowH + kStructRowGap;

    // ── Search bar ──────────────────────────────────────

    float searchH = 28.0f;
    structSearchInput_->setPosition(x, y);
    structSearchInput_->setSize(fullW * 0.4f, searchH);
    structSearchInput_->drawOverlay(renderer, position);

    if (!datasetViewMode_) {
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
    }
    y += searchH + kStructSectionGap;

    UIDrawHelpers::drawDivider(renderer, {x, y}, fullW);
    y += kStructLabelSpace;  // Space for "Raw Source" / "Structured Output" labels above text areas

    // ── Dual text areas ─────────────────────────────────

    float areaH = content.origin.y + content.size.y - y - 40.0f;
    if (areaH < 100.0f) areaH = 100.0f;
    float halfW = (fullW - 10.0f) / 2.0f;

    rawTextArea_->setPosition(x, y);
    rawTextArea_->setSize(halfW, areaH);
    rawTextArea_->drawOverlay(renderer, position);

    structuredTextArea_->setPosition(x + halfW + 10.0f, y);
    structuredTextArea_->setSize(halfW, areaH);
    structuredTextArea_->drawOverlay(renderer, position);
    y += areaH + 5.0f;

    // ── Status bar ──────────────────────────────────────

    renderer.drawRoundedRect({x, y}, {fullW, 25.0f},
                             UITheme::Colors::Background, UITheme::Sizes::SmallRadius);

    std::string status = "Total: " + std::to_string(totalSequences_)
                       + "  |  Assigned: " + std::to_string(assignedSequences_)
                       + "  |  Structured: " + std::to_string(structuredCount_)
                       + "  |  Failed: " + std::to_string(failedCount_);
    renderer.drawText({x + 10.0f, y + 5.0f}, status, UITheme::Colors::TextSecondary);
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
    GRIM::Config::GrimTextPaths paths;
    if (!GRIM::Config::loadGrimTextPaths(paths)) {
        datasetSizeInfo_ = "Dataset: Config error";
        hudFileSize_ = "N/A";
        return;
    }

    try {
        if (std::filesystem::exists(paths.training_data)) {
            auto sz = std::filesystem::file_size(paths.training_data);
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

        std::string ckptDir = getResourcePath() + "/models/GRIM-text/training/checkpoints";
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
        [this](float) { sourcesDirty_ = true; });

    card.depthSlider = std::make_shared<UISlider>(
        "Crawl Depth", 1.0f, 5.0f, static_cast<float>(depth),
        [this](float) { sourcesDirty_ = true; });

    card.limitSlider = std::make_shared<UISlider>(
        "Fetch Limit", 0.0f, 5000.0f, static_cast<float>(limit),
        [this](float) { sourcesDirty_ = true; });

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

    std::string query = hfSearchBuffer_;
    std::string token = hfTokenBuffer_;

    std::thread([this, query, token]() {
        try {
            if (!token.empty() && hfWebhook_) hfWebhook_->setApiToken(token);
            auto results = hfWebhook_->searchDatasets(query, 20, "task_categories:text-generation");
            hfSearchResults_ = std::move(results);
            if (hfSearchResults_.empty()) {
                lastSearchError_ = "No datasets found for: " + query;
                addLog("No results for: " + query, 1);
            } else {
                lastSearchError_.clear();
                addLog("Found " + std::to_string(hfSearchResults_.size()) + " datasets", 0);
            }
        } catch (const std::exception& e) {
            lastSearchError_ = "Search failed: " + std::string(e.what());
            addLog("HF search error: " + std::string(e.what()), 2);
            hfSearchResults_.clear();
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

    std::string token = hfTokenBuffer_;

    std::thread([this, category, token]() {
        try {
            if (!token.empty() && hfWebhook_) hfWebhook_->setApiToken(token);
            auto results = hfWebhook_->searchDatasets("", 20, "task_categories:" + category);
            hfSearchResults_ = std::move(results);
            if (hfSearchResults_.empty()) {
                lastSearchError_ = "No datasets in category: " + category;
            } else {
                lastSearchError_.clear();
                addLog("Found " + std::to_string(hfSearchResults_.size()) + " datasets", 0);
            }
        } catch (const std::exception& e) {
            lastSearchError_ = "Category search failed: " + std::string(e.what());
            hfSearchResults_.clear();
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

    std::string token = hfTokenBuffer_;

    std::thread([this, token]() {
        try {
            if (!token.empty() && hfWebhook_) hfWebhook_->setApiToken(token);
            auto results = hfWebhook_->searchDatasets("", 20, "task_categories:text-generation");
            hfSearchResults_ = std::move(results);
            if (hfSearchResults_.empty()) {
                lastSearchError_ = "Could not load popular datasets";
            } else {
                lastSearchError_.clear();
                addLog("Loaded " + std::to_string(hfSearchResults_.size()) + " popular datasets", 0);
            }
        } catch (const std::exception& e) {
            lastSearchError_ = "Browse failed: " + std::string(e.what());
            hfSearchResults_.clear();
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
        std::string label = ds.name + "\n " + ds.author
            + "\n Downloads: " + std::to_string(ds.downloads / 1000) + "k | Likes: " + std::to_string(ds.likes);
        auto btn = std::make_shared<UIButton>(label,
            [this, id = ds.id, name = ds.name]() { addToDownloadQueue(id, name); });
        btn->setSize(containerWidth - 10.0f, 70.0f);
        hfResultsScrollBox_->addChild(btn);
    }
    hfResultsScrollBox_->autoLayoutChildren();
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
// Config persistence
// =========================================================

void UIDataHubPanel::loadUIConfig() {
    std::string path = GRIM::Config::getCheckpointDir() + "/collection_state/ui_config.json";
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
    std::string stateDir = GRIM::Config::getCheckpointDir() + "/collection_state";
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
    std::string path = GRIM::Config::getCheckpointDir() + "/collection_state/download_queue.json";
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
    std::string stateDir = GRIM::Config::getCheckpointDir() + "/collection_state";
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

void UIDataHubPanel::loadHFTokenFromConfig() {
    try {
        if (!std::filesystem::exists("ai_config.json")) return;
        std::ifstream f("ai_config.json");
        nlohmann::json cfg;
        f >> cfg;
        if (cfg.contains("api_keys") && cfg["api_keys"].contains("huggingface")) {
            std::string token = cfg["api_keys"]["huggingface"];
            if (!token.empty()) {
                hfTokenInput_->setText(token);
                hfTokenBuffer_ = token;
                if (hfWebhook_) hfWebhook_->setApiToken(token);
                addLog("HF token loaded from config", 0);
            }
        }
    } catch (const std::exception& e) {
        addLog("Error loading HF token: " + std::string(e.what()), 1);
    }
}
