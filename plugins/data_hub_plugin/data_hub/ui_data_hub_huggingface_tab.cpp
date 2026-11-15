#include "ui_data_hub_internal.hpp"

using namespace UIDataHubDetail;

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

// =========================================================
// HuggingFace tab actions
// =========================================================


void UIDataHubPanel::searchHuggingFaceDatasets() {
    if (hfSearchBuffer_.empty()) { addLog("Enter a search query", 1); return; }
    searchHuggingFaceByCategory("");
}

void UIDataHubPanel::searchHuggingFaceByCategory(const std::string& category) {
    if (category == "All Categories") return;
    try {
        DataHubJobs::QueryRequest request;
        request.kind = DataHubJobs::QueryKind::Search;
        request.query = hfSearchBuffer_;
        if (!category.empty()) { request.query.clear(); request.filter = "task_categories:" + category; }
        else request.filter = "task_categories:text-generation";
        request.token = hfTokenBuffer_;
        const auto id = DataHubJobs::submit(std::move(request));
        if (searchRequest_) DataHubJobs::release(searchRequest_);
        if (previewRequest_) { DataHubJobs::release(previewRequest_); previewRequest_ = 0; }
        searchRequest_ = id;
        hfSearching_.store(true); hfSearchResults_.clear(); lastSearchError_.clear();
        hfSelectedResultIndex_ = -1; hfPreviewDatasetId_.clear(); hfPreviewDisplayName_.clear();
        hfPreviewArea_->setText("Searching...");
    } catch (const std::exception& error) { addLog(error.what(), 2); }
}

void UIDataHubPanel::browseHuggingFaceDatasets() {
    const auto query = hfSearchBuffer_;
    hfSearchBuffer_.clear();
    searchHuggingFaceByCategory("");
    hfSearchBuffer_ = query;
}

void UIDataHubPanel::populateHFResults(float containerWidth) {
    hfResultsScrollBox_->clearChildren();
    for (size_t i = 0; i < hfSearchResults_.size(); ++i) {
        const auto& item = hfSearchResults_[i];
        std::string name = item.name;
        if (name.empty()) name = item.id;
        std::string label = "  " + name + "\n " + item.author + "\n Downloads: " +
            std::to_string(item.downloads / 1000) + "k | Likes: " + std::to_string(item.likes);
        if (hfSelectedResultIndex_ == static_cast<int>(i)) label.replace(0, 2, "> ");
        auto button = std::make_shared<UIButton>(label, [this, i] { selectHuggingFaceResult(i); });
        button->setSize(containerWidth - 10.0f, 70.0f);
        hfResultsScrollBox_->addChild(button);
    }
    hfResultsScrollBox_->autoLayoutChildren();
}

void UIDataHubPanel::selectHuggingFaceResult(size_t index) {
    if (index >= hfSearchResults_.size()) throw std::out_of_range("DataHub HF result no longer exists");
    try {
        const auto& item = hfSearchResults_[index];
        DataHubJobs::QueryRequest request;
        request.kind = DataHubJobs::QueryKind::Preview;
        request.query = item.id; request.token = hfTokenBuffer_;
        const auto id = DataHubJobs::submit(std::move(request));
        if (previewRequest_) DataHubJobs::release(previewRequest_);
        previewRequest_ = id;
        hfSelectedResultIndex_ = static_cast<int>(index);
        hfPreviewDatasetId_ = item.id; hfPreviewDisplayName_ = item.name;
        if (hfPreviewDisplayName_.empty()) hfPreviewDisplayName_ = item.id;
        hfResultsNeedsPopulate_.store(true);
        hfPreviewArea_->setText("Loading sample rows...");
    } catch (const std::exception& error) { addLog(error.what(), 2); }
}

void UIDataHubPanel::addToDownloadQueue(const std::string& id, const std::string& name) {
    try { DataHubJobs::enqueueDownload(id, name); loadDownloadQueue(); }
    catch (const std::exception& error) { addLog(error.what(), 2); }
}
void UIDataHubPanel::processDownloadQueue() {
    try { DataHubJobs::processDownloads(hfTokenBuffer_); loadDownloadQueue(); }
    catch (const std::exception& error) { addLog(error.what(), 2); }
}
void UIDataHubPanel::clearDownloadQueue() {
    try { DataHubJobs::clearDownloads(); loadDownloadQueue(); }
    catch (const std::exception& error) { addLog(error.what(), 2); }
}
void UIDataHubPanel::removeFromQueue(int index) {
    if (index < 0 || static_cast<size_t>(index) >= downloadQueue_.size())
        throw std::out_of_range("DataHub queue item no longer exists");
    try { DataHubJobs::removeDownload(downloadQueue_[index].datasetId); loadDownloadQueue(); }
    catch (const std::exception& error) { addLog(error.what(), 2); }
}
void UIDataHubPanel::retryQueueItem(int index) {
    if (index < 0 || static_cast<size_t>(index) >= downloadQueue_.size())
        throw std::out_of_range("DataHub queue item no longer exists");
    try { DataHubJobs::retryDownload(downloadQueue_[index].datasetId); loadDownloadQueue(); }
    catch (const std::exception& error) { addLog(error.what(), 2); }
}
void UIDataHubPanel::clearCompletedFromQueue() {
    try { DataHubJobs::clearCompletedDownloads(); loadDownloadQueue(); }
    catch (const std::exception& error) { addLog(error.what(), 2); }
}
void UIDataHubPanel::updateQueueInteraction(const InputState& input, float x, float y, float width, float height) {
    hoveredQueueItem_ = -1;
    if (input.mousePos.x < x || input.mousePos.x > x + width ||
        input.mousePos.y < y || input.mousePos.y > y + height) return;
    if (input.mousePressed[1]) { clearCompletedFromQueue(); return; }
    const float rowHeight = UITheme::Sizes::WidgetHeight;
    for (size_t i = 0; i < downloadQueue_.size(); ++i) {
        const float top = y + 5.0f + static_cast<float>(i) * rowHeight;
        if (input.mousePos.y < top || input.mousePos.y > top + rowHeight) continue;
        hoveredQueueItem_ = static_cast<int>(i);
        if (input.mousePressed[0] && input.mousePos.x >= x + width - 25.0f &&
            input.mousePos.x <= x + width - 7.0f) removeFromQueue(static_cast<int>(i));
        break;
    }
}
