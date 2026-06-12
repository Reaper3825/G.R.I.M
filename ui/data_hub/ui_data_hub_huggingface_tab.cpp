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
