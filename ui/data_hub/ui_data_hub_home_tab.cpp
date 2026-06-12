#include "ui_data_hub_internal.hpp"

using namespace UIDataHubDetail;

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
    const std::filesystem::path trainingDataPath = resolvePathFromGrimRoot(
        aiConfig.at("paths").at("grim_text").at("training_data").get<std::string>());
    const std::filesystem::path checkpointDir = resolvePathFromGrimRoot(
        aiConfig.at("paths").at("grim_text").at("checkpoints").get<std::string>());

    try {
        if (std::filesystem::exists(trainingDataPath)) {
            auto sz = std::filesystem::file_size(trainingDataPath);
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

        if (std::filesystem::exists(checkpointDir)) {
            int count = 0;
            for (const auto& e : std::filesystem::directory_iterator(checkpointDir))
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

    const std::string directoryCollectionPath = resolvePathFromGrimRoot(
        aiConfig.at("paths").at("grim_text").at("directory_collection").get<std::string>()).string();

    dirPathInput_->setText(directoryCollectionPath);
    dirScanPath_ = directoryCollectionPath;
    dirNeedsScan_ = true;
    addLog("Loaded directory collection path from ai_config.json", 0);
}

void UIDataHubPanel::scanDirectory() {
    dirFileEntries_.clear();
    dirScrollOffset_ = 0.0f;

    if (dirScanPath_.empty()) return;

    std::error_code ec;
    const auto files = listFiles(dirScanPath_, ec);
    if (ec == std::errc::no_such_file_or_directory) {
        addLog("Directory collection path does not exist: " + dirScanPath_, 2);
        return;
    }
    if (ec == std::errc::not_a_directory) {
        addLog("Directory collection path is not a directory: " + dirScanPath_, 2);
        return;
    }
    if (ec) {
        addLog("Failed to scan directory path: " + dirScanPath_ + " (" + ec.message() + ")", 2);
        return;
    }

    dirFileEntries_.reserve(files.size());
    for (const auto& filePath : files) {
        DirFileEntry fe;
        fe.path        = filePath.string();
        fe.filename    = filePath.filename().string();
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

        fs::path filePath = it->path;
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
