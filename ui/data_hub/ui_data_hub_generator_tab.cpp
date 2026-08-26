#include "ui_data_hub_internal.hpp"

using namespace UIDataHubDetail;
using namespace GRIM::DataCollection;

namespace {

uint64_t parseUnsignedOr(const std::string& text, uint64_t fallback) {
    try {
        size_t used = 0;
        const uint64_t value = std::stoull(text, &used);
        return used == text.size() ? value : fallback;
    } catch (...) {
        return fallback;
    }
}

std::string generatorResourceRelativePath(const char* filename) {
    return std::string("models/GRIM-text/training/data/") + filename;
}

std::filesystem::path generatorResourcePath(const char* filename) {
    return std::filesystem::path(getResourcePath())
        / "models" / "GRIM-text" / "training" / "data" / filename;
}

GRIM::DataCollection::GeneratorAdapterOptions generatorAdapterOptions(uint64_t seed) {
    GRIM::DataCollection::GeneratorAdapterOptions options;
    options.seed = seed;
    if (aiConfig.contains("ollama_options") && aiConfig["ollama_options"].is_object()) {
        const auto& ollama = aiConfig["ollama_options"];
        options.temperature = ollama.value("temperature", 0.2f);
        options.topP = ollama.value("top_p", 0.9f);
        options.topK = ollama.value("top_k", 40);
        options.maxTokens = ollama.value("num_predict", 2048);
    }
    if (aiConfig.contains("data_collection")
        && aiConfig["data_collection"].contains("data_structuring")) {
        options.timeoutMs = aiConfig["data_collection"]["data_structuring"]
            .value("timeout_ms", 60000);
    }
    // Slot filling favors stable structure even when the global chat setting
    // uses a more creative temperature.
    options.temperature = std::min(options.temperature, 0.3f);
    options.maxTokens = std::max(options.maxTokens, 1024);
    return options;
}

} // namespace

void UIDataHubPanel::refreshGeneratorState() {
    if (!conceptGenerator_) return;

    const auto catalogPath = generatorResourcePath("LexiconPieces.json");
    const auto sourcePath = generatorResourcePath("GeneratorDataSource.json");
    if (conceptGenerator_->loadCatalog(catalogPath)) {
        generatorCatalogStatus_ = std::to_string(conceptGenerator_->groups().size())
            + " language groups  |  "
            + std::to_string(conceptGenerator_->frames().size()) + " recipe";
        if (conceptGenerator_->frames().size() != 1) generatorCatalogStatus_ += "s";

        std::vector<std::string> frameLabels;
        for (const auto& frame : conceptGenerator_->frames()) frameLabels.push_back(frame.label);
        if (genFrameDropdown_) genFrameDropdown_->setItems(frameLabels);
        rebuildGeneratorGroupViews();
    } else {
        generatorCatalogStatus_ = conceptGenerator_->lastError();
        if (genPreviewArea_) genPreviewArea_->setText(generatorCatalogStatus_);
        addLog("Generator catalog failed to load: " + conceptGenerator_->lastError(), 2);
        return;
    }

    if (conceptGenerator_->loadDataSource(sourcePath)) {
        const size_t count = conceptGenerator_->documents().size();
        generatorSourceStatus_ = count == 0
            ? "0 documents — paste one below"
            : std::to_string(count) + " raw document" + (count == 1 ? " loaded" : "s loaded");
    } else {
        generatorSourceStatus_ = conceptGenerator_->lastError();
        addLog("Generator data source failed to load: " + conceptGenerator_->lastError(), 2);
    }
    if (generatorAdapter_) generatorAdapterStatus_ = generatorAdapter_->adapterId();
}

void UIDataHubPanel::rebuildGeneratorGroupViews() {
    for (const auto& view : generatorGroupViews_) {
        if (!view.toggleButton) continue;
        generatorWidgets_.erase(
            std::remove(generatorWidgets_.begin(), generatorWidgets_.end(), view.toggleButton),
            generatorWidgets_.end());
    }
    generatorGroupViews_.clear();
    generatorLexiconScroll_ = 0.0f;

    if (!conceptGenerator_) return;
    const auto& groups = conceptGenerator_->groups();
    generatorGroupViews_.reserve(groups.size());
    for (size_t index = 0; index < groups.size(); ++index) {
        GeneratorGroupView view;
        view.expanded = index == 0;
        view.toggleButton = std::make_shared<UIButton>(
            std::string(view.expanded ? "[-] " : "[+] ") + groups[index].label
                + " (" + std::to_string(groups[index].entries.size()) + ")",
            [this, index]() {
                if (index >= generatorGroupViews_.size() || !conceptGenerator_) return;
                auto& groupView = generatorGroupViews_[index];
                groupView.expanded = !groupView.expanded;
                const auto& group = conceptGenerator_->groups()[index];
                groupView.toggleButton->setText(
                    std::string(groupView.expanded ? "[-] " : "[+] ")
                    + group.label + " (" + std::to_string(group.entries.size()) + ")");
            });
        view.toggleButton->setVisible(activeView_ == DataHubView::Generator);
        generatorWidgets_.push_back(view.toggleButton);
        generatorGroupViews_.push_back(std::move(view));
    }
}

void UIDataHubPanel::startGeneratorRun() {
    if (!conceptGenerator_ || conceptGenerator_->frames().empty()) {
        addLog("Generator has no loaded recipe", 1);
        return;
    }
    if (!generatorAdapter_) {
        addLog("Generator adapter is unavailable: " + generatorAdapterStatus_, 2);
        return;
    }

    const bool hasInlineDocument = genDocumentArea_ && !genDocumentArea_->getText().empty();
    if (!hasInlineDocument && conceptGenerator_->documents().empty()) {
        addLog("Paste a document or add raw documents to GeneratorDataSource.json", 1);
        return;
    }

    if (generatorPaused_ && generatorAttemptCount_ > 0) {
        generatorPaused_ = false;
        generatorRunning_ = true;
        if (btnGenRun_) btnGenRun_->setText(generatorFillActive_ ? "Working" : "Running");
        return;
    }

    generatorBatchCount_ = std::clamp<uint64_t>(
        parseUnsignedOr(genCountInput_ ? genCountInput_->getText() : "25", 25),
        1, 100000);
    if (genCountInput_) genCountInput_->setText(std::to_string(generatorBatchCount_));

    const bool autoSeed = !genSeedPolicyDropdown_
        || genSeedPolicyDropdown_->getSelectedIndex() == 0;
    if (autoSeed) {
        generatorBaseSeed_ = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
    } else {
        generatorBaseSeed_ = parseUnsignedOr(
            genSeedInput_ ? genSeedInput_->getText() : "1337", 1337);
    }
    if (genSeedInput_) genSeedInput_->setText(std::to_string(generatorBaseSeed_));

    generatorIteration_ = 0;
    generatorGeneratedCount_ = 0;
    generatorAttemptCount_ = 0;
    generatorFailedCount_ = 0;
    generatorLoopTimer_ = 0.0f;
    generatorPaused_ = false;

    const int runMode = genRunModeDropdown_ ? genRunModeDropdown_->getSelectedIndex() : 0;
    generatorRunning_ = runMode != 0;
    requestGeneratorFill();
    addLog("Generator model-fill run started with seed "
           + std::to_string(generatorBaseSeed_), 0);
}

void UIDataHubPanel::pauseGeneratorRun() {
    if (!generatorRunning_ && !generatorFillActive_) return;
    generatorRunning_ = false;
    generatorPaused_ = true;
    if (btnGenRun_) btnGenRun_->setText(generatorFillActive_ ? "Finishing" : "Resume");
    addLog("Generator paused after " + std::to_string(generatorAttemptCount_)
           + " attempts", 0);
}

void UIDataHubPanel::generateNextFrame() {
    if (generatorFillActive_) return;
    if (!generatorAdapter_ || !conceptGenerator_ || conceptGenerator_->frames().empty()) return;
    const bool hasInlineDocument = genDocumentArea_ && !genDocumentArea_->getText().empty();
    if (!hasInlineDocument && conceptGenerator_->documents().empty()) {
        addLog("No raw document is available for model filling", 1);
        return;
    }
    requestGeneratorFill();
}

void UIDataHubPanel::saveGeneratorQueue() {
    if (generatorFillActive_) {
        addLog("Wait for the current model fill before saving the queue", 1);
        return;
    }
    if (!datasetTarget_ || generatorQueuedBlocks_.empty()) {
        addLog("Generator queue is empty", 1);
        return;
    }
    const size_t queued = generatorQueuedBlocks_.size();
    size_t added = 0;
    if (!datasetTarget_->addConceptBlocks(generatorQueuedBlocks_, added)) {
        addLog("Generator queue save failed; the queue was kept", 2);
        return;
    }
    generatorQueuedBlocks_.clear();
    cbFilterDirty_ = true;
    addLog("Saved " + std::to_string(added) + " of " + std::to_string(queued)
           + " queued ConceptBlocks (duplicates skipped)", 0);
}

void UIDataHubPanel::requestGeneratorFill() {
    if (generatorFillActive_ || !generatorAdapter_ || !conceptGenerator_) return;
    const int selectedFrame = genFrameDropdown_
        ? std::max(0, genFrameDropdown_->getSelectedIndex()) : 0;
    if (selectedFrame >= static_cast<int>(conceptGenerator_->frames().size())) return;

    GeneratorDocument document;
    const std::string inlineText = genDocumentArea_ ? genDocumentArea_->getText() : "";
    if (!inlineText.empty()) {
        document.id = "inline_document";
        document.title = "Inline Document";
        document.text = inlineText;
    } else if (!conceptGenerator_->documents().empty()) {
        document = conceptGenerator_->documents()[
            static_cast<size_t>(generatorIteration_ % conceptGenerator_->documents().size())];
    } else {
        return;
    }

    generatorPendingDocument_ = document;
    generatorPendingFrame_ = static_cast<size_t>(selectedFrame);
    generatorPendingIteration_ = generatorIteration_;
    const auto recipe = conceptGenerator_->frames()[generatorPendingFrame_];
    const uint64_t requestSeed = generatorBaseSeed_
        + generatorPendingIteration_ * 0x9E3779B97F4A7C15ULL;
    const auto options = generatorAdapterOptions(requestSeed);
    const GeneratorMemoryContext memory; // Adaptation point for the memory system.
    const auto adapter = generatorAdapter_;

    generatorFillActive_ = true;
    generatorAdapterStatus_ = adapter->adapterId() + " — filling " + document.id;
    if (btnGenRun_) btnGenRun_->setText("Working");
    generatorFillFuture_ = std::async(std::launch::async,
        [adapter, document, recipe, memory, options]() mutable {
            return adapter->fill(document, recipe, memory, options);
        });
}

void UIDataHubPanel::completeGeneratorFill() {
    if (!generatorFillActive_ || !generatorFillFuture_.valid()) return;
    if (generatorFillFuture_.wait_for(std::chrono::milliseconds(0))
        != std::future_status::ready) return;

    GeneratorBindingSet bindings;
    try {
        bindings = generatorFillFuture_.get();
    } catch (const std::exception& error) {
        bindings.documentId = generatorPendingDocument_.id;
        bindings.error = error.what();
    }
    generatorFillActive_ = false;
    ++generatorAttemptCount_;

    bool valid = bindings.error.empty()
        && generatorPendingFrame_ < conceptGenerator_->frames().size()
        && conceptGenerator_->validateBindings(
            generatorPendingDocument_,
            conceptGenerator_->frames()[generatorPendingFrame_],
            bindings);

    if (valid) {
        const auto traversal = genTraversalDropdown_
            && genTraversalDropdown_->getSelectedIndex() == 1
            ? GeneratorTraversal::RoundRobin
            : GeneratorTraversal::SeededRandom;
        generatorLastBindings_ = bindings;
        generatorLastFrame_ = conceptGenerator_->assemble(
            generatorPendingFrame_, bindings, generatorBaseSeed_,
            generatorPendingIteration_, traversal);
        generatorQueuedBlocks_.push_back(generatorLastFrame_.conceptBlock);
        if (genPreviewArea_) {
            genPreviewArea_->setText(
                GRIM::ConceptCanonical::renderLogicalTrainingPreview(
                    generatorLastFrame_.conceptBlock));
        }
        ++generatorGeneratedCount_;
        generatorAdapterStatus_ = bindings.adapterId + " — valid bindings";
    } else {
        ++generatorFailedCount_;
        generatorLastBindings_ = bindings;
        generatorAdapterStatus_ = bindings.error.empty()
            ? "Binding validation failed" : bindings.error;
        if (genPreviewArea_) {
            std::string preview = "BINDING ERROR\n" + generatorAdapterStatus_;
            if (!bindings.rawResponse.empty()) preview += "\n\nRAW MODEL RESPONSE\n" + bindings.rawResponse;
            genPreviewArea_->setText(preview);
        }
        addLog("Generator binding failed: " + generatorAdapterStatus_, 2);
    }

    ++generatorIteration_;
    if (btnGenRun_) {
        btnGenRun_->setText(generatorRunning_ ? "Running"
            : (generatorPaused_ ? "Resume" : "Run"));
    }
}

void UIDataHubPanel::updateGeneratorLoop(float dt) {
    completeGeneratorFill();
    if (generatorFillActive_ || !generatorRunning_ || generatorPaused_) return;

    const int runMode = genRunModeDropdown_ ? genRunModeDropdown_->getSelectedIndex() : 0;
    if (runMode == 0) {
        generatorRunning_ = false;
        if (btnGenRun_) btnGenRun_->setText("Run");
        return;
    }
    if (runMode == 1 && generatorAttemptCount_ >= generatorBatchCount_) {
        generatorRunning_ = false;
        if (btnGenRun_) btnGenRun_->setText("Run");
        addLog("Generator batch completed: " + std::to_string(generatorGeneratedCount_)
               + " valid, " + std::to_string(generatorFailedCount_) + " failed", 0);
        return;
    }

    generatorLoopTimer_ += dt;
    if (generatorLoopTimer_ < 0.12f) return;
    generatorLoopTimer_ = 0.0f;
    requestGeneratorFill();
}

void UIDataHubPanel::drawGeneratorTab(OverlayRenderer& renderer,
                                      const PanelRect& content) {
    const float x = content.origin.x + 15.0f;
    const float fullW = content.size.x - 30.0f;
    const float controlLabelY = content.origin.y + 4.0f;
    const float controlY = content.origin.y + 22.0f;
    constexpr float controlH = 32.0f;
    constexpr float gap = 8.0f;

    float cx = x;
    auto placeControl = [&](const std::shared_ptr<Widget>& widget,
                            const std::string& label, float width) {
        renderer.drawText({cx + 2.0f, controlLabelY}, label,
                          UITheme::Colors::TextSecondary);
        widget->setPosition(cx, controlY);
        widget->setSize(width, controlH);
        widget->drawOverlay(renderer, position);
        cx += width + gap;
    };
    placeControl(genFrameDropdown_, "RECIPE", 165.0f);
    placeControl(genRunModeDropdown_, "RUN MODE", 125.0f);
    placeControl(genSeedPolicyDropdown_, "SEED POLICY", 130.0f);
    placeControl(genTraversalDropdown_, "TRAVERSAL", 145.0f);
    placeControl(genSeedInput_, "SEED", 150.0f);
    placeControl(genCountInput_, "COUNT", 75.0f);
    placeControl(btnGenRun_, "", 78.0f);
    placeControl(btnGenPause_, "", 78.0f);
    placeControl(btnGenNext_, "", 72.0f);

    const float headerY = content.origin.y + 72.0f;
    const float bodyY = content.origin.y + 92.0f;
    const float bodyH = content.size.y - 125.0f;
    const float leftW = fullW * 0.28f;
    const float middleW = fullW * 0.30f;
    const float rightW = fullW - leftW - middleW - 20.0f;
    const float middleX = x + leftW + 10.0f;
    const float rightX = middleX + middleW + 10.0f;

    renderer.drawText({x + 2.0f, headerY}, "RAW DOCUMENT INPUT",
                      UITheme::Colors::TextSecondary);
    renderer.drawText({middleX + 2.0f, headerY}, "RECIPE + LEXICON",
                      UITheme::Colors::TextSecondary);
    renderer.drawText({rightX + 2.0f, headerY}, "GENERATED CONCEPTBLOCK",
                      UITheme::Colors::TextSecondary);
    if (btnGenSaveQueue_) {
        btnGenSaveQueue_->setPosition(rightX + rightW - 108.0f, headerY - 7.0f);
        btnGenSaveQueue_->setSize(106.0f, 27.0f);
        btnGenSaveQueue_->drawOverlay(renderer, position);
    }

    for (const auto& box : std::vector<std::pair<Vec2, Vec2>>{
             {{x, bodyY}, {leftW, bodyH}},
             {{middleX, bodyY}, {middleW, bodyH}},
             {{rightX, bodyY}, {rightW, bodyH}}}) {
        renderer.drawRoundedRect(box.first, box.second,
                                 UITheme::Colors::ContentAreaBg,
                                 UITheme::Sizes::WidgetRadius);
        renderer.drawRoundedBorder(box.first, box.second,
                                   UITheme::Colors::BorderSubtle,
                                   UITheme::Sizes::WidgetRadius);
    }

    renderer.drawText({x + 12.0f, bodyY + 11.0f}, generatorSourceStatus_,
                      conceptGenerator_ && !conceptGenerator_->documents().empty()
                          ? UITheme::Colors::Success : UITheme::Colors::Warning);
    if (genDocumentArea_) {
        genDocumentArea_->setPosition(x + 10.0f, bodyY + 35.0f);
        genDocumentArea_->setSize(leftW - 20.0f, bodyH * 0.62f);
        genDocumentArea_->drawOverlay(renderer, position);
    }
    float sourceY = bodyY + bodyH * 0.62f + 52.0f;
    renderer.drawText({x + 12.0f, sourceY}, "Paste text above to override the source file.",
                      UITheme::Colors::TextMuted);
    sourceY += 24.0f;
    renderer.drawText({x + 12.0f, sourceY}, "DATASET RESOURCE", UITheme::Colors::TextMuted);
    sourceY += 19.0f;
    for (const auto& line : renderer.wrapText(
             generatorResourceRelativePath("GeneratorDataSource.json"), leftW - 24.0f)) {
        renderer.drawText({x + 12.0f, sourceY}, line, UITheme::Colors::TextValue);
        sourceY += 18.0f;
    }
    sourceY += 15.0f;
    renderer.drawText({x + 12.0f, sourceY}, "Rows: { id, title, text, metadata }",
                      UITheme::Colors::TextSecondary);

    renderer.pushClipRect({middleX, bodyY}, {middleW, bodyH});
    float groupY = bodyY + 8.0f - generatorLexiconScroll_;
    if (conceptGenerator_) {
        const auto& groups = conceptGenerator_->groups();
        for (size_t index = 0;
             index < groups.size() && index < generatorGroupViews_.size(); ++index) {
            auto& view = generatorGroupViews_[index];
            const auto& group = groups[index];
            view.toggleButton->setPosition(middleX + 8.0f, groupY);
            view.toggleButton->setSize(middleW - 24.0f, 28.0f);
            if (groupY + 28.0f >= bodyY && groupY <= bodyY + bodyH)
                view.toggleButton->drawOverlay(renderer, position);
            groupY += 33.0f;
            if (!view.expanded) continue;
            for (const auto& entry : group.entries) {
                if (groupY + 19.0f >= bodyY && groupY <= bodyY + bodyH) {
                    renderer.drawText({middleX + 18.0f, groupY + 1.0f},
                                      cbSingleLinePreview(entry.text, 34),
                                      UITheme::Colors::TextSecondary);
                    const std::string type = ConceptGenerator::valueTypeName(entry.type);
                    const float typeW = renderer.measureTextWidth(type);
                    renderer.drawText({middleX + middleW - typeW - 18.0f, groupY + 1.0f},
                                      type, UITheme::Colors::TextValue);
                }
                groupY += 21.0f;
            }
            groupY += 5.0f;
        }
    }
    generatorLexiconContentH_ = groupY + generatorLexiconScroll_ - bodyY + 8.0f;
    renderer.popClipRect();

    if (generatorLexiconContentH_ > bodyH) {
        const float maxScroll = generatorLexiconContentH_ - bodyH + 12.0f;
        const float ratio = maxScroll > 0.0f ? generatorLexiconScroll_ / maxScroll : 0.0f;
        const float thumbH = std::max(24.0f, bodyH * bodyH / generatorLexiconContentH_);
        renderer.drawRoundedRect(
            {middleX + middleW - 6.0f, bodyY + ratio * (bodyH - thumbH)},
            {4.0f, thumbH}, UITheme::Colors::ScrollThumb, 2.0f);
    }

    renderer.drawText({rightX + 12.0f, bodyY + 11.0f}, generatorAdapterStatus_,
                      generatorLastBindings_.valid
                          ? UITheme::Colors::Success : UITheme::Colors::TextMuted);
    if (genPreviewArea_) {
        genPreviewArea_->setPosition(rightX + 10.0f, bodyY + 35.0f);
        genPreviewArea_->setSize(rightW - 20.0f, bodyH * 0.66f);
        genPreviewArea_->drawOverlay(renderer, position);
    }

    float previewY = bodyY + bodyH * 0.66f + 52.0f;
    renderer.drawText({rightX + 12.0f, previewY},
                      "DOCUMENT  " + generatorLastBindings_.documentId,
                      UITheme::Colors::TextSecondary);
    previewY += 20.0f;
    renderer.drawText({rightX + 12.0f, previewY},
                      "ATOMS  " + std::to_string(generatorLastBindings_.atoms.size())
                      + "   EVIDENCE  " + std::to_string(generatorLastBindings_.evidence.size()),
                      UITheme::Colors::TextSecondary);
    previewY += 20.0f;
    renderer.drawText({rightX + 12.0f, previewY},
                      "ATTEMPTS  " + std::to_string(generatorAttemptCount_)
                      + "   VALID  " + std::to_string(generatorGeneratedCount_)
                      + "   FAILED  " + std::to_string(generatorFailedCount_)
                      + "   QUEUED  " + std::to_string(generatorQueuedBlocks_.size()),
                      UITheme::Colors::TextSecondary);
    previewY += 26.0f;
    renderer.drawText({rightX + 12.0f, previewY}, "LEXICON SELECTIONS",
                      UITheme::Colors::TextMuted);
    previewY += 19.0f;
    for (size_t index = 0;
         index < generatorLastFrame_.selections.size() && index < 5; ++index) {
        const auto& selection = generatorLastFrame_.selections[index];
        renderer.drawText({rightX + 16.0f, previewY},
                          selection.groupId + " → "
                              + cbSingleLinePreview(selection.text, 34),
                          UITheme::Colors::TextSecondary);
        previewY += 19.0f;
    }

    const std::string runStatus = generatorFillActive_ ? "MODEL FILLING"
        : (generatorRunning_ ? "RUNNING" : (generatorPaused_ ? "PAUSED" : "READY"));
    renderer.drawText({x + 2.0f, content.origin.y + content.size.y - 20.0f},
                      runStatus + "  |  Ollama adapter now; native GRIM/memory adapter later",
                      generatorFillActive_ || generatorRunning_
                          ? UITheme::Colors::Success : UITheme::Colors::TextMuted);
}
