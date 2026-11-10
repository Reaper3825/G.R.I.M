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
    return (std::filesystem::path(aiConfig.at("paths").at("grim_text").at("training_data")
        .get<std::string>()) / filename).string();
}

} // namespace

void UIDataHubPanel::refreshGeneratorState() {
    if (!creationService_) return;
    const bool loaded = creationService_->reloadAssets();
    conceptGenerator_ = creationService_->assets();
    if (!loaded) addLog("Generator resources: " + creationService_->assetError(), 2);
    if (!conceptGenerator_) {
        generatorCatalogStatus_ = creationService_->assetError();
        return;
    }
    generatorCatalogStatus_ = std::to_string(conceptGenerator_->groups().size())
        + " language groups | " + std::to_string(conceptGenerator_->frames().size()) + " recipes";
    generatorSourceStatus_ = std::to_string(conceptGenerator_->documents().size()) + " raw documents";
    std::vector<std::string> labels;
    for (const auto& frame : conceptGenerator_->frames()) labels.push_back(frame.label);
    if (genFrameDropdown_) genFrameDropdown_->setItems(labels);
    rebuildGeneratorGroupViews();
    updateGeneratorLoop(0.0f);
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
    if (!creationService_) return;
    if (creationService_->status().paused && creationService_->resume()) return;
    generatorBatchCount_ = std::clamp<uint64_t>(
        parseUnsignedOr(genCountInput_ ? genCountInput_->getText() : "25", 25), 1, 100000);
    if (genCountInput_) genCountInput_->setText(std::to_string(generatorBatchCount_));
    const bool autoSeed = !genSeedPolicyDropdown_ || genSeedPolicyDropdown_->getSelectedIndex() == 0;
    generatorBaseSeed_ = autoSeed
        ? static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())
        : parseUnsignedOr(genSeedInput_ ? genSeedInput_->getText() : "1337", 1337);
    if (genSeedInput_) genSeedInput_->setText(std::to_string(generatorBaseSeed_));
    requestGeneratorFill();
}

void UIDataHubPanel::pauseGeneratorRun() {
    if (creationService_) creationService_->pause();
}

void UIDataHubPanel::generateNextFrame() {
    if (!creationService_) return;
    DatasetCreationService::Request request;
    request.frameIndex = genFrameDropdown_ ? std::max(0, genFrameDropdown_->getSelectedIndex()) : 0;
    request.seed = generatorBaseSeed_;
    request.traversal = genTraversalDropdown_ && genTraversalDropdown_->getSelectedIndex() == 1
        ? GeneratorTraversal::RoundRobin : GeneratorTraversal::SeededRandom;
    if (genDocumentArea_) request.document = {"inline_document", "Inline Document", genDocumentArea_->getText(), ""};
    if (!creationService_->start(request)) addLog("Generator is busy or its resources are unavailable", 1);
}

void UIDataHubPanel::saveGeneratorQueue() {
    if (!creationService_) return;
    size_t added = 0;
    if (!creationService_->saveQueue(added)) {
        addLog("Generator queue save failed: " + datasetTarget_->lastError(), 2);
        return;
    }
    cbFilterDirty_ = true;
    addLog("Saved " + std::to_string(added) + " queued ConceptBlocks", 0);
    updateGeneratorLoop(0.0f);
}

void UIDataHubPanel::requestGeneratorFill() {
    if (!creationService_) return;
    DatasetCreationService::Request request;
    request.frameIndex = genFrameDropdown_ ? std::max(0, genFrameDropdown_->getSelectedIndex()) : 0;
    request.seed = generatorBaseSeed_;
    const int mode = genRunModeDropdown_ ? genRunModeDropdown_->getSelectedIndex() : 0;
    request.attempts = mode == 0 ? 1 : generatorBatchCount_;
    request.continuous = mode == 2;
    request.traversal = genTraversalDropdown_ && genTraversalDropdown_->getSelectedIndex() == 1
        ? GeneratorTraversal::RoundRobin : GeneratorTraversal::SeededRandom;
    if (genDocumentArea_) request.document = {"inline_document", "Inline Document", genDocumentArea_->getText(), ""};
    if (!creationService_->start(request)) addLog("Generator is busy or its resources are unavailable", 1);
}

void UIDataHubPanel::updateGeneratorLoop(float) {
    if (!creationService_) return;
    const auto status = creationService_->status();
    if (status.revision == generatorStatusRevision_) return;
    generatorStatusRevision_ = status.revision;
    generatorRunning_ = status.running;
    generatorPaused_ = status.paused;
    generatorFillActive_ = status.filling;
    generatorAttemptCount_ = status.attempts;
    generatorGeneratedCount_ = status.generated;
    generatorFailedCount_ = status.failed;
    generatorQueuedCount_ = status.queued;
    generatorLastFrame_ = status.lastFrame;
    generatorLastBindings_ = status.lastBindings;
    generatorAdapterStatus_ = status.message;
    if (btnGenRun_) btnGenRun_->setText(status.filling ? "Working" : (status.paused ? "Resume" : (status.running ? "Running" : "Run")));
    if (genPreviewArea_ && !status.filling && status.attempts > 0) {
        if (status.lastBindings.error.empty() && status.lastBindings.valid)
            genPreviewArea_->setText(buildTrainingPreview(status.lastFrame.conceptBlock, true));
        else genPreviewArea_->setText("BINDING ERROR\n" + status.message + "\n\n" + status.lastBindings.rawResponse);
    }
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
                      + "   QUEUED  " + std::to_string(generatorQueuedCount_),
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
