#include "ui_data_hub_internal.hpp"

void UIDataHubPanel::requestStructure(const std::string& raw, const std::string& mode,
    const std::string& prompt, StructureTarget target, size_t card) {
    if (structureRequest_) { addLog("A draft generation is already running", 1); return; }
    try {
        structureTarget_ = target; structureCard_ = card; structureMode_ = mode;
        structureDraft_ = structureDraft();
        DataHubJobs::QueryRequest request;
        request.kind = DataHubJobs::QueryKind::Structure;
        request.query = raw; request.mode = mode; request.prompt = prompt;
        structureRequest_ = DataHubJobs::submit(std::move(request));
        addLog("Generation submitted to host", 0);
    } catch (const std::exception& error) { addLog(error.what(), 2); }
}

std::string UIDataHubPanel::structureDraft() const {
    nlohmann::json draft = nlohmann::json::array();
    if (structureTarget_ == StructureTarget::Concept) {
        draft = {cbEditorBlockId_, cbEditorRevision_, cbPromptArea_->getText(),
                 cbAnswerArea_->getText(), cbCustomPromptArea_->getText()};
        for (const auto& area : cbIntermediateAreas_) draft.push_back(area->getText());
    } else if (structureTarget_ == StructureTarget::Card) {
        draft = {rawTextArea_->getText(), customPromptArea_->getText()};
        const auto found = std::find_if(sequenceCards_.begin(), sequenceCards_.end(),
            [this](const auto& card) { return card.cardId == structureCard_; });
        if (found == sequenceCards_.end()) return "removed-card";
        draft.push_back(found->formatIndex);
        for (const auto& entry : found->entries) draft.push_back({entry.prefix, entry.textArea->getText()});
    } else {
        draft = {currentSequenceIndex_, sequenceEditorRevision_, rawTextArea_->getText(),
                 structuredTextArea_->getText(), customPromptArea_->getText(),
                 formatDropdown_->getSelectedIndex()};
    }
    return draft.dump();
}

void UIDataHubPanel::pollHostJobs() {
    if (searchRequest_) {
        const auto result = DataHubJobs::poll(searchRequest_);
        if (result.ready) {
            DataHubJobs::release(searchRequest_); searchRequest_ = 0;
            hfSearching_.store(false);
            hfSearchResults_ = result.datasets;
            lastSearchError_ = result.error;
            if (!result.error.empty()) addLog(result.error, 2);
            else if (hfSearchResults_.empty()) lastSearchError_ = "No datasets found";
            hfResultsNeedsPopulate_.store(true);
            hfPreviewArea_->setText("Select a result to preview its sample rows");
        }
    }
    if (previewRequest_) {
        const auto result = DataHubJobs::poll(previewRequest_);
        if (result.ready) {
            DataHubJobs::release(previewRequest_); previewRequest_ = 0;
            if (!result.error.empty()) {
                hfPreviewArea_->setText(result.error); addLog(result.error, 2);
            } else hfPreviewArea_->setText(result.preview);
        }
    }
    if (structureRequest_) {
        const auto result = DataHubJobs::poll(structureRequest_);
        if (result.ready) {
            DataHubJobs::release(structureRequest_); structureRequest_ = 0;
            if (!result.error.empty()) addLog(result.error, 2);
            else if (structureDraft_ != structureDraft())
                addLog("Generation discarded because its target draft changed or was removed", 1);
            else if (result.structured.empty()) addLog("Generation returned no structured output", 2);
            else if (structureTarget_ == StructureTarget::Card)
                applyCardResult(structureCard_, structureMode_, result.structured);
            else if (structureTarget_ == StructureTarget::Concept)
                applyConceptResult(result.structured);
            else {
                std::string combined;
                for (const auto& value : result.structured) {
                    if (!combined.empty()) combined += "\n\n---\n\n";
                    combined += value;
                }
                structuredTextArea_->setText(combined);
                addLog("Generation completed", 0);
            }
        }
    }
    const auto revision = downloadsRevision_;
    loadDownloadQueue();
    if (downloadsRevision_ != revision && activeView_ != DataHubView::Sources && !sourcesDirty_)
        loadSourceCards();
    const auto batch = DataHubJobs::batchStatus();
    if (batch.revision != batchRevision_) {
        batchRevision_ = batch.revision;
        failedCount_ = batch.failed;
        if (!batch.message.empty()) addLog(batch.message, batch.failed == 0 ? 0 : 2);
    }
}
