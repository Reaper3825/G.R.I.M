#include "ui_data_hub_internal.hpp"

using namespace UIDataHubDetail;

UIDataHubPanel::UIDataHubPanel()
    : UIPanel("DataHub", true)
{
    position = {200, 500};
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
    cbExecutionGateDropdown_ = std::make_shared<UIDropdown>(
        "Execution Gate",
        std::vector<std::string>{"Unsupervised", "Noop", "Execute"},
        0, [](int, const std::string&) {});
    cbExecutionGateDropdown_->setMaxVisibleItems(3);
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
        if (cbExecutionGateDropdown_)
            cbExecutionGateDropdown_->setSelectedIndex(
                static_cast<int>(GRIM::ConceptExecutionGateTarget::Execute));
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
        std::string validationError;
        if (!buildConceptBlockFromEditor(cb, validationError)) {
            addLog("Cannot save ConceptBlock: " + validationError, 2);
            return;
        }
        cb.name = name;
        cb.question = question;
        cb.format_type = formatKey;

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
        cbExecutionGateDropdown_, cbState0TypeInput_, cbState0AtomsInput_,
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
        fs::path modelStoreRoot = resolvePathFromGrimRoot(
            aiConfig.at("paths").at("grim_text").at("model_store").get<std::string>());

        fs::path massDatasetPath = resolvePathFromGrimRoot(
            aiConfig.at("paths").at("grim_text").at("training_data").get<std::string>()).parent_path() / "mass_dataset.jsonl";

        datasetTarget_ = std::make_unique<DatasetTarget>(modelStoreRoot, massDatasetPath);

        GRIM::DataCollection::DataStructuringConfig structCfg;
        if (aiConfig.contains("data_collection") && aiConfig["data_collection"].contains("data_structuring")) {
            structCfg = GRIM::DataCollection::DataStructuringConfig::fromJson(
                aiConfig["data_collection"]["data_structuring"]);
        }
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

// =========================================================
// Config persistence
// =========================================================

void UIDataHubPanel::loadUIConfig() {
    std::string path = (resolvePathFromGrimRoot(aiConfig.at("paths").at("grim_text").at("checkpoints").get<std::string>()) / "collection_state" / "ui_config.json").string();
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
    std::string stateDir = (resolvePathFromGrimRoot(aiConfig.at("paths").at("grim_text").at("checkpoints").get<std::string>()) / "collection_state").string();
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
    std::string path = (resolvePathFromGrimRoot(aiConfig.at("paths").at("grim_text").at("checkpoints").get<std::string>()) / "collection_state" / "download_queue.json").string();
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
    std::string stateDir = (resolvePathFromGrimRoot(aiConfig.at("paths").at("grim_text").at("checkpoints").get<std::string>()) / "collection_state").string();
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
