// UITrainingPanel: shared lifecycle, tab dispatch, and server polling
#include "training_panel/ui_training_panel_internal.hpp"

using namespace GRIMText;
using namespace UITheme;
using namespace UITrainingPanelDetail;

// ============================================================
// Constructor
// ============================================================

UITrainingPanel::UITrainingPanel()
    : UIPanel("GRIM-text Training Control", true)
{
    position = { 200, 80 };
    size     = { 1100, 800 };
    setVisible(false);
    setBackground(Colors::PanelBg);

    if (position.y < 50.0f) position.y = 50.0f;

    // ── Training controller ──
    std::string host = "127.0.0.1";
    int port = 11436;
    try {
        host = aiConfig.at("training").value("server_host", host);
        port = aiConfig.at("training").value("server_port", port);
    } catch (...) {
    }
    try {
        trainingController = std::make_unique<GRIM::UI::UITrainingController>(host, port);
    } catch (const std::exception& e) {
        LOG_ERROR("UITrainingPanel", std::string("Controller init failed: ") + e.what());
    }

    // ══════════════════════════════════════════════════════
    //  Tab buttons (DataHub pattern)
    // ══════════════════════════════════════════════════════

    tabHomeBtn_ = std::make_shared<UIButton>("Home", [this]() {
        setView(TrainingPanelTab::Home);
    });
    tabHomeBtn_->setSize(90.0f, 28.0f);

    tabKnowledgeGapsBtn_ = std::make_shared<UIButton>("Knowledge Gaps", [this]() {
        setView(TrainingPanelTab::KnowledgeGaps);
    });
    tabKnowledgeGapsBtn_->setSize(120.0f, 28.0f);

    tabToolGapsBtn_ = std::make_shared<UIButton>("Tool Gaps", [this]() {
        setView(TrainingPanelTab::ToolGaps);
    });
    tabToolGapsBtn_->setSize(90.0f, 28.0f);

    tabModelConfigBtn_ = std::make_shared<UIButton>("Model Config", [this]() {
        setView(TrainingPanelTab::ModelConfig);
    });
    tabModelConfigBtn_->setSize(115.0f, 28.0f);

    tabTokenizerBtn_ = std::make_shared<UIButton>("Tokenizer", [this]() {
        setView(TrainingPanelTab::Tokenizer);
    });
    tabTokenizerBtn_->setSize(90.0f, 28.0f);

    // ══════════════════════════════════════════════════════
    //  Home tab — Model Browser widgets
    // ══════════════════════════════════════════════════════

    browserScrollBox_ = std::make_shared<UIScrollBox>();
    browserScrollBox_->setSize(1060.0f, 500.0f);

    vramBar_ = std::make_shared<UIProgressBar>("VRAM", 1.0f);
    vramBar_->setSize(500.0f, 20.0f);
    vramBar_->setShowPercentage(true);
    vramBar_->setFillColor(Colors::AccentBlue);
    vramBar_->setBackgroundColor(Colors::ContentAreaBg);

    ramBar_ = std::make_shared<UIProgressBar>("RAM", 1.0f);
    ramBar_->setSize(500.0f, 20.0f);
    ramBar_->setShowPercentage(true);
    ramBar_->setFillColor(Colors::Success);
    ramBar_->setBackgroundColor(Colors::ContentAreaBg);

    createModelBtn_ = std::make_shared<UIButton>("+ New Model", [this]() {
        showCreatorForm_ = !showCreatorForm_;
    });
    createModelBtn_->setSize(120.0f, 30.0f);

    // Model Creator form widgets
    creatorNameInput_ = std::make_shared<UIInputBox>(&bufName_);
    creatorNameInput_->setPlaceholder("Display name");
    creatorNameInput_->setSize(400.0f, 26.0f);

    bufParamStr_ = "0M";
    creatorParamInput_ = std::make_shared<UIInputBox>(&bufParamStr_);
    creatorParamInput_->setPlaceholder("Parameters (e.g. 7B, 405M)");
    creatorParamInput_->setSize(200.0f, 26.0f);

    creatorSubjectInput_ = std::make_shared<UIInputBox>(&bufSubject_);
    creatorSubjectInput_->setPlaceholder("Subject (e.g. medical)");
    creatorSubjectInput_->setSize(400.0f, 26.0f);

    creatorTagsInput_ = std::make_shared<UIInputBox>(&bufTags_);
    creatorTagsInput_->setPlaceholder("Tags (comma-separated)");
    creatorTagsInput_->setSize(400.0f, 26.0f);

    creatorDescInput_ = std::make_shared<UIInputBox>(&bufDesc_);
    creatorDescInput_->setPlaceholder("Description");
    creatorDescInput_->setSize(400.0f, 26.0f);

    creatorStatusLabel_ = std::make_shared<UILabel>("", Colors::TextPrimary);
    creatorStatusLabel_->setSize(400.0f, 20.0f);

    creatorRegisterBtn_ = std::make_shared<UIButton>("Register", [this]() {
        submitNewModel();
    });
    creatorRegisterBtn_->setSize(100.0f, 30.0f);

    creatorCancelBtn_ = std::make_shared<UIButton>("Cancel", [this]() {
        clearCreatorFields();
        showCreatorForm_ = false;
    });
    creatorCancelBtn_->setSize(100.0f, 30.0f);

    // ══════════════════════════════════════════════════════
    //  Knowledge Gap queue scroll box
    // ══════════════════════════════════════════════════════

    gapScrollBox_ = std::make_shared<UIScrollBox>();
    gapScrollBox_->setSize(1060.0f, 600.0f);

    // ══════════════════════════════════════════════════════
    //  Tool Gap queue scroll box
    // ══════════════════════════════════════════════════════

    toolGapScrollBox_ = std::make_shared<UIScrollBox>();
    toolGapScrollBox_->setSize(1060.0f, 600.0f);

    // ══════════════════════════════════════════════════════
    //  Model Config tab widgets
    // ══════════════════════════════════════════════════════

    configModelDropdown_ = std::make_shared<UIDropdown>(
        "Existing model", std::vector<std::string>{"(new model)"}, 0,
        [this](int idx, const std::string&) {
            const int modelIndex = idx - 1;
            if (modelIndex >= 0 && modelIndex < static_cast<int>(configModelIds_.size())) {
                configModelIdBuffer_ = configModelIds_[static_cast<size_t>(modelIndex)];
                if (configModelIdInput_) configModelIdInput_->setText(configModelIdBuffer_);
            }
        });
    configModelDropdown_->setSize(300.0f, 26.0f);

    configModelIdInput_ = std::make_shared<UIInputBox>(&configModelIdBuffer_);
    configModelIdInput_->setPlaceholder("model-id");
    configModelIdInput_->setSize(420.0f, 28.0f);

    try {
        configVocabPathBuffer_ = aiConfig.at("paths").at("grim_text").at("vocab").get<std::string>();
    } catch (...) {
        configVocabPathBuffer_ = "resources/models/GRIM-text/training/data/vocab.bin";
    }
    configVocabPathInput_ = std::make_shared<UIInputBox>(&configVocabPathBuffer_);
    configVocabPathInput_->setPlaceholder("Path to KTMG v4 vocab.bin");
    configVocabPathInput_->setSize(420.0f, 28.0f);

    configCompilerPathBuffer_ = findConfigCompilerExecutable();
    configCompilerPathInput_ = std::make_shared<UIInputBox>(&configCompilerPathBuffer_);
    configCompilerPathInput_->setPlaceholder("Path to compile_model_config executable");
    configCompilerPathInput_->setSize(420.0f, 28.0f);

    configCompileButton_ = std::make_shared<UIButton>("Create .grimcfg", [this]() {
        beginConfigCompile();
    });
    configCompileButton_->setSize(150.0f, 30.0f);

    configReloadButton_ = std::make_shared<UIButton>("Reload Schema Defaults", [this]() {
        reloadConfigPresetTemplate();
    });
    configReloadButton_->setSize(190.0f, 30.0f);

    loadHyperparamSnapshot();
    refreshConfigModelDropdown();

    // FlatBuffer table filter — built after loading the model schema fields.
    std::vector<std::string> filterItems = {"All"};
    if (hyperparamsLoaded_) {
        for (const auto& cat : hyperparamRegistry_.categories()) {
            filterItems.push_back(cat);
        }
    }
    paramCategoryFilter_ = std::make_shared<UIDropdown>(
        "Schema table", filterItems, 0,
        [this](int idx, const std::string& /*name*/) {
            selectedParamCategory_ = idx;
            paramScrollOffset_ = 0.0f;
            hoveredParamRow_ = -1;
        });
    paramCategoryFilter_->setSize(200.0f, 26.0f);

    // Inline param editor — shared input box, hidden until a row is clicked
    paramEditInput_ = std::make_shared<UIInputBox>(&editParamBuffer_);
    paramEditInput_->setSize(120.0f, 20.0f);
    paramEditInput_->OnTextSubmitted.Bind([this](const std::string&) { commitParamEdit(); });

    // ══════════════════════════════════════════════════════
    //  Tokenizer tab widgets
    // ══════════════════════════════════════════════════════

    encodeInputBox_ = std::make_shared<UIInputBox>(&encodeInputBuffer_);
    encodeInputBox_->setSize(500.0f, 28.0f);
    encodeInputBox_->setPlaceholder("Enter text to encode...");

    encodeButton_ = std::make_shared<UIButton>("Encode", [this]() {
        handleEncodeText();
    });
    encodeButton_->setSize(90.0f, Sizes::ButtonHeight);

    clearEncodeButton_ = std::make_shared<UIButton>("Clear", [this]() {
        encodeInputBuffer_.clear();
        encodeComplete_ = false;
        encodeSuccess_ = false;
        encodeErrorMessage_.clear();
        lastEncodeResult_ = {};
    });
    clearEncodeButton_->setSize(70.0f, Sizes::ButtonHeight);

    tokenizerRunValidationBtn_ = std::make_shared<UIButton>("Run Validation", [this]() {
        handleRunTokenizer();
    });
    tokenizerRunValidationBtn_->setSize(120.0f, Sizes::ButtonHeight);

    tokenizerCloseBtn_ = std::make_shared<UIButton>("Close", [this]() {
        setVisible(false);
    });
    tokenizerCloseBtn_->setSize(90.0f, Sizes::ButtonHeight);

    tokenizerScrollBox_ = std::make_shared<UIScrollBox>();
    tokenizerScrollBox_->setSize(600.0f, 300.0f);
}

UITrainingPanel::~UITrainingPanel() = default;

// ============================================================
// View Control (DataHub pattern)
// ============================================================

void UITrainingPanel::setView(TrainingPanelTab tab) {
    activeTab_ = tab;
    if (tab == TrainingPanelTab::Home) {
        refreshModelList();
        updateResourceBars();
    } else if (tab == TrainingPanelTab::ModelConfig) {
        refreshConfigModelDropdown();
    }
}

// ============================================================
// Update
// ============================================================

void UITrainingPanel::update(const InputState& input, float dt) {
    if (!isVisible()) return;
    UIPanel::update(input, dt);

    // Tab buttons — always update
    tabHomeBtn_->update(input, dt);
    tabKnowledgeGapsBtn_->update(input, dt);
    tabToolGapsBtn_->update(input, dt);
    tabModelConfigBtn_->update(input, dt);
    tabTokenizerBtn_->update(input, dt);

    // Bottom bar buttons — always active

    // Tab-specific updates
    switch (activeTab_) {
        case TrainingPanelTab::Home: {
            browserScrollBox_->update(input, dt);
            vramBar_->update(input, dt);
            ramBar_->update(input, dt);
            createModelBtn_->update(input, dt);

            processBrowserClicks(input, getContentRect());

            browserRefreshTimer_ += dt;
            if (browserRefreshTimer_ >= browserRefreshInterval_) {
                browserRefreshTimer_ = 0.0f;
                refreshModelList();
                updateResourceBars();
            }

            if (showCreatorForm_) {
                creatorNameInput_->update(input, dt);
                creatorParamInput_->update(input, dt);
                creatorSubjectInput_->update(input, dt);
                creatorTagsInput_->update(input, dt);
                creatorDescInput_->update(input, dt);
                creatorRegisterBtn_->update(input, dt);
                creatorCancelBtn_->update(input, dt);

                // Regenerate ID whenever subject or param count changes
                static std::string lastBufName;
                static std::string lastBufParam;
                if (bufSubject_ != lastBufName || bufParamStr_ != lastBufParam) {
                    lastBufName  = bufSubject_;
                    lastBufParam = bufParamStr_;
                    regenerateId();
                }
            }
            break;
        }
        case TrainingPanelTab::KnowledgeGaps: {
            gapScrollBox_->update(input, dt);
            processGapClicks(input, getContentRect());
            break;
        }
        case TrainingPanelTab::ToolGaps: {
            toolGapScrollBox_->update(input, dt);
            processToolGapClicks(input, getContentRect());
            break;
        }
        case TrainingPanelTab::ModelConfig: {
            updateModelConfigTab(input, dt);
            break;
        }
        case TrainingPanelTab::Tokenizer: {
            if (encodeInputBox_) encodeInputBox_->update(input, dt);
            if (encodeButton_) encodeButton_->update(input, dt);
            if (clearEncodeButton_) clearEncodeButton_->update(input, dt);
            if (tokenizerRunValidationBtn_) tokenizerRunValidationBtn_->update(input, dt);
            if (tokenizerCloseBtn_) tokenizerCloseBtn_->update(input, dt);
            if (tokenizerScrollBox_) tokenizerScrollBox_->update(input, dt);
            break;
        }
    }

    applyConfigCompileResult();

    if (activeTab_ == TrainingPanelTab::Tokenizer) {
        pollTimer += dt;
        if (pollTimer >= pollInterval) {
            pollTimer = 0.0f;
            pollServer();
        }
    }
}

// ============================================================
// Server Polling
// ============================================================

void UITrainingPanel::pollServer() {
    serverConnected = trainingController && trainingController->isServerRunning();
}

// ============================================================
// Draw — Main entry + tab dispatch (DataHub pattern)
// ============================================================

bool UITrainingPanel::drawOverlay(OverlayRenderer& renderer) {
    if (!UIPanel::drawOverlay(renderer)) return false;

    // ── Tab buttons ──
    float tabX = position.x + 10.0f;
    tabHomeBtn_->setPosition(tabX, position.y + kTabBarY);
    tabKnowledgeGapsBtn_->setPosition(tabX + 95.0f, position.y + kTabBarY);
    tabToolGapsBtn_->setPosition(tabX + 220.0f, position.y + kTabBarY);
    tabModelConfigBtn_->setPosition(tabX + 315.0f, position.y + kTabBarY);
    tabTokenizerBtn_->setPosition(tabX + 435.0f, position.y + kTabBarY);

    tabHomeBtn_->drawOverlay(renderer, position);
    tabKnowledgeGapsBtn_->drawOverlay(renderer, position);
    tabToolGapsBtn_->drawOverlay(renderer, position);
    tabModelConfigBtn_->drawOverlay(renderer, position);
    tabTokenizerBtn_->drawOverlay(renderer, position);

    // Active tab indicator (2px underline)
    float indicatorX = tabX;
    float indicatorW = 90.0f;
    switch (activeTab_) {
        case TrainingPanelTab::Home:          indicatorX = tabX;           indicatorW = 90.0f;  break;
        case TrainingPanelTab::KnowledgeGaps: indicatorX = tabX + 95.0f;  indicatorW = 120.0f; break;
        case TrainingPanelTab::ToolGaps:      indicatorX = tabX + 220.0f; indicatorW = 90.0f;  break;
        case TrainingPanelTab::ModelConfig:   indicatorX = tabX + 315.0f; indicatorW = 115.0f; break;
        case TrainingPanelTab::Tokenizer:     indicatorX = tabX + 435.0f; indicatorW = 90.0f;  break;
    }
    renderer.drawRect({indicatorX, position.y + kTabBarY + 28.0f}, {indicatorW, 2.0f},
                      Colors::Primary);

    // ── Status pill (top-right) ──
    if (activeTab_ == TrainingPanelTab::Tokenizer) {
        const std::string stateText = serverConnected
            ? "Tokenizer service online"
            : "Tokenizer service offline";
        const uint32_t stateColor = serverConnected ? Colors::Success : Colors::TextMuted;
        const float textWidth = UIDrawHelpers::getTextWidth(stateText);
        const float textX = position.x + size.x - textWidth - Spacing::Large;
        renderer.drawText({textX, position.y + kTabBarY + 4.0f}, stateText, stateColor);
    }

    // ── Tab content area ──
    PanelRect content = getContentRect();
    content.origin.y += (kContentTopY - kTabBarY);
    content.size.y   -= (kContentTopY - kTabBarY);
    // Reserve space for the tokenizer validation action bar.
    if (activeTab_ == TrainingPanelTab::Tokenizer) {
        content.size.y -= kBottomBarH;
    }

    switch (activeTab_) {
        case TrainingPanelTab::Home:          drawHomeTab(renderer, content);          break;
        case TrainingPanelTab::KnowledgeGaps: drawKnowledgeGapsTab(renderer, content); break;
        case TrainingPanelTab::ToolGaps:      drawToolGapsTab(renderer, content);      break;
        case TrainingPanelTab::ModelConfig:   drawModelConfigTab(renderer, content);   break;
        case TrainingPanelTab::Tokenizer:     drawTokenizerTab(renderer, content);     break;
    }

    // ── Bottom action bar (Training tab) ──


    // ── Bottom action bar (Tokenizer tab) ──
    if (activeTab_ == TrainingPanelTab::Tokenizer) {
        float barY = position.y + size.y - kBottomBarH - 10.0f;
        float barW = size.x - 20.0f;
        float barX = position.x + 10.0f;
        UIDrawHelpers::drawDivider(renderer, {barX, barY}, barW);
        drawTokenizerBottomBar(renderer, barY + 8.0f, barW, barX);
    }

    // ── Dropdowns on top ──
    if (paramCategoryFilter_ && paramCategoryFilter_->isExpanded())
        paramCategoryFilter_->drawExpandedList(renderer, position);
    if (configModelDropdown_ && configModelDropdown_->isExpanded())
        configModelDropdown_->drawExpandedList(renderer, position);

    renderer.popClipRect();
    return true;
}
