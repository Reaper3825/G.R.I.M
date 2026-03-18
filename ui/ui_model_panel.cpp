// UIModelPanel — Model Creator Suite
// Three-tab interface: Browser, Creator, GapQueue.
// See ui_model_panel.hpp for full documentation.
//======================================================//
#include "ui_model_panel.hpp"
#include "overlay_renderer.hpp"
#include "../core/input_parser.hpp"
#include "../MMO/Core/ModelRegistry.hpp"
#include "../MMO/Core/ModelLoader.hpp"
#include "../MMO/Core/ResourceSignal.hpp"
#include "../resources.hpp"
#include "../logger.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

// External globals from bootstrap
extern GRIM::MMO::ModelLoader*    g_modelLoader;
extern GRIM::MMO::ResourceSignal* g_resourceSignal;

// =========================================================
// Helpers
// =========================================================

static std::string backendDisplayName(GRIM::MMO::BackendType bt) {
    return GRIM::MMO::ModelRegistry::backendTypeToString(bt);
}

static uint32_t statusColor(GRIM::MMO::ResidencyState state) {
    switch (state) {
        case GRIM::MMO::ResidencyState::Unloaded:      return 0xFF888888;  // gray
        case GRIM::MMO::ResidencyState::Loading:       return 0xFFE8D050;  // gold
        case GRIM::MMO::ResidencyState::Loaded:        return 0xFF5AD07A;  // green
        case GRIM::MMO::ResidencyState::InUse:         return 0xFF5B8DEF;  // blue
        case GRIM::MMO::ResidencyState::Idle:          return 0xFF5AD0A0;  // teal
        case GRIM::MMO::ResidencyState::EvictEligible: return 0xFFE8A840;  // amber
        case GRIM::MMO::ResidencyState::Unloading:     return 0xFFE05555;  // coral
    }
    return 0xFFFFFFFF;
}

static std::vector<std::string> splitCommaTags(const std::string& input) {
    std::vector<std::string> tags;
    std::istringstream ss(input);
    std::string tag;
    while (std::getline(ss, tag, ',')) {
        // trim whitespace
        size_t start = tag.find_first_not_of(" \t");
        size_t end   = tag.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos) {
            tags.push_back(tag.substr(start, end - start + 1));
        }
    }
    return tags;
}

static const std::vector<std::string> kBackendOptions = {
    "grim_text_server", "llama_cpp", "ollama", "external"
};

// =========================================================
// Constructor / Destructor
// =========================================================

UIModelPanel::UIModelPanel()
    : UIPanel("Model Registry", true)
{
    setVisible(false);
    setBackground(0xF0202020);  // [GLASS_PHASE4] Opaque dark card

    // Tab buttons — positions set each frame in update()/drawOverlay()
    tabBrowserBtn_ = std::make_shared<UIButton>("Browse", [this]() {
        setView(ModelPanelView::Browser);
    });
    tabBrowserBtn_->setSize(80.0f, 28.0f);

    tabCreatorBtn_ = std::make_shared<UIButton>("Create", [this]() {
        setView(ModelPanelView::Creator);
    });
    tabCreatorBtn_->setSize(80.0f, 28.0f);

    tabGapQueueBtn_ = std::make_shared<UIButton>("Gaps", [this]() {
        setView(ModelPanelView::GapQueue);
    });
    tabGapQueueBtn_->setSize(80.0f, 28.0f);

    addChild(tabBrowserBtn_);
    addChild(tabCreatorBtn_);
    addChild(tabGapQueueBtn_);

    // Browser view widgets — positions set each frame in update()/drawOverlay()
    browserScrollBox_ = std::make_shared<UIScrollBox>();
    browserScrollBox_->setSize(680.0f, 390.0f);

    vramBar_ = std::make_shared<UIProgressBar>("VRAM", 1.0f);
    vramBar_->setSize(330.0f, 20.0f);
    vramBar_->setShowPercentage(true);
    vramBar_->setFillColor(0xFF5B8DEF);

    ramBar_ = std::make_shared<UIProgressBar>("RAM", 1.0f);
    ramBar_->setSize(330.0f, 20.0f);
    ramBar_->setShowPercentage(true);
    ramBar_->setFillColor(0xFF5AD07A);

    // Creator view widgets — positions set each frame in update()/drawOverlay()
    creatorIdInput_ = std::make_shared<UIInputBox>(&bufId_);
    creatorIdInput_->setPlaceholder("model-id (e.g. medical-7b)");
    creatorIdInput_->setSize(400.0f, 26.0f);

    creatorNameInput_ = std::make_shared<UIInputBox>(&bufName_);
    creatorNameInput_->setPlaceholder("Display name");
    creatorNameInput_->setSize(400.0f, 26.0f);

    creatorSubjectInput_ = std::make_shared<UIInputBox>(&bufSubject_);
    creatorSubjectInput_->setPlaceholder("Subject (e.g. medical)");
    creatorSubjectInput_->setSize(400.0f, 26.0f);

    creatorTagsInput_ = std::make_shared<UIInputBox>(&bufTags_);
    creatorTagsInput_->setPlaceholder("Tags (comma-separated)");
    creatorTagsInput_->setSize(400.0f, 26.0f);

    creatorDescInput_ = std::make_shared<UIInputBox>(&bufDesc_);
    creatorDescInput_->setPlaceholder("Description");
    creatorDescInput_->setSize(400.0f, 26.0f);

    creatorPathInput_ = std::make_shared<UIInputBox>(&bufPath_);
    creatorPathInput_->setPlaceholder("Model path (weights file)");
    creatorPathInput_->setSize(400.0f, 26.0f);

    creatorUrlInput_ = std::make_shared<UIInputBox>(&bufUrl_);
    creatorUrlInput_->setPlaceholder("URL (host:port)");
    creatorUrlInput_->setSize(400.0f, 26.0f);

    creatorBackendDropdown_ = std::make_shared<UIDropdown>(
        "Backend", kBackendOptions, 0,
        [](int /*idx*/, const std::string& /*item*/) {});
    creatorBackendDropdown_->setSize(200.0f, 26.0f);

    creatorRamSlider_ = std::make_shared<UISlider>(
        "Est. RAM (MB)", 0.0f, 64000.0f, 0.0f, [](float) {});
    creatorRamSlider_->setSize(400.0f, 26.0f);

    creatorVramSlider_ = std::make_shared<UISlider>(
        "Est. VRAM (MB)", 0.0f, 48000.0f, 0.0f, [](float) {});
    creatorVramSlider_->setSize(400.0f, 26.0f);

    creatorStatusLabel_ = std::make_shared<UILabel>("", 0xFFFFFFFF);
    creatorStatusLabel_->setSize(400.0f, 20.0f);

    creatorRegisterBtn_ = std::make_shared<UIButton>("Register", [this]() {
        submitNewModel();
    });
    creatorRegisterBtn_->setSize(100.0f, 30.0f);

    creatorCancelBtn_ = std::make_shared<UIButton>("Clear", [this]() {
        clearCreatorFields();
    });
    creatorCancelBtn_->setSize(100.0f, 30.0f);

    // Gap queue scroll box — position set each frame in update()/drawOverlay()
    gapScrollBox_ = std::make_shared<UIScrollBox>();
    gapScrollBox_->setSize(680.0f, 430.0f);
}

UIModelPanel::~UIModelPanel() = default;

// =========================================================
// Update
// =========================================================

void UIModelPanel::update(const InputState& input, float dt) {
    if (!isVisible()) return;

    UIPanel::update(input, dt);

    // Reposition tab buttons relative to panel position each frame
    tabBrowserBtn_->setPosition(position.x + 10.0f, position.y + 35.0f);
    tabCreatorBtn_->setPosition(position.x + 95.0f, position.y + 35.0f);
    tabGapQueueBtn_->setPosition(position.x + 180.0f, position.y + 35.0f);

    switch (activeView_) {
        case ModelPanelView::Browser: {
            browserScrollBox_->setPosition(position.x + 10.0f, position.y + 70.0f);
            vramBar_->setPosition(position.x + 10.0f, position.y + 470.0f);
            ramBar_->setPosition(position.x + 350.0f, position.y + 470.0f);

            browserScrollBox_->update(input, dt);
            vramBar_->update(input, dt);
            ramBar_->update(input, dt);

            processBrowserClicks(input);

            browserRefreshTimer_ += dt;
            if (browserRefreshTimer_ >= browserRefreshInterval_) {
                browserRefreshTimer_ = 0.0f;
                refreshModelList();
                updateResourceBars();
            }
            break;
        }
        case ModelPanelView::Creator: {
            float inputX = position.x + 120.0f;
            creatorIdInput_->setPosition(inputX, position.y + 80.0f);
            creatorNameInput_->setPosition(inputX, position.y + 115.0f);
            creatorSubjectInput_->setPosition(inputX, position.y + 150.0f);
            creatorTagsInput_->setPosition(inputX, position.y + 185.0f);
            creatorDescInput_->setPosition(inputX, position.y + 220.0f);
            creatorPathInput_->setPosition(inputX, position.y + 255.0f);
            creatorUrlInput_->setPosition(inputX, position.y + 290.0f);
            creatorBackendDropdown_->setPosition(inputX, position.y + 325.0f);
            creatorRamSlider_->setPosition(inputX, position.y + 365.0f);
            creatorVramSlider_->setPosition(inputX, position.y + 400.0f);
            creatorStatusLabel_->setPosition(inputX, position.y + 440.0f);
            creatorRegisterBtn_->setPosition(inputX, position.y + 470.0f);
            creatorCancelBtn_->setPosition(position.x + 230.0f, position.y + 470.0f);

            creatorIdInput_->update(input, dt);
            creatorNameInput_->update(input, dt);
            creatorSubjectInput_->update(input, dt);
            creatorTagsInput_->update(input, dt);
            creatorDescInput_->update(input, dt);
            creatorPathInput_->update(input, dt);
            creatorUrlInput_->update(input, dt);
            creatorBackendDropdown_->update(input, dt);
            creatorRamSlider_->update(input, dt);
            creatorVramSlider_->update(input, dt);
            creatorRegisterBtn_->update(input, dt);
            creatorCancelBtn_->update(input, dt);
            break;
        }
        case ModelPanelView::GapQueue: {
            gapScrollBox_->setPosition(position.x + 10.0f, position.y + 70.0f);
            gapScrollBox_->update(input, dt);
            processGapClicks(input);
            break;
        }
    }
}

// =========================================================
// Draw
// =========================================================

bool UIModelPanel::drawOverlay(OverlayRenderer& renderer) {
    if (!UIPanel::drawOverlay(renderer)) return false;

    tabBrowserBtn_->setPosition(position.x + 10.0f, position.y + 35.0f);
    tabCreatorBtn_->setPosition(position.x + 95.0f, position.y + 35.0f);
    tabGapQueueBtn_->setPosition(position.x + 180.0f, position.y + 35.0f);
    tabBrowserBtn_->drawOverlay(renderer, position);
    tabCreatorBtn_->drawOverlay(renderer, position);
    tabGapQueueBtn_->drawOverlay(renderer, position);

    float tabW = 80.0f;
    float indicatorY = position.y + 63.0f;
    float indicatorX = position.x + 10.0f;
    if (activeView_ == ModelPanelView::Creator) indicatorX = position.x + 95.0f;
    else if (activeView_ == ModelPanelView::GapQueue) indicatorX = position.x + 180.0f;
    renderer.drawRect({indicatorX, indicatorY}, {tabW, 2.0f}, 0xFF6B8CFF);

    switch (activeView_) {
        case ModelPanelView::Browser:
            drawBrowserView(renderer);
            break;
        case ModelPanelView::Creator:
            drawCreatorView(renderer);
            break;
        case ModelPanelView::GapQueue:
            drawGapQueueView(renderer);
            break;
    }
    
    renderer.popClipRect();
    return true;
}

// =========================================================
// View control
// =========================================================

void UIModelPanel::setView(ModelPanelView view) {
    activeView_ = view;
    if (view == ModelPanelView::Browser) {
        refreshModelList();
        updateResourceBars();
    }
}

// =========================================================
// Knowledge gap intake
// =========================================================

void UIModelPanel::pushKnowledgeGap(KnowledgeGapEntry entry) {
    std::lock_guard<std::mutex> lock(gapMutex_);
    gapQueue_.push_back(std::move(entry));
}

size_t UIModelPanel::pendingGapCount() const {
    std::lock_guard<std::mutex> lock(gapMutex_);
    return gapQueue_.size();
}

// =========================================================
// Browser view
// =========================================================

void UIModelPanel::refreshModelList() {
    auto& registry = GRIM::MMO::ModelRegistry::instance();
    auto allModels = registry.getAllModels();
    const auto* router = registry.getRouter();

    modelEntries_.clear();
    modelEntries_.reserve(allModels.size());

    for (const auto* model : allModels) {
        ModelListEntry entry;
        entry.id       = model->id;
        entry.name     = model->name;
        entry.subject  = model->subject;
        entry.backend  = backendDisplayName(model->backend_type);
        entry.ram_mb   = model->estimated_ram_mb;
        entry.vram_mb  = model->estimated_vram_mb;
        entry.is_router = (router && model->id == router->id);

        if (g_modelLoader) {
            auto state = g_modelLoader->getState(model->id);
            entry.status      = GRIM::MMO::residencyStateToString(state);
            entry.statusColor = statusColor(state);
        } else {
            entry.status      = "N/A";
            entry.statusColor = 0xFF888888;
        }

        modelEntries_.push_back(std::move(entry));
    }
}

void UIModelPanel::drawBrowserView(OverlayRenderer& renderer) {
    Vec2 pos = getPosition();
    float x = pos.x + 10.0f;
    float y = pos.y + kHeaderY;

    // Reposition scroll box relative to panel
    browserScrollBox_->setPosition(pos.x + 10.0f, pos.y + 70.0f);

    // Column headers
    renderer.drawText({x, y}, "Name", 0xFFCCCCCC);
    renderer.drawText({x + 180.0f, y}, "Status", 0xFFCCCCCC);
    renderer.drawText({x + 300.0f, y}, "Backend", 0xFFCCCCCC);
    renderer.drawText({x + 420.0f, y}, "RAM", 0xFFCCCCCC);
    renderer.drawText({x + 490.0f, y}, "VRAM", 0xFFCCCCCC);
    renderer.drawText({x + 560.0f, y}, "Actions", 0xFFCCCCCC);
    y += 22.0f;

    renderer.drawLine({x, y}, {x + 660.0f, y}, 0x10FFFFFF, 1.0f);  // [GLASS_PHASE4] Faint separator
    y += 4.0f;

    if (modelEntries_.empty()) {
        renderer.drawText({x, y}, "No models registered. Use the Create tab to add one.", 0xFF888888);
    }

    for (size_t i = 0; i < modelEntries_.size(); ++i) {
        const auto& entry = modelEntries_[i];

        // Hover highlight
        if (static_cast<int>(i) == hoveredBrowserRow_) {
            renderer.drawRect({x, y - 1.0f}, {660.0f, kRowHeight}, 0x15FFFFFF);  // [GLASS_PHASE4] Dim hover
        }

        std::string label = entry.name;
        if (entry.is_router) label += " [R]";

        renderer.drawText({x, y}, label, 0xFFFFFFFF);
        renderer.drawText({x + 180.0f, y}, entry.status, entry.statusColor);
        renderer.drawText({x + 300.0f, y}, entry.backend, 0xFFAAAAAA);
        renderer.drawText({x + 420.0f, y},
            std::to_string(entry.ram_mb) + " MB", 0xFFAAAAAA);
        renderer.drawText({x + 490.0f, y},
            std::to_string(entry.vram_mb) + " MB", 0xFFAAAAAA);

        // Action buttons — different depending on state
        if (!entry.is_router) {
            // Load/Unload toggle
            bool isLoaded = (entry.status != "Unloaded" && entry.status != "N/A");
            if (isLoaded) {
                renderer.drawText({x + 560.0f, y}, "[Unload]", 0xFFE8A840);
            } else {
                renderer.drawText({x + 560.0f, y}, "[Load]", 0xFF5AD07A);
            }
            renderer.drawText({x + 622.0f, y}, "[x]", 0xFFE05555);
        }

        y += kRowHeight;
    }

    // Resource bars — reposition relative to panel
    vramBar_->setPosition(position.x + 10.0f, position.y + 470.0f);
    ramBar_->setPosition(position.x + 350.0f, position.y + 470.0f);
    vramBar_->drawOverlay(renderer, pos);
    ramBar_->drawOverlay(renderer, pos);
}

// =========================================================
// Browser click processing
// =========================================================

void UIModelPanel::processBrowserClicks(const InputState& input) {
    Vec2 m = input.mousePos;
    float x = position.x + 10.0f;
    float dataY = position.y + kDataStartY;

    // Reset hover
    hoveredBrowserRow_ = -1;

    // Check if mouse is in the data area
    if (m.x < x || m.x > x + 660.0f || m.y < dataY) return;

    int row = static_cast<int>((m.y - dataY) / kRowHeight);
    if (row < 0 || row >= static_cast<int>(modelEntries_.size())) return;

    hoveredBrowserRow_ = row;
    const auto& entry = modelEntries_[static_cast<size_t>(row)];

    // Only process clicks
    if (!input.mousePressed[0]) return;
    if (entry.is_router) return;

    // Check action column hit regions
    float actionLoadX = x + 560.0f;
    float actionRemoveX = x + 622.0f;

    if (m.x >= actionRemoveX && m.x <= actionRemoveX + 30.0f) {
        handleModelAction(entry.id, "remove");
    } else if (m.x >= actionLoadX && m.x <= actionLoadX + 60.0f) {
        bool isLoaded = (entry.status != "Unloaded" && entry.status != "N/A");
        handleModelAction(entry.id, isLoaded ? "unload" : "load");
    }
}

void UIModelPanel::handleModelAction(const std::string& model_id,
                                     const std::string& action) {
    auto& registry = GRIM::MMO::ModelRegistry::instance();

    if (action == "load") {
        if (g_modelLoader) {
            auto result = g_modelLoader->ensureLoaded(model_id);
            LOG_DEBUG("UIModelPanel", "ensureLoaded('" + model_id + "') → "
                + std::string(GRIM::MMO::loadResultToString(result)));
        }
    } else if (action == "unload") {
        if (g_modelLoader) {
            g_modelLoader->unload(model_id);
            LOG_DEBUG("UIModelPanel", "unloaded '" + model_id + "'");
        }
    } else if (action == "remove") {
        // Unload first if loaded
        if (g_modelLoader) {
            auto state = g_modelLoader->getState(model_id);
            if (state != GRIM::MMO::ResidencyState::Unloaded) {
                g_modelLoader->unload(model_id);
            }
        }
        // Remove from registry
        registry.removeModel(model_id);
        // Remove from config file
        removeSubModelFromConfig(model_id);
        LOG_DEBUG("UIModelPanel", "removed model '" + model_id + "'");
        refreshModelList();
    }
}

// =========================================================
// Creator view
// =========================================================

void UIModelPanel::drawCreatorView(OverlayRenderer& renderer) {
    Vec2 pos = getPosition();
    float labelX = pos.x + 15.0f;
    float inputX = pos.x + 120.0f;

    renderer.drawText({labelX, pos.y + 84.0f},  "ID:",      0xFFCCCCCC);
    renderer.drawText({labelX, pos.y + 119.0f}, "Name:",    0xFFCCCCCC);
    renderer.drawText({labelX, pos.y + 154.0f}, "Subject:", 0xFFCCCCCC);
    renderer.drawText({labelX, pos.y + 189.0f}, "Tags:",    0xFFCCCCCC);
    renderer.drawText({labelX, pos.y + 224.0f}, "Desc:",    0xFFCCCCCC);
    renderer.drawText({labelX, pos.y + 259.0f}, "Path:",    0xFFCCCCCC);
    renderer.drawText({labelX, pos.y + 294.0f}, "URL:",     0xFFCCCCCC);

    // Reposition all creator widgets relative to panel
    creatorIdInput_->setPosition(inputX, pos.y + 80.0f);
    creatorNameInput_->setPosition(inputX, pos.y + 115.0f);
    creatorSubjectInput_->setPosition(inputX, pos.y + 150.0f);
    creatorTagsInput_->setPosition(inputX, pos.y + 185.0f);
    creatorDescInput_->setPosition(inputX, pos.y + 220.0f);
    creatorPathInput_->setPosition(inputX, pos.y + 255.0f);
    creatorUrlInput_->setPosition(inputX, pos.y + 290.0f);
    creatorBackendDropdown_->setPosition(inputX, pos.y + 325.0f);
    creatorRamSlider_->setPosition(inputX, pos.y + 365.0f);
    creatorVramSlider_->setPosition(inputX, pos.y + 400.0f);
    creatorStatusLabel_->setPosition(inputX, pos.y + 440.0f);
    creatorRegisterBtn_->setPosition(inputX, pos.y + 470.0f);
    creatorCancelBtn_->setPosition(pos.x + 230.0f, pos.y + 470.0f);

    creatorIdInput_->drawOverlay(renderer, pos);
    creatorNameInput_->drawOverlay(renderer, pos);
    creatorSubjectInput_->drawOverlay(renderer, pos);
    creatorTagsInput_->drawOverlay(renderer, pos);
    creatorDescInput_->drawOverlay(renderer, pos);
    creatorPathInput_->drawOverlay(renderer, pos);
    creatorUrlInput_->drawOverlay(renderer, pos);
    creatorBackendDropdown_->drawOverlay(renderer, pos);
    creatorRamSlider_->drawOverlay(renderer, pos);
    creatorVramSlider_->drawOverlay(renderer, pos);
    creatorStatusLabel_->drawOverlay(renderer, pos);
    creatorRegisterBtn_->drawOverlay(renderer, pos);
    creatorCancelBtn_->drawOverlay(renderer, pos);
}

void UIModelPanel::clearCreatorFields() {
    bufId_.clear();
    bufName_.clear();
    bufSubject_.clear();
    bufTags_.clear();
    bufDesc_.clear();
    bufPath_.clear();
    bufUrl_.clear();

    creatorIdInput_->clear();
    creatorNameInput_->clear();
    creatorSubjectInput_->clear();
    creatorTagsInput_->clear();
    creatorDescInput_->clear();
    creatorPathInput_->clear();
    creatorUrlInput_->clear();
    creatorRamSlider_->setValue(0.0f);
    creatorVramSlider_->setValue(0.0f);
    creatorBackendDropdown_->setSelectedIndex(0);
    creatorStatusLabel_->setText("");
}

bool UIModelPanel::validateCreatorFields(std::string& out_error) const {
    if (bufId_.empty()) {
        out_error = "Model ID is required";
        return false;
    }
    if (bufName_.empty()) {
        out_error = "Name is required";
        return false;
    }
    if (bufUrl_.empty()) {
        out_error = "URL is required";
        return false;
    }

    // Check ID uniqueness
    auto& registry = GRIM::MMO::ModelRegistry::instance();
    if (registry.getModelById(bufId_)) {
        out_error = "Model ID '" + bufId_ + "' already exists";
        return false;
    }

    return true;
}

void UIModelPanel::submitNewModel() {
    std::string error;
    if (!validateCreatorFields(error)) {
        creatorStatusLabel_->setText(error);
        creatorStatusLabel_->setColor(0xFFE05555);
        return;
    }

    GRIM::MMO::ModelInfo model;
    model.id          = bufId_;
    model.name        = bufName_;
    model.subject     = bufSubject_;
    model.description = bufDesc_;
    model.model_path  = bufPath_;
    model.url         = bufUrl_;
    model.subject_tags = splitCommaTags(bufTags_);
    model.estimated_ram_mb  = static_cast<long>(creatorRamSlider_->getValue());
    model.estimated_vram_mb = static_cast<long>(creatorVramSlider_->getValue());

    // Parse backend type from dropdown selection
    std::string backendStr = creatorBackendDropdown_->getSelectedItem();
    model.backend_type = GRIM::MMO::ModelRegistry::parseBackendType(backendStr);

    // Register in runtime registry
    auto& registry = GRIM::MMO::ModelRegistry::instance();
    try {
        registry.registerModel(std::move(model));
    } catch (const std::runtime_error& e) {
        creatorStatusLabel_->setText(e.what());
        creatorStatusLabel_->setColor(0xFFE05555);
        return;
    }

    // Persist to ai_config.json
    const auto* registered = registry.getModelById(bufId_);
    if (registered && !persistSubModel(*registered)) {
        creatorStatusLabel_->setText("Registered but failed to save config");
        creatorStatusLabel_->setColor(0xFFE8A840);
        return;
    }

    creatorStatusLabel_->setText("Registered: " + bufId_);
    creatorStatusLabel_->setColor(0xFF5AD07A);
    LOG_DEBUG("UIModelPanel", "Registered new sub-model '" + bufId_ + "'");

    clearCreatorFields();
    creatorStatusLabel_->setText("Registered successfully");
    creatorStatusLabel_->setColor(0xFF5AD07A);
}

void UIModelPanel::prefillCreatorFromGap(const KnowledgeGapEntry& gap) {
    setView(ModelPanelView::Creator);
    bufSubject_ = gap.subject;
    creatorSubjectInput_->setText(gap.subject);

    std::string tagStr;
    for (size_t i = 0; i < gap.tags.size(); ++i) {
        if (i > 0) tagStr += ", ";
        tagStr += gap.tags[i];
    }
    bufTags_ = tagStr;
    creatorTagsInput_->setText(tagStr);

    creatorStatusLabel_->setText("Pre-filled from knowledge gap");
    creatorStatusLabel_->setColor(0xFFE8D050);
}

// =========================================================
// Gap queue view
// =========================================================

void UIModelPanel::drawGapQueueView(OverlayRenderer& renderer) {
    std::lock_guard<std::mutex> lock(gapMutex_);

    Vec2 pos = getPosition();
    float x = pos.x + 10.0f;
    float y = pos.y + kHeaderY;

    // Reposition scroll box relative to panel
    gapScrollBox_->setPosition(pos.x + 10.0f, pos.y + 70.0f);

    if (gapQueue_.empty()) {
        renderer.drawText({x, y}, "No knowledge gaps queued.", 0xFF888888);
        renderer.drawText({x, y + 20.0f},
            "Gaps appear when the router cannot find a matching sub-model.", 0xFF666666);
        return;
    }

    renderer.drawText({x, y}, "Subject", 0xFFCCCCCC);
    renderer.drawText({x + 200.0f, y}, "Tags", 0xFFCCCCCC);
    renderer.drawText({x + 480.0f, y}, "Actions", 0xFFCCCCCC);
    y += 22.0f;

    renderer.drawLine({x, y}, {x + 660.0f, y}, 0x10FFFFFF, 1.0f);  // [GLASS_PHASE4] Faint separator
    y += 4.0f;

    for (size_t i = 0; i < gapQueue_.size(); ++i) {
        const auto& gap = gapQueue_[i];

        // Hover highlight
        if (static_cast<int>(i) == hoveredGapRow_) {
            renderer.drawRect({x, y - 1.0f}, {660.0f, kRowHeight}, 0x15FFFFFF);  // [GLASS_PHASE4] Dim hover
        }

        renderer.drawText({x, y}, gap.subject, 0xFFFFFFFF);

        std::string tagStr;
        for (size_t t = 0; t < gap.tags.size(); ++t) {
            if (t > 0) tagStr += ", ";
            tagStr += gap.tags[t];
        }
        renderer.drawText({x + 200.0f, y}, tagStr, 0xFFAAAAAA);
        renderer.drawText({x + 480.0f, y}, "[Create]", 0xFF6B8CFF);
        renderer.drawText({x + 550.0f, y}, "[Dismiss]", 0xFFE05555);

        y += kRowHeight;
    }
}

// =========================================================
// Gap queue click processing
// =========================================================

void UIModelPanel::processGapClicks(const InputState& input) {
    Vec2 m = input.mousePos;
    float x = position.x + 10.0f;
    float dataY = position.y + kDataStartY;

    hoveredGapRow_ = -1;

    if (m.x < x || m.x > x + 660.0f || m.y < dataY) return;

    // Scoped lock for reading gap queue state
    KnowledgeGapEntry gapCopy;
    int action = 0; // 0=none, 1=dismiss, 2=create
    {
        std::lock_guard<std::mutex> lock(gapMutex_);
        if (gapQueue_.empty()) return;

        int row = static_cast<int>((m.y - dataY) / kRowHeight);
        if (row < 0 || row >= static_cast<int>(gapQueue_.size())) return;

        hoveredGapRow_ = row;

        if (!input.mousePressed[0]) return;

        float createX  = x + 480.0f;
        float dismissX = x + 550.0f;

        if (m.x >= dismissX && m.x <= dismissX + 70.0f) {
            gapQueue_.erase(gapQueue_.begin() + static_cast<ptrdiff_t>(row));
            hoveredGapRow_ = -1;
            action = 1;
        } else if (m.x >= createX && m.x <= createX + 60.0f) {
            gapCopy = gapQueue_[static_cast<size_t>(row)];
            gapQueue_.erase(gapQueue_.begin() + static_cast<ptrdiff_t>(row));
            hoveredGapRow_ = -1;
            action = 2;
        }
    }

    if (action == 2) {
        prefillCreatorFromGap(gapCopy);
    }
}

void UIModelPanel::dismissGap(size_t index) {
    std::lock_guard<std::mutex> lock(gapMutex_);
    if (index < gapQueue_.size()) {
        gapQueue_.erase(gapQueue_.begin() + static_cast<ptrdiff_t>(index));
    }
}

void UIModelPanel::createFromGap(size_t index) {
    KnowledgeGapEntry gap;
    {
        std::lock_guard<std::mutex> lock(gapMutex_);
        if (index >= gapQueue_.size()) return;
        gap = gapQueue_[index];
        gapQueue_.erase(gapQueue_.begin() + static_cast<ptrdiff_t>(index));
    }
    prefillCreatorFromGap(gap);
}

// =========================================================
// Config persistence (ai_config.json → mmo.sub_models)
// =========================================================

bool UIModelPanel::persistSubModel(const GRIM::MMO::ModelInfo& model) {
    namespace fs = std::filesystem;

    try {
        // Read current config
        nlohmann::json config;
        {
            std::ifstream f(AI_CONFIG_FILE);
            if (!f.is_open()) {
                LOG_ERROR("UIModelPanel", "Cannot open ai_config.json for reading");
                return false;
            }
            f >> config;
        }

        if (!config.contains("mmo") || !config["mmo"].is_object()) {
            LOG_ERROR("UIModelPanel", "ai_config.json missing 'mmo' section");
            return false;
        }

        // Ensure sub_models array exists
        if (!config["mmo"].contains("sub_models")) {
            config["mmo"]["sub_models"] = nlohmann::json::array();
        }

        // Append new model
        config["mmo"]["sub_models"].push_back(
            GRIM::MMO::ModelRegistry::serializeModelToJson(model));

        // Write-temp + rename for atomicity
        const std::string tmpPath = std::string(AI_CONFIG_FILE) + ".tmp";
        {
            std::ofstream out(tmpPath);
            if (!out.is_open()) {
                LOG_ERROR("UIModelPanel", "Cannot write temp config file");
                return false;
            }
            out << config.dump(4);
        }
        fs::rename(tmpPath, AI_CONFIG_FILE);

        // Update in-memory global
        aiConfig = config;

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("UIModelPanel", std::string("persistSubModel failed: ") + e.what());
        return false;
    }
}

bool UIModelPanel::removeSubModelFromConfig(const std::string& model_id) {
    namespace fs = std::filesystem;

    try {
        nlohmann::json config;
        {
            std::ifstream f(AI_CONFIG_FILE);
            if (!f.is_open()) return false;
            f >> config;
        }

        if (!config.contains("mmo") || !config["mmo"].contains("sub_models")) {
            return false;
        }

        auto& subs = config["mmo"]["sub_models"];
        bool found = false;
        for (auto it = subs.begin(); it != subs.end(); ++it) {
            if (it->value("id", "") == model_id) {
                subs.erase(it);
                found = true;
                break;
            }
        }

        if (!found) return false;

        // Write-temp + rename for atomicity
        const std::string tmpPath = std::string(AI_CONFIG_FILE) + ".tmp";
        {
            std::ofstream out(tmpPath);
            if (!out.is_open()) return false;
            out << config.dump(4);
        }
        fs::rename(tmpPath, AI_CONFIG_FILE);

        // Update in-memory global
        aiConfig = config;

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("UIModelPanel", std::string("removeSubModelFromConfig failed: ") + e.what());
        return false;
    }
}

// =========================================================
// Resource display
// =========================================================

void UIModelPanel::updateResourceBars() {
    if (!g_resourceSignal) return;

    auto snap = g_resourceSignal->latest();

    // RAM
    totalRamMb_ = static_cast<float>(snap.ram_used_mb + snap.ram_available_mb);
    usedRamMb_  = static_cast<float>(snap.ram_used_mb);
    ramBar_->setMaxValue(totalRamMb_);
    ramBar_->setValue(usedRamMb_);
    ramBar_->setLabel("RAM: " + std::to_string(snap.ram_used_mb) + " / "
        + std::to_string(snap.ram_used_mb + snap.ram_available_mb) + " MB");

    // VRAM (sum across GPUs)
    long totalVram = 0;
    long usedVram  = 0;
    for (const auto& gpu : snap.gpus) {
        totalVram += gpu.vram_used_mb + gpu.vram_free_mb;
        usedVram  += gpu.vram_used_mb;
    }
    totalVramMb_ = static_cast<float>(totalVram);
    usedVramMb_  = static_cast<float>(usedVram);
    vramBar_->setMaxValue(totalVramMb_);
    vramBar_->setValue(usedVramMb_);
    vramBar_->setLabel("VRAM: " + std::to_string(usedVram) + " / "
        + std::to_string(totalVram) + " MB");
}

// =========================================================
// Focus query
// =========================================================

bool UIModelPanel::isAnyInputEditing() const {
    if (activeView_ != ModelPanelView::Creator) return false;
    return creatorIdInput_->isFocused()
        || creatorNameInput_->isFocused()
        || creatorSubjectInput_->isFocused()
        || creatorTagsInput_->isFocused()
        || creatorDescInput_->isFocused()
        || creatorPathInput_->isFocused()
        || creatorUrlInput_->isFocused();
}
