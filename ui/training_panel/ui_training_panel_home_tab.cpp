// UITrainingPanel: Home tab
#include "ui_training_panel_internal.hpp"

using namespace GRIMText;
using namespace UITheme;
using namespace UITrainingPanelDetail;

namespace {

// =========================================================
// Helpers
// =========================================================

static std::string backendDisplayName(GRIM::MMO::BackendType bt) {
    return GRIM::MMO::ModelRegistry::backendTypeToString(bt);
}

static uint32_t residencyStatusColor(GRIM::MMO::ResidencyState state) {
    switch (state) {
        case GRIM::MMO::ResidencyState::Unloaded:      return Colors::TextMuted;
        case GRIM::MMO::ResidencyState::Loading:       return Colors::Warning;
        case GRIM::MMO::ResidencyState::Loaded:        return Colors::Success;
        case GRIM::MMO::ResidencyState::InUse:         return Colors::AccentBlue;
        case GRIM::MMO::ResidencyState::Idle:          return Colors::Info;
        case GRIM::MMO::ResidencyState::EvictEligible: return Colors::Warning;
        case GRIM::MMO::ResidencyState::Unloading:     return Colors::Danger;
    }
    return Colors::TextPrimary;
}

static std::vector<std::string> splitCommaTags(const std::string& input) {
    std::vector<std::string> tags;
    std::istringstream ss(input);
    std::string tag;
    while (std::getline(ss, tag, ',')) {
        size_t start = tag.find_first_not_of(" \t");
        size_t end   = tag.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos)
            tags.push_back(tag.substr(start, end - start + 1));
    }
    return tags;
}

}  // namespace

// ============================================================
// Home Tab — Model Browser
// ============================================================

void UITrainingPanel::drawHomeTab(OverlayRenderer& renderer, const PanelRect& content) {
    float x = content.origin.x + Spacing::PaddingX;
    float y = content.origin.y + Spacing::Small;
    float w = content.size.x - 2.0f * Spacing::PaddingX;

    // Section header + create button
    UIDrawHelpers::drawSectionHeader(renderer, {x - Spacing::PaddingX, y}, content.size.x,
                                     "Registered Models", Colors::SectionAI);
    y += Sizes::HeaderHeight + Spacing::Small;

    // Create model button (top-right of section)
    createModelBtn_->setPosition(x + w - 130.0f, y - Sizes::HeaderHeight - 2.0f);
    createModelBtn_->drawOverlay(renderer, position);

    if (showCreatorForm_) {
        drawCreatorForm(renderer, content);
        return;  // Creator form replaces browser when visible
    }

    drawBrowserView(renderer, content);
}

void UITrainingPanel::drawBrowserView(OverlayRenderer& renderer, const PanelRect& content) {
    float x = content.origin.x + Spacing::PaddingX;
    float y = content.origin.y + Sizes::HeaderHeight + Spacing::Medium + Spacing::Small;
    float w = content.size.x - 2.0f * Spacing::PaddingX;

    // Column headers
    renderer.drawText({x, y}, "Name", Colors::TextSecondary);
    renderer.drawText({x + 200.0f, y}, "Status", Colors::TextSecondary);
    renderer.drawText({x + 340.0f, y}, "Backend", Colors::TextSecondary);
    renderer.drawText({x + 480.0f, y}, "RAM", Colors::TextSecondary);
    renderer.drawText({x + 560.0f, y}, "VRAM", Colors::TextSecondary);
    renderer.drawText({x + 640.0f, y}, "Actions", Colors::TextSecondary);
    y += 22.0f;

    UIDrawHelpers::drawDivider(renderer, {x, y}, w);
    y += 4.0f;

    if (modelEntries_.empty()) {
        renderer.drawText({x, y}, "No models registered. Use '+ New Model' to add one.", Colors::TextMuted);
    }

    for (size_t i = 0; i < modelEntries_.size(); ++i) {
        const auto& entry = modelEntries_[i];

        // Hover highlight
        if (static_cast<int>(i) == hoveredBrowserRow_) {
            renderer.drawRoundedRect({x, y - 1.0f}, {w, kRowHeight}, Colors::ContentAreaBg, 4.0f);
        }

        std::string label = entry.name;
        if (entry.is_router) label += " [R]";

        renderer.drawText({x, y}, label, Colors::TextPrimary);
        renderer.drawText({x + 200.0f, y}, entry.status, entry.statusColor);
        renderer.drawText({x + 340.0f, y}, entry.backend, Colors::TextSecondary);
        renderer.drawText({x + 480.0f, y},
            std::to_string(entry.ram_mb) + " MB", Colors::TextSecondary);
        renderer.drawText({x + 560.0f, y},
            std::to_string(entry.vram_mb) + " MB", Colors::TextSecondary);

        // Action buttons
        if (!entry.is_router) {
            bool isLoaded = (entry.status != "Unloaded" && entry.status != "N/A");
            if (isLoaded) {
                renderer.drawText({x + 640.0f, y}, "[Unload]", Colors::Warning);
            } else {
                renderer.drawText({x + 640.0f, y}, "[Load]", Colors::Success);
            }
            renderer.drawText({x + 710.0f, y}, "[x]", Colors::Danger);
        }

        y += kRowHeight;
    }

    // Resource bars at bottom
    float barY = content.origin.y + content.size.y - 30.0f;
    vramBar_->setPosition(content.origin.x + Spacing::PaddingX, barY);
    ramBar_->setPosition(content.origin.x + content.size.x / 2.0f + 10.0f, barY);
    vramBar_->setSize((content.size.x / 2.0f) - Spacing::PaddingX - 10.0f, 20.0f);
    ramBar_->setSize((content.size.x / 2.0f) - Spacing::PaddingX - 10.0f, 20.0f);
    vramBar_->drawOverlay(renderer, position);
    ramBar_->drawOverlay(renderer, position);
}

void UITrainingPanel::processBrowserClicks(const InputState& input, const PanelRect& content) {
    Vec2 m = input.mousePos;
    float x = content.origin.x + Spacing::PaddingX;
    float dataY = content.origin.y + Sizes::HeaderHeight + Spacing::Medium + Spacing::Small + 26.0f;
    float w = content.size.x - 2.0f * Spacing::PaddingX;

    hoveredBrowserRow_ = -1;

    if (m.x < x || m.x > x + w || m.y < dataY) return;

    int row = static_cast<int>((m.y - dataY) / kRowHeight);
    if (row < 0 || row >= static_cast<int>(modelEntries_.size())) return;

    hoveredBrowserRow_ = row;
    const auto& entry = modelEntries_[static_cast<size_t>(row)];

    if (!input.mousePressed[0]) return;
    if (entry.is_router) return;

    float actionLoadX = x + 640.0f;
    float actionRemoveX = x + 710.0f;

    if (m.x >= actionRemoveX && m.x <= actionRemoveX + 30.0f) {
        handleModelAction(entry.id, "remove");
    } else if (m.x >= actionLoadX && m.x <= actionLoadX + 60.0f) {
        bool isLoaded = (entry.status != "Unloaded" && entry.status != "N/A");
        handleModelAction(entry.id, isLoaded ? "unload" : "load");
    }
}

void UITrainingPanel::handleModelAction(const std::string& model_id,
                                        const std::string& action) {
    auto& registry = GRIM::MMO::ModelRegistry::instance();

    if (action == "load") {
        if (g_modelLoader) {
            auto result = g_modelLoader->ensureLoaded(model_id);
            LOG_DEBUG("UITrainingPanel", "ensureLoaded('" + model_id + "') → "
                + std::string(GRIM::MMO::loadResultToString(result)));
        }
    } else if (action == "unload") {
        if (g_modelLoader) {
            g_modelLoader->unload(model_id);
            LOG_DEBUG("UITrainingPanel", "unloaded '" + model_id + "'");
        }
    } else if (action == "remove") {
        if (g_modelLoader) {
            auto state = g_modelLoader->getState(model_id);
            if (state != GRIM::MMO::ResidencyState::Unloaded) {
                g_modelLoader->unload(model_id);
            }
        }
        registry.removeModel(model_id);
        removeSubModelFromConfig(model_id);
        LOG_DEBUG("UITrainingPanel", "removed model '" + model_id + "'");
        refreshModelList();
    }
}

// ============================================================
// Model Creator Form (overlay on Home tab)
// ============================================================

void UITrainingPanel::drawCreatorForm(OverlayRenderer& renderer, const PanelRect& content) {
    float x = content.origin.x + Spacing::PaddingX;
    float y = content.origin.y + Sizes::HeaderHeight + Spacing::Medium + Spacing::Small;
    float inputX = x + 100.0f;

    UIDrawHelpers::drawSectionHeader(renderer, {x - Spacing::PaddingX, y}, content.size.x,
                                     "Register New Model", Colors::SectionPersonality);
    y += Sizes::HeaderHeight + Spacing::Small;

    auto drawFormRow = [&](const std::string& label, std::shared_ptr<UIInputBox>& input) {
        renderer.drawText({x, y + 4.0f}, label, Colors::TextSecondary);
        input->setPosition(inputX, y);
        input->setSize(400.0f, 26.0f);
        input->drawOverlay(renderer, position);
        y += 32.0f;
    };

    // Generated ID display row
    renderer.drawText({x, y + 4.0f}, "ID:", Colors::TextSecondary);
    renderer.drawText({inputX, y + 4.0f},
        bufId_.empty() ? "(enter name to generate)" : bufId_,
        bufId_.empty() ? Colors::TextMuted : Colors::AccentBlue);
    y += 32.0f;

    drawFormRow("Name:", creatorNameInput_);

    // Params row
    renderer.drawText({x, y + 4.0f}, "Params:", Colors::TextSecondary);
    creatorParamInput_->setPosition(inputX, y);
    creatorParamInput_->setSize(200.0f, 26.0f);
    creatorParamInput_->drawOverlay(renderer, position);
    y += 32.0f;
    drawFormRow("Subject:", creatorSubjectInput_);
    drawFormRow("Tags:", creatorTagsInput_);
    drawFormRow("Desc:", creatorDescInput_);

    // Status label
    creatorStatusLabel_->setPosition(inputX, y);
    creatorStatusLabel_->drawOverlay(renderer, position);
    y += 26.0f;

    // Register / Cancel buttons
    creatorRegisterBtn_->setPosition(inputX, y);
    creatorRegisterBtn_->drawOverlay(renderer, position);
    creatorCancelBtn_->setPosition(inputX + 110.0f, y);
    creatorCancelBtn_->drawOverlay(renderer, position);
}

void UITrainingPanel::clearCreatorFields() {
    bufId_.clear(); bufName_.clear(); bufParamStr_ = "0M";
    bufSubject_.clear();
    bufTags_.clear(); bufDesc_.clear();

    creatorNameInput_->clear();
    creatorParamInput_->setText("0M");
    creatorSubjectInput_->clear();
    creatorTagsInput_->clear();
    creatorDescInput_->clear();
    creatorStatusLabel_->setText("");
}

bool UITrainingPanel::validateCreatorFields(std::string& out_error) const {
    if (bufName_.empty())    { out_error = "Name is required"; return false; }
    if (bufSubject_.empty()) { out_error = "Subject is required (used for ID)"; return false; }
    if (bufId_.empty())      { out_error = "Subject is required to generate ID"; return false; }

    auto& registry = GRIM::MMO::ModelRegistry::instance();
    if (registry.getModelById(bufId_)) {
        out_error = "Model ID '" + bufId_ + "' already exists";
        return false;
    }
    return true;
}

void UITrainingPanel::submitNewModel() {
    std::string error;
    if (!validateCreatorFields(error)) {
        creatorStatusLabel_->setText(error);
        creatorStatusLabel_->setColor(Colors::Danger);
        return;
    }

    GRIM::MMO::ModelInfo model;
    model.id          = bufId_;
    model.name        = bufName_;
    model.subject     = bufSubject_;
    model.description = bufDesc_;
    model.subject_tags = splitCommaTags(bufTags_);

    // Auto-create model directory under model_store
    {
        namespace fs = std::filesystem;
        fs::path modelStoreRoot = resolvePathFromGrimRoot(
            aiConfig.at("paths").at("grim_text").at("model_store").get<std::string>());
        fs::path modelDir = modelStoreRoot / bufId_;
        std::error_code ec;
        fs::create_directories(modelDir, ec);
        if (ec) {
            creatorStatusLabel_->setText("Failed to create model dir: " + ec.message());
            creatorStatusLabel_->setColor(Colors::Danger);
            return;
        }
        model.model_path = modelDir.string();
    }

    model.backend_type = GRIM::MMO::BackendType::GrimTextServer;

    auto& registry = GRIM::MMO::ModelRegistry::instance();
    try {
        registry.registerModel(std::move(model));
    } catch (const std::runtime_error& e) {
        creatorStatusLabel_->setText(e.what());
        creatorStatusLabel_->setColor(Colors::Danger);
        return;
    }

    const auto* registered = registry.getModelById(bufId_);
    if (registered && !persistSubModel(*registered)) {
        creatorStatusLabel_->setText("Registered but failed to save config");
        creatorStatusLabel_->setColor(Colors::Warning);
        return;
    }

    LOG_DEBUG("UITrainingPanel", "Registered new sub-model '" + bufId_ + "'");
    clearCreatorFields();
    showCreatorForm_ = false;
    refreshModelList();
    creatorStatusLabel_->setText("Registered successfully");
    creatorStatusLabel_->setColor(Colors::Success);
}

// ============================================================
// ID generation helpers
// ============================================================

std::string UITrainingPanel::slugifyName(const std::string& name) {
    std::string slug;
    slug.reserve(name.size());
    bool lastWasDash = true; // suppress leading dash
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            slug += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            lastWasDash = false;
        } else if (!lastWasDash && !slug.empty()) {
            slug += '-';
            lastWasDash = true;
        }
    }
    // Strip trailing dash
    while (!slug.empty() && slug.back() == '-')
        slug.pop_back();
    return slug;
}

int64_t UITrainingPanel::parseParamStr(const std::string& s) {
    if (s.empty()) return 0;
    double val = 0.0;
    size_t i = 0;
    while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.'))
        ++i;
    try { val = std::stod(s.substr(0, i)); } catch (...) { return 0; }

    // Optional suffix: K / M / B / T (case-insensitive)
    if (i < s.size()) {
        char suffix = static_cast<char>(std::toupper(static_cast<unsigned char>(s[i])));
        if      (suffix == 'K') val *= 1e3;
        else if (suffix == 'M') val *= 1e6;
        else if (suffix == 'B') val *= 1e9;
        else if (suffix == 'T') val *= 1e12;
    }
    return static_cast<int64_t>(val);
}

std::string UITrainingPanel::formatParamCount(int64_t params) {
    // Format to 3 significant figures with unit suffix
    struct Tier { double divisor; const char* suffix; };
    static const Tier tiers[] = {
        { 1e12, "T" }, { 1e9, "B" }, { 1e6, "M" }, { 1e3, "K" }
    };
    for (const auto& t : tiers) {
        if (params >= static_cast<int64_t>(t.divisor * 0.5)) {
            double v = params / t.divisor;
            char buf[32];
            // 3 significant figures
            if      (v >= 100.0) std::snprintf(buf, sizeof(buf), "%.0f%s", v, t.suffix);
            else if (v >= 10.0)  std::snprintf(buf, sizeof(buf), "%.1f%s", v, t.suffix);
            else                 std::snprintf(buf, sizeof(buf), "%.2f%s", v, t.suffix);
            return buf;
        }
    }
    // Sub-kilo: just raw count
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%lldP", static_cast<long long>(params));
    return buf;
}

int UITrainingPanel::nextVersionForBase(const std::string& base) {
    auto& registry = GRIM::MMO::ModelRegistry::instance();
    for (int v = 1; v <= 9999; ++v) {
        std::string candidate = base + "-v" + std::to_string(v);
        if (!registry.getModelById(candidate))
            return v;
    }
    return 1;
}

void UITrainingPanel::regenerateId() {
    std::string slug = slugifyName(bufSubject_);
    if (slug.empty()) { bufId_.clear(); return; }

    int64_t params = parseParamStr(bufParamStr_.empty() ? "0M" : bufParamStr_);
    std::string paramFmt = formatParamCount(params);

    std::string base = slug + "-" + paramFmt;
    int ver = nextVersionForBase(base);
    bufId_ = base + "-v" + std::to_string(ver);
}

void UITrainingPanel::prefillCreatorFromGap(const KnowledgeGapEntry& gap) {
    setView(TrainingPanelTab::Home);
    showCreatorForm_ = true;
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
    creatorStatusLabel_->setColor(Colors::Warning);
}

// ============================================================
// Model Browser — Data
// ============================================================

void UITrainingPanel::refreshModelList() {
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
            entry.statusColor = residencyStatusColor(state);
        } else {
            entry.status      = "N/A";
            entry.statusColor = Colors::TextMuted;
        }
        modelEntries_.push_back(std::move(entry));
    }
}

// ============================================================
// Config persistence (GRIM runtime config → mmo.sub_models)
// ============================================================

bool UITrainingPanel::persistSubModel(const GRIM::MMO::ModelInfo& model) {
    try {
        const nlohmann::json config = loadGrimRuntimeAiConfig();

        if (!config.contains("mmo") || !config["mmo"].is_object()) {
            LOG_ERROR("UITrainingPanel", "ai_config.json missing 'mmo' section");
            return false;
        }

        nlohmann::json updatedSubModels = config["mmo"].contains("sub_models")
            ? config["mmo"]["sub_models"]
            : nlohmann::json::array();

        updatedSubModels.push_back(
            GRIM::MMO::ModelRegistry::serializeModelToJson(model));

        nlohmann::json pending;
        pending["mmo"]["sub_models"] = std::move(updatedSubModels);
        saveGrimRuntimeAiConfig(pending);

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("UITrainingPanel", std::string("persistSubModel failed: ") + e.what());
        return false;
    }
}

bool UITrainingPanel::removeSubModelFromConfig(const std::string& model_id) {
    try {
        const nlohmann::json config = loadGrimRuntimeAiConfig();

        if (!config.contains("mmo") || !config["mmo"].contains("sub_models"))
            return false;

        nlohmann::json subs = config["mmo"]["sub_models"];
        bool found = false;
        for (auto it = subs.begin(); it != subs.end(); ++it) {
            if (it->value("id", "") == model_id) {
                subs.erase(it);
                found = true;
                break;
            }
        }
        if (!found) return false;

        nlohmann::json pending;
        pending["mmo"]["sub_models"] = std::move(subs);
        saveGrimRuntimeAiConfig(pending);

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("UITrainingPanel", std::string("removeSubModelFromConfig failed: ") + e.what());
        return false;
    }
}

void UITrainingPanel::updateResourceBars() {
    if (!g_resourceSignal) return;

    auto snap = g_resourceSignal->latest();

    totalRamMb_ = static_cast<float>(snap.ram_used_mb + snap.ram_available_mb);
    usedRamMb_  = static_cast<float>(snap.ram_used_mb);
    ramBar_->setMaxValue(totalRamMb_);
    ramBar_->setValue(usedRamMb_);
    ramBar_->setLabel("RAM: " + std::to_string(snap.ram_used_mb) + " / "
        + std::to_string(snap.ram_used_mb + snap.ram_available_mb) + " MB");

    long totalVram = 0, usedVram = 0;
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
