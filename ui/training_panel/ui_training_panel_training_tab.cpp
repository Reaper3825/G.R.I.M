// UITrainingPanel: Model Config tab and hyperparameter browser
#include "ui_training_panel_internal.hpp"

#include <cctype>
#include <cstdlib>
#include <iterator>
#include <set>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace GRIMText;
using namespace UITheme;
using namespace UITrainingPanelDetail;

// ============================================================
// Hyperparameter Snapshot Loader
// ============================================================

void UITrainingPanel::loadHyperparamSnapshot() {
    try {
        configPresetDocument_ = loadGrimRuntimeAiConfig();
        if (!configPresetDocument_.contains("training") ||
            !configPresetDocument_.at("training").is_object()) {
            throw std::runtime_error("ai_config.json missing object: training");
        }
        nlohmann::json& training = configPresetDocument_.at("training");
        if (!training.contains("config") || !training.at("config").is_object()) {
            throw std::runtime_error("ai_config.json missing object: training.config");
        }
        nlohmann::json& sourceConfig = training["config"];
        if (!sourceConfig.contains("causal_mask")) sourceConfig["causal_mask"] = true;
        if (!sourceConfig.contains("use_pre_norm")) sourceConfig["use_pre_norm"] = true;
        if (!sourceConfig.contains("fuse_qkv")) sourceConfig["fuse_qkv"] = true;
        if (!sourceConfig.contains("rms_epsilon")) sourceConfig["rms_epsilon"] = 1.0e-5;
        hyperparamRegistry_.populateModelConfigSchema(sourceConfig);
        hyperparamsLoaded_ = true;
        configPresetDirty_ = false;
        configCompileStatus_ = "FlatBuffer model schema loaded";
        configCompileSuccess_ = true;
    } catch (const std::exception& e) {
        LOG_ERROR("UITrainingPanel", std::string("Failed to load hyperparameter snapshot: ") + e.what());
        hyperparamsLoaded_ = false;
    }
}

// ============================================================
// Parameter Browser (left-side scroll panel)
// ============================================================

void UITrainingPanel::drawParamBrowser(OverlayRenderer& renderer, const Vec2& origin, const Vec2& sz) {
    float x = origin.x;
    float y = origin.y;
    float w = sz.x;
    float h = sz.y;

    // Header
    UIDrawHelpers::drawSectionHeader(renderer, {x, y}, w,
                                     "FlatBuffer Schema Fields", Colors::SectionAI);
    y += Sizes::HeaderHeight + Spacing::Small;

    // Category filter dropdown
    if (paramCategoryFilter_) {
        paramCategoryFilter_->setPosition(x + Spacing::PaddingX, y);
        paramCategoryFilter_->setSize(w - 2.0f * Spacing::PaddingX, 26.0f);
        paramCategoryFilter_->drawOverlay(renderer, position);
        y += 26.0f + Spacing::Small;
    }

    if (!hyperparamsLoaded_) {
        renderer.drawText({x + Spacing::PaddingX, y + 10.0f},
                          "No model schema fields loaded", Colors::TextMuted);
        return;
    }

    // Determine active category filter
    std::string activeCategory;
    if (selectedParamCategory_ > 0) {
        const auto& cats = hyperparamRegistry_.categories();
        int catIdx = selectedParamCategory_ - 1; // 0 = "All"
        if (catIdx >= 0 && catIdx < static_cast<int>(cats.size())) {
            activeCategory = cats[static_cast<size_t>(catIdx)];
        }
    }

    auto filteredParams = hyperparamRegistry_.filtered(activeCategory);
    if (filteredParams.empty()) {
        renderer.drawText({x + Spacing::PaddingX, y + 10.0f},
                          "No fields in this schema table", Colors::TextMuted);
        return;
    }

    // Scroll area background
    float listH = h - (y - origin.y) - Spacing::Small;
    if (listH < 60.0f) listH = 60.0f;

    renderer.drawRoundedRect({x, y}, {w, listH}, Colors::ContentAreaBg, Sizes::WidgetRadius);
    renderer.drawRoundedBorder({x, y}, {w, listH}, Colors::BorderSubtle, Sizes::WidgetRadius);

    // Row dimensions
    constexpr float rowH = 22.0f;
    constexpr float padX = 8.0f;
    float nameColW = w * 0.55f;
    // Scroll content
    float totalContentH = computeParamBrowserContentHeight(filteredParams, activeCategory, rowH);
    float maxScroll = std::max(0.0f, totalContentH - listH);
    if (paramScrollOffset_ > maxScroll) paramScrollOffset_ = maxScroll;
    if (paramScrollOffset_ < 0.0f) paramScrollOffset_ = 0.0f;

    const ParamScrollbarMetrics scrollbar = computeParamScrollbarMetrics(
        x, y, w, listH, totalContentH, paramScrollOffset_);

    float drawY = y + 2.0f - paramScrollOffset_;
    std::string lastCategory;

    for (size_t i = 0; i < filteredParams.size(); ++i) {
        const auto* entry = filteredParams[i];

        // Category sub-header (when showing "All")
        if (activeCategory.empty() && entry->category != lastCategory) {
            lastCategory = entry->category;
            // Draw category label row
            if (drawY + rowH >= y && drawY <= y + listH) {
                renderer.drawRect({x + 2.0f, drawY}, {w - 4.0f, rowH}, Colors::TableHeaderBg);
                renderer.drawText({x + padX, drawY + 3.0f}, entry->category, Colors::TextHeader);
            }
            drawY += rowH;
            if (drawY > y + listH) break;
        }

        // Skip rows above visible area
        if (drawY + rowH < y) { drawY += rowH; continue; }
        // Stop below visible area
        if (drawY > y + listH) break;

        // Only draw visible rows
        if (drawY >= y && drawY + rowH <= y + listH) {
            // Alternating row background
            uint32_t rowBg = (i % 2 == 0) ? Colors::RowEven : Colors::RowOdd;
            if (static_cast<int>(i) == hoveredParamRow_) {
                rowBg = Colors::RowHover;
            }
            renderer.drawRect({x + 2.0f, drawY}, {w - 4.0f, rowH}, rowBg);

            // Parameter name
            renderer.drawText({x + padX, drawY + 3.0f}, entry->display_name, Colors::TextLight);

            // Value column — inline edit or read-only display
            if (static_cast<int>(i) == editingParamIndex_ && paramEditInput_ &&
                entry->type != GRIM::Config::HyperparamType::Bool) {
                // Draw the input box for the editing row
                float editX = x + nameColW - 2.0f;
                float editRightPad = scrollbar.visible ? (kParamBrowserScrollbarWidth + 8.0f) : 8.0f;
                float editW = w - nameColW - padX - editRightPad;
                paramEditInput_->setPosition(editX, drawY);
                paramEditInput_->setSize(editW, rowH);
                paramEditInput_->drawOverlay(renderer, position);
            } else {
                std::string valStr = entry->valueAsString();

                // Color-code booleans
                uint32_t valColor = Colors::TextValue;
                if (entry->type == GRIM::Config::HyperparamType::Bool) {
                    valColor = entry->value.get<bool>() ? Colors::Success : Colors::TextMuted;
                }

                renderer.drawText({x + nameColW, drawY + 3.0f}, valStr, valColor);
            }
        }

        drawY += rowH;
    }

    // Scrollbar indicator if content overflows
    if (scrollbar.visible) {
        renderer.drawRoundedRect({scrollbar.trackX, scrollbar.trackY},
                                 {scrollbar.trackW, scrollbar.trackH},
                                 Colors::TableHeaderBg,
                                 3.0f);
        const uint32_t thumbColor = draggingParamScrollbar_
            ? Colors::ScrollThumbDrag
            : (hoveredParamScrollbar_ ? Colors::ScrollThumbHover : Colors::ScrollThumb);
        renderer.drawRoundedRect({scrollbar.trackX, scrollbar.thumbY},
                                 {scrollbar.trackW, scrollbar.thumbH},
                                 thumbColor,
                                 3.0f);
        renderer.drawRoundedBorder({scrollbar.trackX, scrollbar.thumbY},
                                   {scrollbar.trackW, scrollbar.thumbH},
                                   Colors::BorderPrimary,
                                   3.0f);
    }
}

// ============================================================
// Param Browser Click Handling
// ============================================================

void UITrainingPanel::processParamBrowserClicks(const InputState& input) {
    using namespace UITheme;

    if (!hyperparamsLoaded_) return;
    if (!input.mousePressed[0]) return;  // left click only

    // Reconstruct the same geometry as drawParamBrowser
    auto content = getContentRect();
    content.origin.y += (kContentTopY - kTabBarY);
    content.size.y -= (kContentTopY - kTabBarY);
    float x = content.origin.x;
    float w = kParamBrowserW;
    float y = content.origin.y;

    // Skip header + dropdown (must match drawParamBrowser layout)
    y += Sizes::HeaderHeight + Spacing::Small;   // "Model Parameters" header
    y += 26.0f + Spacing::Small;                 // category dropdown

    float listH = content.size.y - (y - content.origin.y) - Spacing::Small;
    if (listH < 60.0f) listH = 60.0f;
    float nameColW = w * 0.55f;
    constexpr float rowH = 22.0f;

    Vec2 m = input.mousePos;

    // Check if click is within the list area at all
    if (m.x < x || m.x > x + w || m.y < y || m.y > y + listH) {
        // Clicked outside the param browser — commit any active edit
        if (editingParamIndex_ >= 0) commitParamEdit();
        return;
    }

    // Get the current filtered list
    std::string catFilter;
    if (paramCategoryFilter_ && selectedParamCategory_ > 0) {
        const auto& cats = hyperparamRegistry_.categories();
        if (selectedParamCategory_ - 1 < static_cast<int>(cats.size())) {
            catFilter = cats[selectedParamCategory_ - 1];
        }
    }
    auto params = hyperparamRegistry_.filtered(catFilter);
    const float totalContentH = computeParamBrowserContentHeight(params, catFilter, rowH);
    const ParamScrollbarMetrics scrollbar = computeParamScrollbarMetrics(
        x, y, w, listH, totalContentH, std::clamp(paramScrollOffset_, 0.0f, std::max(0.0f, totalContentH - listH)));

    if (draggingParamScrollbar_ ||
        (scrollbar.visible && pointInRect(input.mousePos, scrollbar.trackX, scrollbar.trackY, scrollbar.trackW, scrollbar.trackH))) {
        return;
    }

    // Find which row was clicked — must match drawParamBrowser geometry exactly
    float drawY = y + 2.0f - paramScrollOffset_;
    std::string lastCategory;
    for (size_t i = 0; i < params.size(); ++i) {
        const auto* entry = params[i];

        // Account for category sub-header rows (when showing "All")
        if (catFilter.empty() && entry->category != lastCategory) {
            lastCategory = entry->category;
            drawY += rowH; // skip category header row
        }

        if (drawY + rowH < y || drawY > y + listH) {
            drawY += rowH;
            continue;  // clipped
        }

        if (m.y >= drawY && m.y < drawY + rowH) {
            bool clickedValue = (m.x >= x + nameColW);

            if (!clickedValue) {
                // Clicked on name column — just commit any open edit
                if (editingParamIndex_ >= 0) commitParamEdit();
                return;
            }

            // Clicked on value column
            if (entry->type == GRIM::Config::HyperparamType::Bool) {
                // Toggle booleans immediately
                if (editingParamIndex_ >= 0) commitParamEdit();
                setPresetHyperparamValue(*entry, !entry->value.get<bool>());
            } else {
                // Open inline editor for non-bool types
                if (editingParamIndex_ >= 0) commitParamEdit();
                editingParamIndex_ = static_cast<int>(i);
                editParamBuffer_ = entry->valueAsString();
                if (paramEditInput_) {
                    paramEditInput_->setText(editParamBuffer_);
                }
            }
            return;
        }
        drawY += rowH;
    }

    // Click was in list area but not on any row — commit
    if (editingParamIndex_ >= 0) commitParamEdit();
}

// ============================================================
// Commit / Cancel inline param edit
// ============================================================

void UITrainingPanel::commitParamEdit() {
    if (editingParamIndex_ < 0) return;

    // Get filtered list to find the entry
    std::string catFilter;
    if (paramCategoryFilter_ && selectedParamCategory_ > 0) {
        const auto& cats = hyperparamRegistry_.categories();
        if (selectedParamCategory_ - 1 < static_cast<int>(cats.size())) {
            catFilter = cats[selectedParamCategory_ - 1];
        }
    }
    auto params = hyperparamRegistry_.filtered(catFilter);

    if (editingParamIndex_ >= 0 && editingParamIndex_ < static_cast<int>(params.size())) {
        const auto* entry = params[editingParamIndex_];
        const std::string& val = editParamBuffer_;

        try {
            setPresetHyperparamValue(*entry, entry->parseEditedValue(val));
        } catch (const std::exception& e) {
            LOG_ERROR("UITrainingPanel", std::string("Invalid hyperparameter edit for ") + entry->key + ": " + e.what());
        }
    }

    editingParamIndex_ = -1;
    editParamBuffer_.clear();
}

void UITrainingPanel::cancelParamEdit() {
    editingParamIndex_ = -1;
    editParamBuffer_.clear();
}

// ============================================================
// Persist single hyperparam value through GRIM runtime config
// ============================================================

bool UITrainingPanel::setPresetHyperparamValue(const GRIM::Config::HyperparamEntry& entry,
                                                const nlohmann::json& value) {
    try {
        configPresetDocument_["training"]["config"][entry.key] = value;
        hyperparamRegistry_.populateModelConfigSchema(
            configPresetDocument_.at("training").at("config"));
        configPresetDirty_ = true;
        configCompileStatus_ = "Preset modified in memory; create .grimcfg to save it";
        configCompileSuccess_ = true;
        return true;

    } catch (const std::exception& e) {
        LOG_ERROR("UITrainingPanel", std::string("Failed to update preset field ") + entry.key + ": " + e.what());
        return false;
    }
}

// ============================================================
// Model configuration preset creator
// ============================================================

namespace {

namespace fs = std::filesystem;

bool isValidModelId(const std::string& modelId) {
    if (modelId.empty() || modelId == "." || modelId == "..") return false;
    return std::all_of(modelId.begin(), modelId.end(), [](unsigned char ch) {
        return std::isalnum(ch) || ch == '-' || ch == '_';
    });
}

std::string quoteCommandPath(const fs::path& path) {
    const std::string value = path.string();
    if (value.find('"') != std::string::npos) {
        throw std::runtime_error("paths containing a double quote are not supported");
    }
    return "\"" + value + "\"";
}

int runConfigCompilerProcess(const fs::path& compiler,
                             const fs::path& input,
                             const fs::path& vocab,
                             const fs::path& output,
                             const fs::path& logPath) {
#ifdef _WIN32
    SECURITY_ATTRIBUTES security{};
    security.nLength = sizeof(security);
    security.bInheritHandle = TRUE;

    HANDLE logHandle = CreateFileW(
        logPath.c_str(),
        GENERIC_WRITE,
        FILE_SHARE_READ,
        &security,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (logHandle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("cannot create compiler output log");
    }

    auto quoteWide = [](const fs::path& path) {
        const std::wstring value = path.wstring();
        if (value.find(L'\"') != std::wstring::npos) {
            throw std::runtime_error("paths containing a double quote are not supported");
        }
        return L"\"" + value + L"\"";
    };

    std::wstring commandLine = quoteWide(compiler) +
        L" --input " + quoteWide(input) +
        L" --vocab " + quoteWide(vocab) +
        L" --output " + quoteWide(output);

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = logHandle;
    startup.hStdError = logHandle;
    PROCESS_INFORMATION process{};

    const BOOL created = CreateProcessW(
        compiler.c_str(),
        commandLine.data(),
        nullptr,
        nullptr,
        TRUE,
        CREATE_NO_WINDOW,
        nullptr,
        nullptr,
        &startup,
        &process);
    if (!created) {
        const DWORD error = GetLastError();
        CloseHandle(logHandle);
        throw std::runtime_error("cannot launch config compiler (Win32 error " +
                                 std::to_string(error) + ")");
    }

    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exitCode = 1;
    GetExitCodeProcess(process.hProcess, &exitCode);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    CloseHandle(logHandle);
    return static_cast<int>(exitCode);
#else
    const std::string command =
        quoteCommandPath(compiler) +
        " --input " + quoteCommandPath(input) +
        " --vocab " + quoteCommandPath(vocab) +
        " --output " + quoteCommandPath(output) +
        " > " + quoteCommandPath(logPath) + " 2>&1";
    return std::system(command.c_str());
#endif
}

std::string lastNonEmptyLine(const std::string& text) {
    std::istringstream stream(text);
    std::string line;
    std::string last;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) last = line;
    }
    return last;
}

std::string relativeDisplayPath(const fs::path& path) {
    std::error_code ec;
    const fs::path relative = fs::relative(path, fs::path(getGrimRootDir()), ec);
    return ec ? path.string() : relative.generic_string();
}

}  // namespace

void UITrainingPanel::reloadConfigPresetTemplate() {
    cancelParamEdit();
    loadHyperparamSnapshot();
    if (paramCategoryFilter_) {
        std::vector<std::string> items = {"All"};
        for (const auto& category : hyperparamRegistry_.categories()) items.push_back(category);
        paramCategoryFilter_->setItems(items);
        paramCategoryFilter_->setSelectedIndex(0);
        selectedParamCategory_ = 0;
        paramScrollOffset_ = 0.0f;
    }
}

std::string UITrainingPanel::findConfigCompilerExecutable() const {
#ifdef _WIN32
    constexpr const char* executableName = "compile_model_config.exe";
#else
    constexpr const char* executableName = "compile_model_config";
#endif

    const fs::path root(getGrimRootDir());
    const std::vector<fs::path> candidates = {
        root / "build" / "grim-config-standalone" / "Release" / executableName,
        root / "build" / "grim-config-standalone" / executableName,
        root / "build" / "grim-config-compiler" / "ConfigCompiler" / "Release" / executableName,
        root / "build" / "grim-config-compiler" / "Release" / executableName,
        root / "resources" / "models" / "GRIM-text" / "training" / "TrainingLoop" /
            "build" / "ConfigCompiler" / "Release" / executableName
    };

    for (const auto& candidate : candidates) {
        if (fs::is_regular_file(candidate)) return relativeDisplayPath(candidate);
    }
    return relativeDisplayPath(candidates.front());
}

void UITrainingPanel::refreshConfigModelDropdown() {
    std::set<std::string> modelIds;
    for (const auto* model : GRIM::MMO::ModelRegistry::instance().getAllModels()) {
        if (model && model->kind == GRIM::MMO::ModelKind::Text && !model->id.empty()) {
            modelIds.insert(model->id);
        }
    }

    try {
        const fs::path store = resolvePathFromGrimRoot(
            configPresetDocument_.at("paths").at("grim_text").at("model_store").get<std::string>());
        if (fs::is_directory(store)) {
            for (const auto& entry : fs::directory_iterator(store)) {
                if (entry.is_directory()) modelIds.insert(entry.path().filename().string());
            }
        }
    } catch (...) {
    }

    configModelIds_.assign(modelIds.begin(), modelIds.end());
    std::vector<std::string> items = {"(new model)"};
    items.insert(items.end(), configModelIds_.begin(), configModelIds_.end());
    if (configModelDropdown_) configModelDropdown_->setItems(items);

    int selectedIndex = 0;
    if (configModelIdBuffer_.empty() && modelIds.count("grim-text-router")) {
        configModelIdBuffer_ = "grim-text-router";
        if (configModelIdInput_) configModelIdInput_->setText(configModelIdBuffer_);
    }
    const auto selected = std::find(configModelIds_.begin(), configModelIds_.end(), configModelIdBuffer_);
    if (selected != configModelIds_.end()) {
        selectedIndex = static_cast<int>(std::distance(configModelIds_.begin(), selected)) + 1;
    }
    if (configModelDropdown_) configModelDropdown_->setSelectedIndex(selectedIndex);
}

std::string UITrainingPanel::configOutputPath() const {
    if (!isValidModelId(configModelIdBuffer_)) return {};
    try {
        const fs::path store = resolvePathFromGrimRoot(
            configPresetDocument_.at("paths").at("grim_text").at("model_store").get<std::string>());
        return relativeDisplayPath(store / configModelIdBuffer_ / "model.grimcfg");
    } catch (...) {
        return {};
    }
}

void UITrainingPanel::beginConfigCompile() {
    applyConfigCompileResult();
    if (configCompileFuture_.valid()) {
        configCompileStatus_ = "A model config compilation is already running";
        configCompileSuccess_ = false;
        return;
    }

    commitParamEdit();
    if (!isValidModelId(configModelIdBuffer_)) {
        configCompileStatus_ = "Model ID may contain only letters, numbers, '-' and '_'";
        configCompileSuccess_ = false;
        return;
    }
    if (!hyperparamsLoaded_) {
        configCompileStatus_ = "No FlatBuffer model schema is loaded";
        configCompileSuccess_ = false;
        return;
    }

    std::string modelStorePath;
    try {
        modelStorePath = configPresetDocument_.at("paths").at("grim_text")
            .at("model_store").get<std::string>();
    } catch (const std::exception& e) {
        configCompileStatus_ = std::string("Missing model-store path: ") + e.what();
        configCompileSuccess_ = false;
        return;
    }

    configCompileStatus_ = "Compiling immutable model configuration...";
    configCompileSuccess_ = false;

    const nlohmann::json source = configPresetDocument_;
    const std::string compilerPath = configCompilerPathBuffer_;
    const std::string vocabPath = configVocabPathBuffer_;
    const std::string modelId = configModelIdBuffer_;
    configCompileFuture_ = std::async(
        std::launch::async,
        &UITrainingPanel::compileConfigPreset,
        source,
        compilerPath,
        vocabPath,
        modelStorePath,
        modelId);
}

void UITrainingPanel::applyConfigCompileResult() {
    if (!configCompileFuture_.valid()) return;
    if (configCompileFuture_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;

    try {
        ConfigCompileResult result = configCompileFuture_.get();
        configCompileSuccess_ = result.success;
        configCompileStatus_ = std::move(result.message);
        if (configCompileSuccess_) {
            configPresetDirty_ = false;
            refreshConfigModelDropdown();
            LOG_DEBUG("UITrainingPanel", configCompileStatus_);
        } else {
            LOG_ERROR("UITrainingPanel", configCompileStatus_);
        }
    } catch (const std::exception& e) {
        configCompileSuccess_ = false;
        configCompileStatus_ = std::string("Config compiler task failed: ") + e.what();
        LOG_ERROR("UITrainingPanel", configCompileStatus_);
    }
}

UITrainingPanel::ConfigCompileResult UITrainingPanel::compileConfigPreset(
    const nlohmann::json& source,
    const std::string& compilerPath,
    const std::string& vocabPath,
    const std::string& modelStorePath,
    const std::string& modelId)
{
    ConfigCompileResult result;
    fs::path inputPath;
    fs::path logPath;

    try {
        if (!isValidModelId(modelId)) throw std::runtime_error("invalid model ID");
        if (!source.contains("training") || !source.at("training").is_object() ||
            !source.at("training").contains("config") ||
            !source.at("training").at("config").is_object()) {
            throw std::runtime_error("preset source does not contain training.config");
        }

        const fs::path compiler = resolvePathFromGrimRoot(compilerPath);
        const fs::path vocab = resolvePathFromGrimRoot(vocabPath);
        const fs::path store = resolvePathFromGrimRoot(modelStorePath);
        const fs::path modelDirectory = store / modelId;
        const fs::path output = modelDirectory / "model.grimcfg";
        inputPath = modelDirectory / ".model.grimcfg.input.tmp.json";
        logPath = modelDirectory / ".model.grimcfg.compile.tmp.log";

        if (!fs::is_regular_file(compiler)) {
            throw std::runtime_error("config compiler not found: " + compiler.string());
        }
        if (!fs::is_regular_file(vocab)) {
            throw std::runtime_error("vocabulary not found: " + vocab.string());
        }

        fs::create_directories(modelDirectory);
        {
            std::ofstream input(inputPath, std::ios::trunc);
            if (!input) throw std::runtime_error("cannot create temporary compiler input");
            input << std::setw(2) << source << '\n';
            if (!input) throw std::runtime_error("failed writing temporary compiler input");
        }

        std::error_code ec;
        fs::remove(logPath, ec);
        const int exitCode = runConfigCompilerProcess(
            compiler, inputPath, vocab, output, logPath);
        std::string compilerOutput;
        {
            std::ifstream log(logPath);
            if (log) {
                compilerOutput.assign(std::istreambuf_iterator<char>(log),
                                      std::istreambuf_iterator<char>());
            }
        }

        fs::remove(inputPath, ec);
        fs::remove(logPath, ec);

        if (exitCode != 0 || !fs::is_regular_file(output)) {
            std::string detail = lastNonEmptyLine(compilerOutput);
            if (detail.empty()) detail = "compiler exited with code " + std::to_string(exitCode);
            throw std::runtime_error(detail);
        }

        result.success = true;
        const std::string outputPath = relativeDisplayPath(output);
        result.message = "Created " + outputPath + " (" +
            std::to_string(fs::file_size(output)) + " bytes)";
        return result;
    } catch (const std::exception& e) {
        std::error_code ec;
        if (!inputPath.empty()) fs::remove(inputPath, ec);
        if (!logPath.empty()) fs::remove(logPath, ec);
        result.message = std::string("Config compilation failed: ") + e.what();
        return result;
    }
}

void UITrainingPanel::updateModelConfigTab(const InputState& input, float dt) {
    if (configModelDropdown_) configModelDropdown_->update(input, dt);
    if (configModelIdInput_) configModelIdInput_->update(input, dt);
    if (configVocabPathInput_) configVocabPathInput_->update(input, dt);
    if (configCompilerPathInput_) configCompilerPathInput_->update(input, dt);
    if (configCompileButton_) configCompileButton_->update(input, dt);
    if (configReloadButton_) configReloadButton_->update(input, dt);
    if (paramCategoryFilter_) paramCategoryFilter_->update(input, dt);

    auto content = getContentRect();
    content.origin.y += (kContentTopY - kTabBarY);
    content.size.y -= (kContentTopY - kTabBarY);
    constexpr float rowH = 22.0f;
    const Vec2 mouse = input.mousePos;

    const float browserX = content.origin.x;
    float listY = content.origin.y + Sizes::HeaderHeight + Spacing::Small;
    listY += 26.0f + Spacing::Small;
    float listH = content.size.y - (listY - content.origin.y) - Spacing::Small;
    if (listH < 60.0f) listH = 60.0f;

    std::string category;
    if (hyperparamsLoaded_ && paramCategoryFilter_ && selectedParamCategory_ > 0) {
        const auto& categories = hyperparamRegistry_.categories();
        if (selectedParamCategory_ - 1 < static_cast<int>(categories.size())) {
            category = categories[static_cast<size_t>(selectedParamCategory_ - 1)];
        }
    }
    const auto params = hyperparamRegistry_.filtered(category);
    const float totalContentH = computeParamBrowserContentHeight(params, category, rowH);
    const float maxScroll = std::max(0.0f, totalContentH - listH);
    paramScrollOffset_ = std::clamp(paramScrollOffset_, 0.0f, maxScroll);

    const ParamScrollbarMetrics scrollbar = computeParamScrollbarMetrics(
        browserX, listY, kParamBrowserW, listH, totalContentH, paramScrollOffset_);
    const bool overPanel = pointInRect(
        mouse, browserX, content.origin.y, kParamBrowserW, content.size.y);
    const bool overList = pointInRect(mouse, browserX, listY, kParamBrowserW, listH);
    const bool overTrack = scrollbar.visible && pointInRect(
        mouse, scrollbar.trackX, scrollbar.trackY, scrollbar.trackW, scrollbar.trackH);
    const bool overThumb = scrollbar.visible && pointInRect(
        mouse, scrollbar.trackX, scrollbar.thumbY, scrollbar.trackW, scrollbar.thumbH);

    hoveredParamScrollbar_ = overThumb;
    if (scrollbar.visible && input.mousePressed[0] && overTrack) {
        draggingParamScrollbar_ = true;
        if (!overThumb && scrollbar.maxThumbTravel > 0.0f) {
            const float targetThumbY = std::clamp(
                mouse.y - scrollbar.thumbH * 0.5f,
                scrollbar.trackY,
                scrollbar.trackY + scrollbar.maxThumbTravel);
            paramScrollOffset_ = ((targetThumbY - scrollbar.trackY) /
                                  scrollbar.maxThumbTravel) * scrollbar.maxScroll;
        }
        paramScrollbarDragStartY_ = mouse.y;
        paramScrollbarDragStartOffset_ = paramScrollOffset_;
    }

    if (draggingParamScrollbar_) {
        if (input.mouseDown[0]) {
            if (scrollbar.maxThumbTravel > 0.0f && scrollbar.maxScroll > 0.0f) {
                const float deltaY = mouse.y - paramScrollbarDragStartY_;
                const float delta = (deltaY / scrollbar.maxThumbTravel) * scrollbar.maxScroll;
                paramScrollOffset_ = std::clamp(
                    paramScrollbarDragStartOffset_ + delta, 0.0f, scrollbar.maxScroll);
            }
        } else {
            draggingParamScrollbar_ = false;
        }
    }

    if (overList && !draggingParamScrollbar_ && input.mouseWheelDelta != 0.0f) {
        const float wheelSteps = normalizeMouseWheelDelta(input.mouseWheelDelta);
        paramScrollOffset_ = std::clamp(
            paramScrollOffset_ - wheelSteps * kParamBrowserWheelPixelsPerStep,
            0.0f,
            maxScroll);
    }
    if (!overPanel && !draggingParamScrollbar_) hoveredParamScrollbar_ = false;

    hoveredParamRow_ = -1;
    if (hyperparamsLoaded_ && overList && !draggingParamScrollbar_ &&
        (!scrollbar.visible || mouse.x < scrollbar.trackX)) {
        float drawY = listY + 2.0f - paramScrollOffset_;
        std::string lastCategory;
        for (size_t i = 0; i < params.size(); ++i) {
            const auto* entry = params[i];
            if (category.empty() && entry->category != lastCategory) {
                lastCategory = entry->category;
                drawY += rowH;
            }
            if (drawY + rowH >= listY && drawY <= listY + listH &&
                mouse.y >= drawY && mouse.y < drawY + rowH) {
                hoveredParamRow_ = static_cast<int>(i);
                break;
            }
            drawY += rowH;
        }
    }

    if (editingParamIndex_ >= 0 && paramEditInput_) paramEditInput_->update(input, dt);
    processParamBrowserClicks(input);
}

void UITrainingPanel::drawModelConfigTab(OverlayRenderer& renderer, const PanelRect& content) {
    drawParamBrowser(renderer,
                     {content.origin.x, content.origin.y},
                     {kParamBrowserW, content.size.y});

    const float gutter = 12.0f;
    const float rightX = content.origin.x + kParamBrowserW + gutter;
    const float rightW = content.size.x - kParamBrowserW - gutter;
    const float innerX = rightX + Spacing::PaddingX;
    const float innerW = rightW - 2.0f * Spacing::PaddingX;
    float y = content.origin.y;

    UIDrawHelpers::drawSectionHeader(
        renderer, {rightX, y}, rightW, "Create Model Configuration", Colors::SectionAI);
    y += Sizes::HeaderHeight + Spacing::Small;

    renderer.drawText(
        {innerX, y},
        "Edit the snapshot on the left, then compile it into the selected model directory.",
        Colors::TextSecondary);
    y += 25.0f;

    if (configModelDropdown_) {
        configModelDropdown_->setPosition(innerX, y);
        configModelDropdown_->setSize(innerW, 34.0f);
        configModelDropdown_->drawOverlay(renderer, position);
        y += 42.0f;
    }

    renderer.drawText({innerX, y}, "Model ID / directory name", Colors::TextSecondary);
    y += 18.0f;
    configModelIdInput_->setPosition(innerX, y);
    configModelIdInput_->setSize(innerW, 28.0f);
    configModelIdInput_->drawOverlay(renderer, position);
    y += 38.0f;

    renderer.drawText({innerX, y}, "Vocabulary", Colors::TextSecondary);
    y += 18.0f;
    configVocabPathInput_->setPosition(innerX, y);
    configVocabPathInput_->setSize(innerW, 28.0f);
    configVocabPathInput_->drawOverlay(renderer, position);
    y += 38.0f;

    renderer.drawText({innerX, y}, "Config compiler", Colors::TextSecondary);
    y += 18.0f;
    configCompilerPathInput_->setPosition(innerX, y);
    configCompilerPathInput_->setSize(innerW, 28.0f);
    configCompilerPathInput_->drawOverlay(renderer, position);
    y += 44.0f;

    UIDrawHelpers::drawSectionHeader(
        renderer, {rightX, y}, rightW, "Artifact", Colors::SectionNeutral);
    y += Sizes::HeaderHeight + Spacing::Small;

    const std::string output = configOutputPath();
    renderer.drawText({innerX, y}, "Output", Colors::TextSecondary);
    y += 18.0f;
    renderer.drawRoundedRect({innerX, y}, {innerW, 32.0f}, Colors::ContentAreaBg, Sizes::WidgetRadius);
    renderer.drawRoundedBorder({innerX, y}, {innerW, 32.0f}, Colors::BorderSubtle, Sizes::WidgetRadius);
    renderer.drawText(
        {innerX + 8.0f, y + 8.0f},
        output.empty() ? "Enter a valid model ID" : output,
        output.empty() ? Colors::TextMuted : Colors::TextValue);
    y += 44.0f;

    configCompileButton_->setPosition(innerX, y);
    configCompileButton_->drawOverlay(renderer, position);
    configReloadButton_->setPosition(innerX + 160.0f, y);
    configReloadButton_->drawOverlay(renderer, position);
    y += 42.0f;

    const bool compiling = configCompileFuture_.valid();
    if (compiling) {
        renderer.drawText({innerX, y}, "Compiling...", Colors::Warning);
    } else if (!configCompileStatus_.empty()) {
        renderer.drawText(
            {innerX, y},
            configCompileStatus_,
            configCompileSuccess_ ? Colors::Success : Colors::Danger);
    }
    y += 28.0f;

    const std::string summary = std::to_string(hyperparamRegistry_.entries().size()) +
        " authored fields from grim_compiled_hyperparameters.fbs" +
        (configPresetDirty_ ? " (modified)" : "");
    renderer.drawText({innerX, y}, summary, Colors::TextSecondary);
    y += 20.0f;
    renderer.drawText(
        {innerX, y},
        "The compiler derives vocabulary geometry, validates invariants, and writes model.grimcfg atomically.",
        Colors::TextMuted);
}
