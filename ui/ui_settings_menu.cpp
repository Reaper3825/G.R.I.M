#include "ui_settings_menu.hpp"
#include "overlay_renderer.hpp"
#include "ui_theme.hpp"
#include "ui_draw_helpers.hpp"
#include "logger.hpp"
#include "input_parser.hpp"
#include "ui_root.hpp"
#include "core/platform_window.hpp"
#include "../settings/runtime_ai_config.hpp"
#include "../settings/settings_apply.hpp"
#include <functional>

static constexpr float kTabBarY     = 35.0f;
static constexpr float kContentTopY = 68.0f;

UISettingsMenu::UISettingsMenu()
    : UIPanel("Settings", true), hasChanges(false), isRefreshing(false), needsWidgetRefresh(false)
{
    position = { 200, 200 };
    size = { 620, 700 };
    setVisible(false);
    setBackground(UITheme::Colors::PanelBg);
    
    scrollBox = std::make_shared<UIScrollBox>();
    scrollBox->setChildSpacing(5.0f);
    
    // Create tab buttons
    tabGeneralBtn_ = std::make_shared<UIButton>("General", [this]() { setTab(SettingsTab::General); });
    tabGeneralBtn_->setSize(70.0f, 26.0f);
    tabVoiceBtn_ = std::make_shared<UIButton>("Voice", [this]() { setTab(SettingsTab::Voice); });
    tabVoiceBtn_->setSize(60.0f, 26.0f);
    tabAudioBtn_ = std::make_shared<UIButton>("Audio", [this]() { setTab(SettingsTab::Audio); });
    tabAudioBtn_->setSize(60.0f, 26.0f);
    tabVisionBtn_ = std::make_shared<UIButton>("Vision", [this]() { setTab(SettingsTab::Vision); });
    tabVisionBtn_->setSize(62.0f, 26.0f);
    tabUIGraphicsBtn_ = std::make_shared<UIButton>("UI", [this]() { setTab(SettingsTab::UIGraphics); });
    tabUIGraphicsBtn_->setSize(40.0f, 26.0f);
    tabPreferencesBtn_ = std::make_shared<UIButton>("Prefs", [this]() { setTab(SettingsTab::Preferences); });
    tabPreferencesBtn_->setSize(55.0f, 26.0f);
    tabMemoryBtn_ = std::make_shared<UIButton>("Memory", [this]() { setTab(SettingsTab::Memory); });
    tabMemoryBtn_->setSize(68.0f, 26.0f);
    
    loadConfig();
}

// =========================================================
// Tab management
// =========================================================

void UISettingsMenu::setTab(SettingsTab tab) {
    if (tab == activeTab_ && !needsWidgetRefresh) return;
    activeTab_ = tab;
    needsWidgetRefresh = true;
    LOG_DEBUG("UISettingsMenu", "Switched to tab " + std::to_string(static_cast<int>(tab)));
}

std::vector<std::string> UISettingsMenu::getSpeakerEmbeddings() {
    return Settings::scanSpeakerEmbeddings();
}

std::vector<std::string> UISettingsMenu::getFontList() {
    return Settings::scanFonts(m_fontPathMap);
}

void UISettingsMenu::loadConfig() {
    config = Settings::loadRuntimeAiConfig();
    pendingConfig = config;
    hasChanges = false;
    needsWidgetRefresh = true;
}

void UISettingsMenu::saveConfig() {
    config = Settings::saveRuntimeAiConfig(pendingConfig);
}

void UISettingsMenu::applyChanges() {
    if (!hasChanges) return;

    saveConfig();
    hasChanges = false;

    if (m_fontPathMap.empty()) getFontList();

    Settings::applyRuntimeSideEffects(
        config,
        m_fontPathMap,
        [](const std::string& path, int sz) {
            UIRoot::get().getRenderer().setFont(path, sz);
        },
        [](bool enabled, float opacity, int intensity) {
            PlatformWindow::setOverlayBlurStyle(
                UIRoot::get().getHWND(), enabled, opacity, intensity);
        });
}

// ? Action handlers - these can safely be called from callbacks
void UISettingsMenu::cycleBackend() {
    try {
        LOG_DEBUG("UISettingsMenu", "cycleBackend() called");
        std::string current = pendingConfig.value("backend", "auto");
        LOG_DEBUG("UISettingsMenu", "Current backend: " + current);
        
        // Valid backends: auto, grim_native, ollama, localai, openai
        std::string next = (current == "auto") ? "grim_native" :
                           (current == "grim_native") ? "ollama" :
                           (current == "ollama") ? "localai" :
                           (current == "localai") ? "openai" : "auto";
        LOG_DEBUG("UISettingsMenu", "Next backend: " + next);
        
        pendingConfig["backend"] = next;
        hasChanges = true;
        needsWidgetRefresh = true;
        
        LOG_DEBUG("UISettingsMenu", "cycleBackend() completed successfully");
    } catch (const std::exception& e) {
        LOG_ERROR("UISettingsMenu", std::string("cycleBackend() exception: ") + e.what());
    } catch (...) {
        LOG_ERROR("UISettingsMenu", "cycleBackend() unknown exception");
    }
}

void UISettingsMenu::cycleVoice() {
    try {
        LOG_DEBUG("UISettingsMenu", "cycleVoice() called");
        
        // Access nested voice.engine instead of top-level voice
        std::string current = "coqui";
        if (pendingConfig.contains("voice") && pendingConfig["voice"].is_object() &&
            pendingConfig["voice"].contains("engine")) {
            current = pendingConfig["voice"]["engine"].get<std::string>();
        }
        
        std::string next = (current == "coqui") ? "sapi" : "coqui";
        LOG_DEBUG("UISettingsMenu", "Voice engine: " + current + " -> " + next);
        
        // Ensure voice object exists
        if (!pendingConfig.contains("voice") || !pendingConfig["voice"].is_object()) {
            pendingConfig["voice"] = nlohmann::json::object();
        }
        pendingConfig["voice"]["engine"] = next;
        
        hasChanges = true;
        needsWidgetRefresh = true;
        LOG_DEBUG("UISettingsMenu", "cycleVoice() completed successfully");
    } catch (const std::exception& e) {
        LOG_ERROR("UISettingsMenu", std::string("cycleVoice() exception: ") + e.what());
    } catch (...) {
        LOG_ERROR("UISettingsMenu", "cycleVoice() unknown exception");
    }
}

void UISettingsMenu::cycleSpeaker() {
    try {
        LOG_DEBUG("UISettingsMenu", "cycleSpeaker() called");
        
        // Ensure voice object exists
        if (!pendingConfig.contains("voice") || !pendingConfig["voice"].is_object()) {
            pendingConfig["voice"] = nlohmann::json::object();
        }
        
        // Get available speakers from config or scan embeddings
        std::vector<std::string> availableSpeakers;
        if (pendingConfig["voice"].contains("available_speakers") && 
            pendingConfig["voice"]["available_speakers"].is_array()) {
            for (const auto& speaker : pendingConfig["voice"]["available_speakers"]) {
                availableSpeakers.push_back(speaker.get<std::string>());
            }
        } else {
            // Default to scanning embeddings if not in config
            availableSpeakers = getSpeakerEmbeddings();
            pendingConfig["voice"]["available_speakers"] = availableSpeakers;
        }
        
        if (availableSpeakers.empty()) {
            LOG_ERROR("UISettingsMenu", "No speakers available");
            return;
        }
        
        // Get current speaker
        std::string current = "default";
        if (pendingConfig["voice"].contains("speaker")) {
            current = pendingConfig["voice"]["speaker"].get<std::string>();
        }
        
        // Find current index
        int currentIndex = 0;
        for (size_t i = 0; i < availableSpeakers.size(); ++i) {
            if (availableSpeakers[i] == current) {
                currentIndex = static_cast<int>(i);
                break;
            }
        }
        
        // Cycle to next
        int nextIndex = (currentIndex + 1) % availableSpeakers.size();
        std::string next = availableSpeakers[nextIndex];
        
        LOG_DEBUG("UISettingsMenu", "Speaker: " + current + " -> " + next);
        
        pendingConfig["voice"]["speaker"] = next;
        
        hasChanges = true;
        needsWidgetRefresh = true;
        LOG_DEBUG("UISettingsMenu", "cycleSpeaker() completed successfully");
    } catch (const std::exception& e) {
        LOG_ERROR("UISettingsMenu", std::string("cycleSpeaker() exception: ") + e.what());
    } catch (...) {
        LOG_ERROR("UISettingsMenu", "cycleSpeaker() unknown exception");
    }
}

void UISettingsMenu::cycleModel() {
    try {
        LOG_DEBUG("UISettingsMenu", "cycleModel() called");
        
        // Access nested whisper.whisper_model instead of top-level model
        std::string current = "ggml-base.en.bin";
        if (pendingConfig.contains("whisper") && pendingConfig["whisper"].is_object() &&
            pendingConfig["whisper"].contains("whisper_model")) {
            current = pendingConfig["whisper"]["whisper_model"].get<std::string>();
        }
        
        std::string next = (current == "ggml-base.en.bin") ? "ggml-medium.en.bin" :
                           (current == "ggml-medium.en.bin") ? "ggml-large-v3.bin" :
                           "ggml-base.en.bin";
        LOG_DEBUG("UISettingsMenu", "Whisper model: " + current + " -> " + next);
        
        // Ensure whisper object exists
        if (!pendingConfig.contains("whisper") || !pendingConfig["whisper"].is_object()) {
            pendingConfig["whisper"] = nlohmann::json::object();
        }
        pendingConfig["whisper"]["whisper_model"] = next;
        
        hasChanges = true;
        needsWidgetRefresh = true;
        LOG_DEBUG("UISettingsMenu", "cycleModel() completed successfully");
    } catch (const std::exception& e) {
        LOG_ERROR("UISettingsMenu", std::string("cycleModel() exception: ") + e.what());
    } catch (...) {
        LOG_ERROR("UISettingsMenu", "cycleModel() unknown exception");
    }
}

void UISettingsMenu::cyclePersonality() {
    try {
        LOG_DEBUG("UISettingsMenu", "cyclePersonality() called");
        
        // Get current custom prompt
        std::string current = "Professional";
        if (pendingConfig.contains("personality") && pendingConfig["personality"].is_object() &&
            pendingConfig["personality"].contains("custom_prompt")) {
            std::string prompt = pendingConfig["personality"]["custom_prompt"].get<std::string>();
            
            // Identify which preset it matches
            if (prompt.find("helpful AI assistant") != std::string::npos) current = "Professional";
            else if (prompt.find("sarcastic") != std::string::npos) current = "Sarcastic";
            else if (prompt.find("friendly and enthusiastic") != std::string::npos) current = "Friendly";
            else if (prompt.find("military") != std::string::npos) current = "Military";
            else if (prompt.find("brief and direct") != std::string::npos) current = "Short";
            else if (prompt.find("concise but warm") != std::string::npos) current = "Short & Sweet";
            else current = "Custom";
        }
        
        // Preset personalities
        std::string nextLabel;
        std::string nextPrompt;
        
        if (current == "Professional") {
            nextLabel = "Friendly";
            nextPrompt = "You are GRIM, a friendly and enthusiastic AI assistant. Use warm, conversational language and show excitement when helping users. Be supportive and encouraging.";
        } else if (current == "Friendly") {
            nextLabel = "Short";
            nextPrompt = "You are GRIM. Be brief and direct. Give concise answers without unnecessary elaboration. Keep responses under 2 sentences when possible.";
        } else if (current == "Short") {
            nextLabel = "Short & Sweet";
            nextPrompt = "You are GRIM. Be concise but warm and friendly. Keep responses brief, goal: 1-2 sentences while maintaining a positive, helpful tone. Add a touch of personality without being wordy.";
        } else if (current == "Short & Sweet") {
            nextLabel = "Sarcastic";
            nextPrompt = "You are GRIM, a sarcastic but helpful AI assistant. Use dry humor and witty remarks while still providing accurate assistance. Don't be mean, just playfully sarcastic.";
        } else if (current == "Sarcastic") {
            nextLabel = "Military";
            nextPrompt = "You are GRIM, a military-grade tactical AI. Use precise, direct language.";
        } else {
            nextLabel = "Professional";
            nextPrompt = "You are GRIM, a helpful AI assistant. Be concise and professional.";
        }
        
        LOG_DEBUG("UISettingsMenu", "Personality: " + current + " -> " + nextLabel);
        
        // Ensure personality object exists
        if (!pendingConfig.contains("personality") || !pendingConfig["personality"].is_object()) {
            pendingConfig["personality"] = nlohmann::json::object();
        }
        pendingConfig["personality"]["custom_prompt"] = nextPrompt;
        
        hasChanges = true;
        needsWidgetRefresh = true;
        LOG_DEBUG("UISettingsMenu", "cyclePersonality() completed successfully");
    } catch (const std::exception& e) {
        LOG_ERROR("UISettingsMenu", std::string("cyclePersonality() exception: ") + e.what());
    } catch (...) {
        LOG_ERROR("UISettingsMenu", "cyclePersonality() unknown exception");
    }
}

void UISettingsMenu::doSaveAndClose() {
    try {
        LOG_DEBUG("UISettingsMenu", "doSaveAndClose() called");
        applyChanges();
        setVisible(false);
        LOG_DEBUG("UISettingsMenu", "doSaveAndClose() completed successfully");
    } catch (const std::exception& e) {
        LOG_ERROR("UISettingsMenu", std::string("doSaveAndClose() exception: ") + e.what());
    } catch (...) {
        LOG_ERROR("UISettingsMenu", "doSaveAndClose() unknown exception");
    }
}

void UISettingsMenu::doCancel() {
    try {
        LOG_DEBUG("UISettingsMenu", "doCancel() called");
        pendingConfig = config;
        hasChanges = false;
        needsWidgetRefresh = true;
        setVisible(false);
        LOG_DEBUG("UISettingsMenu", "doCancel() completed successfully");
    } catch (const std::exception& e) {
        LOG_ERROR("UISettingsMenu", std::string("doCancel() exception: ") + e.what());
    } catch (...) {
        LOG_ERROR("UISettingsMenu", "doCancel() unknown exception");
    }
}

void UISettingsMenu::createWidgets() {
    if (isRefreshing) {
        LOG_DEBUG("UISettingsMenu", "Already refreshing - skipping to prevent recursion");
        return;
    }
    
    isRefreshing = true;
    LOG_DEBUG("UISettingsMenu", "createWidgets() START - tab " + std::to_string(static_cast<int>(activeTab_)));
    
    try {
        scrollBox->clearChildren();
        
        switch (activeTab_) {
            case SettingsTab::General:     createGeneralWidgets();     break;
            case SettingsTab::Voice:       createVoiceWidgets();       break;
            case SettingsTab::Audio:       createAudioWidgets();       break;
            case SettingsTab::Vision:      createVisionWidgets();      break;
            case SettingsTab::UIGraphics:  createUIGraphicsWidgets();  break;
            case SettingsTab::Preferences: createPreferencesWidgets(); break;
            case SettingsTab::Memory:      createMemoryWidgets();      break;
        }
        
        scrollBox->setChildSpacing(5.0f);
        scrollBox->autoLayoutChildren(10.0f);
        
        saveButton = std::make_shared<UIButton>("Save & Close", [this]() { doSaveAndClose(); });
        cancelButton = std::make_shared<UIButton>("Cancel", [this]() { doCancel(); });
        
        LOG_DEBUG("UISettingsMenu", "Widgets created successfully for tab " + std::to_string(static_cast<int>(activeTab_)));
    }
    catch (const std::exception& e) {
        LOG_ERROR("UISettingsMenu", std::string("createWidgets() exception: ") + e.what());
        isRefreshing = false;
        throw;
    }
    catch (...) {
        LOG_ERROR("UISettingsMenu", "createWidgets() unknown exception");
        isRefreshing = false;
        throw;
    }
    
    isRefreshing = false;
}

// =========================================================
// General Tab — system-level settings (backend, URLs, tokens)
// =========================================================
void UISettingsMenu::createGeneralWidgets() {
    float widgetWidth = scrollBox->getSize().x - 30;
    float widgetHeight = 45.0f;
    
    // --- Backend ---
    std::string backend = pendingConfig.value("backend", "auto");
    auto backendButton = std::make_shared<UIButton>(
        "Backend: " + backend,
        [this]() { cycleBackend(); }
    );
    backendButton->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(backendButton);
    
    // --- Default Model ---
    std::string defaultModel = pendingConfig.value("default_model", "llama3.1:8b");
    std::vector<std::string> modelOptions = {"llama3.1:8b", "llama3.2:3b", "mistral:7b", "grim-text"};
    int modelIdx = 0;
    for (size_t i = 0; i < modelOptions.size(); ++i) {
        if (modelOptions[i] == defaultModel) { modelIdx = static_cast<int>(i); break; }
    }
    auto modelDropdown = std::make_shared<UIDropdown>(
        "Default Model:", modelOptions, modelIdx,
        [this](int, const std::string& val) {
            pendingConfig["default_model"] = val;
            hasChanges = true;
        }
    );
    modelDropdown->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(modelDropdown);
    
    // --- Max Tokens ---
    int maxTokens = pendingConfig.value("max_tokens", 256);
    auto maxTokensSlider = std::make_shared<UISlider>(
        "Max Tokens:", 32.0f, 2048.0f, static_cast<float>(maxTokens),
        [this](float val) {
            pendingConfig["max_tokens"] = static_cast<int>(val);
            hasChanges = true;
        }, 32.0f
    );
    maxTokensSlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(maxTokensSlider);
    
    // --- Conversation History Size ---
    int historySize = pendingConfig.value("conversation_history_size", 10);
    auto historySlider = std::make_shared<UISlider>(
        "History Size:", 1.0f, 50.0f, static_cast<float>(historySize),
        [this](float val) {
            pendingConfig["conversation_history_size"] = static_cast<int>(val);
            hasChanges = true;
        }, 1.0f
    );
    historySlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(historySlider);
    
    // --- MMO Enabled ---
    bool mmoEnabled = false;
    if (pendingConfig.contains("mmo") && pendingConfig["mmo"].is_object()) {
        mmoEnabled = pendingConfig["mmo"].value("enabled", false);
    }
    auto mmoToggle = std::make_shared<UIToggle>(
        "MMO (Multi-Model):", mmoEnabled,
        [this](bool val) {
            if (!pendingConfig.contains("mmo") || !pendingConfig["mmo"].is_object())
                pendingConfig["mmo"] = nlohmann::json::object();
            pendingConfig["mmo"]["enabled"] = val;
            hasChanges = true;
        }
    );
    mmoToggle->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(mmoToggle);
}

// =========================================================
// Voice Tab — TTS engine, speaker, speed, voice rules
// =========================================================
void UISettingsMenu::createVoiceWidgets() {
    float widgetWidth = scrollBox->getSize().x - 30;
    float widgetHeight = 45.0f;
    
    // --- Voice Engine ---
    std::string voiceEngine = "coqui";
    if (pendingConfig.contains("voice") && pendingConfig["voice"].is_object() &&
        pendingConfig["voice"].contains("engine")) {
        voiceEngine = pendingConfig["voice"]["engine"].get<std::string>();
    }
    auto voiceButton = std::make_shared<UIButton>(
        "Voice Engine: " + voiceEngine,
        [this]() { cycleVoice(); }
    );
    voiceButton->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(voiceButton);
    
    // --- Speaker ---
    std::string currentSpeaker = "default";
    if (pendingConfig.contains("voice") && pendingConfig["voice"].is_object() &&
        pendingConfig["voice"].contains("speaker")) {
        currentSpeaker = pendingConfig["voice"]["speaker"].get<std::string>();
    }
    auto speakerButton = std::make_shared<UIButton>(
        "Speaker: " + currentSpeaker,
        [this]() { cycleSpeaker(); }
    );
    speakerButton->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(speakerButton);
    
    // --- Voice Mode ---
    std::string voiceMode = "local";
    if (pendingConfig.contains("voice") && pendingConfig["voice"].is_object()) {
        voiceMode = pendingConfig["voice"].value("mode", "local");
    }
    std::vector<std::string> modeOptions = {"local", "remote"};
    int modeIdx = (voiceMode == "remote") ? 1 : 0;
    auto modeDropdown = std::make_shared<UIDropdown>(
        "Voice Mode:", modeOptions, modeIdx,
        [this](int, const std::string& val) {
            if (!pendingConfig.contains("voice") || !pendingConfig["voice"].is_object())
                pendingConfig["voice"] = nlohmann::json::object();
            pendingConfig["voice"]["mode"] = val;
            hasChanges = true;
        }
    );
    modeDropdown->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(modeDropdown);
    
    // --- Speed ---
    float speed = 1.0f;
    if (pendingConfig.contains("voice") && pendingConfig["voice"].is_object()) {
        speed = pendingConfig["voice"].value("speed", 1.0f);
    }
    auto speedSlider = std::make_shared<UISlider>(
        "Voice Speed:", 0.5f, 2.0f, speed,
        [this](float val) {
            if (!pendingConfig.contains("voice") || !pendingConfig["voice"].is_object())
                pendingConfig["voice"] = nlohmann::json::object();
            pendingConfig["voice"]["speed"] = val;
            hasChanges = true;
        }, 0.05f
    );
    speedSlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(speedSlider);
    
    // --- Filter Low Confidence ---
    bool filterConf = true;
    if (pendingConfig.contains("voice") && pendingConfig["voice"].is_object()) {
        filterConf = pendingConfig["voice"].value("filter_low_confidence", true);
    }
    auto filterToggle = std::make_shared<UIToggle>(
        "Filter Low Confidence:", filterConf,
        [this](bool val) {
            if (!pendingConfig.contains("voice") || !pendingConfig["voice"].is_object())
                pendingConfig["voice"] = nlohmann::json::object();
            pendingConfig["voice"]["filter_low_confidence"] = val;
            hasChanges = true;
        }
    );
    filterToggle->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(filterToggle);
    
    // --- Confidence Threshold ---
    float confThresh = 0.6f;
    if (pendingConfig.contains("voice") && pendingConfig["voice"].is_object()) {
        confThresh = pendingConfig["voice"].value("confidence_threshold", 0.6f);
    }
    auto confSlider = std::make_shared<UISlider>(
        "Confidence Threshold:", 0.0f, 1.0f, confThresh,
        [this](float val) {
            if (!pendingConfig.contains("voice") || !pendingConfig["voice"].is_object())
                pendingConfig["voice"] = nlohmann::json::object();
            pendingConfig["voice"]["confidence_threshold"] = val;
            hasChanges = true;
        }, 0.01f
    );
    confSlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(confSlider);
    
    // --- Max Commands Per Input ---
    int maxCmds = 3;
    if (pendingConfig.contains("voice") && pendingConfig["voice"].is_object()) {
        maxCmds = pendingConfig["voice"].value("max_commands_per_input", 3);
    }
    auto maxCmdsSlider = std::make_shared<UISlider>(
        "Max Commands/Input:", 1.0f, 10.0f, static_cast<float>(maxCmds),
        [this](float val) {
            if (!pendingConfig.contains("voice") || !pendingConfig["voice"].is_object())
                pendingConfig["voice"] = nlohmann::json::object();
            pendingConfig["voice"]["max_commands_per_input"] = static_cast<int>(val);
            hasChanges = true;
        }, 1.0f
    );
    maxCmdsSlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(maxCmdsSlider);
    
    // --- Require Multi-Command Confirmation ---
    bool reqConfirm = true;
    if (pendingConfig.contains("voice") && pendingConfig["voice"].is_object()) {
        reqConfirm = pendingConfig["voice"].value("require_multi_command_confirmation", true);
    }
    auto reqConfirmToggle = std::make_shared<UIToggle>(
        "Multi-Cmd Confirm:", reqConfirm,
        [this](bool val) {
            if (!pendingConfig.contains("voice") || !pendingConfig["voice"].is_object())
                pendingConfig["voice"] = nlohmann::json::object();
            pendingConfig["voice"]["require_multi_command_confirmation"] = val;
            hasChanges = true;
        }
    );
    reqConfirmToggle->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(reqConfirmToggle);
}

// =========================================================
// Audio Tab — Whisper STT, silence detection, mic settings
// =========================================================
void UISettingsMenu::createAudioWidgets() {
    float widgetWidth = scrollBox->getSize().x - 30;
    float widgetHeight = 45.0f;
    
    // --- Whisper Model ---
    std::string model = "ggml-base.en.bin";
    if (pendingConfig.contains("whisper") && pendingConfig["whisper"].is_object() &&
        pendingConfig["whisper"].contains("whisper_model")) {
        model = pendingConfig["whisper"]["whisper_model"].get<std::string>();
    }
    auto modelButton = std::make_shared<UIButton>(
        "Whisper Model: " + model,
        [this]() { cycleModel(); }
    );
    modelButton->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(modelButton);
    
    // --- Temperature ---
    float temperature = 0.0f;
    if (pendingConfig.contains("whisper") && pendingConfig["whisper"].is_object()) {
        temperature = pendingConfig["whisper"].value("temperature", 0.0f);
    }
    auto tempSlider = std::make_shared<UISlider>(
        "Whisper Temp:", 0.0f, 1.0f, temperature,
        [this](float val) {
            if (!pendingConfig.contains("whisper") || !pendingConfig["whisper"].is_object())
                pendingConfig["whisper"] = nlohmann::json::object();
            pendingConfig["whisper"]["temperature"] = val;
            hasChanges = true;
        }, 0.05f
    );
    tempSlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(tempSlider);
    
    // --- Beam Size ---
    int beamSize = 5;
    if (pendingConfig.contains("whisper") && pendingConfig["whisper"].is_object()) {
        beamSize = pendingConfig["whisper"].value("beam_size", 5);
    }
    auto beamSlider = std::make_shared<UISlider>(
        "Beam Size:", 1.0f, 10.0f, static_cast<float>(beamSize),
        [this](float val) {
            if (!pendingConfig.contains("whisper") || !pendingConfig["whisper"].is_object())
                pendingConfig["whisper"] = nlohmann::json::object();
            pendingConfig["whisper"]["beam_size"] = static_cast<int>(val);
            hasChanges = true;
        }, 1.0f
    );
    beamSlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(beamSlider);
    
    // --- Suppress Blank ---
    bool suppressBlank = true;
    if (pendingConfig.contains("whisper") && pendingConfig["whisper"].is_object()) {
        suppressBlank = pendingConfig["whisper"].value("suppress_blank", true);
    }
    auto suppressToggle = std::make_shared<UIToggle>(
        "Suppress Blank:", suppressBlank,
        [this](bool val) {
            if (!pendingConfig.contains("whisper") || !pendingConfig["whisper"].is_object())
                pendingConfig["whisper"] = nlohmann::json::object();
            pendingConfig["whisper"]["suppress_blank"] = val;
            hasChanges = true;
        }
    );
    suppressToggle->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(suppressToggle);
    
    // --- Sampling Strategy ---
    std::string sampStrategy = "beam";
    if (pendingConfig.contains("whisper") && pendingConfig["whisper"].is_object()) {
        sampStrategy = pendingConfig["whisper"].value("sampling_strategy", "beam");
    }
    std::vector<std::string> sampOptions = {"beam", "greedy"};
    int sampIdx = (sampStrategy == "greedy") ? 1 : 0;
    auto sampDropdown = std::make_shared<UIDropdown>(
        "Sampling Strategy:", sampOptions, sampIdx,
        [this](int, const std::string& val) {
            if (!pendingConfig.contains("whisper") || !pendingConfig["whisper"].is_object())
                pendingConfig["whisper"] = nlohmann::json::object();
            pendingConfig["whisper"]["sampling_strategy"] = val;
            hasChanges = true;
        }
    );
    sampDropdown->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(sampDropdown);
    
    // --- Language ---
    std::string lang = "en";
    if (pendingConfig.contains("whisper") && pendingConfig["whisper"].is_object()) {
        lang = pendingConfig["whisper"].value("language", "en");
    }
    std::vector<std::string> langOptions = {"en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh", "auto"};
    int langIdx = 0;
    for (size_t i = 0; i < langOptions.size(); ++i) {
        if (langOptions[i] == lang) { langIdx = static_cast<int>(i); break; }
    }
    auto langDropdown = std::make_shared<UIDropdown>(
        "Language:", langOptions, langIdx,
        [this](int, const std::string& val) {
            if (!pendingConfig.contains("whisper") || !pendingConfig["whisper"].is_object())
                pendingConfig["whisper"] = nlohmann::json::object();
            pendingConfig["whisper"]["language"] = val;
            hasChanges = true;
        }
    );
    langDropdown->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(langDropdown);
    
    // --- No Speech Threshold ---
    float noSpeechThresh = 0.6f;
    if (pendingConfig.contains("whisper") && pendingConfig["whisper"].is_object()) {
        noSpeechThresh = pendingConfig["whisper"].value("no_speech_threshold", 0.6f);
    }
    auto noSpeechSlider = std::make_shared<UISlider>(
        "No Speech Threshold:", 0.0f, 1.0f, noSpeechThresh,
        [this](float val) {
            if (!pendingConfig.contains("whisper") || !pendingConfig["whisper"].is_object())
                pendingConfig["whisper"] = nlohmann::json::object();
            pendingConfig["whisper"]["no_speech_threshold"] = val;
            hasChanges = true;
        }, 0.01f
    );
    noSpeechSlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(noSpeechSlider);
    
    // --- Entropy Threshold ---
    float entropyThresh = 2.4f;
    if (pendingConfig.contains("whisper") && pendingConfig["whisper"].is_object()) {
        entropyThresh = pendingConfig["whisper"].value("entropy_threshold", 2.4f);
    }
    auto entropySlider = std::make_shared<UISlider>(
        "Entropy Threshold:", 0.0f, 5.0f, entropyThresh,
        [this](float val) {
            if (!pendingConfig.contains("whisper") || !pendingConfig["whisper"].is_object())
                pendingConfig["whisper"] = nlohmann::json::object();
            pendingConfig["whisper"]["entropy_threshold"] = val;
            hasChanges = true;
        }, 0.1f
    );
    entropySlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(entropySlider);
    
    // --- Silence Threshold (global) ---
    float silenceThresh = pendingConfig.value("silence_threshold", 0.02f);
    auto silenceSlider = std::make_shared<UISlider>(
        "Silence Threshold:", 0.0f, 0.1f, silenceThresh,
        [this](float val) {
            pendingConfig["silence_threshold"] = val;
            hasChanges = true;
        }, 0.001f
    );
    silenceSlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(silenceSlider);
    
    // --- Silence Timeout ---
    int silenceTimeout = pendingConfig.value("silence_timeout_ms", 4000);
    auto silenceTimeoutSlider = std::make_shared<UISlider>(
        "Silence Timeout (ms):", 500.0f, 10000.0f, static_cast<float>(silenceTimeout),
        [this](float val) {
            pendingConfig["silence_timeout_ms"] = static_cast<int>(val);
            hasChanges = true;
        }, 100.0f
    );
    silenceTimeoutSlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(silenceTimeoutSlider);
    
    // --- Max Listening Time ---
    int maxListening = 10000;
    if (pendingConfig.contains("whisper") && pendingConfig["whisper"].is_object()) {
        maxListening = pendingConfig["whisper"].value("max_listening_ms", 10000);
    }
    auto maxListenSlider = std::make_shared<UISlider>(
        "Max Listen (ms):", 1000.0f, 30000.0f, static_cast<float>(maxListening),
        [this](float val) {
            if (!pendingConfig.contains("whisper") || !pendingConfig["whisper"].is_object())
                pendingConfig["whisper"] = nlohmann::json::object();
            pendingConfig["whisper"]["max_listening_ms"] = static_cast<int>(val);
            hasChanges = true;
        }, 500.0f
    );
    maxListenSlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(maxListenSlider);
    
    // --- Min Speech Duration ---
    int minSpeech = 500;
    if (pendingConfig.contains("whisper") && pendingConfig["whisper"].is_object()) {
        minSpeech = pendingConfig["whisper"].value("min_speech_ms", 500);
    }
    auto minSpeechSlider = std::make_shared<UISlider>(
        "Min Speech (ms):", 100.0f, 2000.0f, static_cast<float>(minSpeech),
        [this](float val) {
            if (!pendingConfig.contains("whisper") || !pendingConfig["whisper"].is_object())
                pendingConfig["whisper"] = nlohmann::json::object();
            pendingConfig["whisper"]["min_speech_ms"] = static_cast<int>(val);
            hasChanges = true;
        }, 50.0f
    );
    minSpeechSlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(minSpeechSlider);
}

// =========================================================
// Vision Tab — Vision AI toggle and model
// =========================================================
void UISettingsMenu::createVisionWidgets() {
    float widgetWidth = scrollBox->getSize().x - 30;
    float widgetHeight = 45.0f;
    
    // --- Vision AI Enabled ---
    bool visionEnabled = false;
    if (pendingConfig.contains("vision") && pendingConfig["vision"].is_object()) {
        visionEnabled = pendingConfig["vision"].value("enabled", false);
    }
    auto visionToggle = std::make_shared<UIToggle>(
        "Vision AI:", visionEnabled,
        [this](bool val) {
            if (!pendingConfig.contains("vision") || !pendingConfig["vision"].is_object())
                pendingConfig["vision"] = nlohmann::json::object();
            pendingConfig["vision"]["enabled"] = val;
            hasChanges = true;
        }
    );
    visionToggle->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(visionToggle);
    
    // --- Vision Model ---
    std::string visionModel = pendingConfig.value("vision_model", "llama3.2-vision:11b");
    std::vector<std::string> visionModels = {"llama3.2-vision:11b", "llava:7b", "llava:13b"};
    int vmIdx = 0;
    for (size_t i = 0; i < visionModels.size(); ++i) {
        if (visionModels[i] == visionModel) { vmIdx = static_cast<int>(i); break; }
    }
    auto visionModelDropdown = std::make_shared<UIDropdown>(
        "Vision Model:", visionModels, vmIdx,
        [this](int, const std::string& val) {
            pendingConfig["vision_model"] = val;
            hasChanges = true;
        }
    );
    visionModelDropdown->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(visionModelDropdown);
}

// =========================================================
// UI/Graphics Tab — fonts, blur, window behavior
// =========================================================
void UISettingsMenu::createUIGraphicsWidgets() {
    float widgetWidth = scrollBox->getSize().x - 30;
    float widgetHeight = 45.0f;
    
    // --- Font Selection ---
    std::vector<std::string> availableFonts = getFontList();
    std::string currentFont = "Consolas";
    if (pendingConfig.contains("ui") && pendingConfig["ui"].is_object() &&
        pendingConfig["ui"].contains("font_name")) {
        currentFont = pendingConfig["ui"]["font_name"].get<std::string>();
    }
    int fontIndex = 0;
    for (size_t i = 0; i < availableFonts.size(); ++i) {
        if (availableFonts[i] == currentFont) { fontIndex = static_cast<int>(i); break; }
    }
    auto fontDropdown = std::make_shared<UIDropdown>(
        "Font:", availableFonts, fontIndex,
        [this](int, const std::string& fontName) {
            if (!pendingConfig.contains("ui") || !pendingConfig["ui"].is_object())
                pendingConfig["ui"] = nlohmann::json::object();
            pendingConfig["ui"]["font_name"] = fontName;
            hasChanges = true;
        }
    );
    fontDropdown->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(fontDropdown);
    
    // --- Font Size ---
    int fontSize = 16;
    if (pendingConfig.contains("ui") && pendingConfig["ui"].is_object()) {
        fontSize = pendingConfig["ui"].value("font_size", 16);
    }
    auto fontSizeSlider = std::make_shared<UISlider>(
        "Font Size:", 8.0f, 32.0f, static_cast<float>(fontSize),
        [this](float val) {
            if (!pendingConfig.contains("ui") || !pendingConfig["ui"].is_object())
                pendingConfig["ui"] = nlohmann::json::object();
            pendingConfig["ui"]["font_size"] = static_cast<int>(val);
            hasChanges = true;
        }, 1.0f
    );
    fontSizeSlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(fontSizeSlider);
    
    // --- Minimize Behavior ---
    std::string minimize = "Side";
    if (pendingConfig.contains("ui") && pendingConfig["ui"].is_object()) {
        minimize = pendingConfig["ui"].value("Minimize", "Side");
    }
    std::vector<std::string> minOptions = {"Side", "Taskbar", "Close"};
    int minIdx = 0;
    for (size_t i = 0; i < minOptions.size(); ++i) {
        if (minOptions[i] == minimize) { minIdx = static_cast<int>(i); break; }
    }
    auto minDropdown = std::make_shared<UIDropdown>(
        "Minimize:", minOptions, minIdx,
        [this](int, const std::string& val) {
            if (!pendingConfig.contains("ui") || !pendingConfig["ui"].is_object())
                pendingConfig["ui"] = nlohmann::json::object();
            pendingConfig["ui"]["Minimize"] = val;
            hasChanges = true;
        }
    );
    minDropdown->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(minDropdown);
    
    // --- Glass Blur ---
    bool blurEnabled = true;
    if (pendingConfig.contains("blur") && pendingConfig["blur"].is_object()) {
        blurEnabled = pendingConfig["blur"].value("enabled", true);
    }
    auto blurToggle = std::make_shared<UIToggle>(
        "Glass Blur:", blurEnabled,
        [this](bool val) {
            if (!pendingConfig.contains("blur") || !pendingConfig["blur"].is_object())
                pendingConfig["blur"] = nlohmann::json::object();
            pendingConfig["blur"]["enabled"] = val;
            hasChanges = true;
        }
    );
    blurToggle->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(blurToggle);
    
    // --- Blur Opacity ---
    float blurOpacity = 0.99f;
    if (pendingConfig.contains("blur") && pendingConfig["blur"].is_object()) {
        blurOpacity = pendingConfig["blur"].value("opacity", 0.99f);
    }
    auto blurOpacitySlider = std::make_shared<UISlider>(
        "Blur Opacity:", 0.0f, 1.0f, blurOpacity,
        [this](float val) {
            if (!pendingConfig.contains("blur") || !pendingConfig["blur"].is_object())
                pendingConfig["blur"] = nlohmann::json::object();
            pendingConfig["blur"]["opacity"] = val;
            hasChanges = true;
        }, 0.01f
    );
    blurOpacitySlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(blurOpacitySlider);
    
    // --- Blur Intensity ---
    int blurIntensity = 2;
    if (pendingConfig.contains("blur") && pendingConfig["blur"].is_object()) {
        blurIntensity = pendingConfig["blur"].value("intensity", 2);
    }
    auto blurIntensitySlider = std::make_shared<UISlider>(
        "Blur Intensity:", 1.0f, 5.0f, static_cast<float>(blurIntensity),
        [this](float val) {
            if (!pendingConfig.contains("blur") || !pendingConfig["blur"].is_object())
                pendingConfig["blur"] = nlohmann::json::object();
            pendingConfig["blur"]["intensity"] = static_cast<int>(val);
            hasChanges = true;
        }, 1.0f
    );
    blurIntensitySlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(blurIntensitySlider);
}

// =========================================================
// Preferences Tab — personality, custom prompts, behavior
// =========================================================
void UISettingsMenu::createPreferencesWidgets() {
    float widgetWidth = scrollBox->getSize().x - 30;
    float widgetHeight = 45.0f;
    
    // --- Personality ---
    std::string personalityLabel = "Professional";
    if (pendingConfig.contains("personality") && pendingConfig["personality"].is_object() &&
        pendingConfig["personality"].contains("custom_prompt")) {
        std::string prompt = pendingConfig["personality"]["custom_prompt"].get<std::string>();
        if (prompt.find("helpful AI assistant") != std::string::npos) personalityLabel = "Professional";
        else if (prompt.find("sarcastic") != std::string::npos) personalityLabel = "Sarcastic";
        else if (prompt.find("friendly and enthusiastic") != std::string::npos) personalityLabel = "Friendly";
        else if (prompt.find("military") != std::string::npos) personalityLabel = "Military";
        else if (prompt.find("brief and direct") != std::string::npos) personalityLabel = "Short";
        else if (prompt.find("concise but warm") != std::string::npos) personalityLabel = "Short & Sweet";
        else personalityLabel = "Custom";
    }
    auto personalityButton = std::make_shared<UIButton>(
        "Personality: " + personalityLabel,
        [this]() { cyclePersonality(); }
    );
    personalityButton->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(personalityButton);
    
    // --- Use Custom Prompt ---
    bool useCustom = false;
    if (pendingConfig.contains("personality") && pendingConfig["personality"].is_object()) {
        useCustom = pendingConfig["personality"].value("use_custom_prompt", false);
    }
    auto customToggle = std::make_shared<UIToggle>(
        "Use Custom Prompt:", useCustom,
        [this](bool val) {
            if (!pendingConfig.contains("personality") || !pendingConfig["personality"].is_object())
                pendingConfig["personality"] = nlohmann::json::object();
            pendingConfig["personality"]["use_custom_prompt"] = val;
            hasChanges = true;
        }
    );
    customToggle->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(customToggle);
    
    // --- Custom Prompt Text ---
    std::string customPrompt = "";
    if (pendingConfig.contains("personality") && pendingConfig["personality"].is_object()) {
        customPrompt = pendingConfig["personality"].value("custom_prompt", "");
    }
    auto promptArea = std::make_shared<UITextArea>(
        "Custom Prompt:", customPrompt,
        [this](const std::string& val) {
            if (!pendingConfig.contains("personality") || !pendingConfig["personality"].is_object())
                pendingConfig["personality"] = nlohmann::json::object();
            pendingConfig["personality"]["custom_prompt"] = val;
            hasChanges = true;
        }
    );
    promptArea->setSize(widgetWidth, 100.0f);
    scrollBox->addChild(promptArea);
    
    // --- Ollama Options ---
    // Temperature
    float ollamaTemp = 0.7f;
    if (pendingConfig.contains("ollama_options") && pendingConfig["ollama_options"].is_object()) {
        ollamaTemp = pendingConfig["ollama_options"].value("temperature", 0.7f);
    }
    auto ollamaTempSlider = std::make_shared<UISlider>(
        "Ollama Temperature:", 0.0f, 2.0f, ollamaTemp,
        [this](float val) {
            if (!pendingConfig.contains("ollama_options") || !pendingConfig["ollama_options"].is_object())
                pendingConfig["ollama_options"] = nlohmann::json::object();
            pendingConfig["ollama_options"]["temperature"] = val;
            hasChanges = true;
        }, 0.05f
    );
    ollamaTempSlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(ollamaTempSlider);
    
    // Top P
    float topP = 0.9f;
    if (pendingConfig.contains("ollama_options") && pendingConfig["ollama_options"].is_object()) {
        topP = pendingConfig["ollama_options"].value("top_p", 0.9f);
    }
    auto topPSlider = std::make_shared<UISlider>(
        "Top P:", 0.0f, 1.0f, topP,
        [this](float val) {
            if (!pendingConfig.contains("ollama_options") || !pendingConfig["ollama_options"].is_object())
                pendingConfig["ollama_options"] = nlohmann::json::object();
            pendingConfig["ollama_options"]["top_p"] = val;
            hasChanges = true;
        }, 0.01f
    );
    topPSlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(topPSlider);
    
    // Top K
    int topK = 40;
    if (pendingConfig.contains("ollama_options") && pendingConfig["ollama_options"].is_object()) {
        topK = pendingConfig["ollama_options"].value("top_k", 40);
    }
    auto topKSlider = std::make_shared<UISlider>(
        "Top K:", 1.0f, 100.0f, static_cast<float>(topK),
        [this](float val) {
            if (!pendingConfig.contains("ollama_options") || !pendingConfig["ollama_options"].is_object())
                pendingConfig["ollama_options"] = nlohmann::json::object();
            pendingConfig["ollama_options"]["top_k"] = static_cast<int>(val);
            hasChanges = true;
        }, 1.0f
    );
    topKSlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(topKSlider);
    
    // Repeat Penalty
    float repeatPenalty = 1.1f;
    if (pendingConfig.contains("ollama_options") && pendingConfig["ollama_options"].is_object()) {
        repeatPenalty = pendingConfig["ollama_options"].value("repeat_penalty", 1.1f);
    }
    auto repeatSlider = std::make_shared<UISlider>(
        "Repeat Penalty:", 1.0f, 2.0f, repeatPenalty,
        [this](float val) {
            if (!pendingConfig.contains("ollama_options") || !pendingConfig["ollama_options"].is_object())
                pendingConfig["ollama_options"] = nlohmann::json::object();
            pendingConfig["ollama_options"]["repeat_penalty"] = val;
            hasChanges = true;
        }, 0.05f
    );
    repeatSlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(repeatSlider);
    
    // Context Size
    int numCtx = 4096;
    if (pendingConfig.contains("ollama_options") && pendingConfig["ollama_options"].is_object()) {
        numCtx = pendingConfig["ollama_options"].value("num_ctx", 4096);
    }
    auto ctxSlider = std::make_shared<UISlider>(
        "Context Size:", 512.0f, 32768.0f, static_cast<float>(numCtx),
        [this](float val) {
            if (!pendingConfig.contains("ollama_options") || !pendingConfig["ollama_options"].is_object())
                pendingConfig["ollama_options"] = nlohmann::json::object();
            pendingConfig["ollama_options"]["num_ctx"] = static_cast<int>(val);
            hasChanges = true;
        }, 512.0f
    );
    ctxSlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(ctxSlider);
}

// =========================================================
// Memory Tab — data collection, cache management
// =========================================================
void UISettingsMenu::createMemoryWidgets() {
    float widgetWidth = scrollBox->getSize().x - 30;
    float widgetHeight = 45.0f;
    
    // --- Data Collection: Max New Entries ---
    int maxEntries = 100000;
    if (pendingConfig.contains("data_collection") && pendingConfig["data_collection"].is_object()) {
        maxEntries = pendingConfig["data_collection"].value("max_new_entries_per_run", 100000);
    }
    auto maxEntriesSlider = std::make_shared<UISlider>(
        "Max Entries/Run:", 1000.0f, 500000.0f, static_cast<float>(maxEntries),
        [this](float val) {
            if (!pendingConfig.contains("data_collection") || !pendingConfig["data_collection"].is_object())
                pendingConfig["data_collection"] = nlohmann::json::object();
            pendingConfig["data_collection"]["max_new_entries_per_run"] = static_cast<int>(val);
            hasChanges = true;
        }, 1000.0f
    );
    maxEntriesSlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(maxEntriesSlider);
    
    // --- Clear Merged Cache on Merge ---
    bool clearCache = pendingConfig.value("clear_merged_cache_on_merge", false);
    auto clearCacheToggle = std::make_shared<UIToggle>(
        "Clear Cache on Merge:", clearCache,
        [this](bool val) {
            pendingConfig["clear_merged_cache_on_merge"] = val;
            hasChanges = true;
        }
    );
    clearCacheToggle->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(clearCacheToggle);
    
    // --- Data Structuring Enabled ---
    bool structEnabled = true;
    if (pendingConfig.contains("data_collection") && pendingConfig["data_collection"].is_object() &&
        pendingConfig["data_collection"].contains("data_structuring") && pendingConfig["data_collection"]["data_structuring"].is_object()) {
        structEnabled = pendingConfig["data_collection"]["data_structuring"].value("enabled", true);
    }
    auto structToggle = std::make_shared<UIToggle>(
        "Data Structuring:", structEnabled,
        [this](bool val) {
            if (!pendingConfig.contains("data_collection") || !pendingConfig["data_collection"].is_object())
                pendingConfig["data_collection"] = nlohmann::json::object();
            if (!pendingConfig["data_collection"].contains("data_structuring") || !pendingConfig["data_collection"]["data_structuring"].is_object())
                pendingConfig["data_collection"]["data_structuring"] = nlohmann::json::object();
            pendingConfig["data_collection"]["data_structuring"]["enabled"] = val;
            hasChanges = true;
        }
    );
    structToggle->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(structToggle);
    
    // --- Structuring Mode ---
    std::string structMode = "qa";
    if (pendingConfig.contains("data_collection") && pendingConfig["data_collection"].is_object() &&
        pendingConfig["data_collection"].contains("data_structuring") && pendingConfig["data_collection"]["data_structuring"].is_object()) {
        structMode = pendingConfig["data_collection"]["data_structuring"].value("mode", "qa");
    }
    std::vector<std::string> modeOptions = {"qa", "summary", "raw"};
    int modeIdx = 0;
    for (size_t i = 0; i < modeOptions.size(); ++i) {
        if (modeOptions[i] == structMode) { modeIdx = static_cast<int>(i); break; }
    }
    auto structModeDropdown = std::make_shared<UIDropdown>(
        "Structuring Mode:", modeOptions, modeIdx,
        [this](int, const std::string& val) {
            if (!pendingConfig.contains("data_collection") || !pendingConfig["data_collection"].is_object())
                pendingConfig["data_collection"] = nlohmann::json::object();
            if (!pendingConfig["data_collection"].contains("data_structuring") || !pendingConfig["data_collection"]["data_structuring"].is_object())
                pendingConfig["data_collection"]["data_structuring"] = nlohmann::json::object();
            pendingConfig["data_collection"]["data_structuring"]["mode"] = val;
            hasChanges = true;
        }
    );
    structModeDropdown->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(structModeDropdown);
    
    // --- Parallel Requests ---
    int parallelReq = 4;
    if (pendingConfig.contains("data_collection") && pendingConfig["data_collection"].is_object() &&
        pendingConfig["data_collection"].contains("data_structuring") && pendingConfig["data_collection"]["data_structuring"].is_object()) {
        parallelReq = pendingConfig["data_collection"]["data_structuring"].value("parallel_requests", 4);
    }
    auto parallelSlider = std::make_shared<UISlider>(
        "Parallel Requests:", 1.0f, 16.0f, static_cast<float>(parallelReq),
        [this](float val) {
            if (!pendingConfig.contains("data_collection") || !pendingConfig["data_collection"].is_object())
                pendingConfig["data_collection"] = nlohmann::json::object();
            if (!pendingConfig["data_collection"].contains("data_structuring") || !pendingConfig["data_collection"]["data_structuring"].is_object())
                pendingConfig["data_collection"]["data_structuring"] = nlohmann::json::object();
            pendingConfig["data_collection"]["data_structuring"]["parallel_requests"] = static_cast<int>(val);
            hasChanges = true;
        }, 1.0f
    );
    parallelSlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(parallelSlider);
    
    // --- HuggingFace PDFs per Dataset ---
    int maxPdfsPerDS = 2000;
    if (pendingConfig.contains("data_collection") && pendingConfig["data_collection"].is_object()) {
        maxPdfsPerDS = pendingConfig["data_collection"].value("max_huggingface_pdfs_per_dataset", 2000);
    }
    auto pdfsPerDsSlider = std::make_shared<UISlider>(
        "HF PDFs/Dataset:", 100.0f, 10000.0f, static_cast<float>(maxPdfsPerDS),
        [this](float val) {
            if (!pendingConfig.contains("data_collection") || !pendingConfig["data_collection"].is_object())
                pendingConfig["data_collection"] = nlohmann::json::object();
            pendingConfig["data_collection"]["max_huggingface_pdfs_per_dataset"] = static_cast<int>(val);
            hasChanges = true;
        }, 100.0f
    );
    pdfsPerDsSlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(pdfsPerDsSlider);
    
    // --- HuggingFace Total PDFs ---
    int maxPdfsTotal = 10000;
    if (pendingConfig.contains("data_collection") && pendingConfig["data_collection"].is_object()) {
        maxPdfsTotal = pendingConfig["data_collection"].value("max_huggingface_pdfs_total", 10000);
    }
    auto pdfsTotalSlider = std::make_shared<UISlider>(
        "HF PDFs Total:", 100.0f, 50000.0f, static_cast<float>(maxPdfsTotal),
        [this](float val) {
            if (!pendingConfig.contains("data_collection") || !pendingConfig["data_collection"].is_object())
                pendingConfig["data_collection"] = nlohmann::json::object();
            pendingConfig["data_collection"]["max_huggingface_pdfs_total"] = static_cast<int>(val);
            hasChanges = true;
        }, 500.0f
    );
    pdfsTotalSlider->setSize(widgetWidth, widgetHeight);
    scrollBox->addChild(pdfsTotalSlider);
}

void UISettingsMenu::update(const InputState& input, float dt) {
    UIPanel::update(input, dt);
    
    if (!isVisible()) return;
    
    // Check refresh flag BEFORE updating widgets
    bool shouldRefresh = needsWidgetRefresh;
    if (shouldRefresh) {
        needsWidgetRefresh = false;
        LOG_DEBUG("UISettingsMenu", "Refreshing widgets at START of frame (before update)");
        createWidgets();
        return;
    }
    
    // Position and update tab buttons
    float tabX = position.x + 10.0f;
    float tabW = (size.x - 20.0f) / 7.0f;  // 7 tabs
    
    tabGeneralBtn_->setPosition(tabX, position.y + kTabBarY);
    tabGeneralBtn_->setSize(tabW - 2, 28);
    tabVoiceBtn_->setPosition(tabX + tabW, position.y + kTabBarY);
    tabVoiceBtn_->setSize(tabW - 2, 28);
    tabAudioBtn_->setPosition(tabX + tabW * 2, position.y + kTabBarY);
    tabAudioBtn_->setSize(tabW - 2, 28);
    tabVisionBtn_->setPosition(tabX + tabW * 3, position.y + kTabBarY);
    tabVisionBtn_->setSize(tabW - 2, 28);
    tabUIGraphicsBtn_->setPosition(tabX + tabW * 4, position.y + kTabBarY);
    tabUIGraphicsBtn_->setSize(tabW - 2, 28);
    tabPreferencesBtn_->setPosition(tabX + tabW * 5, position.y + kTabBarY);
    tabPreferencesBtn_->setSize(tabW - 2, 28);
    tabMemoryBtn_->setPosition(tabX + tabW * 6, position.y + kTabBarY);
    tabMemoryBtn_->setSize(tabW - 2, 28);
    
    tabGeneralBtn_->update(input, dt);
    tabVoiceBtn_->update(input, dt);
    tabAudioBtn_->update(input, dt);
    tabVisionBtn_->update(input, dt);
    tabUIGraphicsBtn_->update(input, dt);
    tabPreferencesBtn_->update(input, dt);
    tabMemoryBtn_->update(input, dt);
    
    // Update scroll box position below tab bar
    scrollBox->setPosition(position.x + 10, position.y + kContentTopY);
    scrollBox->setSize(size.x - 20, size.y - kContentTopY - 20 - 50);
    scrollBox->update(input, dt);
    
    // Update Save & Close button
    if (saveButton) {
        float yPos = position.y + size.y - 50;
        saveButton->setPosition(position.x + 10, yPos);
        saveButton->setSize((size.x - 30) / 2, 40);
        saveButton->update(input, dt);
    }
    
    // Update Cancel button
    if (cancelButton) {
        float yPos = position.y + size.y - 50;
        cancelButton->setPosition(position.x + 10 + (size.x - 30) / 2 + 10, yPos);
        cancelButton->setSize((size.x - 30) / 2, 40);
        cancelButton->update(input, dt);
    }
}

bool UISettingsMenu::drawOverlay(OverlayRenderer& renderer)
{
    if (!UIPanel::drawOverlay(renderer)) return false;
    
    using namespace UITheme;
    
    // Draw tab buttons
    tabGeneralBtn_->drawOverlay(renderer, position);
    tabVoiceBtn_->drawOverlay(renderer, position);
    tabAudioBtn_->drawOverlay(renderer, position);
    tabVisionBtn_->drawOverlay(renderer, position);
    tabUIGraphicsBtn_->drawOverlay(renderer, position);
    tabPreferencesBtn_->drawOverlay(renderer, position);
    tabMemoryBtn_->drawOverlay(renderer, position);
    
    // Active tab indicator (2px underline)
    float tabX = position.x + 10.0f;
    float tabW = (size.x - 20.0f) / 7.0f;
    int tabIdx = static_cast<int>(activeTab_);
    float indicatorX = tabX + tabW * tabIdx;
    renderer.drawRect({indicatorX, position.y + kTabBarY + 28.0f}, {tabW - 2, 2.0f}, Colors::Primary);
    
    // Draw separator line under tab bar
    renderer.drawRect({position.x + 10, position.y + kContentTopY - 2}, {size.x - 20, 1.0f}, Colors::BorderSubtle);
    
    // Draw scroll box content
    scrollBox->drawOverlay(renderer, position);
    
    // Save button
    if (saveButton) {
        Vec2 btnPos = saveButton->getPosition();
        Vec2 btnSize = saveButton->getSize();
        
        renderer.drawRoundedRect(btnPos, btnSize, Colors::SuccessBg, Sizes::WidgetRadius);
        renderer.drawRoundedBorder(btnPos, btnSize, Colors::Success, Sizes::WidgetRadius);
        
        float textY = btnPos.y + (btnSize.y / 2.0f) - 8;
        renderer.drawText({btnPos.x + 10, textY}, "Save & Close", Colors::TextWhite);
    }
    
    // Cancel button
    if (cancelButton) {
        Vec2 btnPos = cancelButton->getPosition();
        Vec2 btnSize = cancelButton->getSize();
        
        renderer.drawRoundedRect(btnPos, btnSize, Colors::DangerBg, Sizes::WidgetRadius);
        renderer.drawRoundedBorder(btnPos, btnSize, Colors::Danger, Sizes::WidgetRadius);
        
        float textY = btnPos.y + (btnSize.y / 2.0f) - 8;
        renderer.drawText({btnPos.x + 10, textY}, "Cancel", Colors::TextWhite);
    }
    
    // Unsaved indicator
    if (hasChanges) {
        renderer.drawText({position.x + size.x - 150, position.y + 8}, "* Unsaved", Colors::WarningLight);
    }
    
    renderer.popClipRect();
    return true;
}

