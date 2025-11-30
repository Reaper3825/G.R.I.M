#include "ui_settings_menu.hpp"
#include "overlay_renderer.hpp"
#include "logger.hpp"
#include "input_parser.hpp"
#include "ui_root.hpp"  // ? NEW: For accessing renderer
#include "../voice/voice_speak.hpp"  // ? NEW: For updating speaker dynamically
#include <fstream>
#include <functional>
#include <filesystem>

// External global aiConfig that needs to be updated when settings change
extern nlohmann::json aiConfig;

UISettingsMenu::UISettingsMenu()
    : UIPanel("Settings", true), hasChanges(false), isRefreshing(false), needsWidgetRefresh(false)
{
    position = { 200, 425 };
    size = { 500, 600 };  // Increased height to accommodate scrollbox
    setVisible(false);
    setBackground(0xC0202020);
    
    // Create scroll box for content
    scrollBox = std::make_shared<UIScrollBox>();
    scrollBox->setPosition(position.x + 10, position.y + 40);  // Below title bar
    scrollBox->setSize(size.x - 20, size.y - 50);  // Leave room for title
    scrollBox->setChildSpacing(5.0f);
    
    loadConfig();
}

// ? NEW: Scan for available speaker embeddings
std::vector<std::string> UISettingsMenu::getSpeakerEmbeddings() {
    std::vector<std::string> embeddings;
    embeddings.push_back("default");  // Always include default
    
    try {
        std::string embeddingDir = "D:/G.R.I.M/resources/voices/embeddings";
        
        if (std::filesystem::exists(embeddingDir)) {
            for (const auto& entry : std::filesystem::directory_iterator(embeddingDir)) {
                if (entry.path().extension() == ".npz") {
                    std::string speakerName = entry.path().stem().string();
                    if (speakerName != "default") {  // Don't duplicate default
                        embeddings.push_back(speakerName);
                    }
                }
            }
        }
        
        LOG_DEBUG("UISettingsMenu", "Found " + std::to_string(embeddings.size()) + " speaker embeddings");
    } catch (const std::exception& e) {
        LOG_ERROR("UISettingsMenu", std::string("Failed to scan embeddings: ") + e.what());
    }
    
    // If only default found, add a helpful message option
    if (embeddings.size() == 1) {
        embeddings.push_back("(no custom voices)");
    }
    
    return embeddings;
}

// ? NEW: Scan for available fonts in resources/fonts directory
std::vector<std::string> UISettingsMenu::getFontList() {
    std::vector<std::string> fonts;
    
    // Add system fonts first
    fonts.push_back("Consolas");
    fonts.push_back("Courier New");
    fonts.push_back("Arial");
    fonts.push_back("Segoe UI");
    
    try {
        std::string fontDir = "D:/G.R.I.M/resources/fonts";
        
        if (std::filesystem::exists(fontDir)) {
            for (const auto& entry : std::filesystem::directory_iterator(fontDir)) {
                if (entry.path().extension() == ".ttf" || entry.path().extension() == ".otf") {
                    std::string fontName = entry.path().stem().string();
                    // Add only if not already in list
                    if (std::find(fonts.begin(), fonts.end(), fontName) == fonts.end()) {
                        fonts.push_back(fontName);
                    }
                }
            }
        }
        
        LOG_DEBUG("UISettingsMenu", "Found " + std::to_string(fonts.size()) + " fonts");
    } catch (const std::exception& e) {
        LOG_ERROR("UISettingsMenu", std::string("Failed to scan fonts: ") + e.what());
    }
    
    return fonts;
}

void UISettingsMenu::loadConfig() {
    try {
        std::ifstream f("ai_config.json");
        if (f.is_open()) {
            f >> config;
        } else {
            // Match the actual default structure from bootstrap_config
            config = {
                {"backend", "auto"},
                {"voice", {
                    {"engine", "coqui"},
                    {"speaker", "default"},
                    {"available_speakers", nlohmann::json::array({"default", "p226", "grim"})}
                }},
                {"whisper", {
                    {"whisper_model", "ggml-base.en.bin"},
                    {"temperature", 0.0},
                    {"beam_size", 5},
                    {"suppress_blank", true}
                }},
                {"personality", {
                    {"custom_prompt", "You are GRIM, a helpful AI assistant. Be concise and professional."},
                    {"use_custom_prompt", false}
                }}
            };
        }
        LOG_DEBUG("UISettingsMenu", "Config loaded successfully");
    } catch (const std::exception& e) {
        LOG_ERROR("UISettingsMenu", std::string("Failed to load ai_config.json: ") + e.what());
    } catch (...) {
        LOG_ERROR("UISettingsMenu", "Failed to load ai_config.json (unknown error)");
    }
    
    pendingConfig = config;
    hasChanges = false;
    
    // ? FIX: DON'T create widgets in constructor!
    // createWidgets() will be called on first update() instead
    needsWidgetRefresh = true;  // ? Flag to create widgets on first update
    
    LOG_DEBUG("UISettingsMenu", "Config loaded, widgets will be created on first update");
}

void UISettingsMenu::saveConfig() {
    try {
        std::ofstream f("ai_config.json");
        if (!f.is_open()) {
            LOG_ERROR("UISettingsMenu", "Failed to open ai_config.json for writing");
            return;
        }
        
        f << config.dump(4);
        f.close();
        
        // Update the global aiConfig so changes take effect immediately
        aiConfig = config;
        
        LOG_DEBUG("UISettingsMenu", "Saved ai_config.json and updated global aiConfig");
        
        // Log whisper settings for verification
        if (config.contains("whisper")) {
            LOG_DEBUG("UISettingsMenu", "Whisper settings in saved config: " + config["whisper"].dump());
        }
    } catch (const std::exception& e) {
        LOG_ERROR("UISettingsMenu", std::string("Failed to save ai_config.json: ") + e.what());
    } catch (...) {
        LOG_ERROR("UISettingsMenu", "Failed to save ai_config.json (unknown error)");
    }
}

void UISettingsMenu::applyChanges() {
    if (!hasChanges) {
        LOG_DEBUG("UISettingsMenu", "No changes to apply");
        return;
    }
    
    LOG_DEBUG("UISettingsMenu", "Applying changes to config");
    
    // Log whisper settings being applied for debugging
    if (pendingConfig.contains("whisper")) {
        LOG_DEBUG("UISettingsMenu", "Whisper settings being applied:");
        LOG_DEBUG("UISettingsMenu", "  temperature: " + std::to_string(pendingConfig["whisper"].value("temperature", 0.0f)));
        LOG_DEBUG("UISettingsMenu", "  beam_size: " + std::to_string(pendingConfig["whisper"].value("beam_size", 5)));
        LOG_DEBUG("UISettingsMenu", "  suppress_blank: " + std::string(pendingConfig["whisper"].value("suppress_blank", true) ? "true" : "false"));
        if (pendingConfig["whisper"].contains("whisper_model")) {
            LOG_DEBUG("UISettingsMenu", "  whisper_model: " + pendingConfig["whisper"]["whisper_model"].get<std::string>());
        }
    }
    
    config = pendingConfig;
    saveConfig();
    hasChanges = false;
    
    // ? NEW: Update Voice module speaker if it changed
    if (pendingConfig.contains("voice") && pendingConfig["voice"].is_object() &&
        pendingConfig["voice"].contains("speaker")) {
        std::string newSpeaker = pendingConfig["voice"]["speaker"].get<std::string>();
        Voice::setSpeaker(newSpeaker);
        LOG_DEBUG("UISettingsMenu", "Voice speaker updated to: " + newSpeaker);
    }
    
    // ? NEW: Update UI font if it changed
    if (pendingConfig.contains("ui") && pendingConfig["ui"].is_object() &&
        pendingConfig["ui"].contains("font_name")) {
        std::string newFont = pendingConfig["ui"]["font_name"].get<std::string>();
        int fontSize = 16; // Default
        if (pendingConfig["ui"].contains("font_size")) {
            fontSize = pendingConfig["ui"]["font_size"].get<int>();
        }
        
        // Get UIRoot renderer and update font
        auto& uiRoot = UIRoot::get();
        uiRoot.getRenderer().setFont(newFont, fontSize);
        LOG_DEBUG("UISettingsMenu", "UI font updated to: " + newFont + " (size " + std::to_string(fontSize) + ")");
    }
    
    LOG_DEBUG("UISettingsMenu", "Settings applied and saved successfully");
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
    LOG_DEBUG("UISettingsMenu", "createWidgets() START - using container system");
    
    try {
        // Clear all existing children from scrollbox
        scrollBox->clearChildren();
        
        float widgetWidth = scrollBox->getSize().x - 30;  // Leave room for scrollbar
        float widgetHeight = 45.0f;
        
        // ===== Button 1: Backend =====
        std::string backend = pendingConfig.value("backend", "auto");
        auto backendButton = std::make_shared<UIButton>(
            "Backend: " + backend,
            [this]() { cycleBackend(); }
        );
        backendButton->setSize(widgetWidth, widgetHeight);
        scrollBox->addChild(backendButton);
        
        // ===== Button 2: Voice Engine =====
        std::string voiceEngine = "coqui";
        if (pendingConfig.contains("voice") && pendingConfig["voice"].is_object() &&
            pendingConfig["voice"].contains("engine")) {
            voiceEngine = pendingConfig["voice"]["engine"].get<std::string>();
        }
        auto voiceButton = std::make_shared<UIButton>(
            "Voice: " + voiceEngine,
            [this]() { cycleVoice(); }
        );
        voiceButton->setSize(widgetWidth, widgetHeight);
        scrollBox->addChild(voiceButton);
        
        // ===== Button 3: Speaker (conditional on Coqui) =====
        if (voiceEngine == "coqui") {
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
        }
        
        // ===== Button 4: Whisper Model =====
        std::string model = "ggml-base.en.bin";
        if (pendingConfig.contains("whisper") && pendingConfig["whisper"].is_object() &&
            pendingConfig["whisper"].contains("whisper_model")) {
            model = pendingConfig["whisper"]["whisper_model"].get<std::string>();
        }
        auto modelButton = std::make_shared<UIButton>(
            "Model: " + model,
            [this]() { cycleModel(); }
        );
        modelButton->setSize(widgetWidth, widgetHeight);
        scrollBox->addChild(modelButton);
        
        // ===== Button 5: Personality =====
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
        
        // ===== Dropdown 1: Font Selection =====
        std::vector<std::string> availableFonts = getFontList();
        std::string currentFont = "Consolas";
        if (pendingConfig.contains("ui") && pendingConfig["ui"].is_object() &&
            pendingConfig["ui"].contains("font_name")) {
            currentFont = pendingConfig["ui"]["font_name"].get<std::string>();
        }
        
        // Find current font index
        int fontIndex = 0;
        for (size_t i = 0; i < availableFonts.size(); ++i) {
            if (availableFonts[i] == currentFont) {
                fontIndex = static_cast<int>(i);
                break;
            }
        }
        
        auto fontDropdown = std::make_shared<UIDropdown>(
            "Font:",
            availableFonts,
            fontIndex,
            [this](int idx, const std::string& fontName) {
                if (!pendingConfig.contains("ui") || !pendingConfig["ui"].is_object()) {
                    pendingConfig["ui"] = nlohmann::json::object();
                }
                pendingConfig["ui"]["font_name"] = fontName;
                hasChanges = true;
                LOG_DEBUG("UISettingsMenu", "Font selected: " + fontName);
            }
        );
        fontDropdown->setSize(widgetWidth, widgetHeight);
        scrollBox->addChild(fontDropdown);
        
        // ===== Slider 1: Temperature =====
        float temperature = 0.0f;
        if (pendingConfig.contains("whisper") && pendingConfig["whisper"].is_object() &&
            pendingConfig["whisper"].contains("temperature")) {
            temperature = pendingConfig["whisper"]["temperature"].get<float>();
        }
        auto tempSlider = std::make_shared<UISlider>(
            "Temperature:",
            0.0f, 1.0f, temperature,
            [this](float value) {
                if (!pendingConfig.contains("whisper") || !pendingConfig["whisper"].is_object()) {
                    pendingConfig["whisper"] = nlohmann::json::object();
                }
                pendingConfig["whisper"]["temperature"] = value;
                hasChanges = true;
            }
        );
        tempSlider->setSize(widgetWidth, widgetHeight);
        scrollBox->addChild(tempSlider);
        
        // ===== Slider 2: Beam Size =====
        int beamSize = 5;
        if (pendingConfig.contains("whisper") && pendingConfig["whisper"].is_object() &&
            pendingConfig["whisper"].contains("beam_size")) {
            beamSize = pendingConfig["whisper"]["beam_size"].get<int>();
        }
        auto beamSlider = std::make_shared<UISlider>(
            "Beam Size:",
            1.0f, 10.0f, static_cast<float>(beamSize),
            [this](float value) {
                if (!pendingConfig.contains("whisper") || !pendingConfig["whisper"].is_object()) {
                    pendingConfig["whisper"] = nlohmann::json::object();
                }
                pendingConfig["whisper"]["beam_size"] = static_cast<int>(value);
                hasChanges = true;
            }
        );
        beamSlider->setSize(widgetWidth, widgetHeight);
        scrollBox->addChild(beamSlider);
        
        // ===== Toggle 1: Suppress Blank =====
        bool suppressBlank = true;
        if (pendingConfig.contains("whisper") && pendingConfig["whisper"].is_object() &&
            pendingConfig["whisper"].contains("suppress_blank")) {
            suppressBlank = pendingConfig["whisper"]["suppress_blank"].get<bool>();
        }
        auto suppressToggle = std::make_shared<UIToggle>(
            "Suppress Blank:",
            suppressBlank,
            [this](bool value) {
                if (!pendingConfig.contains("whisper") || !pendingConfig["whisper"].is_object()) {
                    pendingConfig["whisper"] = nlohmann::json::object();
                }
                pendingConfig["whisper"]["suppress_blank"] = value;
                hasChanges = true;
            }
        );
        suppressToggle->setSize(widgetWidth, widgetHeight);
        scrollBox->addChild(suppressToggle);
        
        // ===== Toggle 2: Custom Personality =====
        bool useCustom = false;
        if (pendingConfig.contains("personality") && pendingConfig["personality"].is_object() &&
            pendingConfig["personality"].contains("use_custom_prompt")) {
            useCustom = pendingConfig["personality"]["use_custom_prompt"].get<bool>();
        }
        auto customToggle = std::make_shared<UIToggle>(
            "Custom Personality:",
            useCustom,
            [this](bool value) {
                if (!pendingConfig.contains("personality") || !pendingConfig["personality"].is_object()) {
                    pendingConfig["personality"] = nlohmann::json::object();
                }
                pendingConfig["personality"]["use_custom_prompt"] = value;
                hasChanges = true;
            }
        );
        customToggle->setSize(widgetWidth, widgetHeight);
        scrollBox->addChild(customToggle);
        
        // Auto-layout all children in the scrollbox
        scrollBox->setChildSpacing(5.0f);
        scrollBox->autoLayoutChildren(10.0f);
        
        // Create Save & Close button (outside scrollbox)
        saveButton = std::make_shared<UIButton>("Save & Close", [this]() {
            doSaveAndClose();
        });
        
        // Create Cancel button (outside scrollbox)
        cancelButton = std::make_shared<UIButton>("Cancel", [this]() {
            doCancel();
        });
        
        LOG_DEBUG("UISettingsMenu", "Widgets created successfully - using container system");
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

void UISettingsMenu::update(const InputState& input, float dt) {
    UIPanel::update(input, dt);
    
    if (!isVisible()) return;
    
    // Check refresh flag BEFORE updating widgets
    bool shouldRefresh = needsWidgetRefresh;
    if (shouldRefresh) {
        needsWidgetRefresh = false;
        LOG_DEBUG("UISettingsMenu", "Refreshing widgets at START of frame (before update)");
        createWidgets();
        return;  // Skip this frame's widget updates - fresh widgets will update next frame
    }
    
    // Update scroll box position to follow panel
    scrollBox->setPosition(position.x + 10, position.y + 40);
    scrollBox->setSize(size.x - 20, size.y - 50);
    
    // Scrollbox handles all child updates internally
    scrollBox->update(input, dt);
    
    // Update Save & Close button (outside scrollbox)
    if (saveButton) {
        float yPos = position.y + size.y - 50;
        saveButton->setPosition(position.x + 10, yPos);
        saveButton->setSize((size.x - 30) / 2, 40);
        saveButton->update(input, dt);
    }
    
    // Update Cancel button (outside scrollbox)
    if (cancelButton) {
        float yPos = position.y + size.y - 50;
        cancelButton->setPosition(position.x + 10 + (size.x - 30) / 2 + 10, yPos);
        cancelButton->setSize((size.x - 30) / 2, 40);
        cancelButton->update(input, dt);
    }
}

void UISettingsMenu::drawOverlay(OverlayRenderer& renderer)
{
    if (!isVisible()) return;
    
    UIPanel::drawOverlay(renderer);
    
    // Draw scroll box with all children (scrollbox handles culling and scroll offset)
    scrollBox->drawOverlay(renderer, position);
    
    // Draw Save & Close button
    if (saveButton) {
        Vec2 btnPos = saveButton->getPosition();
        Vec2 btnSize = saveButton->getSize();
        
        uint32_t btnColor = 0xFF2A4A2A;
        uint32_t borderColor = 0xFF00FF00;
        
        renderer.drawRect(btnPos, btnSize, btnColor);
        renderer.drawRect(btnPos, {btnSize.x, 2}, borderColor);
        renderer.drawRect(btnPos, {2, btnSize.y}, borderColor);
        renderer.drawRect({btnPos.x, btnPos.y + btnSize.y - 2}, {btnSize.x, 2}, borderColor);
        renderer.drawRect({btnPos.x + btnSize.x - 2, btnPos.y}, {2, btnSize.y}, borderColor);
        
        float textY = btnPos.y + (btnSize.y / 2.0f) - 8;
        renderer.drawText({btnPos.x + 10, textY}, "Save & Close", 0xFFFFFFFF);
    }
    
    // Draw Cancel button
    if (cancelButton) {
        Vec2 btnPos = cancelButton->getPosition();
        Vec2 btnSize = cancelButton->getSize();
        
        uint32_t btnColor = 0xFF4A2A2A;
        uint32_t borderColor = 0xFFFF0000;
        
        renderer.drawRect(btnPos, btnSize, btnColor);
        renderer.drawRect(btnPos, {btnSize.x, 2}, borderColor);
        renderer.drawRect(btnPos, {2, btnSize.y}, borderColor);
        renderer.drawRect({btnPos.x, btnPos.y + btnSize.y - 2}, {btnSize.x, 2}, borderColor);
        renderer.drawRect({btnPos.x + btnSize.x - 2, btnPos.y}, {2, btnSize.y}, borderColor);
        
        float textY = btnPos.y + (btnSize.y / 2.0f) - 8;
        renderer.drawText({btnPos.x + 10, textY}, "Cancel", 0xFFFFFFFF);
    }
    
    // Unsaved changes indicator
    if (hasChanges) {
        renderer.drawText({position.x + size.x - 150, position.y + 8}, "* Unsaved", 0xFFFFFF00);
    }
}

