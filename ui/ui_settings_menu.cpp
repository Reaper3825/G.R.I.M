#include "ui_settings_menu.hpp"
#include "overlay_renderer.hpp"
#include "logger.hpp"
#include "input_parser.hpp"
#include <fstream>
#include <functional>  // ? ADD: For std::bind
#include <filesystem>  // ? NEW: For directory scanning

UISettingsMenu::UISettingsMenu()
    : UIPanel("Settings", true), hasChanges(false), isRefreshing(false), needsWidgetRefresh(false)
{
    position = { 200, 200 };
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
                    {"engine", "coqui"}
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
        f << config.dump(4);
        LOG_DEBUG("UISettingsMenu", "Saved ai_config.json");
    } catch (...) {
        LOG_ERROR("UISettingsMenu", "Failed to save ai_config.json");
    }
}

void UISettingsMenu::applyChanges() {
    if (!hasChanges) {
        LOG_DEBUG("UISettingsMenu", "No changes to apply");
        return;
    }
    
    config = pendingConfig;
    saveConfig();
    hasChanges = false;
    
    LOG_DEBUG("UISettingsMenu", "Settings applied and saved");
}

// ? Action handlers - these can safely be called from callbacks
void UISettingsMenu::cycleBackend() {
    try {
        LOG_DEBUG("UISettingsMenu", "cycleBackend() called");
        std::string current = pendingConfig.value("backend", "auto");
        LOG_DEBUG("UISettingsMenu", "Current backend: " + current);
        
        // Valid backends: auto, ollama, localai, openai
        std::string next = (current == "auto") ? "ollama" :
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
            nextPrompt = "You are GRIM. Be concise but warm and friendly. Keep responses brief (1-2 sentences) while maintaining a positive, helpful tone. Add a touch of personality without being wordy.";
        } else if (current == "Short & Sweet") {
            nextLabel = "Sarcastic";
            nextPrompt = "You are GRIM, a sarcastic but helpful AI assistant. Use dry humor and witty remarks while still providing accurate assistance. Don't be mean, just playfully sarcastic.";
        } else if (current == "Sarcastic") {
            nextLabel = "Military";
            nextPrompt = "You are GRIM, a military-grade tactical AI. Use precise, direct language. Address user as 'Commander'. Keep responses brief and mission-focused. Acknowledge orders with 'Roger' or 'Affirmative'.";
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
    LOG_DEBUG("UISettingsMenu", "createWidgets() START");
    
    try {
        LOG_DEBUG("UISettingsMenu", "Clearing existing widgets");
        buttons.clear();
        buttonLabels.clear();
        sliders.clear();
        toggles.clear();
        dropdowns.clear();  // ? NEW: Clear dropdowns
        LOG_DEBUG("UISettingsMenu", "Widgets cleared successfully");

        float yOffset = 10.0f;  // Start offset within scroll box
        float widgetHeight = 45.0f;
        float contentX = 10.0f;  // X offset within scroll box
        float widgetWidth = scrollBox->getSize().x - 30;  // Leave room for scrollbar
        
        LOG_DEBUG("UISettingsMenu", "Creating buttons WITH callbacks");
        
        // Button 1: Backend
        LOG_DEBUG("UISettingsMenu", "Creating Button 1: Backend");
        std::string backend = pendingConfig.value("backend", "auto");
        std::string label1 = "Backend: " + backend;
        
        buttons.push_back(std::make_shared<UIButton>(label1, [this]() {
            cycleBackend();
        }));
        buttons.back()->setPosition(contentX, yOffset);
        buttons.back()->setSize(widgetWidth, widgetHeight);
        buttonLabels.push_back(label1);
        yOffset += widgetHeight + 5;
        LOG_DEBUG("UISettingsMenu", "Button 1 created successfully");
        
        // Button 2: Voice Engine (nested in voice.engine)
        LOG_DEBUG("UISettingsMenu", "Creating Button 2: Voice");
        std::string voiceEngine = "coqui";
        if (pendingConfig.contains("voice") && pendingConfig["voice"].is_object() &&
            pendingConfig["voice"].contains("engine")) {
            voiceEngine = pendingConfig["voice"]["engine"].get<std::string>();
        }
        std::string label2 = "Voice: " + voiceEngine;
        buttons.push_back(std::make_shared<UIButton>(label2, [this]() {
            cycleVoice();
        }));
        buttons.back()->setPosition(contentX, yOffset);
        buttons.back()->setSize(widgetWidth, widgetHeight);
        buttonLabels.push_back(label2);
        yOffset += widgetHeight + 5;
        LOG_DEBUG("UISettingsMenu", "Button 2 created successfully");
        
        // ? NEW: Dropdown for Speaker Embedding Selection (only show if Coqui is selected)
        if (voiceEngine == "coqui") {
            LOG_DEBUG("UISettingsMenu", "Creating Dropdown: Speaker Embedding");
            
            // Get available embeddings
            std::vector<std::string> embeddings = getSpeakerEmbeddings();
            
            // Get current speaker from config
            std::string currentSpeaker = "default";
            if (pendingConfig.contains("voice") && pendingConfig["voice"].is_object() &&
                pendingConfig["voice"].contains("speaker")) {
                currentSpeaker = pendingConfig["voice"]["speaker"].get<std::string>();
            }
            
            // Find index of current speaker
            int selectedIndex = 0;
            for (size_t i = 0; i < embeddings.size(); ++i) {
                if (embeddings[i] == currentSpeaker) {
                    selectedIndex = static_cast<int>(i);
                    break;
                }
            }
            
            dropdowns.push_back(std::make_shared<UIDropdown>(
                "Speaker:",
                embeddings,
                selectedIndex,
                [this](int index, const std::string& selected) {
                    if (selected == "(no custom voices)") return;  // Ignore helper text
                    
                    if (!pendingConfig.contains("voice") || !pendingConfig["voice"].is_object()) {
                        pendingConfig["voice"] = nlohmann::json::object();
                    }
                    pendingConfig["voice"]["speaker"] = selected;
                    hasChanges = true;
                    LOG_DEBUG("UISettingsMenu", "Speaker changed to: " + selected);
                }
            ));
            
            dropdowns.back()->setPosition(contentX, yOffset);
            dropdowns.back()->setSize(widgetWidth, widgetHeight);
            yOffset += widgetHeight + 5;
            LOG_DEBUG("UISettingsMenu", "Dropdown created successfully");
        }
        
        // Button 3: Whisper Model (nested in whisper.whisper_model)
        LOG_DEBUG("UISettingsMenu", "Creating Button 3: Model");
        std::string model = "ggml-base.en.bin";
        if (pendingConfig.contains("whisper") && pendingConfig["whisper"].is_object() &&
            pendingConfig["whisper"].contains("whisper_model")) {
            model = pendingConfig["whisper"]["whisper_model"].get<std::string>();
        }
        std::string label3 = "Model: " + model;
        buttons.push_back(std::make_shared<UIButton>(label3, [this]() {
            cycleModel();
        }));
        buttons.back()->setPosition(contentX, yOffset);
        buttons.back()->setSize(widgetWidth, widgetHeight);
        buttonLabels.push_back(label3);
        yOffset += widgetHeight + 5;
        LOG_DEBUG("UISettingsMenu", "Button 3 created successfully");
        
        // Button 4: Personality Preset
        LOG_DEBUG("UISettingsMenu", "Creating Button 4: Personality");
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
        std::string label4 = "Personality: " + personalityLabel;
        buttons.push_back(std::make_shared<UIButton>(label4, [this]() {
            cyclePersonality();
        }));
        buttons.back()->setPosition(contentX, yOffset);
        buttons.back()->setSize(widgetWidth, widgetHeight);
        buttonLabels.push_back(label4);
        yOffset += widgetHeight + 5;
        LOG_DEBUG("UISettingsMenu", "Button 4 created successfully");
        
        // Slider 1: Whisper Temperature (nested in whisper.temperature)
        LOG_DEBUG("UISettingsMenu", "Creating Slider 1: Temperature");
        float temperature = 0.0f;
        if (pendingConfig.contains("whisper") && pendingConfig["whisper"].is_object() &&
            pendingConfig["whisper"].contains("temperature")) {
            temperature = pendingConfig["whisper"]["temperature"].get<float>();
        }
        LOG_DEBUG("UISettingsMenu", "Temperature value extracted");
        
        sliders.push_back(std::make_shared<UISlider>(
            "Temperature:",
            0.0f, 1.0f, temperature,
            [this](float value) {
                if (!pendingConfig.contains("whisper") || !pendingConfig["whisper"].is_object()) {
                    pendingConfig["whisper"] = nlohmann::json::object();
                }
                pendingConfig["whisper"]["temperature"] = value;
                hasChanges = true;
                LOG_DEBUG("UISettingsMenu", "Temperature changed to: " + std::to_string(value));
            }
        ));
        LOG_DEBUG("UISettingsMenu", "Slider 1 constructed");
        sliders.back()->setPosition(contentX, yOffset);
        sliders.back()->setSize(widgetWidth, widgetHeight);
        yOffset += widgetHeight + 5;
        LOG_DEBUG("UISettingsMenu", "Slider 1 created successfully");
        
        // Slider 2: Beam Size (nested in whisper.beam_size)
        LOG_DEBUG("UISettingsMenu", "Creating Slider 2: Beam Size");
        int beamSize = 5;
        if (pendingConfig.contains("whisper") && pendingConfig["whisper"].is_object() &&
            pendingConfig["whisper"].contains("beam_size")) {
            beamSize = pendingConfig["whisper"]["beam_size"].get<int>();
        }
        LOG_DEBUG("UISettingsMenu", "Beam size value extracted");
        
        sliders.push_back(std::make_shared<UISlider>(
            "Beam Size:",
            1.0f, 10.0f, static_cast<float>(beamSize),
            [this](float value) {
                if (!pendingConfig.contains("whisper") || !pendingConfig["whisper"].is_object()) {
                    pendingConfig["whisper"] = nlohmann::json::object();
                }
                pendingConfig["whisper"]["beam_size"] = static_cast<int>(value);
                hasChanges = true;
                LOG_DEBUG("UISettingsMenu", "Beam size changed to: " + std::to_string(static_cast<int>(value)));
            }
        ));
        LOG_DEBUG("UISettingsMenu", "Slider 2 constructed");
        sliders.back()->setPosition(contentX, yOffset);
        sliders.back()->setSize(widgetWidth, widgetHeight);
        yOffset += widgetHeight + 5;
        LOG_DEBUG("UISettingsMenu", "Slider 2 created successfully");
        
        // Toggle 1: Suppress Blank (nested in whisper.suppress_blank)
        LOG_DEBUG("UISettingsMenu", "Creating Toggle 1: Suppress Blank");
        bool suppressBlank = true;
        if (pendingConfig.contains("whisper") && pendingConfig["whisper"].is_object() &&
            pendingConfig["whisper"].contains("suppress_blank")) {
            suppressBlank = pendingConfig["whisper"]["suppress_blank"].get<bool>();
        }
        LOG_DEBUG("UISettingsMenu", "Suppress blank value extracted");
        
        LOG_DEBUG("UISettingsMenu", "About to construct UIToggle 1");
        toggles.push_back(std::make_shared<UIToggle>(
            "Suppress Blank:",
            suppressBlank,
            [this](bool value) {
                if (!pendingConfig.contains("whisper") || !pendingConfig["whisper"].is_object()) {
                    pendingConfig["whisper"] = nlohmann::json::object();
                }
                pendingConfig["whisper"]["suppress_blank"] = value;
                hasChanges = true;
                LOG_DEBUG("UISettingsMenu", "Suppress blank changed to: " + std::string(value ? "true" : "false"));
            }
        ));
        LOG_DEBUG("UISettingsMenu", "UIToggle 1 constructed");
        toggles.back()->setPosition(contentX, yOffset);
        LOG_DEBUG("UISettingsMenu", "Toggle 1 position set");
        toggles.back()->setSize(widgetWidth, widgetHeight);
        yOffset += widgetHeight + 5;
        LOG_DEBUG("UISettingsMenu", "Toggle 1 created successfully");
        
        // Toggle 2: Use Custom Personality
        LOG_DEBUG("UISettingsMenu", "Creating Toggle 2: Use Custom Personality");
        bool useCustomPrompt = false;
        if (pendingConfig.contains("personality") && pendingConfig["personality"].is_object() &&
            pendingConfig["personality"].contains("use_custom_prompt")) {
            useCustomPrompt = pendingConfig["personality"]["use_custom_prompt"].get<bool>();
        }
        LOG_DEBUG("UISettingsMenu", "Use custom personality value extracted");
        
        toggles.push_back(std::make_shared<UIToggle>(
            "Custom Personality:",
            useCustomPrompt,
            [this](bool value) {
                if (!pendingConfig.contains("personality") || !pendingConfig["personality"].is_object()) {
                    pendingConfig["personality"] = nlohmann::json::object();
                }
                pendingConfig["personality"]["use_custom_prompt"] = value;
                hasChanges = true;
                LOG_DEBUG("UISettingsMenu", "Use custom personality changed to: " + std::string(value ? "true" : "false"));
            }
        ));
        toggles.back()->setPosition(contentX, yOffset);
        toggles.back()->setSize(widgetWidth, widgetHeight);
        yOffset += widgetHeight + 20;
        LOG_DEBUG("UISettingsMenu", "Toggle 2 created successfully");
        
        // Save & Close Button
        LOG_DEBUG("UISettingsMenu", "Creating Button 5: Save & Close");
        buttons.push_back(std::make_shared<UIButton>("Save & Close", [this]() {
            doSaveAndClose();
        }));
        buttons.back()->setPosition(contentX, yOffset);
        buttons.back()->setSize((widgetWidth - 10) / 2, 40);
        buttonLabels.push_back("Save & Close");
        LOG_DEBUG("UISettingsMenu", "Button 5 created successfully");
        
        // Cancel Button
        LOG_DEBUG("UISettingsMenu", "Creating Button 6: Cancel");
        buttons.push_back(std::make_shared<UIButton>("Cancel", [this]() {
            doCancel();
        }));
        buttons.back()->setPosition(contentX + (widgetWidth - 10) / 2 + 10, yOffset);
        buttons.back()->setSize((widgetWidth - 10) / 2, 40);
        buttonLabels.push_back("Cancel");
        LOG_DEBUG("UISettingsMenu", "Button 6 created successfully");
        
        yOffset += 40 + 10;  // Add final padding
        
        // Set content height for scrollbox
        scrollBox->setContentHeight(yOffset);
        LOG_DEBUG("UISettingsMenu", "Scroll box content height set to: " + std::to_string(yOffset));
        
        LOG_DEBUG("UISettingsMenu", "Widgets created successfully WITH callbacks");
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
    
    // ? FIX: Make COPIES of widget vectors to prevent iterator invalidation
    auto buttonsCopy = buttons;
    auto slidersCopy = sliders;
    auto togglesCopy = toggles;
    auto dropdownsCopy = dropdowns;  // ? NEW: Copy dropdowns too
    
    // ? Check refresh flag BEFORE updating widgets
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
    
    // Update scroll box (handles scrollbar interaction)
    scrollBox->update(input, dt);
    
    // Get scroll offset
    float scrollOffset = scrollBox->getScrollOffset();
    Vec2 scrollBoxPos = scrollBox->getPosition();
    
    // Update widget positions with scroll offset applied
    float yOffset = 10.0f;  // Start offset within scroll box
    float widgetHeight = 45.0f;
    float contentX = scrollBoxPos.x + 10.0f;
    
    // Update cycle buttons (first 4) using COPY
    for (size_t i = 0; i < 4 && i < buttonsCopy.size(); ++i) {
        buttonsCopy[i]->setPosition(contentX, scrollBoxPos.y + yOffset - scrollOffset);
        buttonsCopy[i]->update(input, dt);
        yOffset += widgetHeight + 5;
        
        // ? NEW: If this is the voice button (index 1) and Coqui is selected, update dropdown
        if (i == 1 && !dropdownsCopy.empty()) {
            std::string voiceEngine = "coqui";
            if (pendingConfig.contains("voice") && pendingConfig["voice"].is_object() &&
                pendingConfig["voice"].contains("engine")) {
                voiceEngine = pendingConfig["voice"]["engine"].get<std::string>();
            }
            
            if (voiceEngine == "coqui") {
                dropdownsCopy[0]->setPosition(contentX, scrollBoxPos.y + yOffset - scrollOffset);
                dropdownsCopy[0]->update(input, dt);
                yOffset += widgetHeight + 5;
            }
        }
    }
    
    // Update sliders using COPY
    for (auto& slider : slidersCopy) {
        slider->setPosition(contentX, scrollBoxPos.y + yOffset - scrollOffset);
        slider->update(input, dt);
        yOffset += widgetHeight + 5;
    }
    
    // Update toggles using COPY
    for (auto& toggle : togglesCopy) {
        toggle->setPosition(contentX, scrollBoxPos.y + yOffset - scrollOffset);
        toggle->update(input, dt);
        yOffset += widgetHeight + 5;
    }
    
    yOffset += 15;
    
    // Update Save/Cancel buttons (last 2) using COPY
    if (buttonsCopy.size() >= 6) {
        float widgetWidth = scrollBox->getSize().x - 30;
        
        buttonsCopy[4]->setPosition(contentX, scrollBoxPos.y + yOffset - scrollOffset);
        buttonsCopy[4]->update(input, dt);
        
        buttonsCopy[5]->setPosition(contentX + (widgetWidth - 10) / 2 + 10, scrollBoxPos.y + yOffset - scrollOffset);
        buttonsCopy[5]->update(input, dt);
    }
}
void UISettingsMenu::drawOverlay(OverlayRenderer& renderer)
{
    if (!isVisible()) return;
    
    UIPanel::drawOverlay(renderer);
    
    // Draw scroll box background and scrollbar
    scrollBox->drawOverlay(renderer, position);
    
    // Get scroll offset for rendering
    float scrollOffset = scrollBox->getScrollOffset();
    Vec2 scrollBoxPos = scrollBox->getPosition();
    Vec2 scrollBoxSize = scrollBox->getSize();
    
    // Note: In a full implementation, we would use scissor/clipping here
    // For now, we just draw everything and let things outside be visible
    // (A proper implementation would require OpenGL/DirectX scissor test)
    
    // ? NEW: Draw dropdowns (if voice engine is Coqui)
    std::string voiceEngine = "coqui";
    if (pendingConfig.contains("voice") && pendingConfig["voice"].is_object() &&
        pendingConfig["voice"].contains("engine")) {
        voiceEngine = pendingConfig["voice"]["engine"].get<std::string>();
    }
    
    if (voiceEngine == "coqui" && !dropdowns.empty()) {
        for (auto& dropdown : dropdowns) {
            Vec2 dropdownPos = dropdown->getPosition();
            
            // Only draw if within visible area
            if (dropdownPos.y + dropdown->getSize().y >= scrollBoxPos.y &&
                dropdownPos.y <= scrollBoxPos.y + scrollBoxSize.y) {
                dropdown->drawOverlay(renderer, position);
            }
        }
    }
    
    // Draw sliders
    for (auto& slider : sliders) {
        Vec2 sliderPos = slider->getPosition();
        
        // Only draw if within visible area (simple culling)
        if (sliderPos.y + slider->getSize().y >= scrollBoxPos.y &&
            sliderPos.y <= scrollBoxPos.y + scrollBoxSize.y) {
            slider->drawOverlay(renderer, position);
        }
    }
    
    // Draw toggles
    for (auto& toggle : toggles) {
        Vec2 togglePos = toggle->getPosition();
        
        // Only draw if within visible area
        if (togglePos.y + toggle->getSize().y >= scrollBoxPos.y &&
            togglePos.y <= scrollBoxPos.y + scrollBoxSize.y) {
            toggle->drawOverlay(renderer, position);
        }
    }
    
    // Draw all buttons with color coding
    for (size_t i = 0; i < buttons.size() && i < buttonLabels.size(); ++i) {
        Vec2 btnPos = buttons[i]->getPosition();
        Vec2 btnSize = buttons[i]->getSize();
        
        // Only draw if within visible area
        if (btnPos.y + btnSize.y < scrollBoxPos.y ||
            btnPos.y > scrollBoxPos.y + scrollBoxSize.y) {
            continue;  // Skip drawing if outside visible area
        }
        
        uint32_t btnColor = 0xFF303030;
        uint32_t borderColor = 0xFF00FFFF;
        
        if (i < 4) {  // Cycle buttons (Backend, Voice, Model, Personality)
            btnColor = 0xFF202030;
            borderColor = 0xFF00FFFF;
        } else if (i == 4) {  // Save
            btnColor = 0xFF2A4A2A;
            borderColor = 0xFF00FF00;
        } else if (i == 5) {  // Cancel
            btnColor = 0xFF4A2A2A;
            borderColor = 0xFFFF0000;
        }
        
        renderer.drawRect(btnPos, btnSize, btnColor);
        renderer.drawRect(btnPos, {btnSize.x, 2}, borderColor);
        renderer.drawRect(btnPos, {2, btnSize.y}, borderColor);
        renderer.drawRect({btnPos.x, btnPos.y + btnSize.y - 2}, {btnSize.x, 2}, borderColor);
        renderer.drawRect({btnPos.x + btnSize.x - 2, btnPos.y}, {2, btnSize.y}, borderColor);
        
        float textY = btnPos.y + (btnSize.y / 2.0f) - 8;
        renderer.drawText({btnPos.x + 10, textY}, buttonLabels[i], 0xFFFFFFFF);
    }
    
    // Unsaved changes indicator
    if (hasChanges) {
        renderer.drawText({position.x + size.x - 150, position.y + 8}, "* Unsaved", 0xFFFFFF00);
    }
}
