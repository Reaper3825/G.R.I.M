// ? CORRECTED VERSION - Replace the update() method in ui_settings_menu.cpp

void UISettingsMenu::update(const InputState& input, float dt) {
    UIPanel::update(input, dt);
    
    if (!isVisible()) return;
    
    // Check refresh flag BEFORE updating widgets
    if (needsWidgetRefresh) {
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
    
    // ? FIX: Update widgets directly (no copying)
    float yOffset = 10.0f;
    float widgetHeight = 45.0f;
    float contentX = scrollBoxPos.x + 10.0f;
    
    // Update cycle buttons (first 4)
    for (size_t i = 0; i < 4 && i < buttons.size(); ++i) {
        buttons[i]->setPosition(contentX, scrollBoxPos.y + yOffset - scrollOffset);
        buttons[i]->update(input, dt);
        yOffset += widgetHeight + 5;
        
        // If voice button and Coqui selected, update dropdown
        if (i == 1 && !dropdowns.empty()) {
            std::string voiceEngine = "coqui";
            if (pendingConfig.contains("voice") && pendingConfig["voice"].is_object() &&
                pendingConfig["voice"].contains("engine")) {
                voiceEngine = pendingConfig["voice"]["engine"].get<std::string>();
            }
            
            if (voiceEngine == "coqui") {
                dropdowns[0]->setPosition(contentX, scrollBoxPos.y + yOffset - scrollOffset);
                dropdowns[0]->update(input, dt);
                yOffset += widgetHeight + 5;
            }
        }
    }
    
    // Update sliders
    for (auto& slider : sliders) {
        slider->setPosition(contentX, scrollBoxPos.y + yOffset - scrollOffset);
        slider->update(input, dt);
        yOffset += widgetHeight + 5;
    }
    
    // Update toggles
    for (auto& toggle : toggles) {
        toggle->setPosition(contentX, scrollBoxPos.y + yOffset - scrollOffset);
        toggle->update(input, dt);
        yOffset += widgetHeight + 5;
    }
    
    yOffset += 15;
    
    // Update Save/Cancel buttons (last 2)
    if (buttons.size() >= 6) {
        float widgetWidth = scrollBox->getSize().x - 30;
        
        buttons[4]->setPosition(contentX, scrollBoxPos.y + yOffset - scrollOffset);
        buttons[4]->update(input, dt);
        
        buttons[5]->setPosition(contentX + (widgetWidth - 10) / 2 + 10, scrollBoxPos.y + yOffset - scrollOffset);
        buttons[5]->update(input, dt);
    }
}
