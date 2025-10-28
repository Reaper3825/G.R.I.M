// Simplified update method - scrollbox handles children
void UISettingsMenu::update(const InputState& input, float dt) {
    UIPanel::update(input, dt);
    
    if (!isVisible()) return;
    
    // Check refresh flag
    if (needsWidgetRefresh) {
        needsWidgetRefresh = false;
        createWidgets();
        return;
    }
    
    // Update scrollbox position and size
    scrollBox->setPosition(position.x + 10, position.y + 40);
    scrollBox->setSize(size.x - 20, size.y - 50);
    
    // Scrollbox updates all its children
    scrollBox->update(input, dt);
}

// Simplified draw - scrollbox handles children  
void UISettingsMenu::drawOverlay(OverlayRenderer& renderer) {
    if (!isVisible()) return;
    
    // Draw panel background and title
    UIPanel::drawOverlay(renderer);
    
    // Scrollbox draws itself and all children
    scrollBox->drawOverlay(renderer, position);
    
    // Unsaved changes indicator
    if (hasChanges) {
        renderer.drawText({position.x + size.x - 150, position.y + 8}, "* Unsaved", 0xFFFFFF00);
    }
}
