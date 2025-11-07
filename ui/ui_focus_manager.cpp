#include "ui_focus_manager.hpp"
#include "logger.hpp"
#include <sstream>
#include <iomanip>
#include <chrono>

UIFocusManager::UIFocusManager() 
    : focusedWidgetID(0),
      focusedPanelID(0),
      rng(std::chrono::steady_clock::now().time_since_epoch().count()),
      dist(0x1000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL)
{
    LOG_DEBUG("UIFocusManager", "Focus manager initialized");
}

uint64_t UIFocusManager::generateUniqueID() {
    uint64_t id = dist(rng);
    
    std::ostringstream oss;
    oss << "Generated unique ID: 0x" << std::hex << std::uppercase << id;
    LOG_DEBUG("UIFocusManager", oss.str());
    
    return id;
}

void UIFocusManager::setFocusedWidget(uint64_t widgetID, uint64_t panelID) {
    focusedWidgetID = widgetID;
    focusedPanelID = panelID;
    
    std::ostringstream oss;
    oss << "Focus set - Widget: 0x" << std::hex << std::uppercase << widgetID 
        << " Panel: 0x" << panelID;
    LOG_DEBUG("UIFocusManager", oss.str());
}

void UIFocusManager::clearFocus() {
    if (focusedWidgetID != 0 || focusedPanelID != 0) {
        LOG_DEBUG("UIFocusManager", "Focus cleared");
        focusedWidgetID = 0;
        focusedPanelID = 0;
    }
}

bool UIFocusManager::isWidgetFocused(uint64_t widgetID) const {
    return focusedWidgetID == widgetID && widgetID != 0;
}

bool UIFocusManager::isPanelFocused(uint64_t panelID) const {
    return focusedPanelID == panelID && panelID != 0;
}

bool UIFocusManager::hasAnyFocus() const {
    return focusedWidgetID != 0 || focusedPanelID != 0;
}

std::string UIFocusManager::getFocusInfo() const {
    std::ostringstream oss;
    if (hasAnyFocus()) {
        oss << "Focused Widget: 0x" << std::hex << std::uppercase << focusedWidgetID 
            << " Panel: 0x" << focusedPanelID;
    } else {
        oss << "No focus";
    }
    return oss.str();
}
