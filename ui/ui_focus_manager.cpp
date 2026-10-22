#include "ui_focus_manager.hpp"
#include "logger.hpp"
#include <sstream>
#include <iomanip>
#include <chrono>

UIFocusManager::UIFocusManager()
    : rng(std::chrono::steady_clock::now().time_since_epoch().count()),
      dist(0x1000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL)
{
    LOG_DEBUG("UIFocusManager", "Focus manager initialized");
}

uint64_t UIFocusManager::generateUniqueID() {
    uint64_t id = dist(rng);

    return id;
}

void UIFocusManager::setFocusedWidget(uint64_t widgetID, uint64_t panelID) {
    GRIM::UI::InputRouter::get().focus = {widgetID, panelID};

    std::ostringstream oss;
    oss << "Focus set - Widget: 0x" << std::hex << std::uppercase << widgetID
        << " Panel: 0x" << panelID;
    LOG_DEBUG("UIFocusManager", oss.str());
}

void UIFocusManager::clearFocus() {
    if (getFocusedWidget() != 0 || getFocusedPanel() != 0) {
        LOG_DEBUG("UIFocusManager", "Focus cleared");
        GRIM::UI::InputRouter::get().focus = {};
    }
}

bool UIFocusManager::isWidgetFocused(uint64_t widgetID) const {
    return getFocusedWidget() == widgetID && widgetID != 0;
}

bool UIFocusManager::isPanelFocused(uint64_t panelID) const {
    return getFocusedPanel() == panelID && panelID != 0;
}

bool UIFocusManager::hasAnyFocus() const {
    return getFocusedWidget() != 0 || getFocusedPanel() != 0;
}

std::string UIFocusManager::getFocusInfo() const {
    std::ostringstream oss;
    if (hasAnyFocus()) {
        oss << "Focused Widget: 0x" << std::hex << std::uppercase << getFocusedWidget()
            << " Panel: 0x" << getFocusedPanel();
    } else {
        oss << "No focus";
    }
    return oss.str();
}
