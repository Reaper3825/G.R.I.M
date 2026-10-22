#pragma once
#include <cstdint>
#include <string>
#include <random>
#include <unordered_map>
#include "input/ui_input_router.hpp"

class Widget;
class UIPanel;

/**
 * Focus Manager - Tracks which UI element currently has input focus
 * Uses unique hex IDs to identify panels and widgets without name collisions
 */
class UIFocusManager {
public:
    static UIFocusManager& getInstance() {
        static UIFocusManager instance;
        return instance;
    }

    // Generate unique hex ID for a widget/panel
    uint64_t generateUniqueID();

    // Focus management
    void setFocusedWidget(uint64_t widgetID, uint64_t panelID);
    void clearFocus();
    
    // Query focus state
    bool isWidgetFocused(uint64_t widgetID) const;
    bool isPanelFocused(uint64_t panelID) const;
    bool hasAnyFocus() const;
    
    uint64_t getFocusedWidget() const { return GRIM::UI::InputRouter::get().focus.widget; }
    uint64_t getFocusedPanel() const { return GRIM::UI::InputRouter::get().focus.panel; }
    
    // Debug
    std::string getFocusInfo() const;

private:
    UIFocusManager();
    ~UIFocusManager() = default;
    
    // Prevent copying
    UIFocusManager(const UIFocusManager&) = delete;
    UIFocusManager& operator=(const UIFocusManager&) = delete;

    
    // Random number generator for unique IDs
    std::mt19937_64 rng;
    std::uniform_int_distribution<uint64_t> dist;
};
