#pragma once
#include "widget.hpp"
#include <functional>
#include <string>
#include <vector>

class UIDropdown : public Widget {
public:
    UIDropdown(const std::string& label, const std::vector<std::string>& items, 
               int initialIndex, std::function<void(int, const std::string&)> onChange);

    void update(const InputState& input, float dt) override;
    void draw(class UIRenderer& renderer) override;
    
    void setSelectedIndex(int idx);
    int getSelectedIndex() const { return selectedIndex; }
    std::string getSelectedItem() const;
    std::string getLabel() const { return label; }
    
    // Set maximum visible items before scrolling (default: 8)
    void setMaxVisibleItems(int max) { maxVisibleItems = std::max(1, max); }
    
    // For overlay rendering
    void drawOverlay(class OverlayRenderer& renderer, const Vec2& panelPos);
    
    // Separate method to draw expanded list on top of everything
    void drawExpandedList(class OverlayRenderer& renderer, const Vec2& panelPos);
    
    // Check if dropdown is currently expanded
    bool isExpanded() const { return expanded; }

private:
    std::string label;
    std::vector<std::string> options;
    int selectedIndex;
    std::function<void(int, const std::string&)> callback;
    
    bool expanded = false;
    Vec2 dropdownPos{0, 0};
    Vec2 dropdownSize{0, 0};
    
    // Scrolling support
    int maxVisibleItems = 8;  // Maximum items to show before scrolling
    int scrollOffset = 0;     // Current scroll position (in items)
};

