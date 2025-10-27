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
    
    // For overlay rendering
    void drawOverlay(class OverlayRenderer& renderer, const Vec2& panelPos);

private:
    std::string label;
    std::vector<std::string> options;
    int selectedIndex;
    std::function<void(int, const std::string&)> callback;
    
    bool expanded = false;
    Vec2 dropdownPos{0, 0};
    Vec2 dropdownSize{0, 0};
};
