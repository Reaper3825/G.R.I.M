#include "ui_dropdown.hpp"
#include "overlay_renderer.hpp"
#include "input_parser.hpp"
#include "helpers/mouse.hpp"
#include "logger.hpp"  // ? ADD: For debugging
#include <algorithm>

UIDropdown::UIDropdown(const std::string& lbl, const std::vector<std::string>& items,
                       int initialIndex, std::function<void(int, const std::string&)> onChange)
    : label(lbl), options(items), selectedIndex(initialIndex), callback(std::move(onChange))
{
    if (selectedIndex < 0 || selectedIndex >= static_cast<int>(options.size())) {
        selectedIndex = 0;
    }
    LOG_DEBUG("UIDropdown", "Constructed: " + label);  // ? ADD: Debug output
}

void UIDropdown::setSelectedIndex(int idx) {
    if (idx >= 0 && idx < static_cast<int>(options.size())) {
        selectedIndex = idx;
    }
}

std::string UIDropdown::getSelectedItem() const {
    if (selectedIndex >= 0 && selectedIndex < static_cast<int>(options.size())) {
        return options[selectedIndex];
    }
    return "";
}

void UIDropdown::update(const InputState& input, float dt) {
    Vec2 m = input.mousePos;
    
    dropdownPos = {position.x + 150, position.y + 5};
    dropdownSize = {size.x - 160, 30};
    
    bool overDropdown = (m.x >= dropdownPos.x && m.x <= dropdownPos.x + dropdownSize.x &&
                        m.y >= dropdownPos.y && m.y <= dropdownPos.y + dropdownSize.y);
    
    bool leftPressed = Mouse::wasPressed(MouseButton::Left);
    
    if (leftPressed) {
        if (overDropdown) {
            // Toggle dropdown
            expanded = !expanded;
        } else if (expanded) {
            // Check if clicked on an option
            for (size_t i = 0; i < options.size(); ++i) {
                Vec2 optPos = {dropdownPos.x, dropdownPos.y + dropdownSize.y + i * 25};
                Vec2 optSize = {dropdownSize.x, 25};
                
                bool overOption = (m.x >= optPos.x && m.x <= optPos.x + optSize.x &&
                                  m.y >= optPos.y && m.y <= optPos.y + optSize.y);
                
                if (overOption) {
                    selectedIndex = static_cast<int>(i);
                    if (callback) callback(selectedIndex, options[selectedIndex]);
                    expanded = false;
                    break;
                }
            }
            
            // Clicked outside - close dropdown
            if (expanded) {
                expanded = false;
            }
        }
    }
}

void UIDropdown::draw(UIRenderer& renderer) {
    // Not used - we use drawOverlay instead
}

void UIDropdown::drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos) {
    // Draw label
    renderer.drawText({position.x, position.y + 15}, label, 0xFFFFFFFF);
    
    // Recalculate dropdown position based on current position (for scrolling)
    Vec2 currentDropdownPos = {position.x + 150, position.y + 5};
    Vec2 currentDropdownSize = {size.x - 160, 30};
    
    // Draw dropdown box
    renderer.drawRect(currentDropdownPos, currentDropdownSize, 0xFF303030);
    renderer.drawRect(currentDropdownPos, {currentDropdownSize.x, 2}, 0xFF00FFFF);
    renderer.drawRect(currentDropdownPos, {2, currentDropdownSize.y}, 0xFF00FFFF);
    renderer.drawRect({currentDropdownPos.x, currentDropdownPos.y + currentDropdownSize.y - 2}, {currentDropdownSize.x, 2}, 0xFF00FFFF);
    renderer.drawRect({currentDropdownPos.x + currentDropdownSize.x - 2, currentDropdownPos.y}, {2, currentDropdownSize.y}, 0xFF00FFFF);
    
    // Draw selected item
    std::string selected = getSelectedItem();
    renderer.drawText({currentDropdownPos.x + 8, currentDropdownPos.y + 8}, selected, 0xFFFFFFFF);
    
    // Draw arrow indicator
    std::string arrow = expanded ? "?" : "?";
    renderer.drawText({currentDropdownPos.x + currentDropdownSize.x - 20, currentDropdownPos.y + 8}, arrow, 0xFF00FFFF);
    
    // Draw expanded options
    if (expanded) {
        for (size_t i = 0; i < options.size(); ++i) {
            Vec2 optPos = {currentDropdownPos.x, currentDropdownPos.y + currentDropdownSize.y + i * 25};
            Vec2 optSize = {currentDropdownSize.x, 25};
            
            uint32_t bgColor = (i == static_cast<size_t>(selectedIndex)) ? 0xFF404040 : 0xFF202020;
            renderer.drawRect(optPos, optSize, bgColor);
            renderer.drawRect(optPos, {optSize.x, 1}, 0xFF00FFFF);
            
            renderer.drawText({optPos.x + 8, optPos.y + 5}, options[i], 0xFFFFFFFF);
        }
    }
}
