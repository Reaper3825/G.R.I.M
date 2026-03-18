#include "ui_dropdown.hpp"
#include "ui_theme.hpp"
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
    
    // Calculate expanded list bounds for scrolling
    int visibleItems = std::min(maxVisibleItems, static_cast<int>(options.size()));
    float expandedHeight = visibleItems * 25.0f;
    Vec2 expandedPos = {dropdownPos.x, dropdownPos.y + dropdownSize.y};
    Vec2 expandedSize = {dropdownSize.x, expandedHeight};
    
    bool overExpandedList = expanded && 
                           (m.x >= expandedPos.x && m.x <= expandedPos.x + expandedSize.x &&
                            m.y >= expandedPos.y && m.y <= expandedPos.y + expandedSize.y);
    
    // Handle mouse wheel scrolling when over expanded list
    if (expanded && overExpandedList && input.mouseWheelDelta != 0.0f) {
        // Scroll by 1 item per 120 wheel delta (standard Windows scroll)
        int scrollAmount = -static_cast<int>(input.mouseWheelDelta / 120.0f);
        scrollOffset = std::clamp(scrollOffset + scrollAmount, 0, 
                                 static_cast<int>(options.size()) - visibleItems);
    }
    
    if (leftPressed) {
        if (overDropdown) {
            // Toggle dropdown
            expanded = !expanded;
            if (expanded) {
                // Reset scroll to show selected item
                scrollOffset = std::max(0, selectedIndex - maxVisibleItems / 2);
                scrollOffset = std::clamp(scrollOffset, 0, 
                                         static_cast<int>(options.size()) - visibleItems);
            }
        } else if (expanded) {
            if (overExpandedList) {
                // Check if clicked on an option (accounting for scroll)
                float relativeY = m.y - expandedPos.y;
                int clickedItem = scrollOffset + static_cast<int>(relativeY / 25.0f);
                
                if (clickedItem >= 0 && clickedItem < static_cast<int>(options.size())) {
                    selectedIndex = clickedItem;
                    if (callback) callback(selectedIndex, options[selectedIndex]);
                    expanded = false;
                    scrollOffset = 0;
                }
            } else {
                // Clicked outside - close dropdown
                expanded = false;
                scrollOffset = 0;
            }
        }
    }
}

void UIDropdown::draw(UIRenderer& renderer) {
    // Not used - we use drawOverlay instead
}

void UIDropdown::drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos) {
    using namespace UITheme;
    
    // Draw label
    renderer.drawText({position.x, position.y + 15}, label, Colors::TextPrimary);
    
    // Recalculate dropdown position based on current position (for scrolling)
    Vec2 currentDropdownPos = {position.x + 150, position.y + 5};
    Vec2 currentDropdownSize = {size.x - 160, 30};
    
    // Draw dropdown box
    renderer.drawRoundedRect(currentDropdownPos, currentDropdownSize, Colors::WidgetBg, Sizes::WidgetRadius);
    renderer.drawRoundedBorder(currentDropdownPos, currentDropdownSize, Colors::BorderPrimary, Sizes::WidgetRadius);
    
    // Draw selected item
    std::string selected = getSelectedItem();
    renderer.drawText({currentDropdownPos.x + 8, currentDropdownPos.y + 8}, selected, Colors::TextPrimary);
    
    // Draw arrow indicator
    std::string arrow = expanded ? "^" : "v";
    renderer.drawText({currentDropdownPos.x + currentDropdownSize.x - 20, currentDropdownPos.y + 8}, arrow, Colors::TextValue);
    
    // Note: Expanded list is NOT drawn here - it's drawn in drawExpandedList() 
    // which is called after all other widgets to ensure proper z-order
}

void UIDropdown::drawExpandedList(OverlayRenderer& renderer, const Vec2& panelPos) {
    using namespace UITheme;
    if (!expanded) return;
    
    // Recalculate dropdown position
    Vec2 currentDropdownPos = {position.x + 150, position.y + 5};
    Vec2 currentDropdownSize = {size.x - 160, 30};
    
    int visibleItems = std::min(maxVisibleItems, static_cast<int>(options.size()));
    int totalItems = static_cast<int>(options.size());
    bool needsScroll = totalItems > maxVisibleItems;
    
    // Draw scrollable options list
    for (int i = 0; i < visibleItems; ++i) {
        int itemIndex = scrollOffset + i;
        if (itemIndex >= totalItems) break;
        
        Vec2 optPos = {currentDropdownPos.x, currentDropdownPos.y + currentDropdownSize.y + i * 25};
        Vec2 optSize = {currentDropdownSize.x, 25};
        
        uint32_t bgColor = (itemIndex == selectedIndex) ? Colors::WidgetBgHover : Colors::ContentAreaBg;
        renderer.drawRect(optPos, optSize, bgColor);
        renderer.drawRect(optPos, {optSize.x, 1}, Colors::GlassHighlight);
        
        renderer.drawText({optPos.x + 8, optPos.y + 5}, options[itemIndex], Colors::TextPrimary);
    }
    
    // Draw scroll indicators if needed
    if (needsScroll) {
        float scrollbarX = currentDropdownPos.x + currentDropdownSize.x - 12;
        float listY = currentDropdownPos.y + currentDropdownSize.y;
        float listHeight = visibleItems * 25.0f;
        
        // Scrollbar background
        renderer.drawRoundedRect({scrollbarX, listY}, {8, listHeight}, Colors::ContentAreaBg, Sizes::SmallRadius);
        
        // Scrollbar thumb
        float scrollRatio = static_cast<float>(scrollOffset) / (totalItems - visibleItems);
        float thumbHeight = (static_cast<float>(visibleItems) / totalItems) * listHeight;
        float thumbY = listY + scrollRatio * (listHeight - thumbHeight);
        
        renderer.drawRoundedRect({scrollbarX, thumbY}, {8, thumbHeight}, Colors::ScrollThumbDrag, Sizes::SmallRadius);
        
        // Draw scroll arrows/indicators
        if (scrollOffset > 0) {
            // Up arrow indicator at top
            renderer.drawText({scrollbarX - 5, listY + 2}, "^", Colors::TextValue);
        }
        if (scrollOffset < totalItems - visibleItems) {
            // Down arrow indicator at bottom
            renderer.drawText({scrollbarX - 5, listY + listHeight - 15}, "v", Colors::TextValue);
        }
    }
}

