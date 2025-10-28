# UI Polish Implementation Guide

## Files Created

1. **`ui/ui_theme.hpp`** - Centralized theme constants
2. **`ui/ui_draw_helpers.hpp`** - Helper functions for consistent drawing

## Implementation Steps

### Step 1: Add Theme Headers to Settings Menu

```cpp
// In ui_settings_menu.cpp, add at top:
#include "ui_theme.hpp"
#include "ui_draw_helpers.hpp"

using namespace UITheme;
using namespace UIDrawHelpers;
```

### Step 2: Update createWidgets() Spacing

Replace current spacing with theme constants:

```cpp
void UISettingsMenu::createWidgets() {
    // ...
    
    float yOffset = Spacing::PaddingY;  // Start with proper padding
    float widgetHeight = Sizes::WidgetHeight;  // Consistent height
    float contentX = Spacing::PaddingX;  // Proper side padding
    float widgetWidth = scrollBox->getSize().x - 40;  // Room for scrollbar
    
    // Section 1: AI Backend
    drawSectionHeader(renderer, {contentX, yOffset}, widgetWidth, 
                     "AI BACKEND", Colors::SectionAI);
    yOffset += Sizes::HeaderHeight + Spacing::Medium;
    
    // Backend button
    // ... create button ...
    yOffset += widgetHeight + Spacing::Medium;
    
    // Divider before next section
    yOffset += Spacing::Large;
    drawDivider(renderer, {contentX, yOffset}, widgetWidth);
    yOffset += Spacing::Large;
    
    // Section 2: Voice Engine
    drawSectionHeader(renderer, {contentX, yOffset}, widgetWidth,
                     "VOICE ENGINE", Colors::SectionVoice);
    yOffset += Sizes::HeaderHeight + Spacing::Medium;
    
    // ... and so on
}
```

### Step 3: Enhance Button Drawing

Update `drawOverlay()` to use theme colors and hover states:

```cpp
void UISettingsMenu::drawOverlay(OverlayRenderer& renderer) {
    if (!isVisible()) return;
    
    UIPanel::drawOverlay(renderer);
    scrollBox->drawOverlay(renderer, position);
    
    Vec2 scrollBoxPos = scrollBox->getPosition();
    float scrollOffset = scrollBox->getScrollOffset();
    
    // Draw widgets with proper theming
    for (size_t i = 0; i < buttons.size(); ++i) {
        Vec2 btnPos = buttons[i]->getPosition();
        Vec2 btnSize = buttons[i]->getSize();
        
        // Skip if outside visible area
        if (btnPos.y + btnSize.y < scrollBoxPos.y ||
            btnPos.y > scrollBoxPos.y + scrollBox->getSize().y) {
            continue;
        }
        
        // Determine button type and colors
        uint32_t tintColor = Colors::WidgetBg;
        uint32_t borderColor = Colors::BorderPrimary;
        
        if (i < 4) {  // Cycle buttons
            tintColor = Colors::SectionAI;  // Or section-specific color
        } else if (i == 4) {  // Save button
            tintColor = Colors::Success;
            borderColor = Colors::Success;
        } else if (i == 5) {  // Cancel button
            tintColor = Colors::Danger;
            borderColor = Colors::Danger;
        }
        
        // Draw with helper (includes hover state)
        bool isHovered = false;  // TODO: Track hover state
        drawWidgetBackground(renderer, btnPos, btnSize, isHovered, false, false);
        
        // Draw category indicator for cycle buttons
        if (i < 4) {
            drawCategoryIndicator(renderer, btnPos, btnSize.y, borderColor);
        }
        
        // Draw button text
        float textY = btnPos.y + (btnSize.y / 2.0f) - 8;
        renderer.drawText({btnPos.x + 10, textY}, buttonLabels[i], Colors::TextPrimary);
    }
    
    // Unsaved changes indicator
    if (hasChanges) {
        renderer.drawText({position.x + size.x - 150, position.y + 8}, 
                         "? Unsaved", Colors::Warning);
    }
}
```

### Step 4: Add Hover State Tracking

Add to UISettingsMenu class:

```cpp
// In ui_settings_menu.hpp:
private:
    std::vector<bool> buttonHoverStates;
    std::vector<bool> sliderHoverStates;
    std::vector<bool> toggleHoverStates;
    std::vector<bool> dropdownHoverStates;
```

Update in `update()`:

```cpp
void UISettingsMenu::update(const InputState& input, float dt) {
    // ... existing code ...
    
    // Track hover states
    Vec2 m = input.mousePos;
    
    buttonHoverStates.resize(buttons.size(), false);
    for (size_t i = 0; i < buttons.size(); ++i) {
        Vec2 btnPos = buttons[i]->getPosition();
        Vec2 btnSize = buttons[i]->getSize();
        
        buttonHoverStates[i] = (m.x >= btnPos.x && m.x <= btnPos.x + btnSize.x &&
                                m.y >= btnPos.y && m.y <= btnPos.y + btnSize.y);
    }
    
    // Similar for sliders, toggles, dropdowns...
}
```

### Step 5: Improve Widget Rendering

**Sliders:**
```cpp
// In ui_slider.cpp drawOverlay():
using namespace UITheme;

// Background
drawWidgetBackground(renderer, position, size, isHovered, isDragging, false);

// Label with consistent styling
renderer.drawText({position.x + Spacing::PaddingX, position.y + 10}, 
                 label, Colors::TextSecondary);

// Value display (right-aligned)
std::string valueStr = std::to_string(static_cast<int>(currentValue));
float valueWidth = getTextWidth(valueStr);
renderer.drawText({position.x + size.x - valueWidth - Spacing::PaddingX, position.y + 10},
                 valueStr, Colors::TextValue);

// Track
Vec2 trackPos = {position.x + 150, position.y + size.y / 2 - 2};
Vec2 trackSize = {size.x - 170, 4};
renderer.drawRect(trackPos, trackSize, Colors::BorderSubtle);

// Filled portion
float fillWidth = (currentValue - minValue) / (maxValue - minValue) * trackSize.x;
renderer.drawRect(trackPos, {fillWidth, trackSize.y}, Colors::Primary);

// Handle
Vec2 handlePos = {trackPos.x + fillWidth - 6, trackPos.y - 6};
Vec2 handleSize = {12, 16};
uint32_t handleColor = isDragging ? Colors::Primary : Colors::WidgetBgHover;
renderer.drawRect(handlePos, handleSize, handleColor);
renderer.drawRect(handlePos, {handleSize.x, 2}, Colors::BorderFocus);
```

**Toggles:**
```cpp
// In ui_toggle.cpp drawOverlay():
using namespace UITheme;

// Label
renderer.drawText({position.x + Spacing::PaddingX, position.y + 10}, 
                 label, Colors::TextSecondary);

// Switch background
Vec2 switchPos = {position.x + size.x - 60, position.y + 5};
Vec2 switchSize = {50, 20};
uint32_t switchBg = isOn ? Colors::Success : Colors::WidgetBg;
renderer.drawRect(switchPos, switchSize, switchBg);

// Switch handle
float handleX = isOn ? switchPos.x + 32 : switchPos.x + 2;
Vec2 handlePos = {handleX, switchPos.y + 2};
Vec2 handleSize = {16, 16};
renderer.drawRect(handlePos, handleSize, Colors::TextPrimary);

// Border
renderer.drawRect(switchPos, {switchSize.x, 2}, Colors::BorderSubtle);
```

**Dropdowns:**
```cpp
// In ui_dropdown.cpp drawOverlay():
using namespace UITheme;

// Label
renderer.drawText({position.x + Spacing::PaddingX, position.y + 15}, 
                 label, Colors::TextSecondary);

// Dropdown box
Vec2 boxPos = {position.x + 150, position.y + 5};
Vec2 boxSize = {size.x - 160, 30};

drawWidgetBackground(renderer, boxPos, boxSize, isHovered, expanded, false);

// Selected value
renderer.drawText({boxPos.x + 8, boxPos.y + 8}, 
                 getSelectedItem(), Colors::TextPrimary);

// Arrow indicator
std::string arrow = expanded ? "?" : "?";
renderer.drawText({boxPos.x + boxSize.x - 20, boxPos.y + 8}, 
                 arrow, Colors::Primary);

// Expanded options
if (expanded) {
    for (size_t i = 0; i < options.size(); ++i) {
        Vec2 optPos = {boxPos.x, boxPos.y + boxSize.y + i * 25};
        Vec2 optSize = {boxSize.x, 25};
        
        bool isSelected = (i == selectedIndex);
        bool isHoveredOpt = false;  // Track this
        
        uint32_t optBg = isSelected ? Colors::WidgetBgActive : 
                         isHoveredOpt ? Colors::WidgetBgHover : Colors::WidgetBg;
        
        renderer.drawRect(optPos, optSize, optBg);
        renderer.drawRect(optPos, {optSize.x, 1}, Colors::BorderSubtle);
        
        renderer.drawText({optPos.x + 8, optPos.y + 5}, 
                         options[i], isSelected ? Colors::TextValue : Colors::TextPrimary);
    }
}
```

### Step 6: Add Section Grouping

Organize widgets into logical sections:

```cpp
void UISettingsMenu::createWidgets() {
    // ... initialization ...
    
    // ???????????????????????????????????????
    // SECTION 1: AI Backend
    // ???????????????????????????????????????
    drawSectionHeader(renderer, {contentX, yOffset}, widgetWidth,
                     "AI BACKEND", Colors::SectionAI);
    yOffset += Sizes::HeaderHeight + Spacing::Medium;
    
    // Backend selection button
    // ...
    yOffset += widgetHeight + Spacing::Large;
    
    // Divider
    drawDivider(renderer, {contentX, yOffset}, widgetWidth);
    yOffset += Spacing::Large;
    
    // ???????????????????????????????????????
    // SECTION 2: Voice Engine
    // ???????????????????????????????????????
    drawSectionHeader(renderer, {contentX, yOffset}, widgetWidth,
                     "VOICE ENGINE", Colors::SectionVoice);
    yOffset += Sizes::HeaderHeight + Spacing::Medium;
    
    // Voice engine button
    // Speaker dropdown (if applicable)
    // ...
    yOffset += Spacing::Large;
    
    // Divider
    drawDivider(renderer, {contentX, yOffset}, widgetWidth);
    yOffset += Spacing::Large;
    
    // ???????????????????????????????????????
    // SECTION 3: Whisper STT
    // ???????????????????????????????????????
    drawSectionHeader(renderer, {contentX, yOffset}, widgetWidth,
                     "SPEECH RECOGNITION", Colors::SectionWhisper);
    yOffset += Sizes::HeaderHeight + Spacing::Medium;
    
    // Model selection
    // Temperature slider
    // Beam size slider
    // Suppress blank toggle
    // ...
    yOffset += Spacing::Large;
    
    // Divider
    drawDivider(renderer, {contentX, yOffset}, widgetWidth);
    yOffset += Spacing::Large;
    
    // ???????????????????????????????????????
    // SECTION 4: Personality
    // ???????????????????????????????????????
    drawSectionHeader(renderer, {contentX, yOffset}, widgetWidth,
                     "PERSONALITY", Colors::SectionPersonality);
    yOffset += Sizes::HeaderHeight + Spacing::Medium;
    
    // Personality preset button
    // Custom personality toggle
    // ...
    yOffset += Spacing::XLarge;  // Extra space before buttons
    
    // ???????????????????????????????????????
    // Action Buttons
    // ???????????????????????????????????????
    // Save & Close, Cancel
    // ...
}
```

## Summary of Improvements

? **Consistent Spacing** - All using theme constants  
? **Visual Hierarchy** - Section headers and dividers  
? **Color Coding** - Different sections have distinct tints  
? **Hover States** - Tracked and rendered for all widgets  
? **Better Typography** - Consistent label/value display  
? **Theme System** - Centralized colors and sizes  
? **Helper Functions** - Reusable drawing utilities  

## Next Steps

1. Apply these changes to `ui_settings_menu.cpp`
2. Update individual widget classes (slider, toggle, dropdown)
3. Add smooth transitions (future enhancement)
4. Add keyboard navigation (future enhancement)
5. Add tooltips on hover (future enhancement)
