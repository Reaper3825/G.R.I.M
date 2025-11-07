#include "ui_panel.hpp"
#include "overlay_renderer.hpp"
#include "input_parser.hpp"
#include "helpers/mouse.hpp"
#include "logger.hpp"
#include "ui_focus_manager.hpp"
#include <algorithm>

UIPanel::UIPanel(const std::string& t, bool drag)
    : title(t), draggable(drag) 
{
    // Generate unique panel ID
    panelID = UIFocusManager::getInstance().generateUniqueID();
}

void UIPanel::addChild(std::shared_ptr<Widget> w) {
    // Set the panel ID for the widget so it knows which panel it belongs to
    if (w) {
        w->setPanelID(panelID);
    }
    children.push_back(std::move(w));
}

void UIPanel::removeChild(Widget* w) {
    children.erase(std::remove_if(children.begin(), children.end(),
        [&](const std::shared_ptr<Widget>& ptr) { return ptr.get() == w; }),
        children.end());
}

bool UIPanel::isOverResizeHandle(const Vec2& mousePos) const {
    // Bottom-right corner resize handle
    float handleX = position.x + size.x - resizeHandleSize;
    float handleY = position.y + size.y - resizeHandleSize;
    
    return (mousePos.x >= handleX && mousePos.x <= position.x + size.x &&
            mousePos.y >= handleY && mousePos.y <= position.y + size.y);
}

void UIPanel::update(const InputState& input, float dt) {
    Vec2 m = input.mousePos;
    bool inside = (m.x >= position.x && m.x <= position.x + size.x &&
                   m.y >= position.y && m.y <= position.y + size.y);
    hovered = inside;
    
    bool inTitleBar = (m.x >= position.x && m.x <= position.x + size.x &&
                       m.y >= position.y && m.y <= position.y + titleBarHeight);
    
    // Use Mouse class for reliable button detection
    bool leftDown = Mouse::isDown(MouseButton::Left);
    
    // Handle resizing FIRST (higher priority)
    if (resizable) {
        bool overHandle = isOverResizeHandle(m);
        
        // Start resizing if clicking on handle
        if (overHandle && leftDown && !resizing && !dragging) {
            resizing = true;
            resizeStartPos = m;
            resizeStartSize = size;
        }
        
        // While resizing, update size continuously
        if (resizing) {
            if (leftDown) {
                // Continue resizing while mouse is held down
                Vec2 delta = { m.x - resizeStartPos.x, m.y - resizeStartPos.y };
                size.x = std::max(300.0f, resizeStartSize.x + delta.x);
                size.y = std::max(200.0f, resizeStartSize.y + delta.y);
            } else {
                // Stop resizing when mouse is released
                resizing = false;
            }
            return; // Don't process dragging while resizing
        }
    }

    // Handle dragging (lower priority than resize)
    if (draggable) {
        // Start dragging if clicking on title bar
        if (inTitleBar && leftDown && !dragging && !resizing) {
            dragging = true;
            dragOffset = {m.x - position.x, m.y - position.y};
        }
        
        // While dragging, update position continuously
        if (dragging) {
            if (leftDown) {
                // Smooth dragging - update position directly
                position.x = m.x - dragOffset.x;
                position.y = m.y - dragOffset.y;
            } else {
                // Stop dragging when mouse is released
                dragging = false;
            }
        }
    }

    // Update children
    for (auto& c : children) {
        if (c->isVisible()) {
            c->update(input, dt);
        }
    }
}

void UIPanel::drawOverlay(OverlayRenderer& renderer) {
    if (!isVisible()) return;
    
    // Panel background
    renderer.drawRect(position, size, bgColor);

    // Title bar
    renderer.drawRect(position, {size.x, titleBarHeight}, 0xFF303030);
    renderer.drawText({position.x + 8, position.y + 6}, title, 0xFFFFFFFF);

    // Border (4 rectangles)
    renderer.drawRect({position.x, position.y}, {size.x, 2}, borderColor);
    renderer.drawRect({position.x, position.y + size.y - 2}, {size.x, 2}, borderColor);
    renderer.drawRect({position.x, position.y}, {2, size.y}, borderColor);
    renderer.drawRect({position.x + size.x - 2, position.y}, {2, size.y}, borderColor);
    
    // Draw resize handle (visual indicator in bottom-right corner)
    if (resizable) {
        float handleX = position.x + size.x - resizeHandleSize;
        float handleY = position.y + size.y - resizeHandleSize;
        
        // Draw three diagonal lines as resize grip
        for (int i = 0; i < 3; ++i) {
            float offset = i * 3.0f;
            renderer.drawRect({handleX + offset, handleY + resizeHandleSize - offset - 2}, 
                            {resizeHandleSize - offset, 2}, 0xFF888888);
        }
    }

    // Draw children would go here - not implemented yet
}
