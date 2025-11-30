#include "ui_panel.hpp"
#include "overlay_renderer.hpp"
#include "input_parser.hpp"
#include "helpers/mouse.hpp"
#include "logger.hpp"
#include "ui_focus_manager.hpp"
#include "ui_root.hpp"
#include <algorithm>

Vec2 UIPanel::s_canvasSize{1920.0f, 1080.0f};

UIPanel::UIPanel(const std::string& t, bool drag)
    : title(t), draggable(drag) 
{
    // Generate unique panel ID
    panelID = UIFocusManager::getInstance().generateUniqueID();
}

void UIPanel::setCanvasSize(const Vec2& canvasSize)
{
    s_canvasSize = canvasSize;
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

bool UIPanel::handleChromeButtons(const InputState& input) {
    bool consumed = false;
    Vec2 btnSize{chromeButtonSize, chromeButtonSize};
    
    auto isInside = [](const Vec2& pt, const Vec2& pos, const Vec2& sz) {
        return pt.x >= pos.x && pt.x <= pos.x + sz.x &&
               pt.y >= pos.y && pt.y <= pos.y + sz.y;
    };

    Vec2 minOrigin = chromeButtonOrigin(true);
    Vec2 maxOrigin = chromeButtonOrigin(false);

    minimizeHovered = chromeOptions.enableMinimize && isInside(input.mousePos, minOrigin, btnSize);
    maximizeHovered = chromeOptions.enableMaximize && isInside(input.mousePos, maxOrigin, btnSize);

    bool pressed = input.mousePressed[0];

    if (pressed) {
        if (minimizeHovered && chromeOptions.enableMinimize) {
            toggleMinimize();
            consumed = true;
        } else if (maximizeHovered && chromeOptions.enableMaximize) {
            toggleMaximize();
            consumed = true;
        }
    }

    return consumed;
}

void UIPanel::toggleMinimize() {
    if (!chromeOptions.enableMinimize)
        return;

    if (!minimized) {
        storedHeight = size.y;
        minimized = true;
        size.y = titleBarHeight + 6.0f;
    } else {
        minimized = false;
        if (storedHeight > 0.0f)
            size.y = storedHeight;
    }
}

void UIPanel::toggleMaximize() {
    if (!chromeOptions.enableMaximize)
        return;

    if (!maximized) {
        if (minimized) {
            minimized = false;
            if (storedHeight > 0.0f)
                size.y = storedHeight;
        }
        storedPosition = position;
        storedSize = size;
        maximized = true;
        Vec2 center{ position.x + size.x * 0.5f,
                     position.y + size.y * 0.5f };
        auto monitorRect = UIRoot::get().getMonitorRectAt(center);
        float margin = 10.0f;
        position.x = monitorRect.origin.x + margin;
        position.y = monitorRect.origin.y + margin;
        size.x = std::max(200.0f, monitorRect.size.x - margin * 2.0f);
        size.y = std::max(titleBarHeight + 20.0f, monitorRect.size.y - margin * 2.0f);
    } else {
        maximized = false;
        position = storedPosition;
        size = storedSize;
    }
}

Vec2 UIPanel::chromeButtonOrigin(bool minimizeButton) const {
    const float padding = 6.0f;
    float offset = minimizeButton ? (chromeButtonSize + padding) * 2.0f : (chromeButtonSize + padding);
    return {position.x + size.x - offset, position.y + padding};
}

void UIPanel::update(const InputState& input, float dt) {
    Vec2 m = input.mousePos;
    bool inside = (m.x >= position.x && m.x <= position.x + size.x &&
                   m.y >= position.y && m.y <= position.y + size.y);
    hovered = inside;
    
    bool inTitleBar = (m.x >= position.x && m.x <= position.x + size.x &&
                       m.y >= position.y && m.y <= position.y + titleBarHeight);
    
    bool chromeConsumed = handleChromeButtons(input);
    
    bool leftDown = Mouse::isDown(MouseButton::Left);
    
    // Handle resizing FIRST (higher priority)
    if (resizable && !chromeConsumed) {
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
    if (draggable && !chromeConsumed) {
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

    // Clamp to canvas bounds
    position.x = std::clamp(position.x, 0.0f, std::max(0.0f, s_canvasSize.x - size.x));
    position.y = std::clamp(position.y, 0.0f, std::max(0.0f, s_canvasSize.y - size.y));

    if (minimized)
        return;

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
    
    // Draw chrome buttons
    if (chromeOptions.enableMinimize) {
        Vec2 btn = chromeButtonOrigin(true);
        uint32_t color = minimizeHovered ? 0xFF505050 : 0xFF3A3A3A;
        renderer.drawRect(btn, {chromeButtonSize, chromeButtonSize}, color);
        renderer.drawText({btn.x + 6.0f, btn.y + 2.0f}, "-", 0xFFFFFFFF);
    }

    if (chromeOptions.enableMaximize) {
        Vec2 btn = chromeButtonOrigin(false);
        uint32_t color = maximizeHovered ? 0xFF505050 : 0xFF3A3A3A;
        renderer.drawRect(btn, {chromeButtonSize, chromeButtonSize}, color);
        renderer.drawText({btn.x + 5.0f, btn.y + 2.0f}, "[]", 0xFFFFFFFF);
    }

    // Draw resize handle (visual indicator in bottom-right corner)
    if (resizable && !maximized) {
        float handleX = position.x + size.x - resizeHandleSize;
        float handleY = position.y + size.y - resizeHandleSize;
        
        // Draw three diagonal lines as resize grip
        for (int i = 0; i < 3; ++i) {
            float offset = i * 3.0f;
            renderer.drawRect({handleX + offset, handleY + resizeHandleSize - offset - 2}, 
                            {resizeHandleSize - offset, 2}, 0xFF888888);
        }
    }

    if (minimized)
        return;

    // Draw children would go here - not implemented yet
}
