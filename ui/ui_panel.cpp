#include "ui_panel.hpp"
#include "ui_theme.hpp"
#include "overlay_renderer.hpp"
#include "input_parser.hpp"
#include "helpers/mouse.hpp"
#include "core/platform_window.hpp"
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

    Vec2 closeOrigin = chromeButtonOrigin(0);
    Vec2 minOrigin   = chromeButtonOrigin(1);
    Vec2 maxOrigin   = chromeButtonOrigin(2);

    closeHovered    = chromeOptions.enableClose    && isInside(input.mousePos, closeOrigin, btnSize);
    maximizeHovered = chromeOptions.enableMaximize && isInside(input.mousePos, maxOrigin, btnSize);
    minimizeHovered = chromeOptions.enableMinimize && isInside(input.mousePos, minOrigin, btnSize);

    bool pressed = input.mousePressed[0];

    if (pressed) {
        if (closeHovered && chromeOptions.enableClose) {
            setVisible(false);
            if (onClose) onClose();
            consumed = true;
        } else if (minimizeHovered && chromeOptions.enableMinimize) {
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
        float menuBarH = PlatformWindow::getMenuBarHeight();
        position.x = monitorRect.origin.x;
        position.y = monitorRect.origin.y + menuBarH;
        size.x = monitorRect.size.x;
        size.y = monitorRect.size.y - menuBarH;
    } else {
        maximized = false;
        position = storedPosition;
        size = storedSize;
    }
}

Vec2 UIPanel::chromeButtonOrigin(int slot) const {
    const float padding = 10.0f;
    const float spacing = chromeButtonSize + 8.0f;
    float x = position.x + padding + spacing * static_cast<float>(slot);
    float y = position.y + (titleBarHeight - chromeButtonSize) * 0.5f;
    return {x, y};
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

PanelRect UIPanel::getContentRect() const {
    float inset = 6.0f;
    return {
        { position.x + inset, position.y + titleBarHeight },
        { size.x - inset * 2.0f, size.y - titleBarHeight - inset }
    };
}

bool UIPanel::drawOverlay(OverlayRenderer& renderer) {
    using namespace UITheme;
    if (!isVisible()) return false;
    
    // Full glassmorphism panel: blur + shadow + translucent fill + glow + border
    float cornerRadius = maximized ? 0.0f : Sizes::BorderRadius;
    renderer.drawGlassPanel(position, size, cornerRadius,
                            bgColor,
                            Colors::BorderPrimary,
                            Colors::GlassHighlight,
                            Colors::BlurRadius,
                            4.0f,
                            reinterpret_cast<uintptr_t>(this));

    // Chrome buttons — macOS traffic light dots (left side: close, minimize, maximize)
    // slot 0 = close (leftmost), 1 = minimize, 2 = maximize
    float dotRadius = chromeButtonSize * 0.5f;
    
    if (chromeOptions.enableClose) {
        Vec2 btn = chromeButtonOrigin(0);
        uint32_t color = closeHovered ? Colors::ChromeClose : Colors::ChromeBtn;
        renderer.drawRoundedRect(btn, {chromeButtonSize, chromeButtonSize}, color, dotRadius);
    }

    if (chromeOptions.enableMinimize) {
        Vec2 btn = chromeButtonOrigin(1);
        uint32_t color = minimizeHovered ? Colors::ChromeMinimize : Colors::ChromeBtn;
        renderer.drawRoundedRect(btn, {chromeButtonSize, chromeButtonSize}, color, dotRadius);
    }

    if (chromeOptions.enableMaximize) {
        Vec2 btn = chromeButtonOrigin(2);
        uint32_t color = maximizeHovered ? Colors::ChromeMaximize : Colors::ChromeBtn;
        renderer.drawRoundedRect(btn, {chromeButtonSize, chromeButtonSize}, color, dotRadius);
    }

    // Title text — centered in title bar
    float titleX = position.x + (size.x - title.length() * 8.0f) * 0.5f;
    renderer.drawText({titleX, position.y + 8.0f}, title, Colors::TextHeader);

    // Resize grip — subtle dots
    if (resizable && !maximized) {
        float handleX = position.x + size.x - 14.0f;
        float handleY = position.y + size.y - 14.0f;
        
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j <= i; ++j) {
                float dx = handleX + (2 - i) * 4.0f;
                float dy = handleY + j * 4.0f;
                renderer.drawRoundedRect({dx, dy}, {2, 2}, Colors::BorderSubtle, 1.0f);
            }
        }
    }

    if (minimized)
        return false;

    // Push clip rect for the content area so subclass draws stay inside the panel
    PanelRect content = getContentRect();
    renderer.pushClipRect(content.origin, content.size);
    return true;
}
