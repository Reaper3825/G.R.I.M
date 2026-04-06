#pragma once
#include "widget.hpp"
#include "ui_theme.hpp"
#include <vector>
#include <memory>
#include <functional>
#include <cstdint>

// Include plugin.hpp for GRIM_HOST_API macro
#include "../core/plugin.hpp"
class OverlayRenderer;  // Forward declaration

struct PanelChromeOptions {
    bool enableMinimize = true;
    bool enableMaximize = true;
    bool enableClose = true;
};

enum class MinimizeType {
    Unknown,
    SIDE,
    TITLE_BAR,
    CLOSE,
    
};

struct PanelRect {
    Vec2 origin;
    Vec2 size;
};

class GRIM_HOST_API UIPanel : public Widget {
public:
    UIPanel(const std::string& title = "", bool draggable = true);
    
    static void setCanvasSize(const Vec2& canvasSize);
    
    // Focus management
    uint64_t getPanelID() const { return panelID; }
    void setPanelID(uint64_t id) { panelID = id; }

    void addChild(std::shared_ptr<Widget> w);
    void removeChild(Widget* w);

    void update(const InputState& input, float dt) override;
    
    // Override base widget draw (does nothing, we use drawOverlay instead)
    void draw(UIRenderer& renderer) override { /* unused */ }
    
    // Draws panel chrome (background, title bar, borders, buttons).
    // Returns true if subclass should draw content (i.e. panel is visible
    // and not minimized). When true, a clip rect for the content area has
    // been pushed onto the renderer -- the caller MUST call
    // renderer.popClipRect() when finished drawing content.
    virtual bool drawOverlay(OverlayRenderer& renderer);
    
    // The drawable region inside the panel, below the title bar and inside borders.
    PanelRect getContentRect() const;

    void setBackground(uint32_t color) { bgColor = color; }
    void setBorder(uint32_t color) { borderColor = color; }

    void setOnClose(std::function<void()> cb) { onClose = std::move(cb); }
    void setTitle(const std::string& t) { title = t; }
    std::string getTitle() const { return title; }
    
    int getZOrder() const { return zOrder; }
    void setZOrder(int z) { zOrder = z; }
    
    void setDraggable(bool drag) { draggable = drag; }
    void setResizable(bool resize) { resizable = resize; }
    void setChromeOptions(const PanelChromeOptions& opts) { chromeOptions = opts; }
    bool isMinimized() const { return minimized; }
    bool isMaximized() const { return maximized; }
    bool isDragging() const { return dragging; }
    bool isResizing() const { return resizing; }

protected:
    std::string title;
    std::vector<std::shared_ptr<Widget>> children;
    bool draggable = true;
    bool resizable = true;
    bool dragging = false;
    bool resizing = false;
    bool hovered = false;
    Vec2 dragOffset{0, 0};
    Vec2 resizeStartPos{0, 0};
    Vec2 resizeStartSize{0, 0};

    uint32_t bgColor = UITheme::Colors::PanelBg;
    uint32_t borderColor = UITheme::Colors::BorderPrimary;

    float titleBarHeight = 30.0f;
    float resizeHandleSize = 20.0f;  // Increased from 10 to 20 pixels
    std::function<void()> onClose = nullptr;
    
    uint64_t panelID = 0;  // Unique panel identifier for focus tracking
    int zOrder = 0;  // Z-order for rendering (higher = on top)

    PanelChromeOptions chromeOptions{};
    bool minimized = false;
    bool maximized = false;
    bool minimizeHovered = false;
    bool maximizeHovered = false;
    bool closeHovered = false;
    float storedHeight = 0.0f;
    Vec2 storedPosition{0.0f, 0.0f};
    Vec2 storedSize{0.0f, 0.0f};
    static Vec2 s_canvasSize;
    static constexpr float chromeButtonSize = 12.0f;
    
    // Helper to check if mouse is over resize handle (bottom-right corner)
    bool isOverResizeHandle(const Vec2& mousePos) const;
    bool handleChromeButtons(const InputState& input);
    void toggleMinimize();
    void toggleMaximize();
    // slot 0 = close (rightmost), 1 = maximize, 2 = minimize (leftmost)
    Vec2 chromeButtonOrigin(int slot) const;
};
