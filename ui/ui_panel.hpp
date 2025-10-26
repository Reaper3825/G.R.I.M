#pragma once
#include "widget.hpp"
#include <vector>
#include <memory>
#include <functional>

class OverlayRenderer;  // Forward declaration

class UIPanel : public Widget {
public:
    UIPanel(const std::string& title = "", bool draggable = true);

    void addChild(std::shared_ptr<Widget> w);
    void removeChild(Widget* w);

    void update(const InputState& input, float dt) override;
    
    // Override base widget draw (does nothing, we use drawOverlay instead)
    void draw(UIRenderer& renderer) override { /* unused */ }
    
    // New method for overlay rendering
    virtual void drawOverlay(OverlayRenderer& renderer);

    void setBackground(uint32_t color) { bgColor = color; }
    void setBorder(uint32_t color) { borderColor = color; }

    void setOnClose(std::function<void()> cb) { onClose = std::move(cb); }
    void setTitle(const std::string& t) { title = t; }
    std::string getTitle() const { return title; }
    
    void setDraggable(bool drag) { draggable = drag; }
    void setResizable(bool resize) { resizable = resize; }

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

    uint32_t bgColor = 0xFF202020;
    uint32_t borderColor = 0xFFFFFFFF;

    float titleBarHeight = 30.0f;
    float resizeHandleSize = 20.0f;  // Increased from 10 to 20 pixels
    std::function<void()> onClose = nullptr;
    
    // Helper to check if mouse is over resize handle (bottom-right corner)
    bool isOverResizeHandle(const Vec2& mousePos) const;
};
