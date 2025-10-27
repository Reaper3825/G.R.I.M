#pragma once
#include "widget.hpp"
#include <vector>
#include <memory>
#include <functional>

class OverlayRenderer;
struct InputState;

class UIScrollBox : public Widget {
public:
    UIScrollBox();

    void update(const InputState& input, float dt) override;
    void draw(class UIRenderer& renderer) override;
    
    // Overlay rendering
    void drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos);
    
    // Child management
    void addChild(std::shared_ptr<Widget> child);
    void removeChild(Widget* child);
    void clearChildren();
    
    // Scrolling
    void setContentHeight(float height);
    float getContentHeight() const { return contentHeight; }
    float getScrollOffset() const { return scrollOffset; }
    void scrollTo(float offset);
    void scrollBy(float delta);
    
    // Get children for external rendering
    const std::vector<std::shared_ptr<Widget>>& getChildren() const { return children; }
    
    // Layout helper
    void setChildSpacing(float spacing) { childSpacing = spacing; }
    void autoLayoutChildren(float startY = 0.0f);

private:
    std::vector<std::shared_ptr<Widget>> children;
    
    float scrollOffset = 0.0f;
    float contentHeight = 0.0f;
    float childSpacing = 5.0f;
    
    bool isDraggingScrollbar = false;
    bool isHoveringScrollbar = false;
    float scrollbarWidth = 12.0f;
    float scrollbarDragStartY = 0.0f;
    float scrollbarDragStartOffset = 0.0f;
    
    // Scrollbar calculations
    bool needsScrollbar() const;
    float getScrollbarHeight() const;
    float getScrollbarY() const;
    Vec2 getScrollbarPos() const;
    Vec2 getScrollbarSize() const;
    bool isMouseOverScrollbar(const Vec2& mousePos) const;
};
