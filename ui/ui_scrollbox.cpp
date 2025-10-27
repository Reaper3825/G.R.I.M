#include "ui_scrollbox.hpp"
#include "overlay_renderer.hpp"
#include "input_parser.hpp"
#include "helpers/mouse.hpp"
#include "logger.hpp"
#include <algorithm>

UIScrollBox::UIScrollBox()
    : scrollOffset(0.0f), contentHeight(0.0f), childSpacing(5.0f)
{
    LOG_DEBUG("UIScrollBox", "Constructed");
}

void UIScrollBox::addChild(std::shared_ptr<Widget> child) {
    if (child) {
        children.push_back(child);
        LOG_DEBUG("UIScrollBox", "Added child widget");
    }
}

void UIScrollBox::removeChild(Widget* child) {
    children.erase(
        std::remove_if(children.begin(), children.end(),
            [child](const std::shared_ptr<Widget>& ptr) { return ptr.get() == child; }),
        children.end()
    );
}

void UIScrollBox::clearChildren() {
    children.clear();
    scrollOffset = 0.0f;
    contentHeight = 0.0f;
    LOG_DEBUG("UIScrollBox", "Cleared all children");
}

void UIScrollBox::setContentHeight(float height) {
    contentHeight = height;
    
    // Clamp scroll offset to valid range
    float maxScroll = std::max(0.0f, contentHeight - size.y);
    scrollOffset = std::min(scrollOffset, maxScroll);
}

void UIScrollBox::scrollTo(float offset) {
    float maxScroll = std::max(0.0f, contentHeight - size.y);
    scrollOffset = std::clamp(offset, 0.0f, maxScroll);
}

void UIScrollBox::scrollBy(float delta) {
    scrollTo(scrollOffset + delta);
}

bool UIScrollBox::needsScrollbar() const {
    return contentHeight > size.y;
}

float UIScrollBox::getScrollbarHeight() const {
    if (!needsScrollbar()) return size.y;
    
    // Scrollbar height proportional to visible area
    float visibleRatio = size.y / contentHeight;
    return std::max(30.0f, size.y * visibleRatio);
}

float UIScrollBox::getScrollbarY() const {
    if (!needsScrollbar()) return 0.0f;
    
    // Scrollbar position based on scroll offset
    float maxScroll = contentHeight - size.y;
    float scrollRatio = (maxScroll > 0.0f) ? (scrollOffset / maxScroll) : 0.0f;
    
    float maxScrollbarY = size.y - getScrollbarHeight();
    return scrollRatio * maxScrollbarY;
}

Vec2 UIScrollBox::getScrollbarPos() const {
    return {
        position.x + size.x - scrollbarWidth - 2,
        position.y + getScrollbarY() + 2
    };
}

Vec2 UIScrollBox::getScrollbarSize() const {
    return { scrollbarWidth, getScrollbarHeight() };
}

bool UIScrollBox::isMouseOverScrollbar(const Vec2& mousePos) const {
    if (!needsScrollbar()) return false;
    
    Vec2 sbPos = getScrollbarPos();
    Vec2 sbSize = getScrollbarSize();
    
    return (mousePos.x >= sbPos.x && mousePos.x <= sbPos.x + sbSize.x &&
            mousePos.y >= sbPos.y && mousePos.y <= sbPos.y + sbSize.y);
}

void UIScrollBox::update(const InputState& input, float dt) {
    Vec2 m = input.mousePos;
    
    // Check if mouse is over the scrollbox
    bool inside = (m.x >= position.x && m.x <= position.x + size.x &&
                   m.y >= position.y && m.y <= position.y + size.y);
    
    if (!inside && !isDraggingScrollbar) {
        return; // Don't process if mouse is outside and not dragging
    }
    
    // Handle scrollbar dragging
    if (needsScrollbar()) {
        isHoveringScrollbar = isMouseOverScrollbar(m);
        
        if (isHoveringScrollbar && Mouse::wasPressed(MouseButton::Left)) {
            isDraggingScrollbar = true;
            scrollbarDragStartY = m.y;
            scrollbarDragStartOffset = scrollOffset;
        }
        
        if (isDraggingScrollbar) {
            if (Mouse::isDown(MouseButton::Left)) {
                // Calculate new scroll position based on mouse drag
                float deltaY = m.y - scrollbarDragStartY;
                float maxScrollbarY = size.y - getScrollbarHeight();
                float maxScroll = contentHeight - size.y;
                
                if (maxScrollbarY > 0.0f) {
                    float scrollDelta = (deltaY / maxScrollbarY) * maxScroll;
                    scrollTo(scrollbarDragStartOffset + scrollDelta);
                }
            } else {
                isDraggingScrollbar = false;
            }
        }
    }
    
    // Handle mouse wheel scrolling
    if (inside && needsScrollbar()) {
        // Note: Mouse wheel delta would need to be added to InputState
        // For now, this is a placeholder for future mouse wheel support
    }
    
    // Update children (with scroll offset applied)
    for (auto& child : children) {
        if (child->isVisible()) {
            child->update(input, dt);
        }
    }
}

void UIScrollBox::draw(UIRenderer& renderer) {
    // Not used - we use drawOverlay instead
}

void UIScrollBox::drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos) {
    // Draw background
    renderer.drawRect(position, size, 0xFF1A1A1A);
    
    // Draw border
    renderer.drawRect(position, {size.x, 1}, 0xFF333333);
    renderer.drawRect(position, {1, size.y}, 0xFF333333);
    renderer.drawRect({position.x, position.y + size.y - 1}, {size.x, 1}, 0xFF333333);
    renderer.drawRect({position.x + size.x - 1, position.y}, {1, size.y}, 0xFF333333);
    
    // Note: Children rendering is handled by parent (UISettingsMenu)
    // because we need to apply scissor/clipping which is not available here
    
    // Draw scrollbar if needed
    if (needsScrollbar()) {
        Vec2 sbPos = getScrollbarPos();
        Vec2 sbSize = getScrollbarSize();
        
        // Scrollbar track
        Vec2 trackPos = {position.x + size.x - scrollbarWidth - 2, position.y + 2};
        Vec2 trackSize = {scrollbarWidth, size.y - 4};
        renderer.drawRect(trackPos, trackSize, 0xFF0A0A0A);
        
        // Scrollbar thumb
        uint32_t thumbColor = isDraggingScrollbar ? 0xFF6A6A6A : 
                              (isHoveringScrollbar ? 0xFF5A5A5A : 0xFF4A4A4A);
        renderer.drawRect(sbPos, sbSize, thumbColor);
        
        // Scrollbar thumb border
        renderer.drawRect(sbPos, {sbSize.x, 1}, 0xFF7A7A7A);
        renderer.drawRect(sbPos, {1, sbSize.y}, 0xFF7A7A7A);
        renderer.drawRect({sbPos.x, sbPos.y + sbSize.y - 1}, {sbSize.x, 1}, 0xFF2A2A2A);
        renderer.drawRect({sbPos.x + sbSize.x - 1, sbPos.y}, {1, sbSize.y}, 0xFF2A2A2A);
    }
}

void UIScrollBox::autoLayoutChildren(float startY) {
    float currentY = startY;
    
    for (auto& child : children) {
        if (child->isVisible()) {
            Vec2 childSize = child->getSize();
            child->setPosition(position.x, currentY - scrollOffset);
            currentY += childSize.y + childSpacing;
        }
    }
    
    setContentHeight(currentY - startY);
}
