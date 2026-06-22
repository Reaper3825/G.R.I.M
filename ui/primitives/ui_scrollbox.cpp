#include "ui_scrollbox.hpp"
#include "../ui_theme.hpp"
#include "../overlay_renderer.hpp"
#include "../../core/input_parser.hpp"
#include "ui_dropdown.hpp"  // For checking if child is a dropdown
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
    
    // Update children with positions adjusted for scrolling
    for (auto& child : children) {
        if (child->isVisible()) {
            Vec2 childPos = child->getPosition();  // Relative to content area
            
            // Calculate absolute screen position for update
            Vec2 absolutePos = {position.x + childPos.x, position.y + childPos.y - scrollOffset};
            
            // Temporarily update child position for input handling
            child->setPosition(absolutePos.x, absolutePos.y);
            child->update(input, dt);
            
            // Restore relative position
            child->setPosition(childPos.x, childPos.y);
        }
    }
}

void UIScrollBox::draw(UIRenderer& renderer) {
    // Not used - we use drawOverlay instead
}

void UIScrollBox::drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos) {
    using namespace UITheme;
    
    // Draw background — translucent recessed glass
    renderer.drawRoundedRect(position, size, Colors::ContentAreaBg, Sizes::WidgetRadius);
    
    // Draw border — visible glass edges
    renderer.drawRoundedBorder(position, size, Colors::BorderPrimary, Sizes::WidgetRadius);
    

    
    // Render children with scroll offset and culling
    int drawnCount = 0;
    for (auto& child : children) {
        if (!child->isVisible()) continue;
        
        Vec2 childPos = child->getPosition();  // Relative to content area
        Vec2 childSize = child->getSize();
        
        // Calculate absolute screen position: scrollbox position + child relative position - scroll offset
        Vec2 absolutePos = {position.x + childPos.x, position.y + childPos.y - scrollOffset};


        
        // Simple culling - only draw if visible in scrollbox
        if (absolutePos.y + childSize.y < position.y || 
            absolutePos.y > position.y + size.y) {
            if (drawnCount == 0) {

            }
            continue; // Child is outside visible area
        }
        
        // Temporarily update child position for rendering
        Vec2 originalPos = childPos;
        child->setPosition(absolutePos.x, absolutePos.y);
        
        // Render the child
        child->drawOverlay(renderer, panelPos);
        drawnCount++;
        
        // Restore original position
        child->setPosition(originalPos.x, originalPos.y);
    }
    

    
    // Draw scrollbar if needed
    if (needsScrollbar()) {
        Vec2 sbPos = getScrollbarPos();
        Vec2 sbSize = getScrollbarSize();
        
        // Scrollbar track
        Vec2 trackPos = {position.x + size.x - scrollbarWidth - 2, position.y + 2};
        Vec2 trackSize = {scrollbarWidth, size.y - 4};
        renderer.drawRoundedRect(trackPos, trackSize, Colors::ContentAreaBg, Sizes::SmallRadius);
        
        // Scrollbar thumb
        uint32_t thumbColor = isDraggingScrollbar ? Colors::ScrollThumbDrag : 
                              (isHoveringScrollbar ? Colors::ScrollThumbHover : Colors::ScrollThumb);
        renderer.drawRoundedRect(sbPos, sbSize, thumbColor, Sizes::SmallRadius);
        
        // Glass thumb border
        renderer.drawRoundedBorder(sbPos, sbSize, Colors::BorderPrimary, Sizes::SmallRadius);
    }
    
    // SECOND PASS: Draw expanded dropdown lists on top of everything
    for (auto& child : children) {
        if (!child->isVisible()) continue;
        
        // Check if this is a dropdown widget
        auto dropdown = std::dynamic_pointer_cast<UIDropdown>(child);
        if (dropdown && dropdown->isExpanded()) {
            Vec2 childPos = child->getPosition();
            Vec2 absolutePos = {position.x + childPos.x, position.y + childPos.y - scrollOffset};
            
            // Temporarily update position
            Vec2 originalPos = childPos;
            child->setPosition(absolutePos.x, absolutePos.y);
            
            // Draw the expanded list
            dropdown->drawExpandedList(renderer, panelPos);
            
            // Restore position
            child->setPosition(originalPos.x, originalPos.y);
        }
    }
}

void UIScrollBox::autoLayoutChildren(float startY) {
    float currentY = startY;
    float contentX = 10.0f;  // X offset within scrollbox content area
    
    for (auto& child : children) {
        if (child->isVisible()) {
            Vec2 childSize = child->getSize();
            // Set position relative to scrollbox content area (not screen position)
            child->setPosition(contentX, currentY);
            currentY += childSize.y + childSpacing;
        }
    }
    
    setContentHeight(currentY - startY);
}
