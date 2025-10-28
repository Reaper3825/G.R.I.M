#pragma once
#include "overlay_renderer.hpp"
#include "ui_theme.hpp"
#include "helpers/vector2.hpp"
#include <string>

namespace UIDrawHelpers {
    
    // Draw a section header with background tint
    inline void drawSectionHeader(OverlayRenderer& renderer, 
                                   const Vec2& pos, 
                                   float width, 
                                   const std::string& title,
                                   uint32_t tintColor = UITheme::Colors::SectionAI) {
        using namespace UITheme;
        
        float headerHeight = Sizes::HeaderHeight;
        
        // Draw tinted background
        renderer.drawRect(pos, {width, headerHeight}, tintColor);
        
        // Draw top border
        renderer.drawRect(pos, {width, Sizes::BorderWidth}, Colors::BorderPrimary);
        
        // Draw bottom border
        renderer.drawRect({pos.x, pos.y + headerHeight - Sizes::BorderWidth}, 
                         {width, Sizes::BorderWidth}, Colors::BorderPrimary);
        
        // Draw header text (centered vertically)
        float textY = pos.y + (headerHeight / 2.0f) - 8;
        renderer.drawText({pos.x + Spacing::PaddingX, textY}, 
                         title, Colors::TextHeader);
    }
    
    // Draw a horizontal divider line
    inline void drawDivider(OverlayRenderer& renderer,
                           const Vec2& pos,
                           float width,
                           uint32_t color = UITheme::Colors::BorderSubtle) {
        renderer.drawRect(pos, {width, 1}, color);
    }
    
    // Draw a labeled value (e.g., "Temperature: 0.5")
    inline void drawLabeledValue(OverlayRenderer& renderer,
                                 const Vec2& pos,
                                 const std::string& label,
                                 const std::string& value,
                                 float labelWidth = 120.0f) {
        using namespace UITheme;
        
        // Draw label
        renderer.drawText(pos, label, Colors::TextSecondary);
        
        // Draw value (aligned to right of label area)
        renderer.drawText({pos.x + labelWidth, pos.y}, value, Colors::TextValue);
    }
    
    // Draw a widget background with optional hover state
    inline void drawWidgetBackground(OverlayRenderer& renderer,
                                     const Vec2& pos,
                                     const Vec2& size,
                                     bool isHovered = false,
                                     bool isActive = false,
                                     bool isDisabled = false) {
        using namespace UITheme;
        
        uint32_t bgColor = Colors::WidgetBg;
        if (isDisabled) {
            bgColor = Colors::WidgetBgDisabled;
        } else if (isActive) {
            bgColor = Colors::WidgetBgActive;
        } else if (isHovered) {
            bgColor = Colors::WidgetBgHover;
        }
        
        // Draw background
        renderer.drawRect(pos, size, bgColor);
        
        // Draw border
        uint32_t borderColor = isHovered ? Colors::BorderFocus : Colors::BorderSubtle;
        if (isActive) borderColor = Colors::Primary;
        if (isDisabled) borderColor = Colors::BorderSubtle;
        
        float borderW = Sizes::BorderWidth;
        renderer.drawRect(pos, {size.x, borderW}, borderColor);  // Top
        renderer.drawRect(pos, {borderW, size.y}, borderColor);  // Left
        renderer.drawRect({pos.x, pos.y + size.y - borderW}, {size.x, borderW}, borderColor);  // Bottom
        renderer.drawRect({pos.x + size.x - borderW, pos.y}, {borderW, size.y}, borderColor);  // Right
    }
    
    // Draw a category indicator (small colored bar on left)
    inline void drawCategoryIndicator(OverlayRenderer& renderer,
                                      const Vec2& pos,
                                      float height,
                                      uint32_t color) {
        renderer.drawRect(pos, {3.0f, height}, color);
    }
    
    // Calculate text width (approximate for monospace Consolas)
    inline float getTextWidth(const std::string& text, float fontSize = 14.0f) {
        // Approximate: Consolas is ~8.4 pixels per character at 14pt
        return text.length() * (fontSize * 0.6f);
    }
    
    // Truncate text to fit width
    inline std::string truncateText(const std::string& text, float maxWidth, float fontSize = 14.0f) {
        float charWidth = fontSize * 0.6f;
        size_t maxChars = static_cast<size_t>(maxWidth / charWidth);
        
        if (text.length() <= maxChars) {
            return text;
        }
        
        // Leave room for "..."
        if (maxChars < 4) return "...";
        
        return text.substr(0, maxChars - 3) + "...";
    }
}
