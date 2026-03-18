#pragma once
#include "overlay_renderer.hpp"
#include "ui_theme.hpp"
#include "helpers/vector2.hpp"
#include <string>

namespace UIDrawHelpers {
    
    // Draw a section header with subtle divider  [GLASS_PHASE5]
    inline void drawSectionHeader(OverlayRenderer& renderer, 
                                   const Vec2& pos, 
                                   float width, 
                                   const std::string& title,
                                   uint32_t tintColor = UITheme::Colors::SectionAI) {
        using namespace UITheme;
        
        float headerHeight = Sizes::HeaderHeight;
        
        // Draw header text (centered vertically)
        float textY = pos.y + (headerHeight / 2.0f) - 8;
        renderer.drawText({pos.x + Spacing::PaddingX, textY}, 
                         title, Colors::TextHeader);
        
        // Draw single subtle divider at bottom  [GLASS_PHASE5]
        renderer.drawRect({pos.x, pos.y + headerHeight - 1}, 
                         {width, 1}, Colors::DividerLine);
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
    
    // Draw a widget background with optional hover state (glassmorphism)
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
        
        float r = Sizes::WidgetRadius;
        
        // Glass background (rounded)
        renderer.drawRoundedRect(pos, size, bgColor, r);
        
        // Glass border (rounded)
        uint32_t borderColor = isActive ? Colors::Primary : 
                               (isHovered ? Colors::BorderFocus : Colors::BorderPrimary);
        if (isDisabled) borderColor = Colors::BorderSubtle;
        
        renderer.drawRoundedBorder(pos, size, borderColor, r, Sizes::BorderWidth);
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
