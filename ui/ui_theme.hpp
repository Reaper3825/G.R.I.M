#pragma once
#include <cstdint>

namespace UITheme {
    // ====================================================
    // Color Palette
    // ====================================================
    namespace Colors {
        // Backgrounds
        constexpr uint32_t PanelBg         = 0xE0202020;  // Main panel (semi-transparent)
        constexpr uint32_t ScrollboxBg     = 0xFF151515;  // Darker scrollbox
        constexpr uint32_t WidgetBg        = 0xFF252525;  // Widget default
        constexpr uint32_t WidgetBgHover   = 0xFF303030;  // Widget on hover
        constexpr uint32_t WidgetBgActive  = 0xFF353535;  // Widget when clicked
        constexpr uint32_t WidgetBgDisabled= 0xFF1A1A1A;  // Disabled widget
        constexpr uint32_t Background      = 0xFF0A0A0A;  // Deep background
        constexpr uint32_t SliderTrack     = 0xFF202020;  // Slider track background
        
        // Accents
        constexpr uint32_t Primary         = 0xFF00DDFF;  // Cyan
        constexpr uint32_t Success         = 0xFF00FF88;  // Green
        constexpr uint32_t Warning         = 0xFFFFAA00;  // Orange
        constexpr uint32_t Danger          = 0xFFFF4444;  // Red
        constexpr uint32_t Info            = 0xFF4488FF;  // Blue
        
        // Text
        constexpr uint32_t TextPrimary     = 0xFFFFFFFF;  // Main text
        constexpr uint32_t TextSecondary   = 0xFFAAAAAA;  // Labels
        constexpr uint32_t TextDisabled    = 0xFF666666;  // Disabled
        constexpr uint32_t TextValue       = 0xFF00DDFF;  // Current values
        constexpr uint32_t TextHeader      = 0xFFFFFFFF;  // Section headers
        
        // Section tints
        constexpr uint32_t SectionNeutral  = 0xFF2A3540;  // Neutral (Data Collection, generic panels)
        constexpr uint32_t SectionAI       = 0xFF2A3A4A;  // Blue tint
        constexpr uint32_t SectionVoice    = 0xFF3A2A4A;  // Purple tint
        constexpr uint32_t SectionWhisper  = 0xFF4A3A2A;  // Orange tint
        constexpr uint32_t SectionPersonality = 0xFF2A4A3A; // Green tint
        
        // Borders
        constexpr uint32_t BorderPrimary   = 0xFF00DDFF;  // Accent border
        constexpr uint32_t BorderSubtle    = 0xFF333333;  // Subtle border
        constexpr uint32_t BorderFocus     = 0xFF00FFFF;  // Focused element
    }
    
    // ====================================================
    // Spacing & Layout
    // ====================================================
    namespace Spacing {
        constexpr float Tiny       = 4.0f;   // Minimal spacing
        constexpr float Small      = 8.0f;   // Between related items
        constexpr float Medium     = 12.0f;  // Standard widget spacing
        constexpr float Large      = 20.0f;  // Section spacing
        constexpr float XLarge     = 30.0f;  // Major section breaks
        
        constexpr float PaddingX   = 15.0f;  // Horizontal padding
        constexpr float PaddingY   = 15.0f;  // Vertical padding
    }
    
    // ====================================================
    // Sizes
    // ====================================================
    namespace Sizes {
        constexpr float WidgetHeight       = 40.0f;  // Standard widget height
        constexpr float HeaderHeight       = 30.0f;  // Section header height
        constexpr float ButtonHeight       = 35.0f;  // Button height
        constexpr float SliderHeight       = 40.0f;  // Slider with label
        constexpr float ToggleHeight       = 30.0f;  // Toggle switch
        constexpr float DropdownHeight     = 35.0f;  // Dropdown box
        constexpr float ProgressBarHeight  = 30.0f;  // Progress bar height
        
        constexpr float BorderWidth        = 2.0f;   // Standard border
        constexpr float BorderWidthSubtle  = 1.0f;   // Subtle border
        constexpr float BorderRadius       = 4.0f;   // Corner radius (not used in GDI)
        
        constexpr float ScrollbarWidth     = 14.0f;  // Scrollbar width
    }
    
    // ====================================================
    // Typography
    // ====================================================
    namespace Typography {
        constexpr float TitleSize      = 18.0f;  // Panel title
        constexpr float HeaderSize     = 16.0f;  // Section header
        constexpr float BodySize       = 14.0f;  // Normal text
        constexpr float LabelSize      = 13.0f;  // Small labels
        constexpr float ValueSize      = 14.0f;  // Value display
    }
    
    // ====================================================
    // Animation & Timing
    // ====================================================
    namespace Timing {
        constexpr float HoverDelay     = 0.05f;    // Instant hover
        constexpr float TransitionMs   = 100.0f;  // State transitions
        constexpr float ScrollSmooth   = 0.1f;    // Scroll smoothing
    }
    
    // ====================================================
    // Z-Index Layers
    // ====================================================
    namespace ZIndex {
        constexpr int Panel            = 100;
        constexpr int Widget           = 110;
        constexpr int Dropdown         = 200;
        constexpr int Tooltip          = 300;
        constexpr int Modal            = 400;
    }
}
