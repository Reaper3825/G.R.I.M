#pragma once
#include <cstdint>

namespace UITheme {
    // ====================================================
    // Neutral Dark Color Palette
    // ====================================================
    // Quiet charcoal surfaces inspired by modern ChatGPT/Cursor
    // style UIs: restrained contrast, soft borders, and muted
    // blue-gray accents instead of neon color.
    // ====================================================
    namespace Colors {
        // --- Neutral Backgrounds ---
        constexpr uint32_t PanelBg         = 0xF01F1F1F;  // Main panel surface
        constexpr uint32_t ScrollboxBg     = 0xE81A1A1A;  // Recessed scroll/list surface
        constexpr uint32_t WidgetBg        = 0xEA2A2A2A;  // Input/button surface
        constexpr uint32_t WidgetBgHover   = 0xF0333333;  // Subtle lift on hover
        constexpr uint32_t WidgetBgActive  = 0xF53A3A3A;  // Pressed/selected widget
        constexpr uint32_t WidgetBgDisabled= 0x701F1F1F;  // Faded but legible
        constexpr uint32_t Background      = 0xFF111111;  // App base
        constexpr uint32_t SliderTrack     = 0x70303030;  // Recessed track
        constexpr uint32_t SliderFill      = 0xFF7F8EA3;  // Muted blue-gray accent
        constexpr uint32_t SliderHandle    = 0xE0D8D8D8;  // Handle default
        constexpr uint32_t SliderHandleActive = 0xFFE4E7EA; // Handle while dragging

        // --- Toggle States ---
        constexpr uint32_t ToggleBgOn      = 0x90404A42;  // Desaturated green tint
        constexpr uint32_t ToggleBgOff     = 0x90303030;  // Neutral off state
        constexpr uint32_t ToggleHandle    = 0xE0E1E1E1;  // Knob color
        constexpr uint32_t ToggleText      = 0xFF121212;  // ON/OFF label (dark)

        // --- Scrollbar ---
        constexpr uint32_t ScrollThumb         = 0x20FFFFFF;  // Resting thumb
        constexpr uint32_t ScrollThumbHover    = 0x34FFFFFF;  // Hovered thumb
        constexpr uint32_t ScrollThumbDrag     = 0x48FFFFFF;  // Dragging thumb

        // --- Card & Content Area ---
        constexpr uint32_t ContentAreaBg   = 0xE0181818;  // Recessed content well
        constexpr uint32_t CardSurface     = 0xEC242424;  // Card surface
        constexpr uint32_t TableHeaderBg   = 0xC02A2A2A;  // Table/list header row
        constexpr uint32_t RowEven         = 0x8C1C1C1C;  // Alternating row (even)
        constexpr uint32_t RowOdd          = 0x94222222;  // Alternating row (odd)
        constexpr uint32_t RowHover        = 0xB0303030;  // Hovered row
        constexpr uint32_t RowSelected     = 0xC83A424C;  // Selected row
        constexpr uint32_t DividerLine     = 0x20FFFFFF;  // Section dividers within cards
        constexpr uint32_t PanelShadow     = 0x70000000;  // Drop shadow behind panels

        // --- Accent Palette (subtle, desaturated) ---
        constexpr uint32_t Primary         = 0xFF7F8EA3;  // Muted blue-gray
        constexpr uint32_t PrimaryLight    = 0xFFA6B0BE;  // Light slate accent
        constexpr uint32_t Success         = 0xFF7BAA86;  // Soft green
        constexpr uint32_t SuccessBg       = 0x80404A42;  // Muted green button bg
        constexpr uint32_t Warning         = 0xFFD0A85F;  // Soft amber
        constexpr uint32_t WarningLight    = 0xFFE0C57A;  // Light amber
        constexpr uint32_t Danger          = 0xFFD06F6F;  // Soft red
        constexpr uint32_t DangerBg        = 0x804A3434;  // Muted red button bg
        constexpr uint32_t DangerBright    = 0xFFB84A4A;  // Solid critical alert bg
        constexpr uint32_t AccentBlue      = 0xFF8FA3BF;  // Active filter / selection accent
        constexpr uint32_t Info            = 0xFF8FA3BF;  // Soft slate blue

        // --- Text ---
        constexpr uint32_t TextPrimary     = 0xFFECECEC;  // Primary text
        constexpr uint32_t TextSecondary   = 0xFFB7B7B7;  // Secondary text
        constexpr uint32_t TextMuted       = 0xFF9A9A9A;  // Muted body text
        constexpr uint32_t TextLight       = 0xFFD8D8D8;  // Light gray body text
        constexpr uint32_t TextLabel       = 0xFFC7C7C7;  // Field labels
        constexpr uint32_t TextDisabled    = 0xFF666666;  // Disabled text
        constexpr uint32_t TextValue       = 0xFFA6B0BE;  // Values and subtle emphasis
        constexpr uint32_t TextLink        = 0xFF9AAABD;  // Link/domain text
        constexpr uint32_t TextHeader      = 0xFFF2F2F2;  // Header text
        constexpr uint32_t TextWhite       = 0xFFFFFFFF;  // Pure white

        // --- Section Tints ---
        constexpr uint32_t SectionNeutral     = 0x70242424;  // Neutral wash
        constexpr uint32_t SectionAI          = 0x70283038;  // Cool slate tint
        constexpr uint32_t SectionVoice       = 0x70302A2A;  // Warm neutral tint
        constexpr uint32_t SectionWhisper     = 0x70302C22;  // Soft amber tint
        constexpr uint32_t SectionPersonality = 0x7026302A;  // Soft green-gray tint

        // --- Borders ---
        constexpr uint32_t BorderPrimary   = 0x2EFFFFFF;  // Clean neutral edge
        constexpr uint32_t BorderSubtle    = 0x18FFFFFF;  // Subtle separator
        constexpr uint32_t BorderFocus     = 0xFF7F8EA3;  // Focus ring matches accent
        constexpr uint32_t BorderShadow    = 0x30000000;  // Bottom/right shadow edge
        
        // --- Highlight & Depth ---
        constexpr uint32_t GlassHighlight  = 0x18FFFFFF;  // Top edge highlight
        constexpr uint32_t GlassInnerGlow  = 0x0AFFFFFF;  // Inner diffuse highlight
        constexpr uint32_t DividerFaint    = 0x0AFFFFFF;  // Faint section divider
        constexpr uint32_t BorderDecorative= 0x18FFFFFF;  // Decorative border / divider
        constexpr uint32_t BorderMedium    = 0x24FFFFFF;  // Medium separator
        constexpr uint32_t DepthShadow     = 0x70000000;  // Drop shadow for panels
        constexpr uint32_t ShadowLight     = 0x50000000;  // Lighter shadow
        constexpr uint32_t ShadowSubtle    = 0x18000000;  // Very subtle shadow edge
        
        // --- Chrome Button States (macOS traffic light style) ---
        constexpr uint32_t ChromeClose     = 0xFFD06F6F;  // Red close dot
        constexpr uint32_t ChromeMinimize  = 0xFFD0A85F;  // Yellow minimize dot
        constexpr uint32_t ChromeMaximize  = 0xFF7BAA86;  // Green maximize dot
        constexpr uint32_t ChromeBtn       = 0x28FFFFFF;  // Unfocused chrome dot
        constexpr uint32_t ChromeBtnHover  = 0x40FFFFFF;  // Brighten on hover

        // --- Blur (distortion strength) ---
        constexpr int BlurRadius           = 65;          // Softer, less frosted blur kernel radius
    }
    
    // ====================================================
    // Spacing & Layout
    // ====================================================
    namespace Spacing {
        constexpr float Tiny       = 4.0f;
        constexpr float Small      = 8.0f;
        constexpr float Medium     = 12.0f;
        constexpr float Large      = 20.0f;
        constexpr float XLarge     = 30.0f;
        
        constexpr float PaddingX   = 18.0f;
        constexpr float PaddingY   = 16.0f;
    }
    
    // ====================================================
    // Sizes
    // ====================================================
    namespace Sizes {
        constexpr float WidgetHeight       = 38.0f;
        constexpr float HeaderHeight       = 34.0f;
        constexpr float ButtonHeight       = 34.0f;
        constexpr float SliderHeight       = 38.0f;
        constexpr float ToggleHeight       = 28.0f;
        constexpr float DropdownHeight     = 34.0f;
        constexpr float ProgressBarHeight  = 28.0f;
        
        constexpr float BorderWidth        = 1.0f;   // Thin glass edges
        constexpr float BorderWidthSubtle  = 1.0f;
        constexpr float BorderRadius       = 22.0f;   // Large rounded glass corners (bubbly)
        constexpr float WidgetRadius       = 10.0f;   // Widget-level rounding (pill-like)
        constexpr float SmallRadius        = 6.0f;    // Small element rounding
        
        constexpr float ChromeDotSize      = 12.0f;   // macOS-style traffic light dots
        constexpr float ScrollbarWidth     = 8.0f;    // Slim modern scrollbar
    }
    
    // ====================================================
    // Typography
    // ====================================================
    namespace Typography {
        constexpr float TitleSize      = 17.0f;
        constexpr float HeaderSize     = 15.0f;
        constexpr float BodySize       = 14.0f;
        constexpr float LabelSize      = 13.0f;
        constexpr float ValueSize      = 14.0f;
    }
    
    // ====================================================
    // Animation & Timing
    // ====================================================
    namespace Timing {
        constexpr float HoverDelay     = 0.03f;
        constexpr float TransitionMs   = 120.0f;
        constexpr float ScrollSmooth   = 0.12f;
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
