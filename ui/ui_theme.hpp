#pragma once
#include <cstdint>

namespace UITheme {
    // ====================================================
    // Glassmorphism Color Palette
    // ====================================================
    // Design language: frosted glass panels, soft depth cues,
    // warm neutrals with a refined blue accent family.
    // Semi-transparent backgrounds simulate light diffusion.
    // Beveled borders (lighter top/left, darker bottom/right)
    // give interactive elements tactile, skeuomorphic depth.
    // ====================================================
    namespace Colors {
        // --- Glass Backgrounds (translucent for frosted glass effect) ---
        constexpr uint32_t PanelBg         = 0x99181820;  // Dark card surface (60% opacity, slight blue tint)
        constexpr uint32_t ScrollboxBg     = 0x80141418;  // Recessed content area (50% opacity)
        constexpr uint32_t WidgetBg        = 0xA0222228;  // Widget surface (63% opacity)
        constexpr uint32_t WidgetBgHover   = 0xB02A2A32;  // Lifted on hover (69% opacity)
        constexpr uint32_t WidgetBgActive  = 0xD0323238;  // Pressed (81% opacity)
        constexpr uint32_t WidgetBgDisabled= 0x601A1A1A;  // Faded (disabled)
        constexpr uint32_t Background      = 0xFF0E0E0E;  // Deep base (opaque)
        constexpr uint32_t SliderTrack     = 0x80141418;  // Recessed track
        constexpr uint32_t SliderFill      = 0xFF6B8CFF;  // Filled portion (accent)
        constexpr uint32_t SliderHandle    = 0xD9D0D0D0;  // Handle default
        constexpr uint32_t SliderHandleActive = 0xFF8BABFF; // Handle while dragging

        // --- Toggle States ---
        constexpr uint32_t ToggleBgOn      = 0xA01A3020;  // Translucent green tint
        constexpr uint32_t ToggleBgOff     = 0xA02A2230;  // Translucent purple tint
        constexpr uint32_t ToggleHandle    = 0xE0D0D0D0;  // Knob color
        constexpr uint32_t ToggleText      = 0xFF0E0E0E;  // ON/OFF label (dark)

        // --- Scrollbar ---
        constexpr uint32_t ScrollThumb         = 0x15FFFFFF;  // Resting thumb
        constexpr uint32_t ScrollThumbHover    = 0x20FFFFFF;  // Hovered thumb
        constexpr uint32_t ScrollThumbDrag     = 0x30FFFFFF;  // Dragging thumb

        // --- Card & Content Area ---
        constexpr uint32_t ContentAreaBg   = 0x80141418;  // Recessed inner areas
        constexpr uint32_t CardSurface     = 0xA0202028;  // Card surface
        constexpr uint32_t TableHeaderBg   = 0x90202028;  // Table/list header row
        constexpr uint32_t RowEven         = 0x80141418;  // Alternating row (even)
        constexpr uint32_t RowOdd          = 0x88181820;  // Alternating row (odd)
        constexpr uint32_t RowHover        = 0xA0222228;  // Hovered row
        constexpr uint32_t RowSelected     = 0xB02A2A32;  // Selected row
        constexpr uint32_t DividerLine     = 0x18FFFFFF;  // Section dividers within cards
        constexpr uint32_t PanelShadow     = 0x50000000;  // Drop shadow behind panels

        // --- Accent Palette (muted, modern) ---
        constexpr uint32_t Primary         = 0xFF6B8CFF;  // Periwinkle blue
        constexpr uint32_t PrimaryLight    = 0xFF8BABFF;  // Light periwinkle (active slider handle, etc)
        constexpr uint32_t Success         = 0xFF5AD07A;  // Soft green
        constexpr uint32_t SuccessBg       = 0xA01A3020;  // Translucent green button bg
        constexpr uint32_t Warning         = 0xFFE8A840;  // Warm amber
        constexpr uint32_t WarningLight    = 0xFFE8D050;  // Bright golden yellow
        constexpr uint32_t Danger          = 0xFFE05555;  // Muted coral red
        constexpr uint32_t DangerBg        = 0xA04A1A22;  // Translucent red button bg
        constexpr uint32_t DangerBright    = 0xFFAA2020;  // Solid critical alert bg
        constexpr uint32_t AccentBlue      = 0xFF4080FF;  // Active filter / selection accent
        constexpr uint32_t Info            = 0xFF5B8DEF;  // Cornflower blue

        // --- Text (off-white for reduced glare) ---
        constexpr uint32_t TextPrimary     = 0xFFEAEAEA;  // Soft white
        constexpr uint32_t TextSecondary   = 0xFF909090;  // Neutral gray
        constexpr uint32_t TextMuted       = 0xFF999999;  // Dimmed body text
        constexpr uint32_t TextLight       = 0xFFCCCCCC;  // Light gray body text
        constexpr uint32_t TextLabel       = 0xFFAAAAAA;  // Field labels
        constexpr uint32_t TextDisabled    = 0xFF505050;  // Muted
        constexpr uint32_t TextValue       = 0xFF8BABFF;  // Light periwinkle
        constexpr uint32_t TextLink        = 0xFF8888FF;  // Link/domain text
        constexpr uint32_t TextHeader      = 0xFFF0F0F0;  // Bright off-white
        constexpr uint32_t TextWhite       = 0xFFFFFFFF;  // Pure white

        // --- Section Tints (subtle neutral washes) ---
        constexpr uint32_t SectionNeutral     = 0x90181820;  // Neutral wash
        constexpr uint32_t SectionAI          = 0x901A2028;  // Cool grey
        constexpr uint32_t SectionVoice       = 0x90281820;  // Warm grey
        constexpr uint32_t SectionWhisper     = 0x90282018;  // Warm grey
        constexpr uint32_t SectionPersonality = 0x901A2820;  // Cool grey

        // --- Glass Borders (visible frosted edges) ---
        constexpr uint32_t BorderPrimary   = 0x40FFFFFF;  // Visible glass edge (25% white)
        constexpr uint32_t BorderSubtle    = 0x20FFFFFF;  // Subtle frost line (12% white)
        constexpr uint32_t BorderFocus     = 0xFF6B8CFF;  // Focus ring matches accent
        constexpr uint32_t BorderShadow    = 0x20000000;  // Bottom/right shadow edge
        
        // --- Glass Highlight & Depth ---
        constexpr uint32_t GlassHighlight  = 0x28FFFFFF;  // Top edge glow (15% white)
        constexpr uint32_t GlassInnerGlow  = 0x10FFFFFF;  // Inner diffuse glow
        constexpr uint32_t DividerFaint    = 0x08FFFFFF;  // Faint section divider (3% white)
        constexpr uint32_t BorderDecorative= 0x20FFFFFF;  // Decorative border / divider (12% white)
        constexpr uint32_t BorderMedium    = 0x14FFFFFF;  // Medium frost line (8% white)
        constexpr uint32_t DepthShadow     = 0x50000000;  // Drop shadow for panels
        constexpr uint32_t ShadowLight     = 0x40000000;  // Lighter shadow (detail panels)
        constexpr uint32_t ShadowSubtle    = 0x10000000;  // Very subtle shadow edge
        
        // --- Chrome Button States ---
        constexpr uint32_t ChromeBtn       = 0x20FFFFFF;  // Dim window control button
        constexpr uint32_t ChromeBtnHover  = 0x35FFFFFF;  // Brighten on hover
        constexpr uint32_t ChromeClose     = 0xC0E05555;  // Close button hovered

        // --- Blur ---
        constexpr int BlurRadius           = 12;          // Frosted glass blur kernel radius
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
        
        constexpr float PaddingX   = 16.0f;
        constexpr float PaddingY   = 14.0f;
    }
    
    // ====================================================
    // Sizes
    // ====================================================
    namespace Sizes {
        constexpr float WidgetHeight       = 38.0f;
        constexpr float HeaderHeight       = 32.0f;
        constexpr float ButtonHeight       = 34.0f;
        constexpr float SliderHeight       = 38.0f;
        constexpr float ToggleHeight       = 28.0f;
        constexpr float DropdownHeight     = 34.0f;
        constexpr float ProgressBarHeight  = 28.0f;
        
        constexpr float BorderWidth        = 1.0f;   // Thin glass edges
        constexpr float BorderWidthSubtle  = 1.0f;
        constexpr float BorderRadius       = 16.0f;   // Panel-level rounded glass corners
        constexpr float WidgetRadius       = 6.0f;    // Widget-level rounding (buttons, inputs, etc)
        constexpr float SmallRadius        = 4.0f;    // Small element rounding (handles, text boxes)
        
        constexpr float ScrollbarWidth     = 10.0f;  // Slim modern scrollbar
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
