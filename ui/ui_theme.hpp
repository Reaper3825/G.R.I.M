#pragma once
#include <cstdint>

namespace UITheme {
    // ====================================================
    // Glassmorphism Color Palette
    // ====================================================
    // Frosted glass panels over a dark scene with neon color
    // blobs bleeding through. High corner radii for a bubbly,
    // organic feel. Clean 1px borders with visible white frost.
    // Translucent fills let background blur show through.
    // ====================================================
    namespace Colors {
        // --- Glass Backgrounds (frost = blur distortion shows through; not just tint) ---
        constexpr uint32_t PanelBg         = 0xB81C1A28;  // 72% opacity so blur is visible
        constexpr uint32_t ScrollboxBg     = 0xA0181624;  // 63% opacity, recessed
        constexpr uint32_t WidgetBg        = 0xB0201E30;  // 69% opacity, slight purple
        constexpr uint32_t WidgetBgHover   = 0x90302E44;  // 56% opacity, lifted
        constexpr uint32_t WidgetBgActive  = 0xA8403E58;  // 66% opacity, pressed
        constexpr uint32_t WidgetBgDisabled= 0x40181828;  // 25% opacity, faded
        constexpr uint32_t Background      = 0xFF0A0A12;  // Deep base (opaque, blue-black)
        constexpr uint32_t SliderTrack     = 0x60141420;  // Recessed track
        constexpr uint32_t SliderFill      = 0xFF7B6EF6;  // Filled portion (violet accent)
        constexpr uint32_t SliderHandle    = 0xE0E0E0F0;  // Handle default (cool white)
        constexpr uint32_t SliderHandleActive = 0xFFA090FF; // Handle while dragging

        // --- Toggle States ---
        constexpr uint32_t ToggleBgOn      = 0x90184028;  // Translucent green tint
        constexpr uint32_t ToggleBgOff     = 0x90281838;  // Translucent purple tint
        constexpr uint32_t ToggleHandle    = 0xE0E0E0F0;  // Knob color
        constexpr uint32_t ToggleText      = 0xFF0E0E14;  // ON/OFF label (dark)

        // --- Scrollbar ---
        constexpr uint32_t ScrollThumb         = 0x18FFFFFF;  // Resting thumb
        constexpr uint32_t ScrollThumbHover    = 0x28FFFFFF;  // Hovered thumb
        constexpr uint32_t ScrollThumbDrag     = 0x3CFFFFFF;  // Dragging thumb

        // --- Card & Content Area ---
        constexpr uint32_t ContentAreaBg   = 0xB0141420;  // 69% opacity, recessed
        constexpr uint32_t CardSurface     = 0xD01E1C2C;  // 82% opacity, card surface
        constexpr uint32_t TableHeaderBg   = 0x70201E30;  // Table/list header row
        constexpr uint32_t RowEven         = 0x50141420;  // Alternating row (even)
        constexpr uint32_t RowOdd          = 0x581A1828;  // Alternating row (odd)
        constexpr uint32_t RowHover        = 0x78282640;  // Hovered row
        constexpr uint32_t RowSelected     = 0x90302E50;  // Selected row
        constexpr uint32_t DividerLine     = 0x18FFFFFF;  // Section dividers within cards
        constexpr uint32_t PanelShadow     = 0x60000008;  // Drop shadow behind panels

        // --- Accent Palette (neon-inspired, vibrant) ---
        constexpr uint32_t Primary         = 0xFF7B6EF6;  // Soft violet
        constexpr uint32_t PrimaryLight    = 0xFFA090FF;  // Light lavender
        constexpr uint32_t Success         = 0xFF50E080;  // Neon green
        constexpr uint32_t SuccessBg       = 0x80183828;  // Translucent green button bg
        constexpr uint32_t Warning         = 0xFFF0B040;  // Warm amber
        constexpr uint32_t WarningLight    = 0xFFF0D860;  // Bright golden yellow
        constexpr uint32_t Danger          = 0xFFE84060;  // Vibrant pink-red
        constexpr uint32_t DangerBg        = 0x80481828;  // Translucent red button bg
        constexpr uint32_t DangerBright    = 0xFFCC2040;  // Solid critical alert bg
        constexpr uint32_t AccentBlue      = 0xFF5090FF;  // Active filter / selection accent
        constexpr uint32_t Info            = 0xFF6B90F0;  // Cornflower blue

        // --- Text (cool off-white for glass contrast) ---
        constexpr uint32_t TextPrimary     = 0xFFEEEEF4;  // Cool soft white
        constexpr uint32_t TextSecondary   = 0xFF8888A0;  // Cool gray
        constexpr uint32_t TextMuted       = 0xFF9090A8;  // Dimmed body text
        constexpr uint32_t TextLight       = 0xFFCCCCD8;  // Light gray body text
        constexpr uint32_t TextLabel       = 0xFFAAAABB;  // Field labels
        constexpr uint32_t TextDisabled    = 0xFF484860;  // Muted
        constexpr uint32_t TextValue       = 0xFFA090FF;  // Light lavender
        constexpr uint32_t TextLink        = 0xFF9080FF;  // Link/domain text
        constexpr uint32_t TextHeader      = 0xFFF4F4FF;  // Bright cool white
        constexpr uint32_t TextWhite       = 0xFFFFFFFF;  // Pure white

        // --- Section Tints (purple-shifted washes) ---
        constexpr uint32_t SectionNeutral     = 0x701A1828;  // Cool neutral wash
        constexpr uint32_t SectionAI          = 0x70181830;  // Cool violet tint
        constexpr uint32_t SectionVoice       = 0x70281828;  // Warm magenta tint
        constexpr uint32_t SectionWhisper     = 0x70282018;  // Warm amber tint
        constexpr uint32_t SectionPersonality = 0x70182820;  // Cool teal tint

        // --- Glass Borders (visible frosted edges, clean) ---
        constexpr uint32_t BorderPrimary   = 0x50FFFFFF;  // Clean glass edge (31% white)
        constexpr uint32_t BorderSubtle    = 0x28FFFFFF;  // Subtle frost line (16% white)
        constexpr uint32_t BorderFocus     = 0xFF7B6EF6;  // Focus ring matches accent
        constexpr uint32_t BorderShadow    = 0x28000010;  // Bottom/right shadow edge
        
        // --- Glass Highlight & Depth ---
        constexpr uint32_t GlassHighlight  = 0x38FFFFFF;  // Top edge glow (22% white)
        constexpr uint32_t GlassInnerGlow  = 0x14FFFFFF;  // Inner diffuse glow
        constexpr uint32_t DividerFaint    = 0x0CFFFFFF;  // Faint section divider (5% white)
        constexpr uint32_t BorderDecorative= 0x28FFFFFF;  // Decorative border / divider (16% white)
        constexpr uint32_t BorderMedium    = 0x1CFFFFFF;  // Medium frost line (11% white)
        constexpr uint32_t DepthShadow     = 0x60000010;  // Drop shadow for panels (blue tint)
        constexpr uint32_t ShadowLight     = 0x48000008;  // Lighter shadow
        constexpr uint32_t ShadowSubtle    = 0x14000008;  // Very subtle shadow edge
        
        // --- Chrome Button States (macOS traffic light style) ---
        constexpr uint32_t ChromeClose     = 0xFFE84060;  // Red close dot
        constexpr uint32_t ChromeMinimize  = 0xFFF0B040;  // Yellow minimize dot
        constexpr uint32_t ChromeMaximize  = 0xFF50E080;  // Green maximize dot
        constexpr uint32_t ChromeBtn       = 0x30FFFFFF;  // Unfocused chrome dot
        constexpr uint32_t ChromeBtnHover  = 0x50FFFFFF;  // Brighten on hover

        // --- Blur (distortion strength) ---
        constexpr int BlurRadius           = 90;          // Frosted glass blur kernel radius
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
