#pragma once

// ============================================================
// GRIM UI constants (shared between draw + console UI modules)
// ============================================================

// Panel heights
constexpr float kTitleBarH   = 40.0f;
constexpr float kInputBarH   = 40.0f;

// Padding
constexpr float kSidePad     = 12.0f;
constexpr float kTopPad      = 8.0f;
constexpr float kBottomPad   = 8.0f;

// Font and spacing (for dbgText lines, 16px default cell height)
constexpr float kFontSize    = 16.0f;
constexpr float kLineSpacing = 1.2f;

// Glass theme colors (ARGB → BGFX uses ABGR internally)  [GLASS_PHASE6]
constexpr uint32_t kColorBackground = 0xFF0E0E0E;
constexpr uint32_t kColorTitleBar   = 0xF0202020;
constexpr uint32_t kColorInputBar   = 0xF01A1A1A;
