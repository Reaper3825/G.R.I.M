#pragma once

// ---------------- UI Tunables ----------------
// All constants here control layout and appearance of the GRIM UI.
// Adjust them in one place to rescale the interface consistently.

/// Height of the top title bar (pixels).
inline constexpr float kTitleBarH = 48.f;

/// Height of the bottom input bar (pixels).
inline constexpr float kInputBarH = 44.f;

/// Horizontal padding on left/right edges (pixels).
inline constexpr float kSidePad = 14.f;

/// Padding between top of window and first history line (pixels).
inline constexpr float kTopPad = 8.f;

/// Padding between bottom of window and last history line (pixels).
inline constexpr float kBottomPad = 8.f;

/// Line spacing multiplier (applied to font size).
inline constexpr float kLineSpacing = 1.3f;

/// Font size for console text (points).
inline constexpr unsigned kFontSize = 18;

/// Font size for title text (points).
inline constexpr unsigned kTitleFontSize = 24;

/// Maximum number of history entries retained.
inline constexpr size_t kMaxHistory = 1000;
