#include "ui_helpers.hpp"
#include "core/grim_platform.h"
#include <chrono>
#include <string>

// ============================================================
// Caret blink (500ms toggle using steady_clock)
// ============================================================
bool updateCaretBlink(uint64_t& lastToggleTime, bool caretVisible)
{
    uint64_t now = GetTickCount64();
    if (now - lastToggleTime > 500)
    {
        caretVisible = !caretVisible;
        lastToggleTime = now;
    }
    return caretVisible;
}

// ============================================================
// Scroll clamp
// ============================================================
void clampScroll(float& scrollOffsetLines, float maxScroll)
{
    if (scrollOffsetLines < 0.0f)
        scrollOffsetLines = 0.0f;
    else if (scrollOffsetLines > maxScroll)
        scrollOffsetLines = maxScroll;
}

// ============================================================
// Textbox update (BGFX version)
// ============================================================
void ui_set_textbox(std::string& buffer, const std::string& newText)
{
    buffer = newText;
}
