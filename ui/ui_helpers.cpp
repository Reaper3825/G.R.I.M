#include "ui_helpers.hpp"
#include "core/grim_platform.h"
#include <chrono>
#include <string>

// ============================================================
// Caret blink (500ms toggle using steady_clock)
// ============================================================
bool updateCaretBlink(uint64_t& lastToggleTime, bool caretVisible)
{
#ifdef _WIN32
    uint64_t now = GetTickCount64();
#else
    auto tp = std::chrono::steady_clock::now().time_since_epoch();
    uint64_t now = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(tp).count());
#endif
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
