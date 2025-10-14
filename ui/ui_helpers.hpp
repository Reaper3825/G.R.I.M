#pragma once
#include <string>
#include <cstdint>

// ============================================================
// GRIM UI Helper functions (BGFX version)
// ============================================================

// Caret blink — returns new caret visibility state
bool updateCaretBlink(uint64_t& lastToggleTime, bool caretVisible);

// Clamps scroll offset between 0 and maxScroll
void clampScroll(float& scrollOffsetLines, float maxScroll);

// Sets text input buffer directly
void ui_set_textbox(std::string& buffer, const std::string& newText);
