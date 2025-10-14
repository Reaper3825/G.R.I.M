#pragma once
#include <cstdint>
#include "console_history.hpp"
#include "ui_constants.hpp"
#include "console_ui.hpp"

void drawUI(const GRIMConsole::ConsoleState& state,
            ConsoleHistory& history,
            uint32_t width,
            uint32_t height);
