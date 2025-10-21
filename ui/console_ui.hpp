#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <nlohmann/json.hpp>
#include "commands/commands_core.hpp"
#include "console_history.hpp"
#include "timer.hpp"

// ===========================================================
// BGFX-Based GRIM Console
// ===========================================================
namespace GRIMConsole
{
    // Main entry point — render loop
    void runConsoleUI(int width, int height);

    // Show/hide console window
    void showConsole();
    void hideConsole();
    void toggleConsole();
    
    // Internal helpers
    void notifyConsoleActivity();

    constexpr float kTitleBarH   = 40.f;
    constexpr float kInputBarH   = 48.f;
    constexpr float kLineSpacing = 1.2f;
    constexpr int   kFontSize    = 16;
    constexpr float kSidePad     = 16.f;
    constexpr float kTopPad      = 8.f;
    constexpr float kBottomPad   = 8.f;

    // State
    struct ConsoleState
    {
        std::string inputBuffer;
        bool caretVisible = true;
        bool running = true;
        float scrollOffsetLines = 0.0f;
        uint64_t lastCaretToggle = 0;
    };

    // Global console data
    extern ConsoleState g_state;
    extern ConsoleHistory g_history;
    extern std::vector<Timer> g_uiTimers;
    extern nlohmann::json g_longTermMemory;
}

