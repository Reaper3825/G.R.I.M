#pragma once
#include <windows.h>
#include <unordered_map>
#include <string>
#include <array>
#include <cstdint>
#include "helpers/vector2.hpp"
#include "helpers/mouse.hpp"

struct InputState {
    Vec2 mousePos{};
    Vec2 mouseDelta{};
    bool mouseDown[3]{};
    bool mousePressed[3]{};
    bool mouseReleased[3]{};

    std::unordered_map<int, bool> keysDown;
    std::unordered_map<int, bool> keyPressed;
    std::unordered_map<int, bool> keyReleased;

    std::string textInput; // For typed characters (console input)
    bool ctrl = false;
    bool shift = false;
    bool alt = false;

    // ? NEW: Mouse input filtering flag
    bool mouseInputEnabled = true;  // If false, mouse events are suppressed

    static InputState capture();                // Existing static method
    void captureFromHWND(HWND hwnd);            // New method for overlay
    void resetFrameState();                     // Clears transient inputs

    // ? NEW: Check if mouse input should be active
    static bool shouldProcessMouseInput();
};
