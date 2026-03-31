#pragma once
#include "grim_platform.h"
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

    // NEW: Mouse wheel scroll delta
    float mouseWheelDelta = 0.0f;  // Positive = scroll up, Negative = scroll down

    std::array<bool, 256> keysDown{};
    std::array<bool, 256> keyPressed{};
    std::array<bool, 256> keyReleased{};

    std::string textInput; // For typed characters (console input)
    bool ctrl = false;
    bool shift = false;
    bool alt = false;

    // ? NEW: Mouse input filtering flag
    bool mouseInputEnabled = true;  // If false, mouse events are suppressed

    // ? NEW: Clipboard operations state
    bool copyRequested = false;   // Ctrl+C pressed this frame
    bool pasteRequested = false;  // Ctrl+V pressed this frame
    bool cutRequested = false;    // Ctrl+X pressed this frame
    std::string pastedText;       // Text pasted this frame (if any)

    static InputState capture();                // Existing static method
    void captureFromHWND(HWND hwnd);            // New method for overlay
    void resetFrameState();                     // Clears transient inputs

    // ? NEW: Check if mouse input should be active
    static bool shouldProcessMouseInput();
};
