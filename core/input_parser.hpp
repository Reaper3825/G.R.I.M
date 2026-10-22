#pragma once
#include "grim_platform.h"
#include <unordered_map>
#include <string>
#include <array>
#include <cstdint>
#include <algorithm>
#include <limits>
#include "helpers/vector2.hpp"
#include "helpers/mouse.hpp"

struct InputState {
    Vec2 mousePos{};
    Vec2 mouseDelta{};
    bool mouseDown[3]{};
    bool mousePressed[3]{};
    bool mouseReleased[3]{};

    // NEW: Mouse wheel scroll delta
    // Compatibility vertical wheel units (120 per logical step). New controls
    // use scrollSteps(): never guess units from the magnitude of a delta.
    float mouseWheelDelta = 0.0f;
    float mouseWheelDeltaX = 0.0f; // Positive = right; same units as Y.
    bool interactionCancelled = false;
    bool keyboardInputEnabled = true;
    bool hasPointerClip = false;
    Vec2 pointerClipMin{}, pointerClipMax{};
    Vec2 scrollSteps() const { return {mouseWheelDeltaX / 120.0f, mouseWheelDelta / 120.0f}; }
    bool pointerInClip() const {
        return !hasPointerClip || (mousePos.x >= pointerClipMin.x && mousePos.y >= pointerClipMin.y &&
            mousePos.x < pointerClipMax.x && mousePos.y < pointerClipMax.y);
    }
    InputState clipped(Vec2 origin, Vec2 extent) const {
        InputState result = *this;
        result.pointerClipMin = hasPointerClip ? Vec2{std::max(pointerClipMin.x, origin.x), std::max(pointerClipMin.y, origin.y)} : origin;
        const Vec2 end{origin.x + extent.x, origin.y + extent.y};
        result.pointerClipMax = hasPointerClip ? Vec2{std::min(pointerClipMax.x, end.x), std::min(pointerClipMax.y, end.y)} : end;
        result.hasPointerClip = true;
        return result;
    }
    void suppressPointer() {
        mouseInputEnabled = false;
        std::fill_n(mouseDown, 3, false);
        std::fill_n(mousePressed, 3, false);
        std::fill_n(mouseReleased, 3, false);
        mouseWheelDelta = mouseWheelDeltaX = 0;
        mouseDelta = {};
        mousePos = {-1000000.0f, -1000000.0f};
    }
    void suppressKeyboard() {
        keyboardInputEnabled = false;
        keysDown.fill(false); keyPressed.fill(false); keyReleased.fill(false);
        textInput.clear(); pastedText.clear();
        copyRequested = pasteRequested = cutRequested = false;
        // Modifiers remain available for pointer gestures (Shift-click, etc.).
    }

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
