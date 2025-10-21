#pragma once
#include "key.hpp"
#include "mouse.hpp"
#include "widget.hpp"




struct InputState {
    Vec2 mousePos;
    bool mouseDown[5] = {};
    bool mousePressed[5] = {};
    bool mouseReleased[5] = {};
    std::unordered_map<KeyCode, bool> keysDown;

    static InputState capture();
};
