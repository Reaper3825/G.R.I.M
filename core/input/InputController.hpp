#pragma once
#include "grim_platform.h"
#include <string>
#include <vector>

namespace GRIM {

enum class MouseButton {
    Left,
    Right,
    Middle
};

enum class KeyAction {
    Press,
    Release,
    Tap
};

class InputController {
public:
    static void moveMouse(int x, int y);
    static void click(MouseButton button);
    static void doubleClick(MouseButton button);
    static void scroll(int delta);
    static void typeText(const std::string& text, int delayMs = 10);
    static void keyEvent(WORD vk, KeyAction action);
    static void combo(const std::vector<WORD>& keys);

private:
    static void sendMouseEvent(DWORD flags, int dx = 0, int dy = 0, DWORD data = 0);
};

} // namespace GRIM
