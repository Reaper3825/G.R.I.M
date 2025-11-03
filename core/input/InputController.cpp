#include "InputController.hpp"
#include <thread>
#include <cstdlib>

namespace GRIM {

void InputController::moveMouse(int x, int y) {
    SetCursorPos(x, y);
}

void InputController::click(MouseButton button) {
    DWORD downFlag = 0, upFlag = 0;
    switch (button) {
        case MouseButton::Left:
            downFlag = MOUSEEVENTF_LEFTDOWN;
            upFlag = MOUSEEVENTF_LEFTUP;
            break;
        case MouseButton::Right:
            downFlag = MOUSEEVENTF_RIGHTDOWN;
            upFlag = MOUSEEVENTF_RIGHTUP;
            break;
        case MouseButton::Middle:
            downFlag = MOUSEEVENTF_MIDDLEDOWN;
            upFlag = MOUSEEVENTF_MIDDLEUP;
            break;
    }

    sendMouseEvent(downFlag);
    std::this_thread::sleep_for(std::chrono::milliseconds(50 + rand() % 50));
    sendMouseEvent(upFlag);
}

void InputController::doubleClick(MouseButton button) {
    click(button);
    std::this_thread::sleep_for(std::chrono::milliseconds(120 + rand() % 60));
    click(button);
}

void InputController::scroll(int delta) {
    sendMouseEvent(MOUSEEVENTF_WHEEL, 0, 0, static_cast<DWORD>(delta));
}

void InputController::keyEvent(WORD vk, KeyAction action) {
    INPUT input{};
    input.type = INPUT_KEYBOARD;
    input.ki.wVk = vk;

    if (action == KeyAction::Release)
        input.ki.dwFlags = KEYEVENTF_KEYUP;

    SendInput(1, &input, sizeof(INPUT));
}

void InputController::combo(const std::vector<WORD>& keys) {
    for (auto key : keys) {
        keyEvent(key, KeyAction::Press);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    for (auto it = keys.rbegin(); it != keys.rend(); ++it) {
        keyEvent(*it, KeyAction::Release);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

void InputController::typeText(const std::string& text, int delayMs) {
    for (char c : text) {
        SHORT vk = VkKeyScanA(c);
        if (vk == -1)
            continue;

        BYTE shift = (vk >> 8) & 1;
        WORD key = vk & 0xFF;

        if (shift)
            keyEvent(VK_SHIFT, KeyAction::Press);

        keyEvent(key, KeyAction::Press);
        std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
        keyEvent(key, KeyAction::Release);

        if (shift)
            keyEvent(VK_SHIFT, KeyAction::Release);
    }
}

void InputController::sendMouseEvent(DWORD flags, int dx, int dy, DWORD data) {
    INPUT input{};
    input.type = INPUT_MOUSE;
    input.mi.dx = dx;
    input.mi.dy = dy;
    input.mi.mouseData = data;
    input.mi.dwFlags = flags;
    SendInput(1, &input, sizeof(INPUT));
}

} // namespace GRIM
