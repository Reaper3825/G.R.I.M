#include "InputController.hpp"
#include "core/input/checked_click.hpp"
#include <stdexcept>

#ifdef _WIN32
#include <thread>
#include <cstdlib>

namespace GRIM {

void InputController::moveMouse(int x, int y) {
    if (!SetCursorPos(x, y)) throw std::runtime_error("Pointer movement rejected by OS");
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

bool InputController::clickAtChecked(int x, int y, MouseButton button) {
    DWORD down = 0, up = 0;
    switch (button) {
    case MouseButton::Left: down = MOUSEEVENTF_LEFTDOWN; up = MOUSEEVENTF_LEFTUP; break;
    case MouseButton::Right: down = MOUSEEVENTF_RIGHTDOWN; up = MOUSEEVENTF_RIGHTUP; break;
    case MouseButton::Middle: down = MOUSEEVENTF_MIDDLEDOWN; up = MOUSEEVENTF_MIDDLEUP; break;
    default: return false;
    }
    INPUT events[2]{};
    events[0].type = events[1].type = INPUT_MOUSE;
    events[0].mi.dwFlags = down;
    events[1].mi.dwFlags = up;
    return Input::SendCheckedClick(x, y, {
        [](int px, int py) { return SetCursorPos(px, py) != FALSE; },
        [](int px, int py) {
            POINT actual{};
            return GetCursorPos(&actual) && actual.x == px && actual.y == py;
        },
        [&] { return SendInput(2, events, sizeof(INPUT)); },
        [&] { SendInput(1, &events[1], sizeof(INPUT)); } // Best-effort release after partial delivery.
    });
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

    if (SendInput(1, &input, sizeof(INPUT)) != 1) throw std::runtime_error("Input event rejected by OS");
}

void InputController::combo(const std::vector<WORD>& keys) {
    try {
        for (auto key : keys) {
            keyEvent(key, KeyAction::Press);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        for (auto it = keys.rbegin(); it != keys.rend(); ++it) {
            keyEvent(*it, KeyAction::Release);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    } catch (...) {
        for (auto it = keys.rbegin(); it != keys.rend(); ++it) {
            try { keyEvent(*it, KeyAction::Release); } catch (...) {}
        }
        throw;
    }
}

void InputController::typeText(const std::string& text, int delayMs) {
    for (char c : text) {
        SHORT vk = VkKeyScanA(c);
        if (vk == -1)
            continue;

        BYTE shift = (vk >> 8) & 1;
        WORD key = vk & 0xFF;

        try {
            if (shift) keyEvent(VK_SHIFT, KeyAction::Press);
            keyEvent(key, KeyAction::Press);
            std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
            keyEvent(key, KeyAction::Release);
            if (shift) keyEvent(VK_SHIFT, KeyAction::Release);
        } catch (...) {
            try { keyEvent(key, KeyAction::Release); } catch (...) {}
            if (shift) { try { keyEvent(VK_SHIFT, KeyAction::Release); } catch (...) {} }
            throw;
        }
    }
}

void InputController::sendMouseEvent(DWORD flags, int dx, int dy, DWORD data) {
    INPUT input{};
    input.type = INPUT_MOUSE;
    input.mi.dx = dx;
    input.mi.dy = dy;
    input.mi.mouseData = data;
    input.mi.dwFlags = flags;
    if (SendInput(1, &input, sizeof(INPUT)) != 1) throw std::runtime_error("Input event rejected by OS");
}

} // namespace GRIM
#else // !_WIN32

namespace GRIM {

void InputController::moveMouse(int, int) {}
void InputController::click(MouseButton) {}
bool InputController::clickAtChecked(int, int, MouseButton) { return false; }
void InputController::doubleClick(MouseButton) {}
void InputController::scroll(int) {}
void InputController::keyEvent(WORD, KeyAction) {}
void InputController::typeText(const std::string&, int) {}
void InputController::combo(const std::vector<WORD>&) {}

void InputController::sendMouseEvent(DWORD, int, int, DWORD) {}

} // namespace GRIM
#endif // _WIN32
