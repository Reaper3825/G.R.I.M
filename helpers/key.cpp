#include "key.hpp"
#include <windows.h>
#include <iostream>

std::unordered_map<KeyCode, Key::KeyState> Key::keyStates;
HHOOK Key::keyboardHook = nullptr;

// Translate Win32 virtual key to your enum
static KeyCode fromVK(WPARAM vk)
{
    switch (vk)
    {
        case VK_SHIFT: return KeyCode::SHIFT;
        case VK_CONTROL: return KeyCode::CTRL;
        case VK_MENU: return KeyCode::ALT;
        case VK_SPACE: return KeyCode::SPACE;
        case VK_RETURN: return KeyCode::ENTER;
        case VK_ESCAPE: return KeyCode::ESCAPE;
        case VK_F1: return KeyCode::F1;
        case VK_F2: return KeyCode::F2;
        case VK_F3: return KeyCode::F3;
        case VK_F4: return KeyCode::F4;
        case VK_F5: return KeyCode::F5;
        case VK_F6: return KeyCode::F6;
        case VK_F7: return KeyCode::F7;
        case VK_F8: return KeyCode::F8;
        case VK_F9: return KeyCode::F9;
        case VK_F10: return KeyCode::F10;
        case VK_F11: return KeyCode::F11;
        case VK_F12: return KeyCode::F12;
        default:
            if (vk >= 'A' && vk <= 'Z')
                return static_cast<KeyCode>(vk - 'A' + static_cast<int>(KeyCode::A));
            return KeyCode::UNKNOWN;
    }
}

// Hook callback
LRESULT CALLBACK Key::LowLevelKeyboardProc(int nCode, WPARAM wParam, LPARAM lParam)
{
    if (nCode == HC_ACTION)
    {
        auto* kb = reinterpret_cast<KBDLLHOOKSTRUCT*>(lParam);
        KeyCode code = fromVK(kb->vkCode);
        if (code != KeyCode::UNKNOWN)
        {
            if (wParam == WM_KEYDOWN || wParam == WM_SYSKEYDOWN)
                setDown(code);
            else if (wParam == WM_KEYUP || wParam == WM_SYSKEYUP)
                setUp(code);
        }
    }
    return CallNextHookEx(nullptr, nCode, wParam, lParam);
}

// Initialize hook
void Key::initialize()
{
    if (!keyboardHook)
        keyboardHook = SetWindowsHookExW(WH_KEYBOARD_LL, LowLevelKeyboardProc, nullptr, 0);
}

// Shutdown hook
void Key::shutdown()
{
    if (keyboardHook)
    {
        UnhookWindowsHookEx(keyboardHook);
        keyboardHook = nullptr;
    }
}

// Event logic
void Key::setDown(KeyCode code)
{
    auto& state = keyStates[code];
    if (!state.down)
    {
        state.pressed = true;
        for (auto& cb : state.pressCallbacks)
            cb(code);
    }
    state.down = true;
}

void Key::setUp(KeyCode code)
{
    auto& state = keyStates[code];
    if (state.down)
    {
        state.released = true;
        for (auto& cb : state.releaseCallbacks)
            cb(code);
    }
    state.down = false;
}

bool Key::isDown(KeyCode code)      { return keyStates[code].down; }
bool Key::wasPressed(KeyCode code)  { return keyStates[code].pressed; }
bool Key::wasReleased(KeyCode code) { return keyStates[code].released; }

void Key::endFrame()
{
    for (auto& [_, s] : keyStates)
    {
        s.pressed = false;
        s.released = false;
    }
}

void Key::onPress(KeyCode code, Callback cb)
{
    keyStates[code].pressCallbacks.push_back(std::move(cb));
}

void Key::onRelease(KeyCode code, Callback cb)
{
    keyStates[code].releaseCallbacks.push_back(std::move(cb));
}
