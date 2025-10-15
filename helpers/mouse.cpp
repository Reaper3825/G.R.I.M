// mouse.cpp
#include "mouse.hpp"
#include "logger.hpp"

std::unordered_map<MouseButton, MouseState> Mouse::buttonStates;
HHOOK Mouse::mouseHook = nullptr;

static DWORD g_mouseThreadId = 0;
static std::thread g_mouseThread;

static MouseButton fromButton(WPARAM wp) {
    switch (wp) {
        case WM_LBUTTONDOWN:
        case WM_LBUTTONUP: return MouseButton::Left;
        case WM_RBUTTONDOWN:
        case WM_RBUTTONUP: return MouseButton::Right;
        case WM_MBUTTONDOWN:
        case WM_MBUTTONUP: return MouseButton::Middle;
        case WM_XBUTTONDOWN:
        case WM_XBUTTONUP:
            return HIWORD(wp) == XBUTTON1 ? MouseButton::X1 : MouseButton::X2;
        default: return MouseButton::Unknown;
    }
}

LRESULT CALLBACK Mouse::LowLevelMouseProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode == HC_ACTION) {
        auto* ms = reinterpret_cast<MSLLHOOKSTRUCT*>(lParam);
        POINT pt = ms->pt;

        switch (wParam) {
            case WM_LBUTTONDOWN: case WM_RBUTTONDOWN: case WM_MBUTTONDOWN: case WM_XBUTTONDOWN:
                setDown(fromButton(wParam)); break;
            case WM_LBUTTONUP: case WM_RBUTTONUP: case WM_MBUTTONUP: case WM_XBUTTONUP:
                setUp(fromButton(wParam)); break;
            case WM_MOUSEMOVE:
                for (auto& [btn, st] : buttonStates)
                    st.position = pt;
                break;
        }
    }
    return CallNextHookEx(nullptr, nCode, wParam, lParam);
}

void Mouse::initialize() {
    if (mouseHook) return;

    g_mouseThread = std::thread([] {
        g_mouseThreadId = GetCurrentThreadId();
        HINSTANCE hInst = GetModuleHandleW(nullptr);
        mouseHook = SetWindowsHookExW(WH_MOUSE_LL, LowLevelMouseProc, hInst, 0);
        MSG msg;
        while (GetMessageW(&msg, nullptr, 0, 0)) { }
        if (mouseHook) { UnhookWindowsHookEx(mouseHook); mouseHook = nullptr; }
    });
    g_mouseThread.detach();
}

void Mouse::shutdown() {
    if (g_mouseThreadId) PostThreadMessageW(g_mouseThreadId, WM_QUIT, 0, 0);
}

void Mouse::setDown(MouseButton btn) {
    auto& s = buttonStates[btn];
    if (!s.down) {
        s.pressed = true;
        for (auto& cb : s.pressCallbacks) cb(btn);
    }
    s.down = true;
}

void Mouse::setUp(MouseButton btn) {
    auto& s = buttonStates[btn];
    if (s.down) {
        s.released = true;
        for (auto& cb : s.releaseCallbacks) cb(btn);
    }
    s.down = false;
}

bool Mouse::isDown(MouseButton btn)      { return buttonStates[btn].down; }
bool Mouse::wasPressed(MouseButton btn)  { return buttonStates[btn].pressed; }
bool Mouse::wasReleased(MouseButton btn) { return buttonStates[btn].released; }

POINT Mouse::getPosition() {
    POINT pt;
    GetCursorPos(&pt);
    return pt;
}

void Mouse::endFrame() {
    for (auto& [_, s] : buttonStates) {
        s.pressed = false;
        s.released = false;
    }
}

void Mouse::onPress(MouseButton btn, std::function<void(MouseButton)> cb) {
    buttonStates[btn].pressCallbacks.push_back(std::move(cb));
}
void Mouse::onRelease(MouseButton btn, std::function<void(MouseButton)> cb) {
    buttonStates[btn].releaseCallbacks.push_back(std::move(cb));
}
