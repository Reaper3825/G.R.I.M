// mouse.hpp
#pragma once
#include <unordered_map>
#include <functional>
#include <Windows.h>

enum class MouseButton {
    Left, Right, Middle, X1, X2, Unknown
};

struct MouseState {
    bool down = false;
    bool pressed = false;
    bool released = false;
    POINT position = {0,0};
    std::vector<std::function<void(MouseButton)>> pressCallbacks;
    std::vector<std::function<void(MouseButton)>> releaseCallbacks;
};

class Mouse {
public:
    static void initialize();
    static void shutdown();
    static void updatePosition();
    static POINT getPosition();
    static bool isDown(MouseButton btn);
    static bool wasPressed(MouseButton btn);
    static bool wasReleased(MouseButton btn);
    static void endFrame();
    static void onPress(MouseButton btn, std::function<void(MouseButton)> cb);
    static void onRelease(MouseButton btn, std::function<void(MouseButton)> cb);

private:
    static std::unordered_map<MouseButton, MouseState> buttonStates;
    static HHOOK mouseHook;
    static LRESULT CALLBACK LowLevelMouseProc(int nCode, WPARAM wParam, LPARAM lParam);
    static void setDown(MouseButton btn);
    static void setUp(MouseButton btn);
};
