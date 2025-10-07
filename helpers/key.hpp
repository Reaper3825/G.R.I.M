#pragma once
#include <unordered_map>
#include <vector>
#include <functional>
#include <windows.h>

enum class KeyCode
{
    A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,
    F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,
    SHIFT,CTRL,ALT,SPACE,ENTER,ESCAPE,
    UNKNOWN
};

class Key
{
public:
    using Callback = std::function<void(KeyCode)>;

    static void initialize();   // start listening
    static void shutdown();     // stop listening
    static void endFrame();

    static bool isDown(KeyCode code);
    static bool wasPressed(KeyCode code);
    static bool wasReleased(KeyCode code);

    static void onPress(KeyCode code, Callback cb);
    static void onRelease(KeyCode code, Callback cb);

private:
    struct KeyState
    {
        bool down = false;
        bool pressed = false;
        bool released = false;
        std::vector<Callback> pressCallbacks;
        std::vector<Callback> releaseCallbacks;
    };

    static std::unordered_map<KeyCode, KeyState> keyStates;
    static HHOOK keyboardHook;

    static LRESULT CALLBACK LowLevelKeyboardProc(int nCode, WPARAM wParam, LPARAM lParam);
    static void setDown(KeyCode code);
    static void setUp(KeyCode code);
};
