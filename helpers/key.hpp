#pragma once
#include <unordered_map>
#include <vector>
#include <functional>
#include <windows.h>

// =====================================================
// Comprehensive keyboard key enumeration for GRIM
// =====================================================
enum class KeyCode
{
    // --- Letters ---
    A, B, C, D, E, F, G, H, I, J, K, L, M,
    N, O, P, Q, R, S, T, U, V, W, X, Y, Z,

    // --- Numbers (top row) ---
    Num0, Num1, Num2, Num3, Num4,
    Num5, Num6, Num7, Num8, Num9,

    // --- Function Keys ---
    F1,  F2,  F3,  F4,  F5,  F6,
    F7,  F8,  F9,  F10, F11, F12,

    // --- Modifier Keys ---
    LShift, RShift,
    LCtrl,  RCtrl,
    LAlt,   RAlt,
    LSystem, RSystem, // Windows / Command keys
    CapsLock, NumLock, ScrollLock,

    // --- Navigation / Editing ---
    Enter,
    Escape,
    Space,
    Backspace,
    Tab,
    Insert,
    Delete,
    Home,
    End,
    PageUp,
    PageDown,

    // --- Arrow Keys ---
    Left,
    Right,
    Up,
    Down,

    // --- Symbols / Punctuation ---
    Dash,       // -
    Equal,      // =
    LBracket,   // [
    RBracket,   // ]
    Backslash,  // 
    Semicolon,  // ;
    Apostrophe, // '
    Comma,      // ,
    Period,     // .
    Slash,      // /
    Grave,      // `

    // --- Keypad (Numpad) ---
    Numpad0, Numpad1, Numpad2, Numpad3, Numpad4,
    Numpad5, Numpad6, Numpad7, Numpad8, Numpad9,
    NumpadAdd, NumpadSubtract, NumpadMultiply, NumpadDivide,
    NumpadDecimal, NumpadEnter,

    // --- Multimedia / System ---
    PrintScreen,
    Pause,
    Menu,

    // --- Fallback ---
    Unknown
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
