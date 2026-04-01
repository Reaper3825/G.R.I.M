#pragma once
#include <cstdint>

// Cross-platform input polling abstraction
// Replaces Windows-specific GetAsyncKeyState and hooks

namespace PlatformInput {
    
    // Initialize platform-specific input system
    void initialize();
    
    // Shutdown platform-specific input system
    void shutdown();
    
    // Check if a mouse button is currently pressed
    // button: 0 = left, 1 = right, 2 = middle
    bool isMouseButtonDown(int button);
    
    // Check if a keyboard key is currently pressed
    // keyCode: Virtual key code (VK_* on Windows, platform-specific elsewhere)
    bool isKeyDown(int keyCode);
    
    // Get current cursor position in screen coordinates
    void getCursorPos(int& x, int& y);
    
    // Get cursor position relative to a window
    void getCursorPosRelative(void* windowHandle, int& x, int& y);
    
    // On macOS, returns true when Command is held (used so Cmd maps to Ctrl for shortcuts)
    bool isCommandDown();

    // Event-driven key state tracking (called from platform event pump)
    void setKeyDownFromEvent(int keyCode, bool down);
    void setCommandDownFromEvent(bool down);

    // Virtual key code mappings (cross-platform)
    enum class Key : int {
        // Mouse buttons
        LeftButton = 0x01,
        RightButton = 0x02,
        MiddleButton = 0x04,
        
        // Modifier keys
        Shift = 0x10,
        Control = 0x11,
        Alt = 0x12,
        
        // Common keys
        Escape = 0x1B,
        Space = 0x20,
        Enter = 0x0D,
        Tab = 0x09,
        Backspace = 0x08,
        Delete = 0x2E,
        
        // Arrow keys
        Left = 0x25,
        Up = 0x26,
        Right = 0x27,
        Down = 0x28,
        
        // Function keys
        F1 = 0x70,
        F2 = 0x71,
        F3 = 0x72,
        F4 = 0x73,
        F5 = 0x74,
        F6 = 0x75,
        F7 = 0x76,
        F8 = 0x77,
        F9 = 0x78,
        F10 = 0x79,
        F11 = 0x7A,
        F12 = 0x7B,
    };
}
