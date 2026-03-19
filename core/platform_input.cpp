#include "platform_input.hpp"

#ifdef _WIN32
#include "grim_platform.h"

namespace PlatformInput {
    
    void initialize() {
        // No initialization needed for GetAsyncKeyState approach
    }
    
    void shutdown() {
        // No cleanup needed
    }
    
    bool isMouseButtonDown(int button) {
        int vk = 0;
        switch (button) {
            case 0: vk = VK_LBUTTON; break;
            case 1: vk = VK_RBUTTON; break;
            case 2: vk = VK_MBUTTON; break;
            default: return false;
        }
        return (GetAsyncKeyState(vk) & 0x8000) != 0;
    }
    
    bool isKeyDown(int keyCode) {
        return (GetAsyncKeyState(keyCode) & 0x8000) != 0;
    }

    bool isCommandDown() {
        return false; // Command key is macOS-only
    }
    
    void getCursorPos(int& x, int& y) {
        POINT p{};
        ::GetCursorPos(&p);
        x = p.x;
        y = p.y;
    }
    
    void getCursorPosRelative(void* windowHandle, int& x, int& y) {
        POINT p{};
        ::GetCursorPos(&p);
        if (windowHandle) {
            ::ScreenToClient(static_cast<HWND>(windowHandle), &p);
        }
        x = p.x;
        y = p.y;
    }
}

#elif __linux__
// Linux implementation using X11 or Wayland
#include <X11/Xlib.h>
#include <X11/keysym.h>

namespace PlatformInput {
    static Display* display = nullptr;
    static Window root;
    
    void initialize() {
        display = XOpenDisplay(nullptr);
        if (display) {
            root = DefaultRootWindow(display);
        }
    }
    
    void shutdown() {
        if (display) {
            XCloseDisplay(display);
            display = nullptr;
        }
    }
    
    bool isMouseButtonDown(int button) {
        if (!display) return false;
        
        Window root_return, child_return;
        int root_x, root_y, win_x, win_y;
        unsigned int mask;
        
        XQueryPointer(display, root, &root_return, &child_return,
                     &root_x, &root_y, &win_x, &win_y, &mask);
        
        switch (button) {
            case 0: return (mask & Button1Mask) != 0;
            case 1: return (mask & Button3Mask) != 0;
            case 2: return (mask & Button2Mask) != 0;
            default: return false;
        }
    }
    
    bool isKeyDown(int keyCode) {
        if (!display) return false;
        
        char keys[32];
        XQueryKeymap(display, keys);
        
        KeyCode kc = XKeysymToKeycode(display, keyCode);
        return (keys[kc / 8] & (1 << (kc % 8))) != 0;
    }

    bool isCommandDown() {
        return false; // Command key is macOS-only
    }
    
    void getCursorPos(int& x, int& y) {
        if (!display) {
            x = y = 0;
            return;
        }
        
        Window root_return, child_return;
        int root_x, root_y, win_x, win_y;
        unsigned int mask;
        
        XQueryPointer(display, root, &root_return, &child_return,
                     &root_x, &root_y, &win_x, &win_y, &mask);
        
        x = root_x;
        y = root_y;
    }
    
    void getCursorPosRelative(void* windowHandle, int& x, int& y) {
        if (!display || !windowHandle) {
            x = y = 0;
            return;
        }
        
        Window root_return, child_return;
        int root_x, root_y, win_x, win_y;
        unsigned int mask;
        
        XQueryPointer(display, static_cast<Window>(reinterpret_cast<uintptr_t>(windowHandle)),
                     &root_return, &child_return,
                     &root_x, &root_y, &win_x, &win_y, &mask);
        
        x = win_x;
        y = win_y;
    }
}

#elif __APPLE__
// macOS implementation using Cocoa/Quartz
#include <ApplicationServices/ApplicationServices.h>
#include <Carbon/Carbon.h>

namespace PlatformInput {

    // Map Windows VK code to macOS CGKeyCode (from Carbon Events.h).
    // Returns -1 if no mapping (key not supported for polling).
    static int vkToMacKeyCode(int vk) {
        switch (vk) {
            case 0x08: return kVK_Delete;           // Backspace
            case 0x09: return kVK_Tab;
            case 0x0D: return kVK_Return;
            case 0x10: return kVK_Shift;            // Shift
            case 0x11: return kVK_Control;
            case 0x12: return kVK_Option;          // Alt -> Option
            case 0x1B: return kVK_Escape;
            case 0x20: return kVK_Space;
            case 0x25: return kVK_LeftArrow;
            case 0x26: return kVK_UpArrow;
            case 0x27: return kVK_RightArrow;
            case 0x28: return kVK_DownArrow;
            case 0x2E: return kVK_ForwardDelete;    // Delete
            case 0x30: return kVK_ANSI_0;
            case 0x31: return kVK_ANSI_1;
            case 0x32: return kVK_ANSI_2;
            case 0x33: return kVK_ANSI_3;
            case 0x34: return kVK_ANSI_4;
            case 0x35: return kVK_ANSI_5;
            case 0x36: return kVK_ANSI_6;
            case 0x37: return kVK_ANSI_7;
            case 0x38: return kVK_ANSI_8;
            case 0x39: return kVK_ANSI_9;
            case 0x41: return kVK_ANSI_A;
            case 0x42: return kVK_ANSI_B;
            case 0x43: return kVK_ANSI_C;
            case 0x44: return kVK_ANSI_D;
            case 0x45: return kVK_ANSI_E;
            case 0x46: return kVK_ANSI_F;
            case 0x47: return kVK_ANSI_G;
            case 0x48: return kVK_ANSI_H;
            case 0x49: return kVK_ANSI_I;
            case 0x4A: return kVK_ANSI_J;
            case 0x4B: return kVK_ANSI_K;
            case 0x4C: return kVK_ANSI_L;
            case 0x4D: return kVK_ANSI_M;
            case 0x4E: return kVK_ANSI_N;
            case 0x4F: return kVK_ANSI_O;
            case 0x50: return kVK_ANSI_P;
            case 0x51: return kVK_ANSI_Q;
            case 0x52: return kVK_ANSI_R;
            case 0x53: return kVK_ANSI_S;
            case 0x54: return kVK_ANSI_T;
            case 0x55: return kVK_ANSI_U;
            case 0x56: return kVK_ANSI_V;
            case 0x57: return kVK_ANSI_W;
            case 0x58: return kVK_ANSI_X;
            case 0x59: return kVK_ANSI_Y;
            case 0x5A: return kVK_ANSI_Z;
            case 0x70: return kVK_F1;
            case 0x71: return kVK_F2;
            case 0x72: return kVK_F3;
            case 0x73: return kVK_F4;
            case 0x74: return kVK_F5;
            case 0x75: return kVK_F6;
            case 0x76: return kVK_F7;
            case 0x77: return kVK_F8;
            case 0x78: return kVK_F9;
            case 0x79: return kVK_F10;
            case 0x7A: return kVK_F11;
            case 0x7B: return kVK_F12;
            case 0xC0: return kVK_ANSI_Grave;     // VK_OEM_3 = ` / ~ (console toggle hotkey)
            default:   return -1;
        }
    }

    void initialize() {
        // No initialization needed
    }

    void shutdown() {
        // No cleanup needed
    }

    bool isMouseButtonDown(int button) {
        CGEventSourceRef eventSource = CGEventSourceCreate(kCGEventSourceStateHIDSystemState);
        bool result = false;

        switch (button) {
            case 0: result = CGEventSourceButtonState(kCGEventSourceStateHIDSystemState, kCGMouseButtonLeft); break;
            case 1: result = CGEventSourceButtonState(kCGEventSourceStateHIDSystemState, kCGMouseButtonRight); break;
            case 2: result = CGEventSourceButtonState(kCGEventSourceStateHIDSystemState, kCGMouseButtonCenter); break;
        }

        CFRelease(eventSource);
        return result;
    }

    bool isKeyDown(int keyCode) {
        int macCode = vkToMacKeyCode(keyCode);
        if (macCode < 0) return false;
        return CGEventSourceKeyState(kCGEventSourceStateHIDSystemState, static_cast<CGKeyCode>(macCode));
    }

    bool isCommandDown() {
        return CGEventSourceKeyState(kCGEventSourceStateHIDSystemState, static_cast<CGKeyCode>(kVK_Command))
            || CGEventSourceKeyState(kCGEventSourceStateHIDSystemState, static_cast<CGKeyCode>(kVK_RightCommand));
    }
    
    void getCursorPos(int& x, int& y) {
        CGEventRef event = CGEventCreate(nullptr);
        CGPoint cursor = CGEventGetLocation(event);
        CFRelease(event);
        
        x = static_cast<int>(cursor.x);
        y = static_cast<int>(cursor.y);
    }
    
    void getCursorPosRelative(void* windowHandle, int& x, int& y) {
        // Window-relative positioning requires NSWindow access
        // For now, just return screen coordinates
        getCursorPos(x, y);
    }
}

#else
// Fallback implementation for unsupported platforms
namespace PlatformInput {
    void initialize() {}
    void shutdown() {}
    bool isMouseButtonDown(int) { return false; }
    bool isKeyDown(int) { return false; }
    bool isCommandDown() { return false; }
    void getCursorPos(int& x, int& y) { x = y = 0; }
    void getCursorPosRelative(void*, int& x, int& y) { x = y = 0; }
}

#endif
