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

namespace PlatformInput {
    
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
        // macOS key state checking requires more complex implementation
        // For now, return false (to be implemented)
        return false;
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
    void getCursorPos(int& x, int& y) { x = y = 0; }
    void getCursorPosRelative(void*, int& x, int& y) { x = y = 0; }
}

#endif
