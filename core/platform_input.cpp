#include "platform_input.hpp"
#include "grim_platform.h"

#ifdef _WIN32
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

    void setKeyDownFromEvent(int, bool) {} // No-op on Windows (uses GetAsyncKeyState)
    void setCommandDownFromEvent(bool) {}
    
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

    bool moveCursorRelative(int deltaX, int deltaY) {
        INPUT input{};
        input.type = INPUT_MOUSE;
        input.mi.dx = deltaX;
        input.mi.dy = deltaY;
        input.mi.dwFlags = MOUSEEVENTF_MOVE;
        return ::SendInput(1, &input, sizeof(INPUT)) == 1;
    }

    bool emitMouseClick(int button) {
        DWORD downFlag = 0;
        DWORD upFlag = 0;
        switch (button) {
            case 0: downFlag = MOUSEEVENTF_LEFTDOWN;   upFlag = MOUSEEVENTF_LEFTUP; break;
            case 1: downFlag = MOUSEEVENTF_RIGHTDOWN;  upFlag = MOUSEEVENTF_RIGHTUP; break;
            case 2: downFlag = MOUSEEVENTF_MIDDLEDOWN; upFlag = MOUSEEVENTF_MIDDLEUP; break;
            default: return false;
        }
        INPUT inputs[2]{};
        inputs[0].type = INPUT_MOUSE;
        inputs[0].mi.dwFlags = downFlag;
        inputs[1].type = INPUT_MOUSE;
        inputs[1].mi.dwFlags = upFlag;
        return ::SendInput(2, inputs, sizeof(INPUT)) == 2;
    }
}

#elif __linux__
// Linux implementation using X11 or Wayland
#include <X11/Xlib.h>
#include <X11/keysym.h>

namespace PlatformInput {
    static Display* display = nullptr;
    static Window root;

    static KeySym logicalKeyToKeysym(int keyCode) {
        if ((keyCode >= '0' && keyCode <= '9') || (keyCode >= 'A' && keyCode <= 'Z'))
            return static_cast<KeySym>(keyCode);
        if (keyCode >= VK_F1 && keyCode <= VK_F12)
            return static_cast<KeySym>(XK_F1 + keyCode - VK_F1);

        switch (keyCode) {
            case VK_BACK: return XK_BackSpace; case VK_TAB: return XK_Tab;
            case VK_RETURN: return XK_Return; case VK_ESCAPE: return XK_Escape;
            case VK_SPACE: return XK_space; case VK_INSERT: return XK_Insert;
            case VK_DELETE: return XK_Delete; case VK_HOME: return XK_Home;
            case VK_END: return XK_End; case VK_PRIOR: return XK_Page_Up;
            case VK_NEXT: return XK_Page_Down; case VK_LEFT: return XK_Left;
            case VK_RIGHT: return XK_Right; case VK_UP: return XK_Up;
            case VK_DOWN: return XK_Down;
            case VK_LSHIFT: return XK_Shift_L; case VK_RSHIFT: return XK_Shift_R;
            case VK_LCONTROL: return XK_Control_L; case VK_RCONTROL: return XK_Control_R;
            case VK_LMENU: return XK_Alt_L; case VK_RMENU: return XK_Alt_R;
            case VK_LWIN: return XK_Super_L; case VK_RWIN: return XK_Super_R;
            case VK_CAPITAL: return XK_Caps_Lock; case VK_NUMLOCK: return XK_Num_Lock;
            case VK_SCROLL: return XK_Scroll_Lock;
            case VK_OEM_MINUS: return XK_minus; case VK_OEM_PLUS: return XK_equal;
            case VK_OEM_4: return XK_bracketleft; case VK_OEM_6: return XK_bracketright;
            case VK_OEM_5: return XK_backslash; case VK_OEM_1: return XK_semicolon;
            case VK_OEM_7: return XK_apostrophe; case VK_OEM_COMMA: return XK_comma;
            case VK_OEM_PERIOD: return XK_period; case VK_OEM_2: return XK_slash;
            case VK_OEM_3: return XK_grave;
            case VK_NUMPAD0: return XK_KP_0; case VK_NUMPAD1: return XK_KP_1;
            case VK_NUMPAD2: return XK_KP_2; case VK_NUMPAD3: return XK_KP_3;
            case VK_NUMPAD4: return XK_KP_4; case VK_NUMPAD5: return XK_KP_5;
            case VK_NUMPAD6: return XK_KP_6; case VK_NUMPAD7: return XK_KP_7;
            case VK_NUMPAD8: return XK_KP_8; case VK_NUMPAD9: return XK_KP_9;
            case VK_ADD: return XK_KP_Add; case VK_SUBTRACT: return XK_KP_Subtract;
            case VK_MULTIPLY: return XK_KP_Multiply; case VK_DIVIDE: return XK_KP_Divide;
            case VK_DECIMAL: return XK_KP_Decimal; case VK_SEPARATOR: return XK_KP_Enter;
            case VK_SNAPSHOT: return XK_Print; case VK_PAUSE: return XK_Pause;
            case VK_APPS: return XK_Menu;
            default: return NoSymbol;
        }
    }

    static bool keysymIsDown(const char keys[32], KeySym symbol) {
        if (symbol == NoSymbol) return false;
        const ::KeyCode nativeCode = XKeysymToKeycode(display, symbol);
        return nativeCode != 0 && (keys[nativeCode / 8] & (1 << (nativeCode % 8))) != 0;
    }
    
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

        if (keyCode == VK_SHIFT)
            return keysymIsDown(keys, XK_Shift_L) || keysymIsDown(keys, XK_Shift_R);
        if (keyCode == VK_CONTROL)
            return keysymIsDown(keys, XK_Control_L) || keysymIsDown(keys, XK_Control_R);
        if (keyCode == VK_MENU)
            return keysymIsDown(keys, XK_Alt_L) || keysymIsDown(keys, XK_Alt_R);
        return keysymIsDown(keys, logicalKeyToKeysym(keyCode));
    }

    bool isCommandDown() {
        return false; // Command key is macOS-only
    }

    void setKeyDownFromEvent(int, bool) {} // No-op on Linux (uses X11 polling)
    void setCommandDownFromEvent(bool) {}
    
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

    bool moveCursorRelative(int deltaX, int deltaY) {
        if (!display) return false;
        XWarpPointer(display, None, root, 0, 0, 0, 0, deltaX, deltaY);
        XFlush(display);
        return true;
    }

    bool emitMouseClick(int button) {
        if (!display) return false;
        unsigned int xButton = 0;
        switch (button) {
            case 0: xButton = Button1; break;
            case 1: xButton = Button3; break;
            case 2: xButton = Button2; break;
            default: return false;
        }

        Window rootReturn = 0, childReturn = 0;
        int rootX = 0, rootY = 0, winX = 0, winY = 0;
        unsigned int mask = 0;
        if (!XQueryPointer(display, root, &rootReturn, &childReturn,
                           &rootX, &rootY, &winX, &winY, &mask)) return false;
        const Window target = childReturn != None ? childReturn : root;
        XEvent event{};
        event.xbutton.display = display;
        event.xbutton.window = target;
        event.xbutton.root = root;
        event.xbutton.subwindow = None;
        event.xbutton.time = CurrentTime;
        event.xbutton.x = winX;
        event.xbutton.y = winY;
        event.xbutton.x_root = rootX;
        event.xbutton.y_root = rootY;
        event.xbutton.same_screen = True;
        event.xbutton.button = xButton;

        event.xbutton.type = ButtonPress;
        const Status pressed = XSendEvent(
            display, target, True, ButtonPressMask, &event);
        event.xbutton.type = ButtonRelease;
        const Status released = XSendEvent(
            display, target, True, ButtonReleaseMask, &event);
        XFlush(display);
        return pressed != 0 && released != 0;
    }
}

#elif __APPLE__
// macOS implementation — event-driven key state from NSEvent callbacks
// CGEventSourceKeyState polling removed (requires Accessibility permissions)
#include <ApplicationServices/ApplicationServices.h>
#include <Carbon/Carbon.h>

namespace PlatformInput {

    // Event-driven key state arrays, indexed by VK code (0-255)
    static bool s_eventKeyStates[256] = {};
    static bool s_eventCommandDown = false;

    // Reverse mapping: macOS keyCode (kVK_*) → Windows VK code
    static int macKeyCodeToVK(int macCode) {
        switch (macCode) {
            case kVK_Delete:         return 0x08; // VK_BACK (Backspace)
            case kVK_Tab:            return 0x09;
            case kVK_Return:         return 0x0D;
            case kVK_Escape:         return 0x1B;
            case kVK_Space:          return 0x20;
            case kVK_LeftArrow:      return 0x25;
            case kVK_UpArrow:        return 0x26;
            case kVK_RightArrow:     return 0x27;
            case kVK_DownArrow:      return 0x28;
            case kVK_ForwardDelete:  return 0x2E; // VK_DELETE
            case kVK_Home:           return 0x24;
            case kVK_End:            return 0x23;
            case kVK_PageUp:         return 0x21;
            case kVK_PageDown:       return 0x22;

            case kVK_Shift:          return 0xA0; // VK_LSHIFT
            case kVK_RightShift:     return 0xA1; // VK_RSHIFT
            case kVK_Control:        return 0xA2; // VK_LCONTROL
            case kVK_RightControl:   return 0xA3; // VK_RCONTROL
            case kVK_Option:         return 0xA4; // VK_LMENU
            case kVK_RightOption:    return 0xA5; // VK_RMENU
            case kVK_Command:        return 0xA2; // Map Cmd → LCtrl VK (for shortcuts)
            case kVK_RightCommand:   return 0xA3; // Map Right Cmd → RCtrl VK (wake key)

            case kVK_ANSI_A: return 'A'; case kVK_ANSI_B: return 'B';
            case kVK_ANSI_C: return 'C'; case kVK_ANSI_D: return 'D';
            case kVK_ANSI_E: return 'E'; case kVK_ANSI_F: return 'F';
            case kVK_ANSI_G: return 'G'; case kVK_ANSI_H: return 'H';
            case kVK_ANSI_I: return 'I'; case kVK_ANSI_J: return 'J';
            case kVK_ANSI_K: return 'K'; case kVK_ANSI_L: return 'L';
            case kVK_ANSI_M: return 'M'; case kVK_ANSI_N: return 'N';
            case kVK_ANSI_O: return 'O'; case kVK_ANSI_P: return 'P';
            case kVK_ANSI_Q: return 'Q'; case kVK_ANSI_R: return 'R';
            case kVK_ANSI_S: return 'S'; case kVK_ANSI_T: return 'T';
            case kVK_ANSI_U: return 'U'; case kVK_ANSI_V: return 'V';
            case kVK_ANSI_W: return 'W'; case kVK_ANSI_X: return 'X';
            case kVK_ANSI_Y: return 'Y'; case kVK_ANSI_Z: return 'Z';

            case kVK_ANSI_0: return '0'; case kVK_ANSI_1: return '1';
            case kVK_ANSI_2: return '2'; case kVK_ANSI_3: return '3';
            case kVK_ANSI_4: return '4'; case kVK_ANSI_5: return '5';
            case kVK_ANSI_6: return '6'; case kVK_ANSI_7: return '7';
            case kVK_ANSI_8: return '8'; case kVK_ANSI_9: return '9';

            case kVK_F1:  return 0x70; case kVK_F2:  return 0x71;
            case kVK_F3:  return 0x72; case kVK_F4:  return 0x73;
            case kVK_F5:  return 0x74; case kVK_F6:  return 0x75;
            case kVK_F7:  return 0x76; case kVK_F8:  return 0x77;
            case kVK_F9:  return 0x78; case kVK_F10: return 0x79;
            case kVK_F11: return 0x7A; case kVK_F12: return 0x7B;

            case kVK_ANSI_Grave: return 0xC0; // VK_OEM_3

            default: return -1;
        }
    }

    void initialize() {
        memset(s_eventKeyStates, 0, sizeof(s_eventKeyStates));
        s_eventCommandDown = false;
    }

    void shutdown() {}

    void setKeyDownFromEvent(int vkCode, bool down) {
        if (vkCode >= 0 && vkCode < 256)
            s_eventKeyStates[vkCode] = down;
    }

    void setCommandDownFromEvent(bool down) {
        s_eventCommandDown = down;
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
        if (keyCode == 0x10)
            return s_eventKeyStates[0xA0] || s_eventKeyStates[0xA1];
        if (keyCode == 0x11)
            return s_eventKeyStates[0xA2] || s_eventKeyStates[0xA3];
        if (keyCode == 0x12)
            return s_eventKeyStates[0xA4] || s_eventKeyStates[0xA5];
        if (keyCode >= 0 && keyCode < 256)
            return s_eventKeyStates[keyCode];
        return false;
    }

    bool isCommandDown() {
        return s_eventCommandDown;
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

    bool moveCursorRelative(int deltaX, int deltaY) {
        CGEventRef current = CGEventCreate(nullptr);
        if (!current) return false;
        CGPoint position = CGEventGetLocation(current);
        CFRelease(current);
        position.x += deltaX;
        position.y += deltaY;
        CGEventRef move = CGEventCreateMouseEvent(
            nullptr, kCGEventMouseMoved, position, kCGMouseButtonLeft);
        if (!move) return false;
        CGEventPost(kCGHIDEventTap, move);
        CFRelease(move);
        return true;
    }

    bool emitMouseClick(int button) {
        CGMouseButton mouseButton;
        CGEventType downType;
        CGEventType upType;
        switch (button) {
            case 0:
                mouseButton = kCGMouseButtonLeft;
                downType = kCGEventLeftMouseDown;
                upType = kCGEventLeftMouseUp;
                break;
            case 1:
                mouseButton = kCGMouseButtonRight;
                downType = kCGEventRightMouseDown;
                upType = kCGEventRightMouseUp;
                break;
            case 2:
                mouseButton = kCGMouseButtonCenter;
                downType = kCGEventOtherMouseDown;
                upType = kCGEventOtherMouseUp;
                break;
            default:
                return false;
        }
        CGEventRef current = CGEventCreate(nullptr);
        if (!current) return false;
        const CGPoint position = CGEventGetLocation(current);
        CFRelease(current);
        CGEventRef down = CGEventCreateMouseEvent(nullptr, downType, position, mouseButton);
        CGEventRef up = CGEventCreateMouseEvent(nullptr, upType, position, mouseButton);
        if (!down || !up) {
            if (down) CFRelease(down);
            if (up) CFRelease(up);
            return false;
        }
        CGEventPost(kCGHIDEventTap, down);
        CGEventPost(kCGHIDEventTap, up);
        CFRelease(down);
        CFRelease(up);
        return true;
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
    void setKeyDownFromEvent(int, bool) {}
    void setCommandDownFromEvent(bool) {}
    void getCursorPos(int& x, int& y) { x = y = 0; }
    void getCursorPosRelative(void*, int& x, int& y) { x = y = 0; }
    bool moveCursorRelative(int, int) { return false; }
    bool emitMouseClick(int) { return false; }
}

#endif
