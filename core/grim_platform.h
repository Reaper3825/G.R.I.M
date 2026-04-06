#pragma once

// =============================================================================
// G.R.I.M Platform Compatibility Layer
// =============================================================================
// Provides Windows type stubs on non-Windows platforms so headers that use
// HWND, HANDLE, etc. in interfaces can compile. Actual Windows API usage
// remains inside #ifdef _WIN32 blocks in .cpp files.
// =============================================================================

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif
#else

#include <cstddef>
#include <cstdint>

#ifdef __OBJC__
#import <objc/objc.h>
#endif

// Window/display types
typedef void* HWND;
typedef void* HDC;
typedef void* HGDIOBJ;
typedef void* HBITMAP;
typedef void* HINSTANCE;
typedef void* HMODULE;
typedef void* HANDLE;

// Simple types
typedef unsigned long DWORD;
typedef unsigned short WORD;
typedef unsigned int UINT;
#ifndef __OBJC__
typedef int BOOL;
#endif
typedef unsigned char BYTE;
typedef long LONG;
typedef unsigned long ULONG;
typedef void* LPVOID;
typedef const void* LPCVOID;
typedef unsigned long long ULONG_PTR;
typedef long long LRESULT;
typedef unsigned long long WPARAM;
typedef long long LPARAM;
typedef void* HHOOK;
typedef wchar_t WCHAR;
#define CALLBACK

// Constants
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

#ifndef WM_APP
#define WM_APP 0x8000
#endif

// Minimal PROCESS_INFORMATION for struct layout when not used
struct PROCESS_INFORMATION {
    void* hProcess;
    void* hThread;
    unsigned long dwProcessId;
    unsigned long dwThreadId;
};

// Minimal STARTUPINFO (not used on non-Windows, but some headers may reference)
struct STARTUPINFOA {
    unsigned long cb;
    char* lpReserved;
    char* lpDesktop;
    char* lpTitle;
    unsigned long dwX;
    unsigned long dwY;
    unsigned long dwXSize;
    unsigned long dwYSize;
    unsigned long dwXCountChars;
    unsigned long dwYCountChars;
    unsigned long dwFillAttribute;
    unsigned long dwFlags;
    unsigned short wShowWindow;
    unsigned short cbReserved2;
    unsigned char* lpReserved2;
    void* hStdInput;
    void* hStdOutput;
    void* hStdError;
};

// Virtual key codes (for key.cpp cross-platform)
#define VK_F1        0x70
#define VK_F2        0x71
#define VK_F3        0x72
#define VK_F4        0x73
#define VK_F5        0x74
#define VK_F6        0x75
#define VK_F7        0x76
#define VK_F8        0x77
#define VK_F9        0x78
#define VK_F10       0x79
#define VK_F11       0x7A
#define VK_F12       0x7B
#define VK_SHIFT     0x10
#define VK_CONTROL   0x11
#define VK_MENU      0x12
#define VK_LSHIFT    0xA0
#define VK_RSHIFT    0xA1
#define VK_LCONTROL  0xA2
#define VK_RCONTROL  0xA3
#define VK_LMENU     0xA4
#define VK_RMENU     0xA5
#define VK_LWIN      0x5B
#define VK_RWIN      0x5C
#define VK_CAPITAL   0x14
#define VK_NUMLOCK   0x90
#define VK_SCROLL    0x91
#define VK_RETURN    0x0D
#define VK_ESCAPE    0x1B
#define VK_SPACE     0x20
#define VK_BACK      0x08
#define VK_TAB       0x09
#define VK_INSERT    0x2D
#define VK_DELETE    0x2E
#define VK_HOME      0x24
#define VK_END       0x23
#define VK_PRIOR     0x21
#define VK_NEXT      0x22
#define VK_LEFT      0x25
#define VK_UP        0x26
#define VK_RIGHT     0x27
#define VK_DOWN      0x28
#define VK_OEM_MINUS  0xBD
#define VK_OEM_PLUS   0xBB
#define VK_OEM_4      0xDB
#define VK_OEM_6      0xDD
#define VK_OEM_5      0xDC
#define VK_OEM_1      0xBA
#define VK_OEM_7      0xDE
#define VK_OEM_COMMA  0xBC
#define VK_OEM_PERIOD 0xBE
#define VK_OEM_2      0xBF
#define VK_OEM_3      0xC0
#define VK_NUMPAD0    0x60
#define VK_NUMPAD1    0x61
#define VK_NUMPAD2    0x62
#define VK_NUMPAD3    0x63
#define VK_NUMPAD4    0x64
#define VK_NUMPAD5    0x65
#define VK_NUMPAD6    0x66
#define VK_NUMPAD7    0x67
#define VK_NUMPAD8    0x68
#define VK_NUMPAD9    0x69
#define VK_MULTIPLY   0x6A
#define VK_ADD        0x6B
#define VK_SEPARATOR  0x6C
#define VK_SUBTRACT   0x6D
#define VK_DECIMAL    0x6E
#define VK_DIVIDE     0x6F
#define VK_SNAPSHOT   0x2C
#define VK_PAUSE      0x13
#define VK_APPS       0x5D

// POINT for mouse position (MouseState)
struct POINT {
    LONG x;
    LONG y;
};

#endif
