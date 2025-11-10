#include "platform_clipboard.hpp"
#include <cstring>

#ifdef _WIN32
#include <windows.h>

namespace PlatformClipboard {
    
    void initialize() {
        // No initialization needed for Windows clipboard API
    }
    
    void shutdown() {
        // No cleanup needed for Windows clipboard API
    }
    
    bool copyText(const std::string& text) {
        if (text.empty()) {
            return false;
        }
        
        // Open clipboard
        if (!OpenClipboard(nullptr)) {
            return false;
        }
        
        // Empty existing clipboard contents
        if (!EmptyClipboard()) {
            CloseClipboard();
            return false;
        }
        
        // Allocate global memory for text (including null terminator)
        size_t size = text.size() + 1;
        HGLOBAL hMem = GlobalAlloc(GMEM_MOVEABLE, size);
        if (!hMem) {
            CloseClipboard();
            return false;
        }
        
        // Lock memory and copy text
        char* pMem = static_cast<char*>(GlobalLock(hMem));
        if (!pMem) {
            GlobalFree(hMem);
            CloseClipboard();
            return false;
        }
        
        memcpy(pMem, text.c_str(), size);
        GlobalUnlock(hMem);
        
        // Set clipboard data
        if (!SetClipboardData(CF_TEXT, hMem)) {
            GlobalFree(hMem);
            CloseClipboard();
            return false;
        }
        
        // Close clipboard (ownership transfers to system)
        CloseClipboard();
        return true;
    }
    
    std::string getText() {
        // Open clipboard
        if (!OpenClipboard(nullptr)) {
            return "";
        }
        
        // Check if clipboard has text
        if (!IsClipboardFormatAvailable(CF_TEXT)) {
            CloseClipboard();
            return "";
        }
        
        // Get clipboard data handle
        HANDLE hData = GetClipboardData(CF_TEXT);
        if (!hData) {
            CloseClipboard();
            return "";
        }
        
        // Lock memory and get text
        char* pText = static_cast<char*>(GlobalLock(hData));
        if (!pText) {
            CloseClipboard();
            return "";
        }
        
        std::string result(pText);
        GlobalUnlock(hData);
        CloseClipboard();
        
        return result;
    }
    
    bool hasText() {
        return IsClipboardFormatAvailable(CF_TEXT) != 0;
    }
    
    void clear() {
        if (OpenClipboard(nullptr)) {
            EmptyClipboard();
            CloseClipboard();
        }
    }
    
    bool hasFormat(Format format) {
        switch (format) {
            case Format::Text:
                return IsClipboardFormatAvailable(CF_TEXT) != 0;
            case Format::RichText:
                return IsClipboardFormatAvailable(CF_UNICODETEXT) != 0;
            default:
                return false;
        }
    }
}

#elif __linux__
// Linux implementation using X11
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <vector>

namespace PlatformClipboard {
    static Display* display = nullptr;
    static Window window;
    static Atom clipboard_atom;
    static Atom utf8_atom;
    static Atom targets_atom;
    static std::string clipboard_cache;
    
    void initialize() {
        display = XOpenDisplay(nullptr);
        if (!display) {
            return;
        }
        
        // Create invisible window for clipboard operations
        int screen = DefaultScreen(display);
        window = XCreateSimpleWindow(display, RootWindow(display, screen),
                                     0, 0, 1, 1, 0, 0, 0);
        
        // Get clipboard atoms
        clipboard_atom = XInternAtom(display, "CLIPBOARD", False);
        utf8_atom = XInternAtom(display, "UTF8_STRING", False);
        targets_atom = XInternAtom(display, "TARGETS", False);
    }
    
    void shutdown() {
        if (display) {
            if (window) {
                XDestroyWindow(display, window);
            }
            XCloseDisplay(display);
            display = nullptr;
        }
    }
    
    bool copyText(const std::string& text) {
        if (!display || text.empty()) {
            return false;
        }
        
        // Cache the text
        clipboard_cache = text;
        
        // Take ownership of clipboard
        XSetSelectionOwner(display, clipboard_atom, window, CurrentTime);
        
        // Verify ownership
        if (XGetSelectionOwner(display, clipboard_atom) != window) {
            return false;
        }
        
        XFlush(display);
        return true;
    }
    
    std::string getText() {
        if (!display) {
            return "";
        }
        
        // Check if we own the clipboard
        if (XGetSelectionOwner(display, clipboard_atom) == window) {
            return clipboard_cache;
        }
        
        // Request clipboard content
        Atom property = XInternAtom(display, "CLIPBOARD_CONTENT", False);
        XConvertSelection(display, clipboard_atom, utf8_atom, property, window, CurrentTime);
        XFlush(display);
        
        // Wait for SelectionNotify event (with timeout)
        XEvent event;
        int timeout = 100; // 100ms timeout
        while (timeout-- > 0) {
            if (XCheckTypedWindowEvent(display, window, SelectionNotify, &event)) {
                if (event.xselection.property == None) {
                    return "";
                }
                
                // Read the property
                Atom actual_type;
                int actual_format;
                unsigned long nitems, bytes_after;
                unsigned char* data = nullptr;
                
                XGetWindowProperty(display, window, property, 0, ~0L, False,
                                 AnyPropertyType, &actual_type, &actual_format,
                                 &nitems, &bytes_after, &data);
                
                if (data) {
                    std::string result(reinterpret_cast<char*>(data), nitems);
                    XFree(data);
                    XDeleteProperty(display, window, property);
                    return result;
                }
                break;
            }
            usleep(1000); // 1ms sleep
        }
        
        return "";
    }
    
    bool hasText() {
        if (!display) {
            return false;
        }
        
        // Check if we own the clipboard
        if (XGetSelectionOwner(display, clipboard_atom) == window) {
            return !clipboard_cache.empty();
        }
        
        // Check if someone else owns it
        return XGetSelectionOwner(display, clipboard_atom) != None;
    }
    
    void clear() {
        if (display) {
            clipboard_cache.clear();
            XSetSelectionOwner(display, clipboard_atom, None, CurrentTime);
            XFlush(display);
        }
    }
    
    bool hasFormat(Format format) {
        switch (format) {
            case Format::Text:
                return hasText();
            default:
                return false;
        }
    }
}

#elif __APPLE__
// macOS implementation using Cocoa/AppKit
#include <ApplicationServices/ApplicationServices.h>
#include <CoreFoundation/CoreFoundation.h>

namespace PlatformClipboard {
    
    void initialize() {
        // No initialization needed for macOS pasteboard API
    }
    
    void shutdown() {
        // No cleanup needed for macOS pasteboard API
    }
    
    bool copyText(const std::string& text) {
        if (text.empty()) {
            return false;
        }
        
        // Get general pasteboard
        PasteboardRef pasteboard;
        if (PasteboardCreate(kPasteboardClipboard, &pasteboard) != noErr) {
            return false;
        }
        
        // Clear existing contents
        if (PasteboardClear(pasteboard) != noErr) {
            CFRelease(pasteboard);
            return false;
        }
        
        // Create CFString from text
        CFStringRef cfText = CFStringCreateWithCString(kCFAllocatorDefault, 
                                                       text.c_str(), 
                                                       kCFStringEncodingUTF8);
        if (!cfText) {
            CFRelease(pasteboard);
            return false;
        }
        
        // Create CFData from CFString
        CFDataRef cfData = CFStringCreateExternalRepresentation(kCFAllocatorDefault,
                                                                cfText,
                                                                kCFStringEncodingUTF8,
                                                                0);
        CFRelease(cfText);
        
        if (!cfData) {
            CFRelease(pasteboard);
            return false;
        }
        
        // Put data to pasteboard
        OSStatus status = PasteboardPutItemFlavor(pasteboard, 
                                                  (PasteboardItemID)1,
                                                  CFSTR("public.utf8-plain-text"),
                                                  cfData,
                                                  0);
        
        CFRelease(cfData);
        CFRelease(pasteboard);
        
        return status == noErr;
    }
    
    std::string getText() {
        // Get general pasteboard
        PasteboardRef pasteboard;
        if (PasteboardCreate(kPasteboardClipboard, &pasteboard) != noErr) {
            return "";
        }
        
        // Synchronize to get latest contents
        PasteboardSynchronize(pasteboard);
        
        // Get item count
        ItemCount itemCount;
        if (PasteboardGetItemCount(pasteboard, &itemCount) != noErr || itemCount < 1) {
            CFRelease(pasteboard);
            return "";
        }
        
        // Get first item
        PasteboardItemID itemID;
        if (PasteboardGetItemIdentifier(pasteboard, 1, &itemID) != noErr) {
            CFRelease(pasteboard);
            return "";
        }
        
        // Get text flavor data
        CFDataRef cfData;
        if (PasteboardCopyItemFlavorData(pasteboard, itemID, 
                                        CFSTR("public.utf8-plain-text"), 
                                        &cfData) != noErr) {
            CFRelease(pasteboard);
            return "";
        }
        
        // Convert CFData to string
        CFIndex length = CFDataGetLength(cfData);
        const UInt8* bytes = CFDataGetBytePtr(cfData);
        std::string result(reinterpret_cast<const char*>(bytes), length);
        
        CFRelease(cfData);
        CFRelease(pasteboard);
        
        return result;
    }
    
    bool hasText() {
        // Get general pasteboard
        PasteboardRef pasteboard;
        if (PasteboardCreate(kPasteboardClipboard, &pasteboard) != noErr) {
            return false;
        }
        
        // Synchronize
        PasteboardSynchronize(pasteboard);
        
        // Check item count
        ItemCount itemCount;
        if (PasteboardGetItemCount(pasteboard, &itemCount) != noErr) {
            CFRelease(pasteboard);
            return false;
        }
        
        bool hasText = itemCount > 0;
        CFRelease(pasteboard);
        
        return hasText;
    }
    
    void clear() {
        PasteboardRef pasteboard;
        if (PasteboardCreate(kPasteboardClipboard, &pasteboard) == noErr) {
            PasteboardClear(pasteboard);
            CFRelease(pasteboard);
        }
    }
    
    bool hasFormat(Format format) {
        switch (format) {
            case Format::Text:
                return hasText();
            default:
                return false;
        }
    }
}

#else
// Fallback implementation for unsupported platforms
namespace PlatformClipboard {
    static std::string clipboard_fallback;
    
    void initialize() {}
    void shutdown() {}
    
    bool copyText(const std::string& text) {
        clipboard_fallback = text;
        return true;
    }
    
    std::string getText() {
        return clipboard_fallback;
    }
    
    bool hasText() {
        return !clipboard_fallback.empty();
    }
    
    void clear() {
        clipboard_fallback.clear();
    }
    
    bool hasFormat(Format format) {
        return format == Format::Text && hasText();
    }
}

#endif
