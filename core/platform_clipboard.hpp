#pragma once
#include <string>

// Cross-platform clipboard abstraction
// Provides copy/paste functionality for text across Windows, Linux, and macOS

namespace PlatformClipboard {
    
    // Initialize platform-specific clipboard system
    // Must be called before using clipboard functions
    void initialize();
    
    // Shutdown platform-specific clipboard system
    void shutdown();
    
    // Copy text to clipboard
    // Returns true on success, false on failure
    bool copyText(const std::string& text);
    
    // Get text from clipboard
    // Returns clipboard contents, or empty string if clipboard is empty or error occurs
    std::string getText();
    
    // Check if clipboard contains text data
    bool hasText();
    
    // Clear clipboard contents
    void clear();
    
    // Clipboard format types (for future expansion)
    enum class Format {
        Text,           // Plain text (UTF-8)
        RichText,       // Rich text format (HTML/RTF)
        Image,          // Image data (future)
        Files           // File paths (future)
    };
    
    // Check if clipboard contains specific format (future expansion)
    bool hasFormat(Format format);
}
