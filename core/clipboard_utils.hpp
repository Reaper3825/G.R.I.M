#pragma once
#include <string>
#include "platform_clipboard.hpp"

// High-level clipboard utilities for UI components
// Provides convenient helpers for copy/paste operations

namespace ClipboardUtils {
    
    // Initialize clipboard system (call once at startup)
    inline void initialize() {
        PlatformClipboard::initialize();
    }
    
    // Shutdown clipboard system (call once at shutdown)
    inline void shutdown() {
        PlatformClipboard::shutdown();
    }
    
    // Copy text to clipboard with feedback
    // Returns true on success
    inline bool copy(const std::string& text) {
        if (text.empty()) {
            return false;
        }
        return PlatformClipboard::copyText(text);
    }
    
    // Paste text from clipboard
    // Returns clipboard contents or empty string if unavailable
    inline std::string paste() {
        return PlatformClipboard::getText();
    }
    
    // Cut operation helper: copy text and return true to signal deletion
    // UI component should delete the text if this returns true
    inline bool cut(const std::string& text) {
        return copy(text);
    }
    
    // Check if paste operation is available
    inline bool canPaste() {
        return PlatformClipboard::hasText();
    }
    
    // Copy with line normalization (remove trailing whitespace, ensure single LF)
    inline bool copyNormalized(const std::string& text) {
        std::string normalized = text;
        
        // Remove trailing whitespace from each line
        size_t pos = 0;
        while (pos < normalized.length()) {
            size_t lineEnd = normalized.find('\n', pos);
            if (lineEnd == std::string::npos) {
                lineEnd = normalized.length();
            }
            
            // Trim trailing spaces before newline
            while (lineEnd > pos && normalized[lineEnd - 1] == ' ') {
                normalized.erase(lineEnd - 1, 1);
                lineEnd--;
            }
            
            pos = (lineEnd < normalized.length()) ? lineEnd + 1 : lineEnd;
        }
        
        return copy(normalized);
    }
    
    // Paste with automatic line ending conversion (CRLF -> LF)
    inline std::string pasteNormalized() {
        std::string text = paste();
        
        // Convert CRLF to LF
        size_t pos = 0;
        while ((pos = text.find("\r\n", pos)) != std::string::npos) {
            text.replace(pos, 2, "\n");
            pos++;
        }
        
        // Remove standalone CR
        pos = 0;
        while ((pos = text.find('\r', pos)) != std::string::npos) {
            text.erase(pos, 1);
        }
        
        return text;
    }
    
    // Copy multiple lines as a single string
    inline bool copyLines(const std::vector<std::string>& lines, const std::string& separator = "\n") {
        if (lines.empty()) {
            return false;
        }
        
        std::string combined;
        for (size_t i = 0; i < lines.size(); ++i) {
            combined += lines[i];
            if (i < lines.size() - 1) {
                combined += separator;
            }
        }
        
        return copy(combined);
    }
    
    // Paste and split into lines
    inline std::vector<std::string> pasteLines() {
        std::string text = pasteNormalized();
        std::vector<std::string> lines;
        
        if (text.empty()) {
            return lines;
        }
        
        size_t pos = 0;
        size_t lastPos = 0;
        
        while ((pos = text.find('\n', lastPos)) != std::string::npos) {
            lines.push_back(text.substr(lastPos, pos - lastPos));
            lastPos = pos + 1;
        }
        
        // Don't forget last line if no trailing newline
        if (lastPos < text.length()) {
            lines.push_back(text.substr(lastPos));
        }
        
        return lines;
    }
    
    // Clear clipboard (useful for sensitive data)
    inline void clear() {
        PlatformClipboard::clear();
    }
}
