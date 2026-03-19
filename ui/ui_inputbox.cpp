#include "ui_inputbox.hpp"
#include "ui_theme.hpp"
#include "ui_renderer.hpp"
#include "overlay_renderer.hpp"
#include "input_parser.hpp"
#include <chrono>
#include "helpers/mouse.hpp"
#include "helpers/key.hpp"
#include "ui_focus_manager.hpp"
#include "logger.hpp"
#include "core/grim_platform.h"

// Static guard to prevent event propagation between input boxes
static UIInputBox* g_activeInputBox = nullptr;

UIInputBox::UIInputBox(std::string* bind)
    : externalBind(bind) 
{
    // Generate unique focus ID
    focusID = UIFocusManager::getInstance().generateUniqueID();
    
    // Initialize buffer from external bind if provided
    if (externalBind && !externalBind->empty()) {
        buffer = *externalBind;
    }
}

void UIInputBox::setPlaceholder(const std::string& t) {
    placeholder = t;
}

void UIInputBox::update(const InputState& input, float dt) {
    if (!isVisible()) return;
    
    // Caret blink
#ifdef _WIN32
    uint64_t now = GetTickCount64();
#else
    auto tp = std::chrono::steady_clock::now().time_since_epoch();
    uint64_t now = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(tp).count());
#endif
    if (now - lastBlink > 500) {
        caretVisible = !caretVisible;
        lastBlink = now;
    }
    
    Vec2 m = input.mousePos;
    
    // Check if mouse is over the input box
    bool overBox = (m.x >= position.x && m.x <= position.x + size.x &&
                    m.y >= position.y && m.y <= position.y + size.y);
    
    bool leftPressed = Mouse::wasPressed(MouseButton::Left);
    
    // Handle focus change
    if (leftPressed) {
        if (overBox) {
            // Clicking on this box
            if (!focused) {
                // If another box was active, clear it first
                if (g_activeInputBox != nullptr && g_activeInputBox != this) {
                    // Commit the other box's changes
                    if (g_activeInputBox->externalBind) {
                        *(g_activeInputBox->externalBind) = g_activeInputBox->buffer;
                    }
                    g_activeInputBox->setFocused(false);  // Use base class method
                    LOG_DEBUG("UIInputBox", "Previous input box auto-committed on focus transfer");
                }
                
                // Gain focus - set as active input box
                setFocused(true);  // Use base class method
                g_activeInputBox = this;
                LOG_DEBUG("UIInputBox", "Input box gained focus");
            }
        } else {
            // Clicking outside this box
            if (focused && g_activeInputBox == this) {
                // Lose focus - commit changes and clear active guard
                setFocused(false);  // Use base class method
                g_activeInputBox = nullptr;
                if (externalBind) {
                    *externalBind = buffer;
                }
                LOG_DEBUG("UIInputBox", "Input box lost focus, committed: " + buffer);
            }
        }
    }
    
    // Only process input when this box is the active one
    if (!focused || g_activeInputBox != this) return;
    
    // Handle keyboard input
    
    // Handle Backspace
    if (Key::wasPressed(KeyCode::Backspace)) {
        if (!buffer.empty()) {
            buffer.pop_back();
        }
    }
    
    // Handle Delete (same as backspace for now)
    if (Key::wasPressed(KeyCode::Delete)) {
        if (!buffer.empty()) {
            buffer.pop_back();
        }
    }
    
    // Handle Enter - commit and unfocus
    if (Key::wasPressed(KeyCode::Enter)) {
        std::string submittedText = buffer;  // Capture before clear
        
        // Clear the buffer for next input
        buffer.clear();
        if (externalBind) {
            externalBind->clear();
        }
        
        setFocused(false);  // Use base class method
        g_activeInputBox = nullptr;  // Clear active guard
        
        // Fire the submit delegate event
        submitTextInput(submittedText);
        
        LOG_DEBUG("UIInputBox", "Enter pressed, submitted and cleared: " + submittedText);
        return;
    }
    
    // Handle Escape - cancel edit and restore previous value
    if (Key::wasPressed(KeyCode::Escape)) {
        setFocused(false);  // Use base class method
        g_activeInputBox = nullptr;  // Clear active guard
        if (externalBind) {
            buffer = *externalBind;  // Restore from external bind
        }
        LOG_DEBUG("UIInputBox", "Escape pressed, cancelled edit");
        return;
    }
    
    // Handle text input from InputState
    for (char c : input.textInput) {
        // Allow all printable ASCII characters for file paths
        if (c >= 32 && c <= 126) {
            buffer += c;
        }
    }
    
    // Sync buffer to external bind in real-time
    if (externalBind) {
        *externalBind = buffer;
    }
}

void UIInputBox::draw(UIRenderer& renderer) {
    if (!isVisible()) return;
    
    // Background
    uint32_t bgColor = focused ? 0xFF2A2A38 : 0xD9222238;
    renderer.drawRect(position, size, bgColor);
    
    // Text
    std::string display = buffer.empty() ? placeholder : buffer;
    uint32_t textColor = buffer.empty() ? 0xFF505050 : 0xFFEAEAEA;
    
    if (caretVisible && focused && !buffer.empty())
        display.push_back('|');
    else if (caretVisible && focused && buffer.empty())
        display = "|";
        
    renderer.drawText({position.x + 6, position.y + 6}, display, textColor);
}

void UIInputBox::drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos) {
    using namespace UITheme;
    if (!isVisible()) return;
    
    std::string display = buffer.empty() ? placeholder : buffer;
    uint32_t textColor = buffer.empty() ? Colors::TextDisabled : Colors::TextLight;
    
    if (caretVisible && focused) {
        display += "|";
    }
    
    int maxChars = static_cast<int>(size.x / 8) - 2;
    if (static_cast<int>(display.length()) > maxChars) {
        display = "..." + display.substr(display.length() - maxChars + 3);
    }
    
    renderer.drawText({position.x + 6, position.y + 6}, display, textColor);
}

void UIInputBox::setText(const std::string& text) {
    buffer = text;
    if (externalBind) {
        *externalBind = text;
    }
}

void UIInputBox::clear() {
    buffer.clear();
    if (externalBind) {
        externalBind->clear();
    }
}

void UIInputBox::submitTextInput(const std::string& text) {
    // Validation checks could go here
    if (text.empty()) {
        LOG_DEBUG("UIInputBox", "Skipping submit - empty text");
        return;
    }
    
    // Execute the delegate if bound
    if (OnTextSubmitted.IsBound()) {
        OnTextSubmitted.Execute(text);
        LOG_DEBUG("UIInputBox", "Text submitted via delegate: " + text);
    } else {
        LOG_DEBUG("UIInputBox", "Text submitted but no delegate bound: " + text);
    }
}
