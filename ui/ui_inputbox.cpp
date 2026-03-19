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
#include "core/platform_clipboard.hpp"
#include <algorithm>

static UIInputBox* g_activeInputBox = nullptr;

UIInputBox::UIInputBox(std::string* bind)
    : externalBind(bind) 
{
    focusID = UIFocusManager::getInstance().generateUniqueID();
    if (externalBind && !externalBind->empty()) {
        buffer = *externalBind;
    }
}

void UIInputBox::setPlaceholder(const std::string& t) {
    placeholder = t;
}

// ---------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------

int UIInputBox::charIndexAtX(float clickX) const {
    float textStartX = position.x + 6.0f;
    float relX = clickX - textStartX;
    if (relX <= 0.0f) return 0;

    // Account for display truncation (leading "..." when text overflows)
    int maxChars = static_cast<int>(size.x / kCharWidth) - 2;
    int displayOffset = 0;
    if (static_cast<int>(buffer.length()) > maxChars && maxChars > 3) {
        displayOffset = static_cast<int>(buffer.length()) - maxChars + 3;
    }

    int idx = static_cast<int>(relX / kCharWidth);
    if (displayOffset > 0) {
        idx += displayOffset; // map from display space to buffer space
    }
    return std::clamp(idx, 0, static_cast<int>(buffer.length()));
}

void UIInputBox::deleteSelection() {
    if (!hasSelection()) return;
    int lo = std::min(selStart, selEnd);
    int hi = std::max(selStart, selEnd);
    buffer.erase(lo, hi - lo);
    cursorPos = lo;
    clearSelection();
}

std::string UIInputBox::selectedText() const {
    if (!hasSelection()) return "";
    int lo = std::min(selStart, selEnd);
    int hi = std::max(selStart, selEnd);
    return buffer.substr(lo, hi - lo);
}

// ---------------------------------------------------------------
// Update
// ---------------------------------------------------------------

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
    bool overBox = (m.x >= position.x && m.x <= position.x + size.x &&
                    m.y >= position.y && m.y <= position.y + size.y);

    bool leftPressed = Mouse::wasPressed(MouseButton::Left);

    // -------------------------------------------------------
    // Focus management
    // -------------------------------------------------------
    if (leftPressed) {
        if (overBox) {
            if (!focused) {
                if (g_activeInputBox != nullptr && g_activeInputBox != this) {
                    if (g_activeInputBox->externalBind)
                        *(g_activeInputBox->externalBind) = g_activeInputBox->buffer;
                    g_activeInputBox->setFocused(false);
                }
                setFocused(true);
                g_activeInputBox = this;
            }
            // Click-to-position cursor
            cursorPos = charIndexAtX(m.x);
            if (input.shift) {
                selEnd = cursorPos;
            } else {
                clearSelection();
            }
            caretVisible = true;
            lastBlink = now;
        } else {
            if (focused && g_activeInputBox == this) {
                setFocused(false);
                g_activeInputBox = nullptr;
                if (externalBind) *externalBind = buffer;
            }
        }
    }

    if (!focused || g_activeInputBox != this) return;

    // -------------------------------------------------------
    // Key repeat logic
    // -------------------------------------------------------
    // We track a "held" action key (Backspace, Delete, Left, Right)
    // and fire repeats after an initial delay.

    auto isActionKeyDown = [&](KeyCode kc) -> bool { return Key::isDown(kc); };

    auto processActionKey = [&](KeyCode kc, int vk) {
        if (Key::wasPressed(kc)) {
            heldKeyVK = vk;
            heldKeyTimer = 0.0f;
            repeatFiring = false;
            return true; // fire once immediately
        }
        if (heldKeyVK == vk && isActionKeyDown(kc)) {
            heldKeyTimer += dt;
            if (!repeatFiring && heldKeyTimer >= repeatDelay) {
                repeatFiring = true;
                heldKeyTimer = 0.0f;
                return true;
            }
            if (repeatFiring && heldKeyTimer >= repeatRate) {
                heldKeyTimer = 0.0f;
                return true;
            }
        }
        return false;
    };

    // Release tracking
    if (heldKeyVK >= 0) {
        KeyCode heldKC = KeyCode::Unknown;
        switch (heldKeyVK) {
            case VK_BACK:   heldKC = KeyCode::Backspace; break;
            case VK_DELETE: heldKC = KeyCode::Delete; break;
            case VK_LEFT:   heldKC = KeyCode::Left; break;
            case VK_RIGHT:  heldKC = KeyCode::Right; break;
            default: break;
        }
        if (heldKC != KeyCode::Unknown && !isActionKeyDown(heldKC)) {
            heldKeyVK = -1;
            heldKeyTimer = 0.0f;
            repeatFiring = false;
        }
    }

    // -------------------------------------------------------
    // Backspace
    // -------------------------------------------------------
    if (processActionKey(KeyCode::Backspace, VK_BACK)) {
        if (hasSelection()) {
            deleteSelection();
        } else if (cursorPos > 0) {
            buffer.erase(cursorPos - 1, 1);
            cursorPos--;
            clearSelection();
        }
        caretVisible = true;
        lastBlink = now;
    }

    // -------------------------------------------------------
    // Delete
    // -------------------------------------------------------
    if (processActionKey(KeyCode::Delete, VK_DELETE)) {
        if (hasSelection()) {
            deleteSelection();
        } else if (cursorPos < static_cast<int>(buffer.length())) {
            buffer.erase(cursorPos, 1);
            clearSelection();
        }
        caretVisible = true;
        lastBlink = now;
    }

    // -------------------------------------------------------
    // Arrow keys (with shift for selection, with repeat)
    // -------------------------------------------------------
    if (processActionKey(KeyCode::Left, VK_LEFT)) {
        if (cursorPos > 0) cursorPos--;
        if (input.shift) {
            selEnd = cursorPos;
        } else {
            clearSelection();
        }
        caretVisible = true;
        lastBlink = now;
    }

    if (processActionKey(KeyCode::Right, VK_RIGHT)) {
        if (cursorPos < static_cast<int>(buffer.length())) cursorPos++;
        if (input.shift) {
            selEnd = cursorPos;
        } else {
            clearSelection();
        }
        caretVisible = true;
        lastBlink = now;
    }

    // Home / End
    if (Key::wasPressed(KeyCode::Home)) {
        cursorPos = 0;
        if (input.shift) selEnd = cursorPos;
        else clearSelection();
    }
    if (Key::wasPressed(KeyCode::End)) {
        cursorPos = static_cast<int>(buffer.length());
        if (input.shift) selEnd = cursorPos;
        else clearSelection();
    }

    // -------------------------------------------------------
    // Select All (Ctrl+A / Cmd+A)
    // -------------------------------------------------------
    if (input.ctrl && Key::wasPressed(KeyCode::A)) {
        selStart = 0;
        selEnd = static_cast<int>(buffer.length());
        cursorPos = selEnd;
    }

    // -------------------------------------------------------
    // Clipboard (Ctrl+C / Ctrl+V / Ctrl+X)
    // -------------------------------------------------------
    if (input.copyRequested) {
        if (hasSelection()) {
            PlatformClipboard::copyText(selectedText());
        }
    }

    if (input.cutRequested) {
        if (hasSelection()) {
            PlatformClipboard::copyText(selectedText());
            deleteSelection();
        }
    }

    if (input.pasteRequested && !input.pastedText.empty()) {
        if (hasSelection()) deleteSelection();
        // Filter to printable ASCII
        std::string filtered;
        for (char c : input.pastedText) {
            if (c >= 32 && c <= 126) filtered += c;
        }
        buffer.insert(cursorPos, filtered);
        cursorPos += static_cast<int>(filtered.length());
        clearSelection();
    }

    // -------------------------------------------------------
    // Enter - commit and unfocus
    // -------------------------------------------------------
    if (Key::wasPressed(KeyCode::Enter)) {
        std::string submittedText = buffer;
        buffer.clear();
        cursorPos = 0;
        clearSelection();
        if (externalBind) externalBind->clear();

        setFocused(false);
        g_activeInputBox = nullptr;
        submitTextInput(submittedText);
        return;
    }

    // -------------------------------------------------------
    // Escape - cancel edit
    // -------------------------------------------------------
    if (Key::wasPressed(KeyCode::Escape)) {
        setFocused(false);
        g_activeInputBox = nullptr;
        if (externalBind) buffer = *externalBind;
        cursorPos = static_cast<int>(buffer.length());
        clearSelection();
        return;
    }

    // -------------------------------------------------------
    // Text input from platform (WM_CHAR / macOS text callback)
    // -------------------------------------------------------
    for (char c : input.textInput) {
        if (c >= 32 && c <= 126) {
            if (hasSelection()) deleteSelection();
            buffer.insert(buffer.begin() + cursorPos, c);
            cursorPos++;
            clearSelection();
            caretVisible = true;
            lastBlink = now;
        }
    }

    // Sync to external bind
    if (externalBind) *externalBind = buffer;
}

// ---------------------------------------------------------------
// Draw (legacy UIRenderer path)
// ---------------------------------------------------------------

void UIInputBox::draw(UIRenderer& renderer) {
    if (!isVisible()) return;
    
    uint32_t bgColor = focused ? 0xFF2A2A38 : 0xD9222238;
    renderer.drawRect(position, size, bgColor);
    
    std::string display = buffer.empty() ? placeholder : buffer;
    uint32_t textColor = buffer.empty() ? 0xFF505050 : 0xFFEAEAEA;
    
    if (caretVisible && focused && !buffer.empty())
        display.push_back('|');
    else if (caretVisible && focused && buffer.empty())
        display = "|";
        
    renderer.drawText({position.x + 6, position.y + 6}, display, textColor);
}

// ---------------------------------------------------------------
// drawOverlay (OverlayRenderer path with cursor + selection)
// ---------------------------------------------------------------

void UIInputBox::drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos) {
    using namespace UITheme;
    if (!isVisible()) return;

    int maxChars = static_cast<int>(size.x / kCharWidth) - 2;
    if (maxChars < 1) maxChars = 1;

    // Build display string, handling overflow with leading "..."
    std::string display;
    int displayOffset = 0; // first buffer index shown
    if (buffer.empty()) {
        display = placeholder;
    } else if (static_cast<int>(buffer.length()) > maxChars && maxChars > 3) {
        displayOffset = static_cast<int>(buffer.length()) - maxChars + 3;
        display = "..." + buffer.substr(displayOffset);
    } else {
        display = buffer;
    }

    uint32_t textColor = buffer.empty() ? Colors::TextDisabled : Colors::TextLight;

    // Draw selection highlight
    if (focused && hasSelection() && !buffer.empty()) {
        int lo = std::min(selStart, selEnd);
        int hi = std::max(selStart, selEnd);
        // Map buffer indices to display coordinates
        int dispLo = std::max(0, lo - displayOffset);
        int dispHi = std::min(static_cast<int>(display.length()), hi - displayOffset);
        if (displayOffset > 0) {
            dispLo = std::max(3, dispLo + 3);
            dispHi = std::min(static_cast<int>(display.length()), dispHi + 3);
        }
        if (dispHi > dispLo) {
            float selX = position.x + 6.0f + dispLo * kCharWidth;
            float selW = (dispHi - dispLo) * kCharWidth;
            renderer.drawRect({selX, position.y + 3.0f}, {selW, size.y - 6.0f}, 0x604488CC);
        }
    }

    renderer.drawText({position.x + 6, position.y + 6}, display, textColor);

    // Draw caret
    if (focused && caretVisible && !buffer.empty()) {
        int dispCursorPos = cursorPos - displayOffset;
        if (displayOffset > 0) dispCursorPos += 3;
        dispCursorPos = std::clamp(dispCursorPos, 0, static_cast<int>(display.length()));
        float caretX = position.x + 6.0f + dispCursorPos * kCharWidth;
        renderer.drawRect({caretX, position.y + 4.0f}, {1.5f, size.y - 8.0f}, Colors::TextLight);
    } else if (focused && caretVisible && buffer.empty()) {
        renderer.drawRect({position.x + 6.0f, position.y + 4.0f}, {1.5f, size.y - 8.0f}, Colors::TextLight);
    }
}

// ---------------------------------------------------------------
// Public API
// ---------------------------------------------------------------

void UIInputBox::setText(const std::string& text) {
    buffer = text;
    cursorPos = static_cast<int>(buffer.length());
    clearSelection();
    if (externalBind) *externalBind = text;
}

void UIInputBox::clear() {
    buffer.clear();
    cursorPos = 0;
    clearSelection();
    if (externalBind) externalBind->clear();
}

void UIInputBox::submitTextInput(const std::string& text) {
    if (text.empty()) return;
    if (OnTextSubmitted.IsBound()) {
        OnTextSubmitted.Execute(text);
    }
}
