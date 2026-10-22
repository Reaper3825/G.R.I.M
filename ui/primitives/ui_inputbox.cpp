#include "helpers/utf8_edit.hpp"
#include "ui_inputbox.hpp"
#include "../ui_theme.hpp"
#include "../ui_renderer.hpp"
#include "../overlay_renderer.hpp"
#include "../../core/input_parser.hpp"
#include <chrono>
#include "helpers/mouse.hpp"
#include "helpers/key.hpp"
#include "../ui_focus_manager.hpp"
#include "../ui_root.hpp"
#include "logger.hpp"
#include "core/grim_platform.h"
#include "core/platform_clipboard.hpp"
#include <algorithm>



namespace {
constexpr float kTextPaddingX = 6.0f;
constexpr const char* kOverflowPrefix = "...";

struct InputBoxDisplayLayout {
    std::string prefix;
    std::string display;
    int displayOffset = 0;
};

InputBoxDisplayLayout buildDisplayLayout(const std::string& buffer,
                                         float boxWidth,
                                         const OverlayRenderer& renderer)
{
    InputBoxDisplayLayout layout;
    if (buffer.empty()) return layout;

    const float availableWidth = std::max(0.0f, boxWidth - (kTextPaddingX * 2.0f));
    if (availableWidth <= 0.0f) {
        layout.displayOffset = static_cast<int>(buffer.length());
        return layout;
    }

    if (renderer.measureTextWidth(buffer) <= availableWidth) {
        layout.display = buffer;
        return layout;
    }

    const float ellipsisWidth = renderer.measureTextWidth(kOverflowPrefix);
    if (ellipsisWidth <= availableWidth) {
        layout.prefix = kOverflowPrefix;
    }

    float visibleWidth = layout.prefix.empty() ? 0.0f : ellipsisWidth;
    int suffixStart = static_cast<int>(buffer.length());

    for (int end = static_cast<int>(buffer.length()); end > 0;) {
        const int i = UTF8Edit::previous(buffer, end);
        const float charWidth = renderer.measureTextWidth(buffer.substr(i, end - i));
        end = i;
        if (visibleWidth + charWidth > availableWidth) {
            break;
        }

        visibleWidth += charWidth;
        suffixStart = i;
    }

    layout.displayOffset = suffixStart;
    layout.display = layout.prefix + buffer.substr(static_cast<size_t>(layout.displayOffset));
    return layout;
}

float measureDisplayPrefixWidth(const OverlayRenderer& renderer,
                                const InputBoxDisplayLayout& layout,
                                int displayIndex)
{
    const int clampedIndex = std::clamp(displayIndex, 0, static_cast<int>(layout.display.length()));
    return renderer.measureTextWidth(layout.display.substr(0, static_cast<size_t>(clampedIndex)));
}
}

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
    if (buffer.empty()) return 0;

    const OverlayRenderer& renderer = UIRoot::get().getRenderer();
    const InputBoxDisplayLayout layout = buildDisplayLayout(buffer, size.x, renderer);

    float textStartX = position.x + kTextPaddingX;
    float relX = clickX - textStartX;
    if (relX <= 0.0f) return 0;

    const std::string visibleBuffer = buffer.substr(static_cast<size_t>(layout.displayOffset));
    float prevCaretX = layout.prefix.empty() ? 0.0f : renderer.measureTextWidth(layout.prefix);

    for (int i = 0; i < static_cast<int>(visibleBuffer.length()); i = UTF8Edit::next(visibleBuffer, i)) {
        const float nextCaretX = renderer.measureTextWidth(
            layout.prefix + visibleBuffer.substr(0, static_cast<size_t>(UTF8Edit::next(visibleBuffer, i))));
        const float midpoint = prevCaretX + ((nextCaretX - prevCaretX) * 0.5f);

        if (relX < midpoint) {
            return layout.displayOffset + i;
        }

        prevCaretX = nextCaretX;
    }

    return static_cast<int>(buffer.length());
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

void UIInputBox::update(const InputState& source, float dt) {
    const auto input = routeInput(source);
    if (!hasPointerCapture()) dragging = false;
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

    // -------------------------------------------------------
    // Unified mouse: click-to-cursor + drag-to-select
    // Uses InputState directly (no Mouse:: static class mixing)
    // -------------------------------------------------------
    if (input.mousePressed[0]) {
        if (overBox) {
            if (!focused) {
                setFocused(true);

            }
            cursorPos = charIndexAtX(m.x);
            if (input.shift) {
                selEnd = cursorPos;
            } else {
                selStart = cursorPos;
                selEnd = cursorPos;
            }
            dragging = capturePointer();
            caretVisible = true;
            lastBlink = now;
        } else {
            if (focused) {
                setFocused(false);

                if (externalBind) *externalBind = buffer;
            }
            dragging = false;
            releasePointer();
        }
    } else if (dragging) {
        if (input.mouseDown[0]) {
            int dragPos = charIndexAtX(m.x);
            if (dragPos != selEnd) {
                selEnd = dragPos;
                cursorPos = dragPos;
                caretVisible = true;
                lastBlink = now;
            }
        } else {
            dragging = false;
            releasePointer();
        }
    }

    if (!focused) return;

    // While mouse is actively dragging, skip keyboard input so
    // arrow keys / text input don't clobber the drag selection.
    if (dragging) return;

    // -------------------------------------------------------
    // Key repeat logic
    // -------------------------------------------------------
    // We track a "held" action key (Backspace, Delete, Left, Right)
    // and fire repeats after an initial delay.

    auto isActionKeyDown = [&](KeyCode kc) -> bool { return input.keysDown[Key::toPlatformCode(kc)]; };

    auto processActionKey = [&](KeyCode kc, int vk) {
        if (input.keyPressed[Key::toPlatformCode(kc)]) {
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
            const int previous = UTF8Edit::previous(buffer, cursorPos);
            buffer.erase(previous, cursorPos - previous);
            cursorPos = previous;
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
            buffer.erase(cursorPos, UTF8Edit::next(buffer, cursorPos) - cursorPos);
            clearSelection();
        }
        caretVisible = true;
        lastBlink = now;
    }

    // -------------------------------------------------------
    // Arrow keys (with shift for selection, with repeat)
    // -------------------------------------------------------
    if (processActionKey(KeyCode::Left, VK_LEFT)) {
        if (input.shift) {
            if (cursorPos > 0) cursorPos = UTF8Edit::previous(buffer, cursorPos);
            selEnd = cursorPos;
        } else if (hasSelection()) {
            cursorPos = std::min(selStart, selEnd);
            clearSelection();
        } else if (cursorPos > 0) {
            cursorPos = UTF8Edit::previous(buffer, cursorPos);
            clearSelection();
        }
        caretVisible = true;
        lastBlink = now;
    }

    if (processActionKey(KeyCode::Right, VK_RIGHT)) {
        if (input.shift) {
            if (cursorPos < static_cast<int>(buffer.length())) cursorPos = UTF8Edit::next(buffer, cursorPos);
            selEnd = cursorPos;
        } else if (hasSelection()) {
            cursorPos = std::max(selStart, selEnd);
            clearSelection();
        } else if (cursorPos < static_cast<int>(buffer.length())) {
            cursorPos = UTF8Edit::next(buffer, cursorPos);
            clearSelection();
        }
        caretVisible = true;
        lastBlink = now;
    }

    // Home / End
    if (input.keyPressed[Key::toPlatformCode(KeyCode::Home)]) {
        cursorPos = 0;
        if (input.shift) selEnd = cursorPos;
        else clearSelection();
    }
    if (input.keyPressed[Key::toPlatformCode(KeyCode::End)]) {
        cursorPos = static_cast<int>(buffer.length());
        if (input.shift) selEnd = cursorPos;
        else clearSelection();
    }

    // -------------------------------------------------------
    // Select All (Ctrl+A / Cmd+A)
    // -------------------------------------------------------
    if (input.ctrl && input.keyPressed[Key::toPlatformCode(KeyCode::A)]) {
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
        const auto filtered = UTF8Edit::printable(input.pastedText, false);
        buffer.insert(cursorPos, filtered);
        cursorPos += static_cast<int>(filtered.length());
        clearSelection();
    }

    // -------------------------------------------------------
    // Enter - commit and unfocus
    // -------------------------------------------------------
    if (input.keyPressed[Key::toPlatformCode(KeyCode::Enter)]) {
        std::string submittedText = buffer;
        if (externalBind) *externalBind = submittedText;

        setFocused(false);

        submitTextInput(submittedText);
        if (clearOnSubmit) {
            buffer.clear();
            if (externalBind) externalBind->clear();
        }
        cursorPos = static_cast<int>(buffer.length());
        clearSelection();
        return;
    }

    // -------------------------------------------------------
    // Escape - cancel edit
    // -------------------------------------------------------
    if (input.keyPressed[Key::toPlatformCode(KeyCode::Escape)]) {
        setFocused(false);

        if (externalBind) buffer = *externalBind;
        cursorPos = static_cast<int>(buffer.length());
        clearSelection();
        return;
    }

    // -------------------------------------------------------
    // Text input from platform (WM_CHAR / macOS text callback)
    // -------------------------------------------------------
    const auto inserted = UTF8Edit::printable(input.textInput);
    if (!inserted.empty()) {
        if (hasSelection()) deleteSelection();
        buffer.insert(cursorPos, inserted);
        cursorPos += static_cast<int>(inserted.size());
        clearSelection();
        caretVisible = true;
        lastBlink = now;
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

    const InputBoxDisplayLayout layout = buildDisplayLayout(buffer, size.x, renderer);
    const std::string display = buffer.empty() ? placeholder : layout.display;

    uint32_t textColor = buffer.empty() ? Colors::TextDisabled : Colors::TextLight;

    // Draw selection highlight
    if (focused && hasSelection() && !buffer.empty()) {
        int lo = std::min(selStart, selEnd);
        int hi = std::max(selStart, selEnd);

        const int prefixChars = static_cast<int>(layout.prefix.length());
        const int visibleBufferLen = static_cast<int>(buffer.length()) - layout.displayOffset;
        const int dispLo = prefixChars + std::clamp(lo - layout.displayOffset, 0, visibleBufferLen);
        const int dispHi = prefixChars + std::clamp(hi - layout.displayOffset, 0, visibleBufferLen);

        if (dispHi > dispLo) {
            float selX = position.x + kTextPaddingX + measureDisplayPrefixWidth(renderer, layout, dispLo);
            float selEndX = position.x + kTextPaddingX + measureDisplayPrefixWidth(renderer, layout, dispHi);
            float selW = std::max(0.0f, selEndX - selX);
            renderer.drawRect({selX, position.y + 3.0f}, {selW, size.y - 6.0f}, 0xE63366AA);
        }
    }

    renderer.drawText({position.x + kTextPaddingX, position.y + 6}, display, textColor);

    // Draw caret
    if (focused && caretVisible && !buffer.empty()) {
        int dispCursorPos = cursorPos - layout.displayOffset + static_cast<int>(layout.prefix.length());
        dispCursorPos = std::clamp(dispCursorPos, 0, static_cast<int>(layout.display.length()));
        float caretX = position.x + kTextPaddingX + measureDisplayPrefixWidth(renderer, layout, dispCursorPos);
        renderer.drawRect({caretX, position.y + 4.0f}, {1.5f, size.y - 8.0f}, Colors::TextLight);
    } else if (focused && caretVisible && buffer.empty()) {
        renderer.drawRect({position.x + kTextPaddingX, position.y + 4.0f}, {1.5f, size.y - 8.0f}, Colors::TextLight);
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
