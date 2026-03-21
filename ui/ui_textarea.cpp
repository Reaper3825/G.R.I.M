#include "ui_textarea.hpp"
#include "ui_theme.hpp"
#include "overlay_renderer.hpp"
#include "input_parser.hpp"
#include "helpers/mouse.hpp"
#include "helpers/key.hpp"
#include "core/grim_platform.h"
#include "core/platform_clipboard.hpp"
#include "logger.hpp"
#include <algorithm>
#include <chrono>

UITextArea::UITextArea(const std::string& lbl, const std::string& initialText,
                       std::function<void(const std::string&)> onChange)
    : label(lbl), text(initialText), callback(std::move(onChange))
{
    placeholder = "";
    updateLines();
}

void UITextArea::setText(const std::string& newText) {
    text = newText;
    updateLines();
    cursorPos = std::min(cursorPos, static_cast<int>(text.length()));
}

// ─── Line management ────────────────────────────────────

void UITextArea::updateLines() {
    lines.clear();
    std::string current;
    for (char c : text) {
        if (c == '\n') {
            lines.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    lines.push_back(current);
    if (lines.empty()) lines.push_back("");
    rebuildWrappedLines();
}

void UITextArea::rebuildWrappedLines() {
    wrappedLines.clear();
    float availW = size.x - kTextPad * 2.0f;
    lastWrapWidth = availW;
    size_t maxChars = (availW > kCharWidth) ? static_cast<size_t>(availW / kCharWidth) : 1;

    for (int li = 0; li < static_cast<int>(lines.size()); ++li) {
        const auto& line = lines[li];
        if (line.size() <= maxChars) {
            wrappedLines.push_back({li, 0, line});
        } else {
            size_t pos = 0;
            while (pos < line.size()) {
                size_t end = pos + maxChars;
                if (end >= line.size()) {
                    wrappedLines.push_back({li, static_cast<int>(pos), line.substr(pos)});
                    break;
                }
                size_t breakAt = line.rfind(' ', end);
                if (breakAt != std::string::npos && breakAt > pos) {
                    wrappedLines.push_back({li, static_cast<int>(pos), line.substr(pos, breakAt - pos)});
                    pos = breakAt + 1;
                } else {
                    wrappedLines.push_back({li, static_cast<int>(pos), line.substr(pos, maxChars)});
                    pos = end;
                }
            }
        }
    }
}

// ─── Cursor helpers ─────────────────────────────────────

void UITextArea::cursorToLineCol(int pos, int& line, int& col) const {
    int p = 0;
    for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
        int lineLen = static_cast<int>(lines[i].size());
        if (pos <= p + lineLen || i == static_cast<int>(lines.size()) - 1) {
            line = i;
            col = pos - p;
            return;
        }
        p += lineLen + 1;
    }
    line = static_cast<int>(lines.size()) - 1;
    col = static_cast<int>(lines.back().size());
}

int UITextArea::lineColToCursor(int line, int col) const {
    int pos = 0;
    for (int i = 0; i < line && i < static_cast<int>(lines.size()); ++i)
        pos += static_cast<int>(lines[i].size()) + 1;
    pos += std::min(col, static_cast<int>(lines[std::min(line, static_cast<int>(lines.size()) - 1)].size()));
    return std::min(pos, static_cast<int>(text.size()));
}

int UITextArea::lineStartOffset(int lineIdx) const {
    int pos = 0;
    for (int i = 0; i < lineIdx && i < static_cast<int>(lines.size()); ++i)
        pos += static_cast<int>(lines[i].size()) + 1;
    return pos;
}

int UITextArea::lineEndOffset(int lineIdx) const {
    return lineStartOffset(lineIdx) +
           static_cast<int>(lines[std::min(lineIdx, static_cast<int>(lines.size()) - 1)].size());
}

int UITextArea::wrapIndexForCursor() const {
    int curLine, curCol;
    cursorToLineCol(cursorPos, curLine, curCol);

    int wrapIdx = -1;
    for (int i = 0; i < static_cast<int>(wrappedLines.size()); ++i) {
        if (wrappedLines[i].logicalLine == curLine) {
            int wrapEnd = wrappedLines[i].colStart + static_cast<int>(wrappedLines[i].text.size());
            bool isLast = (i + 1 >= static_cast<int>(wrappedLines.size()) ||
                           wrappedLines[i + 1].logicalLine != curLine);
            if (curCol >= wrappedLines[i].colStart &&
                (curCol < wrapEnd || (curCol == wrapEnd && isLast))) {
                wrapIdx = i;
                break;
            }
        }
    }
    if (wrapIdx < 0) {
        for (int i = static_cast<int>(wrappedLines.size()) - 1; i >= 0; --i) {
            if (wrappedLines[i].logicalLine == curLine) { wrapIdx = i; break; }
        }
    }
    return wrapIdx;
}

int UITextArea::cursorFromWrappedClick(int wrappedIdx, int clickCol) const {
    if (wrappedIdx < 0 || wrappedIdx >= static_cast<int>(wrappedLines.size()))
        return cursorPos;
    const auto& wl = wrappedLines[wrappedIdx];
    int col = wl.colStart + std::clamp(clickCol, 0, static_cast<int>(wl.text.size()));
    return lineColToCursor(wl.logicalLine, col);
}

void UITextArea::ensureCursorVisible() {
    int wIdx = wrapIndexForCursor();
    if (wIdx < 0) return;

    float availH = size.y - kTextPad * 2.0f;
    int maxVisLines = std::max(1, static_cast<int>(availH / kLineHeight));

    int startLine = static_cast<int>(scrollOffset);
    if (wIdx < startLine)
        scrollOffset = static_cast<float>(wIdx);
    else if (wIdx >= startLine + maxVisLines)
        scrollOffset = static_cast<float>(wIdx - maxVisLines + 1);
}

// ─── Input handling ─────────────────────────────────────

void UITextArea::handleInput(const InputState& input, float dt) {
#ifdef _WIN32
    uint64_t now = GetTickCount64();
#else
    auto tp = std::chrono::steady_clock::now().time_since_epoch();
    uint64_t now = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(tp).count());
#endif

    auto processActionKey = [&](KeyCode kc, int vk) -> bool {
        if (Key::wasPressed(kc)) {
            heldKeyVK = vk;
            heldKeyTimer = 0.0f;
            repeatFiring = false;
            return true;
        }
        if (heldKeyVK == vk && Key::isDown(kc)) {
            heldKeyTimer += dt;
            if (!repeatFiring && heldKeyTimer >= kRepeatDelay) {
                repeatFiring = true;
                heldKeyTimer = 0.0f;
                return true;
            }
            if (repeatFiring && heldKeyTimer >= kRepeatRate) {
                heldKeyTimer = 0.0f;
                return true;
            }
        }
        return false;
    };

    if (heldKeyVK >= 0) {
        KeyCode heldKC = KeyCode::Unknown;
        switch (heldKeyVK) {
            case VK_BACK:   heldKC = KeyCode::Backspace; break;
            case VK_DELETE: heldKC = KeyCode::Delete; break;
            case VK_LEFT:   heldKC = KeyCode::Left; break;
            case VK_RIGHT:  heldKC = KeyCode::Right; break;
            case VK_UP:     heldKC = KeyCode::Up; break;
            case VK_DOWN:   heldKC = KeyCode::Down; break;
            default: break;
        }
        if (heldKC != KeyCode::Unknown && !Key::isDown(heldKC)) {
            heldKeyVK = -1;
            heldKeyTimer = 0.0f;
            repeatFiring = false;
        }
    }

    bool changed = false;

    if (processActionKey(KeyCode::Backspace, VK_BACK)) {
        if (cursorPos > 0) {
            text.erase(cursorPos - 1, 1);
            cursorPos--;
            changed = true;
        }
        caretVisible = true; lastBlink = now;
    }

    if (processActionKey(KeyCode::Delete, VK_DELETE)) {
        if (cursorPos < static_cast<int>(text.length())) {
            text.erase(cursorPos, 1);
            changed = true;
        }
        caretVisible = true; lastBlink = now;
    }

    if (processActionKey(KeyCode::Left, VK_LEFT)) {
        if (cursorPos > 0) cursorPos--;
        caretVisible = true; lastBlink = now;
    }

    if (processActionKey(KeyCode::Right, VK_RIGHT)) {
        if (cursorPos < static_cast<int>(text.length())) cursorPos++;
        caretVisible = true; lastBlink = now;
    }

    // Up/Down navigate visual wrapped lines
    if (processActionKey(KeyCode::Up, VK_UP)) {
        int wIdx = wrapIndexForCursor();
        if (wIdx > 0) {
            int curLine, curCol;
            cursorToLineCol(cursorPos, curLine, curCol);
            int visualCol = curCol - wrappedLines[wIdx].colStart;
            cursorPos = cursorFromWrappedClick(wIdx - 1, visualCol);
        }
        caretVisible = true; lastBlink = now;
    }

    if (processActionKey(KeyCode::Down, VK_DOWN)) {
        int wIdx = wrapIndexForCursor();
        if (wIdx >= 0 && wIdx + 1 < static_cast<int>(wrappedLines.size())) {
            int curLine, curCol;
            cursorToLineCol(cursorPos, curLine, curCol);
            int visualCol = curCol - wrappedLines[wIdx].colStart;
            cursorPos = cursorFromWrappedClick(wIdx + 1, visualCol);
        }
        caretVisible = true; lastBlink = now;
    }

    if (Key::wasPressed(KeyCode::Home)) {
        int wIdx = wrapIndexForCursor();
        if (wIdx >= 0)
            cursorPos = cursorFromWrappedClick(wIdx, 0);
    }
    if (Key::wasPressed(KeyCode::End)) {
        int wIdx = wrapIndexForCursor();
        if (wIdx >= 0)
            cursorPos = cursorFromWrappedClick(wIdx, static_cast<int>(wrappedLines[wIdx].text.size()));
    }

    if (input.ctrl && Key::wasPressed(KeyCode::A)) {
        cursorPos = static_cast<int>(text.length());
    }

    if (input.copyRequested) {
        PlatformClipboard::copyText(text);
    }

    if (input.pasteRequested && !input.pastedText.empty()) {
        std::string filtered;
        for (char c : input.pastedText) {
            if ((c >= 32 && c <= 126) || c == '\n' || c == '\t') filtered += c;
        }
        text.insert(cursorPos, filtered);
        cursorPos += static_cast<int>(filtered.length());
        changed = true;
    }

    if (Key::wasPressed(KeyCode::Enter)) {
        text.insert(text.begin() + cursorPos, '\n');
        cursorPos++;
        changed = true;
        caretVisible = true; lastBlink = now;
    }

    for (char c : input.textInput) {
        if (c >= 32 && c <= 126) {
            text.insert(text.begin() + cursorPos, c);
            cursorPos++;
            changed = true;
            caretVisible = true; lastBlink = now;
        }
    }

    if (changed) {
        updateLines();
        if (callback) callback(text);
    }
    ensureCursorVisible();
}

// ─── Update ─────────────────────────────────────────────

void UITextArea::update(const InputState& input, float dt) {
    Vec2 m = input.mousePos;
    hovered = (m.x >= position.x && m.x <= position.x + size.x &&
               m.y >= position.y && m.y <= position.y + size.y);

    // Rebuild wrapped lines if widget width changed
    float availW = size.x - kTextPad * 2.0f;
    if (availW != lastWrapWidth) rebuildWrappedLines();

    if (hovered && Mouse::wasPressed(MouseButton::Left)) {
        focused = true;

        // Map click to a visual wrapped line
        float relY = m.y - (position.y + kTextPad);
        int clickedVisRow = static_cast<int>(relY / kLineHeight);
        int clickedWrapIdx = clickedVisRow + static_cast<int>(scrollOffset);
        clickedWrapIdx = std::clamp(clickedWrapIdx, 0,
                                    std::max(0, static_cast<int>(wrappedLines.size()) - 1));

        float relX = m.x - (position.x + kTextPad);
        int clickedCol = static_cast<int>(relX / kCharWidth);

        cursorPos = cursorFromWrappedClick(clickedWrapIdx, clickedCol);

#ifdef _WIN32
        lastBlink = GetTickCount64();
#else
        auto tp2 = std::chrono::steady_clock::now().time_since_epoch();
        lastBlink = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(tp2).count());
#endif
        caretVisible = true;
    } else if (!hovered && Mouse::wasPressed(MouseButton::Left)) {
        focused = false;
    }

    // Mouse wheel scroll (against wrapped line count)
    if (hovered && input.mouseWheelDelta != 0.0f) {
        scrollOffset -= input.mouseWheelDelta;
        float maxScroll = std::max(0.0f, static_cast<float>(wrappedLines.size()) -
                          (size.y - kTextPad * 2.0f) / kLineHeight);
        scrollOffset = std::clamp(scrollOffset, 0.0f, maxScroll);
    }

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

    if (focused) handleInput(input, dt);
}

// ─── Draw ───────────────────────────────────────────────

void UITextArea::draw(UIRenderer& renderer) {}

void UITextArea::drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos) {
    using namespace UITheme;

    renderer.drawText({position.x, position.y - 18}, label, Colors::TextPrimary);

    uint32_t bgColor = focused ? Colors::WidgetBgHover : (hovered ? Colors::WidgetBgHover : Colors::WidgetBg);
    renderer.drawRoundedRect(position, size, bgColor, Sizes::WidgetRadius);

    uint32_t borderColor = focused ? Colors::BorderFocus : Colors::BorderPrimary;
    renderer.drawRoundedBorder(position, size, borderColor, Sizes::WidgetRadius);

    // Rebuild if width changed since last wrap
    float availW = size.x - kTextPad * 2.0f;
    if (availW != lastWrapWidth) rebuildWrappedLines();

    float availH = size.y - kTextPad * 2.0f;
    int maxVisLines = std::max(1, static_cast<int>(availH / kLineHeight));

    if (text.empty() && !focused) {
        if (!placeholder.empty())
            renderer.drawText({position.x + kTextPad, position.y + kTextPad}, placeholder, Colors::TextDisabled);
    } else {
        int startLine = static_cast<int>(scrollOffset);
        float yPos = position.y + kTextPad;
        for (int i = startLine; i < static_cast<int>(wrappedLines.size()) && i < startLine + maxVisLines; ++i) {
            renderer.drawText({position.x + kTextPad, yPos}, wrappedLines[i].text, 0xFFD0D0D0);
            yPos += kLineHeight;
        }

        // Caret
        if (focused && caretVisible) {
            int wIdx = wrapIndexForCursor();
            if (wIdx >= 0) {
                int visRow = wIdx - startLine;
                if (visRow >= 0 && visRow < maxVisLines) {
                    int curLine, curCol;
                    cursorToLineCol(cursorPos, curLine, curCol);
                    int dispCol = curCol - wrappedLines[wIdx].colStart;
                    dispCol = std::clamp(dispCol, 0, static_cast<int>(wrappedLines[wIdx].text.size()));
                    float caretX = position.x + kTextPad + dispCol * kCharWidth;
                    float caretY = position.y + kTextPad + visRow * kLineHeight;
                    renderer.drawRect({caretX, caretY}, {1.5f, kLineHeight}, Colors::TextLight);
                }
            }
        }
    }

    std::string charCount = std::to_string(text.length()) + " chars";
    renderer.drawText({position.x + size.x - 80, position.y + size.y - 18}, charCount, 0xFF909090);
}
