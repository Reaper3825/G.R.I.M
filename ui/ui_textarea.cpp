#include "ui_textarea.hpp"
#include "ui_theme.hpp"
#include "overlay_renderer.hpp"
#include "input_parser.hpp"
#include "helpers/mouse.hpp"
#include "helpers/key.hpp"
#include "core/grim_platform.h"
#include "core/platform_clipboard.hpp"
#include "ui_root.hpp"
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

// ─── Selection helpers ──────────────────────────────────

std::string UITextArea::selectedText() const {
    if (!hasSelection()) return "";
    int lo = std::min(selStart, selEnd);
    int hi = std::max(selStart, selEnd);
    return text.substr(lo, hi - lo);
}

void UITextArea::deleteSelection() {
    if (!hasSelection()) return;
    int lo = std::min(selStart, selEnd);
    int hi = std::max(selStart, selEnd);
    text.erase(lo, hi - lo);
    cursorPos = lo;
    clearSelection();
}

int UITextArea::cursorFromMousePos(const Vec2& mousePos, const OverlayRenderer& overlayRenderer) const {
    float relY = mousePos.y - (position.y + kTextPad);
    int clickedVisRow = static_cast<int>(relY / kLineHeight);
    int clickedWrapIdx = clickedVisRow + static_cast<int>(scrollOffset);
    clickedWrapIdx = std::clamp(clickedWrapIdx, 0,
                                std::max(0, static_cast<int>(wrappedLines.size()) - 1));

    float relX = mousePos.x - (position.x + kTextPad);
    int clickedCol = 0;
    if (clickedWrapIdx >= 0 && clickedWrapIdx < static_cast<int>(wrappedLines.size())) {
        const auto& wl = wrappedLines[clickedWrapIdx];
        float prevW = 0.0f;
        for (int ci = 0; ci < static_cast<int>(wl.text.size()); ++ci) {
            float nextW = overlayRenderer.measureTextWidth(wl.text.substr(0, ci + 1));
            float midpoint = prevW + (nextW - prevW) * 0.5f;
            if (relX < midpoint) break;
            clickedCol = ci + 1;
            prevW = nextW;
        }
    }

    return cursorFromWrappedClick(clickedWrapIdx, clickedCol);
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
        if (hasSelection()) {
            deleteSelection();
            changed = true;
        } else if (cursorPos > 0) {
            text.erase(cursorPos - 1, 1);
            cursorPos--;
            changed = true;
        }
        caretVisible = true; lastBlink = now;
    }

    if (processActionKey(KeyCode::Delete, VK_DELETE)) {
        if (hasSelection()) {
            deleteSelection();
            changed = true;
        } else if (cursorPos < static_cast<int>(text.length())) {
            text.erase(cursorPos, 1);
            changed = true;
        }
        caretVisible = true; lastBlink = now;
    }

    if (processActionKey(KeyCode::Left, VK_LEFT)) {
        if (input.shift) {
            if (cursorPos > 0) cursorPos--;
            selEnd = cursorPos;
        } else if (hasSelection()) {
            cursorPos = std::min(selStart, selEnd);
            clearSelection();
        } else if (cursorPos > 0) {
            cursorPos--;
            clearSelection();
        }
        caretVisible = true; lastBlink = now;
    }

    if (processActionKey(KeyCode::Right, VK_RIGHT)) {
        if (input.shift) {
            if (cursorPos < static_cast<int>(text.length())) cursorPos++;
            selEnd = cursorPos;
        } else if (hasSelection()) {
            cursorPos = std::max(selStart, selEnd);
            clearSelection();
        } else if (cursorPos < static_cast<int>(text.length())) {
            cursorPos++;
            clearSelection();
        }
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
        if (input.shift) selEnd = cursorPos;
        else if (hasSelection()) clearSelection();
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
        if (input.shift) selEnd = cursorPos;
        else if (hasSelection()) clearSelection();
        caretVisible = true; lastBlink = now;
    }

    if (Key::wasPressed(KeyCode::Home)) {
        int wIdx = wrapIndexForCursor();
        if (wIdx >= 0)
            cursorPos = cursorFromWrappedClick(wIdx, 0);
        if (input.shift) selEnd = cursorPos;
        else clearSelection();
    }
    if (Key::wasPressed(KeyCode::End)) {
        int wIdx = wrapIndexForCursor();
        if (wIdx >= 0)
            cursorPos = cursorFromWrappedClick(wIdx, static_cast<int>(wrappedLines[wIdx].text.size()));
        if (input.shift) selEnd = cursorPos;
        else clearSelection();
    }

    if (input.ctrl && Key::wasPressed(KeyCode::A)) {
        selStart = 0;
        selEnd = static_cast<int>(text.length());
        cursorPos = selEnd;
    }

    if (input.copyRequested) {
        if (hasSelection()) {
            PlatformClipboard::copyText(selectedText());
        }
    }

    if (input.cutRequested) {
        if (hasSelection()) {
            PlatformClipboard::copyText(selectedText());
            deleteSelection();
            changed = true;
        }
    }

    if (input.pasteRequested && !input.pastedText.empty()) {
        if (hasSelection()) {
            deleteSelection();
            changed = true;
        }
        std::string filtered;
        for (char c : input.pastedText) {
            if ((c >= 32 && c <= 126) || c == '\n' || c == '\t') filtered += c;
        }
        text.insert(cursorPos, filtered);
        cursorPos += static_cast<int>(filtered.length());
        clearSelection();
        changed = true;
    }

    if (Key::wasPressed(KeyCode::Enter)) {
        if (hasSelection()) { deleteSelection(); changed = true; }
        text.insert(text.begin() + cursorPos, '\n');
        cursorPos++;
        clearSelection();
        changed = true;
        caretVisible = true; lastBlink = now;
    }

    for (char c : input.textInput) {
        if (c >= 32 && c <= 126) {
            if (hasSelection()) { deleteSelection(); changed = true; }
            text.insert(text.begin() + cursorPos, c);
            cursorPos++;
            clearSelection();
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

    // ─── Unified mouse: click-to-cursor + drag-to-select ────
    // Uses InputState directly (no Mouse:: static class mixing)
    if (input.mousePressed[0]) {
        if (hovered) {
            focused = true;

            const OverlayRenderer& overlayRenderer = UIRoot::get().getRenderer();
            cursorPos = cursorFromMousePos(m, overlayRenderer);

            if (input.shift) {
                selEnd = cursorPos;
            } else {
                selStart = cursorPos;
                selEnd = cursorPos;
            }
            dragging = true;

#ifdef _WIN32
            lastBlink = GetTickCount64();
#else
            auto tp2 = std::chrono::steady_clock::now().time_since_epoch();
            lastBlink = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(tp2).count());
#endif
            caretVisible = true;
        } else {
            focused = false;
            dragging = false;
        }
    } else if (dragging) {
        if (input.mouseDown[0]) {
            const OverlayRenderer& overlayRenderer = UIRoot::get().getRenderer();
            int dragPos = cursorFromMousePos(m, overlayRenderer);
            if (dragPos != selEnd) {
                selEnd = dragPos;
                cursorPos = dragPos;
                caretVisible = true;
#ifdef _WIN32
                lastBlink = GetTickCount64();
#else
                auto tp3 = std::chrono::steady_clock::now().time_since_epoch();
                lastBlink = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(tp3).count());
#endif
                ensureCursorVisible();
            }
        } else {
            dragging = false;
        }
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

    // While mouse is actively dragging, skip keyboard input so
    // arrow keys / text input don't clobber the drag selection.
    if (focused && !dragging) handleInput(input, dt);
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

        // Draw selection highlight
        if (focused && hasSelection()) {
            int lo = std::min(selStart, selEnd);
            int hi = std::max(selStart, selEnd);

            for (int i = startLine; i < static_cast<int>(wrappedLines.size()) && i < startLine + maxVisLines; ++i) {
                const auto& wl = wrappedLines[i];
                // Compute the absolute text range this wrapped line covers
                int wrapAbsStart = lineStartOffset(wl.logicalLine) + wl.colStart;
                int wrapAbsEnd = wrapAbsStart + static_cast<int>(wl.text.size());

                // Intersect with selection
                int selLo = std::max(lo, wrapAbsStart);
                int selHi = std::min(hi, wrapAbsEnd);

                if (selLo < selHi) {
                    int localLo = selLo - wrapAbsStart;
                    int localHi = selHi - wrapAbsStart;
                    float selX = position.x + kTextPad + renderer.measureTextWidth(wl.text.substr(0, localLo));
                    float selEndX = position.x + kTextPad + renderer.measureTextWidth(wl.text.substr(0, localHi));
                    float selW = std::max(0.0f, selEndX - selX);
                    float selY = position.y + kTextPad + (i - startLine) * kLineHeight;
                    renderer.drawRect({selX, selY}, {selW, kLineHeight}, 0xE63366AA);
                }
            }
        }

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
                    float caretX = position.x + kTextPad + renderer.measureTextWidth(wrappedLines[wIdx].text.substr(0, dispCol));
                    float caretY = position.y + kTextPad + visRow * kLineHeight;
                    renderer.drawRect({caretX, caretY}, {1.5f, kLineHeight}, Colors::TextLight);
                }
            }
        }
    }

    std::string charCount = std::to_string(text.length()) + " chars";
    renderer.drawText({position.x + size.x - 80, position.y + size.y - 18}, charCount, 0xFF909090);
}
