#include "helpers/utf8_edit.hpp"
#include "ui_textarea.hpp"
#include "../ui_theme.hpp"
#include "../overlay_renderer.hpp"
#include "../../core/input_parser.hpp"
#include "helpers/mouse.hpp"
#include "helpers/key.hpp"
#include "core/grim_platform.h"
#include "core/platform_clipboard.hpp"
#include "../ui_root.hpp"
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
    if (text == newText) return;
    text = newText;
    updateLines();
    cursorPos = UTF8Edit::boundary(text, std::min(cursorPos, static_cast<int>(text.length())));
    clearSelection();
    scrollOffset = 0.0f;
    dragging = false;
    releasePointer();
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
    wrapDirty = true;
    reflowAnchorLine = -1;
}

void UITextArea::rebuildWrappedLines(const OverlayRenderer& renderer) {
    // Preserve a source-text anchor across successive resize frames.
    if (!wrapDirty && reflowAnchorLine < 0 && !wrappedLines.empty()) {
        const auto& top = wrappedLines[std::clamp(static_cast<int>(scrollOffset),
            0, static_cast<int>(wrappedLines.size()) - 1)];
        reflowAnchorLine = top.logicalLine;
        reflowAnchorCol = top.colStart;
    }
    wrappedLines.clear();
    const float availW = std::max(0.0f, size.x - kTextPad * 2.0f);
    lastWrapWidth = size.x - kTextPad * 2.0f;
    wrapDirty = false;
    for (int li = 0; li < static_cast<int>(lines.size()); ++li) {
        const auto& line = lines[li];
        if (line.empty()) wrappedLines.push_back({li, 0, ""});
        int pos = 0;
        while (pos < static_cast<int>(line.size())) {
            int end = pos;
            int lastSpace = -1;
            while (end < static_cast<int>(line.size())) {
                const int next = UTF8Edit::next(line, end);
                if (renderer.measureTextWidth(line.substr(pos, next - pos)) > availW && end > pos)
                    break;
                if (line[end] == ' ') lastSpace = end;
                end = next;
            }
            int nextPos = end;
            if (end < static_cast<int>(line.size()) && lastSpace > pos) {
                end = lastSpace;
                nextPos = lastSpace + 1;
            }
            wrappedLines.push_back({li, pos, line.substr(pos, end - pos)});
            pos = nextPos;
        }
    }
    if (reflowAnchorLine >= 0) {
        for (int i = 0; i < static_cast<int>(wrappedLines.size()); ++i) {
            const auto& row = wrappedLines[i];
            if (row.logicalLine == reflowAnchorLine && row.colStart <= reflowAnchorCol)
                scrollOffset = static_cast<float>(i);
        }
    }
    const int visible = std::max(1, static_cast<int>((size.y - kTextPad * 2.0f) / kLineHeight));
    scrollOffset = std::clamp(scrollOffset, 0.0f,
        static_cast<float>(std::max(0, static_cast<int>(wrappedLines.size()) - visible)));
}

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
    return UTF8Edit::boundary(text, std::min(pos, static_cast<int>(text.size())));
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
    reflowAnchorLine = -1;
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
        for (int ci = 0; ci < static_cast<int>(wl.text.size()); ci = UTF8Edit::next(wl.text, ci)) {
            float nextW = overlayRenderer.measureTextWidth(wl.text.substr(0, UTF8Edit::next(wl.text, ci)));
            float midpoint = prevW + (nextW - prevW) * 0.5f;
            if (relX < midpoint) break;
            clickedCol = UTF8Edit::next(wl.text, ci);
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
        if (input.keyPressed[Key::toPlatformCode(kc)]) {
            heldKeyVK = vk;
            heldKeyTimer = 0.0f;
            repeatFiring = false;
            return true;
        }
        if (heldKeyVK == vk && input.keysDown[Key::toPlatformCode(kc)]) {
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
        if (heldKC != KeyCode::Unknown && !input.keysDown[Key::toPlatformCode(heldKC)]) {
            heldKeyVK = -1;
            heldKeyTimer = 0.0f;
            repeatFiring = false;
        }
    }

    const int previousCursor = cursorPos;
    bool changed = false;

    if (!readOnly && processActionKey(KeyCode::Backspace, VK_BACK)) {
        if (hasSelection()) {
            deleteSelection();
            changed = true;
        } else if (cursorPos > 0) {
            const int previous = UTF8Edit::previous(text, cursorPos);
            text.erase(previous, cursorPos - previous);
            cursorPos = previous;
            changed = true;
        }
        caretVisible = true; lastBlink = now;
    }

    if (!readOnly && processActionKey(KeyCode::Delete, VK_DELETE)) {
        if (hasSelection()) {
            deleteSelection();
            changed = true;
        } else if (cursorPos < static_cast<int>(text.length())) {
            text.erase(cursorPos, UTF8Edit::next(text, cursorPos) - cursorPos);
            changed = true;
        }
        caretVisible = true; lastBlink = now;
    }

    if (processActionKey(KeyCode::Left, VK_LEFT)) {
        if (input.shift) {
            if (cursorPos > 0) cursorPos = UTF8Edit::previous(text, cursorPos);
            selEnd = cursorPos;
        } else if (hasSelection()) {
            cursorPos = std::min(selStart, selEnd);
            clearSelection();
        } else if (cursorPos > 0) {
            cursorPos = UTF8Edit::previous(text, cursorPos);
            clearSelection();
        }
        caretVisible = true; lastBlink = now;
    }

    if (processActionKey(KeyCode::Right, VK_RIGHT)) {
        if (input.shift) {
            if (cursorPos < static_cast<int>(text.length())) cursorPos = UTF8Edit::next(text, cursorPos);
            selEnd = cursorPos;
        } else if (hasSelection()) {
            cursorPos = std::max(selStart, selEnd);
            clearSelection();
        } else if (cursorPos < static_cast<int>(text.length())) {
            cursorPos = UTF8Edit::next(text, cursorPos);
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
        else clearSelection();
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
        else clearSelection();
        caretVisible = true; lastBlink = now;
    }

    if (input.keyPressed[Key::toPlatformCode(KeyCode::Home)]) {
        int wIdx = wrapIndexForCursor();
        if (wIdx >= 0)
            cursorPos = cursorFromWrappedClick(wIdx, 0);
        if (input.shift) selEnd = cursorPos;
        else clearSelection();
    }
    if (input.keyPressed[Key::toPlatformCode(KeyCode::End)]) {
        int wIdx = wrapIndexForCursor();
        if (wIdx >= 0)
            cursorPos = cursorFromWrappedClick(wIdx, static_cast<int>(wrappedLines[wIdx].text.size()));
        if (input.shift) selEnd = cursorPos;
        else clearSelection();
    }

    if (input.ctrl && input.keyPressed[Key::toPlatformCode(KeyCode::A)]) {
        selStart = 0;
        selEnd = static_cast<int>(text.length());
        cursorPos = selEnd;
    }

    if (input.copyRequested) {
        if (hasSelection()) {
            PlatformClipboard::copyText(selectedText());
        }
    }

    if (!readOnly && input.cutRequested) {
        if (hasSelection()) {
            PlatformClipboard::copyText(selectedText());
            deleteSelection();
            changed = true;
        }
    }

    if (!readOnly && input.pasteRequested && !input.pastedText.empty()) {
        if (hasSelection()) {
            deleteSelection();
            changed = true;
        }
        const auto filtered = UTF8Edit::printable(input.pastedText, true);
        text.insert(cursorPos, filtered);
        cursorPos += static_cast<int>(filtered.length());
        clearSelection();
        changed = true;
    }

    if (!readOnly && input.keyPressed[Key::toPlatformCode(KeyCode::Enter)]) {
        if (hasSelection()) { deleteSelection(); changed = true; }
        text.insert(text.begin() + cursorPos, '\n');
        cursorPos++;
        clearSelection();
        changed = true;
        caretVisible = true; lastBlink = now;
    }

    const auto inserted = UTF8Edit::printable(input.textInput);
    if (!readOnly && !inserted.empty()) {
        if (hasSelection()) deleteSelection();
        text.insert(cursorPos, inserted);
        cursorPos += static_cast<int>(inserted.size());
        clearSelection();
        changed = true;
        caretVisible = true; lastBlink = now;
    }

    if (changed) {
        updateLines();
        rebuildWrappedLines(UIRoot::get().getRenderer());
        if (callback) callback(text);
    }
    if (changed || cursorPos != previousCursor) ensureCursorVisible();
}

// ─── Update ─────────────────────────────────────────────

void UITextArea::update(const InputState& source, float dt) {
    const auto input = routeInput(source);
    if (!hasPointerCapture()) dragging = false;
    Vec2 m = input.mousePos;
    hovered = (m.x >= position.x && m.x <= position.x + size.x &&
               m.y >= position.y && m.y <= position.y + size.y);

    // Rebuild wrapped lines if widget width changed
    float availW = size.x - kTextPad * 2.0f;
    if (wrapDirty || availW != lastWrapWidth) rebuildWrappedLines(UIRoot::get().getRenderer());

    // ─── Unified mouse: click-to-cursor + drag-to-select ────
    // Uses InputState directly (no Mouse:: static class mixing)
    if (input.mousePressed[0]) {
        if (hovered) {
            setFocused(true);

            const OverlayRenderer& overlayRenderer = UIRoot::get().getRenderer();
            cursorPos = cursorFromMousePos(m, overlayRenderer);

            if (input.shift) {
                selEnd = cursorPos;
            } else {
                selStart = cursorPos;
                selEnd = cursorPos;
            }
            dragging = capturePointer();

#ifdef _WIN32
            lastBlink = GetTickCount64();
#else
            auto tp2 = std::chrono::steady_clock::now().time_since_epoch();
            lastBlink = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(tp2).count());
#endif
            caretVisible = true;
        } else {
            setFocused(false);
            dragging = false;
            releasePointer();
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
            releasePointer();
        }
    }

    // Mouse wheel scroll (against wrapped line count)
    if (hovered && input.mouseWheelDelta != 0.0f) {
        reflowAnchorLine = -1;
        const float previousScroll = scrollOffset;
        scrollOffset -= input.scrollSteps().y * 3.0f;
        const int visibleLines = std::max(1, static_cast<int>((size.y - kTextPad * 2.0f) / kLineHeight));
        float maxScroll = static_cast<float>(std::max(0, static_cast<int>(wrappedLines.size()) - visibleLines));
        scrollOffset = std::clamp(scrollOffset, 0.0f, maxScroll);
        if (scrollOffset != previousScroll) consumeScroll();
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
    if (wrapDirty || availW != lastWrapWidth) rebuildWrappedLines(renderer);

    float availH = size.y - kTextPad * 2.0f;
    int maxVisLines = std::max(1, static_cast<int>(availH / kLineHeight));

    scrollOffset = std::clamp(scrollOffset, 0.0f,
        static_cast<float>(std::max(0, static_cast<int>(wrappedLines.size()) - maxVisLines)));
    renderer.pushClipRect(position, size);
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

    renderer.popClipRect();
    if (readOnly) return;
    std::string charCount = std::to_string(text.length()) + " chars";
    renderer.drawText({position.x + size.x - 80, position.y + size.y - 18}, charCount, 0xFF909090);
}
