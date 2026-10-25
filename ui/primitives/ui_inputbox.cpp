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
#include <cmath>



namespace {
constexpr float kTextPaddingX = 6.0f;
constexpr float kWrapLineHeight = 20.0f;
std::string singleLine(std::string text) {
    std::string result;
    for(size_t i=0;i<text.size();++i) {
        if(text[i]=='\r' || text[i]=='\n') {
            if(text[i]=='\r' && i+1<text.size() && text[i+1]=='\n')++i;
            result+=' ';
        } else result+=text[i];
    }
    return UTF8Edit::printable(result);
}
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

void UIInputBox::setWordWrap(bool enabled) {
    wordWrap_=enabled;wrapWidth_=-1;firstWrapRow_=0;lastWrapCursor_=-1;
    if(enabled)setText(buffer);
}
void UIInputBox::refreshWrap(const OverlayRenderer& renderer) {
    if(wrapText_!=buffer || wrapWidth_!=size.x || wrapLines_.empty()) {
        wrapText_=buffer;wrapWidth_=size.x;wrapLines_.clear();
        const float width=std::max(1.f,size.x-12);
        const int length=static_cast<int>(buffer.size());
        for(int begin=0;begin<length;) {
            int end=begin,lastSpace=-1;
            while(end<length) {
                const int next=UTF8Edit::next(buffer,end);
                if(renderer.measureTextWidth(buffer.substr(begin,next-begin))>width && end>begin)break;
                if(buffer[end]==' ')lastSpace=next;
                end=next;
            }
            if(end<length && lastSpace>begin)end=lastSpace;
            wrapLines_.push_back({begin,end});begin=end;
        }
        if(wrapLines_.empty())wrapLines_.push_back({0,0});
        // Reflow shows the caret while editing, and the beginning otherwise.
        if(focused)revealWrappedCaret();else firstWrapRow_=0;
    }
    const int rows=std::max(1,static_cast<int>((size.y-8)/kWrapLineHeight));
    firstWrapRow_=std::clamp(firstWrapRow_,0,std::max(0,static_cast<int>(wrapLines_.size())-rows));
    if(wrapHeight_!=size.y) {
        wrapHeight_=size.y;
        if(focused)revealWrappedCaret();
    }
    if(lastWrapCursor_!=cursorPos) {
        if(focused)revealWrappedCaret();
        lastWrapCursor_=cursorPos;
    }
}
int UIInputBox::wrappedRow(int offset) const {
    for(int i=0;i<static_cast<int>(wrapLines_.size())-1;++i)if(offset<wrapLines_[i].end)return i;
    return static_cast<int>(wrapLines_.size())-1;
}
int UIInputBox::wrappedOffset(int row,float x,const OverlayRenderer& renderer) const {
    row=std::clamp(row,0,static_cast<int>(wrapLines_.size())-1);
    const auto line=wrapLines_[row];float previous=0;
    for(int i=line.begin;i<line.end;) {
        const int next=UTF8Edit::next(buffer,i);
        const float width=renderer.measureTextWidth(buffer.substr(line.begin,next-line.begin));
        if(x<(previous+width)/2)return i;
        previous=width;i=next;
    }
    // Shared wrap boundaries belong to the next row. Keep clicks at the right
    // edge on this row, before its trailing space/codepoint.
    return row+1<static_cast<int>(wrapLines_.size())?UTF8Edit::previous(buffer,line.end):line.end;
}
void UIInputBox::revealWrappedCaret() {
    const int row=wrappedRow(cursorPos),rows=std::max(1,static_cast<int>((size.y-8)/kWrapLineHeight));
    if(row<firstWrapRow_)firstWrapRow_=row;
    else if(row>=firstWrapRow_+rows)firstWrapRow_=row-rows+1;
}
int UIInputBox::charIndexAtPoint(Vec2 point) {
    if(!wordWrap_)return charIndexAtX(point.x);
    const auto& renderer=UIRoot::get().getRenderer();refreshWrap(renderer);
    const int row=firstWrapRow_+static_cast<int>(std::floor((point.y-position.y-6)/kWrapLineHeight));
    return wrappedOffset(row,point.x-position.x-6,renderer);
}
float UIInputBox::getContentHeight() {
    if(!wordWrap_)return 28;
    refreshWrap(UIRoot::get().getRenderer());
    return std::max(28.f,8.f+wrapLines_.size()*kWrapLineHeight);
}
void UIInputBox::drawWrapped(OverlayRenderer& renderer) {
    refreshWrap(renderer);
    using namespace UITheme::Colors;
    renderer.pushClipRect(position,size);
    if(buffer.empty())renderer.drawText({position.x+6,position.y+6},placeholder,TextDisabled);
    for(int row=firstWrapRow_;row<static_cast<int>(wrapLines_.size());++row) {
        const auto line=wrapLines_[row];
        const float y=position.y+6+(row-firstWrapRow_)*kWrapLineHeight;
        if(y>=position.y+size.y)break;
        const auto width=[&](int offset){return renderer.measureTextWidth(buffer.substr(line.begin,offset-line.begin));};
        const int lo=std::max(line.begin,std::min(selStart,selEnd)),hi=std::min(line.end,std::max(selStart,selEnd));
        if(focused && hi>lo)renderer.drawRect({position.x+6+width(lo),y},{width(hi)-width(lo),kWrapLineHeight},0xE63366AA);
        renderer.drawText({position.x+6,y},buffer.substr(line.begin,line.end-line.begin),TextLight);
        if(focused && caretVisible && wrappedRow(cursorPos)==row)
            renderer.drawRect({position.x+6+width(cursorPos),y},{1.5f,kWrapLineHeight-2},TextLight);
    }
    renderer.popClipRect();
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
            const int clicked=charIndexAtPoint(m);
            if (!focused) {
                setFocused(true);

            }
            cursorPos = clicked;
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
            int dragPos = charIndexAtPoint(m);
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

    if(wordWrap_) {
        refreshWrap(UIRoot::get().getRenderer());
        if(overBox && input.mouseInputEnabled && input.mouseWheelDelta!=0) {
            const int rows=std::max(1,static_cast<int>((size.y-8)/kWrapLineHeight));
            firstWrapRow_=std::clamp(firstWrapRow_-static_cast<int>(input.scrollSteps().y*3),0,std::max(0,static_cast<int>(wrapLines_.size())-rows));
            consumeScroll();
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

    if(wordWrap_ && (input.keyPressed[VK_UP] || input.keyPressed[VK_DOWN])) {
        const auto& renderer=UIRoot::get().getRenderer();refreshWrap(renderer);
        const int row=wrappedRow(cursorPos);
        const float x=renderer.measureTextWidth(buffer.substr(wrapLines_[row].begin,cursorPos-wrapLines_[row].begin));
        cursorPos=wrappedOffset(row+(input.keyPressed[VK_UP]?-1:1),x,renderer);
        if(input.shift)selEnd=cursorPos;else clearSelection();
        caretVisible=true;lastBlink=now;
    }
    if(wordWrap_)refreshWrap(UIRoot::get().getRenderer());
    // Home / End
    if (input.keyPressed[Key::toPlatformCode(KeyCode::Home)]) {
        cursorPos = wordWrap_ && !input.ctrl ? wrapLines_[wrappedRow(cursorPos)].begin : 0;
        if (input.shift) selEnd = cursorPos;
        else clearSelection();
    }
    if (input.keyPressed[Key::toPlatformCode(KeyCode::End)]) {
        cursorPos = wordWrap_ && !input.ctrl ? wrappedOffset(wrappedRow(cursorPos),1e9f,UIRoot::get().getRenderer()) : static_cast<int>(buffer.length());
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
        const auto filtered = wordWrap_ ? singleLine(input.pastedText) : UTF8Edit::printable(input.pastedText, false);
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
    const auto inserted = wordWrap_ ? singleLine(input.textInput) : UTF8Edit::printable(input.textInput);
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
    
    if(wordWrap_) {
        refreshWrap(UIRoot::get().getRenderer());
        if(buffer.empty())renderer.drawText({position.x+6,position.y+6},placeholder,0xFF505050);
        for(int row=firstWrapRow_;row<static_cast<int>(wrapLines_.size());++row) {
            const float y=position.y+6+(row-firstWrapRow_)*kWrapLineHeight;
            if(y+kWrapLineHeight>position.y+size.y+2)break;
            const auto line=wrapLines_[row];
            const auto width=[&](int offset){return UIRoot::get().getRenderer().measureTextWidth(buffer.substr(line.begin,offset-line.begin));};
            const int lo=std::max(line.begin,std::min(selStart,selEnd)),hi=std::min(line.end,std::max(selStart,selEnd));
            if(focused && hi>lo)renderer.drawRect({position.x+6+width(lo),y},{width(hi)-width(lo),kWrapLineHeight},0xE63366AA);
            renderer.drawText({position.x+6,y},buffer.substr(line.begin,line.end-line.begin),0xFFEAEAEA);
            if(focused && caretVisible && wrappedRow(cursorPos)==row)
                renderer.drawRect({position.x+6+width(cursorPos),y},{1.5f,kWrapLineHeight-2},0xFFEAEAEA);
        }
        return;
    }
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

    if(wordWrap_){drawWrapped(renderer);return;}
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
    buffer = wordWrap_ ? singleLine(text) : text;
    cursorPos = static_cast<int>(buffer.length());
    clearSelection();
    if (externalBind) *externalBind = buffer;
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
