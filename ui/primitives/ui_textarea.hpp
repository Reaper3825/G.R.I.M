#pragma once
#include "widget.hpp"
#include <functional>
#include <string>
#include <vector>
#include <cstdint>

class UITextArea : public Widget {
public:
    UITextArea(const std::string& label, const std::string& initialText, 
               std::function<void(const std::string&)> onChange);

    void update(const InputState& input, float dt) override;
    void draw(class UIRenderer& renderer) override;
    
    void setText(const std::string& text);
    std::string getText() const { return text; }
    std::string getLabel() const { return label; }
    
    bool wantsFocus() const override { return true; }

    void drawOverlay(class OverlayRenderer& renderer, const Vec2& panelPos);

private:
    std::string label;
    std::string text;
    std::string placeholder;
    std::function<void(const std::string&)> callback;
    
    bool focused = false;
    bool hovered = false;
    int cursorPos = 0;
    float scrollOffset = 0.0f;
    
    // Selection state
    int selStart = 0;    // selection anchor
    int selEnd = 0;      // selection moving end
    bool dragging = false;
    
    bool hasSelection() const { return selStart != selEnd; }
    void clearSelection() { selStart = selEnd = cursorPos; }
    std::string selectedText() const;
    void deleteSelection();
    
    // Logical lines (split on '\n')
    std::vector<std::string> lines;

    // Wrapped visual lines (rebuilt when text or size changes)
    struct WrapLine {
        int logicalLine;
        int colStart;
        std::string text;
    };
    std::vector<WrapLine> wrappedLines;
    float lastWrapWidth = 0.0f;

    // Caret blink
    bool caretVisible = true;
    uint64_t lastBlink = 0;

    // Key repeat
    int heldKeyVK = -1;
    float heldKeyTimer = 0.0f;
    static constexpr float kRepeatDelay = 0.40f;
    static constexpr float kRepeatRate = 0.035f;
    bool repeatFiring = false;

    static constexpr float kCharWidth = 8.4f;
    static constexpr float kLineHeight = 16.0f;
    static constexpr float kTextPad = 8.0f;
    
    void updateLines();
    void rebuildWrappedLines();
    void handleInput(const InputState& input, float dt);

    // Cursor helpers
    void cursorToLineCol(int pos, int& line, int& col) const;
    int lineColToCursor(int line, int col) const;
    int lineStartOffset(int lineIdx) const;
    int lineEndOffset(int lineIdx) const;
    void ensureCursorVisible();

    int wrapIndexForCursor() const;
    int cursorFromWrappedClick(int wrappedIdx, int clickCol) const;
    int cursorFromMousePos(const Vec2& mousePos, const class OverlayRenderer& renderer) const;
};
