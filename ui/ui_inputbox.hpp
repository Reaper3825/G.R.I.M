#pragma once
#include "widget.hpp"
#include "../core/delegate.hpp"
#include <string>
#include <cstdint>

class UIInputBox : public Widget {
public:
    UIInputBox(std::string* bindTarget = nullptr);

    void setPlaceholder(const std::string& text);
    void update(const InputState& input, float dt) override;
    void draw(UIRenderer& renderer) override;
    
    void drawOverlay(class OverlayRenderer& renderer, const Vec2& panelPos);

    const std::string& getText() const { return buffer; }
    void setText(const std::string& text);
    void clear();
    
    bool wantsFocus() const override { return true; }

    Delegate<const std::string&> OnTextSubmitted;

private:
    void submitTextInput(const std::string& text);
    int charIndexAtX(float clickX) const;
    void deleteSelection();
    bool hasSelection() const { return selStart != selEnd; }
    std::string selectedText() const;
    void clearSelection() { selStart = selEnd = cursorPos; }

    std::string buffer;
    std::string placeholder;
    std::string* externalBind = nullptr;

    int cursorPos = 0;   // byte offset into buffer
    int selStart = 0;    // selection anchor
    int selEnd = 0;      // selection moving end (== cursorPos during shift-click/arrow)

    bool caretVisible = true;
    uint64_t lastBlink = 0;

    // Key repeat state
    int heldKeyVK = -1;
    float heldKeyTimer = 0.0f;
    float repeatDelay = 0.40f;  // seconds before first repeat
    float repeatRate = 0.035f;  // seconds between repeats
    bool repeatFiring = false;

    static constexpr float kCharWidth = 8.0f; // approximate monospace glyph advance
};
