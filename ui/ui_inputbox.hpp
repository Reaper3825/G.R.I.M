#pragma once
#include "widget.hpp"
#include "../core/delegate.hpp"
#include <string>

class UIInputBox : public Widget {
public:
    UIInputBox(std::string* bindTarget = nullptr);

    void setPlaceholder(const std::string& text);
    void update(const InputState& input, float dt) override;
    void draw(UIRenderer& renderer) override;
    
    // For overlay rendering
    void drawOverlay(class OverlayRenderer& renderer, const Vec2& panelPos);

    const std::string& getText() const { return buffer; }
    void setText(const std::string& text);
    void clear();
    
    // Override base class focus methods
    bool wantsFocus() const override { return true; }  // Input boxes want focus

    // Delegate event fired when Enter is pressed and text is submitted
    // Signature: void(const std::string& submittedText)
    Delegate<const std::string&> OnTextSubmitted;

    // Optional: Multicast version if you need multiple listeners
    // MulticastDelegate<const std::string&> OnTextSubmittedMulticast;

private:
    void submitTextInput(const std::string& text);

    std::string buffer;
    std::string placeholder;
    std::string* externalBind = nullptr;
    bool caretVisible = true;
    uint64_t lastBlink = 0;
};
