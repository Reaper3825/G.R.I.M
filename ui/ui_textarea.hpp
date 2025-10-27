#pragma once
#include "widget.hpp"
#include <functional>
#include <string>
#include <vector>

class UITextArea : public Widget {
public:
    UITextArea(const std::string& label, const std::string& initialText, 
               std::function<void(const std::string&)> onChange);

    void update(const InputState& input, float dt) override;
    void draw(class UIRenderer& renderer) override;
    
    void setText(const std::string& text);
    std::string getText() const { return text; }
    std::string getLabel() const { return label; }
    
    // For overlay rendering
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
    
    // Multi-line support
    std::vector<std::string> lines;
    int currentLine = 0;
    
    void updateLines();
    void handleInput(const InputState& input);
};
