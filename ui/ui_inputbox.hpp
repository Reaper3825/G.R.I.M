#pragma once
#include "widget.hpp"
#include <string>

class UIInputBox : public Widget {
public:
    UIInputBox(std::string* bindTarget = nullptr);

    void setPlaceholder(const std::string& text);
    void update(const InputState& input, float dt) override;
    void draw(UIRenderer& renderer) override;

    const std::string& getText() const { return buffer; }

private:
    std::string buffer;
    std::string placeholder;
    std::string* externalBind = nullptr;
    bool caretVisible = true;
    uint64_t lastBlink = 0;
};
