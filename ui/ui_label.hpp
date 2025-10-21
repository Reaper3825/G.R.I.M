#pragma once
#include "widget.hpp"
#include <string>

class UILabel : public Widget {
public:
    UILabel(const std::string& text = "", uint32_t color = 0xFFFFFFFF);

    void setText(const std::string& t);
    void setColor(uint32_t c);

    void update(const InputState& input, float dt) override;
    void draw(UIRenderer& renderer) override;

private:
    std::string text;
    uint32_t color;
};
