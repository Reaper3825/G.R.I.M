#pragma once
#include "widget.hpp"
#include <string>
#include <functional>

class UIButton : public Widget {
public:
    UIButton(const std::string& label, std::function<void()> onClick);

    void update(const InputState& input, float dt) override;
    void draw(UIRenderer& renderer) override;

private:
    std::string label;
    std::function<void()> callback;
    bool pressed = false;
    uint32_t baseColor = 0xFF303030;
    uint32_t hoverColor = 0xFF404040;
    uint32_t pressColor = 0xFF505050;
};
