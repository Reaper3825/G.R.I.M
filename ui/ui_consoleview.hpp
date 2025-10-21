#pragma once
#include "widget.hpp"
#include "console_history.hpp"
#include <vector>

class UIConsoleView : public Widget {
public:
    UIConsoleView(ConsoleHistory* history);

    void update(const InputState& input, float dt) override;
    void draw(UIRenderer& renderer) override;

    void scroll(float deltaLines);

private:
    ConsoleHistory* history = nullptr;
    float scrollOffset = 0.0f;
    float lineHeight = 16.0f;
};
