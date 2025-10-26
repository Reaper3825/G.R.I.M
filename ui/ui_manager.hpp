#pragma once
#include <vector>
#include <memory>
#include "widget.hpp"
#include "ui_renderer.hpp"
#include "input_parser.hpp"

struct InputState; // Forward declaration

class UIManager {
public:
    void add(std::shared_ptr<Widget> w);
    void remove(Widget* w);

    void update(float dt);
    void draw();

    void setRenderer(UIRenderer* r) { renderer = r; }
    void setInputState(const InputState& in) { input = in; }

    Widget* getFocused() const { return focused; }

private:
    std::vector<std::shared_ptr<Widget>> widgets;
    InputState input;
    UIRenderer* renderer = nullptr;
    Widget* focused = nullptr;
};
