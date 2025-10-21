#pragma once
#include "widget.hpp"
#include <vector>
#include <memory>
#include <functional>

class UIPanel : public Widget {
public:
    UIPanel(const std::string& title = "", bool draggable = true);

    void addChild(std::shared_ptr<Widget> w);
    void removeChild(Widget* w);

    void update(const InputState& input, float dt) override;
    void draw(UIRenderer& renderer) override;

    void setBackground(uint32_t color) { bgColor = color; }
    void setBorder(uint32_t color) { borderColor = color; }

    void setOnClose(std::function<void()> cb) { onClose = std::move(cb); }
    void setTitle(const std::string& t) { title = t; }

private:
    std::string title;
    std::vector<std::shared_ptr<Widget>> children;
    bool draggable = true;
    bool dragging = false;
    Vec2 dragOffset{0, 0};

    uint32_t bgColor = 0xFF202020;
    uint32_t borderColor = 0xFFFFFFFF;

    float titleBarHeight = 24.0f;
    std::function<void()> onClose = nullptr;
};
