#pragma once

#include "widget.hpp"
#include "core/input/InputBindings.hpp"
#include <functional>
#include <optional>
#include <string>

class UIKeybind : public Widget {
public:
    using ChangeCallback = std::function<void(const std::optional<GRIM::InputBindings::Binding>&)>;

    UIKeybind(std::string actionLabel,
              GRIM::InputBindings::Binding binding,
              ChangeCallback onChange);
    ~UIKeybind() override;

    void update(const InputState& input, float dt) override;
    void drawOverlay(class OverlayRenderer& renderer, const Vec2& panelPos) override;

private:
    void stopListening();

    std::string actionLabel_;
    GRIM::InputBindings::Binding binding_;
    ChangeCallback onChange_;
    bool hovered_ = false;
    bool listening_ = false;
};
