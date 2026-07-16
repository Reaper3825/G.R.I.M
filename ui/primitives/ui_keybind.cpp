#include "ui_keybind.hpp"
#include "core/input_parser.hpp"
#include "helpers/mouse.hpp"
#include "overlay_renderer.hpp"
#include "ui_theme.hpp"

UIKeybind::UIKeybind(std::string actionLabel,
                     GRIM::InputBindings::Binding binding,
                     ChangeCallback onChange)
    : actionLabel_(std::move(actionLabel)),
      binding_(binding),
      onChange_(std::move(onChange))
{
}

UIKeybind::~UIKeybind()
{
    stopListening();
}

void UIKeybind::stopListening()
{
    if (!listening_) return;
    listening_ = false;
    GRIM::InputBindings::endCapture();
}

void UIKeybind::update(const InputState& input, float)
{
    const float buttonWidth = 190.0f;
    const Vec2 buttonPos{position.x + size.x - buttonWidth - 10.0f, position.y + 7.0f};
    const Vec2 buttonSize{buttonWidth, size.y - 14.0f};
    hovered_ = input.mousePos.x >= buttonPos.x && input.mousePos.x <= buttonPos.x + buttonSize.x &&
               input.mousePos.y >= buttonPos.y && input.mousePos.y <= buttonPos.y + buttonSize.y;

    if (!listening_ && hovered_ &&
        (input.mousePressed[0] || Mouse::wasPressed(MouseButton::Left)))
    {
        listening_ = true;
        GRIM::InputBindings::beginCapture();
        return;
    }

    if (!listening_) return;

    // Escape cancels; Backspace/Delete deliberately clear the action.
    if (input.keyPressed[VK_ESCAPE]) {
        stopListening();
        return;
    }
    if (input.keyPressed[VK_BACK] || input.keyPressed[VK_DELETE]) {
        stopListening();
        if (onChange_) onChange_(std::nullopt);
        return;
    }

    const auto captured = GRIM::InputBindings::capturePressedBinding(input);
    if (!captured) return;
    binding_ = *captured;
    stopListening();
    if (onChange_) onChange_(binding_);
}

void UIKeybind::drawOverlay(OverlayRenderer& renderer, const Vec2&)
{
    using namespace UITheme;
    renderer.drawRoundedRect(position, size, Colors::WidgetBg, Sizes::WidgetRadius);
    renderer.drawRoundedBorder(position, size, Colors::BorderSubtle, Sizes::WidgetRadius);
    renderer.drawText({position.x + 14.0f, position.y + size.y * 0.5f - 8.0f},
                      actionLabel_, Colors::TextPrimary);

    const float buttonWidth = 190.0f;
    const Vec2 buttonPos{position.x + size.x - buttonWidth - 10.0f, position.y + 7.0f};
    const Vec2 buttonSize{buttonWidth, size.y - 14.0f};
    const uint32_t fill = listening_ ? Colors::WidgetBgActive :
                          (hovered_ ? Colors::WidgetBgHover : Colors::ContentAreaBg);
    renderer.drawRoundedRect(buttonPos, buttonSize, fill, buttonSize.y * 0.5f);
    renderer.drawRoundedBorder(buttonPos, buttonSize,
                               listening_ ? Colors::Primary : Colors::BorderPrimary,
                               buttonSize.y * 0.5f);

    const std::string value = listening_ ? "Press a key..." :
                              GRIM::InputBindings::toDisplayString(binding_);
    const float textWidth = static_cast<float>(value.size()) * 7.5f;
    renderer.drawText({buttonPos.x + (buttonSize.x - textWidth) * 0.5f,
                       buttonPos.y + buttonSize.y * 0.5f - 8.0f},
                      value, listening_ ? Colors::TextWhite : Colors::TextPrimary);
}
