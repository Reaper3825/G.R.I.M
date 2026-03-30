#include "ui_action_menu.hpp"
#include "ui_theme.hpp"
#include "overlay_renderer.hpp"
#include "input_parser.hpp"
#include "helpers/mouse.hpp"
#include <algorithm>
#include <cmath>

UIActionMenu::UIActionMenu(const std::string& title)
    : title_(title) {}

void UIActionMenu::addItem(const std::string& label,
                           std::function<void()> callback,
                           uint32_t textColor) {
    items_.push_back({label, std::move(callback), textColor, false});
}

void UIActionMenu::addSeparator() {
    items_.push_back({"", nullptr, 0, true});
}

void UIActionMenu::clearItems() {
    items_.clear();
}

void UIActionMenu::setItemLabel(int index, const std::string& label) {
    if (index >= 0 && index < static_cast<int>(items_.size()))
        items_[index].label = label;
}

// ─────────────────────────────────────────────────────────
//  Geometry helpers
// ─────────────────────────────────────────────────────────

float UIActionMenu::popupHeight() const {
    float h = 6.0f; // top padding
    for (const auto& item : items_)
        h += item.separator ? kSepH : kItemH;
    h += 6.0f; // bottom padding
    return h;
}

float UIActionMenu::popupWidth() const {
    float maxW = kMinPopupW;
    for (const auto& item : items_) {
        if (item.separator) continue;
        float w = item.label.length() * 7.5f + kPadX * 2.0f + 16.0f;
        if (w > maxW) maxW = w;
    }
    return maxW;
}

// ─────────────────────────────────────────────────────────
//  Update — hit test the pill button + popup items
// ─────────────────────────────────────────────────────────

void UIActionMenu::update(const InputState& input, float /*dt*/) {
    Vec2 m = input.mousePos;

    // Pill button hit test
    bool overButton = (m.x >= position.x && m.x <= position.x + size.x &&
                       m.y >= position.y && m.y <= position.y + size.y);
    hovered_ = overButton;

    bool leftPressed = Mouse::wasPressed(MouseButton::Left);

    // Popup geometry (anchored below the pill button)
    float pw = popupWidth();
    float ph = popupHeight();
    float px = position.x;
    float py = position.y + size.y + 4.0f;

    bool overPopup = expanded_ &&
                     (m.x >= px && m.x <= px + pw &&
                      m.y >= py && m.y <= py + ph);

    // Track hovered item
    hoveredItem_ = -1;
    if (expanded_ && overPopup) {
        float iy = py + 6.0f; // top padding
        for (int i = 0; i < static_cast<int>(items_.size()); ++i) {
            float ih = items_[i].separator ? kSepH : kItemH;
            if (m.y >= iy && m.y < iy + ih && !items_[i].separator)
                hoveredItem_ = i;
            iy += ih;
        }
    }

    if (leftPressed) {
        if (overButton) {
            expanded_ = !expanded_;
        } else if (expanded_) {
            if (hoveredItem_ >= 0 && hoveredItem_ < static_cast<int>(items_.size())) {
                auto& item = items_[hoveredItem_];
                if (item.callback) item.callback();
            }
            expanded_ = false;
            hoveredItem_ = -1;
        }
    }
}

// ─────────────────────────────────────────────────────────
//  Draw — pill button only
// ─────────────────────────────────────────────────────────

void UIActionMenu::drawOverlay(OverlayRenderer& renderer, const Vec2& /*panelPos*/) {
    using namespace UITheme;

    uint32_t bg = Colors::WidgetBg;
    uint32_t textCol = Colors::TextPrimary;
    if (expanded_) {
        bg = Colors::WidgetBgActive;
        textCol = Colors::TextWhite;
    } else if (hovered_) {
        bg = Colors::WidgetBgHover;
        textCol = Colors::TextWhite;
    }

    float pillR = size.y * 0.5f;
    renderer.drawRoundedRect(position, size, bg, pillR);

    uint32_t border = (hovered_ || expanded_)
                          ? Colors::BorderPrimary
                          : Colors::BorderSubtle;
    renderer.drawRoundedBorder(position, size, border, pillR);

    // Label + down-arrow indicator
    std::string display = expanded_ ? (title_ + "  ^") : (title_ + "  v");
    float textW = display.length() * 7.0f;
    float tx = position.x + (size.x - textW) * 0.5f;
    float ty = position.y + size.y * 0.5f - 8.0f;
    renderer.drawText({tx, ty}, display, textCol);
}

// ─────────────────────────────────────────────────────────
//  Draw popup — call after all other widgets (z-order)
// ─────────────────────────────────────────────────────────

void UIActionMenu::drawExpandedList(OverlayRenderer& renderer,
                                     const Vec2& /*panelPos*/) {
    if (!expanded_) return;
    using namespace UITheme;

    float pw = popupWidth();
    float ph = popupHeight();
    float px = position.x;
    float py = position.y + size.y + 4.0f;

    // Glass panel background
    renderer.drawRoundedRect({px, py}, {pw, ph},
                             Colors::PanelBg, Sizes::SmallRadius);
    renderer.drawRoundedBorder({px, py}, {pw, ph},
                               Colors::BorderPrimary, Sizes::SmallRadius);

    // Items
    float iy = py + 6.0f;
    for (int i = 0; i < static_cast<int>(items_.size()); ++i) {
        const auto& item = items_[i];

        if (item.separator) {
            float sepY = iy + kSepH * 0.5f;
            renderer.drawRect({px + kPadX, sepY}, {pw - kPadX * 2.0f, 1.0f},
                              Colors::DividerLine);
            iy += kSepH;
            continue;
        }

        // Hover highlight
        if (i == hoveredItem_) {
            renderer.drawRoundedRect({px + 4.0f, iy + 1.0f},
                                     {pw - 8.0f, kItemH - 2.0f},
                                     Colors::WidgetBgHover, 4.0f);
        }

        renderer.drawText({px + kPadX + 4.0f, iy + 6.0f},
                          item.label, item.textColor);
        iy += kItemH;
    }
}
