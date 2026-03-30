#pragma once
#include "widget.hpp"
#include "ui_theme.hpp"
#include <functional>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────
//  UIActionMenu — Compact pill-button with expandable
//  action list.  Replaces toolbar button clusters with a
//  single trigger that opens a floating glass menu.
//
//  Usage:
//      auto menu = std::make_shared<UIActionMenu>("Curriculum");
//      menu->addItem("New Curriculum",   [this]{ ... });
//      menu->addItem("Delete",           [this]{ ... }, UITheme::Colors::Danger);
//      menu->addSeparator();
//      menu->addItem("Assign to Model",  [this]{ ... });
//
//  Drawing:
//      menu->drawOverlay(renderer, panelPos);   // pill button
//      // After all other widgets:
//      menu->drawExpandedList(renderer, panelPos); // popup
// ─────────────────────────────────────────────────────────

struct ActionMenuItem {
    std::string         label;
    std::function<void()> callback;
    uint32_t            textColor  = UITheme::Colors::TextPrimary;
    bool                separator  = false;  // true = horizontal line, label/callback ignored
};

class UIActionMenu : public Widget {
public:
    explicit UIActionMenu(const std::string& title);

    // ── Item management ─────────────────────────────
    void addItem(const std::string& label,
                 std::function<void()> callback,
                 uint32_t textColor = UITheme::Colors::TextPrimary);
    void addSeparator();
    void clearItems();
    void setItemLabel(int index, const std::string& label);
    void setTitle(const std::string& title) { title_ = title; }

    // ── Widget interface ────────────────────────────
    void update(const InputState& input, float dt) override;
    void draw(UIRenderer& renderer) override {}
    void drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos) override;

    /// Draw the popup list.  Call AFTER all other widgets
    /// in the same z-order pass as dropdown expanded lists.
    void drawExpandedList(OverlayRenderer& renderer, const Vec2& panelPos);

    bool isExpanded() const { return expanded_; }

private:
    std::string              title_;
    std::vector<ActionMenuItem> items_;
    bool   expanded_    = false;
    bool   hovered_     = false;
    bool   pressed_     = false;
    int    hoveredItem_ = -1;

    static constexpr float kItemH     = 28.0f;
    static constexpr float kSepH      =  9.0f;
    static constexpr float kPadX      = 10.0f;
    static constexpr float kMinPopupW = 140.0f;

    float popupHeight() const;
    float popupWidth()  const;
};
