#include "provider_defined_panel.hpp"
#include "panel_definition.hpp"
#include "primitives/ui_button.hpp"
#include "primitives/ui_inputbox.hpp"
#include "overlay_renderer.hpp"
#include "core/input_parser.hpp"
#include <algorithm>

namespace GRIM::UI {
ProviderDefinedPanel::ProviderDefinedPanel(nlohmann::json definition, std::shared_ptr<ProviderPanelLease> owner)
    : UIPanel(definition.at("title").get<std::string>(), true, definition.at("panel_id").get<std::string>()),
      owner_(std::move(owner)), scroll_(std::make_shared<UIScrollBox>()) {
    if (!owner_) throw std::invalid_argument("Provider panel requires an owner");
    setPosition(240, 180);
    setSize(definition.at("width").get<float>(), definition.at("height").get<float>());
    UIPanel::setVisible(false);
    addChild(scroll_);
    applyDefinition(std::move(definition));
}

void ProviderDefinedPanel::applyDefinition(nlohmann::json definition) {
    validatePanelDefinition(definition);
    if (definition.at("panel_id") != getPanelKey()) throw std::invalid_argument("Panel identity cannot change");
    std::map<std::string, std::shared_ptr<Widget>> next;
    for (const auto& spec : definition.at("widgets")) {
        const auto id = spec.at("id").get<std::string>();
        const auto type = spec.at("type").get<std::string>();
        if (type == "label") continue;
        const auto found = widgets_.find(id);
        std::shared_ptr<Widget> widget;
        if (found != widgets_.end()) {
            const bool matches = (type == "button" && dynamic_cast<UIButton*>(found->second.get())) ||
                                 (type == "input" && dynamic_cast<UIInputBox*>(found->second.get()));
            if (!matches) throw std::invalid_argument("Widget type cannot change for an existing ID");
            widget = found->second;
        } else if (type == "button") {
            widget = std::make_shared<UIButton>(spec.at("text").get<std::string>(), [this, id] { interact(id); });
        } else {
            auto input = std::make_shared<UIInputBox>();
            input->setText(spec.at("text").get<std::string>());
            widget = std::move(input);
        }
        next.emplace(id, std::move(widget));
    }
    for (const auto& [id, widget] : widgets_) {
        if (!next.count(id)) { widget->cancelInput(); removeChild(widget.get()); }
    }
    for (const auto& [id, widget] : next) if (!widgets_.count(id)) addChild(widget);
    for (const auto& spec : definition.at("widgets")) {
        const auto id = spec.at("id").get<std::string>();
        if (spec.at("type") == "button") {
            auto& button = static_cast<UIButton&>(*next.at(id));
            button.setText(spec.at("text").get<std::string>());
            button.setColors(spec.at("color").get<uint32_t>(), UITheme::Colors::WidgetBgHover, UITheme::Colors::WidgetBgActive);
        } else if (spec.at("type") == "input" && widgets_.count(id)) {
            const auto& old = definition_.at("widgets");
            const auto previous = std::find_if(old.begin(), old.end(), [&](const auto& item) { return item.at("id") == id; });
            if (previous == old.end()) throw std::logic_error("Missing previous input definition");
            if (previous->at("value_revision") != spec.at("value_revision"))
                static_cast<UIInputBox&>(*next.at(id)).setText(spec.at("text").get<std::string>());
        }
    }
    widgets_ = std::move(next);
    setTitle(definition.at("title").get<std::string>());
    definition_ = std::move(definition);
    layout();
}

void ProviderDefinedPanel::emit(nlohmann::json event) {
    std::lock_guard lock(owner_->eventsMutex);
    if (owner_->retired) return;
    if (owner_->events.size() >= 4096) throw std::runtime_error("Provider panel event queue exhausted");
    event["panel_id"] = getPanelKey();
    event["runtime_id"] = getPanelID();
    event["revision"] = definition_.at("revision");
    owner_->events.push_back(std::move(event));
}

void ProviderDefinedPanel::setVisible(bool visible) {
    const bool changed = visible != isVisible();
    UIPanel::setVisible(visible);
    if (changed) emit({{"type", "visibility"}, {"visible", visible}});
}

void ProviderDefinedPanel::interact(const std::string& id) {
    for (const auto& spec : definition_.at("widgets")) if (spec.at("id") == id) {
        auto values = nlohmann::json::object();
        for (const auto& input : definition_.at("widgets")) if (input.at("type") == "input") {
            const auto key = input.at("id").get<std::string>();
            values[key] = {{"value", static_cast<UIInputBox&>(*widgets_.at(key)).getText()},
                           {"value_revision", input.at("value_revision")}};
        }
        emit({{"type", "click"}, {"widget_id", id}, {"action", spec.at("action")}, {"values", std::move(values)}});
        return;
    }
    throw std::logic_error("Interaction targets a missing widget");
}

void ProviderDefinedPanel::layout() {
    const auto content = getContentRect();
    scroll_->setPosition(content.origin);
    scroll_->setSize(content.size);
    scroll_->setContentHeight(definition_.at("content_height").get<float>());
    for (const auto& spec : definition_.at("widgets")) {
        if (spec.at("type") == "label") continue;
        auto& widget = widgets_.at(spec.at("id").get<std::string>());
        const auto& r = spec.at("rect");
        widget->setPosition(content.origin.x + r[0].get<float>(),
                            content.origin.y + r[1].get<float>() - scroll_->getScrollOffset());
        widget->setSize(r[2].get<float>(), r[3].get<float>());
    }
}

void ProviderDefinedPanel::update(const InputState& input, float dt) {
    if (!isVisible()) return;
    const auto size = getContentRect().size;
    if (reportedSize_.x != size.x || reportedSize_.y != size.y) {
        reportedSize_ = size;
        emit({{"type", "geometry"}, {"width", size.x}, {"height", size.y}});
    }
    layout();
    std::map<std::string, std::string> before;
    for (const auto& spec : definition_.at("widgets"))
        if (spec.at("type") == "input") {
            const auto id = spec.at("id").get<std::string>();
            before.emplace(id, static_cast<UIInputBox&>(*widgets_.at(id)).getText());
        }
    UIPanel::update(input, dt);
    for (const auto& spec : definition_.at("widgets"))
        if (spec.at("type") == "input") {
            const auto id = spec.at("id").get<std::string>();
            const auto& value = static_cast<UIInputBox&>(*widgets_.at(id)).getText();
            if (value != before.at(id))
                emit({{"type", "change"}, {"widget_id", id}, {"value", value},
                      {"value_revision", spec.at("value_revision")}, {"action", spec.at("action")}});
        }
}

bool ProviderDefinedPanel::drawOverlay(OverlayRenderer& renderer) {
    if (!UIPanel::drawOverlay(renderer)) return false;
    layout();
    scroll_->drawOverlay(renderer, getPosition());
    const auto content = getContentRect();
    for (const auto& spec : definition_.at("widgets")) {
        if (spec.at("type") != "label") {
            widgets_.at(spec.at("id").get<std::string>())->drawOverlay(renderer, getPosition());
            continue;
        }
        const auto& r = spec.at("rect");
        const Vec2 origin{content.origin.x + r[0].get<float>(),
                          content.origin.y + r[1].get<float>() - scroll_->getScrollOffset()};
        renderer.pushClipRect(origin, {r[2].get<float>(), r[3].get<float>()});
        renderer.drawText(origin, spec.at("text").get<std::string>(), spec.at("color").get<uint32_t>());
        renderer.popClipRect();
    }
    renderer.popClipRect();
    return true;
}
}
