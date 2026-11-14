#pragma once
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <nlohmann/json.hpp>

namespace GRIM::UI {
inline void validatePanelDefinition(const nlohmann::json& value) {
    using Json = nlohmann::json;
    const auto fields = [](const Json& object, std::initializer_list<const char*> names) {
        if (!object.is_object() || object.size() != names.size())
            throw std::invalid_argument("Panel definition has missing or unknown fields");
        for (const auto* name : names)
            if (!object.contains(name)) throw std::invalid_argument("Missing panel field: " + std::string(name));
    };
    const auto text = [](const Json& field, size_t maximum, bool empty) {
        const auto& s = field.get_ref<const std::string&>();
        if ((!empty && s.empty()) || s.size() > maximum || s.find('\0') != std::string::npos)
            throw std::invalid_argument("Invalid panel text");
    };
    const auto number = [](const Json& field, double minimum, double maximum) {
        if (!field.is_number()) throw std::invalid_argument("Panel geometry must be numeric");
        const double n = field.get<double>();
        if (!std::isfinite(n) || n < minimum || n > maximum)
            throw std::invalid_argument("Panel geometry out of range");
    };
    fields(value, {"version", "panel_id", "runtime_id", "revision", "title", "width", "height", "content_height", "widgets"});
    if (value.at("version") != 1 || !value.at("runtime_id").is_number_unsigned() ||
        !value.at("revision").is_number_unsigned() || value.at("revision").get<uint64_t>() == 0)
        throw std::invalid_argument("Invalid panel definition version or identity");
    text(value.at("panel_id"), 256, false);
    const auto& key = value.at("panel_id").get_ref<const std::string&>();
    if (key.find_first_of("\r\n") != std::string::npos) throw std::invalid_argument("Invalid panel key");
    text(value.at("title"), 512, false);
    number(value.at("width"), 300, 16384);
    number(value.at("height"), 200, 16384);
    number(value.at("content_height"), 0, 1000000);
    const auto& widgets = value.at("widgets");
    if (!widgets.is_array() || widgets.size() > 256) throw std::invalid_argument("Panel requires at most 256 widgets");
    std::set<std::string> ids;
    for (const auto& widget : widgets) {
        fields(widget, {"id", "type", "text", "rect", "color", "action", "value_revision"});
        text(widget.at("id"), 256, false);
        if (!ids.insert(widget.at("id").get<std::string>()).second) throw std::invalid_argument("Duplicate widget ID");
        const auto type = widget.at("type").get<std::string>();
        if (type != "label" && type != "button" && type != "input") throw std::invalid_argument("Unsupported panel widget");
        text(widget.at("text"), 16384, true);
        const auto& rect = widget.at("rect");
        if (!rect.is_array() || rect.size() != 4) throw std::invalid_argument("Widget rect requires x, y, width, height");
        for (const auto& n : rect) number(n, 0, 1000000);
        if (!widget.at("color").is_number_unsigned() || widget.at("color").get<uint64_t>() > UINT32_MAX ||
            !widget.at("value_revision").is_number_unsigned() || !widget.at("action").is_object())
            throw std::invalid_argument("Invalid widget color, action or value revision");
    }
}
}
