#pragma once
#include "ui/ui_object_tools.hpp"

namespace GRIM::HostObjectActions {
inline nlohmann::json tools() { return UI::ObjectTools::ids; }

inline nlohmann::json execute(const std::string& id, const std::string& target,
                              const std::function<bool()>& cancelled) {
    try {
        auto result = UI::ObjectTools::execute(id, target, MMO::ToolRegistry::instance().version(), cancelled);
        result["status"] = "completed";
        return result;
    } catch (const UI::CompletionUnknown& error) {
        return {{"status", "unknown"}, {"error", error.what()}, {"feedback", error.what()}};
    }
}
}
