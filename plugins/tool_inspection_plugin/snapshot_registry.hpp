#pragma once
#include "core/plugin_api.hpp"
#include "MMO/Core/ToolDescriptorJson.hpp"
#include "commands/command_result.hpp"
#include <algorithm>
#include <optional>
#include <stdexcept>

namespace inspection {
inline const GrimPluginAPI* host = nullptr;
inline std::vector<GRIM::MMO::ToolDescriptor> snapshot(const char* id = nullptr) {
    struct OwnedString {
        const char* text;
        ~OwnedString() { if (text) host->string_free(host, text); }
    } text{host->get_tool_snapshot(host, id)};
    if (!text.text) throw std::runtime_error("Tool registry snapshot unavailable");
    const auto value = nlohmann::json::parse(text.text);
    if (value.at("format") != "grim-tool-snapshot" || value.at("version") != 1)
        throw std::runtime_error("Unsupported tool registry snapshot");
    return value.at("tools").get<std::vector<GRIM::MMO::ToolDescriptor>>();
}
struct Registry {
    auto getAllTools() const { return snapshot(); }
    auto getByCategory(const std::string& category) const {
        auto tools = snapshot();
        tools.erase(std::remove_if(tools.begin(), tools.end(), [&](const auto& tool) {
            return tool.category != category;
        }), tools.end());
        return tools;
    }
    std::optional<GRIM::MMO::ToolDescriptor> getTool(const std::string& id) const {
        auto tools = snapshot(id.c_str());
        if (tools.empty()) return std::nullopt;
        return std::move(tools.front());
    }
};
inline Registry snapshotRegistry() { return {}; }
}
CommandResult cmdListTools(const std::string&);
CommandResult cmdToolInfo(const std::string&);
CommandResult cmdToolStats(const std::string&);
