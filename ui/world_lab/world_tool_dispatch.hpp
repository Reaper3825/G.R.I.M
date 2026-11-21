#pragma once
#include "MMO/Core/Registries/ToolRegistry.hpp"
#include "MMO/Core/ToolResultJson.hpp"
#include "commands/command_result.hpp"

namespace GRIM::WorldLab {
class WorldToolDispatch {
public:
    static nlohmann::json execute(const std::string& command, uint64_t actionId,
                                  uint64_t revision) {
        auto& registry = MMO::ToolRegistry::instance();
        nlohmann::json output = {{"action_id", actionId}, {"status", "failed"},
                                 {"feedback", MMO::formatToolResult(nullptr)}};
        if (command.empty() || command.size() > 512 ||
            command.find_first_of("\r\n\0", 0, 3) != std::string::npos)
            return output;

        const auto space = command.find(' ');
        const auto id = command.substr(0, space);
        output["tool_id"] = id;
        const auto tool = registry.getTool(id);
        if (!tool || tool->tool_id != id || tool->usage_type != MMO::ToolUsage::Model)
            return output;

        const auto start = space == std::string::npos ? space : command.find_first_not_of(' ', space);
        auto arguments = start == std::string::npos ? std::string{} : command.substr(start);
        if (arguments.empty() && tool->argument_format == "json") arguments = "{}";

        const auto result = registry.invoke(id, arguments,
            {.expected_registry_version = revision, .require_canonical_model = true, .wait_for_completion = true});
        output["status"] = result.at("status");
        output["feedback"] = MMO::formatToolResult(result.at("payload"));
        return output;
    }
};
}
