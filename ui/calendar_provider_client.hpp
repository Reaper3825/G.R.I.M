#pragma once
#include "MMO/Core/Registries/ToolRegistry.hpp"
#include "commands/command_result.hpp"
#include <nlohmann/json.hpp>
namespace GRIM::UI {
// Copy results through ordinary registry dispatch; never retain a DLL callback or provider object.
inline nlohmann::json callSchedule(const std::string& operation, const nlohmann::json& args) {
    const auto result = MMO::ToolRegistry::instance().invoke("schedule." + operation, args.dump());
    if (!result.success) throw std::runtime_error(result.message.empty() ? "Schedule provider unavailable" : result.message);
    return nlohmann::json::parse(result.structuredOutputJson);
}
}
