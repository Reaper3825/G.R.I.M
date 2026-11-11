#pragma once
#include "MMO/Core/ToolRegistry.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace GRIM::WorldLab::UITools {
using Json = nlohmann::json;
inline void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
inline bool sameProvider(const MMO::RegisteredTool& a, const MMO::RegisteredTool& b) {
    return a.provider.type == b.provider.type && a.provider.name == b.provider.name &&
           a.provider.instance == b.provider.instance;
}
inline MMO::RegisteredTool descriptor(const std::string& id, uint64_t version) {
    auto& registry = MMO::ToolRegistry::instance();
    const auto tool = registry.getTool(id);
    require(registry.version() == version && tool && tool->tool_id == id &&
            tool->usage_type == MMO::ToolUsage::Model,
            "Tool unavailable or registry changed; run ui.list again");
    require(tool->argument_format == "json", "Unsupported UI tool argument format");
    return *tool;
}
struct CompletionUnknown : std::runtime_error { using std::runtime_error::runtime_error; };
}
