#pragma once
#include "MMO/Core/ToolRegistry.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace GRIM::WorldLab::UITools {
using Json = nlohmann::json;
inline void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
inline bool sameProvider(const MMO::ToolDescriptor& a, const MMO::ToolDescriptor& b) {
    return a.provider_name == b.provider_name && a.provider_instance == b.provider_instance &&
           a.provider_type == b.provider_type;
}
inline MMO::ToolDescriptor descriptor(const std::string& id, uint64_t version) {
    auto& registry = MMO::ToolRegistry::instance();
    const auto tool = registry.getTool(id);
    require(registry.version() == version && tool && tool->tool_id == id &&
            tool->usage_type == MMO::ToolUsage::Model && tool->swap_state == MMO::ToolSwapState::Loaded,
            "Tool unavailable or registry changed; run ui.list again");
    require(tool->argument_format == "json", "Unsupported UI tool argument format");
    return *tool;
}
struct CompletionUnknown : std::runtime_error { using std::runtime_error::runtime_error; };
}
