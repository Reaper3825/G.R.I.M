// Tool metadata shared by registration, discovery, and policy consumers.
// Independent of ToolRegistry and host/plugin execution interfaces.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace GRIM::MMO {

// =========================================================
// ToolProviderType — who provides this tool
// =========================================================
enum class ToolProviderType : uint8_t {
    Builtin,    // Compiled-in command
    Plugin      // Hot-loadable DLL plugin
};

// =========================================================
// ToolSwapState — hot-swap lifecycle
// =========================================================
enum class ToolSwapState : uint8_t {
    Loaded,
    Loading,
    Unloading,
    Unavailable
};

// =========================================================
// ToolParameter — describes one parameter of a tool
// =========================================================
struct ToolParameter {
    std::string name;
    std::string type;           // "string", "file", "app", "number", "path", "url", etc.
    std::string description;
    bool        required = false;
};

// =========================================================
// ToolDescriptor — the canonical per-tool record
//
// Fields from MMO plan Section "Unified ToolRegistry":
//   tool_id, display_name, provider_type, provider_name, version,
//   description, usage, examples, category, permission_bits,
//   needs_confirmation, argument_schema, preconditions,
//   context_requirements, affordance_tokens, aliases, keywords,
//   capability_tags, hot_swap_state, success_stats.
// =========================================================
struct ToolDescriptor {
    // Identity
    std::string           tool_id;          // stable canonical identifier
    std::string           display_name;
    ToolProviderType      provider_type = ToolProviderType::Builtin;
    std::string           provider_name;    // plugin name or "builtin"
    std::string           version;          // semver or monotonic counter

    // Documentation
    std::string           description;
    std::string           usage;
    std::vector<std::string> examples;
    std::string           category;         // "action", "information", "system", "ui"

    // Policy
    uint32_t              permission_bits = 0;  // GrimPermission flags
    bool                  needs_confirmation = false;
    bool                  is_informational   = true;

    // Arguments
    std::vector<ToolParameter> parameters;
    std::vector<std::string>   preconditions;

    // Context / affordance
    std::vector<std::string> context_requirements;  // e.g. "app:blender", "ui:selected_object"
    std::vector<std::string> affordance_tokens;     // e.g. "<press_hold_drag>", "<open_app>"
    std::vector<std::string> aliases;
    std::vector<std::string> keywords;
    std::vector<std::string> capability_tags;

    // Hot-swap
    ToolSwapState         swap_state = ToolSwapState::Loaded;

    // Stats
    int                   usage_count   = 0;
    int                   success_count = 0;
    int                   failure_count = 0;
    float                 success_rate  = 1.0f;
};

} // namespace GRIM::MMO
