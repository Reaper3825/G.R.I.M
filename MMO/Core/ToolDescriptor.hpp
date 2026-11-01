// Tool metadata shared by registration, discovery, and policy consumers.
// Independent of ToolRegistry and host/plugin execution interfaces.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace GRIM::MMO {

// Intended consumer, independent of effects, permissions, and provider.
enum class ToolUsage : uint32_t {
    Debug = 0,  // Developer diagnostics and manual inspection.
    Model = 1,  // Capabilities exposed to the model.
    System = 2, // Host-controlled operations.
    Ignore = 3  // Retained registration excluded from model discovery.
};
constexpr bool validToolUsage(ToolUsage usage) {
    return usage == ToolUsage::Debug || usage == ToolUsage::Model ||
           usage == ToolUsage::System || usage == ToolUsage::Ignore;
}
constexpr const char* toolUsageName(ToolUsage usage) {
    switch (usage) {
        case ToolUsage::Debug: return "Debug";
        case ToolUsage::Model: return "Model";
        case ToolUsage::System: return "System";
        case ToolUsage::Ignore: return "Ignore";
    }
    return "Unknown";
}

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
    std::string           provider_instance; // host-owned loaded generation
    std::string           provider_path;    // host-owned plugin source file
    std::string           version;          // semver or monotonic counter

    // Documentation
    std::string           description;
    std::string           usage; // Model-facing action label/template, independent of usage_type.
    // Usage placeholder -> existing world type (t, o, c, s, r, d, k, f).
    // Labels only: no entity resolution, argument conversion, or dispatch.
    std::map<std::string, std::string> bindings;
    std::vector<std::string> examples;
    std::string           category;         // "action", "information", "system", "ui"

    // Policy
    ToolUsage             usage_type = ToolUsage::Model;
    uint32_t              permission_bits = 0;  // GrimPermission flags
    bool                  needs_confirmation = false;
    bool                  is_informational   = true;

    // Arguments
    std::string argument_format = "raw_text";
    std::string argument_schema_json;
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

inline bool validToolBindings(const ToolDescriptor& tool) {
    if (tool.bindings.size() > 256) return false;
    for (const auto& [placeholder, type] : tool.bindings) {
        if (placeholder.empty() || placeholder.find_first_not_of(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") != std::string::npos)
            return false;
        if (type != "t" && type != "o" && type != "c" && type != "s" &&
            type != "r" && type != "d" && type != "k" && type != "f") return false;
        if (tool.usage.find("{" + placeholder + "}") == std::string::npos) return false;
    }
    return true;
}

} // namespace GRIM::MMO
