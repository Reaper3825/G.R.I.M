// Multi-Model Orchestration (MMO) - ToolRegistry
// Unified, authoritative registry of all tools the model can select.
//
// Unifies:
//   - CommandRegistry::ToolMetadata (built-in commands)
//   - Plugin-registered commands (currently bypass CommandRegistry)
//
// One canonical registry that the model, action policy, hot-swap
// manager, and training pipeline all share. Commands and plugins
// are not separate worlds — they are all tools.
//
// Hot-swap rules:
//   - Built-in tools are static (registered at startup).
//   - Plugin tools are hot-swappable (load/unload/reload).
//   - Registry updates from plugin state changes are atomic.
//   - In-flight tool calls pin the tool_id + version they started with.
//   - A hot reload invalidates cached model-visible tool prompts.
//======================================================//
#pragma once

#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
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

// =========================================================
// ToolRegistryListener — notified on atomic registry changes
// =========================================================
struct ToolRegistryListener {
    virtual ~ToolRegistryListener() = default;
    virtual void onToolRegistered(const std::string& tool_id) = 0;
    virtual void onToolUnregistered(const std::string& tool_id) = 0;
    virtual void onToolUpdated(const std::string& tool_id) = 0;
};

// =========================================================
// ToolRegistry
//
// Usage:
//   auto& reg = ToolRegistry::instance();
//   reg.registerTool(descriptor);
//   auto tool = reg.getTool("open");
//   auto prompt = reg.generateCompactPrompt();
//
// Thread-safe: all public methods serialized under mutex.
// =========================================================
class ToolRegistry {
public:
    static ToolRegistry& instance();

    // ─── Registration ─────────────────────────────────────

    // Register or replace a tool descriptor.
    void registerTool(const ToolDescriptor& descriptor);

    // Register a simple built-in tool with minimal metadata.
    void registerSimple(const std::string& tool_id,
                        const std::string& description,
                        const std::string& category,
                        bool is_informational = true);

    // Unregister a tool (e.g. plugin unload).
    void unregisterTool(const std::string& tool_id);

    // ─── Atomic batch updates (for plugin hot-swap) ───────

    // Atomically register multiple tools from a plugin load.
    void registerBatch(const std::string& plugin_name,
                       const std::vector<ToolDescriptor>& tools);

    // Atomically unregister all tools from a plugin unload.
    void unregisterByProvider(const std::string& plugin_name);

    // ─── Query ────────────────────────────────────────────

    std::optional<ToolDescriptor> getTool(const std::string& tool_id) const;
    std::vector<ToolDescriptor>   getAllTools() const;
    std::vector<ToolDescriptor>   getByCategory(const std::string& category) const;
    std::vector<ToolDescriptor>   getByCapabilityTag(const std::string& tag) const;
    std::vector<ToolDescriptor>   getActionTools() const;
    std::vector<ToolDescriptor>   getInformationalTools() const;
    std::vector<std::string>      getCategories() const;
    bool                          isRegistered(const std::string& tool_id) const;
    size_t                        toolCount() const;

    // Resolve a tool_id from an alias or keyword.
    std::optional<std::string> resolveAlias(const std::string& alias) const;

    // ─── Model-facing prompt generation ───────────────────

    // Generate compact prompt for the model (tool_id + description).
    std::string generateCompactPrompt() const;

    // Generate full prompt with parameters and examples.
    std::string generateFullPrompt() const;

    // Prompt cache invalidation version counter.
    // Incremented on every registry mutation. The model/router
    // should regenerate cached tool summaries when this changes.
    uint64_t version() const;

    // ─── Analytics ────────────────────────────────────────

    void recordSuccess(const std::string& tool_id);
    void recordFailure(const std::string& tool_id);

    // ─── Listeners ────────────────────────────────────────

    void addListener(ToolRegistryListener* listener);
    void removeListener(ToolRegistryListener* listener);

private:
    ToolRegistry() = default;

    void notifyRegistered(const std::string& tool_id);
    void notifyUnregistered(const std::string& tool_id);
    void notifyUpdated(const std::string& tool_id);

    mutable std::mutex mutex_;
    std::unordered_map<std::string, ToolDescriptor> tools_;
    std::unordered_map<std::string, std::string> alias_index_; // alias -> tool_id
    uint64_t version_ = 0;
    std::vector<ToolRegistryListener*> listeners_;
};

} // namespace GRIM::MMO
