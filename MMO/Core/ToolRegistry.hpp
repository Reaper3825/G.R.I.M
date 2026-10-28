// The authoritative store for tool metadata and execution bindings.
#pragma once
#include "ToolDescriptor.hpp"
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

struct CommandResult;

namespace GRIM::MMO {

using ToolHandler = std::function<CommandResult(const std::string&)>;

struct ToolRegistration {
    ToolDescriptor descriptor;
    ToolHandler handler;
};

struct ToolRegistryListener {
    virtual ~ToolRegistryListener() = default;
    virtual void onToolRegistered(const std::string& tool_id) = 0;
    virtual void onToolUnregistered(const std::string& tool_id) = 0;
    virtual void onToolUpdated(const std::string& tool_id) = 0;
};

class ToolRegistry {
public:
    static ToolRegistry& instance();

    // Metadata and a callable are published together. Invalid definitions throw;
    // name/alias collisions return false. Replacement must retain the provider.
    bool registerTool(const ToolDescriptor& descriptor, ToolHandler handler, bool replace = false);
    bool registerSimple(const std::string& tool_id, const std::string& description,
                        const std::string& category, ToolHandler handler, bool is_informational = true);
    bool registerBatch(const std::string& plugin_name, const std::vector<ToolRegistration>& tools,
                       bool replace = false);
    // Atomically replace one loaded generation, including removing vanished tools.
    bool replaceProvider(const std::string& provider, const std::string& old_instance,
                         const std::vector<ToolRegistration>& tools);
    void unregisterInstance(const std::string& provider, const std::string& instance);
    // Update documentation/policy without losing the callable or execution stats.
    bool updateMetadata(const ToolDescriptor& descriptor);
    // Empty provider is for host callers; plugin callers must supply ownership.
    bool unregisterTool(const std::string& tool_id, const std::string& provider = {},
                        ToolProviderType type = ToolProviderType::Plugin, const std::string& instance = {});
    void unregisterByProvider(const std::string& provider, ToolProviderType type = ToolProviderType::Plugin);

    // Explicit selected-tool execution. Does not infer intent or run policy gates.
    // Handlers run outside the registry mutex and can query/register other tools.
    CommandResult invoke(const std::string& tool_id, const std::string& arguments);
    bool isExecutable(const std::string& tool_id) const;
    std::vector<std::string> getExecutableNames(bool include_aliases = false) const;

    std::optional<ToolDescriptor> getTool(const std::string& tool_id) const;
    std::vector<ToolDescriptor> getAllTools() const;
    std::vector<ToolDescriptor> getByCategory(const std::string& category) const;
    std::vector<ToolDescriptor> getByCapabilityTag(const std::string& tag) const;
    std::vector<ToolDescriptor> getActionTools() const;
    std::vector<ToolDescriptor> getInformationalTools() const;
    std::vector<std::string> getCategories() const;
    bool isRegistered(const std::string& tool_id) const;
    size_t toolCount() const;
    std::optional<std::string> resolveAlias(const std::string& alias) const;
    std::string generateCompactPrompt() const;
    std::string generateFullPrompt() const;
    uint64_t version() const;
    void addListener(ToolRegistryListener* listener);
    void removeListener(ToolRegistryListener* listener);

private:
    ToolRegistry() = default;
    struct Entry { ToolDescriptor descriptor; ToolHandler handler; };
    using Entries = std::unordered_map<std::string, std::shared_ptr<Entry>>;
    using Aliases = std::unordered_map<std::string, std::string>;
    static bool buildAliases(const Entries& entries, Aliases& aliases);
    std::string resolveLocked(const std::string& name) const;
    mutable std::mutex mutex_;
    Entries tools_;
    Aliases alias_index_;
    uint64_t version_ = 0;
    std::vector<ToolRegistryListener*> listeners_;
};
} // namespace GRIM::MMO
