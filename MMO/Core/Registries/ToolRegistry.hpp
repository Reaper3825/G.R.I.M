// Global lookup and dispatch for provider-owned tools.
#pragma once
#include "../Descriptors/RegisteredTool.hpp"
#include "ProviderRegistry.hpp"
#include "../ToolExecutionMode.hpp"
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace GRIM::MMO {

struct ToolInvocationOptions {
    ToolExecutionMode mode = ToolExecutionMode::Production;
    std::optional<uint64_t> expected_registry_version;
    bool require_canonical_model = false;
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

    // Update documentation/policy without losing the callable or execution stats.
    // An inspection copy may update only the provider generation it came from.
    bool updateMetadata(const ToolDescriptor& descriptor, const ProviderDescriptor& owner);
    bool updateMetadata(const RegisteredTool& tool) { return updateMetadata(tool, tool.provider); }
    bool unregisterTool(const std::string& tool_id, const std::string& provider,
                        const std::string& instance);

    // Explicit selected-tool execution. Does not infer intent or run policy gates.
    // Handlers run outside the registry mutex and can query/register other tools.
    // Production uses the registered handler. Sandbox is reserved and returns
    // an explicit not-implemented result without invoking the handler.
    // Optional selection checks run under the same lock as handler selection.
    CommandResult invoke(const std::string& name, const std::string& arguments,
                         const ToolInvocationOptions& options = {});
    bool isExecutable(const std::string& tool_id) const;
    std::vector<std::string> getExecutableNames(bool include_aliases = false) const;

    std::optional<RegisteredTool> getTool(const std::string& tool_id) const;
    // Snapshots and their filtered subsets are ordered alphabetically by tool ID.
    std::vector<ToolDescriptor> getAllTools() const;
    std::vector<RegisteredTool> getRegisteredTools() const;
    // Loaded tools explicitly classified for model use. Host discovery is unfiltered.
    std::vector<ToolDescriptor> getModelTools() const;
    std::vector<ToolDescriptor> getByCategory(const std::string& category) const;
    std::vector<ToolDescriptor> getByCapabilityTag(const std::string& tag) const;
    std::vector<ToolDescriptor> getActionTools() const;
    std::vector<ToolDescriptor> getInformationalTools() const;
    std::vector<std::string> getCategories() const;
    bool isRegistered(const std::string& tool_id) const;
    size_t toolCount() const;
    std::optional<std::string> resolveAlias(const std::string& alias) const;
    uint64_t version() const;
    void addListener(ToolRegistryListener* listener);
    void removeListener(ToolRegistryListener* listener);

private:
    friend class ProviderRegistry;
    ToolRegistry() = default;
    using Entry = ProviderRegistry::Entry;
    using ToolIndex = std::unordered_map<std::string, Entry*>;
    using Aliases = std::unordered_map<std::string, std::string>;
    static bool buildAliases(const ToolIndex& entries, Aliases& aliases);
    std::string resolveLocked(const std::string& name) const;
    mutable std::mutex mutex_;
    ProviderRegistry providers_{*this};
    ToolIndex tool_index_; // Non-owning. Provider records retain the entries.
    Aliases alias_index_;
    uint64_t version_ = 0;
    std::vector<ToolRegistryListener*> listeners_;
};
} // namespace GRIM::MMO
