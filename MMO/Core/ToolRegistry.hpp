// The authoritative store for tool metadata and execution bindings.
#pragma once
#include "ToolDescriptor.hpp"
#include "ToolRegistrySnapshot.hpp"
#include "ToolExecutionMode.hpp"
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>
#include <tuple>

struct CommandResult;

namespace GRIM::MMO {

using ToolHandler = std::function<CommandResult(const std::string&)>;

struct ToolRegistration {
    ToolDescriptor descriptor;
    ToolHandler handler;
};

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

    // Metadata and a callable are published together. Invalid definitions throw;
    // name/alias collisions return false. Replacement must retain the provider.
    bool registerTool(const ToolDescriptor& descriptor, ToolHandler handler, bool replace = false,
                      const ProviderDescriptor& provider = {ToolProviderType::Builtin, "builtin", {}, {}});
    bool registerSimple(const std::string& tool_id, const std::string& description,
                        const std::string& category, ToolHandler handler, bool is_informational = true);
    // Publish provider identity and tools in the same transaction, including zero tools.
    bool publishProvider(const ProviderDescriptor&, const std::vector<ToolRegistration>&,
                         const std::string& old_instance = {});
    ToolRegistrySnapshot snapshot() const;
    // Host publication only. Resource changes do not invalidate tool selections.
    bool publishResource(const ResourceDescriptor& resource);
    // Atomically replace one resource kind for an exact provider generation.
    bool replaceResources(const ProviderDescriptor& provider, const std::string& kind,
                          const std::vector<ResourceDescriptor>& resources);
    void removeResource(const ResourceDescriptor& resource);
    uint64_t resourceVersion() const;
    void unregisterInstance(const std::string& provider, const std::string& instance);
    // Update documentation/policy without losing the callable or execution stats.
    // An inspection copy may update only the provider generation it came from.
    bool updateMetadata(const ToolDescriptor& descriptor, const ProviderDescriptor* owner = nullptr);
    bool updateMetadata(const RegisteredTool& tool) { return updateMetadata(tool, &tool.provider); }
    // Empty provider is for host callers; plugin callers must supply ownership.
    bool unregisterTool(const std::string& tool_id, const std::string& provider = {},
                        ToolProviderType type = ToolProviderType::Plugin, const std::string& instance = {});
    void unregisterByProvider(const std::string& provider, ToolProviderType type = ToolProviderType::Plugin);

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
    ToolRegistry() = default;
    bool registerBatch(const ProviderDescriptor&, const std::vector<ToolRegistration>&, bool replace, bool publish);
    bool replaceProvider(const ProviderDescriptor&, const std::string&, const std::vector<ToolRegistration>&);
    using Provider = std::shared_ptr<const ProviderDescriptor>;
    struct Entry { ToolDescriptor descriptor; ToolHandler handler; Provider provider; };
    using Entries = std::unordered_map<std::string, std::shared_ptr<Entry>>;
    using Aliases = std::unordered_map<std::string, std::string>;
    static bool buildAliases(const Entries& entries, Aliases& aliases);
    std::string resolveLocked(const std::string& name) const;
    mutable std::mutex mutex_;
    Entries tools_;
    std::map<std::pair<ToolProviderType, std::string>, Provider> providers_;
    Aliases alias_index_;
    uint64_t version_ = 0;
    uint64_t resourceVersion_ = 0;
    using ResourceKey = std::tuple<ToolProviderType, std::string, std::string>;
    std::map<ResourceKey, ResourceDescriptor> resources_;
    void pruneResourcesLocked();
    std::vector<ToolRegistryListener*> listeners_;
};
} // namespace GRIM::MMO
