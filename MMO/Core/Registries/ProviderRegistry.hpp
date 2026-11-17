#pragma once
#include "ProviderRegistrySnapshot.hpp"
#include <functional>
#include <map>
#include <memory>
#include <unordered_map>

struct CommandResult;

namespace GRIM::MMO {

class ToolRegistry;

using ToolHandler = std::function<CommandResult(const std::string&)>;

struct ToolRegistration {
    ToolDescriptor descriptor;
    ToolHandler handler;
};

class ProviderRegistry {
public:
    static ProviderRegistry& instance();

    // Register a live provider and its manifest tools in one operation.
    bool publish(const ProviderDescriptor&, const std::vector<ToolRegistration>&);
    void retire(const std::string& provider, const std::string& instance);

    ProviderRegistrySnapshot snapshot() const;
    bool hasChanged(const ProviderRegistrySnapshot& snapshot) const;

    // Resource changes do not invalidate tool selections.
    // Single-resource operations address a published generation by its unique instance ID.
    bool publishResource(const std::string& instance, const ResourceDescriptor& resource);
    bool replaceResources(const ProviderDescriptor& provider, const std::string& kind,
                          const std::vector<ResourceDescriptor>& resources);
    void removeResource(const std::string& instance, const ResourceDescriptor& resource);
    uint64_t resourceVersion() const;

private:
    friend class ToolRegistry;
    explicit ProviderRegistry(ToolRegistry& tools) : tools_(tools) {}

    using Provider = std::shared_ptr<const ProviderDescriptor>;
    struct Entry { ToolDescriptor descriptor; ToolHandler handler; Provider provider; };
    using Resources = std::map<std::string, ResourceDescriptor>;
    struct ProviderRecord {
        Provider descriptor;
        std::unordered_map<std::string, std::shared_ptr<Entry>> tools;
        Resources resources;
    };
    static Resources makeResourceBatch(const std::string& kind, const std::vector<ResourceDescriptor>& resources);
    Resources* providerResourcesLocked(const ProviderDescriptor& provider);
    Resources* publishedResourcesForInstanceLocked(const std::string& instance);

    // Provider records and tool indexes share the ToolRegistry mutex.
    ToolRegistry& tools_;
    std::map<std::string, std::unique_ptr<ProviderRecord>> records_;
    uint64_t resourceVersion_ = 0;
};

} // namespace GRIM::MMO
