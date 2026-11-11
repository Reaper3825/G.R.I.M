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

    // Publish a plugin generation and its tools together, including zero-tool providers.
    bool publish(const ProviderDescriptor&, const std::vector<ToolRegistration>&,
                 const std::string& old_instance = {});
    void retire(const std::string& provider, const std::string& instance);

    ProviderRegistrySnapshot snapshot() const;
    bool hasChanged(const ProviderRegistrySnapshot& snapshot) const;

    // Host publication only. Resource changes do not invalidate tool selections.
    bool publishResource(const ProviderDescriptor& provider, const ResourceDescriptor& resource);
    bool replaceResources(const ProviderDescriptor& provider, const std::string& kind,
                          const std::vector<ResourceDescriptor>& resources);
    void removeResource(const ProviderDescriptor& provider, const ResourceDescriptor& resource);
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
    Resources* providerResourcesLocked(const ProviderDescriptor& provider);

    // Provider records and tool indexes share the ToolRegistry mutex.
    ToolRegistry& tools_;
    std::map<std::string, std::unique_ptr<ProviderRecord>> records_;
    uint64_t resourceVersion_ = 0;
};

} // namespace GRIM::MMO
