#pragma once
#include "ProviderRegistrySnapshot.hpp"
#include "../../../core/plugin_api.hpp"
#include "../../../ui/ui_panel_service.hpp"
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>
#include <unordered_map>

namespace PluginAPI { class ProviderBatch; }

struct CommandResult;

namespace GRIM::MMO {

class ToolRegistry;

using ToolHandler = std::function<CommandResult(const std::string&)>;

using CompletionToolHandler = std::function<CommandResult(const std::string&, bool wait_for_completion)>;

struct ToolRegistration {
    ToolDescriptor descriptor;
    ToolHandler handler;
    CompletionToolHandler completion_handler;
};

class ProviderRegistry {
public:
    struct CallGate {
        enum class State { Initializing, Ready, Stopping };
        std::mutex mutex;
        std::condition_variable idle;
        State state = State::Initializing;
        size_t active = 0;
    };
    struct NativeState {
        explicit NativeState(const ProviderDescriptor& descriptor) : identity(descriptor) {}
        const ProviderDescriptor& identity;
        std::unique_ptr<void, void (*)(void*)> module{nullptr, +[](void*) {}};
        void (*shutdown)() = nullptr;
        GrimPluginHandle handle = 0;
        CallGate gate;
        std::shared_ptr<GRIM::UI::ProviderPanelLease> panelLease = std::make_shared<GRIM::UI::ProviderPanelLease>();
        std::mutex panelRequestsMutex;
        std::unordered_map<uint64_t, GRIM::UI::PanelServiceRequest> panelRequests;
        GrimSystemTickCallback tick = nullptr;
        void* tickData = nullptr;
    };
    std::shared_ptr<NativeState> nativeProvider(GrimPluginHandle handle) const;
    std::vector<std::shared_ptr<NativeState>> readyNativeProviders() const;

    static ProviderRegistry& instance();

#ifdef GRIM_PROVIDER_REGISTRY_TESTS
    bool publish(const ProviderDescriptor&, const std::vector<ToolRegistration>&);
    void retire(const std::string& provider, const std::string& instance);
#endif

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
    friend class ::PluginAPI::ProviderBatch;
    explicit ProviderRegistry(ToolRegistry& tools) : tools_(tools) {}

    using Provider = std::shared_ptr<const ProviderDescriptor>;
    struct Entry { ToolDescriptor descriptor; ToolHandler handler; CompletionToolHandler completion_handler; Provider provider; };
    using Resources = std::map<std::string, ResourceDescriptor>;
    struct ProviderRecord {
        Provider descriptor;
        std::unordered_map<std::string, std::shared_ptr<Entry>> tools;
        Resources resources;
        std::optional<NativeState> native;
        bool published = false;
    };
    using Record = std::shared_ptr<ProviderRecord>;
    void publishPrepared(const std::vector<Record>& records);
    void unpublishPrepared(const std::vector<Record>& records);
    void validateNativeCandidateLocked(const ProviderRecord& candidate) const;
    static void prepareTools(ProviderRecord& record, const std::vector<ToolRegistration>& registrations);
    static Resources makeResourceBatch(const std::string& kind, const std::vector<ResourceDescriptor>& resources);
    ProviderRecord* providerResourcesLocked(const ProviderDescriptor& provider);
    ProviderRecord* providerForInstanceLocked(const std::string& instance);

    // Provider records and tool indexes share the ToolRegistry mutex.
    ToolRegistry& tools_;
    std::map<std::string, Record> records_;
    GrimPluginHandle nextHandle_ = 1;
    uint64_t resourceVersion_ = 0;
};

} // namespace GRIM::MMO
