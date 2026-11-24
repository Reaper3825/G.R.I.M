#include "ProviderRegistry.hpp"
#include "ToolRegistry.hpp"
#include "../ToolResultJson.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace GRIM::MMO {

struct ProviderRegistry::UpdateSubscription {
    struct Provider {
        std::weak_ptr<ProviderRecord> record;
        bool toolsChanged = false;
        bool resourcesChanged = false;
    };
    UpdateSink sink;
    std::map<std::string, Provider> providers;
};

ProviderRegistry::CallGate::ActiveCall::ActiveCall(CallGate& gate, Kind kind) : gate_(gate) {
    std::lock_guard lock(gate_.mutex);
    if (gate_.state == State::Ready ||
        (kind == Kind::HostService && gate_.state == State::Initializing)) {
        ++gate_.active;
        entered_ = true;
    }
}

ProviderRegistry::CallGate::ActiveCall::~ActiveCall() {
    if (!entered_) return;
    std::lock_guard lock(gate_.mutex);
    if (--gate_.active == 0) gate_.idle.notify_all();
}

nlohmann::json ProviderRegistry::NativeState::invokeTool(const std::string& id,
                                                        const std::string& arguments,
                                                        bool waitForCompletion) {
    CallGate::ActiveCall active(gate, CallGate::ActiveCall::Kind::Dispatch);
    if (!active) return makeToolResult(false, "Plugin is stopping", "ERR_TOOL_UNAVAILABLE", "");

    const GrimToolCall call{.arguments = arguments.c_str(), .arguments_size = arguments.size(),
                            .wait_for_completion = waitForCompletion ? 1u : 0u};
    const auto reply = dispatch(id.c_str(), &call);
    // Release the buffer through the DLL before the active call ends.
    const std::unique_ptr<const char, decltype(reply.release)> bytes(reply.release ? reply.json : nullptr,
                                                                     reply.release);
    if (reply.delivery == GRIM_OK && bytes && reply.size) {
        auto result = nlohmann::json::parse(bytes.get(), bytes.get() + reply.size, nullptr, false);
        if (!result.is_discarded()) return result;
    }
    return makeToolResult(false, "Plugin did not return a JSON result", "ERR_PLUGIN_RESULT", "");
}

ProviderRegistry& ProviderRegistry::instance() {
    return ToolRegistry::instance().providers_;
}

std::shared_ptr<ProviderRegistry::NativeState> ProviderRegistry::nativeProvider(GrimPluginHandle handle) const {
    std::lock_guard lock(tools_.mutex_);
    for (const auto& [name, record] : records_) {
        if (record->native && record->native->handle == handle)
            return {record, &*record->native};
    }
    return {};
}

std::vector<std::shared_ptr<ProviderRegistry::NativeState>> ProviderRegistry::readyNativeProviders() const {
    std::lock_guard lock(tools_.mutex_);
    std::vector<std::shared_ptr<NativeState>> providers;
    for (const auto& [name, record] : records_) {
        if (record->native && record->published)
            providers.emplace_back(record, &*record->native);
    }
    return providers;
}

void ProviderRegistry::prepareTools(ProviderRecord& record, std::vector<ToolDescriptor> descriptors) {
    for (auto& desc : descriptors) {
        if (desc.tool_id.empty())
            throw std::invalid_argument("Tool registration requires an ID");
        if (!validToolUsage(desc.usage_type) || !validToolBindings(desc) ||
            !validToolShortDescription(desc.short_description))
            throw std::invalid_argument("Invalid tool metadata");
        auto entry = std::make_shared<Entry>(Entry{std::move(desc), record.descriptor});
        const auto& id = entry->descriptor.tool_id;
        if (!record.tools.try_emplace(id, entry).second)
            throw std::invalid_argument("Duplicate tool ID: " + id);
    }
}

void ProviderRegistry::validateNativeCandidateLocked(const ProviderRecord& candidate) const {
    if (records_.contains(candidate.descriptor->name))
        throw std::runtime_error("Duplicate provider name: " + candidate.descriptor->name);
    auto tools = tools_.tool_index_;
    for (const auto& [name, staged] : records_) {
        if (staged->native && !staged->published) {
            for (const auto& [id, tool] : staged->tools) tools.emplace(id, tool.get());
        }
    }
    for (const auto& [id, tool] : candidate.tools) {
        if (!tools.emplace(id, tool.get()).second)
            throw std::runtime_error("Duplicate tool ID: " + id);
    }
    ToolRegistry::Aliases aliases;
    if (!ToolRegistry::buildAliases(tools, aliases))
        throw std::runtime_error("Duplicate tool alias");
}

#ifdef GRIM_PROVIDER_REGISTRY_TESTS
bool ProviderRegistry::publishResourceFixture(const ProviderDescriptor& provider) {
    if (provider.type != ToolProviderType::Plugin || provider.name.empty() ||
        provider.instance.empty()) return false;
    auto record = std::make_shared<ProviderRecord>();
    record->descriptor = std::make_shared<const ProviderDescriptor>(provider);
    {
        std::lock_guard lock(tools_.mutex_);
        if (!records_.emplace(provider.name, record).second) return false;
    }
    try {
        publishPrepared({record});
    } catch (...) {
        retire(provider.name, provider.instance);
        throw;
    }
    return true;
}

#endif

void ProviderRegistry::publishPrepared(const std::vector<Record>& records, bool startup) {
    if (records.empty()) {
        if (startup) {
            std::lock_guard lock(tools_.mutex_);
            startupProviders_.clear();
        }
        return;
    }
    std::vector<ProviderDescriptor> startupProviders;
    if (startup) {
        startupProviders.reserve(records.size());
        for (const auto& record : records) startupProviders.push_back(*record->descriptor);
    }
    std::vector<ToolRegistryListener*> listeners;
    std::vector<std::string> published;
    {
        std::lock_guard lock(tools_.mutex_);
        auto nextTools = tools_.tool_index_;
        // Staging handles admission. Build the final indexes before making any record visible.
        for (const auto& record : records) {
            for (const auto& [id, tool] : record->tools) {
                if (!nextTools.emplace(id, tool.get()).second)
                    throw std::logic_error("Prepared provider tool ID conflicts at commit: " + id);
                published.push_back(id);
            }
        }
        ToolRegistry::Aliases nextAliases;
        if (!ToolRegistry::buildAliases(nextTools, nextAliases))
            throw std::logic_error("Prepared provider tool alias conflicts at commit");
        listeners = tools_.listeners_;
        bool resourcesChanged = false;
        for (const auto& record : records) {
            record->published = true;
            if (record->native) {
                auto& gate = record->native->gate;
                std::lock_guard gateLock(gate.mutex);
                gate.state = CallGate::State::Ready;
            }
            resourcesChanged |= !record->resources.empty();
        }
        if (resourcesChanged) ++resourceVersion_;
        tools_.tool_index_.swap(nextTools);
        tools_.alias_index_.swap(nextAliases);
        if (startup) startupProviders_ = std::move(startupProviders);
        ++tools_.version_;
    }
    for (const auto& id : published) for (auto* listener : listeners) {
        try { listener->onToolRegistered(id); } catch (...) {}
    }
}

void ProviderRegistry::unpublishPrepared(const std::vector<Record>& records) {
    std::vector<ToolRegistryListener*> listeners;
    std::vector<std::string> removed;
    {
        std::lock_guard lock(tools_.mutex_);
        listeners = tools_.listeners_;
        for (const auto& record : records) {
            if (!record->published) continue;
            for (const auto& [id, tool] : record->tools) removed.push_back(id);
        }
        bool changed = false;
        bool resourcesChanged = false;
        for (const auto& record : records) {
            if (!record->published) continue;
            record->published = false;
            markProviderUpdatedLocked(*record, true, true);
            changed = true;
            for (const auto& [id, tool] : record->tools) tools_.tool_index_.erase(id);
            std::erase_if(tools_.alias_index_, [&](const auto& alias) {
                return record->tools.contains(alias.second);
            });
            resourcesChanged |= !record->resources.empty();
        }
        if (resourcesChanged) ++resourceVersion_;
        if (changed) ++tools_.version_;
    }
    for (const auto& id : removed) for (auto* listener : listeners) {
        try { listener->onToolUnregistered(id); } catch (...) {}
    }
}

std::vector<ToolDescriptor> ProviderRegistry::copyTools(const ProviderRecord& provider) {
    std::vector<ToolDescriptor> result;
    result.reserve(provider.tools.size());
    for (const auto& [id, entry] : provider.tools) result.push_back(entry->descriptor);
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.tool_id < b.tool_id;
    });
    return result;
}

std::vector<ResourceDescriptor> ProviderRegistry::copyResources(const ProviderRecord& provider) {
    std::vector<ResourceDescriptor> result;
    result.reserve(provider.resources.size());
    for (const auto& [id, resource] : provider.resources) result.push_back(resource);
    return result;
}

ProviderRegistrySnapshot ProviderRegistry::snapshot() const {
    std::lock_guard lock(tools_.mutex_);
    ProviderRegistrySnapshot result;
    for (const auto& [name, record] : records_) {
        if (record->published)
            result.providers.push_back({*record->descriptor, copyTools(*record), copyResources(*record)});
    }
    return result;
}

std::vector<ProviderDescriptor> ProviderRegistry::startupProviders() const {
    std::lock_guard lock(tools_.mutex_);
    return startupProviders_;
}

ProviderRegistry::ProviderUpdates ProviderRegistry::subscribeProviderUpdates(
    const std::vector<ProviderDescriptor>& providers, UpdateSink sink) {
    auto subscription = std::make_shared<UpdateSubscription>();
    subscription->sink = std::move(sink);
    ProviderUpdates result{{}, subscription};
    result.initial.reserve(providers.size());
    std::lock_guard lock(tools_.mutex_);
    for (const auto& provider : providers) {
        ProviderSnapshot initial{provider};
        const auto found = records_.find(provider.name);
        const bool active = found != records_.end() && found->second->published &&
            found->second->descriptor->instance == provider.instance;
        if (active) {
            const auto& record = found->second;
            initial.tools = copyTools(*record);
            initial.resources = copyResources(*record);
            subscription->providers.try_emplace(provider.instance, UpdateSubscription::Provider{record});
        }
        result.initial.push_back(std::move(initial));
    }
    std::erase_if(updateSubscriptions_, [](const auto& entry) { return entry.expired(); });
    updateSubscriptions_.push_back(subscription);
    return result;
}

void ProviderRegistry::markProviderUpdatedLocked(const ProviderRecord& provider, bool tools, bool resources) {
    for (const auto& entry : updateSubscriptions_) {
        if (const auto subscription = entry.lock()) {
            const auto tracked = subscription->providers.find(provider.descriptor->instance);
            if (tracked == subscription->providers.end() || tracked->second.record.lock().get() != &provider)
                continue;
            tracked->second.toolsChanged |= tools;
            tracked->second.resourcesChanged |= resources;
            providerUpdatesPending_ = true;
        }
    }
}

void ProviderRegistry::dispatchProviderUpdates() {
    if (!providerUpdatesPending_) return;
    std::vector<std::pair<std::weak_ptr<UpdateSubscription>, std::vector<ProviderUpdate>>> deliveries;
    {
        std::lock_guard lock(tools_.mutex_);
        std::erase_if(updateSubscriptions_, [](const auto& entry) { return entry.expired(); });
        for (const auto& entry : updateSubscriptions_) {
            const auto subscription = entry.lock();
            if (!subscription) continue;
            std::vector<ProviderUpdate> updates;
            for (const auto& [id, provider] : subscription->providers) {
                if (!provider.toolsChanged && !provider.resourcesChanged) continue;
                const auto record = provider.record.lock();
                const bool active = record && record->published;
                ProviderUpdate update{id};
                if (provider.toolsChanged)
                    update.tools = active ? copyTools(*record) : std::vector<ToolDescriptor>{};
                if (provider.resourcesChanged)
                    update.resources = active ? copyResources(*record) : std::vector<ResourceDescriptor>{};
                updates.push_back(std::move(update));
            }
            if (!updates.empty()) deliveries.emplace_back(entry, std::move(updates));
        }
        for (const auto& [entry, updates] : deliveries) {
            if (const auto subscription = entry.lock()) {
                for (const auto& update : updates) {
                    auto& provider = subscription->providers.at(update.providerId);
                    provider.toolsChanged = false;
                    provider.resourcesChanged = false;
                }
            }
        }
        providerUpdatesPending_ = false;
    }
    for (auto& [entry, updates] : deliveries) {
        if (const auto subscription = entry.lock()) subscription->sink(std::move(updates));
    }
}

uint64_t ProviderRegistry::resourceVersion() const {
    std::lock_guard lock(tools_.mutex_);
    return resourceVersion_;
}
ProviderRegistry::ProviderRecord* ProviderRegistry::providerResourcesLocked(const ProviderDescriptor& provider) {
    const auto owner = records_.find(provider.name);
    if (owner == records_.end() || owner->second->descriptor->instance != provider.instance) return nullptr;
    return owner->second.get();
}

ProviderRegistry::ProviderRecord* ProviderRegistry::providerForInstanceLocked(const std::string& instance) {
    const auto found = std::find_if(records_.begin(), records_.end(), [&](const auto& entry) {
        return entry.second->descriptor->instance == instance;
    });
    return found == records_.end() ? nullptr : found->second.get();
}

static bool validResourceText(const std::string& value) {
    return !value.empty() && value.size() <= 256 &&
        value.find_first_of("\r\n\0", 0, 3) == std::string::npos;
}

static bool validResource(const ResourceDescriptor& resource) {
    return validResourceText(resource.resource_id) && validResourceText(resource.name) &&
        validResourceText(resource.kind) && resource.parent_id.empty() &&
        resource.runtime_id && resource.state.is_object();
}

bool ProviderRegistry::publishResource(const std::string& instance, const ResourceDescriptor& resource) {
    if (!validResource(resource)) throw std::invalid_argument("Invalid live resource");
    std::lock_guard lock(tools_.mutex_);
    auto* owner = providerForInstanceLocked(instance);
    if (!owner) return false;
    auto* resources = &owner->resources;
    const auto existing = resources->find(resource.resource_id);
    if (existing != resources->end() && existing->second == resource) return true;
    resources->insert_or_assign(resource.resource_id, resource);
    if (owner->published) {
        ++resourceVersion_;
        markProviderUpdatedLocked(*owner, false, true);
    }
    return true;
}

void ProviderRegistry::removeResource(const std::string& instance, const ResourceDescriptor& resource) {
    std::lock_guard lock(tools_.mutex_);
    auto* owner = providerForInstanceLocked(instance);
    if (!owner) return;
    auto* resources = &owner->resources;
    const auto existing = resources->find(resource.resource_id);
    if (existing == resources->end() || existing->second.runtime_id != resource.runtime_id) return;
    resources->erase(existing);
    if (owner->published) {
        ++resourceVersion_;
        markProviderUpdatedLocked(*owner, false, true);
    }
}

ProviderRegistry::Resources ProviderRegistry::makeResourceBatch(
    const std::string& kind, const std::vector<ResourceDescriptor>& resources) {
    if (!validResourceText(kind) || resources.size() > 10000)
        throw std::invalid_argument("Invalid resource batch");
    Resources batch;
    for (const auto& resource : resources) {
        if (resource.kind != kind || !validResource(resource) ||
            !batch.emplace(resource.resource_id, resource).second)
            throw std::invalid_argument("Invalid resource batch member");
    }
    return batch;
}

bool ProviderRegistry::replaceResources(const ProviderDescriptor& provider, const std::string& kind,
                                    const std::vector<ResourceDescriptor>& resources) {
    auto batch = makeResourceBatch(kind, resources);

    std::lock_guard lock(tools_.mutex_);
    auto* owner = providerResourcesLocked(provider);
    if (!owner) return false;
    auto* current = &owner->resources;
    auto next = *current;
    std::erase_if(next, [&](const auto& entry) { return entry.second.kind == kind; });
    next.merge(batch);
    if (!batch.empty()) throw std::invalid_argument("Resource ID belongs to another kind");
    if (next.size() > 10000) throw std::invalid_argument("Provider resource limit exceeded");
    if (next != *current) {
        current->swap(next);
        if (owner->published) {
            ++resourceVersion_;
            markProviderUpdatedLocked(*owner, false, true);
        }
    }
    return true;
}

#ifdef GRIM_PROVIDER_REGISTRY_TESTS
void ProviderRegistry::retire(const std::string& provider, const std::string& instance) {
    Record record;
    {
        std::lock_guard lock(tools_.mutex_);
        const auto owner = records_.find(provider);
        if (owner == records_.end() || owner->second->descriptor->instance != instance ||
            owner->second->native) return;
        record = owner->second;
    }
    unpublishPrepared({record});
    std::lock_guard lock(tools_.mutex_);
    const auto owner = records_.find(provider);
    if (owner != records_.end() && owner->second == record) records_.erase(owner);
}

#endif

} // namespace GRIM::MMO
