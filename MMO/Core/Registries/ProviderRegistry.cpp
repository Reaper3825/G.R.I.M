#include "ProviderRegistry.hpp"
#include "ToolRegistry.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace GRIM::MMO {

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

void ProviderRegistry::prepareTools(ProviderRecord& record, const std::vector<ToolRegistration>& registrations) {
    for (const auto& registration : registrations) {
        const auto& desc = registration.descriptor;
        if (desc.tool_id.empty() || !registration.handler)
            throw std::invalid_argument("Tool registration requires an ID and handler");
        if (!validToolUsage(desc.usage_type) || !validToolBindings(desc) ||
            !validToolShortDescription(desc.short_description))
            throw std::invalid_argument("Invalid tool metadata");
        auto entry = std::make_shared<Entry>(Entry{desc, registration.handler,
            record.descriptor});
        if (!record.tools.emplace(desc.tool_id, std::move(entry)).second)
            throw std::invalid_argument("Duplicate tool ID: " + desc.tool_id);
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
bool ProviderRegistry::publish(const ProviderDescriptor& provider,
                               const std::vector<ToolRegistration>& registrations) {
    if (provider.type != ToolProviderType::Plugin || provider.name.empty() ||
        provider.instance.empty()) return false;
    auto record = std::make_shared<ProviderRecord>();
    record->descriptor = std::make_shared<const ProviderDescriptor>(provider);
    prepareTools(*record, registrations);
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
    if (record->published) return true;
    retire(provider.name, provider.instance);
    return false;
}

#endif

void ProviderRegistry::publishPrepared(const std::vector<Record>& records) {
    std::vector<ToolRegistryListener*> listeners;
    std::vector<std::string> published;
    std::vector<Record> accepted;
    {
        std::lock_guard lock(tools_.mutex_);
        auto nextTools = tools_.tool_index_;
        auto nextAliases = tools_.alias_index_;
        for (const auto& record : records) {
            auto candidateTools = nextTools;
            bool collision = false;
            for (const auto& [id, tool] : record->tools) {
                if (!candidateTools.emplace(id, tool.get()).second) {
                    collision = true;
                    break;
                }
            }
            ToolRegistry::Aliases aliases;
            if (collision || !ToolRegistry::buildAliases(candidateTools, aliases)) continue;
            for (const auto& [id, tool] : record->tools) published.push_back(id);
            accepted.push_back(record);
            nextTools.swap(candidateTools);
            nextAliases.swap(aliases);
        }
        if (accepted.empty()) return;
        listeners = tools_.listeners_;
        bool resourcesChanged = false;
        for (const auto& record : accepted) {
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

ProviderRegistrySnapshot ProviderRegistry::snapshot() const {
    std::lock_guard lock(tools_.mutex_);
    ProviderRegistrySnapshot result;
    result.revision = tools_.version_;
    result.resource_revision = resourceVersion_;
    for (const auto& [name, record] : records_) {
        if (!record->published) continue;
        ProviderSnapshot group;
        group.provider = *record->descriptor;
        for (const auto& [id, resource] : record->resources)
            group.resources.push_back(resource);
        for (const auto& [id, entry] : record->tools)
            group.tools.push_back(entry->descriptor);
        std::sort(group.tools.begin(), group.tools.end(), [](const auto& a, const auto& b) {
            return a.tool_id < b.tool_id;
        });
        result.providers.push_back(std::move(group));
    }
    return result;
}

bool ProviderRegistry::hasChanged(const ProviderRegistrySnapshot& snapshot) const {
    std::lock_guard lock(tools_.mutex_);
    return snapshot.revision != tools_.version_ || snapshot.resource_revision != resourceVersion_;
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
    if (owner->published) ++resourceVersion_;
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
    if (owner->published) ++resourceVersion_;
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
        if (owner->published) ++resourceVersion_;
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
