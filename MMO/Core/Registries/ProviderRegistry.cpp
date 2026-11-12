#include "ProviderRegistry.hpp"
#include "ToolRegistry.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace GRIM::MMO {

ProviderRegistry& ProviderRegistry::instance() {
    return ToolRegistry::instance().providers_;
}

static bool validResource(const ResourceDescriptor& resource);

bool ProviderRegistry::publish(const ProviderDescriptor& provider,
                               const std::vector<ToolRegistration>& registrations,
                               const std::string& old_instance,
                               const std::vector<ResourceDescriptor>& resources) {
    if (provider.type != ToolProviderType::Plugin || provider.name.empty() ||
        provider.instance.empty() || provider.instance == old_instance) return false;

    auto record = std::make_unique<ProviderRecord>();
    record->descriptor = std::make_shared<const ProviderDescriptor>(provider);
    if (resources.size() > 10000) throw std::invalid_argument("Provider resource limit exceeded");
    for (const auto& resource : resources)
        if (!validResource(resource) || !record->resources.emplace(resource.resource_id, resource).second)
            throw std::invalid_argument("Invalid initial resources");
    std::unique_ptr<ProviderRecord> retired;
    std::vector<ToolRegistryListener*> listeners;
    std::vector<std::pair<std::string, bool>> published;
    std::vector<std::string> removed;
    {
        std::lock_guard lock(tools_.mutex_);
        const auto owner = records_.find(provider.name);
        if (old_instance.empty()) {
            if (owner != records_.end()) return false;
        } else if (owner == records_.end() || owner->second->descriptor->instance != old_instance) {
            return false;
        }

        auto candidate = tools_.tool_index_;
        if (owner != records_.end()) {
            for (const auto& [id, entry] : owner->second->tools) {
                removed.push_back(id);
                candidate.erase(id);
            }
        }
        for (const auto& registration : registrations) {
            const auto& desc = registration.descriptor;
            const char* error = nullptr;
            if (desc.tool_id.empty() || !registration.handler)
                error = "Tool registration requires an ID and handler";
            else if (!validToolUsage(desc.usage_type))
                error = "Invalid tool usage type";
            else if (!validToolBindings(desc))
                error = "Invalid tool bindings";
            else if (!validToolShortDescription(desc.short_description))
                error = "Invalid short description";
            if (error) {
                if (old_instance.empty()) throw std::invalid_argument(error);
                return false;
            }
            if (candidate.count(desc.tool_id)) return false;
            auto entry = std::make_shared<Entry>(Entry{desc, registration.handler, record->descriptor});
            candidate.emplace(desc.tool_id, entry.get());
            record->tools.emplace(desc.tool_id, std::move(entry));
            published.emplace_back(desc.tool_id, tools_.tool_index_.count(desc.tool_id) != 0);
        }
        ToolRegistry::Aliases aliases;
        if (!ToolRegistry::buildAliases(candidate, aliases)) return false;
        listeners = tools_.listeners_;
        std::erase_if(removed, [&](const auto& id) { return record->tools.count(id) != 0; });
        if (owner == records_.end()) {
            records_.emplace(provider.name, std::move(record));
        } else {
            retired = std::exchange(owner->second, std::move(record));
            if (!retired->resources.empty()) ++resourceVersion_;
        }
        if (!resources.empty()) ++resourceVersion_;
        tools_.tool_index_.swap(candidate);
        tools_.alias_index_.swap(aliases);
        ++tools_.version_;
    }
    retired.reset();
    for (const auto& id : removed) for (auto* listener : listeners) {
        try { listener->onToolUnregistered(id); } catch (...) {}
    }
    for (const auto& [id, updated] : published) for (auto* listener : listeners) {
        try {
            if (updated) listener->onToolUpdated(id);
            else listener->onToolRegistered(id);
        } catch (...) {} // Observers cannot veto a committed mutation.
    }
    return true;
}

ProviderRegistrySnapshot ProviderRegistry::snapshot() const {
    std::lock_guard lock(tools_.mutex_);
    ProviderRegistrySnapshot result;
    result.revision = tools_.version_;
    result.resource_revision = resourceVersion_;
    for (const auto& [name, record] : records_) {
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
ProviderRegistry::Resources* ProviderRegistry::providerResourcesLocked(const ProviderDescriptor& provider) {
    const auto owner = records_.find(provider.name);
    if (owner == records_.end() || owner->second->descriptor->instance != provider.instance) return nullptr;
    return &owner->second->resources;
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

bool ProviderRegistry::publishResource(const ProviderDescriptor& provider, const ResourceDescriptor& resource) {
    if (!validResource(resource)) throw std::invalid_argument("Invalid live resource");
    std::lock_guard lock(tools_.mutex_);
    auto* resources = providerResourcesLocked(provider);
    if (!resources) return false;
    const auto existing = resources->find(resource.resource_id);
    if (existing != resources->end() && existing->second == resource) return true;
    resources->insert_or_assign(resource.resource_id, resource);
    ++resourceVersion_;
    return true;
}

void ProviderRegistry::removeResource(const ProviderDescriptor& provider, const ResourceDescriptor& resource) {
    std::lock_guard lock(tools_.mutex_);
    auto* resources = providerResourcesLocked(provider);
    if (!resources) return;
    const auto existing = resources->find(resource.resource_id);
    if (existing == resources->end() || existing->second.runtime_id != resource.runtime_id) return;
    resources->erase(existing);
    ++resourceVersion_;
}

void ProviderRegistry::validateResources(const std::string& kind, const std::vector<ResourceDescriptor>& resources) {
    if (!validResourceText(kind) || resources.size() > 10000)
        throw std::invalid_argument("Invalid resource batch");
    Resources batch;
    for (const auto& resource : resources) {
        if (resource.kind != kind || !validResource(resource) ||
            !batch.emplace(resource.resource_id, resource).second)
            throw std::invalid_argument("Invalid resource batch member");
    }

}

bool ProviderRegistry::replaceResources(const ProviderDescriptor& provider, const std::string& kind,
                                    const std::vector<ResourceDescriptor>& resources) {
    validateResources(kind, resources);
    Resources batch;
    for (const auto& resource : resources) batch.emplace(resource.resource_id, resource);

    std::lock_guard lock(tools_.mutex_);
    auto* current = providerResourcesLocked(provider);
    if (!current) return false;
    auto next = *current;
    std::erase_if(next, [&](const auto& entry) { return entry.second.kind == kind; });
    next.merge(batch);
    if (!batch.empty()) throw std::invalid_argument("Resource ID belongs to another kind");
    if (next.size() > 10000) throw std::invalid_argument("Provider resource limit exceeded");
    if (next != *current) {
        current->swap(next);
        ++resourceVersion_;
    }
    return true;
}

void ProviderRegistry::retire(const std::string& provider, const std::string& instance) {
    std::unique_ptr<ProviderRecord> retired;
    std::vector<ToolRegistryListener*> listeners;
    {
        std::lock_guard lock(tools_.mutex_);
        const auto owner = records_.find(provider);
        if (owner == records_.end() || owner->second->descriptor->instance != instance) return;
        listeners = tools_.listeners_;
        retired = std::move(owner->second);
        records_.erase(owner);
        for (const auto& [id, entry] : retired->tools) tools_.tool_index_.erase(id);
        std::erase_if(tools_.alias_index_, [&](const auto& alias) {
            return retired->tools.count(alias.second) != 0;
        });
        if (!retired->resources.empty()) ++resourceVersion_;
        ++tools_.version_;
    }
    for (const auto& [id, entry] : retired->tools) for (auto* listener : listeners) {
        try { listener->onToolUnregistered(id); } catch (...) {}
    }
}

} // namespace GRIM::MMO
