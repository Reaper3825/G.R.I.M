#include "ToolRegistry.hpp"
#include "../../commands/command_result.hpp"
#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace GRIM::MMO {

ToolRegistry& ToolRegistry::instance() {
    static ToolRegistry registry;
    return registry;
}

bool ToolRegistry::publishProvider(const ProviderDescriptor& provider,
                                  const std::vector<ToolRegistration>& tools,
                                  const std::string& old_instance) {
    if (old_instance.empty()) return registerProviderTools(provider, tools, false, true);
    return replaceProvider(provider, old_instance, tools);
}

ToolRegistrySnapshot ToolRegistry::snapshot() const {
    std::lock_guard lock(mutex_);
    ToolRegistrySnapshot result;
    result.revision = version_;
    result.resource_revision = resourceVersion_;
    using Key = std::tuple<ToolProviderType, std::string, std::string>;
    std::map<Key, ProviderSnapshot> groups;
    const auto groupFor = [&](const ProviderDescriptor& provider) -> ProviderSnapshot& {
        const auto [it, inserted] = groups.try_emplace({provider.type, provider.name, provider.instance});
        if (inserted) it->second.provider = provider;
        return it->second;
    };
    for (const auto& [key, record] : providers_) {
        auto& group = groupFor(*record->descriptor);
        group.published = true;
        for (const auto& [id, resource] : record->resources)
            group.resources.push_back(resource);
    }
    for (const auto& [id, entry] : tools_)
        groupFor(*entry->provider).tools.push_back(entry->descriptor);
    for (auto& [key, group] : groups) {
        std::sort(group.tools.begin(), group.tools.end(), [](const auto& a, const auto& b) {
            return a.tool_id < b.tool_id;
        });
        result.providers.push_back(std::move(group));
    }
    return result;
}

uint64_t ToolRegistry::resourceVersion() const {
    std::lock_guard lock(mutex_);
    return resourceVersion_;
}
ToolRegistry::Resources* ToolRegistry::providerResourcesLocked(const ProviderDescriptor& provider) {
    const auto owner = providers_.find({provider.type, provider.name});
    if (owner == providers_.end() || owner->second->descriptor->instance != provider.instance) return nullptr;
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

bool ToolRegistry::publishResource(const ProviderDescriptor& provider, const ResourceDescriptor& resource) {
    if (!validResource(resource)) throw std::invalid_argument("Invalid live resource");
    std::lock_guard lock(mutex_);
    auto* resources = providerResourcesLocked(provider);
    if (!resources) return false;
    const auto existing = resources->find(resource.resource_id);
    if (existing != resources->end() && existing->second == resource) return true;
    resources->insert_or_assign(resource.resource_id, resource);
    ++resourceVersion_;
    return true;
}

void ToolRegistry::removeResource(const ProviderDescriptor& provider, const ResourceDescriptor& resource) {
    std::lock_guard lock(mutex_);
    auto* resources = providerResourcesLocked(provider);
    if (!resources) return;
    const auto existing = resources->find(resource.resource_id);
    if (existing == resources->end() || existing->second.runtime_id != resource.runtime_id) return;
    resources->erase(existing);
    ++resourceVersion_;
}

bool ToolRegistry::replaceResources(const ProviderDescriptor& provider, const std::string& kind,
                                    const std::vector<ResourceDescriptor>& resources) {
    if (!validResourceText(kind) || resources.size() > 10000)
        throw std::invalid_argument("Invalid resource batch");
    Resources batch;
    for (const auto& resource : resources) {
        if (resource.kind != kind || !validResource(resource) ||
            !batch.emplace(resource.resource_id, resource).second)
            throw std::invalid_argument("Invalid resource batch member");
    }

    std::lock_guard lock(mutex_);
    auto* current = providerResourcesLocked(provider);
    if (!current) return false;
    auto next = *current;
    std::erase_if(next, [&](const auto& entry) { return entry.second.kind == kind; });
    next.merge(batch);
    if (!batch.empty()) throw std::invalid_argument("Resource ID belongs to another kind");
    if (next != *current) {
        current->swap(next);
        ++resourceVersion_;
    }
    return true;
}

bool ToolRegistry::buildAliases(const Entries& entries, Aliases& aliases) {
    for (const auto& [id, entry] : entries) {
        auto add = [&](const std::string& alias) {
            if (alias.empty()) return false;
            if (alias == id) return true;
            if (entries.count(alias)) return false;
            const auto [it, inserted] = aliases.emplace(alias, id);
            return inserted || it->second == id;
        };
        for (const auto& alias : entry->descriptor.aliases) if (!add(alias)) return false;
        const auto& display = entry->descriptor.display_name;
        if (!display.empty() && !add(display)) return false;
    }
    return true;
}

std::string ToolRegistry::resolveLocked(const std::string& name) const {
    if (tools_.count(name)) return name;
    const auto it = alias_index_.find(name);
    return it == alias_index_.end() ? std::string{} : it->second;
}

bool ToolRegistry::registerTool(const ToolDescriptor& descriptor, ToolHandler handler, bool replace,
                                const ProviderDescriptor& provider) {
    return registerProviderTools(provider, {{descriptor, std::move(handler)}}, replace, false);
}

bool ToolRegistry::registerProviderTools(const ProviderDescriptor& provider,
                                 const std::vector<ToolRegistration>& registrations,
                                 bool replace, bool publish) {
    if (registrations.empty() && !publish) return true;
    if (provider.name.empty()) {
        if (publish) return false;
        throw std::invalid_argument("Tool registration requires a provider");
    }
    std::vector<ToolRegistryListener*> listeners;
    std::vector<std::pair<std::string, bool>> changes;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto providers = providers_;
        const auto key = std::make_pair(provider.type, provider.name);
        const auto published = providers.find(key);
        if (publish) {
            if (provider.type != ToolProviderType::Plugin || provider.instance.empty() ||
                published != providers.end()) return false;
            for (const auto& [id, entry] : tools_)
                if (entry->provider->type == provider.type && entry->provider->name == provider.name) return false;
        } else if (published != providers.end() && *published->second->descriptor != provider) return false;
        const auto owner = published == providers.end()
            ? std::make_shared<const ProviderDescriptor>(provider) : published->second->descriptor;
        if (publish) providers.emplace(key, std::make_shared<ProviderRecord>(ProviderRecord{owner, {}}));
        auto candidate = tools_;
        std::unordered_set<std::string> seen;
        for (const auto& registration : registrations) {
            const auto& desc = registration.descriptor;
            if (desc.tool_id.empty() || !registration.handler)
                throw std::invalid_argument("Tool registration requires an ID and handler");
            if (!validToolUsage(desc.usage_type)) throw std::invalid_argument("Invalid tool usage type");
            if (!validToolBindings(desc)) throw std::invalid_argument("Invalid tool bindings");
            if (!validToolShortDescription(desc.short_description)) throw std::invalid_argument("Invalid short description");
            if (!seen.insert(desc.tool_id).second) return false;
            const auto it = candidate.find(desc.tool_id);
            const bool exists = it != candidate.end();
            if (exists && (!replace || *it->second->provider != provider)) return false;
            candidate[desc.tool_id] = std::make_shared<Entry>(Entry{desc, registration.handler, owner});
            changes.emplace_back(desc.tool_id, exists);
        }
        Aliases aliases;
        if (!buildAliases(candidate, aliases)) return false;
        listeners = listeners_;
        tools_.swap(candidate);
        providers_.swap(providers);
        alias_index_.swap(aliases);
        ++version_;
    }
    for (const auto& [id, updated] : changes)
        for (auto* listener : listeners) {
            try {
                if (updated) listener->onToolUpdated(id);
                else listener->onToolRegistered(id);
            } catch (...) {} // Observers cannot veto an already committed mutation.
        }
    return true;
}

bool ToolRegistry::replaceProvider(const ProviderDescriptor& provider, const std::string& old_instance,
                                   const std::vector<ToolRegistration>& registrations) {
    if (provider.type != ToolProviderType::Plugin || provider.name.empty() ||
        provider.instance.empty() || provider.instance == old_instance) return false;
    std::vector<ToolRegistryListener*> listeners;
    std::vector<std::pair<std::string, bool>> published;
    std::vector<std::string> removed;
    {
        std::lock_guard lock(mutex_);
        auto providers = providers_;
        const auto owner = providers.find({provider.type, provider.name});
        if (owner == providers.end() || owner->second->descriptor->instance != old_instance) return false;
        const auto previousProvider = owner->second;
        const auto nextProvider = std::make_shared<const ProviderDescriptor>(provider);
        owner->second = std::make_shared<ProviderRecord>(ProviderRecord{nextProvider, {}});
        auto candidate = tools_;
        for (auto it = candidate.begin(); it != candidate.end();) {
            if (it->second->provider == previousProvider->descriptor) {
                removed.push_back(it->first);
                it = candidate.erase(it);
            } else ++it;
        }
        std::unordered_set<std::string> seen;
        for (const auto& item : registrations) {
            const auto& d = item.descriptor;
            if (d.tool_id.empty() || !item.handler || !seen.insert(d.tool_id).second ||
                candidate.count(d.tool_id) || !validToolUsage(d.usage_type) || !validToolBindings(d) ||
                !validToolShortDescription(d.short_description)) return false;
            published.emplace_back(d.tool_id, tools_.count(d.tool_id) != 0);
            candidate.emplace(d.tool_id, std::make_shared<Entry>(Entry{d, item.handler, nextProvider}));
        }
        Aliases aliases;
        if (!buildAliases(candidate, aliases)) return false;
        listeners = listeners_;
        tools_.swap(candidate);
        providers_.swap(providers);
        if (!previousProvider->resources.empty()) ++resourceVersion_;
        alias_index_.swap(aliases);
        ++version_;
        std::erase_if(removed, [&](const auto& id){ return seen.count(id) != 0; });
    }
    for (const auto& id : removed) for (auto* listener : listeners) {
        try { listener->onToolUnregistered(id); } catch (...) {}
    }
    for (const auto& [id, updated] : published) for (auto* listener : listeners) {
        try {
            if (updated) listener->onToolUpdated(id);
            else listener->onToolRegistered(id);
        } catch (...) {}
    }
    return true;
}
void ToolRegistry::unregisterInstance(const std::string& provider, const std::string& instance) {
    std::vector<ToolRegistryListener*> listeners;
    std::vector<std::string> removed;
    {
        std::lock_guard lock(mutex_);
        for (const auto& [id, item] : tools_) {
            const auto& owner = *item->provider;
            if (owner.type == ToolProviderType::Plugin && owner.name == provider && owner.instance == instance) removed.push_back(id);
        }
        const auto owner = providers_.find({ToolProviderType::Plugin, provider});
        const bool removeProvider = owner != providers_.end() && owner->second->descriptor->instance == instance;
        if (removed.empty() && !removeProvider) return;
        listeners = listeners_;
        if (removeProvider) {
            if (!owner->second->resources.empty()) ++resourceVersion_;
            providers_.erase(owner);
        }
        for (const auto& id : removed) tools_.erase(id);
        for (auto it = alias_index_.begin(); it != alias_index_.end();) {
            if (!tools_.count(it->second)) it = alias_index_.erase(it);
            else ++it;
        }
        ++version_;
    }
    for (const auto& id : removed) for (auto* listener : listeners) {
        try { listener->onToolUnregistered(id); } catch (...) {}
    }
}

bool ToolRegistry::registerSimple(const std::string& id, const std::string& description,
                                 const std::string& category, ToolHandler handler, bool informational) {
    ToolDescriptor desc;
    desc.tool_id = desc.display_name = desc.usage = id;
    desc.description = description;
    desc.category = category;
    desc.version = "1.0";
    desc.is_informational = informational;
    desc.needs_confirmation = !informational;
    return registerTool(desc, std::move(handler));
}

bool ToolRegistry::updateMetadata(const ToolDescriptor& descriptor, const ProviderDescriptor* owner) {
    if (!validToolUsage(descriptor.usage_type)) return false;
    if (!validToolBindings(descriptor)) return false;
    if (!validToolShortDescription(descriptor.short_description)) return false;
    std::vector<ToolRegistryListener*> listeners;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = tools_.find(descriptor.tool_id);
        if (it == tools_.end() || (owner && *it->second->provider != *owner)) return false;
        auto updated = std::make_shared<Entry>(*it->second);
        const auto& previous = it->second->descriptor;
        updated->descriptor = descriptor;
        updated->descriptor.usage_count = previous.usage_count;
        updated->descriptor.success_count = previous.success_count;
        updated->descriptor.failure_count = previous.failure_count;
        updated->descriptor.success_rate = previous.success_rate;
        auto candidate = tools_;
        candidate[descriptor.tool_id] = updated;
        Aliases aliases;
        if (!buildAliases(candidate, aliases)) return false;
        listeners = listeners_;
        // Keep this registration identity so an in-flight call retains its stats.
        it->second->descriptor = std::move(updated->descriptor);
        alias_index_.swap(aliases);
        ++version_;
    }
    for (auto* listener : listeners) listener->onToolUpdated(descriptor.tool_id);
    return true;
}

bool ToolRegistry::unregisterTool(const std::string& id, const std::string& provider, ToolProviderType type, const std::string& instance) {
    std::vector<ToolRegistryListener*> listeners;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = tools_.find(id);
        if (it == tools_.end()) return false;
        if (!provider.empty() && (it->second->provider->name != provider ||
                                 it->second->provider->type != type)) return false;
        if (!instance.empty() && it->second->provider->instance != instance) return false;
        listeners = listeners_;
        tools_.erase(it);
        for (auto alias = alias_index_.begin(); alias != alias_index_.end();) {
            if (alias->second == id) alias = alias_index_.erase(alias);
            else ++alias;
        }
        ++version_;
    }
    for (auto* listener : listeners) listener->onToolUnregistered(id);
    return true;
}

void ToolRegistry::unregisterByProvider(const std::string& provider, ToolProviderType type) {
    std::vector<std::string> removed;
    std::vector<ToolRegistryListener*> listeners;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& [id, entry] : tools_)
            if (entry->provider->name == provider && entry->provider->type == type) removed.push_back(id);
        const auto owner = providers_.find({type, provider});
        if (removed.empty() && owner == providers_.end()) return;
        listeners = listeners_;
        if (owner != providers_.end()) {
            if (!owner->second->resources.empty()) ++resourceVersion_;
            providers_.erase(owner);
        }
        for (const auto& id : removed) tools_.erase(id);
        for (auto it = alias_index_.begin(); it != alias_index_.end();) {
            if (!tools_.count(it->second)) it = alias_index_.erase(it);
            else ++it;
        }
        ++version_;
    }
    for (const auto& id : removed)
        for (auto* listener : listeners) listener->onToolUnregistered(id);
}

CommandResult ToolRegistry::invoke(const std::string& name, const std::string& arguments,
                                   const ToolInvocationOptions& options) {
    // Route at the shared boundary; a reserved mode must never fall through
    // to a real handler or affect its production execution statistics.
    switch (options.mode) {
        case ToolExecutionMode::Production:
            break;
        case ToolExecutionMode::Sandbox:
            return {false, "Sandbox tool execution is not implemented", "ERR_TOOL_SANDBOX_NOT_IMPLEMENTED", "", "", Colors::Red};
        default:
            return {false, "Invalid tool execution mode", "ERR_TOOL_EXECUTION_MODE", "", "", Colors::Red};
    }
    std::shared_ptr<Entry> entry;
    std::string id;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (options.expected_registry_version && *options.expected_registry_version != version_)
            return {false, "Tool registry changed; inspect and retry", "ERR_TOOL_STALE", "", "", Colors::Red};
        id = resolveLocked(name);
        if (id.empty()) return {false, "Unknown command: " + name, "ERR_UNKNOWN_CMD", "", "", Colors::Red};
        entry = tools_.at(id);
        if (options.require_canonical_model && (id != name || entry->descriptor.usage_type != ToolUsage::Model))
            return {false, "Canonical Model tool required", "ERR_TOOL_NOT_MODEL", "", "", Colors::Red};
    }
    CommandResult result;
    try { result = entry->handler(arguments); }
    catch (const std::exception& error) {
        result = {false, "[Error] Exception while running command: " + name + ": " + error.what(), "ERR_CMD_EXCEPTION", "", "", Colors::Red};
    } catch (...) {
        result = {false, "[Error] Exception while running command: " + name, "ERR_CMD_EXCEPTION", "", "", Colors::Red};
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = tools_.find(id);
        if (it != tools_.end() && it->second == entry) {
            auto& stats = entry->descriptor;
            ++stats.usage_count;
            if (result.success) ++stats.success_count;
            else ++stats.failure_count;
            stats.success_rate = static_cast<float>(stats.success_count) / (stats.success_count + stats.failure_count);
        }
    }
    return result;
}

std::optional<RegisteredTool> ToolRegistry::getTool(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto id = resolveLocked(name);
    if (id.empty()) return std::nullopt;
    const auto& entry = *tools_.at(id);
    return RegisteredTool{entry.descriptor, *entry.provider};
}

std::vector<ToolDescriptor> ToolRegistry::getAllTools() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<ToolDescriptor> result;
    for (const auto& [id, entry] : tools_) result.push_back(entry->descriptor);
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.tool_id < b.tool_id;
    });
    return result;
}

std::vector<RegisteredTool> ToolRegistry::getRegisteredTools() const {
    std::lock_guard lock(mutex_);
    std::vector<RegisteredTool> result;
    for (const auto& [id, entry] : tools_) result.push_back({entry->descriptor, *entry->provider});
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.tool_id < b.tool_id;
    });
    return result;
}

std::vector<ToolDescriptor> ToolRegistry::getModelTools() const {
    auto result = getAllTools();
    std::erase_if(result, [](const auto& tool) {
        return tool.usage_type != ToolUsage::Model;
    });
    return result;
}
std::vector<ToolDescriptor> ToolRegistry::getByCategory(const std::string& category) const {
    auto result = getAllTools();
    std::erase_if(result, [&](const auto& tool) { return tool.category != category; });
    return result;
}
std::vector<ToolDescriptor> ToolRegistry::getByCapabilityTag(const std::string& tag) const {
    auto result = getAllTools();
    std::erase_if(result, [&](const auto& tool) { return std::find(tool.capability_tags.begin(), tool.capability_tags.end(), tag) == tool.capability_tags.end(); });
    return result;
}
std::vector<ToolDescriptor> ToolRegistry::getActionTools() const {
    auto result = getAllTools();
    std::erase_if(result, [](const auto& tool) { return tool.is_informational; });
    return result;
}
std::vector<ToolDescriptor> ToolRegistry::getInformationalTools() const {
    auto result = getAllTools();
    std::erase_if(result, [](const auto& tool) { return !tool.is_informational; });
    return result;
}
std::vector<std::string> ToolRegistry::getCategories() const {
    std::vector<std::string> result;
    for (const auto& tool : getAllTools()) result.push_back(tool.category);
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}
bool ToolRegistry::isRegistered(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !resolveLocked(name).empty();
}
bool ToolRegistry::isExecutable(const std::string& name) const {
    return isRegistered(name);
}
std::vector<std::string> ToolRegistry::getExecutableNames(bool include_aliases) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> result;
    for (const auto& [id, entry] : tools_) result.push_back(id);
    if (include_aliases)
        for (const auto& [alias, id] : alias_index_) result.push_back(alias);
    std::sort(result.begin(), result.end());
    return result;
}
size_t ToolRegistry::toolCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tools_.size();
}
std::optional<std::string> ToolRegistry::resolveAlias(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto id = resolveLocked(name);
    if (id.empty()) return std::nullopt;
    return id;
}

uint64_t ToolRegistry::version() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return version_;
}

void ToolRegistry::addListener(ToolRegistryListener* listener) {
    std::lock_guard<std::mutex> lock(mutex_);
    listeners_.push_back(listener);
}
void ToolRegistry::removeListener(ToolRegistryListener* listener) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::erase(listeners_, listener);
}
} // namespace GRIM::MMO
