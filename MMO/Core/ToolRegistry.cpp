#include "ToolRegistry.hpp"
#include "../../commands/command_result.hpp"
#include <algorithm>
#include <sstream>
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
    if (old_instance.empty()) return registerBatch(provider.name, tools, false, &provider);
    return replaceProvider(provider.name, old_instance, tools, &provider);
}

ToolRegistrySnapshot ToolRegistry::snapshot() const {
    std::lock_guard lock(mutex_);
    ToolRegistrySnapshot result;
    result.revision = version_;
    result.resource_revision = resourceVersion_;
    for (const auto& [key, resource] : resources_) result.resources.push_back(resource);
    auto providers = providers_;
    for (const auto& [id, entry] : tools_) {
        const auto& tool = entry->descriptor;
        result.tools.push_back(tool);
        // Built-ins have no DLL lifecycle; their registrations define availability.
        if (tool.provider_type == ToolProviderType::Builtin)
            providers.try_emplace(std::make_pair(tool.provider_type, tool.provider_name),
                ProviderDescriptor{tool.provider_type, tool.provider_name, tool.provider_instance, tool.provider_path});
    }
    for (const auto& [key, provider] : providers) result.providers.push_back(provider);
    std::sort(result.tools.begin(), result.tools.end(), [](const auto& a, const auto& b) {
        return a.tool_id < b.tool_id;
    });
    return result;
}

uint64_t ToolRegistry::resourceVersion() const {
    std::lock_guard lock(mutex_);
    return resourceVersion_;
}
void ToolRegistry::pruneResourcesLocked() {
    const auto removed = std::erase_if(resources_, [&](const auto& entry) {
        const auto& r = entry.second;
        const auto p = providers_.find({r.provider_type, r.provider_name});
        return p == providers_.end() || p->second.instance != r.provider_instance;
    });
    if (removed) ++resourceVersion_;
}
bool ToolRegistry::publishResource(const ResourceDescriptor& r) {
    const auto validText = [](const std::string& value) {
        return !value.empty() && value.size() <= 256 &&
            value.find_first_of("\r\n\0", 0, 3) == std::string::npos;
    };
    if (!validText(r.resource_id) || !validText(r.name) || !validText(r.kind) ||
        !r.parent_id.empty() || !r.runtime_id || !r.state.is_object())
        throw std::invalid_argument("Invalid live resource");
    std::lock_guard lock(mutex_);
    const auto p = providers_.find({r.provider_type, r.provider_name});
    if (p == providers_.end() || p->second.instance != r.provider_instance) return false;
    const ResourceKey key{r.provider_type, r.provider_name, r.resource_id};
    const auto existing = resources_.find(key);
    if (existing != resources_.end() && existing->second == r) return true;
    resources_.insert_or_assign(key, r);
    ++resourceVersion_;
    return true;
}
void ToolRegistry::removeResource(const ResourceDescriptor& r) {
    std::lock_guard lock(mutex_);
    const auto it = resources_.find({r.provider_type, r.provider_name, r.resource_id});
    if (it == resources_.end() || it->second.provider_instance != r.provider_instance ||
        it->second.runtime_id != r.runtime_id) return;
    resources_.erase(it);
    ++resourceVersion_;
}

bool ToolRegistry::replaceResources(const ProviderDescriptor& provider, const std::string& kind,
                                    const std::vector<ResourceDescriptor>& resources) {
    const auto valid = [](const std::string& s) {
        return !s.empty() && s.size() <= 256 && s.find_first_of("\r\n\0", 0, 3) == std::string::npos;
    };
    if (!valid(kind) || resources.size() > 10000) throw std::invalid_argument("Invalid resource batch");
    std::map<ResourceKey, ResourceDescriptor> batch;
    for (const auto& r : resources) {
        if (r.provider_type != provider.type || r.provider_name != provider.name ||
            r.provider_instance != provider.instance || r.kind != kind || !valid(r.resource_id) ||
            !valid(r.name) || !r.parent_id.empty() || !r.runtime_id || !r.state.is_object() ||
            !batch.emplace(ResourceKey{r.provider_type, r.provider_name, r.resource_id}, r).second)
            throw std::invalid_argument("Invalid resource batch member");
    }
    std::lock_guard lock(mutex_);
    const auto p = providers_.find({provider.type, provider.name});
    if (p == providers_.end() || p->second.instance != provider.instance) return false;
    auto next = resources_;
    std::erase_if(next, [&](const auto& entry) {
        const auto& r = entry.second;
        return r.provider_type == provider.type && r.provider_name == provider.name && r.kind == kind;
    });
    for (const auto& [key, r] : batch) {
        if (next.count(key)) throw std::invalid_argument("Resource ID belongs to another kind");
        next.emplace(key, r);
    }
    if (next != resources_) { resources_.swap(next); ++resourceVersion_; }
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

bool ToolRegistry::registerTool(const ToolDescriptor& descriptor, ToolHandler handler, bool replace) {
    return registerBatch({}, {{descriptor, std::move(handler)}}, replace);
}

bool ToolRegistry::registerBatch(const std::string& plugin_name,
                               const std::vector<ToolRegistration>& registrations, bool replace,
                               const ProviderDescriptor* publication) {
    if (registrations.empty() && !publication) return true;
    std::vector<ToolRegistryListener*> listeners;
    std::vector<std::pair<std::string, bool>> changes;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto providers = providers_;
        if (publication) {
            if (publication->type != ToolProviderType::Plugin || publication->name != plugin_name ||
                publication->name.empty() || publication->instance.empty() ||
                providers.count({publication->type, publication->name})) return false;
            for (const auto& [id, entry] : tools_)
                if (entry->descriptor.provider_type == publication->type &&
                    entry->descriptor.provider_name == publication->name) return false;
            providers.emplace(std::make_pair(publication->type, publication->name), *publication);
        }
        auto candidate = tools_;
        std::unordered_set<std::string> seen;
        for (const auto& registration : registrations) {
            const auto& desc = registration.descriptor;
            const auto owner = providers.find({desc.provider_type, desc.provider_name});
            if (owner != providers.end() && (owner->second.instance != desc.provider_instance ||
                                            owner->second.path != desc.provider_path)) return false;
            if (desc.tool_id.empty() || !registration.handler || desc.provider_name.empty())
                throw std::invalid_argument("Tool registration requires an ID, provider, and handler");
            if (!validToolUsage(desc.usage_type)) throw std::invalid_argument("Invalid tool usage type");
            if (!validToolBindings(desc)) throw std::invalid_argument("Invalid tool bindings");
            if (!validToolShortDescription(desc.short_description)) throw std::invalid_argument("Invalid short description");
            if (!plugin_name.empty() && (desc.provider_name != plugin_name || desc.provider_type != ToolProviderType::Plugin))
                throw std::invalid_argument("Tool batch provider mismatch");
            if (!seen.insert(desc.tool_id).second) return false;
            const auto it = candidate.find(desc.tool_id);
            const bool exists = it != candidate.end();
            if (exists && (!replace || it->second->descriptor.provider_name != desc.provider_name ||
                           it->second->descriptor.provider_type != desc.provider_type)) return false;
            candidate[desc.tool_id] = std::make_shared<Entry>(Entry{desc, registration.handler});
            changes.emplace_back(desc.tool_id, exists);
        }
        Aliases aliases;
        if (!buildAliases(candidate, aliases)) return false;
        listeners = listeners_;
        tools_.swap(candidate);
        providers_.swap(providers);
        pruneResourcesLocked();
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

bool ToolRegistry::replaceProvider(const std::string& provider, const std::string& old_instance,
                                   const std::vector<ToolRegistration>& registrations,
                                   const ProviderDescriptor* publication) {
    if (provider.empty() || old_instance.empty()) return false;
    std::vector<ToolRegistryListener*> listeners;
    std::vector<std::pair<std::string, bool>> published;
    std::vector<std::string> removed;
    {
        std::lock_guard lock(mutex_);
        auto providers = providers_;
        const auto owner = providers.find({ToolProviderType::Plugin, provider});
        if (publication) {
            if (publication->type != ToolProviderType::Plugin || publication->name != provider ||
                publication->instance.empty() || publication->instance == old_instance ||
                owner == providers.end() || owner->second.instance != old_instance) return false;
            owner->second = *publication;
        } else if (owner != providers.end()) return false; // Tracked providers require a generation record.
        auto candidate = tools_;
        for (auto it = candidate.begin(); it != candidate.end();) {
            const auto& d = it->second->descriptor;
            if (d.provider_type == ToolProviderType::Plugin && d.provider_name == provider &&
                d.provider_instance == old_instance) {
                removed.push_back(it->first);
                it = candidate.erase(it);
            } else ++it;
        }
        std::unordered_set<std::string> seen;
        for (const auto& item : registrations) {
            const auto& d = item.descriptor;
            if (publication && (d.provider_instance != publication->instance ||
                                d.provider_path != publication->path)) return false;
            if (d.provider_name != provider || d.provider_type != ToolProviderType::Plugin ||
                d.provider_instance.empty() || d.provider_instance == old_instance ||
                d.tool_id.empty() || !item.handler || !seen.insert(d.tool_id).second ||
                candidate.count(d.tool_id) || !validToolUsage(d.usage_type) || !validToolBindings(d) ||
                !validToolShortDescription(d.short_description)) return false;
            published.emplace_back(d.tool_id, tools_.count(d.tool_id) != 0);
            candidate.emplace(d.tool_id, std::make_shared<Entry>(Entry{d, item.handler}));
        }
        Aliases aliases;
        if (!buildAliases(candidate, aliases)) return false;
        listeners = listeners_;
        tools_.swap(candidate);
        providers_.swap(providers);
        pruneResourcesLocked();
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
            const auto& d = item->descriptor;
            if (d.provider_type == ToolProviderType::Plugin && d.provider_name == provider &&
                d.provider_instance == instance) removed.push_back(id);
        }
        const auto owner = providers_.find({ToolProviderType::Plugin, provider});
        const bool removeProvider = owner != providers_.end() && owner->second.instance == instance;
        if (removed.empty() && !removeProvider) return;
        listeners = listeners_;
        if (removeProvider) providers_.erase(owner);
        pruneResourcesLocked();
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
    desc.provider_name = "builtin";
    desc.version = "1.0";
    desc.is_informational = informational;
    desc.needs_confirmation = !informational;
    return registerTool(desc, std::move(handler));
}

bool ToolRegistry::updateMetadata(const ToolDescriptor& descriptor) {
    if (!validToolUsage(descriptor.usage_type)) return false;
    if (!validToolBindings(descriptor)) return false;
    if (!validToolShortDescription(descriptor.short_description)) return false;
    std::vector<ToolRegistryListener*> listeners;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = tools_.find(descriptor.tool_id);
        if (it == tools_.end() || it->second->descriptor.provider_name != descriptor.provider_name ||
            it->second->descriptor.provider_type != descriptor.provider_type) return false;
        auto updated = std::make_shared<Entry>(*it->second);
        const auto& previous = it->second->descriptor;
        updated->descriptor = descriptor;
        updated->descriptor.provider_instance = previous.provider_instance;
        updated->descriptor.provider_path = previous.provider_path;
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
        if (!provider.empty() && (it->second->descriptor.provider_name != provider ||
                                 it->second->descriptor.provider_type != type)) return false;
        if (!instance.empty() && it->second->descriptor.provider_instance != instance) return false;
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
            if (entry->descriptor.provider_name == provider && entry->descriptor.provider_type == type) removed.push_back(id);
        const auto owner = providers_.find({type, provider});
        if (removed.empty() && owner == providers_.end()) return;
        listeners = listeners_;
        if (owner != providers_.end()) providers_.erase(owner);
        pruneResourcesLocked();
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
                                   ToolExecutionMode mode) {
    return invokeChecked(name, arguments, std::nullopt, mode);
}

CommandResult ToolRegistry::invokeModel(const std::string& id, const std::string& arguments,
                                       uint64_t expected_version, ToolExecutionMode mode) {
    return invokeChecked(id, arguments, expected_version, mode);
}

CommandResult ToolRegistry::invokeChecked(const std::string& name, const std::string& arguments,
                                          std::optional<uint64_t> expected_version, ToolExecutionMode mode) {
    // Route at the shared boundary; a reserved mode must never fall through
    // to a real handler or affect its production execution statistics.
    switch (mode) {
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
        if (expected_version && *expected_version != version_)
            return {false, "Tool registry changed; inspect and retry", "ERR_TOOL_STALE", "", "", Colors::Red};
        id = resolveLocked(name);
        if (id.empty()) return {false, "Unknown command: " + name, "ERR_UNKNOWN_CMD", "", "", Colors::Red};
        entry = tools_.at(id);
        if (expected_version && (id != name || entry->descriptor.usage_type != ToolUsage::Model))
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

std::optional<ToolDescriptor> ToolRegistry::getTool(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto id = resolveLocked(name);
    if (id.empty()) return std::nullopt;
    return tools_.at(id)->descriptor;
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

std::string ToolRegistry::generateCompactPrompt() const {
    std::ostringstream ss;
    ss << "Available tools:\n";

    const auto tools = getModelTools();
    // Sort by category for readability
    std::vector<const ToolDescriptor*> sorted;
    sorted.reserve(tools.size());
    for (const auto& desc : tools) sorted.push_back(&desc);
    std::sort(sorted.begin(), sorted.end(),
              [](const auto* a, const auto* b) {
                  if (a->category != b->category) return a->category < b->category;
                  return a->tool_id < b->tool_id;
              });

    std::string current_cat;
    for (const auto* desc : sorted) {
        if (desc->category != current_cat) {
            current_cat = desc->category;
            ss << "\n[" << current_cat << "]\n";
        }
        ss << "  " << desc->tool_id << " - " << desc->description;
        if (!desc->is_informational) ss << " [ACTION]";
        if (desc->needs_confirmation) ss << " [CONFIRM]";
        ss << "\n";
    }
    return ss.str();
}

std::string ToolRegistry::generateFullPrompt() const {
    std::ostringstream ss;
    ss << "Available tools (detailed):\n";
    const auto tools = getModelTools();

    for (const auto& desc : tools) {
        ss << "\n## " << desc.tool_id << "\n";
        ss << "Description: " << desc.description << "\n";
        if (!desc.usage.empty()) ss << "Usage: " << desc.usage << "\n";
        ss << "Category: " << desc.category << "\n";
        ss << "Argument format: " << desc.argument_format << "\n";
        if (!desc.argument_schema_json.empty()) ss << "Argument schema: " << desc.argument_schema_json << "\n";
        ss << "Type: " << (desc.is_informational ? "information" : "action") << "\n";

        if (!desc.parameters.empty()) {
            ss << "Parameters:\n";
            for (const auto& p : desc.parameters) {
                ss << "  - " << p.name << " (" << p.type << ")"
                   << (p.required ? " [required]" : " [optional]")
                   << ": " << p.description << "\n";
            }
        }

        if (!desc.preconditions.empty()) {
            ss << "Preconditions: ";
            for (size_t i = 0; i < desc.preconditions.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << desc.preconditions[i];
            }
            ss << "\n";
        }
    }
    return ss.str();
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
