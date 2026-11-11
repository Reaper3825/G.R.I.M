#include "ToolRegistry.hpp"
#include "../../commands/command_result.hpp"
#include <algorithm>

namespace GRIM::MMO {

ToolRegistry& ToolRegistry::instance() {
    static ToolRegistry registry;
    return registry;
}

bool ToolRegistry::buildAliases(const ToolIndex& entries, Aliases& aliases) {
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
    if (tool_index_.count(name)) return name;
    const auto it = alias_index_.find(name);
    return it == alias_index_.end() ? std::string{} : it->second;
}

bool ToolRegistry::updateMetadata(const ToolDescriptor& descriptor, const ProviderDescriptor& owner) {
    if (!validToolUsage(descriptor.usage_type)) return false;
    if (!validToolBindings(descriptor)) return false;
    if (!validToolShortDescription(descriptor.short_description)) return false;
    std::vector<ToolRegistryListener*> listeners;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = tool_index_.find(descriptor.tool_id);
        if (it == tool_index_.end() || *it->second->provider != owner) return false;
        Entry updated = *it->second;
        const auto& previous = it->second->descriptor;
        updated.descriptor = descriptor;
        updated.descriptor.usage_count = previous.usage_count;
        updated.descriptor.success_count = previous.success_count;
        updated.descriptor.failure_count = previous.failure_count;
        updated.descriptor.success_rate = previous.success_rate;
        auto candidate = tool_index_;
        candidate[descriptor.tool_id] = &updated;
        Aliases aliases;
        if (!buildAliases(candidate, aliases)) return false;
        listeners = listeners_;
        // Keep this registration identity so an in-flight call retains its stats.
        it->second->descriptor = std::move(updated.descriptor);
        alias_index_.swap(aliases);
        ++version_;
    }
    for (auto* listener : listeners) listener->onToolUpdated(descriptor.tool_id);
    return true;
}

bool ToolRegistry::unregisterTool(const std::string& id, const std::string& provider, const std::string& instance) {
    std::vector<ToolRegistryListener*> listeners;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = tool_index_.find(id);
        if (it == tool_index_.end()) return false;
        const auto& descriptor = *it->second->provider;
        if (descriptor.name != provider || descriptor.instance != instance) return false;
        listeners = listeners_;
        const auto& owner = providers_.records_.at(provider);
        tool_index_.erase(it);
        owner->tools.erase(id);
        for (auto alias = alias_index_.begin(); alias != alias_index_.end();) {
            if (alias->second == id) alias = alias_index_.erase(alias);
            else ++alias;
        }
        ++version_;
    }
    for (auto* listener : listeners) listener->onToolUnregistered(id);
    return true;
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
        const auto* indexed = tool_index_.at(id);
        entry = providers_.records_.at(indexed->provider->name)->tools.at(id);
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
        const auto it = tool_index_.find(id);
        if (it != tool_index_.end() && it->second == entry.get()) {
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
    const auto& entry = *tool_index_.at(id);
    return RegisteredTool{entry.descriptor, *entry.provider};
}

std::vector<ToolDescriptor> ToolRegistry::getAllTools() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<ToolDescriptor> result;
    for (const auto& [id, entry] : tool_index_) result.push_back(entry->descriptor);
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.tool_id < b.tool_id;
    });
    return result;
}

std::vector<RegisteredTool> ToolRegistry::getRegisteredTools() const {
    std::lock_guard lock(mutex_);
    std::vector<RegisteredTool> result;
    for (const auto& [id, entry] : tool_index_) result.push_back({entry->descriptor, *entry->provider});
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
    for (const auto& [id, entry] : tool_index_) result.push_back(id);
    if (include_aliases)
        for (const auto& [alias, id] : alias_index_) result.push_back(alias);
    std::sort(result.begin(), result.end());
    return result;
}
size_t ToolRegistry::toolCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tool_index_.size();
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
