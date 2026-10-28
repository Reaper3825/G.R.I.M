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
                               const std::vector<ToolRegistration>& registrations, bool replace) {
    if (registrations.empty()) return true;
    std::vector<ToolRegistryListener*> listeners;
    std::vector<std::pair<std::string, bool>> changes;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto candidate = tools_;
        std::unordered_set<std::string> seen;
        for (const auto& registration : registrations) {
            const auto& desc = registration.descriptor;
            if (desc.tool_id.empty() || !registration.handler || desc.provider_name.empty())
                throw std::invalid_argument("Tool registration requires an ID, provider, and handler");
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
        alias_index_.swap(aliases);
        ++version_;
    }
    for (const auto& [id, updated] : changes)
        for (auto* listener : listeners) {
            if (updated) listener->onToolUpdated(id);
            else listener->onToolRegistered(id);
        }
    return true;
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
    std::vector<ToolRegistryListener*> listeners;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = tools_.find(descriptor.tool_id);
        if (it == tools_.end() || it->second->descriptor.provider_name != descriptor.provider_name ||
            it->second->descriptor.provider_type != descriptor.provider_type) return false;
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

bool ToolRegistry::unregisterTool(const std::string& id, const std::string& provider, ToolProviderType type) {
    std::vector<ToolRegistryListener*> listeners;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = tools_.find(id);
        if (it == tools_.end()) return false;
        if (!provider.empty() && (it->second->descriptor.provider_name != provider ||
                                 it->second->descriptor.provider_type != type)) return false;
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
        if (removed.empty()) return;
        listeners = listeners_;
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

CommandResult ToolRegistry::invoke(const std::string& name, const std::string& arguments) {
    std::shared_ptr<Entry> entry;
    std::string id;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        id = resolveLocked(name);
        if (id.empty()) return {false, "Unknown command: " + name, "ERR_UNKNOWN_CMD", "", "", Colors::Red};
        entry = tools_.at(id);
        if (entry->descriptor.swap_state != ToolSwapState::Loaded)
            return {false, "Tool unavailable: " + name, "ERR_TOOL_UNAVAILABLE", "", "", Colors::Red};
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
        return a.category != b.category ? a.category < b.category : a.tool_id < b.tool_id;
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
    for (const auto& tool : getAllTools())
        if (result.empty() || result.back() != tool.category) result.push_back(tool.category);
    return result;
}
bool ToolRegistry::isRegistered(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !resolveLocked(name).empty();
}
bool ToolRegistry::isExecutable(const std::string& name) const {
    const auto tool = getTool(name);
    return tool && tool->swap_state == ToolSwapState::Loaded;
}
std::vector<std::string> ToolRegistry::getExecutableNames(bool include_aliases) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> result;
    for (const auto& [id, entry] : tools_)
        if (entry->descriptor.swap_state == ToolSwapState::Loaded) result.push_back(id);
    if (include_aliases)
        for (const auto& [alias, id] : alias_index_)
            if (tools_.at(id)->descriptor.swap_state == ToolSwapState::Loaded) result.push_back(alias);
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

    const auto tools = getAllTools();
    // Sort by category for readability
    std::vector<const ToolDescriptor*> sorted;
    sorted.reserve(tools.size());
    for (const auto& desc : tools) {
        if (desc.swap_state == ToolSwapState::Loaded) {
            sorted.push_back(&desc);
        }
    }
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
    const auto tools = getAllTools();

    for (const auto& desc : tools) {
        if (desc.swap_state != ToolSwapState::Loaded) continue;

        ss << "\n## " << desc.tool_id << "\n";
        ss << "Description: " << desc.description << "\n";
        if (!desc.usage.empty()) ss << "Usage: " << desc.usage << "\n";
        ss << "Category: " << desc.category << "\n";
        ss << "Type: " << (desc.is_informational ? "information" : "action") << "\n";

        if (!desc.parameters.empty()) {
            ss << "Parameters:\n";
            for (const auto& p : desc.parameters) {
                ss << "  - " << p.name << " (" << p.type << ")"
                   << (p.required ? " [required]" : " [optional]")
                   << ": " << p.description << "\n";
            }
        }

        if (!desc.examples.empty()) {
            ss << "Examples:\n";
            for (const auto& ex : desc.examples) {
                ss << "  " << ex << "\n";
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
