#include "ToolRegistry.hpp"
#include "logger.hpp"

#include <algorithm>
#include <sstream>
#include <unordered_set>

namespace GRIM::MMO {

// =========================================================
// Singleton
// =========================================================

ToolRegistry& ToolRegistry::instance() {
    static ToolRegistry s;
    return s;
}

// =========================================================
// Registration
// =========================================================

void ToolRegistry::registerTool(const ToolDescriptor& descriptor) {
    if (descriptor.tool_id.empty()) {
        throw std::runtime_error("ToolRegistry::registerTool: tool_id is empty");
    }

    std::lock_guard<std::mutex> lock(mutex_);

    bool is_update = tools_.find(descriptor.tool_id) != tools_.end();
    tools_[descriptor.tool_id] = descriptor;

    // Build alias index
    for (const auto& alias : descriptor.aliases) {
        alias_index_[alias] = descriptor.tool_id;
    }
    // Also index by display_name if different from tool_id
    if (!descriptor.display_name.empty() &&
        descriptor.display_name != descriptor.tool_id) {
        alias_index_[descriptor.display_name] = descriptor.tool_id;
    }

    ++version_;

    if (is_update) {
        notifyUpdated(descriptor.tool_id);
    } else {
        notifyRegistered(descriptor.tool_id);
    }

    LOG_DEBUG("ToolRegistry",
              (is_update ? "Updated: " : "Registered: ") + descriptor.tool_id +
              " [" + descriptor.category + "] provider=" + descriptor.provider_name);
}

void ToolRegistry::registerSimple(
    const std::string& tool_id,
    const std::string& description,
    const std::string& category,
    bool is_informational)
{
    ToolDescriptor desc;
    desc.tool_id = tool_id;
    desc.display_name = tool_id;
    desc.description = description;
    desc.usage = tool_id;
    desc.category = category;
    desc.is_informational = is_informational;
    desc.needs_confirmation = !is_informational;
    desc.provider_type = ToolProviderType::Builtin;
    desc.provider_name = "builtin";
    desc.version = "1.0";
    registerTool(desc);
}

void ToolRegistry::unregisterTool(const std::string& tool_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tools_.find(tool_id);
    if (it == tools_.end()) return;

    // Remove alias entries pointing to this tool
    for (const auto& alias : it->second.aliases) {
        alias_index_.erase(alias);
    }
    if (!it->second.display_name.empty()) {
        alias_index_.erase(it->second.display_name);
    }

    tools_.erase(it);
    ++version_;

    notifyUnregistered(tool_id);
    LOG_DEBUG("ToolRegistry", "Unregistered: " + tool_id);
}

// =========================================================
// Atomic batch updates
// =========================================================

void ToolRegistry::registerBatch(
    const std::string& plugin_name,
    const std::vector<ToolDescriptor>& tools)
{
    std::lock_guard<std::mutex> lock(mutex_);

    for (const auto& desc : tools) {
        if (desc.tool_id.empty()) {
            throw std::runtime_error(
                "ToolRegistry::registerBatch: empty tool_id from plugin " + plugin_name);
        }

        tools_[desc.tool_id] = desc;
        for (const auto& alias : desc.aliases) {
            alias_index_[alias] = desc.tool_id;
        }
        if (!desc.display_name.empty() && desc.display_name != desc.tool_id) {
            alias_index_[desc.display_name] = desc.tool_id;
        }
    }

    ++version_;

    // Notify all at once
    for (const auto& desc : tools) {
        notifyRegistered(desc.tool_id);
    }

    LOG_DEBUG("ToolRegistry",
              "Batch registered " + std::to_string(tools.size()) +
              " tools from plugin: " + plugin_name);
}

void ToolRegistry::unregisterByProvider(const std::string& plugin_name) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> to_remove;
    for (const auto& [id, desc] : tools_) {
        if (desc.provider_name == plugin_name) {
            to_remove.push_back(id);
        }
    }

    for (const auto& id : to_remove) {
        auto it = tools_.find(id);
        if (it != tools_.end()) {
            for (const auto& alias : it->second.aliases) {
                alias_index_.erase(alias);
            }
            if (!it->second.display_name.empty()) {
                alias_index_.erase(it->second.display_name);
            }
            tools_.erase(it);
        }
    }

    if (!to_remove.empty()) {
        ++version_;
        for (const auto& id : to_remove) {
            notifyUnregistered(id);
        }
        LOG_DEBUG("ToolRegistry",
                  "Unregistered " + std::to_string(to_remove.size()) +
                  " tools from plugin: " + plugin_name);
    }
}

// =========================================================
// Query
// =========================================================

std::optional<ToolDescriptor> ToolRegistry::getTool(
    const std::string& tool_id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tools_.find(tool_id);
    if (it != tools_.end()) return it->second;

    // Try alias resolution
    auto ait = alias_index_.find(tool_id);
    if (ait != alias_index_.end()) {
        auto tit = tools_.find(ait->second);
        if (tit != tools_.end()) return tit->second;
    }

    return std::nullopt;
}

std::vector<ToolDescriptor> ToolRegistry::getAllTools() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<ToolDescriptor> result;
    result.reserve(tools_.size());
    for (const auto& [id, desc] : tools_) {
        result.push_back(desc);
    }
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) {
                  if (a.category != b.category) return a.category < b.category;
                  return a.tool_id < b.tool_id;
              });
    return result;
}

std::vector<ToolDescriptor> ToolRegistry::getByCategory(
    const std::string& category) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<ToolDescriptor> result;
    for (const auto& [id, desc] : tools_) {
        if (desc.category == category) result.push_back(desc);
    }
    return result;
}

std::vector<ToolDescriptor> ToolRegistry::getByCapabilityTag(
    const std::string& tag) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<ToolDescriptor> result;
    for (const auto& [id, desc] : tools_) {
        for (const auto& t : desc.capability_tags) {
            if (t == tag) {
                result.push_back(desc);
                break;
            }
        }
    }
    return result;
}

std::vector<ToolDescriptor> ToolRegistry::getActionTools() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<ToolDescriptor> result;
    for (const auto& [id, desc] : tools_) {
        if (!desc.is_informational) result.push_back(desc);
    }
    return result;
}

std::vector<ToolDescriptor> ToolRegistry::getInformationalTools() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<ToolDescriptor> result;
    for (const auto& [id, desc] : tools_) {
        if (desc.is_informational) result.push_back(desc);
    }
    return result;
}

std::vector<std::string> ToolRegistry::getCategories() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::unordered_set<std::string> seen;
    std::vector<std::string> result;
    for (const auto& [id, desc] : tools_) {
        if (seen.insert(desc.category).second) {
            result.push_back(desc.category);
        }
    }
    return result;
}

bool ToolRegistry::isRegistered(const std::string& tool_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (tools_.find(tool_id) != tools_.end()) return true;
    auto ait = alias_index_.find(tool_id);
    return (ait != alias_index_.end() &&
            tools_.find(ait->second) != tools_.end());
}

size_t ToolRegistry::toolCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tools_.size();
}

std::optional<std::string> ToolRegistry::resolveAlias(
    const std::string& alias) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = alias_index_.find(alias);
    if (it != alias_index_.end()) return it->second;
    // Also check direct tool_id
    if (tools_.find(alias) != tools_.end()) return alias;
    return std::nullopt;
}

// =========================================================
// Model-facing prompt generation
// =========================================================

std::string ToolRegistry::generateCompactPrompt() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream ss;
    ss << "Available tools:\n";

    // Sort by category for readability
    std::vector<const ToolDescriptor*> sorted;
    sorted.reserve(tools_.size());
    for (const auto& [id, desc] : tools_) {
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
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream ss;
    ss << "Available tools (detailed):\n";

    for (const auto& [id, desc] : tools_) {
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

// =========================================================
// Analytics
// =========================================================

void ToolRegistry::recordSuccess(const std::string& tool_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tools_.find(tool_id);
    if (it == tools_.end()) return;

    it->second.success_count++;
    it->second.usage_count++;
    int total = it->second.success_count + it->second.failure_count;
    it->second.success_rate = static_cast<float>(it->second.success_count) / total;
}

void ToolRegistry::recordFailure(const std::string& tool_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tools_.find(tool_id);
    if (it == tools_.end()) return;

    it->second.failure_count++;
    it->second.usage_count++;
    int total = it->second.success_count + it->second.failure_count;
    it->second.success_rate = static_cast<float>(it->second.success_count) / total;
}

// =========================================================
// Listeners
// =========================================================

void ToolRegistry::addListener(ToolRegistryListener* listener) {
    std::lock_guard<std::mutex> lock(mutex_);
    listeners_.push_back(listener);
}

void ToolRegistry::removeListener(ToolRegistryListener* listener) {
    std::lock_guard<std::mutex> lock(mutex_);
    listeners_.erase(
        std::remove(listeners_.begin(), listeners_.end(), listener),
        listeners_.end());
}

void ToolRegistry::notifyRegistered(const std::string& tool_id) {
    for (auto* l : listeners_) l->onToolRegistered(tool_id);
}

void ToolRegistry::notifyUnregistered(const std::string& tool_id) {
    for (auto* l : listeners_) l->onToolUnregistered(tool_id);
}

void ToolRegistry::notifyUpdated(const std::string& tool_id) {
    for (auto* l : listeners_) l->onToolUpdated(tool_id);
}

} // namespace GRIM::MMO
