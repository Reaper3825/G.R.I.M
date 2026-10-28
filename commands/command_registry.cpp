#include "command_registry.hpp"
#include "logger.hpp"
#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

// Legacy startup staging only. Runtime queries and analytics use MMO::ToolRegistry.
namespace GRIM::CommandRegistry {
static std::unordered_map<std::string, ToolMetadata> g_registry;
static std::mutex g_registryMutex;

void registerTool(const std::string& commandName, const ToolMetadata& metadata) {
    std::lock_guard<std::mutex> lock(g_registryMutex);
    
    ToolMetadata meta = metadata;
    meta.name = commandName; // Ensure name matches key
    
    g_registry[commandName] = meta;
    
    // Also register aliases
    for (const auto& alias : metadata.aliases) {
        ToolMetadata aliasMeta = meta;
        aliasMeta.name = alias;
        g_registry[alias] = aliasMeta;
    }
    
    LOG_DEBUG("CommandRegistry", "Registered: " + commandName + 
              " [" + metadata.category + "]" +
              (metadata.isInformational ? " (info)" : " (action)"));
}

std::vector<ToolMetadata> getAllTools() {
    std::lock_guard<std::mutex> lock(g_registryMutex);
    
    std::vector<ToolMetadata> tools;
    std::unordered_set<std::string> seenNames; // Deduplicate aliases
    
    for (const auto& [name, meta] : g_registry) {
        if (seenNames.find(meta.name) == seenNames.end()) {
            tools.push_back(meta);
            seenNames.insert(meta.name);
        }
    }
    
    // Sort by category then name
    std::sort(tools.begin(), tools.end(), [](const auto& a, const auto& b) {
        if (a.category != b.category) return a.category < b.category;
        return a.name < b.name;
    });
    
    return tools;
}

size_t getToolCount() {
    std::lock_guard<std::mutex> lock(g_registryMutex);
    
    std::unordered_set<std::string> uniqueNames;
    for (const auto& [name, meta] : g_registry) {
        uniqueNames.insert(meta.name);
    }
    
    return uniqueNames.size();
}

} // namespace GRIM::CommandRegistry
