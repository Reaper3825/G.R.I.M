#include "command_registry.hpp"
#include "logger.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <mutex>
#include <unordered_set>

namespace GRIM {
namespace CommandRegistry {

// ====================================================
// Internal storage
// ====================================================
static std::unordered_map<std::string, ToolMetadata> g_registry;
static std::mutex g_registryMutex;

// ====================================================
// Registration
// ====================================================

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

void registerSimpleTool(const std::string& commandName, 
                       const std::string& description,
                       bool isInformational) {
    ToolMetadata meta;
    meta.name = commandName;
    meta.description = description;
    meta.usage = commandName;
    meta.category = isInformational ? "information" : "action";
    meta.isInformational = isInformational;
    meta.needsConfirmation = !isInformational; // Actions need confirmation by default
    
    registerTool(commandName, meta);
}

void unregisterTool(const std::string& commandName) {
    std::lock_guard<std::mutex> lock(g_registryMutex);
    
    auto it = g_registry.find(commandName);
    if (it != g_registry.end()) {
        // Remove aliases too
        for (const auto& alias : it->second.aliases) {
            g_registry.erase(alias);
        }
        g_registry.erase(it);
        LOG_DEBUG("CommandRegistry", "Unregistered: " + commandName);
    }
}

// ====================================================
// Query
// ====================================================

std::optional<ToolMetadata> getTool(const std::string& commandName) {
    std::lock_guard<std::mutex> lock(g_registryMutex);
    
    auto it = g_registry.find(commandName);
    if (it != g_registry.end()) {
        return it->second;
    }
    return std::nullopt;
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

std::vector<ToolMetadata> getToolsByCategory(const std::string& category) {
    auto allTools = getAllTools();
    std::vector<ToolMetadata> filtered;
    
    for (const auto& tool : allTools) {
        if (tool.category == category) {
            filtered.push_back(tool);
        }
    }
    
    return filtered;
}

std::vector<std::string> getCategories() {
    std::lock_guard<std::mutex> lock(g_registryMutex);
    
    std::unordered_set<std::string> uniqueCategories;
    for (const auto& [name, tool] : g_registry) {
        if (!tool.category.empty()) {
            uniqueCategories.insert(tool.category);
        }
    }
    
    std::vector<std::string> categories(uniqueCategories.begin(), uniqueCategories.end());
    std::sort(categories.begin(), categories.end());
    
    return categories;
}

std::vector<ToolMetadata> getInformationalTools() {
    auto allTools = getAllTools();
    std::vector<ToolMetadata> filtered;
    
    for (const auto& tool : allTools) {
        if (tool.isInformational) {
            filtered.push_back(tool);
        }
    }
    
    return filtered;
}

std::vector<ToolMetadata> getActionTools() {
    auto allTools = getAllTools();
    std::vector<ToolMetadata> filtered;
    
    for (const auto& tool : allTools) {
        if (!tool.isInformational) {
            filtered.push_back(tool);
        }
    }
    
    return filtered;
}

std::optional<ToolMetadata> findToolByKeyword(const std::string& input) {
    std::lock_guard<std::mutex> lock(g_registryMutex);
    
    std::string lowerInput = input;
    std::transform(lowerInput.begin(), lowerInput.end(), lowerInput.begin(), ::tolower);
    
    // First pass: exact keyword match
    for (const auto& [name, meta] : g_registry) {
        for (const auto& keyword : meta.keywords) {
            std::string lowerKeyword = keyword;
            std::transform(lowerKeyword.begin(), lowerKeyword.end(), 
                         lowerKeyword.begin(), ::tolower);
            
            if (lowerInput.find(lowerKeyword) != std::string::npos) {
                return meta;
            }
        }
    }
    
    // Second pass: partial name match
    for (const auto& [name, meta] : g_registry) {
        std::string lowerName = meta.name;
        std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
        
        if (lowerInput.find(lowerName) != std::string::npos) {
            return meta;
        }
    }
    
    return std::nullopt;
}

bool isRegistered(const std::string& commandName) {
    std::lock_guard<std::mutex> lock(g_registryMutex);
    return g_registry.find(commandName) != g_registry.end();
}

// ====================================================
// AI Context Generation
// ====================================================

std::string generateAIPrompt() {
    auto tools = getAllTools();
    
    if (tools.empty()) {
        return "[No commands registered]";
    }
    
    std::ostringstream oss;
    oss << "Available commands:\n";
    
    // Group by category
    std::unordered_map<std::string, std::vector<ToolMetadata>> byCategory;
    for (const auto& tool : tools) {
        byCategory[tool.category].push_back(tool);
    }
    
    for (const auto& [category, categoryTools] : byCategory) {
        oss << "\n[" << category << "]\n";
        for (const auto& tool : categoryTools) {
            oss << "  - " << tool.name << ": " << tool.description;
            
            if (!tool.usage.empty() && tool.usage != tool.name) {
                oss << " (usage: " << tool.usage << ")";
            }
            
            if (!tool.examples.empty()) {
                oss << " (e.g., \"" << tool.examples[0] << "\")";
            }
            
            oss << "\n";
        }
    }
    
    return oss.str();
}

std::string generateCompactPrompt() {
    auto tools = getAllTools();
    
    if (tools.empty()) {
        return "[No commands]";
    }
    
    std::ostringstream oss;
    
    // Categorize
    std::vector<std::string> actions, info, system;
    
    for (const auto& tool : tools) {
        std::string entry = tool.name;
        if (tool.category == "action") {
            actions.push_back(entry);
        } else if (tool.category == "information") {
            info.push_back(entry);
        } else if (tool.category == "system") {
            system.push_back(entry);
        }
    }
    
    if (!actions.empty()) {
        oss << "Actions: ";
        for (size_t i = 0; i < actions.size(); ++i) {
            oss << actions[i];
            if (i < actions.size() - 1) oss << ", ";
        }
        oss << "\n";
    }
    
    if (!info.empty()) {
        oss << "Info: ";
        for (size_t i = 0; i < info.size(); ++i) {
            oss << info[i];
            if (i < info.size() - 1) oss << ", ";
        }
        oss << "\n";
    }
    
    if (!system.empty()) {
        oss << "System: ";
        for (size_t i = 0; i < system.size(); ++i) {
            oss << system[i];
            if (i < system.size() - 1) oss << ", ";
        }
        oss << "\n";
    }
    
    return oss.str();
}

std::string generateCategoryPrompt(const std::string& category) {
    auto tools = getToolsByCategory(category);
    
    std::ostringstream oss;
    oss << "[" << category << " commands]\n";
    
    for (const auto& tool : tools) {
        oss << "  " << tool.name << ": " << tool.description;
        if (!tool.examples.empty()) {
            oss << " (e.g., \"" << tool.examples[0] << "\")";
        }
        oss << "\n";
    }
    
    return oss.str();
}

// ====================================================
// Analytics
// ====================================================

void recordSuccess(const std::string& commandName) {
    std::lock_guard<std::mutex> lock(g_registryMutex);
    
    auto it = g_registry.find(commandName);
    if (it != g_registry.end()) {
        it->second.usageCount++;
        
        // Update success rate (rolling average)
        int totalAttempts = it->second.usageCount;
        int successCount = static_cast<int>(it->second.successRate * (totalAttempts - 1)) + 1;
        it->second.successRate = static_cast<float>(successCount) / totalAttempts;
    }
}

void recordFailure(const std::string& commandName) {
    std::lock_guard<std::mutex> lock(g_registryMutex);
    
    auto it = g_registry.find(commandName);
    if (it != g_registry.end()) {
        it->second.usageCount++;
        
        // Update success rate (rolling average)
        int totalAttempts = it->second.usageCount;
        int successCount = static_cast<int>(it->second.successRate * (totalAttempts - 1));
        it->second.successRate = static_cast<float>(successCount) / totalAttempts;
    }
}

std::vector<ToolStats> getUsageStats() {
    auto tools = getAllTools();
    std::vector<ToolStats> stats;
    
    for (const auto& tool : tools) {
        ToolStats s;
        s.name = tool.name;
        s.totalUses = tool.usageCount;
        s.successCount = static_cast<int>(tool.successRate * tool.usageCount);
        s.failureCount = tool.usageCount - s.successCount;
        s.successRate = tool.successRate;
        stats.push_back(s);
    }
    
    // Sort by usage count descending
    std::sort(stats.begin(), stats.end(), [](const auto& a, const auto& b) {
        return a.totalUses > b.totalUses;
    });
    
    return stats;
}

std::vector<std::string> getMostUsedTools(int limit) {
    auto stats = getUsageStats();
    std::vector<std::string> result;
    
    for (int i = 0; i < std::min(limit, static_cast<int>(stats.size())); ++i) {
        result.push_back(stats[i].name);
    }
    
    return result;
}

// ====================================================
// Utilities
// ====================================================

void saveToFile(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(g_registryMutex);
    
    try {
        nlohmann::json j = nlohmann::json::array();
        
        std::unordered_set<std::string> seenNames;
        for (const auto& [name, meta] : g_registry) {
            if (seenNames.find(meta.name) != seenNames.end()) continue;
            seenNames.insert(meta.name);
            
            nlohmann::json toolJson;
            toolJson["name"] = meta.name;
            toolJson["description"] = meta.description;
            toolJson["usage"] = meta.usage;
            toolJson["category"] = meta.category;
            toolJson["isInformational"] = meta.isInformational;
            toolJson["needsConfirmation"] = meta.needsConfirmation;
            toolJson["aliases"] = meta.aliases;
            toolJson["examples"] = meta.examples;
            toolJson["keywords"] = meta.keywords;
            toolJson["usageCount"] = meta.usageCount;
            toolJson["successRate"] = meta.successRate;
            
            // Parameters
            nlohmann::json params = nlohmann::json::array();
            for (const auto& param : meta.parameters) {
                params.push_back({
                    {"name", param.name},
                    {"required", param.required},
                    {"type", param.type},
                    {"description", param.description}
                });
            }
            toolJson["parameters"] = params;
            
            j.push_back(toolJson);
        }
        
        std::ofstream f(filepath);
        f << j.dump(2);
        
        LOG_DEBUG("CommandRegistry", "Saved " + std::to_string(j.size()) + " tools to " + filepath);
    } catch (const std::exception& e) {
        LOG_ERROR("CommandRegistry", "Failed to save: " + std::string(e.what()));
    }
}

void loadFromFile(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(g_registryMutex);
    
    try {
        std::ifstream f(filepath);
        if (!f) {
            LOG_DEBUG("CommandRegistry", "No registry file found: " + filepath);
            return;
        }
        
        nlohmann::json j;
        f >> j;
        
        for (const auto& toolJson : j) {
            ToolMetadata meta;
            meta.name = toolJson.value("name", "");
            meta.description = toolJson.value("description", "");
            meta.usage = toolJson.value("usage", "");
            meta.category = toolJson.value("category", "general");
            meta.isInformational = toolJson.value("isInformational", true);
            meta.needsConfirmation = toolJson.value("needsConfirmation", false);
            meta.usageCount = toolJson.value("usageCount", 0);
            meta.successRate = toolJson.value("successRate", 1.0f);
            
            if (toolJson.contains("aliases")) {
                meta.aliases = toolJson["aliases"].get<std::vector<std::string>>();
            }
            if (toolJson.contains("examples")) {
                meta.examples = toolJson["examples"].get<std::vector<std::string>>();
            }
            if (toolJson.contains("keywords")) {
                meta.keywords = toolJson["keywords"].get<std::vector<std::string>>();
            }
            
            if (toolJson.contains("parameters")) {
                for (const auto& paramJson : toolJson["parameters"]) {
                    ToolMetadata::Parameter param;
                    param.name = paramJson.value("name", "");
                    param.required = paramJson.value("required", false);
                    param.type = paramJson.value("type", "string");
                    param.description = paramJson.value("description", "");
                    meta.parameters.push_back(param);
                }
            }
            
            g_registry[meta.name] = meta;
        }
        
        LOG_DEBUG("CommandRegistry", "Loaded " + std::to_string(j.size()) + " tools from " + filepath);
    } catch (const std::exception& e) {
        LOG_ERROR("CommandRegistry", "Failed to load: " + std::string(e.what()));
    }
}

void clear() {
    std::lock_guard<std::mutex> lock(g_registryMutex);
    g_registry.clear();
    LOG_DEBUG("CommandRegistry", "Registry cleared");
}

size_t getToolCount() {
    std::lock_guard<std::mutex> lock(g_registryMutex);
    
    std::unordered_set<std::string> uniqueNames;
    for (const auto& [name, meta] : g_registry) {
        uniqueNames.insert(meta.name);
    }
    
    return uniqueNames.size();
}

} // namespace CommandRegistry
} // namespace GRIM
