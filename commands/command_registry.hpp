#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace GRIM::CommandRegistry {
// Built-in metadata staged for bootstrap import into MMO::ToolRegistry.
struct ToolMetadata {
    std::string name;                      // Command name (e.g., "open", "search")
    std::string description;               // What it does
    std::string usage;                     // Usage example (e.g., "open <app_name>")
    std::string category;                  // "action", "information", "system", "ui"
    
    bool isInformational = true;           // true = read-only query, false = state-changing action
    bool needsConfirmation = false;        // true = ask before executing
    
    std::vector<std::string> aliases;      // Alternative names
    std::vector<std::string> examples;     // Example usage strings
    std::vector<std::string> keywords;     // Keywords for AI context matching
    
    // Parameter information
    struct Parameter {
        std::string name;
        bool required;
        std::string type;                  // "string", "file", "app", "number", etc.
        std::string description;
    };
    std::vector<Parameter> parameters;
    
    // Stats
    int usageCount = 0;
    float successRate = 1.0f;
    bool fromPlugin = false;
    std::string pluginName;
};

void registerTool(const std::string& commandName, const ToolMetadata& metadata);
std::vector<ToolMetadata> getAllTools();
size_t getToolCount();
} // namespace GRIM::CommandRegistry
