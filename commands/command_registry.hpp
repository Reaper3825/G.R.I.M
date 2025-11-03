#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <functional>
#include "commands_core.hpp"

namespace GRIM {
namespace CommandRegistry {

// ====================================================
// Tool/Command Metadata
// ====================================================
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

// ====================================================
// Registration
// ====================================================

// Register a tool with metadata
void registerTool(const std::string& commandName, const ToolMetadata& metadata);

// Register a simple tool (auto-generates minimal metadata)
void registerSimpleTool(const std::string& commandName, 
                       const std::string& description,
                       bool isInformational = true);

// Unregister a tool (for plugin cleanup)
void unregisterTool(const std::string& commandName);

// ====================================================
// Query
// ====================================================

// Get metadata for a specific command
std::optional<ToolMetadata> getTool(const std::string& commandName);

// Get all registered tools
std::vector<ToolMetadata> getAllTools();

// Get tools by category
std::vector<ToolMetadata> getToolsByCategory(const std::string& category);

// Get all unique categories
std::vector<std::string> getCategories();

// Get informational vs action tools
std::vector<ToolMetadata> getInformationalTools();
std::vector<ToolMetadata> getActionTools();

// Find tool by keyword match
std::optional<ToolMetadata> findToolByKeyword(const std::string& input);

// Check if a command is registered
bool isRegistered(const std::string& commandName);

// ====================================================
// AI Context Generation
// ====================================================

// Generate prompt text listing available commands for AI
std::string generateAIPrompt();

// Generate compact prompt (names + descriptions only)
std::string generateCompactPrompt();

// Generate category-specific prompt
std::string generateCategoryPrompt(const std::string& category);

// ====================================================
// Analytics
// ====================================================

// Record successful command execution
void recordSuccess(const std::string& commandName);

// Record failed command execution
void recordFailure(const std::string& commandName);

// Get usage statistics
struct ToolStats {
    std::string name;
    int totalUses;
    int successCount;
    int failureCount;
    float successRate;
};
std::vector<ToolStats> getUsageStats();

// Get most used tools
std::vector<std::string> getMostUsedTools(int limit = 10);

// ====================================================
// Utilities
// ====================================================

// Save registry to JSON (for persistence)
void saveToFile(const std::string& filepath);

// Load registry from JSON
void loadFromFile(const std::string& filepath);

// Clear all registered tools
void clear();

// Get total command count
size_t getToolCount();

} // namespace CommandRegistry
} // namespace GRIM
