#include "snapshot_registry.hpp"
#include "plugins/plugin_logging.hpp"
#include <sstream>
#include <unordered_map>

CommandResult cmdListTools(const std::string& arg) {
    LOG_DEBUG("Command", "Dispatch: list_tools");
    
    std::ostringstream output;
    
    if (arg.empty()) {
        // List all tools
        auto tools = inspection::snapshotRegistry().getAllTools();
        
        if (tools.empty()) {
            return {
                false,
                "[Error] No tools registered",
                "ERR_NO_TOOLS",
                "error",
                "No tools available",
                Colors::Red
            };
        }
        
        output << "[Registered Tools - " << tools.size() << " total]\n\n";
        
        // Group by category
        std::unordered_map<std::string, std::vector<GRIM::MMO::ToolDescriptor>> byCategory;
        for (const auto& tool : tools) {
            byCategory[tool.category].push_back(tool);
        }
        
        for (const auto& [category, categoryTools] : byCategory) {
            output << "[" << category << "]\n";
            for (const auto& tool : categoryTools) {
                output << "  " << tool.tool_id;
                if (!tool.aliases.empty()) {
                    output << " (aliases: ";
                    for (size_t i = 0; i < tool.aliases.size(); ++i) {
                        output << tool.aliases[i];
                        if (i < tool.aliases.size() - 1) output << ", ";
                    }
                    output << ")";
                }
                output << "\n";
                output << "    " << tool.description << "\n";
            }
            output << "\n";
        }
    } else {
        // List tools in specific category
        auto tools = inspection::snapshotRegistry().getByCategory(arg);
        
        if (tools.empty()) {
            return {
                false,
                "[Error] No tools in category: " + arg,
                "ERR_NO_CATEGORY",
                "error",
                "Category not found",
                Colors::Red
            };
        }
        
        output << "[" << arg << " tools - " << tools.size() << " total]\n\n";
        for (const auto& tool : tools) {
            output << "  " << tool.tool_id << ": " << tool.description << "\n";
        }
    }
    
    return {
        true,
        output.str(),
        "ERR_NONE",
        "information",
        "Tool list displayed",
        Colors::Cyan
    };
}

CommandResult cmdToolInfo(const std::string& arg) {
    LOG_DEBUG("Command", "Dispatch: tool_info arg=" + arg);
    
    if (arg.empty()) {
        return {
            false,
            "[Error] Usage: tool_info <command_name>",
            "ERR_NO_ARGUMENT",
            "error",
            "Please specify a tool name",
            Colors::Red
        };
    }
    
    auto toolOpt = inspection::snapshotRegistry().getTool(arg);
    if (!toolOpt.has_value()) {
        return {
            false,
            "[Error] Tool not found: " + arg,
            "ERR_NOT_FOUND",
            "error",
            "Tool not registered",
            Colors::Red
        };
    }
    
    const auto& tool = toolOpt.value();
    std::ostringstream output;
    
    output << "[Tool: " << tool.tool_id << "]\n\n";
    output << "Description: " << tool.description << "\n";
    output << "Category   : " << tool.category << "\n";
    output << "Type       : " << (tool.is_informational ? "Information" : "Action") << "\n";
    output << "Usage      : " << tool.usage << "\n";
    
    if (!tool.aliases.empty()) {
        output << "Aliases    : ";
        for (size_t i = 0; i < tool.aliases.size(); ++i) {
            output << tool.aliases[i];
            if (i < tool.aliases.size() - 1) output << ", ";
        }
        output << "\n";
    }
    
    if (!tool.keywords.empty()) {
        output << "Keywords   : ";
        for (size_t i = 0; i < tool.keywords.size(); ++i) {
            output << tool.keywords[i];
            if (i < tool.keywords.size() - 1) output << ", ";
        }
        output << "\n";
    }
    
    if (!tool.parameters.empty()) {
        output << "\nParameters:\n";
        for (const auto& param : tool.parameters) {
            output << "  " << param.name << " (" << param.type << ")";
            if (param.required) output << " [required]";
            output << "\n    " << param.description << "\n";
        }
    }
    
    if (!tool.examples.empty()) {
        output << "\nExamples:\n";
        for (const auto& example : tool.examples) {
            output << "  - " << example << "\n";
        }
    }
    
    output << "\nUsage Stats:\n";
    output << "  Total uses  : " << tool.usage_count << "\n";
    output << "  Success rate: " << static_cast<int>(tool.success_rate * 100) << "%\n";
    
    return {
        true,
        output.str(),
        "ERR_NONE",
        "information",
        "Tool information displayed",
        Colors::Cyan
    };
}

CommandResult cmdToolStats(const std::string& arg) {
    LOG_DEBUG("Command", "Dispatch: tool_stats");
    
    auto allTools = inspection::snapshotRegistry().getAllTools();
    
    // Filter to tools with usage
    std::vector<GRIM::MMO::ToolDescriptor> stats;
    for (const auto& t : allTools) {
        if (t.usage_count > 0) stats.push_back(t);
    }
    // Sort by usage count descending
    std::sort(stats.begin(), stats.end(),
              [](const auto& a, const auto& b) { return a.usage_count > b.usage_count; });
    
    if (stats.empty()) {
        return {
            false,
            "[Error] No usage statistics available",
            "ERR_NO_STATS",
            "error",
            "No statistics",
            Colors::Red
        };
    }
    
    std::ostringstream output;
    output << "[Tool Usage Statistics]\n\n";
    
    int limit = arg.empty() ? 20 : std::stoi(arg);
    int count = 0;
    
    for (const auto& stat : stats) {
        if (count++ >= limit) break;
        
        output << count << ". " << stat.tool_id << "\n";
        output << "   Uses: " << stat.usage_count 
               << " (Success: " << stat.success_count
               << ", Fail: " << stat.failure_count << ")\n";
        output << "   Success Rate: " << static_cast<int>(stat.success_rate * 100) << "%\n\n";
    }
    
    return {
        true,
        output.str(),
        "ERR_NONE",
        "information",
        "Statistics displayed",
        Colors::Cyan
    };
}

