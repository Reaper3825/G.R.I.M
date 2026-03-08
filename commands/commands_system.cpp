#include "commands/commands_system.hpp"
#include "../MMO/Core/ToolRegistry.hpp"
#include "../MMO/Core/HardwareInventory.hpp"
#include "resources.hpp"
#include "error_manager.hpp"
#include "logger.hpp"
#include "nlp/nlp.hpp"
#include "memory/unified_memory.hpp"
#include <algorithm>
#include <sstream>
#include <vector>

// Externals
extern ConsoleHistory history;
extern GRIM::UnifiedMemoryStorage g_memoryStorage;

CommandResult cmdSystemInfo([[maybe_unused]] const std::string& arg) {
    LOG_DEBUG("Command", "Dispatch: system_info");

    auto inv = GRIM::MMO::detectHardware();

    std::ostringstream output;
    output << "[System Info]\n";
    output << "OS         : " << inv.os_name << " (" << inv.arch << ")\n";
    output << "CPU Cores  : " << inv.cpu_cores << "\n";
    output << "RAM        : " << inv.ram_total_mb << " MB\n";

    if (inv.hasGPU()) {
        std::string gpuName = inv.gpus.empty() ? "Unknown" : inv.gpus[0].name;
        std::string gpuLine = gpuName + " (" + std::to_string(inv.gpu_count) + " device(s))";
        output << "GPU        : " << gpuLine << "\n";

        if (inv.hasCUDA())  output << "CUDA       : Supported\n";
        if (inv.hasMetal()) output << "Metal      : Supported\n";
        if (inv.hasROCm())  output << "ROCm       : Supported\n";
    } else {
        output << "GPU        : None detected\n";
    }

    output << "Suggested Whisper model: " << inv.suggested_whisper_model << "\n";

    return {
        true,                               // success
        output.str(),                       // message
        "ERR_NONE",                         // errorCode
        "summary",                          // category
        "System information shown",         // voice
        Colors::Cyan                        // color
    };
}

// ====================================================
// Test command for Intent Classifier
// ====================================================
CommandResult cmdTestIntent([[maybe_unused]] const std::string& arg) {
    LOG_DEBUG("TestIntent", "Test command invoked with arg: \"" + arg + "\"");
    
    // Test cases: (input phrase, expected intent type)
    std::vector<std::pair<std::string, std::string>> testCases = {
        {"open notepad", "Command"},
        {"close window", "Command"},
        {"what's the weather", "Question"},
        {"how do I install this", "Question"},
        {"hey there", "Banter"},
        {"good morning", "Banter"},
        {"yes", "Feedback"},
        {"no", "Feedback"},
        {"no I meant chrome", "Correction"},
        {"that's wrong", "Correction"},
        {"remember when I", "MemoryReference"},
        {"what did I say about", "MemoryReference"},
        {"and then", "TaskContinuation"},
        {"continue", "TaskContinuation"},
        {"I'm frustrated", "Emotion"},
        {"that's amazing", "Emotion"},
        {"restart", "MetaCommand"},
        {"reload config", "MetaCommand"},
        {"blank", "Idle"}
    };
    
    std::ostringstream output;
    output << "=== Intent Classifier Test Results ===\n\n";
    
    for (const auto& [input, expected] : testCases) {
        // For now, just log the test cases
        // When classifier is implemented, we'll call classify() here
        output << "Input: \"" << input << "\"\n";
        output << "  Expected: " << expected << "\n";
        output << "  Status: [Pending Implementation]\n\n";
        
        LOG_DEBUG("TestIntent", 
                 "Test case: \"" + input + "\" | Expected: " + expected);
    }
    
    output << "Total Test Cases: " << testCases.size() << "\n";
    output << "Status: Classifier not yet implemented\n";
    output << "\nNext Steps:\n";
    output << "1. Implement ai_intent_classifier.cpp\n";
    output << "2. Integrate into handleCommand()\n";
    output << "3. Re-run this test to measure accuracy\n";
    
    LOG_DEBUG("TestIntent", "Test suite completed. " + std::to_string(testCases.size()) + " cases defined.");
    
    return {
        true,                                    // success
        output.str(),                            // message
        "ERR_NONE",                              // errorCode
        "debug",                                 // category
        "Test results available in console",     // voice
        Colors::Cyan                             // color
    };
}

// ====================================================
// ? NEW: NLP Statistics Command
// ====================================================
CommandResult cmdNlpStats([[maybe_unused]] const std::string& arg) {
    LOG_DEBUG("NLP", "Displaying NLP statistics");
    
    auto stats = g_nlp.getStats();
    
    std::ostringstream output;
    output << "=== NLP System Statistics ===\n\n";
    output << "Total Rules      : " << stats["total_rules"].get<int>() << "\n";
    output << "  - Static       : " << stats["static_rules"].get<int>() << "\n";
    output << "  - Learned      : " << stats["learned_rules"].get<int>() << "\n";
    output << "  - Plugin       : " << stats["plugin_rules"].get<int>() << "\n\n";
    
    output << "Parse Statistics:\n";
    output << "  Total Parses   : " << stats["total_parses"].get<int>() << "\n";
    output << "  Successful     : " << stats["successful_matches"].get<int>() << "\n";
    output << "  Failed         : " << stats["failed_matches"].get<int>() << "\n";
    output << "  Accuracy       : " << (int)(stats["accuracy"].get<double>() * 100) << "%\n";
    
    return {
        true,
        output.str(),
        "ERR_NONE",
        "debug",
        "NLP statistics displayed",
        Colors::Cyan
    };
}

// ====================================================
// ? NEW: NLP Learn Command
// ====================================================
CommandResult cmdNlpLearn(const std::string& arg) {
    // Expect format: "user input" -> "intent_name"
    size_t arrowPos = arg.find("->");
    if (arrowPos == std::string::npos) {
        return {
            false,
            "[Error] Usage: nlp_learn \"user input\" -> \"intent_name\"",
            "ERR_INVALID_FORMAT",
            "error",
            "Invalid format for learning",
            Colors::Red
        };
    }
    
    std::string userInput = arg.substr(0, arrowPos);
    std::string intent = arg.substr(arrowPos + 2);
    
    // Trim whitespace
    userInput.erase(0, userInput.find_first_not_of(" \t\""));
    userInput.erase(userInput.find_last_not_of(" \t\"") + 1);
    intent.erase(0, intent.find_first_not_of(" \t\""));
    intent.erase(intent.find_last_not_of(" \t\"") + 1);
    
    LOG_DEBUG("NLP", "Learning pattern: \"" + userInput + "\" -> " + intent);
    
    if (g_nlp.learnPattern(userInput, intent)) {
        return {
            true,
            "[NLP] Learned new pattern: \"" + userInput + "\" -> " + intent,
            "ERR_NONE",
            "routine",
            "Pattern learned successfully",
            Colors::Green
        };
    } else {
        return {
            false,
            "[NLP] Failed to learn pattern",
            "ERR_NLP_LEARN_FAILED",
            "error",
            "Failed to learn pattern",
            Colors::Red
        };
    }
}

// ====================================================
// ? NEW: NLP Save Command
// ====================================================
CommandResult cmdNlpSave([[maybe_unused]] const std::string& arg) {
    LOG_DEBUG("NLP", "Saving learned rules to memory");
    
    try {
        g_nlp.saveLearnedRules(g_memoryStorage);
        
        return {
            true,
            "[NLP] Learned rules saved to memory",
            "ERR_NONE",
            "routine",
            "Rules saved successfully",
            Colors::Green
        };
    } catch (const std::exception& e) {
        LOG_ERROR("NLP", std::string("Failed to save rules: ") + e.what());
        return {
            false,
            "[NLP] Failed to save learned rules",
            "ERR_NLP_SAVE_FAILED",
            "error",
            "Failed to save rules",
            Colors::Red
        };
    }
}

// ====================================================
// ? NEW: Comprehensive NLP System Test
// ====================================================
CommandResult cmdTestNlp([[maybe_unused]] const std::string& arg) {
    LOG_DEBUG("TestNLP", "Running NLP system tests");
    
    std::ostringstream output;
    output << "=== NLP System Test Suite ===\n\n";
    
    int passed = 0;
    int failed = 0;
    
    // TEST 1: Basic Statistics
    output << "[TEST 1] Checking baseline statistics...\n";
    try {
        auto stats = g_nlp.getStats();
        int totalRules = stats["total_rules"].get<int>();
        if (totalRules > 0) {
            output << "  ? PASS: " << totalRules << " rules loaded\n";
            passed++;
        } else {
            output << "  ? FAIL: No rules loaded\n";
            failed++;
        }
    } catch (const std::exception& e) {
        output << "  ? FAIL: Exception - " << e.what() << "\n";
        failed++;
    }
    output << "\n";
    
    // TEST 2: Static Rule Matching
    output << "[TEST 2] Testing static rule matching...\n";
    std::vector<std::pair<std::string, std::string>> testCases = {
        {"open notepad", "open_app"},
        {"list files", "ls"},
        {"show system", "sysinfo"},
        {"reload nlp", "reload_nlp"}
    };
    
    for (const auto& [input, expectedIntent] : testCases) {
        try {
            Intent result = g_nlp.parse(input);
            if (result.matched && result.name == expectedIntent) {
                output << "  ? PASS: \"" << input << "\" -> " << expectedIntent << "\n";
                passed++;
                g_nlp.recordSuccess(expectedIntent, input);
            } else if (result.matched) {
                output << "  ? FAIL: \"" << input << "\" -> " << result.name 
                       << " (expected " << expectedIntent << ")\n";
                failed++;
            } else {
                output << "  ? FAIL: \"" << input << "\" -> No match\n";
                failed++;
            }
        } catch (const std::exception& e) {
            output << "  ? FAIL: \"" << input << "\" - Exception: " << e.what() << "\n";
            failed++;
        }
    }
    output << "\n";
    
    // TEST 3: Learning New Pattern
    output << "[TEST 3] Testing dynamic learning...\n";
    try {
        std::string testInput = "launch my browser";
        std::string testIntent = "open_app";
        
        bool learned = g_nlp.learnPattern(testInput, testIntent);
        if (learned) {
            output << "  ? PASS: Learned pattern \"" << testInput << "\" -> " << testIntent << "\n";
            passed++;
            
            // Verify the learned pattern works
            Intent result = g_nlp.parse(testInput);
            if (result.matched && result.name == testIntent) {
                output << "  ? PASS: Learned pattern matched successfully\n";
                passed++;
            } else {
                output << "  ? FAIL: Learned pattern did not match\n";
                failed++;
            }
        } else {
            output << "  ? FAIL: Failed to learn pattern\n";
            failed++;
        }
    } catch (const std::exception& e) {
        output << "  ? FAIL: Exception - " << e.what() << "\n";
        failed++;
    }
    output << "\n";
    
    // TEST 4: Context-Aware Parsing
    output << "[TEST 4] Testing context-aware parsing...\n";
    try {
        Intent result1 = g_nlp.parseWithContext("open chrome", "");
        Intent result2 = g_nlp.parseWithContext("open chrome", "open_app");
        
        if (result1.matched && result2.matched) {
            output << "  ? PASS: Context-aware parsing functional\n";
            passed++;
            
            // Context should influence confidence
            if (result2.confidence >= result1.confidence) {
                output << "  ? PASS: Context boosted confidence (" 
                       << result1.confidence << " -> " << result2.confidence << ")\n";
                passed++;
            } else {
                output << "  ? INFO: Context did not boost confidence\n";
            }
        } else {
            output << "  ? FAIL: Context parsing failed\n";
            failed++;
        }
    } catch (const std::exception& e) {
        output << "  ? FAIL: Exception - " << e.what() << "\n";
        failed++;
    }
    output << "\n";
    
    // TEST 5: Fuzzy Matching
    output << "[TEST 5] Testing fuzzy matching...\n";
    try {
        // Should match even with variations
        std::vector<std::string> variations = {
            "hey grim open notepad",
            "grim, open notepad",
            "please open notepad",
            "open notepad"
        };
        
        int matchCount = 0;
        for (const auto& var : variations) {
            Intent result = g_nlp.parse(var);
            if (result.matched) matchCount++;
        }
        
        if (matchCount == variations.size()) {
            output << "  ? PASS: All variations matched (" << matchCount << "/" << variations.size() << ")\n";
            passed++;
        } else {
            output << "  ? PARTIAL: " << matchCount << "/" << variations.size() << " variations matched\n";
            if (matchCount > 0) passed++;
        }
    } catch (const std::exception& e) {
        output << "  ? FAIL: Exception - " << e.what() << "\n";
        failed++;
    }
    output << "\n";
    
    // TEST 6: Statistics Accuracy
    output << "[TEST 6] Verifying statistics...\n";
    try {
        auto stats = g_nlp.getStats();
        int totalParses = stats["total_parses"].get<int>();
        
        if (totalParses > 0) {
            output << "  ? PASS: Statistics tracking functional\n";
            output << "    - Total parses: " << totalParses << "\n";
            output << "    - Successful: " << stats["successful_matches"].get<int>() << "\n";
            output << "    - Failed: " << stats["failed_matches"].get<int>() << "\n";
            output << "    - Accuracy: " << (int)(stats["accuracy"].get<double>() * 100) << "%\n";
            passed++;
        } else {
            output << "  ? FAIL: No parse statistics recorded\n";
            failed++;
        }
    } catch (const std::exception& e) {
        output << "  ? FAIL: Exception - " << e.what() << "\n";
        failed++;
    }
    output << "\n";
    
    // Final Summary
    output << "=== Test Summary ===\n";
    output << "Passed: " << passed << "\n";
    output << "Failed: " << failed << "\n";
    output << "Total:  " << (passed + failed) << "\n";
    
    int percentage = (passed + failed) > 0 ? (passed * 100) / (passed + failed) : 0;
    output << "Success Rate: " << percentage << "%\n";
    
    if (failed == 0) {
        output << "\n? All tests passed!\n";
    } else if (passed > failed) {
        output << "\n??  Most tests passed, but some issues detected.\n";
    } else {
        output << "\n? Multiple test failures detected.\n";
    }
    
    LOG_DEBUG("TestNLP", "Test completed: " + std::to_string(passed) + " passed, " + 
              std::to_string(failed) + " failed");
    
    return {
        failed == 0,                         // success
        output.str(),                        // message
        "ERR_NONE",                          // errorCode
        "debug",                             // category
        std::to_string(passed) + " of " + std::to_string(passed + failed) + " tests passed", // voice
        failed == 0 ? Colors::Green : (passed > failed ? Colors::Yellow : Colors::Red) // color
    };
}

// ====================================================
// ? NEW: Settings Command (Stub)
// ====================================================
CommandResult cmdSettings([[maybe_unused]] const std::string& arg) {
    LOG_DEBUG("Settings", "Settings command called");
    
    std::ostringstream output;
    output << "=== GRIM Settings ===\n\n";
    output << "Configuration File: ai_config.json\n";
    output << "Location: " << getResourcePath() << "/ai_config.json\n\n";
    
    output << "Available Configuration Commands:\n";
    output << "  - ai_backend <backend>    Change AI backend (ollama/localai/openai)\n";
    output << "  - reload_nlp             Reload NLP rules from file\n";
    output << "  - nlp_stats              Show NLP system statistics\n";
    output << "  - sysinfo                Show system information\n\n";
    
    output << "Settings Menu UI:\n";
    output << "  Full settings menu with widgets is planned.\n";
    output << "  For now, edit ai_config.json manually or use commands above.\n\n";
    
    output << "Quick Settings:\n";
    output << "  - Voice Engine: " << "coqui" << " (edit ai_config.json)\n";
    output << "  - AI Backend: " << "auto" << " (use: ai_backend <backend>)\n";
    output << "  - Whisper Model: " << "ggml-base.en.bin" << "\n";
    
    return {
        true,
        output.str(),
        "ERR_NONE",
        "routine",
        "Settings information displayed",
        Colors::Cyan
    };
}

// ====================================================
// ✅ NEW: Command Registry Commands
// ====================================================

CommandResult cmdListTools(const std::string& arg) {
    LOG_DEBUG("Command", "Dispatch: list_tools");
    
    std::ostringstream output;
    
    if (arg.empty()) {
        // List all tools
        auto tools = GRIM::MMO::ToolRegistry::instance().getAllTools();
        
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
        auto tools = GRIM::MMO::ToolRegistry::instance().getByCategory(arg);
        
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
    
    auto toolOpt = GRIM::MMO::ToolRegistry::instance().getTool(arg);
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
    
    auto allTools = GRIM::MMO::ToolRegistry::instance().getAllTools();
    
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
