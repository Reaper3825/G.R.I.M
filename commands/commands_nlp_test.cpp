#include "commands_nlp_test.hpp"
#include "nlp/grammar_parser.hpp"
#include "nlp/nlp.hpp"
#include "logger.hpp"
#include "console_history.hpp"
#include <sstream>

extern ConsoleHistory history;

namespace GRIM {
    extern GrammarParser g_grammarParser;
}

CommandResult cmdTestGrammar([[maybe_unused]] const std::string& arg) {
    LOG_DEBUG("TestGrammar", "Running grammar parser integration test");
    
    std::ostringstream output;
    output << "=== Grammar Parser Live Test ===\n\n";
    
    int passed = 0;
    int failed = 0;
    
    // Test 1: Verbose command
 {
     auto result = GRIM::g_grammarParser.parse("hey grim can you please open notepad");
        if (result.matched && result.intent == "open_app") {
         output << "? Verbose: 'hey grim can you please open notepad' ? open_app\n";
            passed++;
    } else {
     output << "? Verbose command failed\n";
         failed++;
        }
    }
  
    // Test 2: Multi-command
    {
      auto result = GRIM::g_grammarParser.parse("open chrome and then close firefox");
   if (result.intent == "multi_command" && result.subCommands.size() == 2) {
   output << "? Multi: 'open chrome and then close firefox' ? 2 commands\n";
   passed++;
} else {
     output << "? Multi-command failed\n";
      failed++;
  }
    }
    
 // Test 3: Fuzzy matching
    {
        auto result = GRIM::g_grammarParser.parse("opn notepad");
    if (result.matched && result.intent == "open_app") {
          output << "? Fuzzy: 'opn notepad' ? matched despite typo\n";
        passed++;
        } else {
        output << "? Fuzzy matching failed\n";
          failed++;
        }
    }
    
    // Test 4: Synonyms
    {
      auto r1 = GRIM::g_grammarParser.parse("launch chrome");
        auto r2 = GRIM::g_grammarParser.parse("start chrome");
        if (r1.matched && r2.matched && r1.intent == "open_app" && r2.intent == "open_app") {
         output << "? Synonyms: 'launch' and 'start' both ? open_app\n";
  passed++;
        } else {
   output << "? Synonym test failed\n";
      failed++;
        }
    }
    
    // Statistics
    auto stats = GRIM::g_grammarParser.getStats();
    output << "\n=== Statistics ===\n";
    output << "Components: " << stats["components_loaded"].get<int>() << "\n";
    output << "Verbs: " << stats["verbs_loaded"].get<int>() << "\n";
    output << "Templates: " << stats["templates_loaded"].get<int>() << "\n";
    output << "Total Parses: " << stats["total_parses"].get<int>() << "\n";
    output << "Accuracy: " << (int)(stats["accuracy"].get<double>() * 100) << "%\n";
    
    output << "\n=== Results ===\n";
    output << "Passed: " << passed << "/" << (passed + failed) << "\n";
    
    return {
    failed == 0,
        output.str(),
  "ERR_NONE",
   "debug",
        std::to_string(passed) + " of " + std::to_string(passed + failed) + " tests passed",
        failed == 0 ? Colors::Green : Colors::Yellow
    };
}
