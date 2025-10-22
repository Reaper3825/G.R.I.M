#pragma once
#include <string>
#include <vector>
#include <regex>
#include <memory>
#include <unordered_map>
#include "intent.hpp"
#include "memory/memory_storage.hpp"

// Forward declare to avoid heavy include
struct CommandResult;

namespace GRIM {
    class MemoryStorage;
    namespace RL {
        nlohmann::json getAction(const nlohmann::json& observation);
    }
}

class NLP {
public:
    struct Rule {
        std::string intent;        // e.g. "open_app"
        std::string description;   // human-readable ("Open a local application")
        std::string pattern_str;   // raw regex string
        std::regex pattern;        // compiled regex
        double score_boost = 0.0;  // weight to improve ranking
        bool case_insensitive = true; // regex flag
        
        std::vector<std::string> slot_names; // slot names for regex groups
        std::string category;                // optional grouping (system, app, alias)
        
        // ✅ NEW: Dynamic learning metadata
        int usage_count = 0;           // How many times this rule matched
        double success_rate = 1.0;     // Percentage of successful executions
        std::time_t last_used = 0;     // Last time this rule was used
        bool learned = false;          // True if learned from user, not from file
        std::string source = "static"; // "static", "learned", "plugin"
    };

    // --- Construction ---
    NLP();
    ~NLP();

    // --- Core Parsing ---
    Intent parse(const std::string& text);
    
    // ✅ NEW: Parse with context awareness
    Intent parseWithContext(const std::string& text, const std::string& previousCommand = "");
    
    // --- Rule Loading ---
    bool load_rules(const std::string& path, std::string* err = nullptr);
    bool load_rules_from_string(const std::string& rulesText, std::string* err = nullptr);
    
    // ✅ NEW: Plugin support
    bool registerPluginRule(const Rule& rule);
    void unregisterPluginRules(const std::string& pluginName);
    
    // ✅ NEW: Dynamic learning
    bool learnPattern(const std::string& userInput, const std::string& intent, 
                     const std::vector<std::string>& slotNames = {});
    
    // ✅ NEW: Memory integration
    void loadLearnedRules(GRIM::MemoryStorage& storage);
    void saveLearnedRules(GRIM::MemoryStorage& storage);
    
    // ✅ NEW: RL feedback integration
    void recordSuccess(const std::string& intent, const std::string& input);
    void recordFailure(const std::string& intent, const std::string& input);
    
    // ✅ NEW: Rule management
    bool updateRule(const std::string& intent, double scoreBoost);
    bool removeRule(const std::string& intent);
    std::vector<Rule> getRulesByCategory(const std::string& category) const;
    
    // ✅ NEW: Statistics & analytics
    nlohmann::json getStats() const;
    std::vector<std::pair<std::string, double>> getRuleRankings() const;
    
    // --- Debug helpers ---
    size_t rule_count() const { return rules.size(); }
    std::vector<Rule> getAllRules() const { return rules; }

private:
    // ✅ NEW: Fuzzy matching for better intent detection
    std::vector<std::pair<Rule*, double>> fuzzyMatch(const std::string& text);
    
    // ✅ NEW: Context-aware scoring
    double computeContextScore(const Rule& rule, const std::string& previousCommand);
    
    // ✅ NEW: Pattern generation from examples
    std::string generatePattern(const std::string& example);
    
    // ✅ NEW: Rule conflict resolution
    Rule* resolveConflicts(const std::vector<std::pair<Rule*, double>>& candidates);
    
    std::vector<Rule> rules;
    std::unordered_map<std::string, std::vector<Rule*>> pluginRules; // plugin_name -> rules
    
    // ✅ NEW: Learning state
    struct LearningStats {
        int totalParses = 0;
        int successfulMatches = 0;
        int failedMatches = 0;
        std::time_t lastUpdate = 0;
    } stats;
};

// 🔹 Global NLP object declaration (defined in nlp.cpp)
extern NLP g_nlp;

// ---------------------- Reload Wrapper ----------------------
CommandResult reloadNlpRules();
