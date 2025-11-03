#include "nlp.hpp"
#include "grammar_parser.hpp"  // ? CHANGED: Use GrammarParser instead of NativeGrammarParser
#include "intent.hpp"
#include "error_manager.hpp"
#include "resources.hpp"
#include "console_history.hpp"

NLP g_nlp;

// ? CHANGED: External grammar parser (defined in bootstrap_config.cpp)
namespace GRIM {
    extern GrammarParser g_grammarParser;
}

NLP::NLP() {
    stats.lastUpdate = std::time(nullptr);
}

NLP::~NLP() = default;

// ====================================================
// Parse text against loaded NLP rules
// ====================================================
Intent NLP::parse(const std::string& text) {
    return parseWithContext(text, "");
}

// ? ENHANCED: Context-aware parsing with NATIVE grammar parser integration
Intent NLP::parseWithContext(const std::string& text, const std::string& previousCommand) {
    Intent intent;
    intent.matched = false;
    
    stats.totalParses++;
    
    // ? NEW: Try NATIVE grammar parser first (pure C++, no Python!)
    try {
      auto grammarResult = GRIM::g_grammarParser.parse(text);
     
        if (grammarResult.matched && grammarResult.confidence >= 0.65) {
          // Convert NativeGrammarResult to Intent
       intent.matched = true;
     intent.name = grammarResult.intent;
    intent.confidence = grammarResult.confidence;
     intent.slots = grammarResult.slots;  // Direct assignment (both unordered_map)
      intent.category = "grammar";  // Mark as grammar-parsed
          intent.description = "Matched via native C++ grammar parser";
        
      // Handle multi-command results
   if (grammarResult.intent == "multi_command" && !grammarResult.subCommands.empty()) {
  // Store sub-commands in a custom field
    LOG_DEBUG("NLP", "Grammar matched multi-command with " + 
 std::to_string(grammarResult.subCommands.size()) + " sub-commands");
            }
      
      stats.successfulMatches++;
     
     LOG_DEBUG("NLP", "Grammar parser matched: " + intent.name + 
 " (confidence=" + std::to_string(intent.confidence) + ")");
  
         return intent;
     }
   
        LOG_DEBUG("NLP", "Native grammar parser did not match (confidence: " + 
          std::to_string(grammarResult.confidence) + "), falling back to regex");

    } catch (const std::exception& e) {
        LOG_ERROR("NLP", std::string("Native grammar parser exception: ") + e.what());
        // Fall through to regex-based parsing
    }
    
    // ? FALLBACK: Regex-based fuzzy matching (existing code continues below)
    // Get fuzzy matches with confidence scores
    auto candidates = fuzzyMatch(text);
    
    if (candidates.empty()) {
        stats.failedMatches++;
   return intent;
    }
    
  // Apply context scoring
    for (auto& [rule, score] : candidates) {
   double contextBonus = computeContextScore(*rule, previousCommand);
 score += contextBonus;
    }
    
    // Sort by score descending
std::sort(candidates.begin(), candidates.end(),
     [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Resolve conflicts and pick best
    Rule* bestRule = resolveConflicts(candidates);
    
    if (bestRule) {
        // Build intent from matched rule
        intent.matched = true;
intent.name = bestRule->intent;
        intent.description = bestRule->description;
   intent.category = bestRule->category.empty() ? "general" : bestRule->category;
        intent.confidence = candidates[0].second;
 
     // Extract slots
 std::smatch match;
      if (std::regex_match(text, match, bestRule->pattern)) {
            for (size_t i = 1; i < match.size() && i <= bestRule->slot_names.size(); i++) {
    intent.slots[bestRule->slot_names[i - 1]] = match[i].str();
      }
        }
        
   // Update rule statistics
        bestRule->usage_count++;
   bestRule->last_used = std::time(nullptr);
      stats.successfulMatches++;
    } else {
        stats.failedMatches++;
    }
    
    stats.lastUpdate = std::time(nullptr);
    return intent;
}

// ====================================================
// Load rules from JSON file
// ====================================================
bool NLP::load_rules(const std::string& path, std::string* err) {
    std::ifstream in(path);
    if (!in.is_open()) {
     if (err) *err = "Could not open file: " + path;
        return false;
    }

    try {
   nlohmann::json j;
        in >> j;
        
        // ✅ FIX: Parse JSON directly instead of round-tripping through string
        if (!j.is_array()) {
            if (err) *err = "Expected JSON array of rules, got: " + std::string(j.type_name());
            return false;
        }

        std::vector<Rule> newRules;
        for (const auto& ruleJson : j) {
            Rule rule;
            rule.intent = ruleJson.value("intent", "");
            rule.description = ruleJson.value("description", "");
            rule.pattern_str = ruleJson.value("pattern", "");
            rule.category = ruleJson.value("category", "general");
            rule.score_boost = ruleJson.value("score_boost", 0.0);
            rule.case_insensitive = ruleJson.value("case_insensitive", true);

            if (ruleJson.contains("slot_names")) {
                rule.slot_names = ruleJson["slot_names"].get<std::vector<std::string>>();
            }

            // Compile regex
            auto flags = std::regex::ECMAScript;
            if (rule.case_insensitive) {
                flags |= std::regex::icase;
            }

            try {
                rule.pattern = std::regex(rule.pattern_str, flags);
            } catch (const std::regex_error& e) {
                if (err) *err = "Regex error in rule '" + rule.intent + "': " + e.what();
                return false;
            }

            newRules.push_back(rule);
        }

        rules = std::move(newRules);
        return true;
        
    } catch (const std::exception& e) {
        if (err) *err = std::string("Parse error: ") + e.what();
     return false;
 }
}

// ====================================================
// Load rules from JSON string
// ====================================================
bool NLP::load_rules_from_string(const std::string& rulesText, std::string* err) {
    try {
        nlohmann::json j = nlohmann::json::parse(rulesText);

   if (!j.is_array()) {
         if (err) *err = "Expected JSON array of rules";
            return false;
        }

        std::vector<Rule> newRules;
        for (const auto& ruleJson : j) {
Rule rule;
    rule.intent = ruleJson.value("intent", "");
          rule.description = ruleJson.value("description", "");
rule.pattern_str = ruleJson.value("pattern", "");
            rule.category = ruleJson.value("category", "general");
            rule.score_boost = ruleJson.value("score_boost", 0.0);
     rule.case_insensitive = ruleJson.value("case_insensitive", true);

            if (ruleJson.contains("slots")) {
    rule.slot_names = ruleJson["slots"].get<std::vector<std::string>>();
      }

    // Compile regex
 auto flags = std::regex::ECMAScript;
if (rule.case_insensitive) {
    flags |= std::regex::icase;
       }

            try {
            rule.pattern = std::regex(rule.pattern_str, flags);
            } catch (const std::regex_error& e) {
       if (err) *err = "Regex error in rule '" + rule.intent + "': " + e.what();
         return false;
        }

          newRules.push_back(rule);
   }

     rules = std::move(newRules);
        return true;

    } catch (const std::exception& e) {
        if (err) *err = std::string("JSON parse error: ") + e.what();
     return false;
    }
}

// ====================================================
// Register plugin rule
// ====================================================
bool NLP::registerPluginRule(const Rule& rule) {
    rules.push_back(rule);
    if (!rule.source.empty() && rule.source != "static") {
        pluginRules[rule.source].push_back(&rules.back());
    }
    return true;
}

// ====================================================
// Unregister plugin rules
// ====================================================
void NLP::unregisterPluginRules(const std::string& pluginName) {
  auto it = pluginRules.find(pluginName);
    if (it == pluginRules.end()) return;

    // Remove all rules from this plugin
    for (Rule* rulePtr : it->second) {
        rules.erase(std::remove_if(rules.begin(), rules.end(),
       [rulePtr](const Rule& r) { return &r == rulePtr; }), rules.end());
    }

    pluginRules.erase(it);
}

// ====================================================
// Learn pattern dynamically
// ====================================================
bool NLP::learnPattern(const std::string& userInput, const std::string& intent,
         const std::vector<std::string>& slotNames) {
    Rule rule;
    rule.intent = intent;
  rule.description = "Learned from user input";
    rule.pattern_str = generatePattern(userInput);
    rule.slot_names = slotNames;
    rule.learned = true;
    rule.source = "learned";
    rule.case_insensitive = true;

    try {
        rule.pattern = std::regex(rule.pattern_str, std::regex::ECMAScript | std::regex::icase);
rules.push_back(rule);
        return true;
    } catch (const std::regex_error&) {
     return false;
    }
}

// ====================================================
// Load learned rules from memory storage
// ====================================================
void NLP::loadLearnedRules(GRIM::MemoryStorage& storage) {
    // TODO: Implement when memory storage API is available
    (void)storage;
}

// ====================================================
// Save learned rules to memory storage
// ====================================================
void NLP::saveLearnedRules(GRIM::MemoryStorage& storage) {
    // TODO: Implement when memory storage API is available
    (void)storage;
}

// ====================================================
// Record successful intent match
// ====================================================
void NLP::recordSuccess(const std::string& intent, const std::string& input) {
    for (auto& rule : rules) {
        if (rule.intent == intent) {
          std::smatch match;
     if (std::regex_search(input, match, rule.pattern)) {
                rule.usage_count++;
    rule.last_used = std::time(nullptr);
        // Update success rate (simple moving average)
        rule.success_rate = (rule.success_rate * 0.9) + (1.0 * 0.1);
   }
      }
    }
}

// ====================================================
// Record failed intent match
// ====================================================
void NLP::recordFailure(const std::string& intent, const std::string& input) {
    for (auto& rule : rules) {
        if (rule.intent == intent) {
 std::smatch match;
 if (std::regex_search(input, match, rule.pattern)) {
             // Update success rate (simple moving average)
    rule.success_rate = (rule.success_rate * 0.9) + (0.0 * 0.1);
}
        }
    }
}

// ====================================================
// Update rule score boost
// ====================================================
bool NLP::updateRule(const std::string& intent, double scoreBoost) {
    for (auto& rule : rules) {
        if (rule.intent == intent) {
 rule.score_boost = scoreBoost;
   return true;
}
    }
 return false;
}

// ====================================================
// Remove rule by intent
// ====================================================
bool NLP::removeRule(const std::string& intent) {
  auto it = std::remove_if(rules.begin(), rules.end(),
        [&intent](const Rule& r) { return r.intent == intent; });
    
    if (it != rules.end()) {
        rules.erase(it, rules.end());
        return true;
    }
    return false;
}

// ====================================================
// Get rules by category
// ====================================================
std::vector<NLP::Rule> NLP::getRulesByCategory(const std::string& category) const {
    std::vector<Rule> result;
    for (const auto& rule : rules) {
        if (rule.category == category) {
        result.push_back(rule);
        }
    }
    return result;
}

// ====================================================
// Get statistics
// ====================================================
nlohmann::json NLP::getStats() const {
 nlohmann::json j;
    
    int staticRules = 0;
    int learnedRules = 0;
    int pluginRulesCount = 0;
    
    for (const auto& rule : rules) {
        if (rule.source == "static") staticRules++;
        else if (rule.source == "learned") learnedRules++;
        else pluginRulesCount++;
    }
    
    j["total_rules"] = rules.size();
    j["static_rules"] = staticRules;
    j["learned_rules"] = learnedRules;
    j["plugin_rules"] = pluginRulesCount;
    
    j["total_parses"] = stats.totalParses;
    j["successful_matches"] = stats.successfulMatches;
    j["failed_matches"] = stats.failedMatches;
    
    double accuracy = stats.totalParses > 0 
        ? (double)stats.successfulMatches / stats.totalParses 
        : 0.0;
    j["accuracy"] = accuracy;
    
    j["last_update"] = stats.lastUpdate;
    
    return j;
}

// ====================================================
// Get rule rankings
// ====================================================
std::vector<std::pair<std::string, double>> NLP::getRuleRankings() const {
    std::vector<std::pair<std::string, double>> rankings;
    
    for (const auto& rule : rules) {
        double score = rule.success_rate * rule.usage_count + rule.score_boost;
        rankings.emplace_back(rule.intent, score);
    }
    
    std::sort(rankings.begin(), rankings.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    return rankings;
}

// ====================================================
// Private: Fuzzy match
// ====================================================
std::vector<std::pair<NLP::Rule*, double>> NLP::fuzzyMatch(const std::string& text) {
    std::vector<std::pair<Rule*, double>> matches;
    
    for (auto& rule : rules) {
    std::smatch match;
        if (std::regex_search(text, match, rule.pattern)) {
     double score = 1.0 + rule.score_boost;
       
            // Boost score for exact matches
            if (match[0].str() == text) {
          score += 0.5;
            }

            // Apply success rate
     score *= rule.success_rate;
 
            matches.emplace_back(&rule, score);
        }
    }
    
    return matches;
}

// ====================================================
// Private: Compute context score
// ====================================================
double NLP::computeContextScore(const Rule& rule, const std::string& previousCommand) {
  if (previousCommand.empty()) return 0.0;
    
 // Simple heuristic: if previous command used same category, boost slightly
    // This is a placeholder for more sophisticated context analysis
    (void)rule;
    return 0.1;
}

// ====================================================
// Private: Generate pattern from example
// ====================================================
std::string NLP::generatePattern(const std::string& example) {
    // Escape special regex characters
  std::string pattern = example;
    const std::string specialChars = R"([\^$.|?*+(){})";
    
    for (char c : specialChars) {
        std::string from(1, c);
        std::string to = "\\" + from;
      size_t pos = 0;
        while ((pos = pattern.find(c, pos)) != std::string::npos) {
            pattern.replace(pos, 1, to);
       pos += to.length();
        }
    }
    
    // Make it match loosely (allow extra words before/after)
    return ".*" + pattern + ".*";
}

// ====================================================
// Private: Resolve conflicts
// ====================================================
NLP::Rule* NLP::resolveConflicts(const std::vector<std::pair<Rule*, double>>& candidates) {
    if (candidates.empty()) return nullptr;
    
    // Simply return the highest-scoring candidate
    return candidates[0].first;
}

// ====================================================
// Reload NLP rules wrapper
// ====================================================
CommandResult reloadNlpRules() {
    std::string path = getResourcePath() + "/nlp_rules.json";
    std::string err;
    
    if (g_nlp.load_rules(path, &err)) {
    return {
  true,
         "[NLP] Rules reloaded from " + path,
     "ERR_NONE",
          "routine",
 "NLP rules reloaded",
         Colors::Green
    };
    } else {
        return {
            false,
    "[NLP] Failed to reload rules: " + err,
            "ERR_NLP_LOAD_FAILED",
        "error",
        "Failed to reload NLP rules",
            Colors::Red
        };
    }
}
