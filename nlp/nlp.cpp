#include "nlp.hpp"
#include "intent.hpp"
#include "error_manager.hpp"
#include "resources.hpp"
#include "console_history.hpp"

NLP g_nlp;

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

// ? NEW: Context-aware parsing
Intent NLP::parseWithContext(const std::string& text, const std::string& previousCommand) {
    Intent intent;
    intent.matched = false;
    
    stats.totalParses++;
    
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

// ? NEW: Fuzzy matching with confidence scores
std::vector<std::pair<NLP::Rule*, double>> NLP::fuzzyMatch(const std::string& text) {
    std::vector<std::pair<Rule*, double>> matches;
    
    for (auto& rule : rules) {
        std::smatch match;
        if (std::regex_match(text, match, rule.pattern)) {
            double score = 0.5 + rule.score_boost;
            
            // Boost score based on success rate
            score *= rule.success_rate;
            
            // Boost recently used rules
            auto now = std::time(nullptr);
            auto age = now - rule.last_used;
            if (age < 3600) { // Used in last hour
                score += 0.1;
            }
            
            matches.push_back({&rule, score});
        }
    }
    
    return matches;
}

// ? NEW: Context-aware scoring
double NLP::computeContextScore(const Rule& rule, const std::string& previousCommand) {
    if (previousCommand.empty()) return 0.0;
    
    // Boost follow-up commands
    if (rule.category == "continuation" && !previousCommand.empty()) {
        return 0.2;
    }
    
    // Boost related commands in same category
    // (This would need access to previous rule's category - simplified for now)
    return 0.0;
}

// ? NEW: Conflict resolution
NLP::Rule* NLP::resolveConflicts(const std::vector<std::pair<Rule*, double>>& candidates) {
    if (candidates.empty()) return nullptr;
    
    // If top candidate has significantly higher score, use it
    if (candidates.size() == 1 || candidates[0].second > candidates[1].second + 0.3) {
        return candidates[0].first;
    }
    
    // Otherwise prefer more specific (longer pattern) rules
    Rule* best = candidates[0].first;
    for (const auto& [rule, score] : candidates) {
        if (score >= candidates[0].second - 0.1 && 
            rule->pattern_str.length() > best->pattern_str.length()) {
            best = rule;
        }
    }
    
    return best;
}

// ====================================================
// Load rules from file
// ====================================================
bool NLP::load_rules(const std::string& path, std::string* err) {
    try {
        std::ifstream f(path);
        if (!f) {
            if (err) *err = "Could not open file: " + path;
            return false;
        }
        nlohmann::json j;
        f >> j;
        f.close();

        // Don't clear learned rules - only replace static ones
        rules.erase(std::remove_if(rules.begin(), rules.end(),
                                   [](const Rule& r) { return r.source == "static"; }),
                   rules.end());

        for (auto& r : j) {
            Rule rule;
            rule.intent = r.value("intent", "");
            rule.description = r.value("description", "");
            rule.pattern_str = r.value("pattern", "");
            rule.slot_names = r.value("slot_names", std::vector<std::string>{});
            rule.score_boost = r.value("score_boost", 0.0);
            rule.case_insensitive = r.value("case_insensitive", true);
            rule.category = r.value("category", "general");
            rule.source = "static";

            try {
                std::regex::flag_type flags = std::regex::ECMAScript;
                if (rule.case_insensitive) {
                    flags |= std::regex::icase;
                }
                rule.pattern = std::regex(rule.pattern_str, flags);
            } catch (std::exception& e) {
                std::cerr << "[NLP] Invalid regex for intent " << rule.intent
                          << ": " << e.what() << "\n";
                continue;
            }

            rules.push_back(rule);
        }

        std::cerr << "[NLP] Loaded " << rules.size() << " rules from " << path << "\n";
        return true;
    } catch (std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

// ====================================================
// Load rules from a JSON string
// ====================================================
bool NLP::load_rules_from_string(const std::string& rulesText, std::string* err) {
    try {
        nlohmann::json j = nlohmann::json::parse(rulesText);

        rules.clear();

        for (auto& r : j) {
            Rule rule;
            rule.intent = r.value("intent", "");
            rule.description = r.value("description", "");
            rule.pattern_str = r.value("pattern", "");
            rule.slot_names = r.value("slot_names", std::vector<std::string>{});
            rule.score_boost = r.value("score_boost", 0.0);
            rule.case_insensitive = r.value("case_insensitive", true);
            rule.category = r.value("category", "general");

            try {
                std::regex::flag_type flags = std::regex::ECMAScript;
                if (rule.case_insensitive) {
                    flags |= std::regex::icase;
                }
                rule.pattern = std::regex(rule.pattern_str, flags);
            } catch (std::exception& e) {
                std::cerr << "[NLP] Invalid regex for intent " << rule.intent
                          << ": " << e.what() << "\n";
                continue;
            }

            rules.push_back(rule);
        }

        std::cerr << "[NLP] Loaded " << rules.size() << " rules from string\n";
        return true;
    } catch (std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

// ====================================================
// Reload wrapper
// ====================================================
CommandResult reloadNlpRules() {
    std::string err;
    if (!g_nlp.load_rules(getResourcePath() + "/nlp_rules.json", &err)) {
        return {
            false,                                      // success
            "[Error] Failed to reload NLP rules",       // message
            "ERR_NLP_RELOAD_FAILED",                    // errorCode
            "error",                                    // category
            "NLP reload failed",                        // voice
            Colors::Red                                 // color
        };
    }
    return {
        true,                                       // success
        "[NLP] Rules reloaded successfully",        // message
        "ERR_NONE",                                 // errorCode
        "routine",                                  // category
        "NLP rules reloaded",                       // voice
        Colors::Green                               // color
    };
}

// ? NEW: Learn pattern from user input
bool NLP::learnPattern(const std::string& userInput, const std::string& intent,
                      const std::vector<std::string>& slotNames) {
    // Generate pattern from example
    std::string pattern = generatePattern(userInput);
    
    Rule rule;
    rule.intent = intent;
    rule.description = "Learned from user: " + userInput;
    rule.pattern_str = pattern;
    rule.slot_names = slotNames;
    rule.score_boost = 0.3; // Moderate boost for learned rules
    rule.case_insensitive = true;
    rule.category = "learned";
    rule.learned = true;
    rule.source = "learned";
    
    try {
        std::regex::flag_type flags = std::regex::ECMAScript | std::regex::icase;
        rule.pattern = std::regex(pattern, flags);
    } catch (const std::exception& e) {
        std::cerr << "[NLP] Failed to compile learned pattern: " << e.what() << "\n";
        return false;
    }
    
    rules.push_back(rule);
    std::cerr << "[NLP] Learned new pattern: " << pattern << " -> " << intent << "\n";
    return true;
}

// ? NEW: Generate regex pattern from example
std::string NLP::generatePattern(const std::string& example) {
    // Escape special regex characters
    std::string escaped = example;
    static const std::string special = "\\^$.|?*+()[]{}";
    
    for (char c : special) {
        std::string from(1, c);
        std::string to = "\\" + from;
        size_t pos = 0;
        while ((pos = escaped.find(from, pos)) != std::string::npos) {
            escaped.replace(pos, 1, to);
            pos += to.length();
        }
    }
    
    // Add flexible prefix matching (optional wake words)
    std::string pattern = "^(?:hey\\s+grim[, ]*|grim[, ]*|please\\s+)?" + escaped + "$";
    return pattern;
}

// ? NEW: Load learned rules from memory
void NLP::loadLearnedRules(GRIM::MemoryStorage& storage) {
    try {
        auto learnedRules = storage.getByTag("nlp_learned");
        
        for (const auto& memObj : learnedRules) {
            Rule rule;
            
            // Parse rule from memory object
            try {
                auto j = nlohmann::json::parse(memObj.raw);
                rule.intent = j.value("intent", "");
                rule.pattern_str = j.value("pattern", "");
                rule.score_boost = j.value("score_boost", 0.3);
                rule.usage_count = j.value("usage_count", 0);
                rule.success_rate = j.value("success_rate", 1.0);
                rule.learned = true;
                rule.source = "learned";
                rule.case_insensitive = true;
                
                std::regex::flag_type flags = std::regex::ECMAScript | std::regex::icase;
                rule.pattern = std::regex(rule.pattern_str, flags);
                
                rules.push_back(rule);
            } catch (const std::exception& e) {
                std::cerr << "[NLP] Failed to load learned rule: " << e.what() << "\n";
                continue;
            }
        }
        
        std::cerr << "[NLP] Loaded " << learnedRules.size() << " learned rules from memory\n";
    } catch (const std::exception& e) {
        std::cerr << "[NLP] Error loading learned rules: " << e.what() << "\n";
    }
}

// ? NEW: Save learned rules to memory
void NLP::saveLearnedRules(GRIM::MemoryStorage& storage) {
    try {
        for (const auto& rule : rules) {
            if (!rule.learned) continue;
            
            nlohmann::json j = {
                {"intent", rule.intent},
                {"pattern", rule.pattern_str},
                {"score_boost", rule.score_boost},
                {"usage_count", rule.usage_count},
                {"success_rate", rule.success_rate}
            };
            
            GRIM::MemoryObject memObj;
            memObj.id = GRIM::MemoryObject::generateUUID();
            memObj.timestamp = std::time(nullptr);
            memObj.source = GRIM::SourceTag::GrimInternal;
            memObj.type = GRIM::TypeTag::Fact;
            memObj.intent = GRIM::IntentTag::Inform;
            memObj.context = GRIM::ContextTag::CommandLearning;
            memObj.raw = j.dump();
            memObj.normalized = rule.intent;
            memObj.confidence = static_cast<float>(rule.success_rate);
            memObj.tags = {"nlp_learned", "pattern", rule.intent};
            
            storage.storeLongTerm(memObj);
        }
    } catch (const std::exception& e) {
        std::cerr << "[NLP] Error saving learned rules: " << e.what() << "\n";
    }
}

// ? NEW: Record successful match for RL feedback
void NLP::recordSuccess(const std::string& intent, const std::string& input) {
    for (auto& rule : rules) {
        if (rule.intent == intent) {
            int total = rule.usage_count;
            double oldRate = rule.success_rate;
            rule.success_rate = ((oldRate * total) + 1.0) / (total + 1);
            
            // Send to RL system
            try {
                nlohmann::json feedback = {
                    {"type", "nlp_success"},
                    {"intent", intent},
                    {"input", input},
                    {"success_rate", rule.success_rate},
                    {"usage_count", rule.usage_count}
                };
                GRIM::RL::getAction(feedback);
            } catch (...) {}
            
            break;
        }
    }
}

// ? NEW: Record failed match for RL feedback
void NLP::recordFailure(const std::string& intent, const std::string& input) {
    for (auto& rule : rules) {
        if (rule.intent == intent) {
            int total = rule.usage_count;
            double oldRate = rule.success_rate;
            rule.success_rate = (oldRate * total) / (total + 1);
            
            // Send to RL system
            try {
                nlohmann::json feedback = {
                    {"type", "nlp_failure"},
                    {"intent", intent},
                    {"input", input},
                    {"success_rate", rule.success_rate},
                    {"usage_count", rule.usage_count}
                };
                GRIM::RL::getAction(feedback);
            } catch (...) {}
            
            break;
        }
    }
}

// ? NEW: Plugin support
bool NLP::registerPluginRule(const Rule& rule) {
    rules.push_back(rule);
    pluginRules[rule.source].push_back(&rules.back());
    std::cerr << "[NLP] Registered plugin rule: " << rule.intent << " from " << rule.source << "\n";
    return true;
}

void NLP::unregisterPluginRules(const std::string& pluginName) {
    auto it = pluginRules.find(pluginName);
    if (it != pluginRules.end()) {
        for (auto* rule : it->second) {
            rules.erase(std::remove_if(rules.begin(), rules.end(),
                                      [rule](const Rule& r) { return &r == rule; }),
                       rules.end());
        }
        pluginRules.erase(it);
        std::cerr << "[NLP] Unregistered plugin rules from " << pluginName << "\n";
    }
}

// ? NEW: Get statistics
nlohmann::json NLP::getStats() const {
    int staticCount = 0, learnedCount = 0, pluginCount = 0;
    
    for (const auto& rule : rules) {
        if (rule.source == "static") staticCount++;
        else if (rule.source == "learned") learnedCount++;
        else pluginCount++;
    }
    
    return {
        {"total_rules", rules.size()},
        {"static_rules", staticCount},
        {"learned_rules", learnedCount},
        {"plugin_rules", pluginCount},
        {"total_parses", stats.totalParses},
        {"successful_matches", stats.successfulMatches},
        {"failed_matches", stats.failedMatches},
        {"accuracy", stats.totalParses > 0 ? 
            (double)stats.successfulMatches / stats.totalParses : 0.0}
    };
}
