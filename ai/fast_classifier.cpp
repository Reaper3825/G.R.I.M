#include "fast_classifier.hpp"
#include "intent_gate.hpp"
#include "logger.hpp"
#include "memory/context_manager.hpp"
#include "memory/memory_storage.hpp"
#include "nlp/nlp.hpp"  // ? NEW: Access NLP rules
#include <algorithm>
#include <sstream>
#include <cctype>
#include <regex>

namespace GRIM {

// Static member initialization
std::unordered_map<std::string, float> FastClassifier::commandWeights_;
std::unordered_map<std::string, float> FastClassifier::banterWeights_;
std::unordered_map<std::string, float> FastClassifier::learnedWeights_;

void FastClassifier::init() {
    // Command indicators (action verbs)
    commandWeights_ = {
        // Core actions
        {"open", 1.5f}, {"run", 1.5f}, {"launch", 1.5f}, {"start", 1.5f},
        {"close", 1.5f}, {"stop", 1.5f}, {"kill", 1.5f}, {"end", 1.5f},
        {"show", 1.2f}, {"display", 1.2f}, {"list", 1.2f}, {"get", 1.2f},
        {"set", 1.5f}, {"change", 1.5f}, {"update", 1.5f}, {"modify", 1.5f},
        {"create", 1.5f}, {"make", 1.5f}, {"delete", 1.5f}, {"remove", 1.5f},
        {"install", 1.5f}, {"uninstall", 1.5f}, {"download", 1.5f},
        {"restart", 1.5f}, {"reboot", 1.5f}, {"shutdown", 1.5f},
        {"search", 1.3f}, {"find", 1.3f}, {"locate", 1.3f},
        {"play", 1.3f}, {"pause", 1.3f}, {"resume", 1.3f},
        {"save", 1.3f}, {"load", 1.3f}, {"backup", 1.3f},
        
        // Question indicators (commands disguised as questions)
        {"what", 0.8f}, {"how", 0.8f}, {"when", 0.8f}, {"where", 0.8f},
        {"tell", 1.0f}, {"give", 1.0f}, {"show", 1.0f},
        
        // Modal verbs
        {"can", 0.5f}, {"could", 0.5f}, {"would", 0.3f}, {"should", 0.5f}
    };
    
    // Banter indicators (social phrases)
    banterWeights_ = {
        // Greetings
        {"hello", 2.0f}, {"hi", 2.0f}, {"hey", 2.0f}, {"yo", 2.0f},
        {"morning", 1.8f}, {"afternoon", 1.8f}, {"evening", 1.8f},
        {"sup", 2.0f}, {"wassup", 2.0f}, {"howdy", 2.0f},
        
        // Gratitude
        {"thanks", 2.0f}, {"thank", 2.0f}, {"thx", 2.0f}, {"ty", 2.0f},
        {"appreciate", 1.5f}, {"grateful", 1.5f},
        
        // Affirmations
        {"yes", 1.0f}, {"yeah", 1.0f}, {"yep", 1.0f}, {"yup", 1.0f},
        {"no", 1.0f}, {"nope", 1.0f}, {"nah", 1.0f},
        {"ok", 1.2f}, {"okay", 1.2f}, {"sure", 1.2f}, {"alright", 1.2f},
        
        // Reactions
        {"lol", 2.0f}, {"haha", 2.0f}, {"lmao", 2.0f}, {"rofl", 2.0f},
        {"nice", 1.5f}, {"cool", 1.5f}, {"awesome", 1.5f}, {"great", 1.5f},
        {"wow", 1.5f}, {"omg", 1.5f}, {"damn", 1.5f},
        
        // Farewells
        {"bye", 2.0f}, {"goodbye", 2.0f}, {"later", 1.8f}, {"cya", 2.0f},
        {"peace", 1.8f}, {"night", 1.5f},
        
        // Small talk
        {"how", 0.8f}, {"doing", 1.2f}, {"you", 0.5f}, {"your", 0.5f},
        {"feeling", 1.2f}, {"good", 0.8f}, {"bad", 0.8f},
        
        // Internet slang
        {"brb", 2.0f}, {"afk", 2.0f}, {"btw", 1.5f}, {"imo", 1.5f},
        {"tbh", 1.5f}, {"nvm", 1.5f}, {"idk", 1.5f}
    };
    
    // ? INTEGRATION #1: Enhance with NLP rule data
    try {
        extern NLP g_nlp;
        auto nlpRules = g_nlp.getAllRules();
        
        int nlpEnhanced = 0;
        std::regex verbPattern(R"(\b(open|close|launch|run|show|list|set|create|delete|search|find|play|stop|kill)\b)");
        
        for (const auto& rule : nlpRules) {
            std::smatch match;
            std::string pattern = rule.pattern_str;
            
            // Extract command verbs from NLP patterns
            if (std::regex_search(pattern, match, verbPattern)) {
                std::string verb = match[1].str();
                
                // Calculate boost based on usage history
                float usageBoost = rule.usage_count * 0.01f;
                float successBoost = rule.success_rate * 0.5f;
                float totalBoost = 1.5f + usageBoost + successBoost;
                
                // Only boost if better than current weight
                if (commandWeights_.count(verb)) {
                    commandWeights_[verb] = std::max(commandWeights_[verb], totalBoost);
                } else {
                    commandWeights_[verb] = totalBoost;
                }
                
                nlpEnhanced++;
                
                LOG_DEBUG("FastClassifier", "NLP-enhanced '" + verb + "' weight: " + 
                         std::to_string(commandWeights_[verb]) + 
                         " (usage:" + std::to_string(rule.usage_count) + 
                         ", success:" + std::to_string(rule.success_rate) + ")");
            }
        }
        
        LOG_DEBUG("FastClassifier", "Enhanced " + std::to_string(nlpEnhanced) + 
                 " verbs from " + std::to_string(nlpRules.size()) + " NLP rules");
        
    } catch (const std::exception& e) {
        LOG_ERROR("FastClassifier", std::string("NLP enhancement failed: ") + e.what());
    }
    
    LOG_DEBUG("FastClassifier", "Initialized with " + 
             std::to_string(commandWeights_.size()) + " command indicators, " +
             std::to_string(banterWeights_.size()) + " banter indicators");
}

std::vector<std::string> FastClassifier::tokenize(const std::string& line) {
    std::vector<std::string> tokens;
    std::string lower = line;
    
    // Convert to lowercase
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    // Simple whitespace tokenization
    std::istringstream iss(lower);
    std::string word;
    while (iss >> word) {
        // Remove punctuation
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
        if (!word.empty()) {
            tokens.push_back(word);
        }
    }
    
    return tokens;
}

float FastClassifier::calculateScore(const std::vector<std::string>& tokens,
                                     const std::unordered_map<std::string, float>& weights) {
    float score = 0.0f;
    for (const auto& token : tokens) {
        auto it = weights.find(token);
        if (it != weights.end()) {
            score += it->second;
        }
    }
    return score;
}

FastResult FastClassifier::evaluate(const std::string& line, const ContextSnapshot& ctx) {
    auto tokens = tokenize(line);
    
    if (tokens.empty()) {
        return {IntentType::Unknown, 0.0f, "Empty input"};
    }
    
    // Calculate scores
    float commandScore = calculateScore(tokens, commandWeights_);
    float banterScore = calculateScore(tokens, banterWeights_);
    float learnedScore = calculateScore(tokens, learnedWeights_);
    
    // Add learned bias
    commandScore += learnedScore;
    
    // Context boost: if recent command, boost command score
    if (ctx.recentIntents.size() > 0 && ctx.recentIntents.back() == "command") {
        commandScore *= 1.2f;
    }
    
    // ? INTEGRATION #4: Context-aware boosting based on NLP history
    if (!ctx.lastNlpCategory.empty()) {
        LOG_DEBUG("FastClassifier", "Last NLP category: " + ctx.lastNlpCategory);
        
        // If last command was in "app" category, boost app-related verbs
        if (ctx.lastNlpCategory == "app") {
            for (const auto& token : tokens) {
                if (token == "open" || token == "close" || token == "launch" || 
                    token == "start" || token == "run" || token == "kill") {
                    commandScore += 0.3f;  // Context boost for app commands
                    LOG_DEBUG("FastClassifier", "App context boost for '" + token + "'");
                }
            }
        }
        // System category boost
        else if (ctx.lastNlpCategory == "system") {
            for (const auto& token : tokens) {
                if (token == "show" || token == "list" || token == "get" || 
                    token == "set" || token == "restart" || token == "shutdown") {
                    commandScore += 0.3f;
                    LOG_DEBUG("FastClassifier", "System context boost for '" + token + "'");
                }
            }
        }
    }
    
    // If user is in a "command flow", reduce banter threshold
    if (ctx.consecutiveCommands > 2) {
        banterScore *= 0.7f;  // Less likely to be banter mid-task
        LOG_DEBUG("FastClassifier", "Command flow detected (" + 
                 std::to_string(ctx.consecutiveCommands) + " consecutive) - reducing banter score");
    }
    
    // Recent commands boost (temporal context)
    auto now = std::time(nullptr);
    if (ctx.lastCommandTime > 0) {
        auto timeSinceLastCmd = now - ctx.lastCommandTime;
        if (timeSinceLastCmd < 30) {  // Within 30 seconds
            commandScore *= 1.1f;  // Likely another command
            LOG_DEBUG("FastClassifier", "Recent command boost (within 30s)");
        }
    }
    
    // Short inputs are more likely banter UNLESS they have strong command verbs
    bool hasStrongCommandVerb = std::any_of(tokens.begin(), tokens.end(), [](const std::string& t) {
        auto it = commandWeights_.find(t);
        return it != commandWeights_.end() && it->second >= 1.5f;
    });
    
    if (tokens.size() <= 2 && !hasStrongCommandVerb) {
        banterScore *= 1.3f;
    }
    
    // If we have a strong command verb, boost command score significantly
    if (hasStrongCommandVerb) {
        commandScore *= 1.8f;
    }
    
    // Questions can be either - check for command verbs
    bool hasQuestionWord = std::any_of(tokens.begin(), tokens.end(), [](const std::string& t) {
        return t == "what" || t == "how" || t == "when" || t == "where" || t == "why";
    });
    
    if (hasQuestionWord) {
        bool hasCommandVerb = std::any_of(tokens.begin(), tokens.end(), [](const std::string& t) {
            auto it = commandWeights_.find(t);
            return it != commandWeights_.end() && it->second > 1.0f;
        });
        
        if (hasCommandVerb) {
            commandScore *= 1.5f;
        } else {
            banterScore *= 1.2f;
        }
    }
    
    // Determine intent
    FastResult result;
    float total = commandScore + banterScore;
    
    if (total < 0.5f) {
        result.guess = IntentType::Unknown;
        result.confidence = 0.0f;
        result.reasoning = "No strong indicators";
    } else if (commandScore > banterScore) {
        result.guess = IntentType::Command;
        result.confidence = std::min(1.0f, commandScore / (total + 1.0f));
        result.reasoning = "Command score: " + std::to_string(commandScore) + 
                          " > Banter: " + std::to_string(banterScore);
    } else {
        result.guess = IntentType::Banter;
        result.confidence = std::min(1.0f, banterScore / (total + 1.0f));
        result.reasoning = "Banter score: " + std::to_string(banterScore) + 
                          " > Command: " + std::to_string(commandScore);
    }
    
    LOG_DEBUG("FastClassifier", "Evaluated: '" + line + "' ? " + 
             intentTypeToString(result.guess) + " (confidence: " + 
             std::to_string(result.confidence) + ") [Cmd:" + std::to_string(commandScore) + 
             " Ban:" + std::to_string(banterScore) + "]");
    
    return result;
}

void FastClassifier::updateWeights(const std::string& line, IntentType correctType) {
    auto tokens = tokenize(line);
    
    float adjustment = 0.1f;
    for (const auto& token : tokens) {
        if (correctType == IntentType::Command) {
            learnedWeights_[token] += adjustment;
        } else if (correctType == IntentType::Banter) {
            learnedWeights_[token] -= adjustment;
        }
    }
    
    LOG_DEBUG("FastClassifier", "Updated weights for: " + line + 
             " as " + intentTypeToString(correctType));
}

// ? INTEGRATION #5: Public method for AI to boost command weights
void FastClassifier::boostCommandWeight(const std::string& word, float weight) {
    commandWeights_[word] = std::max(commandWeights_[word], weight);
    LOG_DEBUG("FastClassifier", "AI boosted '" + word + "' to weight: " + std::to_string(weight));
}

} // namespace GRIM
