#include "fast_classifier.hpp"
#include "intent_gate.hpp"
#include "logger.hpp"
#include "memory/context_manager.hpp"
#include "memory/memory_storage.hpp"
#include "nlp/nlp.hpp"
#include "weight_provider.hpp"
#include "flatbuffer_weight_provider.hpp"
#include "nlp_grammar_provider.hpp"
#include <algorithm>
#include <sstream>
#include <cctype>
#include <regex>

namespace GRIM {

// Static member initialization
std::vector<std::shared_ptr<IWeightProvider>> FastClassifier::providers_;
std::unordered_map<std::string, float> FastClassifier::commandWeights_;
std::unordered_map<std::string, float> FastClassifier::banterWeights_;
std::unordered_map<std::string, float> FastClassifier::questionWeights_;
std::unordered_map<std::string, float> FastClassifier::learnedWeights_;

void FastClassifier::init() {
    // Initialize weight providers
    providers_.clear();
    
    // 1. FlatBuffer provider (baseline weights from file, priority=50)
    auto fbProvider = std::make_shared<FlatBufferWeightProvider>("resources/classifier_weights.fb", 50);
    if (fbProvider->init()) {
        providers_.push_back(fbProvider);
    } else {
        LOG_DEBUG("FastClassifier", "Failed to load FlatBuffer weights, using minimal defaults");
    }
    
    // 2. NLP Grammar provider (dynamic weights from grammar DB, priority=75)
    auto nlpProvider = std::make_shared<NLPGrammarProvider>(75);
    if (nlpProvider->init()) {
        providers_.push_back(nlpProvider);
    } else {
        LOG_DEBUG("FastClassifier", "Failed to load NLP grammar weights");
    }
    
    // Sort providers by priority (descending)
    std::sort(providers_.begin(), providers_.end(),
              [](const auto& a, const auto& b) {
                  return a->getPriority() > b->getPriority();
              });
    
    // Load and merge weights from all providers
    loadWeightsFromProviders();
    
    LOG_DEBUG("FastClassifier", "Initialized with " + 
             std::to_string(commandWeights_.size()) + " command indicators, " +
             std::to_string(questionWeights_.size()) + " question indicators, " +
             std::to_string(banterWeights_.size()) + " banter indicators");
}

void FastClassifier::loadWeightsFromProviders() {
    // Clear existing weights
    commandWeights_.clear();
    questionWeights_.clear();
    banterWeights_.clear();
    
    // Merge weights from each provider in priority order
    for (const auto& provider : providers_) {
        std::string strategy = provider->getMergeStrategy();
        
        // Load command weights
        auto cmdWeights = provider->getWeights("command");
        if (!cmdWeights.empty()) {
            mergeWeights("command", cmdWeights, strategy);
        }
        
        // Load question weights
        auto qWeights = provider->getWeights("question");
        if (!qWeights.empty()) {
            mergeWeights("question", qWeights, strategy);
        }
        
        // Load banter weights
        auto bWeights = provider->getWeights("banter");
        if (!bWeights.empty()) {
            mergeWeights("banter", bWeights, strategy);
        }
    }
}

void FastClassifier::mergeWeights(const std::string& category, 
                                   const std::unordered_map<std::string, float>& newWeights,
                                   const std::string& strategy) {
    // Get reference to target weight map
    std::unordered_map<std::string, float>* targetWeights = nullptr;
    if (category == "command") {
        targetWeights = &commandWeights_;
    } else if (category == "question") {
        targetWeights = &questionWeights_;
    } else if (category == "banter") {
        targetWeights = &banterWeights_;
    } else {
        return;  // Unknown category
    }
    
    // Merge based on strategy
    if (strategy == "override") {
        // Override: Replace existing weights
        for (const auto& [token, weight] : newWeights) {
            (*targetWeights)[token] = weight;
        }
    } else if (strategy == "additive") {
        // Additive: Add to existing weights
        for (const auto& [token, weight] : newWeights) {
            (*targetWeights)[token] += weight;
        }
    } else if (strategy == "max") {
        // Max: Keep maximum weight
        for (const auto& [token, weight] : newWeights) {
            auto it = targetWeights->find(token);
            if (it == targetWeights->end() || weight > it->second) {
                (*targetWeights)[token] = weight;
            }
        }
    }
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
    float questionScore = calculateScore(tokens, questionWeights_);
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
    
    // Check for question markers (starts with question word, ends with ?)
    bool startsWithQuestionWord = !tokens.empty() && questionWeights_.count(tokens[0]) && 
                                  questionWeights_[tokens[0]] >= 1.8f;
    bool endsWithQuestionMark = !line.empty() && line.back() == '?';
    
    if (startsWithQuestionWord || endsWithQuestionMark) {
        questionScore *= 1.5f;  // Strong question indicator
    }
    
    // Short inputs are more likely banter UNLESS they have strong command verbs or question words
    bool hasStrongCommandVerb = std::any_of(tokens.begin(), tokens.end(), [](const std::string& t) {
        auto it = commandWeights_.find(t);
        return it != commandWeights_.end() && it->second >= 1.5f;
    });
    
    bool hasStrongQuestionWord = std::any_of(tokens.begin(), tokens.end(), [](const std::string& t) {
        auto it = questionWeights_.find(t);
        return it != questionWeights_.end() && it->second >= 1.8f;
    });
    
    if (tokens.size() <= 2 && !hasStrongCommandVerb && !hasStrongQuestionWord) {
        banterScore *= 1.3f;
    }
    
    // If we have a strong command verb, boost command score significantly
    if (hasStrongCommandVerb) {
        commandScore *= 1.8f;
    }
    
    // Questions with command verbs might be commands (e.g., "what is running?")
    if (hasStrongQuestionWord && hasStrongCommandVerb) {
        // Could be either - let scores compete
        questionScore *= 0.8f;  // Slight preference to command
    }
    
    // Determine intent - now with three-way classification
    FastResult result;
    float total = commandScore + banterScore + questionScore;
    
    if (total < 0.5f) {
        result.guess = IntentType::Unknown;
        result.confidence = 0.0f;
        result.reasoning = "No strong indicators";
    } else {
        // Find highest score
        if (questionScore > commandScore && questionScore > banterScore) {
            result.guess = IntentType::Question;
            result.confidence = std::min(1.0f, questionScore / (total + 1.0f));
            result.reasoning = "Question score: " + std::to_string(questionScore) + 
                              " > Command: " + std::to_string(commandScore) + 
                              ", Banter: " + std::to_string(banterScore);
        } else if (commandScore > banterScore) {
            result.guess = IntentType::Command;
            result.confidence = std::min(1.0f, commandScore / (total + 1.0f));
            result.reasoning = "Command score: " + std::to_string(commandScore) + 
                              " > Question: " + std::to_string(questionScore) +
                              ", Banter: " + std::to_string(banterScore);
        } else {
            result.guess = IntentType::Banter;
            result.confidence = std::min(1.0f, banterScore / (total + 1.0f));
            result.reasoning = "Banter score: " + std::to_string(banterScore) + 
                              " > Command: " + std::to_string(commandScore) +
                              ", Question: " + std::to_string(questionScore);
        }
    }
    
    LOG_DEBUG("FastClassifier", "Evaluated: '" + line + "' → " + 
             intentTypeToString(result.guess) + " (confidence: " + 
             std::to_string(result.confidence) + ") [Cmd:" + std::to_string(commandScore) + 
             " Q:" + std::to_string(questionScore) +
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
        } else if (correctType == IntentType::Question) {
            // Boost question weight for this token
            questionWeights_[token] = std::max(questionWeights_[token], 1.5f);
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
