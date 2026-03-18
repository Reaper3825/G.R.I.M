#include "intent_gate.hpp"
#include "fast_classifier.hpp"
#include "lm_intent.hpp"
#include "logger.hpp"
#include "memory/context_snapshot.hpp"
#include "nlp/nlp.hpp"  // ? Access NLP for success rates
#include <unordered_map>
#include <optional>
#include <algorithm>

namespace GRIM {

// Static members
std::unordered_map<std::string, IntentType> IntentGate::cache_;
std::mutex IntentGate::cacheMutex_;

// Configuration
static constexpr float FAST_THRESHOLD = 0.7f;  // Confidence threshold for fast path

// ====================================================
// Initialization
// ====================================================
void IntentGate::init() {
    std::lock_guard<std::mutex> lock(cacheMutex_);  // ? Fixed typo
    FastClassifier::init();
    LOG_PHASE("Intent Gate initialized", true);
}

void IntentGate::shutdown() {
    std::lock_guard<std::mutex> lock(cacheMutex_);  // ? Fixed typo
    cache_.clear();
    LOG_PHASE("Intent Gate shutdown", true);
}

// ====================================================
// Cache helpers
// ====================================================
std::optional<IntentResult> getCached(const std::string& line) {
    std::lock_guard<std::mutex> lock(IntentGate::cacheMutex_);  // ? Fixed typo
    auto it = IntentGate::cache_.find(line);
    if (it != IntentGate::cache_.end()) {
        return IntentResult{it->second, 1.0f, "cached"};
    }
    return std::nullopt;
}

void cache(const std::string& line, const IntentResult& result) {
    std::lock_guard<std::mutex> lock(IntentGate::cacheMutex_);  // ? Fixed typo
    IntentGate::cache_[line] = result.type;
}

// ====================================================
// Main decision logic
// ====================================================
IntentResult IntentGate::decide(const std::string& line, const ContextSnapshot& ctx) {
    // Check cache first
    auto cached = getCached(line);
    if (cached.has_value()) {
        LOG_DEBUG("IntentGate", "Cache hit for: \"" + line + "\"");
        return cached.value();
    }
    
    // Fast path classification
    FastResult fast = FastClassifier::evaluate(line, ctx);
    
    LOG_DEBUG("IntentGate", "Fast result: " + intentTypeToString(fast.guess) + 
             " (confidence: " + std::to_string(fast.confidence) + ")");
    
    // ? INTEGRATION #2: Boost confidence with NLP success rate
    if (fast.confidence > FAST_THRESHOLD) {
        try {
            Intent nlpMatch = ::g_nlp.parse(line);
            
            if (nlpMatch.matched) {
                // Find the matching rule to get success_rate
                auto rules = ::g_nlp.getAllRules();
                for (const auto& rule : rules) {
                    if (rule.intent == nlpMatch.name) {
                        // Adjust confidence based on NLP history
                        float originalConf = fast.confidence;
                        fast.confidence *= rule.success_rate;
                        
                        LOG_DEBUG("IntentGate", "NLP success boost: " + 
                                 std::to_string(rule.success_rate) + 
                                 " (used " + std::to_string(rule.usage_count) + "x) " +
                                 std::to_string(originalConf) + " -> " + std::to_string(fast.confidence));
                        
                        // Store NLP category in cache metadata
                        IntentResult result = {fast.guess, fast.confidence, fast.reasoning + 
                                             " [NLP:" + rule.category + "]"};
                        cache(line, result);
                        return result;
                    }
                }
            }
        } catch (const std::exception& e) {
            LOG_ERROR("IntentGate", std::string("NLP boost failed: ") + e.what());
        }
        
        IntentResult result = {fast.guess, fast.confidence, fast.reasoning};
        cache(line, result);
        return result;
    }
    
    // Ambiguous - use LM fallback (or just return low confidence command)
    LOG_DEBUG("IntentGate", "Confidence below threshold (" + std::to_string(fast.confidence) + 
             " < " + std::to_string(FAST_THRESHOLD) + "), defaulting to low-confidence command");
    
    // For now, treat ambiguous as low-confidence command
    // TODO: Integrate LM classifier when implemented
    IntentResult result = {fast.guess, fast.confidence * 0.5f, fast.reasoning + " [low-conf]"};
    cache(line, result);
    return result;
}

// ====================================================
// Manual correction
// ====================================================
void IntentGate::correct(const std::string& line, IntentType correctType) {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    cache_[line] = correctType;
    
    // Update fast classifier weights
    FastClassifier::updateWeights(line, correctType);
    
    LOG_DEBUG("IntentGate", "Corrected: \"" + line + "\" ? " + intentTypeToString(correctType));
}

// ====================================================
// Cache management
// ====================================================
void IntentGate::clearCache() {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    cache_.clear();
    LOG_DEBUG("IntentGate", "Cache cleared");
}

} // namespace GRIM
