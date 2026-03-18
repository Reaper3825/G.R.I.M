#include "nlp_grammar_provider.hpp"
#include "../logger.hpp"
#include <algorithm>
#include <regex>

namespace GRIM {

NLPGrammarProvider::NLPGrammarProvider(int priority)
    : priority_(priority) {}

bool NLPGrammarProvider::init() {
    try {
        // Extract weights from grammar database
        extractCommandWeights();
        extractQuestionWeights();
        extractLocationWeights();
        extractBanterWeights();

        logDebug("NLPGrammarProvider", "NLPGrammarProvider initialized with " + 
                std::to_string(weightCache_.size()) + " categories");
        return true;
    } catch (const std::exception& e) {
        logError("NLPGrammarProvider", "NLPGrammarProvider init failed: " + std::string(e.what()));
        return false;
    }
}

void NLPGrammarProvider::extractCommandWeights() {
    auto nlpRules = ::g_nlp.getAllRules();
    
    std::unordered_map<std::string, float> commandWeights;
    std::regex verbPattern(R"(\b(open|close|launch|run|show|list|set|create|delete|search|find|play|stop|kill|start|end|save|load|change|update|modify|make|remove|install|uninstall|download|restart|reboot|shutdown|pause|resume|backup)\b)");
    
    for (const auto& rule : nlpRules) {
        std::smatch match;
        std::string pattern = rule.pattern_str;
        
        if (std::regex_search(pattern, match, verbPattern)) {
            std::string verb = match[1].str();
            
            // Calculate weight based on usage history
            float usageBoost = rule.usage_count * 0.01f;
            float successBoost = rule.success_rate * 0.5f;
            float totalBoost = 1.5f + usageBoost + successBoost;
            
            commandWeights[verb] = std::max(commandWeights[verb], totalBoost);
        }
    }
    
    weightCache_["command"] = std::move(commandWeights);
}

void NLPGrammarProvider::extractQuestionWeights() {
    auto nlpRules = ::g_nlp.getAllRules();
    
    std::unordered_map<std::string, float> questionWeights;
    std::regex questionPattern(R"(\b(what|who|where|when|why|how|which|whats|whos|wheres|whens|whys|hows)\b)");
    
    for (const auto& rule : nlpRules) {
        std::smatch match;
        std::string pattern = rule.pattern_str;
        
        if (std::regex_search(pattern, match, questionPattern)) {
            std::string questionWord = match[1].str();
            
            float usageBoost = rule.usage_count * 0.01f;
            float successBoost = rule.success_rate * 0.3f;
            float totalBoost = 2.0f + usageBoost + successBoost;
            
            questionWeights[questionWord] = std::max(questionWeights[questionWord], totalBoost);
        }
    }
    
    weightCache_["question"] = std::move(questionWeights);
}

void NLPGrammarProvider::extractLocationWeights() {
    auto nlpRules = ::g_nlp.getAllRules();
    
    std::unordered_map<std::string, float> locationWeights;
    std::regex locationPattern(R"(\b(weather|temperature|forecast|near|nearby|local|around|location|place|store|restaurant|closest|nearest|find|best|recommendation|area|city|address)\b)");
    
    for (const auto& rule : nlpRules) {
        std::smatch match;
        std::string pattern = rule.pattern_str;
        
        if (std::regex_search(pattern, match, locationPattern)) {
            std::string locationWord = match[1].str();
            
            float usageBoost = rule.usage_count * 0.01f;
            float totalBoost = 1.5f + usageBoost;
            
            locationWeights[locationWord] = std::max(locationWeights[locationWord], totalBoost);
        }
    }
    
    // Location keywords go into question weights (they're query-related)
    auto& questionWeights = weightCache_["question"];
    for (const auto& [token, weight] : locationWeights) {
        questionWeights[token] = std::max(questionWeights[token], weight);
    }
}

void NLPGrammarProvider::extractBanterWeights() {
    // Banter patterns are less structured, use minimal extraction
    // Most banter comes from baseline weights rather than grammar rules
    std::unordered_map<std::string, float> banterWeights;
    
    // Could extract from greetings/farewells in rules if needed
    // For now, keep empty - baseline provider will handle banter
    
    weightCache_["banter"] = std::move(banterWeights);
}

std::unordered_map<std::string, float> NLPGrammarProvider::getWeights(const std::string& category) const {
    auto it = weightCache_.find(category);
    if (it != weightCache_.end()) {
        return it->second;
    }
    return {};
}

} // namespace GRIM
