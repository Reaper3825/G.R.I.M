#pragma once
#include <string>
#include <vector>
#include <unordered_map>

namespace GRIM {

// Forward declarations
enum class IntentType;
struct ContextSnapshot;

// Fast heuristic classifier result
struct FastResult {
    IntentType guess;
    float confidence;
    std::string reasoning;
};

// Lightweight keyword-based classifier
class FastClassifier {
public:
    static void init();
    
    // Evaluate input using heuristics + memory weights
    static FastResult evaluate(const std::string& line, const ContextSnapshot& ctx);
    
    // Update weights based on corrections
    static void updateWeights(const std::string& line, IntentType correctType);
    
    // ? INTEGRATION #5: Allow AI to boost specific command words
    static void boostCommandWeight(const std::string& word, float weight = 1.3f);
    
private:
    // Command indicators (verbs, action words)
    static std::unordered_map<std::string, float> commandWeights_;
    
    // Banter indicators (social phrases, greetings)
    static std::unordered_map<std::string, float> banterWeights_;
    
    // Learned weights from corrections
    static std::unordered_map<std::string, float> learnedWeights_;
    
    // Tokenize and normalize input
    static std::vector<std::string> tokenize(const std::string& line);
    
    // Calculate weighted score
    static float calculateScore(const std::vector<std::string>& tokens, 
                                const std::unordered_map<std::string, float>& weights);
};

} // namespace GRIM
