#pragma once
#include <string>

namespace GRIM {

// Forward declaration
enum class IntentType;

// Local LLM intent classifier (uses existing AI backend: Mistral/LocalAI/OpenAI)
class LMIntent {
public:
    static void init();
    static void shutdown();
    
    // Ask AI backend for intent classification
    static IntentType askOllama(const std::string& line);
    
    // Check if AI backend is available
    static bool isAvailable();
    
private:
    static bool initialized_;
    static bool available_;
};

} // namespace GRIM
