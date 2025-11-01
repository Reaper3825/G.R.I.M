#pragma once
#include <string>
#include <unordered_map>
#include <mutex>

namespace GRIM {

// Forward declaration
struct ContextSnapshot;

// Simple intent classification
enum class IntentType {
    Command,    // Actionable instruction (open, run, close, etc.)
    Question,   // Information request (what is, how do, where can, etc.)
    Banter,     // Casual conversation (hey, thanks, lol, etc.)
    Unknown     // Uncertain - needs clarification
};

// Result from classification
struct IntentResult {
    IntentType type;
    float confidence;  // 0.0 to 1.0
    std::string reason; // Debug info
};

// Central intent routing gate
class IntentGate {
public:
    static void init();
    static void shutdown();
    
    // Main decision function - routes input to command or banter
    static IntentResult decide(const std::string& line, const ContextSnapshot& ctx);
    
    // Manual correction for learning
    static void correct(const std::string& line, IntentType correctType);
    
    // Cache management
    static void clearCache();
    
    // ? FIXED: Made public for access by helper functions
    static std::unordered_map<std::string, IntentType> cache_;
    static std::mutex cacheMutex_;
    
private:
    // Background resolution for uncertain cases
    static void asyncResolve(const std::string& line);
};

// Helper to convert IntentType to string
inline std::string intentTypeToString(IntentType type) {
    switch (type) {
        case IntentType::Command: return "command";
        case IntentType::Question: return "question";
        case IntentType::Banter: return "banter";
        case IntentType::Unknown: return "unknown";
        default: return "unknown";
    }
}

} // namespace GRIM
