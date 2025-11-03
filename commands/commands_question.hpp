#pragma once
#include "commands_core.hpp"
#include <string>
#include <vector>

namespace GRIM {

// Question answer sources
enum class AnswerSource {
    UserMemory,         // From user's personal memories
    BaseMemory,         // From GRIM's base knowledge
    FieldMemory,        // From specialized field knowledge
    ExternalKnowledge,  // From web search or external APIs
    Vision,             // From OCR/image analysis
    OSINT,              // From OSINT/people lookup
    NotFound            // No answer available
};

// Result of question processing
struct QuestionResult {
    bool answered;
    std::string answer;
    AnswerSource source;
    float confidence;
    std::vector<std::string> references; // Source IDs or URLs
};

// Question handler - searches memory and external knowledge
class QuestionHandler {
public:
    // Process a question through memory hierarchy
    static QuestionResult process(const std::string& question);
    
    // Store important Q&A context in memory (public for cmdQuestion)
    static void storeQuestionContext(const std::string& question, 
                                    const QuestionResult& result);
    
private:
    // Check user's personal memories (learned facts, preferences, etc.)
    static QuestionResult searchUserMemory(const std::string& question);
    
    // Check GRIM's base knowledge (system info, built-in facts)
    static QuestionResult searchBaseMemory(const std::string& question);
    
    // Check field-specific knowledge (specialized domains)
    static QuestionResult searchFieldMemory(const std::string& question);
    
    // Search external sources (web, APIs)
    static QuestionResult searchExternalKnowledge(const std::string& question);
    
    // Search using vision/OCR (for "what do you see?" type questions)
    static QuestionResult searchVision(const std::string& question);
    
    // Search OSINT databases for people/usernames
    static QuestionResult searchOSINT(const std::string& query);
    
    // ✅ NEW: Search tool/command knowledge (what can GRIM do)
    static QuestionResult searchToolKnowledge(const std::string& question);
    
    // Helper: Perform web search using DuckDuckGo API
    static std::string performWebSearch(const std::string& query);
    
    // Helper: Fetch weather data using coordinates
    static std::string fetchWeatherData(double lat, double lon);
    
    // Helper: Perform OSINT lookup for username
    static std::string performOSINTLookup(const std::string& username);
    
    // Helper: Perform OCR on screen
    static std::string performScreenOCR();
    
    // Helper: Detect objects on screen
    static std::string performObjectDetection();
    
    // Helper: Determine if Q&A should be remembered
    static bool shouldRememberQA(const std::string& question, 
                                const QuestionResult& result);
    
    // Helper: Format answer based on question context
    static std::string formatAnswer(const std::string& question,
                                   const std::string& rawAnswer,
                                   AnswerSource source);
    
    // Helper: Convert first-person statements to second-person responses
    static std::string convertToSecondPerson(const std::string& text);
    
    // Extract question keywords for search
    static std::vector<std::string> extractKeywords(const std::string& question);
    
    // Determine if question requires external knowledge
    static bool needsExternalKnowledge(const std::string& question);
};

} // namespace GRIM

// Command function for question handling
CommandResult cmdQuestion(const std::string& question);
