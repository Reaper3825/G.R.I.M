#include "commands_question.hpp"
#include "../MMO/Core/ToolRegistry.hpp"
#include "response_manager.hpp"
#include "logger.hpp"
#include "memory/unified_memory.hpp" 
#include "ai/ai.hpp" 
#include "perception/perception.hpp" 
#include "perception/perception_context.hpp" // ✅ NEW: Context-aware perception
#include "external_collector/osit.hpp" 
#include "location.hpp" 
#include <algorithm>
#include <sstream>
#include <map>  // ✅ NEW: For category grouping
#include <regex>
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

// External references
extern GRIM::UnifiedMemoryStorage g_memoryStorage;
#include "memory/memory_buffer_rotation.hpp"

namespace GRIM {

// ============================================================================
// Main Question Processing
// ============================================================================

QuestionResult QuestionHandler::process(const std::string& question) {
    LOG_DEBUG("QuestionHandler", "Processing question: " + question);
    
    // Check for vision-specific questions first
    std::string lowerQ = question;
    std::transform(lowerQ.begin(), lowerQ.end(), lowerQ.begin(), ::tolower);
    
    if (lowerQ.find("see") != std::string::npos || 
        lowerQ.find("screen") != std::string::npos ||
        lowerQ.find("monitor") != std::string::npos ||
        lowerQ.find("display") != std::string::npos ||
        lowerQ.find("what's on") != std::string::npos ||
        lowerQ.find("what is on") != std::string::npos ||
        lowerQ.find("read text") != std::string::npos) {
        QuestionResult visionResult = searchVision(question);
        if (visionResult.answered) {
            return visionResult;
        }
    }
    
    // Check for OSINT-specific questions
    if (lowerQ.find("who is") != std::string::npos || 
        lowerQ.find("find user") != std::string::npos ||
        lowerQ.find("lookup") != std::string::npos) {
        QuestionResult osintResult = searchOSINT(question);
        if (osintResult.answered && osintResult.confidence > 0.5f) {
            return osintResult;
        }
    }
    
    // ✅ NEW: Check for tool/command/capability questions
    if (lowerQ.find("tool") != std::string::npos || 
        lowerQ.find("command") != std::string::npos ||
        lowerQ.find("what can you") != std::string::npos ||
        lowerQ.find("what do you") != std::string::npos ||
        lowerQ.find("what are you able") != std::string::npos ||
        lowerQ.find("capabilities") != std::string::npos ||
        lowerQ.find("functions") != std::string::npos ||
        lowerQ.find("features") != std::string::npos) {
        QuestionResult toolResult = searchToolKnowledge(question);
        if (toolResult.answered) {
            return toolResult;
        }
    }
    
    // Search hierarchy: User → Base → Field → External
    
    // 1. Check user's personal memories first
    QuestionResult result = searchUserMemory(question);
    if (result.answered && result.confidence > 0.7f) {
        LOG_DEBUG("QuestionHandler", "Found answer in user memory");
        return result;
    }
    
    // 2. Check GRIM's base knowledge
    QuestionResult baseResult = searchBaseMemory(question);
    if (baseResult.answered && baseResult.confidence > result.confidence) {
        LOG_DEBUG("QuestionHandler", "Found better answer in base memory");
        result = baseResult;
    }
    
    if (result.answered && result.confidence > 0.6f) {
        return result;
    }
    
    // 3. Check field-specific knowledge
    QuestionResult fieldResult = searchFieldMemory(question);
    if (fieldResult.answered && fieldResult.confidence > result.confidence) {
        LOG_DEBUG("QuestionHandler", "Found better answer in field memory");
        result = fieldResult;
    }
    
    if (result.answered && result.confidence > 0.5f) {
        return result;
    }
    
    // 4. If not found or low confidence, search externally
    if (!result.answered || needsExternalKnowledge(question)) {
        LOG_DEBUG("QuestionHandler", "Searching external knowledge");
        QuestionResult externalResult = searchExternalKnowledge(question);
        if (externalResult.answered) {
            return externalResult;
        }
    }
    
    // 5. Return best result or not found
    return result.answered ? result : QuestionResult{false, "", AnswerSource::NotFound, 0.0f, {}};
}

// ============================================================================
// User Memory Search
// ============================================================================

QuestionResult QuestionHandler::searchUserMemory(const std::string& question) {
    QuestionResult result;
    result.answered = false;
    result.source = AnswerSource::UserMemory;
    result.confidence = 0.0f;
    
    LOG_DEBUG("QuestionHandler", "Searching user memory for: " + question);
    
    // Extract keywords from question
    auto keywords = extractKeywords(question);
    
    if (keywords.empty()) {
        LOG_DEBUG("QuestionHandler", "No keywords extracted from question");
        return result;
    }
    
    std::vector<UnifiedMemoryObject> candidateMemories;
    
    // Search memory storage by keywords
    for (const auto& keyword : keywords) {
        auto memories = g_memoryStorage.search(keyword, 10);
        
        for (const auto& mem : memories) {
            // Check if this memory could answer the question
            // Look for fact-type memories with high confidence
            if (mem.type == TypeTag::FACT || mem.type == TypeTag::COMMAND) {
                candidateMemories.push_back(mem);
            }
        }
    }
    
    // Also search by tags (more specific)
    for (const auto& keyword : keywords) {
        auto tagMemories = g_memoryStorage.getByTag(keyword);
        for (const auto& mem : tagMemories) {
            if (mem.type == TypeTag::FACT) {
                candidateMemories.push_back(mem);
            }
        }
    }
    
    // Find the best matching memory
    UnifiedMemoryObject bestMatch;
    bool foundMatch = false;
    
    for (const auto& mem : candidateMemories) {
        if (mem.confidence > result.confidence) {
            bestMatch = mem;
            foundMatch = true;
            result.confidence = mem.confidence;
        }
    }
    
    if (foundMatch) {
        result.answered = true;
        
        // Format answer with proper perspective conversion
        std::string formattedAnswer = formatAnswer(question, bestMatch.raw, AnswerSource::UserMemory);
        
        std::ostringstream oss;
        oss << formattedAnswer;
        
        // Add timestamp context if memory is old
        auto now = std::time(nullptr);
        auto age = now - bestMatch.timestamp;
        if (age > 86400) { // More than 1 day old
            int days = age / 86400;
            oss << "\n(Remembered " << days << " day" << (days > 1 ? "s" : "") << " ago)";
        }
        
        result.answer = oss.str();
        result.references.push_back(std::to_string(bestMatch.id));
        
        LOG_DEBUG("QuestionHandler", "Found answer in user memory: " + std::to_string(bestMatch.id) + 
                  " (confidence: " + std::to_string(result.confidence) + ")");
    } else {
        LOG_DEBUG("QuestionHandler", "No matching memories found");
    }
    
    return result;
}

// ============================================================================
// Base Memory Search
// ============================================================================

QuestionResult QuestionHandler::searchBaseMemory(const std::string& question) {
    QuestionResult result;
    result.answered = false;
    result.source = AnswerSource::BaseMemory;
    result.confidence = 0.0f;
    
    LOG_DEBUG("QuestionHandler", "Searching base memory for: " + question);
    
    std::string lowerQ = question;
    std::transform(lowerQ.begin(), lowerQ.end(), lowerQ.begin(), ::tolower);
    
    // Check for system-related questions with detailed answers
    if (lowerQ.find("what is grim") != std::string::npos || 
        lowerQ.find("what are you") != std::string::npos ||
        lowerQ.find("who are you") != std::string::npos) {
        result.answered = true;
        result.answer = "I am GRIM - General Responsive Intelligence Module. I'm an AI assistant with:\n"
                       "• Natural language understanding and memory\n"
                       "• Command execution and automation capabilities\n"
                       "• Web search, OSINT, and computer vision\n"
                       "• Learning from interactions to improve over time";
        result.confidence = 1.0f;
        result.references.push_back("grim-identity");
        return result;
    }
    
    if (lowerQ.find("what can you do") != std::string::npos ||
        lowerQ.find("your capabilities") != std::string::npos ||
        lowerQ.find("help") != std::string::npos && lowerQ.find("what") != std::string::npos) {
        result.answered = true;
        result.answer = "My capabilities include:\n"
                       "• Answering questions (searching memory, web, or using AI)\n"
                       "• Remembering facts and preferences\n"
                       "• OSINT lookups across 400+ platforms\n"
                       "• Screen reading and object detection (OCR/vision)\n"
                       "• Executing commands and automation\n"
                       "• Natural conversation and learning\n\n"
                       "Try 'help' for commands or just ask me anything!";
        result.confidence = 1.0f;
        result.references.push_back("grim-capabilities");
        return result;
    }
    
    // Check for memory-related questions
    if (lowerQ.find("do you remember") != std::string::npos ||
        lowerQ.find("what do you know about") != std::string::npos) {
        // Extract the subject being asked about
        auto keywords = extractKeywords(question);
        if (!keywords.empty()) {
            LOG_DEBUG("QuestionHandler", "Memory inquiry about: " + keywords[0]);
            // This will fall through to general memory search below
        }
    }
    
    // Search base memory for system facts stored internally
    auto keywords = extractKeywords(question);
    for (const auto& keyword : keywords) {
        auto memories = g_memoryStorage.search(keyword, 10);
        for (const auto& mem : memories) {
            // Only return memories stored by GRIM itself
            if (mem.source == SourceType::GRIM_INTERNAL && mem.confidence > result.confidence) {
                result.answered = true;
                result.answer = mem.raw;
                result.confidence = mem.confidence;
                result.references.push_back(std::to_string(mem.id));
                
                LOG_DEBUG("QuestionHandler", "Found in base memory: " + std::to_string(mem.id));
            }
        }
    }
    
    return result;
}

// ============================================================================
// Field Memory Search
// ============================================================================

QuestionResult QuestionHandler::searchFieldMemory(const std::string& question) {
    QuestionResult result;
    result.answered = false;
    result.source = AnswerSource::FieldMemory;
    result.confidence = 0.0f;
    
    LOG_DEBUG("QuestionHandler", "Searching field memory for: " + question);
    
    // Search specialized domain knowledge
    // Search all memories for relevant information from Events and Status
    auto keywords = extractKeywords(question);
    
    if (keywords.empty()) {
        return result;
    }
    
    std::vector<UnifiedMemoryObject> candidateMemories;
    
    for (const auto& keyword : keywords) {
        auto memories = g_memoryStorage.search(keyword, 10);
        for (const auto& mem : memories) {
            // Look for informational memories (events, status updates, etc.)
            if (mem.type == TypeTag::EVENT || 
                mem.type == TypeTag::STATUS || 
                mem.type == TypeTag::SUMMARY) {
                candidateMemories.push_back(mem);
            }
        }
    }
    
    // Find best match
    UnifiedMemoryObject bestMatch;
    bool foundMatch = false;
    
    for (const auto& mem : candidateMemories) {
        if (mem.confidence > result.confidence) {
            bestMatch = mem;
            foundMatch = true;
            result.confidence = mem.confidence;
        }
    }
    
    if (foundMatch) {
        result.answered = true;
        result.answer = bestMatch.raw;
        result.references.push_back(std::to_string(bestMatch.id));
        
        // Add context based on memory type
        std::string typeContext;
        switch (bestMatch.type) {
            case TypeTag::EVENT:
                typeContext = " (from past events)";
                break;
            case TypeTag::STATUS:
                typeContext = " (from system status)";
                break;
            case TypeTag::SUMMARY:
                typeContext = " (from summaries)";
                break;
            default:
                typeContext = "";
        }
        
        if (!typeContext.empty()) {
            result.answer += typeContext;
        }
        
        LOG_DEBUG("QuestionHandler", "Found in field memory: " + std::to_string(bestMatch.id) + 
                  " (type: " + toString(bestMatch.type, TypeNames) + ")");
    } else {
        LOG_DEBUG("QuestionHandler", "No field memory found");
    }
    
    return result;
}

// ============================================================================
// External Knowledge Search
// ============================================================================

QuestionResult QuestionHandler::searchExternalKnowledge(const std::string& question) {
    QuestionResult result;
    result.answered = false;
    result.source = AnswerSource::ExternalKnowledge;
    result.confidence = 0.5f; // External results have moderate confidence
    
    LOG_DEBUG("QuestionHandler", "Searching external knowledge for: " + question);
    
    // Determine the type of external search needed
    std::string lowerQ = question;
    std::transform(lowerQ.begin(), lowerQ.end(), lowerQ.begin(), ::tolower);
    
    // Check for person/username lookup (OSINT)
    if (lowerQ.find("who is ") != std::string::npos || 
        lowerQ.find("find user") != std::string::npos ||
        lowerQ.find("lookup ") != std::string::npos) {
        
        // Extract potential username
        std::string username;
        size_t pos = lowerQ.find("who is ");
        if (pos != std::string::npos) {
            username = question.substr(pos + 7);
        } else if ((pos = lowerQ.find("find user ")) != std::string::npos) {
            username = question.substr(pos + 10);
        } else if ((pos = lowerQ.find("lookup ")) != std::string::npos) {
            username = question.substr(pos + 7);
        }
        
        // Clean username
        username.erase(std::remove_if(username.begin(), username.end(), ::ispunct), username.end());
        username.erase(std::remove_if(username.begin(), username.end(), ::isspace), username.end());
        
        if (!username.empty()) {
            LOG_DEBUG("QuestionHandler", "Attempting OSINT lookup for: " + username);
            result.answer = "OSINT lookup for '" + username + "' is available. Use the OSINT command for detailed results.";
            result.answered = true;
            result.confidence = 0.7f;
            result.references.push_back("osint:" + username);
            return result;
        }
    }
    
    // Check for web search queries (factual questions, definitions, etc.)
    if (lowerQ.find("what is") != std::string::npos ||
        lowerQ.find("define ") != std::string::npos ||
        lowerQ.find("meaning of") != std::string::npos ||
        lowerQ.find("how to") != std::string::npos ||
        lowerQ.find("why does") != std::string::npos) {
        
        LOG_DEBUG("QuestionHandler", "Question requires web search");
        
        // Perform actual web search
        std::string webResult = performWebSearch(question);
        if (!webResult.empty()) {
            result.answer = webResult;
            result.answered = true;
            result.confidence = 0.85f;
            result.references.push_back("web-search");
            return result;
        }
        
        // Web search failed, note it for AI fallback
        result.answer = "Web search unavailable. Using AI for answer.";
        result.confidence = 0.4f;
    }
    
    // Check for current/time-sensitive information
    if (lowerQ.find("latest") != std::string::npos ||
        lowerQ.find("current") != std::string::npos ||
        lowerQ.find("today") != std::string::npos ||
        lowerQ.find("news") != std::string::npos ||
        lowerQ.find("weather") != std::string::npos) {
        
        LOG_DEBUG("QuestionHandler", "Time-sensitive query detected");
        
        // ✅ NEW: Handle weather queries with real API
        if (lowerQ.find("weather") != std::string::npos || 
            lowerQ.find("temperature") != std::string::npos ||
            lowerQ.find("forecast") != std::string::npos) {
            
            // Try to get weather using location
            if (g_location.lat != 0.0 && g_location.lon != 0.0) {
                std::string weatherResult = fetchWeatherData(g_location.lat, g_location.lon);
                if (!weatherResult.empty()) {
                    result.answered = true;
                    result.answer = weatherResult;
                    result.confidence = 0.9f;
                    result.references.push_back("weather-api");
                    return result;
                }
            }
        }
        
        result.answer = "This requires real-time data. Using AI for best-effort answer.";
        result.confidence = 0.3f;
    }
    
    // Fall back to AI for general knowledge
    LOG_DEBUG("QuestionHandler", "Using AI for general knowledge question");
    
    try {
        // ✅ NEW: Build location-aware prompt for AI
        std::string prompt = "Please answer this question concisely and accurately: " + question;
        
        // Add explicit location context for location-aware questions
        bool isLocationQuery = (
            lowerQ.find("weather") != std::string::npos ||
            lowerQ.find("near me") != std::string::npos ||
            lowerQ.find("nearby") != std::string::npos ||
            lowerQ.find("local") != std::string::npos ||
            lowerQ.find("in my area") != std::string::npos ||
            lowerQ.find("around here") != std::string::npos
        );
        
        if (isLocationQuery && (g_location.lat != 0.0 || g_location.lon != 0.0)) {
            prompt += "\n\nUser is located in: " + g_location.fullAddress() + 
                     " (Latitude: " + std::to_string(g_location.lat) + 
                     ", Longitude: " + std::to_string(g_location.lon) + ")";
        }
        
        CommandResult aiResult = ai_process(prompt);
        
        if (aiResult.success && !aiResult.message.empty()) {
            result.answered = true;
            result.answer = aiResult.message;
            result.confidence = 0.6f; // AI answers have moderate-high confidence
            result.references.push_back("AI-generated");
        } else {
            result.answer = "I don't have enough information to answer that question.";
            result.confidence = 0.0f;
        }
    } catch (const std::exception& e) {
        LOG_ERROR("QuestionHandler", "AI search failed: " + std::string(e.what()));
        result.answer = "Unable to retrieve external information at this time.";
        result.confidence = 0.0f;
    }
    
    return result;
}

// ============================================================================
// Vision/OCR Search - Enhanced with Context-Aware Perception
// ============================================================================

QuestionResult QuestionHandler::searchVision(const std::string& question) {
    QuestionResult result;
    result.answered = false;
    result.source = AnswerSource::Vision;
    result.confidence = 0.0f;
    
    LOG_DEBUG("QuestionHandler", "Vision search requested: " + question);
    
    // Check if question is about visual content
    std::string lowerQ = question;
    std::transform(lowerQ.begin(), lowerQ.end(), lowerQ.begin(), ::tolower);
    
    bool isVisionQuestion = (
        lowerQ.find("see") != std::string::npos ||
        lowerQ.find("read") != std::string::npos ||
        lowerQ.find("text on screen") != std::string::npos ||
        lowerQ.find("screen") != std::string::npos ||
        lowerQ.find("monitor") != std::string::npos ||
        lowerQ.find("display") != std::string::npos ||
        lowerQ.find("what's on") != std::string::npos ||
        lowerQ.find("what is on") != std::string::npos ||
        lowerQ.find("show me") != std::string::npos ||
        lowerQ.find("looking at") != std::string::npos ||
        lowerQ.find("viewing") != std::string::npos
    );
    
    if (!isVisionQuestion) {
        return result; // Not a vision question
    }
    
    // ✅ NEW: Use context-aware perception system
    try {
        // Use the new perception context manager for intelligent answers
        std::string visionResult = GRIM::Perception::answerVisionQuestionWithContext(question);
        
        if (!visionResult.empty()) {
            result.answer = visionResult;
            result.answered = true;
            result.confidence = 0.85f; // Higher confidence with context
            result.references.push_back("perception_context");
            
            LOG_DEBUG("QuestionHandler", "Vision question answered with context (length: " + 
                      std::to_string(visionResult.length()) + " chars)");
            return result;
        }
    } catch (const std::exception& e) {
        LOG_ERROR("QuestionHandler", "Context-aware vision failed: " + std::string(e.what()));
        // Fall through to legacy methods
    }
    
    // ✅ FALLBACK: Legacy OCR/object detection for compatibility
    // Determine if OCR or object detection is needed
    bool needsOCR = (
        lowerQ.find("read") != std::string::npos ||
        lowerQ.find("text") != std::string::npos ||
        lowerQ.find("words") != std::string::npos
    );
    
    // Attempt to perform vision analysis using legacy methods
    std::string visionResult;
    if (needsOCR) {
        visionResult = performScreenOCR();
    } else {
        visionResult = performObjectDetection();
    }
    
    if (!visionResult.empty()) {
        result.answer = visionResult;
        result.answered = true;
        result.confidence = 0.75f; // Lower confidence for legacy methods
        result.references.push_back("perception");
    }
    
    return result;
}

// ============================================================================
// OSINT Search
// ============================================================================

QuestionResult QuestionHandler::searchOSINT(const std::string& query) {
    QuestionResult result;
    result.answered = false;
    result.source = AnswerSource::OSINT;
    result.confidence = 0.0f;
    
    LOG_DEBUG("QuestionHandler", "OSINT search for: " + query);
    
    // Extract username from query
    std::string username = query;
    std::string lowerQ = query;
    std::transform(lowerQ.begin(), lowerQ.end(), lowerQ.begin(), ::tolower);
    
    // Try to extract username from natural language
    if (lowerQ.find("who is ") != std::string::npos) {
        size_t pos = lowerQ.find("who is ");
        username = query.substr(pos + 7);
    } else if (lowerQ.find("find ") != std::string::npos) {
        size_t pos = lowerQ.find("find ");
        username = query.substr(pos + 5);
    } else if (lowerQ.find("lookup ") != std::string::npos) {
        size_t pos = lowerQ.find("lookup ");
        username = query.substr(pos + 7);
    }
    
    // Clean username
    username.erase(std::remove_if(username.begin(), username.end(), ::ispunct), username.end());
    username.erase(std::remove_if(username.begin(), username.end(), ::isspace), username.end());
    
    if (!username.empty()) {
        std::string osintResult = performOSINTLookup(username);
        if (!osintResult.empty()) {
            result.answer = osintResult;
            result.answered = true;
            result.confidence = 0.7f;
            result.references.push_back("osint");
        }
    }
    
    return result;
}

// ============================================================================
// Tool/Command Knowledge Search
// ============================================================================

QuestionResult QuestionHandler::searchToolKnowledge(const std::string& question) {
    QuestionResult result;
    result.answered = true;
    result.source = AnswerSource::BaseMemory;
    result.confidence = 0.95f;
    
    LOG_DEBUG("QuestionHandler", "Searching tool knowledge for: " + question);
    
    std::string lowerQ = question;
    std::transform(lowerQ.begin(), lowerQ.end(), lowerQ.begin(), ::tolower);
    
    // Get all registered tools
    auto allTools = GRIM::MMO::ToolRegistry::instance().getAllTools();
    
    // Check what type of question this is
    if (lowerQ.find("how many") != std::string::npos || 
        lowerQ.find("count") != std::string::npos) {
        // Count question
        result.answer = "I have " + std::to_string(allTools.size()) + 
                       " registered tools and commands available.";
        return result;
    }
    
    if (lowerQ.find("list") != std::string::npos || 
        lowerQ.find("show all") != std::string::npos ||
        lowerQ.find("what tools") != std::string::npos ||
        lowerQ.find("what commands") != std::string::npos) {
        // List all tools by category
        std::ostringstream answer;
        answer << "I have " << allTools.size() << " tools available:\n\n";
        
        // Group by category
        std::map<std::string, std::vector<std::string>> byCategory;
        for (const auto& tool : allTools) {
            byCategory[tool.category].push_back(tool.tool_id + ": " + tool.description);
        }
        
        for (const auto& [category, tools] : byCategory) {
            answer << "**" << category << "**:\n";
            int count = 0;
            for (const auto& tool : tools) {
                answer << "  - " << tool << "\n";
                if (++count >= 5 && tools.size() > 6) {
                    answer << "  ... and " << (tools.size() - 5) << " more\n";
                    break;
                }
            }
            answer << "\n";
        }
        
        answer << "Use 'list_tools [category]' for filtered lists or 'tool_info <name>' for details.";
        result.answer = answer.str();
        return result;
    }
    
    if (lowerQ.find("what can you") != std::string::npos ||
        lowerQ.find("capabilities") != std::string::npos ||
        lowerQ.find("features") != std::string::npos) {
        // General capabilities question
        std::ostringstream answer;
        answer << "I can help you with:\n\n";
        
        auto categories = GRIM::MMO::ToolRegistry::instance().getCategories();
        for (const auto& category : categories) {
            auto categoryTools = GRIM::MMO::ToolRegistry::instance().getByCategory(category);
            answer << "**" << category << "** (" << categoryTools.size() << " tools): ";
            
            // List first few examples
            int count = 0;
            for (const auto& tool : categoryTools) {
                if (count > 0) answer << ", ";
                answer << tool.tool_id;
                if (++count >= 3) break;
            }
            if (categoryTools.size() > 3) {
                answer << ", ...";
            }
            answer << "\n";
        }
        
        answer << "\nAsk 'what tools do you have' for a complete list.";
        result.answer = answer.str();
        return result;
    }
    
    // Check if asking about a specific capability
    for (const auto& tool : allTools) {
        // Check if tool name or description matches question keywords
        if (lowerQ.find(tool.tool_id) != std::string::npos ||
            tool.description.find(question) != std::string::npos) {
            result.answer = "Yes, I can help with that. Use the '" + tool.tool_id + 
                          "' command: " + tool.description;
            return result;
        }
    }
    
    // General answer
    result.answer = "I have " + std::to_string(allTools.size()) + 
                   " tools for actions, information queries, and system management. " +
                   "Try 'list_tools' to see everything I can do.";
    return result;
}

// ============================================================================
// Utility Functions
// ============================================================================

std::vector<std::string> QuestionHandler::extractKeywords(const std::string& question) {
    std::vector<std::string> keywords;
    
    // Remove question words and common stop words
    std::vector<std::string> stopWords = {
        "what", "where", "when", "who", "why", "how",
        "is", "are", "was", "were", "the", "a", "an",
        "do", "does", "did", "can", "could", "would",
        "should", "my", "your", "me", "you", "i"
    };
    
    // Split question into words
    std::istringstream iss(question);
    std::string word;
    
    while (iss >> word) {
        // Remove punctuation
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
        
        // Convert to lowercase
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        
        // Skip stop words and very short words
        if (word.length() > 2 && 
            std::find(stopWords.begin(), stopWords.end(), word) == stopWords.end()) {
            keywords.push_back(word);
        }
    }
    
    return keywords;
}

bool QuestionHandler::needsExternalKnowledge(const std::string& question) {
    std::string lowerQ = question;
    std::transform(lowerQ.begin(), lowerQ.end(), lowerQ.begin(), ::tolower);
    
    // Patterns that typically require external knowledge
    std::vector<std::string> externalPatterns = {
        "latest", "current", "news", "today",
        "weather", "temperature", "forecast",
        "stock", "price", "market",
        "definition of", "what does", "meaning of",
        "how to", "tutorial", "guide"
    };
    
    for (const auto& pattern : externalPatterns) {
        if (lowerQ.find(pattern) != std::string::npos) {
            return true;
        }
    }
    
    return false;
}

// ============================================================================
// Answer Formatting and Conversion
// ============================================================================

std::string QuestionHandler::convertToSecondPerson(const std::string& text) {
    std::string result = text;
    
    // Common first-person to second-person conversions
    struct Conversion {
        std::string from;
        std::string to;
        bool caseSensitive;
    };
    
    std::vector<Conversion> conversions = {
        // Pronouns
        {"My ", "Your ", false},
        {"my ", "your ", true},
        {" my ", " your ", true},
        {"I'm ", "You're ", false},
        {"I am ", "You are ", false},
        {"i'm ", "you're ", true},
        {"i am ", "you are ", true},
        {" I ", " you ", true},
        {" i ", " you ", true},
        {"Me ", "You ", false},
        {"me ", "you ", true},
        {" me ", " you ", true},
        
        // Possessives
        {"mine ", "yours ", true},
        {" mine", " yours", true},
        
        // Verbs (common patterns)
        {"I like ", "You like ", false},
        {"I prefer ", "You prefer ", false},
        {"I love ", "You love ", false},
        {"I hate ", "You hate ", false},
        {"I want ", "You want ", false},
        {"I need ", "You need ", false},
        {"I have ", "You have ", false},
        {"I've ", "You've ", false},
    };
    
    for (const auto& conv : conversions) {
        size_t pos = 0;
        while ((pos = result.find(conv.from, pos)) != std::string::npos) {
            // Check case sensitivity
            if (conv.caseSensitive || pos == 0 || !std::isalpha(result[pos - 1])) {
                result.replace(pos, conv.from.length(), conv.to);
                pos += conv.to.length();
            } else {
                pos += conv.from.length();
            }
        }
    }
    
    return result;
}

std::string QuestionHandler::formatAnswer(const std::string& question, 
                                         const std::string& rawAnswer,
                                         AnswerSource source) {
    std::string lowerQ = question;
    std::transform(lowerQ.begin(), lowerQ.end(), lowerQ.begin(), ::tolower);
    
    // Check if this is a personal question about the user
    bool isPersonalQuestion = (
        lowerQ.find("my ") != std::string::npos ||
        lowerQ.find("what's my") != std::string::npos ||
        lowerQ.find("what is my") != std::string::npos ||
        lowerQ.find("do i ") != std::string::npos ||
        lowerQ.find("am i ") != std::string::npos
    );
    
    // If it's from user memory and a personal question, convert to second person
    if (source == AnswerSource::UserMemory && isPersonalQuestion) {
        std::string converted = convertToSecondPerson(rawAnswer);
        
        // Handle specific question patterns for cleaner responses
        if (lowerQ.find("my name") != std::string::npos) {
            // Extract just the name if stored as "My name is Austin"
            size_t isPos = converted.find(" is ");
            if (isPos != std::string::npos) {
                std::string name = converted.substr(isPos + 4);
                // Trim whitespace
                name.erase(0, name.find_first_not_of(" \t\n\r"));
                name.erase(name.find_last_not_of(" \t\n\r") + 1);
                return name;  // Just return "Austin" instead of "Your name is Austin"
            }
        }
        
        if (lowerQ.find("my favorite") != std::string::npos || 
            lowerQ.find("my favourite") != std::string::npos) {
            // Extract just the favorite thing
            size_t isPos = converted.find(" is ");
            if (isPos != std::string::npos) {
                std::string favorite = converted.substr(isPos + 4);
                favorite.erase(0, favorite.find_first_not_of(" \t\n\r"));
                favorite.erase(favorite.find_last_not_of(" \t\n\r") + 1);
                return favorite;  // "Blue" instead of "Your favorite color is blue"
            }
        }
        
        return converted;
    }
    
    // For other sources, return as-is
    return rawAnswer;
}

// ============================================================================
// Context Memory Storage
// ============================================================================

bool QuestionHandler::shouldRememberQA(const std::string& question, const QuestionResult& result) {
    // Don't remember failed answers
    if (!result.answered || result.confidence < 0.5f) {
        return false;
    }
    
    std::string lowerQ = question;
    std::transform(lowerQ.begin(), lowerQ.end(), lowerQ.begin(), ::tolower);
    
    // ALWAYS remember personal information questions
    if (lowerQ.find("my name") != std::string::npos ||
        lowerQ.find("my age") != std::string::npos ||
        lowerQ.find("my favorite") != std::string::npos ||
        lowerQ.find("i like") != std::string::npos ||
        lowerQ.find("i prefer") != std::string::npos ||
        lowerQ.find("i am") != std::string::npos ||
        lowerQ.find("i'm ") != std::string::npos) {
        return true;
    }
    
    // Remember important factual questions from external sources
    if (result.source == AnswerSource::ExternalKnowledge && result.confidence >= 0.8f) {
        // Only remember definitions and facts, not time-sensitive data
        if (lowerQ.find("what is") != std::string::npos ||
            lowerQ.find("define") != std::string::npos ||
            lowerQ.find("who is") != std::string::npos ||
            lowerQ.find("who was") != std::string::npos) {
            
            // Skip time-sensitive queries
            if (lowerQ.find("latest") == std::string::npos &&
                lowerQ.find("current") == std::string::npos &&
                lowerQ.find("today") == std::string::npos &&
                lowerQ.find("news") == std::string::npos) {
                return true;
            }
        }
    }
    
    // Remember successful OSINT lookups
    if (result.source == AnswerSource::OSINT && result.confidence >= 0.7f) {
        return true;
    }
    
    // Remember vision analysis results if high confidence
    if (result.source == AnswerSource::Vision && result.confidence >= 0.8f) {
        return true;
    }
    
    // Don't remember base knowledge (already in system)
    if (result.source == AnswerSource::BaseMemory) {
        return false;
    }
    
    // Don't remember if it came from user memory (already stored)
    if (result.source == AnswerSource::UserMemory) {
        return false;
    }
    
    return false;
}

void QuestionHandler::storeQuestionContext(const std::string& question, const QuestionResult& result) {
    if (!shouldRememberQA(question, result)) {
        LOG_DEBUG("QuestionHandler", "Not storing Q&A - criteria not met");
        return;
    }
    
    LOG_DEBUG("QuestionHandler", "Storing Q&A context in memory");
    
    std::string lowerQ = question;
    std::transform(lowerQ.begin(), lowerQ.end(), lowerQ.begin(), ::tolower);
    
    // Create memory object
    UnifiedMemoryObject memory;
    memory.id = UnifiedMemoryObject::generateID();
    memory.timestamp = static_cast<uint64_t>(std::time(nullptr));
    memory.source = SourceType::GRIM_INTERNAL;  // GRIM learned this
    memory.intent = MemoryIntent::INFORM;
    memory.context = ContextType::CONVERSATION;
    memory.confidence = result.confidence;
    
    // Determine memory type and content based on question type
    if (lowerQ.find("my name") != std::string::npos ||
        lowerQ.find("my favorite") != std::string::npos ||
        lowerQ.find("i like") != std::string::npos ||
        lowerQ.find("i prefer") != std::string::npos ||
        lowerQ.find("i am") != std::string::npos ||
        lowerQ.find("i'm ") != std::string::npos) {
        
        // Personal fact about user
        memory.type = TypeTag::FACT;
        memory.raw = result.answer;
        memory.normalized = result.answer;
        
        // Extract tags from question keywords
        auto keywords = extractKeywords(question);
        for (const auto& keyword : keywords) {
            memory.tags.push_back(keyword);
        }
        
        LOG_DEBUG("QuestionHandler", "Storing personal fact: " + memory.raw);
        
    } else if (result.source == AnswerSource::ExternalKnowledge) {
        // External knowledge learned
        memory.type = TypeTag::FACT;
        memory.raw = "Q: " + question + "\nA: " + result.answer;
        memory.normalized = result.answer;
        
        // Tag with subject matter
        auto keywords = extractKeywords(question);
        for (const auto& keyword : keywords) {
            memory.tags.push_back(keyword);
        }
        memory.tags.push_back("learned-fact");
        
        LOG_DEBUG("QuestionHandler", "Storing learned fact from web");
        
    } else if (result.source == AnswerSource::OSINT) {
        // OSINT result
        memory.type = TypeTag::EVENT;  // OSINT lookups are events
        memory.raw = "OSINT: " + question + " -> " + result.answer;
        memory.normalized = result.answer;
        
        auto keywords = extractKeywords(question);
        for (const auto& keyword : keywords) {
            memory.tags.push_back(keyword);
        }
        memory.tags.push_back("osint");
        
        LOG_DEBUG("QuestionHandler", "Storing OSINT result");
        
    } else if (result.source == AnswerSource::Vision) {
        // Vision analysis
        memory.type = TypeTag::EVENT;
        memory.raw = "Vision: " + result.answer;
        memory.normalized = result.answer;
        memory.tags.push_back("vision");
        memory.tags.push_back("ocr");
        
        LOG_DEBUG("QuestionHandler", "Storing vision analysis");
    }
    
    // Stage in rotation pipeline (will flush to long-term on merge/sync)
    try {
        GRIM::MemoryBufferRotation::instance().preprocess(memory);
        LOG_DEBUG("QuestionHandler", "Staged context memory: " + std::to_string(memory.id));
    } catch (const std::exception& e) {
        LOG_ERROR("QuestionHandler", "Failed to stage memory: " + std::string(e.what()));
    }
}

} // namespace GRIM

// ============================================================================
// Command Entry Point
// ============================================================================

CommandResult cmdQuestion(const std::string& question) {
    if (question.empty()) {
        return {
            false,
            "[Question] Please ask a question",
            "ERR_EMPTY_QUESTION",
            "error",
            "Please ask a question",
            Colors::Red
        };
    }
    
    GRIM::QuestionResult result = GRIM::QuestionHandler::process(question);
    
    // Automatically store important context in memory
    if (result.answered && result.confidence >= 0.5f) {
        GRIM::QuestionHandler::storeQuestionContext(question, result);
    }
    
    if (result.answered) {
        std::string sourceStr;
        switch (result.source) {
            case GRIM::AnswerSource::UserMemory:
                sourceStr = "from your memories";
                break;
            case GRIM::AnswerSource::BaseMemory:
                sourceStr = "from base knowledge";
                break;
            case GRIM::AnswerSource::FieldMemory:
                sourceStr = "from specialized knowledge";
                break;
            case GRIM::AnswerSource::ExternalKnowledge:
                sourceStr = "from external sources";
                break;
            case GRIM::AnswerSource::Vision:
                sourceStr = "from vision/OCR analysis";
                break;
            case GRIM::AnswerSource::OSINT:
                sourceStr = "from OSINT lookup";
                break;
            default:
                sourceStr = "";
                break;
        }
        
        std::string confidenceStr = " (confidence: " + 
            std::to_string(static_cast<int>(result.confidence * 100)) + "%)";
        
        LOG_DEBUG("QuestionHandler", "Answered " + sourceStr + confidenceStr);
        
        // ✅ FIX: Truncate long vision responses for TTS (XTTS has 400 token limit ~= 300 chars safe)
        std::string voiceResponse = result.answer;
        const size_t MAX_TTS_LENGTH = 300;
        
        if (result.source == GRIM::AnswerSource::Vision && voiceResponse.length() > MAX_TTS_LENGTH) {
            // Intelligently truncate - find last complete sentence before limit
            size_t truncateAt = voiceResponse.find_last_of(".!?", MAX_TTS_LENGTH);
            if (truncateAt != std::string::npos && truncateAt > 100) {
                voiceResponse = voiceResponse.substr(0, truncateAt + 1);
            } else {
                // No sentence boundary found, hard truncate with ellipsis
                voiceResponse = voiceResponse.substr(0, MAX_TTS_LENGTH - 3) + "...";
            }
            
            LOG_DEBUG("QuestionHandler", "Truncated vision response for TTS: " + 
                      std::to_string(result.answer.length()) + " -> " + 
                      std::to_string(voiceResponse.length()) + " chars");
        }
        
        return {
            true,
            result.answer,  // Full answer for console/UI
            "",  // Empty error code for success
            "answer",
            voiceResponse,  // Truncated answer for voice
            Colors::Cyan
        };
    } else {
        return {
            false,
            "[Question] I don't have enough information to answer that question.",
            "ERR_NO_ANSWER",
            "error",
            "I don't know the answer to that",
            Colors::Yellow
        };
    }
}

// ============================================================================
// Web Search Implementation
// ============================================================================

std::string GRIM::QuestionHandler::performWebSearch(const std::string& query) {
    LOG_DEBUG("QuestionHandler", "Performing web search for: " + query);
    
    try {
        // ✅ NEW: Enhance query with location context if relevant
        std::string enhancedQuery = query;
        std::string lowerQuery = query;
        std::transform(lowerQuery.begin(), lowerQuery.end(), lowerQuery.begin(), ::tolower);
        
        // Check if query is location-dependent
        bool isLocationQuery = (
            lowerQuery.find("near me") != std::string::npos ||
            lowerQuery.find("nearby") != std::string::npos ||
            lowerQuery.find("in my area") != std::string::npos ||
            lowerQuery.find("local") != std::string::npos ||
            lowerQuery.find("weather") != std::string::npos ||
            lowerQuery.find("restaurants") != std::string::npos ||
            lowerQuery.find("stores") != std::string::npos ||
            lowerQuery.find("where can i") != std::string::npos ||
            lowerQuery.find("closest") != std::string::npos
        );
        
        if (isLocationQuery && (g_location.lat != 0.0 || g_location.lon != 0.0)) {
            // Replace "near me" with actual location
            std::string locationStr = " in " + g_location.shortAddress();
            
            if (lowerQuery.find("near me") != std::string::npos) {
                size_t pos = enhancedQuery.find("near me");
                if (pos != std::string::npos) {
                    enhancedQuery = enhancedQuery.substr(0, pos) + locationStr;
                }
            } else if (lowerQuery.find("nearby") != std::string::npos) {
                size_t pos = enhancedQuery.find("nearby");
                if (pos != std::string::npos) {
                    enhancedQuery = enhancedQuery.substr(0, pos) + locationStr;
                }
            } else {
                // Append location to query
                enhancedQuery += locationStr;
            }
            
            LOG_DEBUG("QuestionHandler", "Enhanced query with location: " + enhancedQuery);
        }
        
        // URL encode the query manually (simple version)
        std::string encodedQuery;
        for (char c : enhancedQuery) {
            if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
                encodedQuery += c;
            } else if (c == ' ') {
                encodedQuery += '+';
            } else {
                char hex[4];
                snprintf(hex, sizeof(hex), "%%%02X", static_cast<unsigned char>(c));
                encodedQuery += hex;
            }
        }
        
        std::string url = "https://api.duckduckgo.com/?q=" + encodedQuery + 
                         "&format=json&no_html=1&skip_disambig=1";
        
        LOG_DEBUG("QuestionHandler", "Searching: " + url);
        
        auto response = cpr::Get(
            cpr::Url{url},
            cpr::Timeout{5000},
            cpr::UserAgent{"GRIM/1.0"}
        );
        
        if (response.status_code == 200 && !response.text.empty()) {
            LOG_DEBUG("QuestionHandler", "Got response: " + std::to_string(response.text.length()) + " bytes");
            
            try {
                auto jsonResponse = nlohmann::json::parse(response.text);
                
                // Try different answer fields in priority order
                // 1. Abstract (most comprehensive)
                if (jsonResponse.contains("Abstract") && 
                    jsonResponse["Abstract"].is_string() && 
                    !jsonResponse["Abstract"].get<std::string>().empty()) {
                    
                    std::string answer = jsonResponse["Abstract"].get<std::string>();
                    
                    // Add source if available
                    if (jsonResponse.contains("AbstractURL") && 
                        jsonResponse["AbstractURL"].is_string()) {
                        answer += "\n\nSource: " + jsonResponse["AbstractURL"].get<std::string>();
                    }
                    
                    return answer;
                }
                
                // 2. Definition
                if (jsonResponse.contains("Definition") && 
                    jsonResponse["Definition"].is_string() && 
                    !jsonResponse["Definition"].get<std::string>().empty()) {
                    
                    std::string answer = jsonResponse["Definition"].get<std::string>();
                    
                    if (jsonResponse.contains("DefinitionURL") && 
                        jsonResponse["DefinitionURL"].is_string()) {
                        answer += "\n\nSource: " + jsonResponse["DefinitionURL"].get<std::string>();
                    }
                    
                    return answer;
                }
                
                // 3. Related Topics (first result)
                if (jsonResponse.contains("RelatedTopics") && 
                    jsonResponse["RelatedTopics"].is_array() && 
                    !jsonResponse["RelatedTopics"].empty()) {
                    
                    for (const auto& topic : jsonResponse["RelatedTopics"]) {
                        if (topic.contains("Text") && topic["Text"].is_string() && 
                            !topic["Text"].get<std::string>().empty()) {
                            
                            std::string answer = topic["Text"].get<std::string>();
                            
                            if (topic.contains("FirstURL") && topic["FirstURL"].is_string()) {
                                answer += "\n\nSource: " + topic["FirstURL"].get<std::string>();
                            }
                            
                            return answer;
                        }
                    }
                }
                
                LOG_DEBUG("QuestionHandler", "No usable content in JSON response");
                
            } catch (const nlohmann::json::exception& e) {
                LOG_ERROR("QuestionHandler", "JSON parsing failed: " + std::string(e.what()));
            }
        } else {
            LOG_ERROR("QuestionHandler", "HTTP request failed: " + std::to_string(response.status_code));
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("QuestionHandler", "Web search exception: " + std::string(e.what()));
    }
    
    return ""; // No results found
}

// ============================================================================
// Weather API Implementation
// ============================================================================

std::string GRIM::QuestionHandler::fetchWeatherData(double lat, double lon) {
    LOG_DEBUG("QuestionHandler", "Fetching weather for: " + std::to_string(lat) + ", " + std::to_string(lon));
    
    try {
        // Use Open-Meteo API (free, no API key required)
        std::string url = "https://api.open-meteo.com/v1/forecast?latitude=" + 
                         std::to_string(lat) + "&longitude=" + std::to_string(lon) +
                         "&current_weather=true&temperature_unit=fahrenheit";
        
        LOG_DEBUG("QuestionHandler", "Weather API URL: " + url);
        
        auto response = cpr::Get(
            cpr::Url{url},
            cpr::Timeout{5000},
            cpr::UserAgent{"GRIM/1.0"}
        );
        
        if (response.status_code == 200 && !response.text.empty()) {
            try {
                auto jsonResponse = nlohmann::json::parse(response.text);
                
                if (jsonResponse.contains("current_weather")) {
                    auto weather = jsonResponse["current_weather"];
                    
                    double temp = weather.value("temperature", 0.0);
                    double windSpeed = weather.value("windspeed", 0.0);
                    int weatherCode = weather.value("weathercode", 0);
                    
                    // Weather code to description mapping
                    std::string condition;
                    if (weatherCode == 0) condition = "Clear sky";
                    else if (weatherCode <= 3) condition = "Partly cloudy";
                    else if (weatherCode <= 49) condition = "Foggy";
                    else if (weatherCode <= 69) condition = "Rainy";
                    else if (weatherCode <= 79) condition = "Snowy";
                    else if (weatherCode <= 99) condition = "Thunderstorm";
                    else condition = "Unknown";
                    
                    std::ostringstream result;
                    result << "Current weather in " << g_location.shortAddress() << ":\n";
                    result << "Temperature: " << std::fixed << std::setprecision(1) << temp << "°F\n";
                    result << "Condition: " << condition << "\n";
                    result << "Wind Speed: " << std::fixed << std::setprecision(1) << windSpeed << " mph";
                    
                    return result.str();
                }
                
            } catch (const nlohmann::json::exception& e) {
                LOG_ERROR("QuestionHandler", "Weather JSON parsing failed: " + std::string(e.what()));
            }
        } else {
            LOG_ERROR("QuestionHandler", "Weather API request failed: " + std::to_string(response.status_code));
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("QuestionHandler", "Weather API exception: " + std::string(e.what()));
    }
    
    return ""; // Return empty on failure
}

// ============================================================================
// OSINT Lookup Implementation
// ============================================================================

std::string GRIM::QuestionHandler::performOSINTLookup(const std::string& username) {
    LOG_DEBUG("QuestionHandler", "Performing OSINT lookup for: " + username);
    
    // Check if Sherlock is available
    if (!isSherlockAvailable()) {
        LOG_ERROR("QuestionHandler", "Sherlock not available");
        return "OSINT lookup ready for '" + username + "'. Use command: osint lookup " + username;
    }
    
    try {
        // Run actual OSINT scan
        OSINTConfig config;
        config.useCache = true;
        config.timeoutSeconds = 10; // Quick scan for questions
        config.verboseOutput = false;
        
        LOG_DEBUG("QuestionHandler", "Running Sherlock scan...");
        OSINTReport report = runSelfAudit(username, config);
        
        if (report.success && report.totalFound > 0) {
            std::ostringstream oss;
            oss << "Found " << username << " on " << report.totalFound << " platform(s):\n";
            
            int count = 0;
            for (const auto& finding : report.findings) {
                if (finding.found && count < 5) { // Show top 5
                    oss << "  • " << finding.platform << ": " << finding.url << "\n";
                    count++;
                }
            }
            
            if (report.totalFound > 5) {
                oss << "  ... and " << (report.totalFound - 5) << " more\n";
            }
            
            oss << "\nUse 'osint lookup " << username << "' for full details.";
            return oss.str();
        } else if (report.success) {
            return "No profiles found for '" + username + "' on checked platforms.";
        } else {
            LOG_ERROR("QuestionHandler", "OSINT scan failed: " + report.error);
            return "OSINT lookup encountered an error. Try: osint lookup " + username;
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("QuestionHandler", "OSINT exception: " + std::string(e.what()));
    }
    
    return "OSINT lookup ready for '" + username + "'. Use command: osint lookup " + username;
}

// ============================================================================
// Vision/OCR Implementation
// ============================================================================

std::string GRIM::QuestionHandler::performScreenOCR() {
    LOG_DEBUG("QuestionHandler", "Performing screen OCR");
    
    // Check if perception is available
    if (!GRIM::Perception::isAvailable()) {
        LOG_ERROR("QuestionHandler", "Perception system not initialized");
        return "OCR capabilities ready. Use command: perception read text";
    }
    
    try {
        // Perform actual OCR on full screen
        std::string ocrText = GRIM::Perception::readText();
        
        if (!ocrText.empty() && ocrText.find("[Error]") == std::string::npos) {
            return "Screen text:\n" + ocrText;
        } else {
            LOG_DEBUG("QuestionHandler", "OCR returned no text or error");
            return "No readable text detected on screen.";
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("QuestionHandler", "OCR exception: " + std::string(e.what()));
    }
    
    return "OCR capabilities ready. Use command: perception read text";
}

std::string GRIM::QuestionHandler::performObjectDetection() {
    LOG_DEBUG("QuestionHandler", "Performing object detection");
    
    // Check if perception is available
    if (!GRIM::Perception::isAvailable()) {
        LOG_ERROR("QuestionHandler", "Perception system not initialized");
        return "Object detection ready. Use command: perception what see";
    }
    
    try {
        // Perform actual object detection
        std::string detectionResult = GRIM::Perception::detectObjects();
        
        if (!detectionResult.empty() && detectionResult.find("[Error]") == std::string::npos) {
            return detectionResult;
        } else {
            LOG_DEBUG("QuestionHandler", "Object detection returned no results");
            return "No objects detected on screen.";
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("QuestionHandler", "Detection exception: " + std::string(e.what()));
    }
    
    return "Object detection ready. Use command: perception what see";
}

