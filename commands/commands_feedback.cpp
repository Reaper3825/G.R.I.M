#include "commands_feedback.hpp"
#include "commands_core.hpp"
#include "commands_execution.hpp"
#include "logger.hpp"
#include "memory/unified_memory.hpp"
#include "../MMO/Core/SessionContextManager.hpp"
#include "../MMO/Core/CorrectionTuple.hpp"
#include "ai/ai_rl.hpp"
#include "helpers/grim_input.hpp"
#include "response_manager.hpp"
#include "console_history.hpp"
#include "voice/voice_speak.hpp"
#include "helpers/color.hpp"
#include "input_parser.hpp"
#include <algorithm>
#include <cctype>

extern GRIM::UnifiedMemoryStorage g_memoryStorage;
#define history getConsoleHistory()

static const std::string kDefaultSession = "default";

namespace GRIM {
namespace Feedback {

// ====================================================
// State variables — delegated to SessionContextManager
// ====================================================
bool hasPending() { return GRIM::MMO::SessionContextManager::instance().getPending(kDefaultSession).has_value(); }
bool hasPendingClarification() {
    auto p = GRIM::MMO::SessionContextManager::instance().getPending(kDefaultSession);
    return p.has_value() && p->kind == GRIM::MMO::PendingKind::Clarification;
}

void setPending(const std::string& command) {
    GRIM::MMO::SessionContextManager::instance().setPending(kDefaultSession, {
        GRIM::MMO::PendingKind::Confirmation,
        command, "", "", "", {}, 120
    });
}
void setPendingClarification(const std::string& command) {
    GRIM::MMO::SessionContextManager::instance().setPending(kDefaultSession, {
        GRIM::MMO::PendingKind::Clarification,
        command, "", "", "", {}, 120
    });
}

void clearPending() { GRIM::MMO::SessionContextManager::instance().clearPending(kDefaultSession); }
void clearPendingClarification() { GRIM::MMO::SessionContextManager::instance().clearPending(kDefaultSession); }

std::optional<std::string> getPending() {
    auto p = GRIM::MMO::SessionContextManager::instance().getPending(kDefaultSession);
    if (p.has_value() && p->kind != GRIM::MMO::PendingKind::Clarification)
        return p->original_command;
    return std::nullopt;
}
std::optional<std::string> getPendingClarification() {
    auto p = GRIM::MMO::SessionContextManager::instance().getPending(kDefaultSession);
    if (p.has_value() && p->kind == GRIM::MMO::PendingKind::Clarification)
        return p->original_command;
    return std::nullopt;
}

void setVoiceCommand(bool isVoice) {
    GRIM::MMO::SessionContextManager::instance().setVoiceCommand(kDefaultSession, isVoice);
}
bool isVoiceCommand() {
    return GRIM::MMO::SessionContextManager::instance().isVoiceCommand(kDefaultSession);
}

void setMultiCommandContext(bool isMulti) {
    GRIM::MMO::SessionContextManager::instance().setMultiCommandContext(kDefaultSession, isMulti);
}
bool isMultiCommandContext() {
    return GRIM::MMO::SessionContextManager::instance().isMultiCommandContext(kDefaultSession);
}

// ====================================================
// Helper functions
// ====================================================
static float calculateSimilarity(const std::string& s1, const std::string& s2)
{
    auto levenshtein = [](const std::string& a, const std::string& b) -> int {
        const size_t m = a.size(), n = b.size();
        std::vector<int> prev(n + 1), curr(n + 1);
        for (size_t j = 0; j <= n; ++j) prev[j] = static_cast<int>(j);
        for (size_t i = 1; i <= m; ++i) {
            curr[0] = static_cast<int>(i);
            for (size_t j = 1; j <= n; ++j) {
                int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
                curr[j] = std::min({prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost});
            }
            prev.swap(curr);
        }
        return prev[n];
    };
    
    int distance = levenshtein(s1, s2);
    return 1.0f - (static_cast<float>(distance) / std::max(s1.length(), s2.length()));
}

static std::string normalizeResponse(const std::string& response)
{
    std::string result = GRIMInput::normalizeCommand(response);
    result.erase(std::remove_if(result.begin(), result.end(), 
        [](char c) { return std::isspace(static_cast<unsigned char>(c)) || 
                           std::ispunct(static_cast<unsigned char>(c)); }), 
        result.end());
    std::transform(result.begin(), result.end(), result.begin(), 
        [](char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); });
    return result;
}

// ====================================================
// Clarification handler
// ====================================================
bool processClarificationResponse(const std::string& originalInput, const std::string& userResponse)
{
    LOG_DEBUG("LearnedCmd", "Processing clarification - original: \"" + originalInput + "\", response: \"" + userResponse + "\"");

    // Check if this was a prediction (suggested command)
    try {
        auto predictions = g_memoryStorage.getByTag("prediction");
        GRIM::MemoryObject* prediction = nullptr;
        
        for (auto& pred : predictions) {
            if (pred.raw == originalInput) {
                prediction = &pred;
                break;
            }
        }
        
        if (prediction) {
            std::string answer = normalizeResponse(userResponse);
            
            bool confirmed = (answer == "yes" || answer == "y" || answer == "yeah" || 
                             answer == "correct" || answer == "yep" || answer == "yup");
            bool denied = (answer == "no" || answer == "n" || answer == "nope" || 
                          answer == "wrong" || answer == "nah");
            
            if (confirmed) {
                LOG_DEBUG("Prediction", "User confirmed prediction: \"" + prediction->normalized + "\"");
                
                CommandExecution::storeLearnedCommand(originalInput, prediction->normalized, 0.95f);
                
                // Record successful prediction for RL
                try {
                    nlohmann::json reward = {
                        {"type", "successful_prediction"},
                        {"original_input", originalInput},
                        {"predicted_command", prediction->normalized},
                        {"confidence", prediction->confidence},
                        {"user_confirmed", true},
                        {"reward_multiplier", 1.5}
                    };
                    GRIM::RL::getAction(reward);
                    LOG_DEBUG("RL", "Successful prediction reward sent");
                } catch (const std::exception& e) {
                    LOG_ERROR("RL", std::string("Failed to send prediction reward: ") + e.what());
                }
                
                std::string resp = "Great! Executing " + prediction->normalized;
                history.push(resp, Colors::Green.toUInt());
                Voice::speak(resp, "learned");
                
                clearPendingClarification();
                
                // Execute the predicted command
                handleCommand(prediction->normalized);
                return true;
            }
            else if (denied) {
                LOG_DEBUG("Prediction", "User denied prediction");
                
                // Record failed prediction for RL
                try {
                    nlohmann::json penalty = {
                        {"type", "failed_prediction"},
                        {"original_input", originalInput},
                        {"predicted_command", prediction->normalized},
                        {"confidence", prediction->confidence},
                        {"user_confirmed", false},
                        {"reward_multiplier", -0.5}
                    };
                    GRIM::RL::getAction(penalty);
                    LOG_DEBUG("RL", "Failed prediction penalty sent");
                } catch (const std::exception& e) {
                    LOG_ERROR("RL", std::string("Failed to send prediction penalty: ") + e.what());
                }
                
                std::string resp = "I see. What did you mean?";
                history.push(resp, Colors::Cyan.toUInt());
                Voice::speak(resp, "clarify");
                return true;  // Keep clarification open
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("LearnedCmd", std::string("Failed to check predictions: ") + e.what());
    }

    // Standard clarification - user providing the correct command
    try {
        CommandExecution::storeLearnedCommand(originalInput, userResponse, 0.9f);
        clearPendingClarification();

        // Feed this learning event back to RL
        try {
            nlohmann::json rlFeedback = {
                {"type", "clarification_resolved"},
                {"original_input", originalInput},
                {"corrected_command", userResponse},
                {"confidence", 0.9},
                {"learning_source", "user_clarification"}
            };
            GRIM::RL::getAction(rlFeedback);
            LOG_DEBUG("RL", "Clarification learning signal sent: " + originalInput + " → " + userResponse);
        } catch (const std::exception& e) {
            LOG_ERROR("RL", std::string("Failed to send clarification to RL: ") + e.what());
        }

        std::string resp = "Got it — I'll remember that next time.";
        history.push(resp, Colors::Green.toUInt());
        Voice::speak(resp, "learned");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("LearnedCmd", std::string("Failed to store learned command: ") + e.what());
        std::string resp = "I couldn't save that one — try again later.";
        history.push(resp, Colors::Red.toUInt());
        Voice::speak(resp, "error");
        return true;
    }
}

// ====================================================
// Feedback handler
// ====================================================
bool processFeedbackResponse(const std::string& originalCmd, const std::string& userResponse)
{
    LOG_DEBUG("Feedback", "Processing feedback for command: \"" + originalCmd + "\"");
    LOG_DEBUG("Feedback", "User response: \"" + userResponse + "\"");
    
    // Check if user is simply confirming (yes/no)
    std::string answer = normalizeResponse(userResponse);
    
    bool isPositive = (answer == "yes" || answer == "y" || answer == "yeah" || 
                      answer == "correct" || answer == "yep" || answer == "yup" || answer == "right");
    bool isNegative = (answer == "no" || answer == "n" || answer == "nope" || 
                      answer == "wrong" || answer == "nah" || answer == "incorrect");

    if (isPositive || isNegative) {
        // Simple yes/no confirmation
        auto& scm = GRIM::MMO::SessionContextManager::instance();
        try {
            nlohmann::json fb = {
                {"type", "explicit_feedback"},
                {"command", originalCmd},
                {"feedback", isPositive ? "positive" : "negative"},
                {"user_text", userResponse},
                {"confidence", 1.0}
            };
            GRIM::RL::getAction(fb);
            LOG_DEBUG("Feedback", "Explicit feedback recorded: " + std::string(isPositive ? "positive" : "negative"));
            
            if (isPositive) {
                scm.recordUsage(kDefaultSession, originalCmd);
                // Record acceptance and mark any pending correction tuple as confirmed
                scm.recordUserResponse(kDefaultSession, false, "");
                auto snap = scm.snapshot(kDefaultSession);
                GRIM::MMO::CorrectionTupleCollector::instance().markConfirmed(
                    kDefaultSession, snap.turn_id);
            } else {
                // Record rejection — triggers correction tuple collection
                scm.recordUserResponse(kDefaultSession, true, "");
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Feedback", std::string("Error sending feedback to RL: ") + e.what());
        }

        std::string resp = isPositive ? "Got it — I'll keep doing that." : "Understood — I'll adjust next time.";
        history.push(resp, isPositive ? Colors::Green.toUInt() : Colors::Cyan.toUInt());
        Voice::speak(resp, "feedback");
        clearPending();
        return true;
    }
    
    // User might be repeating/clarifying the command - check similarity
    std::string normalizedResponse = GRIMInput::normalizeCommand(userResponse);
    std::string normalizedOriginal = GRIMInput::normalizeCommand(originalCmd);
    
    float similarity = calculateSimilarity(normalizedResponse, normalizedOriginal);
    LOG_DEBUG("Feedback", "Similarity score: " + std::to_string(similarity));
    
    // Check memory system for learned patterns
    float memoryConfidence = 0.0f;
    try {
        auto learned = g_memoryStorage.findLearnedCommand(normalizedResponse);
        if (learned.has_value()) {
            memoryConfidence = learned->confidence;
            LOG_DEBUG("Feedback", "Found learned command with confidence: " + std::to_string(memoryConfidence));
            
            if (learned->normalized == normalizedOriginal) {
                similarity = std::max(similarity, 0.9f);
                LOG_DEBUG("Feedback", "Memory confirms this is the same command");
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Feedback", std::string("Memory lookup failed: ") + e.what());
    }
    
    // Context confidence check
    float contextConfidence = 0.7f; // Placeholder
    
    // Decision logic based on confidence
    float finalConfidence = (similarity * 0.5f) + (memoryConfidence * 0.3f) + (contextConfidence * 0.2f);
    LOG_DEBUG("Feedback", "Final confidence: " + std::to_string(finalConfidence));
    
    if (similarity > 0.8f || finalConfidence > 0.75f) {
        // High confidence - user is repeating/confirming
        LOG_DEBUG("Feedback", "High confidence match - treating as implicit confirmation");
        
        auto& scm = GRIM::MMO::SessionContextManager::instance();
        try {
            nlohmann::json fb = {
                {"type", "implicit_confirmation"},
                {"command", originalCmd},
                {"feedback", "positive"},
                {"user_text", userResponse},
                {"confidence", finalConfidence},
                {"similarity", similarity},
                {"reason", "command_repetition"}
            };
            GRIM::RL::getAction(fb);
            
            CommandExecution::storeLearnedCommand(normalizedResponse, originalCmd, finalConfidence);
            scm.recordUsage(kDefaultSession, originalCmd);
            // Record acceptance and mark correction tuple confirmed
            scm.recordUserResponse(kDefaultSession, false, "");
            auto snap = scm.snapshot(kDefaultSession);
            GRIM::MMO::CorrectionTupleCollector::instance().markConfirmed(
                kDefaultSession, snap.turn_id);
        } catch (const std::exception& e) {
            LOG_ERROR("Feedback", std::string("Error recording implicit feedback: ") + e.what());
        }
        
        std::string resp = "Got it — command confirmed.";
        history.push(resp, Colors::Green.toUInt());
        Voice::speak(resp, "feedback");
        clearPending();
        return true;
    }
    else if (similarity > 0.5f || finalConfidence > 0.5f) {
        // Medium confidence - ask for clarification
        LOG_DEBUG("Feedback", "Medium confidence - asking for clarification");
        
        std::string clarification = "Did you mean \"" + originalCmd + "\" or \"" + normalizedResponse + "\"?";
        history.push(clarification, Colors::Yellow.toUInt());
        Voice::speak(clarification, "clarify");
        
        setPendingClarification(normalizedResponse);
        clearPending();
        return true;
    }
    else {
        // Low confidence - treat as a new command (user correction)
        LOG_DEBUG("Feedback", "Low confidence - treating as new command");
        
        // Record rejection + correction text → triggers correction tuple collection
        GRIM::MMO::SessionContextManager::instance().recordUserResponse(
            kDefaultSession, true, userResponse);

        try {
            nlohmann::json fb = {
                {"type", "command_change"},
                {"original_command", originalCmd},
                {"new_command", userResponse},
                {"confidence", finalConfidence},
                {"similarity", similarity},
                {"feedback", "negative"}
            };
            GRIM::RL::getAction(fb);
        } catch (const std::exception& e) {
            LOG_ERROR("Feedback", std::string("Error recording command change: ") + e.what());
        }
        
        clearPending();
        return false;  // Continue normal processing
    }
}

} // namespace Feedback
} // namespace GRIM
