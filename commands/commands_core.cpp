#include "commands_ai.hpp"
#include "commands_memory.hpp"
#include "commands_interface.hpp"
#include "commands_filesystem.hpp"
#include "commands_timers.hpp"
#include "commands_voice.hpp"
#include "commands_system.hpp"
#include "commands_aliases.hpp"
#include "response_manager.hpp"
#include "console_history.hpp"
#include "voice/voice_speak.hpp"
#include "error_manager.hpp"
#include "resources.hpp"
#include "nlp/nlp.hpp"
#include "synonyms.hpp"
#include "commands_core.hpp"
#include "aliases.hpp"
#include "ai/personality_manager.hpp"
#include "ai/proactive_dialogue.hpp"
#include "logger.hpp"
#include "memory/context_manager.hpp"
#include "ai/ai_rl.hpp"
#include "memory/memory_manager.hpp"
#include "memory/memory_storage.hpp"
#include "memory/memory_router.hpp"
#include "input_parser.hpp"
#include "ai/ai.hpp"
#include "core/plugin.hpp"
#include "helpers/color.hpp"
#include <crtdbg.h>

#define CHECK_HEAP() _CrtCheckMemory()

using Voice::speak;

// ====================================================
// Globals
// ====================================================
static std::optional<std::string> g_pendingClarifyCmd;
static std::optional<std::string> g_pendingFeedbackCmd;
static bool g_isMultiCommandContext = false;  // Track if we're in multi-command mode

// ✅ Add getter function for external access
bool hasPendingFeedback() {
    return g_pendingFeedbackCmd.has_value();
}

extern nlohmann::json longTermMemory;
extern nlohmann::json aiConfig;
extern NLP g_nlp;
#define history getConsoleHistory()
extern GRIM::MemoryStorage g_memoryStorage;

Intent g_lastIntent;

// ====================================================
// Global learned command map
// ====================================================
static std::unordered_map<std::string, std::string> g_learnedCommandMap;

static CommandResult handleLearnedCommand(const std::string& arg)
{
    for (auto& pair : g_learnedCommandMap)
    {
        const std::string& phrase = pair.first;
        const std::string& action = pair.second;

        if (arg == phrase || arg.find(phrase) != std::string::npos)
        {
            LOG_DEBUG("LearnedCmd", "Executing learned command \"" + phrase + "\" → \"" + action + "\"");
            return dispatchCommand(action, "");
        }
    }

    return CommandResult{
        false,                                            // success
        "[Error] Unknown learned command route.",         // message
        "ERR_NONE",                                       // errorCode
        "",                                               // category
        "",                                               // voice
        Colors::Red                                       // color
    };
}

// ====================================================
// Core Plugin Loader
// ====================================================
void ensureCorePluginsRegistered()
{
    static bool done = false;
    if (done) return;
    done = true;

    LOG_DEBUG("Core", "Ensuring core plugins registered...");
    extern void registerCorePlugin();
    registerCorePlugin();
}

// ====================================================
// Core Dispatch
// ====================================================
CommandResult dispatchCommand(const std::string& cmd, const std::string& arg)
{
    ensureCorePluginsRegistered();

    // 1. Built-in / Plugin command path
    auto it = commandMap.find(cmd);
    if (it != commandMap.end())
    {
        LOG_DEBUG("Dispatch", "Running command \"" + cmd + "\" arg=\"" + arg + "\"");
        try {
            return it->second(arg);
        } catch (const std::exception& e) {
            LOG_ERROR("Dispatch", "Exception in command \"" + cmd + "\": " + e.what());
            return CommandResult{
                false,                                                  // success
                "[Error] Exception while running command: " + cmd,      // message
                "ERR_CMD_EXCEPTION",                                    // errorCode
                "",                                                     // category
                "",                                                     // voice
                Colors::Red                                             // color
            };
        }
    }

    // 2. Learned-command lookup
    try {
        auto learned = g_memoryStorage.findLearnedCommand(cmd);
        if (learned.has_value()) {
            LOG_DEBUG("Dispatch", "Matched learned command: \"" + learned->raw + "\" → \"" + learned->normalized + "\"");
            return dispatchCommand(learned->normalized, arg);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Dispatch", std::string("Learned-command lookup failed: ") + e.what());
    }

    // 3. Record unknown command
    try {
        GRIM::MemoryObject unknown;
        unknown.id = GRIM::MemoryObject::generateUUID();
        unknown.timestamp = std::time(nullptr);
        unknown.source = GRIM::SourceTag::UserText;
        unknown.type = GRIM::TypeTag::UnknownCommand;
        unknown.intent = GRIM::IntentTag::Query;
        unknown.context = GRIM::ContextTag::Conversation;
        unknown.raw = cmd + (arg.empty() ? "" : " " + arg);
        unknown.normalized = normalizeWord(cmd);
        unknown.confidence = 0.4f;
        unknown.tags = {"intercepted", "unclassified", "pending_analysis"};

        g_memoryStorage.storeShortTerm(unknown);
        GRIM::MemoryRouter::dispatch(unknown);

        LOG_DEBUG("Dispatch", "Unknown command recorded for later clarification");
    } catch (const std::exception& e) {
        LOG_ERROR("Dispatch", std::string("Failed to record unknown command: ") + e.what());
    }

    // 4. Try RL bridge reasoning
    try {
        nlohmann::json obs = {
            {"type","unknown_command"},
            {"input", cmd + " " + arg},
            {"context", longTermMemory}
        };

        auto rlRes = GRIM::RL::getAction(obs);
        if (rlRes.contains("suggested_command")) {
            std::string inferred = rlRes["suggested_command"].get<std::string>();
            LOG_DEBUG("RL", "RL suggested command: " + inferred);
            return dispatchCommand(inferred, arg);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Dispatch", std::string("RL reasoning failed: ") + e.what());
    }

    // 5. AI reasoning fallback
    try {
        LOG_TRACE("Dispatch", "Calling ai_interpret() for unknown command...");
        
        // ✅ BEFORE calling AI, check if we have similar learned commands
        std::vector<std::pair<std::string, float>> suggestions;
        try {
            auto learnedCommands = g_memoryStorage.getAllLearnedCommands();
            
            // Calculate similarity to all learned commands
            auto levenshtein = [](const std::string& s1, const std::string& s2) -> int {
                const size_t m = s1.size(), n = s2.size();
                std::vector<int> prev(n + 1), curr(n + 1);
                for (size_t j = 0; j <= n; ++j) prev[j] = static_cast<int>(j);
                for (size_t i = 1; i <= m; ++i) {
                    curr[0] = static_cast<int>(i);
                    for (size_t j = 1; j <= n; ++j) {
                        int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
                        curr[j] = std::min({prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost});
                    }
                    prev.swap(curr);
                }
                return prev[n];
            };
            
            std::string fullInput = cmd + (arg.empty() ? "" : " " + arg);
            
            for (const auto& learned : learnedCommands) {
                int distance = levenshtein(fullInput, learned.raw);
                float similarity = 1.0f - (static_cast<float>(distance) / 
                                  std::max(fullInput.length(), learned.raw.length()));
                
                // Weight by learned confidence
                float score = similarity * learned.confidence;
                
                if (similarity > 0.6f) { // Only consider fairly similar commands
                    suggestions.push_back({learned.normalized, score});
                    LOG_DEBUG("Dispatch", "Similar learned command: \"" + learned.raw + 
                             "\" (similarity=" + std::to_string(similarity) + 
                             ", confidence=" + std::to_string(learned.confidence) + 
                             ", score=" + std::to_string(score) + ")");
                }
            }
            
            // Sort by score descending
            std::sort(suggestions.begin(), suggestions.end(), 
                     [](const auto& a, const auto& b) { return a.second > b.second; });
            
        } catch (const std::exception& e) {
            LOG_ERROR("Dispatch", std::string("Failed to check learned commands: ") + e.what());
        }
        
        // ✅ If we have a high-confidence suggestion, ask user instead of calling AI
        if (!suggestions.empty() && suggestions[0].second > 0.75f) {
            std::string suggestedCmd = suggestions[0].first;
            float confidence = suggestions[0].second;
            
            LOG_DEBUG("Dispatch", "High-confidence suggestion found: \"" + suggestedCmd + 
                     "\" (score=" + std::to_string(confidence) + ")");
            
            std::string question = "Did you mean \"" + suggestedCmd + "\"?";
            history.push(question, Colors::Yellow.toUInt());
            Voice::speak(question, "clarify");
            
            // Store as pending clarification with the suggested command
            g_pendingClarifyCmd = cmd + (arg.empty() ? "" : " " + arg);
            
            // Store the suggestion in memory for the clarification handler
            GRIM::MemoryObject suggestion;
            suggestion.id = GRIM::MemoryObject::generateUUID();
            suggestion.timestamp = std::time(nullptr);
            suggestion.source = GRIM::SourceTag::GrimInternal;
            suggestion.type = GRIM::TypeTag::Prediction;
            suggestion.intent = GRIM::IntentTag::Query;
            suggestion.context = GRIM::ContextTag::Conversation;
            suggestion.raw = cmd + (arg.empty() ? "" : " " + arg);
            suggestion.normalized = suggestedCmd;
            suggestion.confidence = confidence;
            suggestion.tags = {"prediction", "learned_command", "high_confidence"};
            g_memoryStorage.storeShortTerm(suggestion);
            
            return CommandResult{
                true,                                           // success (user interaction pending)
                "Suggestion: " + suggestedCmd,                  // message
                "ERR_NONE",                                     // errorCode
                "clarify",                                      // category
                "",                                             // voice (already spoken)
                Colors::Yellow                                  // color
            };
        }
        
        // ✅ Medium confidence - mention suggestion but still try AI
        if (!suggestions.empty() && suggestions[0].second > 0.5f) {
            LOG_DEBUG("Dispatch", "Medium-confidence suggestion available, trying AI first");
        }
        
        CommandResult aiRes = ai_interpret(cmd + " " + arg, true);

        if (aiRes.success && aiRes.category == "command_infer")
        {
            std::string inferred = GRIMInput::normalizeCommand(aiRes.message);
            LOG_DEBUG("AI", "AI inferred new command: " + inferred);

            g_memoryStorage.storeLearnedCommand(cmd, inferred, 0.75f);
            g_learnedCommandMap[cmd] = inferred;
            commandMap[cmd] = handleLearnedCommand;
            CHECK_HEAP();
            std::string resp = ResponseManager::get(
                "Got it — I've learned that \"" + cmd + "\" means \"" + inferred + "\".");
            history.push(resp, (Colors::Green.a << 24) | (Colors::Green.b << 16) | (Colors::Green.g << 8) | Colors::Green.r);
            Voice::speak(resp, "learned");

            return CommandResult{
                true,                                       // success
                "Learned new command: " + inferred,         // message
                "ERR_NONE",                                 // errorCode
                "",                                         // category
                "",                                         // voice
                Colors::Green                               // color
            };
        }
        else if (aiRes.success && aiRes.category == "conversation")
        {
            LOG_DEBUG("AI", "AI interpret returned conversational intent.");
            return aiRes;
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Dispatch", std::string("AI fallback failed: ") + e.what());
    }

    // 6. Clarification fallback
    try {
        static std::string lastAsked;
        if (cmd != lastAsked && !cmd.empty()) {
            std::string clarification;
            if (GRIM::PersonalityManager::isStable())
                clarification = "Did you mean \"" + GRIMInput::normalizeCommand(cmd) + "\" or something else?";
            else
                clarification = "I didn’t quite catch that. Can you rephrase?";

            std::string resp = ResponseManager::get(clarification);
            history.push(resp, Colors::Cyan.toUInt());
            Voice::speak(resp, "clarify");

            GRIM::MemoryObject clarify;
            clarify.id = GRIM::MemoryObject::generateUUID();
            clarify.timestamp = std::time(nullptr);
            clarify.source = GRIM::SourceTag::GrimInternal;
            clarify.type = GRIM::TypeTag::Event;
            clarify.intent = GRIM::IntentTag::Query;
            clarify.context = GRIM::ContextTag::Conversation;
            clarify.raw = clarification;
            clarify.normalized = "clarification_prompt";
            clarify.confidence = 0.8f;
            clarify.tags = {"clarify", "dialogue"};

            g_memoryStorage.storeShortTerm(clarify);
            lastAsked = cmd;

            LOG_DEBUG("Dispatch", "Clarifying question issued: " + clarification);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Dispatch", std::string("Clarification prompt failed: ") + e.what());
    }

    return CommandResult{
        false,                              // success
        "Unknown command: " + cmd,          // message
        "ERR_UNKNOWN_CMD",                  // errorCode
        "",                                 // category
        "",                                 // voice
        Colors::Red                         // color
    };
}

// ====================================================
// handleCommand: central command + NLP hub
// ====================================================
void handleCommand(const std::string& line)
{
    LOG_TRACE("HandleCommand", "START line=\"" + line + "\"");

    // Ensure core plugins are registered BEFORE any command lookup
    ensureCorePluginsRegistered();

    // ====================================================
    // Multi-command detection and processing
    // ====================================================
    auto commands = GRIMInput::splitCommands(line);
    if (commands.size() > 1)
    {
        LOG_DEBUG("HandleCommand", "Detected " + std::to_string(commands.size()) + " commands in voice input");
        for (size_t i = 0; i < commands.size(); ++i)
        {
            LOG_DEBUG("HandleCommand", "  [" + std::to_string(i + 1) + "] \"" + commands[i] + "\"");
        }
        
        // Safety check: limit max commands per input
        int maxCommands = aiConfig["voice"].value("max_commands_per_input", 3);
        if (commands.size() > static_cast<size_t>(maxCommands)) {
            std::string warning = "Detected " + std::to_string(commands.size()) + 
                                 " commands, but maximum is " + std::to_string(maxCommands) + 
                                 ". Only processing first " + std::to_string(maxCommands) + ".";
            LOG_DEBUG("HandleCommand", warning);
            history.push("[Warning] " + warning, Colors::Yellow.toUInt());
            commands.resize(maxCommands);
        }
        
        // Set multi-command context to suppress feedback during batch processing
        bool wasMultiContext = g_isMultiCommandContext;
        g_isMultiCommandContext = true;
        
        // Process each command sequentially
        std::vector<CommandResult> results;
        for (const auto& cmd : commands)
        {
            handleCommand(cmd); // Recursive call for each sub-command
        }
        
        // Restore context flag
        g_isMultiCommandContext = wasMultiContext;
        
        // Only ask for feedback once after ALL commands complete
        if (!wasMultiContext && !g_pendingFeedbackCmd.has_value())
        {
            std::string ask = "I processed " + std::to_string(commands.size()) + " commands. Was that correct?";
            history.push(ask, Colors::Cyan.toUInt());
            Voice::speak(ask, "feedback");
            g_pendingFeedbackCmd = line; // Store the full multi-command line
            LOG_DEBUG("Feedback", "Opened feedback prompt for multi-command batch");
        }
        
        return;
    }

    auto [cmdRaw, arg] = GRIMInput::parseInput(line);
    LOG_TRACE("HandleCommand", "Parsed → cmdRaw=\"" + cmdRaw + "\" arg=\"" + arg + "\"");

    // === Clarification handler ===
    if (g_pendingClarifyCmd.has_value())
    {
        std::string originalInput = g_pendingClarifyCmd.value();
        std::string userResponse = line;

        LOG_DEBUG("LearnedCmd", "Processing clarification - original: \"" + originalInput + "\", response: \"" + userResponse + "\"");

        // ✅ Check if this was a prediction (suggested command)
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
                // User is responding to a prediction - check if they confirmed
                std::string answer = GRIMInput::normalizeCommand(userResponse);
                answer.erase(std::remove_if(answer.begin(), answer.end(), 
                    [](char c) { return std::isspace(static_cast<unsigned char>(c)) || 
                                       std::ispunct(static_cast<unsigned char>(c)); }), 
                    answer.end());
                std::transform(answer.begin(), answer.end(), answer.begin(), 
                    [](char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); });
                
                bool confirmed = (answer == "yes" || answer == "y" || answer == "yeah" || 
                                 answer == "correct" || answer == "yep" || answer == "yup");
                bool denied = (answer == "no" || answer == "n" || answer == "nope" || 
                              answer == "wrong" || answer == "nah");
                
                if (confirmed) {
                    // ✅ User confirmed the prediction - execute it and give high reward
                    LOG_DEBUG("Prediction", "User confirmed prediction: \"" + prediction->normalized + "\"");
                    
                    // Store as learned command with high confidence
                    g_memoryStorage.storeLearnedCommand(originalInput, prediction->normalized, 0.95f);
                    
                    // ✅ Record successful prediction for RL
                    try {
                        nlohmann::json reward = {
                            {"type", "successful_prediction"},
                            {"original_input", originalInput},
                            {"predicted_command", prediction->normalized},
                            {"confidence", prediction->confidence},
                            {"user_confirmed", true},
                            {"reward_multiplier", 1.5}  // Higher reward for successful prediction
                        };
                        GRIM::RL::getAction(reward);
                        LOG_DEBUG("RL", "Successful prediction reward sent");
                    } catch (const std::exception& e) {
                        LOG_ERROR("RL", std::string("Failed to send prediction reward: ") + e.what());
                    }
                    
                    std::string resp = "Great! Executing " + prediction->normalized;
                    history.push(resp, Colors::Green.toUInt());
                    Voice::speak(resp, "learned");
                    
                    g_pendingClarifyCmd.reset();
                    
                    // Execute the predicted command
                    return handleCommand(prediction->normalized);
                }
                else if (denied) {
                    // User denied the prediction - ask what they meant
                    LOG_DEBUG("Prediction", "User denied prediction");
                    
                    // Record failed prediction for RL
                    try {
                        nlohmann::json penalty = {
                            {"type", "failed_prediction"},
                            {"original_input", originalInput},
                            {"predicted_command", prediction->normalized},
                            {"confidence", prediction->confidence},
                            {"user_confirmed", false},
                            {"reward_multiplier", -0.5}  // Penalty for wrong prediction
                        };
                        GRIM::RL::getAction(penalty);
                        LOG_DEBUG("RL", "Failed prediction penalty sent");
                    } catch (const std::exception& e) {
                        LOG_ERROR("RL", std::string("Failed to send prediction penalty: ") + e.what());
                    }
                    
                    std::string resp = "I see. What did you mean?";
                    history.push(resp, Colors::Cyan.toUInt());
                    Voice::speak(resp, "clarify");
                    return;  // Keep clarification open for user to restate
                }
            }
        } catch (const std::exception& e) {
            LOG_ERROR("LearnedCmd", std::string("Failed to check predictions: ") + e.what());
        }

        // ✅ Standard clarification (user providing the correct command)
        try {
            g_memoryStorage.storeLearnedCommand(originalInput, userResponse, 0.9f);
            g_pendingClarifyCmd.reset();

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
        } catch (const std::exception& e) {
            LOG_ERROR("LearnedCmd", std::string("Failed to store learned command: ") + e.what());
            std::string resp = "I couldn't save that one — try again later.";
            history.push(resp, Colors::Red.toUInt());
            Voice::speak(resp, "error");
        }

        return;
    }

    // === Feedback handler ===
    if (g_pendingFeedbackCmd.has_value())
    {
        std::string originalCmd = g_pendingFeedbackCmd.value();
        std::string userResponse = line;
        
        LOG_DEBUG("Feedback", "Processing feedback for command: \"" + originalCmd + "\"");
        LOG_DEBUG("Feedback", "User response: \"" + userResponse + "\"");
        
        // ========================================================================
        // STEP 1: Check if user is simply confirming (yes/no)
        // ========================================================================
        std::string answer = GRIMInput::normalizeCommand(userResponse);
        answer.erase(std::remove_if(answer.begin(), answer.end(), 
            [](char c) { return std::isspace(static_cast<unsigned char>(c)) || 
                               std::ispunct(static_cast<unsigned char>(c)); }), 
            answer.end());
        std::transform(answer.begin(), answer.end(), answer.begin(), 
            [](char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); });
        
        bool isPositive = (answer == "yes" || answer == "y" || answer == "yeah" || 
                          answer == "correct" || answer == "yep" || answer == "yup" || answer == "right");
        bool isNegative = (answer == "no" || answer == "n" || answer == "nope" || 
                          answer == "wrong" || answer == "nah" || answer == "incorrect");

        if (isPositive || isNegative) {
            // Simple yes/no confirmation
            try {
                nlohmann::json fb = {
                    {"type", "explicit_feedback"},
                    {"command", originalCmd},
                    {"feedback", isPositive ? "positive" : "negative"},
                    {"user_text", userResponse},
                    {"confidence", 1.0}  // User explicitly confirmed
                };
                GRIM::RL::getAction(fb);
                LOG_DEBUG("Feedback", "Explicit feedback recorded: " + std::string(isPositive ? "positive" : "negative"));
                
                // Update context manager with usage
                if (isPositive) {
                    GRIM::ContextManager::recordUsage(originalCmd);
                }
            } catch (const std::exception& e) {
                LOG_ERROR("Feedback", std::string("Error sending feedback to RL: ") + e.what());
            }

            std::string resp = isPositive ? "Got it — I'll keep doing that." : "Understood — I'll adjust next time.";
            history.push(resp, isPositive ? Colors::Green.toUInt() : Colors::Cyan.toUInt());
            Voice::speak(resp, "feedback");
            g_pendingFeedbackCmd.reset();
            return;
        }
        
        // ========================================================================
        // STEP 2: User might be repeating/clarifying the command
        // Check similarity using fuzzy matching
        // ========================================================================
        std::string normalizedResponse = GRIMInput::normalizeCommand(userResponse);
        std::string normalizedOriginal = GRIMInput::normalizeCommand(originalCmd);
        
        // Calculate Levenshtein distance for similarity
        auto levenshtein = [](const std::string& s1, const std::string& s2) -> int {
            const size_t m = s1.size(), n = s2.size();
            std::vector<int> prev(n + 1), curr(n + 1);
            for (size_t j = 0; j <= n; ++j) prev[j] = static_cast<int>(j);
            for (size_t i = 1; i <= m; ++i) {
                curr[0] = static_cast<int>(i);
                for (size_t j = 1; j <= n; ++j) {
                    int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
                    curr[j] = std::min({prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost});
                }
                prev.swap(curr);
            }
            return prev[n];
        };
        
        int distance = levenshtein(normalizedResponse, normalizedOriginal);
        float similarity = 1.0f - (static_cast<float>(distance) / std::max(normalizedResponse.length(), normalizedOriginal.length()));
        
        LOG_DEBUG("Feedback", "Similarity score: " + std::to_string(similarity) + " (distance=" + std::to_string(distance) + ")");
        
        // ========================================================================
        // STEP 3: Check memory system for learned patterns
        // ========================================================================
        float memoryConfidence = 0.0f;
        try {
            auto learned = g_memoryStorage.findLearnedCommand(normalizedResponse);
            if (learned.has_value()) {
                memoryConfidence = learned->confidence;
                LOG_DEBUG("Feedback", "Found learned command with confidence: " + std::to_string(memoryConfidence));
                
                // If memory suggests this is a repeat of the same command
                if (learned->normalized == normalizedOriginal) {
                    similarity = std::max(similarity, 0.9f); // Boost similarity
                    LOG_DEBUG("Feedback", "Memory confirms this is the same command");
                }
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Feedback", std::string("Memory lookup failed: ") + e.what());
        }
        
        // ========================================================================
        // STEP 4: Check usage history for confidence
        // ========================================================================
        float contextConfidence = 0.5f;
        try {
            // Check how often this command is used
            std::string mood = GRIM::ContextManager::getCurrentMood();
            // Higher usage = higher confidence user meant this command
            contextConfidence = 0.7f; // Placeholder - you can expand this
        } catch (const std::exception& e) {
            LOG_ERROR("Feedback", std::string("Context check failed: ") + e.what());
        }
        
        // ========================================================================
        // STEP 5: Decision logic based on confidence
        // ========================================================================
        float finalConfidence = (similarity * 0.5f) + (memoryConfidence * 0.3f) + (contextConfidence * 0.2f);
        LOG_DEBUG("Feedback", "Final confidence: " + std::to_string(finalConfidence));
        
        if (similarity > 0.8f || finalConfidence > 0.75f) {
            // High confidence - user is repeating/confirming the command
            LOG_DEBUG("Feedback", "High confidence match - treating as implicit confirmation");
            
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
                
                // Store learned pattern
                g_memoryStorage.storeLearnedCommand(normalizedResponse, originalCmd, finalConfidence);
                GRIM::ContextManager::recordUsage(originalCmd);
            } catch (const std::exception& e) {
                LOG_ERROR("Feedback", std::string("Error recording implicit feedback: ") + e.what());
            }
            
            std::string resp = "Got it — command confirmed.";
            history.push(resp, Colors::Green.toUInt());
            Voice::speak(resp, "feedback");
            g_pendingFeedbackCmd.reset();
            return;
        }
        else if (similarity > 0.5f || finalConfidence > 0.5f) {
            // Medium confidence - ask for clarification
            LOG_DEBUG("Feedback", "Medium confidence - asking for clarification");
            
            std::string clarification = "Did you mean \"" + originalCmd + "\" or \"" + normalizedResponse + "\"?";
            history.push(clarification, Colors::Yellow.toUInt());
            Voice::speak(clarification, "clarify");
            
            // Store this as a potential learned command but don't confirm yet
            g_pendingClarifyCmd = normalizedResponse;
            g_pendingFeedbackCmd.reset();
            return;
        }
        else {
            // Low confidence - treat as a new command
            LOG_DEBUG("Feedback", "Low confidence - treating as new command");
            
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
            
            g_pendingFeedbackCmd.reset();
            // Fall through to execute the new command
        }
    }

    // ====================================================
    // Normal execution flow
    // ====================================================
    history.push("> " + line, Colors::Default.toUInt());
    CommandResult result;

    // --- RL Pre-dispatch observation (Dynamic PPO Integration) ---
    try {
        nlohmann::json obs = {
            {"input", line},
            {"command_raw", cmdRaw},
            {"argument", arg},
            {"context", longTermMemory},
            {"mood", GRIM::ContextManager::getCurrentMood()}
        };

        nlohmann::json rlRes = GRIM::RL::getAction(obs);

        if (rlRes.contains("suggested_command")) {
            std::string suggested = rlRes["suggested_command"].get<std::string>();
            if (commandMap.find(suggested) != commandMap.end()) {
                LOG_DEBUG("RL", "Dynamic PPO suggested: " + suggested);
                cmdRaw = suggested;
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("RL", std::string("Pre-dispatch RL error: ") + e.what());
    }

    // --- Dispatch ---
    if (commandMap.find(cmdRaw) != commandMap.end())
    {
        LOG_DEBUG("HandleCommand", "Found command in map: \"" + cmdRaw + "\"");
        result = dispatchCommand(cmdRaw, arg);
    }
    else
    {
        LOG_DEBUG("HandleCommand", "Command \"" + cmdRaw + "\" NOT in map, trying NLP...");
        
        std::string normalizedLine = GRIMInput::normalizeLine(line);
        Intent intent = g_nlp.parse(normalizedLine);
        g_lastIntent = intent;

        std::string cmd = intent.matched ? intent.name : GRIMInput::normalizeCommand(cmdRaw);

        if (intent.matched && intent.slots.size())
        {
            for (const auto& [k, v] : intent.slots)
                if (!v.empty()) { arg = GRIMInput::cleanArg(v); break; }
        }

        LOG_DEBUG("HandleCommand", "After normalization: cmd=\"" + cmd + "\"");
        if (commandMap.find(cmd) != commandMap.end())
        {
            LOG_DEBUG("HandleCommand", "Found normalized command in map: \"" + cmd + "\"");
            result = dispatchCommand(cmd, arg);
        }
        else
        {
            LOG_DEBUG("HandleCommand", "No match found, sending to AI interpret");
            result = ai_interpret(line, true);
        }
    }

    if (result.message.empty()) {
        result.message = "[no response configured]";
        result.success = false;
        if (result.errorCode.empty()) result.errorCode = "ERR_NONE";
    }

    std::string finalText = ResponseManager::get(result.message);
    Logger::logResult(result);
    history.push(finalText, (result.color.a << 24) | (result.color.b << 16) | (result.color.g << 8) | result.color.r);
    std::cout << finalText << std::endl;

    if (!result.voice.empty() && result.voice.find("[TRACE]") == std::string::npos)
        Voice::speak(result.voice, result.category.empty() ? "routine" : result.category);

    // Only ask for feedback if NOT in multi-command context AND command was successful
    if (!g_isMultiCommandContext && !g_pendingFeedbackCmd.has_value() && result.success && result.category != "conversation")
    {
        std::string ask = "Was that what you wanted?";  // Direct string, no ResponseManager
        history.push(ask, Colors::Cyan.toUInt());
        Voice::speak(ask, "feedback");
        g_pendingFeedbackCmd = cmdRaw;
        LOG_DEBUG("Feedback", "Opened feedback prompt for command: " + cmdRaw);
    }
    else if (!result.success && result.errorCode != "ERR_UNKNOWN_CMD")
    {
        // ✅ On errors, give audible notification that we're returning to wake word detection
        LOG_DEBUG("Feedback", "Skipping feedback for failed command: " + cmdRaw + " (" + result.errorCode + ")");
        
        // ✅ Tell user we're ready for the next command
        Voice::speak("Ready.", "routine");
    }

    try {
        float execTime = 0.0f; // (optional: measure actual duration later)
        float sentimentScore = (result.success ? 0.5f : -0.5f);
        float diversityFactor = 0.2f; // placeholder — can evolve with context
        std::string mood = GRIM::ContextManager::getCurrentMood();

        GRIM::RL::processCommandResult(result, cmdRaw, execTime, sentimentScore, diversityFactor, mood);
    } catch (const std::exception& e) {
        LOG_ERROR("RL", std::string("Post-dispatch RL feedback error: ") + e.what());
    }

    GRIM::ContextManager::recordUsage(cmdRaw);
    GRIM::PersonalityManager::updateAfterCommand(result.success);
    GRIM::DialogueProactive::checkAfterCommand(line, result);

    LOG_TRACE("HandleCommand", "END");
}
