#include "commands_core.hpp"
#include "commands_execution.hpp"
#include "commands_feedback.hpp"
#include "commands_ai.hpp"
#include "commands_question.hpp"
#include "../MMO/Core/ToolRegistry.hpp"
#include "../MMO/Core/SessionContextManager.hpp"
#include "response_manager.hpp"
#include "console_history.hpp"
#include "voice/voice_speak.hpp"
#include "nlp/nlp.hpp"
#include "input_parser.hpp"
#include "logger.hpp"
#include "error_manager.hpp"
#include "ai/ai.hpp"
#include "ai/intent_gate.hpp"
#include "ai/fast_classifier.hpp"
#include "memory/unified_memory.hpp"
#include "Reward_Learning/grim_rl.hpp"
#include "ai/personality_manager.hpp"
#include "ai/proactive_dialogue.hpp"
#include "core/plugin.hpp"
#include "helpers/color.hpp"
#include <crtdbg.h>
#include "helpers/grim_input.hpp"
#include <algorithm>

#define CHECK_HEAP() _CrtCheckMemory()

using Voice::speak;

// ====================================================
// External access functions
// ====================================================
static const std::string kDefaultSession = "default";

bool hasPendingFeedback() {
    return GRIM::MMO::SessionContextManager::instance().getPending(kDefaultSession).has_value();
}

void setVoiceCommand(bool isVoice) {
    GRIM::MMO::SessionContextManager::instance().setVoiceCommand(kDefaultSession, isVoice);
}

extern nlohmann::json longTermMemory;
extern nlohmann::json aiConfig;
extern NLP g_nlp;
#define history getConsoleHistory()

Intent g_lastIntent;

// ====================================================
// Core Plugin Loader
// ====================================================

// Forward declaration (defined in core_plugin.cpp)
extern void registerCorePlugin();

void ensureCorePluginsRegistered()
{
    static bool done = false;
    if (done) return;
    done = true;

    LOG_DEBUG("Core", "Ensuring core plugins registered...");
    registerCorePlugin();  // ✅ Call the actual registration function
}

// ====================================================
// Core Dispatch - Simplified to route commands
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
            CommandResult result = it->second(arg);
            
            // Record command usage analytics
            if (result.success) {
                GRIM::MMO::ToolRegistry::instance().recordSuccess(cmd);
            } else {
                GRIM::MMO::ToolRegistry::instance().recordFailure(cmd);
            }
            
            return result;
        } catch (const std::exception& e) {
            LOG_ERROR("Dispatch", "Exception in command \"" + cmd + "\": " + e.what());
            GRIM::MMO::ToolRegistry::instance().recordFailure(cmd);
            return CommandResult{
                false,
                "[Error] Exception while running command: " + cmd,
                "ERR_CMD_EXCEPTION",
                "",
                "",
                Colors::Red
            };
        }
    }

    // 2. Try learned commands from memory
    CommandResult learnedResult = GRIM::CommandExecution::tryLearnedCommand(cmd, arg);
    if (learnedResult.success || learnedResult.errorCode != "ERR_NOT_FOUND") {
        return learnedResult;
    }

    // 3. Record unknown command for analysis
    GRIM::CommandExecution::recordUnknownCommand(cmd, arg);

    // 4. Try RL inference
    CommandResult rlResult = GRIM::CommandExecution::tryRLInference(cmd, arg);
    if (rlResult.success || rlResult.errorCode != "ERR_NOT_FOUND") {
        return rlResult;
    }

    // 5. Try AI interpretation with similarity checking
    try {
        LOG_TRACE("Dispatch", "Calling ai_interpret() for unknown command...");
        
        // Check if we have similar learned commands
        auto suggestions = GRIM::CommandExecution::findSimilarLearnedCommands(cmd + " " + arg, 0.6f);
        
        // High-confidence suggestion - ask user instead of calling AI
        if (!suggestions.empty() && suggestions[0].second > 0.75f) {
            std::string suggestedCmd = suggestions[0].first;
            float confidence = suggestions[0].second;
            
            LOG_DEBUG("Dispatch", "High-confidence suggestion found: \"" + suggestedCmd + 
                     "\" (score=" + std::to_string(confidence) + ")");
            
            std::string question = "Did you mean \"" + suggestedCmd + "\"?";
            auto& scm = GRIM::MMO::SessionContextManager::instance();
            scm.setPending(kDefaultSession, {
                GRIM::MMO::PendingKind::Clarification,
                cmd + (arg.empty() ? "" : " " + arg),
                "TypeTag:App", question, "", {}, 120
            });
            history.push(question, Colors::Yellow.toUInt());
            Voice::speak(question, "clarify");
            
            return CommandResult{
                true,
                "Suggestion: " + suggestedCmd,
                "ERR_NONE",
                "clarify",
                "",
                Colors::Yellow
            };
        }
        
        // Try AI interpretation
        CommandResult aiRes = ai_interpret(cmd + " " + arg, true);

        if (aiRes.success && aiRes.category == "command_infer")
        {
            std::string inferred = GRIMInput::normalizeCommand(aiRes.message);
            LOG_DEBUG("AI", "AI inferred new command: " + inferred);

            GRIM::CommandExecution::storeLearnedCommand(cmd, inferred, 0.75f);
            CHECK_HEAP();
            
            std::string resp = ResponseManager::get(
                "Got it — I've learned that \"" + cmd + "\" means \"" + inferred + "\".");
            history.push(resp, Colors::Green.toUInt());
            Voice::speak(resp, "learned");

            return CommandResult{
                true,
                "Learned new command: " + inferred,
                "ERR_NONE",
                "",
                "",
                Colors::Green
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
            std::string clarification = GRIM::PersonalityManager::isStable()
                ? "Did you mean \"" + GRIMInput::normalizeCommand(cmd) + "\" or something else?"
                : "I didn't quite catch that. Can you rephrase?";

            std::string resp = ResponseManager::get(clarification);
            history.push(resp, Colors::Cyan.toUInt());
            Voice::speak(resp, "clarify");
            
            lastAsked = cmd;
            LOG_DEBUG("Dispatch", "Clarifying question issued: " + clarification);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Dispatch", std::string("Clarification prompt failed: ") + e.what());
    }

    return CommandResult{
        false,
        "Unknown command: " + cmd,
        "ERR_UNKNOWN_CMD",
        "",
        "",
        Colors::Red
    };
}

// ====================================================
// handleCommand: central command hub - delegates to subsystems
// ====================================================
void handleCommand(const std::string& line)
{
    LOG_TRACE("HandleCommand", "START line=\"" + line + "\"");

    ensureCorePluginsRegistered();

    // ====================================================
    // ✅ NEW: Intent Classification - Route to command or banter pipeline
    // ====================================================
    try {
        auto& scm = GRIM::MMO::SessionContextManager::instance();
        GRIM::ContextSnapshot ctx = scm.legacySnapshot(kDefaultSession);
        GRIM::IntentResult intentResult = GRIM::IntentGate::decide(line, ctx);
        
        LOG_DEBUG("HandleCommand", "Intent classified as: " + GRIM::intentTypeToString(intentResult.type) + 
                 " (confidence: " + std::to_string(intentResult.confidence) + ")");
        
        // Route based on intent
        if (intentResult.type == GRIM::IntentType::Banter) {
            // This is casual conversation - route to AI chat
            LOG_DEBUG("HandleCommand", "Routing to banter pipeline");
            
            history.push("> " + line, Colors::Default.toUInt());
            CommandResult result = ai_process(line); // AI handles casual conversation
            
            std::string finalText = ResponseManager::get(result.message);
            history.push(finalText, (result.color.a << 24) | (result.color.b << 16) | 
                        (result.color.g << 8) | result.color.r);
            
            if (!result.voice.empty()) {
                Voice::speak(result.voice, "banter");
            }
            
            LOG_TRACE("HandleCommand", "END (banter route)");
            return; // Exit early - don't process as command
        }
        
        // ✅ NEW: Handle questions through memory and external knowledge
        if (intentResult.type == GRIM::IntentType::Question) {
            LOG_DEBUG("HandleCommand", "Routing to question pipeline");
            
            history.push("> " + line, Colors::Default.toUInt());
            CommandResult result = cmdQuestion(line); // Question handler searches memories and external sources
            
            std::string finalText = ResponseManager::get(result.message);
            history.push(finalText, (result.color.a << 24) | (result.color.b << 16) | 
                        (result.color.g << 8) | result.color.r);
            
            if (!result.voice.empty()) {
                Voice::speak(result.voice, "question");
            }
            
            LOG_TRACE("HandleCommand", "END (question route)");
            return; // Exit early - question handled
        }
        
        // If classified as Command, clean up banter words before parsing
        if (intentResult.type == GRIM::IntentType::Command) {
            // Strip common banter prefixes/suffixes for cleaner parsing
            std::string cleaned = line;
            std::string lowerCleaned = line;
            
            // Convert to lowercase for comparison
            std::transform(lowerCleaned.begin(), lowerCleaned.end(), lowerCleaned.begin(), ::tolower);
            
            // Remove common polite prefixes (case-insensitive)
            const std::vector<std::string> banterPrefixes = {
                "hey ", "hi ", "hello ", "yo ", "sup ",
                "hey grim ", "hi grim ", "grim ",
                "please ", "could you ", "can you ", "would you ",
                " can you ", " could you ", " would you ",  // ✅ FIX: Handle leading spaces
                " please "
            };
            
            for (const auto& prefix : banterPrefixes) {
                if (lowerCleaned.find(prefix) == 0) {
                    // ✅ FIX: Use lowerCleaned length to avoid case mismatch issues
                    cleaned = line.substr(prefix.length());
                    
                    // ✅ Trim any remaining leading/trailing spaces
                    cleaned.erase(0, cleaned.find_first_not_of(" \t\n\r"));
                    cleaned.erase(cleaned.find_last_not_of(" \t\n\r") + 1);
                    
                    LOG_DEBUG("HandleCommand", "Stripped banter prefix '" + prefix + "', cleaned: \"" + cleaned + "\"");
                    break; // Only strip first match
                }
            }
            
            // If we stripped banter words, use the cleaned version
            if (cleaned != line && !cleaned.empty()) {
                LOG_DEBUG("HandleCommand", "Using cleaned command: \"" + cleaned + "\"");
                // Recursively call with cleaned input
                handleCommand(cleaned);
                return;
            }
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("HandleCommand", std::string("Intent classification failed: ") + e.what());
        // Fall through to normal command processing on error
    }

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
        auto& scm = GRIM::MMO::SessionContextManager::instance();
        bool wasMultiContext = scm.isMultiCommandContext(kDefaultSession);
        scm.setMultiCommandContext(kDefaultSession, true);
        
        // Process each command sequentially
        for (const auto& cmd : commands)
        {
            handleCommand(cmd); // Recursive call for each sub-command
        }
        
        // Restore context flag
        scm.setMultiCommandContext(kDefaultSession, wasMultiContext);
        
        // Only ask for feedback once after ALL commands complete
        if (!wasMultiContext && !scm.getPending(kDefaultSession).has_value())
        {
            std::string ask = "I processed " + std::to_string(commands.size()) + " commands. Was that correct?";
            history.push(ask, Colors::Cyan.toUInt());
            Voice::speak(ask, "feedback");
            scm.setPending(kDefaultSession, {
                GRIM::MMO::PendingKind::Confirmation,
                line, "", ask, "", {}, 120
            });
            LOG_DEBUG("Feedback", "Opened feedback prompt for multi-command batch");
        }
        
        return;
    }

    auto [cmdRaw, arg] = GRIMInput::parseInput(line);
    // [GRIM CONTEXT] Attempt to resolve contextual references (like "that app")
    if (arg == "that app" || arg == "the app" || arg == "it") {
        auto& scm = GRIM::MMO::SessionContextManager::instance();
        auto ref = scm.resolveReference(kDefaultSession, arg);
        if (ref.has_value()) {
            arg = ref->value;
            LOG_DEBUG("Context", "Resolved pronoun → " + arg);
        } else {
            // Fallback to context memory recall
            auto ctx = scm.recallContextByType(kDefaultSession, "App");
            if (ctx.has_value()) {
                arg = ctx->raw;
                LOG_DEBUG("Context", "Resolved pronoun via context recall → " + arg);
            } else {
                std::string ask = "Which app did you mean?";
                history.push(ask, Colors::Cyan.toUInt());
                Voice::speak(ask, "clarify");

                scm.setPending(kDefaultSession, {
                    GRIM::MMO::PendingKind::MissingSlot,
                    "open_app", "TypeTag:App", ask, "", {}, 120
                });
                return;
            }
        }
    }

    LOG_TRACE("HandleCommand", "Parsed → cmdRaw=\"" + cmdRaw + "\" arg=\"" + arg + "\"");

    // === Pending interaction handler (clarification, feedback, missing slot) ===
    {
        auto& scm = GRIM::MMO::SessionContextManager::instance();
        auto pending = scm.getPending(kDefaultSession);
        if (pending.has_value()) {
            if (pending->kind == GRIM::MMO::PendingKind::Clarification) {
                if (GRIM::Feedback::processClarificationResponse(pending->original_command, line)) {
                    scm.clearPending(kDefaultSession);
                    return;
                }
            }
            else if (pending->kind == GRIM::MMO::PendingKind::MissingSlot) {
                LOG_DEBUG("Context", "Pending missing-slot active → " + pending->original_command);
                scm.clearPending(kDefaultSession);
                handleCommand(pending->original_command + " " + line);
                return;
            }
            else if (pending->kind == GRIM::MMO::PendingKind::Confirmation ||
                     pending->kind == GRIM::MMO::PendingKind::FollowUp) {
                std::string originalCmd = pending->original_command;
                if (GRIM::Feedback::processFeedbackResponse(originalCmd, line)) {
                    scm.clearPending(kDefaultSession);
                    return;
                }
                scm.clearPending(kDefaultSession);
            }
        }
    }


    // ====================================================
    // Normal execution flow
    // ====================================================
    history.push("> " + line, Colors::Default.toUInt());
    CommandResult result;

    if (auto suggestion = GRIM::RewardLearning::suggestPreDispatchCommand(
            line,
            cmdRaw,
            arg,
            longTermMemory,
            GRIM::MMO::SessionContextManager::instance().legacySnapshot(kDefaultSession).currentMood))
    {
        cmdRaw = *suggestion;
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
        
        // ✅ INTEGRATION #3: Feedback loop - teach classifier from NLP failures
        if (!intent.matched) {
            LOG_DEBUG("Feedback", "NLP failed to match - analyzing for classifier update");
            
            // Tokenize input to check for command indicators
            std::string lowerLine = line;
            std::transform(lowerLine.begin(), lowerLine.end(), lowerLine.begin(), ::tolower);
            
            bool hasCommandWords = false;
            const std::vector<std::string> commandVerbs = {
                "open", "close", "run", "launch", "show", "list", "set", 
                "create", "delete", "search", "find", "play", "stop",
                "nevermind", "cancel", "stop", "forget", "undo", "clear"  // ✅ ADD: cancellation verbs
            };
            
            for (const auto& verb : commandVerbs) {
                if (lowerLine.find(verb) != std::string::npos) {
                    hasCommandWords = true;
                    break;
                }
            }
            
            // ✅ FIX: Don't teach as banter if NLP will match it later
            // Check if command is in the command map (even if NLP didn't match)
            bool isKnownCommand = (commandMap.find(cmd) != commandMap.end());
            
            if (!hasCommandWords && !isKnownCommand) {
                // Likely banter that slipped through - teach classifier
                GRIM::FastClassifier::updateWeights(line, GRIM::IntentType::Banter);
                LOG_DEBUG("Feedback", "Taught classifier: \"" + line + "\" → Banter (no command verbs)");
            } else if (isKnownCommand) {
                // Known command - teach as command
                GRIM::FastClassifier::updateWeights(line, GRIM::IntentType::Command);
                LOG_DEBUG("Feedback", "Taught classifier: \"" + line + "\" → Command (found in command map)");
            } else {
                // Has command words but NLP didn't match - might be new pattern
                LOG_DEBUG("Feedback", "\"" + line + "\" has command words but no NLP match - potential learning opportunity");
            }
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

    // Ensure result has a message
    if (result.message.empty()) {
        result.message = "[no response configured]";
        result.success = false;
        if (result.errorCode.empty()) result.errorCode = "ERR_NONE";
    }

    // Output result
    std::string finalText = ResponseManager::get(result.message);
    Logger::logResult(result);
    history.push(finalText, (result.color.a << 24) | (result.color.b << 16) | (result.color.g << 8) | result.color.r);
    // [GRIM CONTEXT] Record successful command context
    if (result.success) {
        GRIM::UnifiedMemoryObject contextObj;
        contextObj.id = GRIM::UnifiedMemoryObject::generateID();
        contextObj.timestamp = static_cast<uint64_t>(std::time(nullptr));
        contextObj.source = GRIM::SourceType::GRIM_INTERNAL;
        contextObj.type = GRIM::TypeTag::COMMAND;
        contextObj.intent = GRIM::MemoryIntent::INFORM;
        contextObj.context = GRIM::ContextType::CONVERSATION;
        contextObj.raw = cmdRaw;
        contextObj.normalized = GRIMInput::normalizeCommand(cmdRaw);
        contextObj.confidence = 1.0f;
        contextObj.tags = {"context_command", "session"};

        GRIM::MMO::SessionContextManager::instance().rememberContextObject(kDefaultSession, contextObj);
    }



    std::cout << finalText << std::endl;

    if (!result.voice.empty() && result.voice.find("[TRACE]") == std::string::npos)
        Voice::speak(result.voice, result.category.empty() ? "routine" : result.category);

    // Request feedback for successful voice commands
    {
        auto& scm = GRIM::MMO::SessionContextManager::instance();
        if (!scm.isMultiCommandContext(kDefaultSession) && !scm.getPending(kDefaultSession).has_value() && 
            result.success && result.category != "conversation" && result.category != "cancellation" && 
            result.category != "banter" && scm.isVoiceCommand(kDefaultSession))
        {
            std::string ask = "Was that what you wanted?";
            history.push(ask, Colors::Cyan.toUInt());
            Voice::speak(ask, "feedback");
            scm.setPending(kDefaultSession, {
                GRIM::MMO::PendingKind::Confirmation,
                cmdRaw, "", ask, "", {}, 120
            });
            LOG_DEBUG("Feedback", "Opened feedback prompt for command: " + cmdRaw);
        }
        else if (!result.success && result.errorCode != "ERR_UNKNOWN_CMD" && scm.isVoiceCommand(kDefaultSession))
        {
            LOG_DEBUG("Feedback", "Skipping feedback for failed command: " + cmdRaw + " (" + result.errorCode + ")");
            
            if (result.errorCode == "ERR_APP_NO_ARGUMENT" || 
                result.errorCode == "ERR_APP_LAUNCH_FAILED" ||
                result.errorCode == "ERR_FS_MISSING_DIR" ||
                result.errorCode == "ERR_FS_DIR_NOT_FOUND") {
                Voice::speak("Ready.", "routine");
            }
        }
        
        // Reset voice flag for next command
        scm.setVoiceCommand(kDefaultSession, false);
    }

    // Post-command RL feedback
    GRIM::RewardLearning::sendCommandFeedback(
        result,
        cmdRaw,
        0.0f,
        (result.success ? 0.5f : -0.5f),
        0.2f,
        GRIM::MMO::SessionContextManager::instance().legacySnapshot(kDefaultSession).currentMood);

    // Update context and personality
    GRIM::MMO::SessionContextManager::instance().recordUsage(kDefaultSession, cmdRaw);
    GRIM::PersonalityManager::updateAfterCommand(result.success);
    GRIM::DialogueProactive::checkAfterCommand(line, result);
    GRIM::MMO::SessionContextManager::instance().decayOldContext(kDefaultSession, 60);
    LOG_TRACE("HandleCommand", "END");
}
