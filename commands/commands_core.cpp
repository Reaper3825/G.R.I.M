#include "commands_core.hpp"
#include "commands_execution.hpp"
#include "commands_feedback.hpp"
#include "commands_ai.hpp"
#include "response_manager.hpp"
#include "console_history.hpp"
#include "voice/voice_speak.hpp"
#include "nlp/nlp.hpp"
#include "input_parser.hpp"
#include "logger.hpp"
#include "error_manager.hpp"
#include "ai/ai.hpp"
#include "ai/intent_gate.hpp"
#include "ai/fast_classifier.hpp"  // ✅ NEW: For updateWeights()
#include "memory/context_manager.hpp"
#include "memory/memory_manager.hpp"
#include "ai/ai_rl.hpp"
#include "ai/personality_manager.hpp"
#include "ai/proactive_dialogue.hpp"
#include "core/plugin.hpp"
#include "helpers/color.hpp"
#include <crtdbg.h>
#include "helpers/grim_input.hpp"
#include <algorithm>  // ✅ NEW: For std::transform

#define CHECK_HEAP() _CrtCheckMemory()

using Voice::speak;

// ====================================================
// External access functions
// ====================================================
bool hasPendingFeedback() {
    return GRIM::Feedback::hasPending();
}

void setVoiceCommand(bool isVoice) {
    GRIM::Feedback::setVoiceCommand(isVoice);
}

extern nlohmann::json longTermMemory;
extern nlohmann::json aiConfig;
extern NLP g_nlp;
#define history getConsoleHistory()

Intent g_lastIntent;

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
            return it->second(arg);
        } catch (const std::exception& e) {
            LOG_ERROR("Dispatch", "Exception in command \"" + cmd + "\": " + e.what());
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
            GRIM::ContextManager::setPendingIntent({ "open_app", "TypeTag:App", std::chrono::steady_clock::now() });
            history.push(question, Colors::Yellow.toUInt());
            Voice::speak(question, "clarify");
            
            GRIM::Feedback::setPendingClarification(cmd + (arg.empty() ? "" : " " + arg));
            
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
        GRIM::ContextSnapshot ctx = GRIM::ContextManager::getSnapshot();
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
        bool wasMultiContext = GRIM::Feedback::isMultiCommandContext();
        GRIM::Feedback::setMultiCommandContext(true);
        
        // Process each command sequentially
        for (const auto& cmd : commands)
        {
            handleCommand(cmd); // Recursive call for each sub-command
        }
        
        // Restore context flag
        GRIM::Feedback::setMultiCommandContext(wasMultiContext);
        
        // Only ask for feedback once after ALL commands complete
        if (!wasMultiContext && !GRIM::Feedback::hasPending())
        {
            std::string ask = "I processed " + std::to_string(commands.size()) + " commands. Was that correct?";
            history.push(ask, Colors::Cyan.toUInt());
            Voice::speak(ask, "feedback");
            GRIM::Feedback::setPending(line); // Store the full multi-command line
            LOG_DEBUG("Feedback", "Opened feedback prompt for multi-command batch");
        }
        
        return;
    }

    auto [cmdRaw, arg] = GRIMInput::parseInput(line);
    // [GRIM CONTEXT] Attempt to resolve contextual references (like "that app")
    if (arg == "that app" || arg == "the app" || arg == "it") {
        auto ctx = GRIM::ContextManager::recallContextByType("App");
        if (ctx.has_value()) {
            arg = ctx->raw;
            LOG_DEBUG("Context", "Resolved pronoun → " + arg);
        } else {
            std::string ask = "Which app did you mean?";
            history.push(ask, Colors::Cyan.toUInt());
            Voice::speak(ask, "clarify");

            GRIM::ContextManager::setPendingIntent({ "open_app", "TypeTag:App", std::chrono::steady_clock::now() });
            return;
        }
    }

    LOG_TRACE("HandleCommand", "Parsed → cmdRaw=\"" + cmdRaw + "\" arg=\"" + arg + "\"");

    // === Clarification handler ===
    if (GRIM::Feedback::hasPendingClarification())
    {
        std::string originalInput = GRIM::Feedback::getPendingClarification().value();
        if (GRIM::Feedback::processClarificationResponse(originalInput, line)) {
            return; // Clarification was handled
        }
    }
    // [GRIM CONTEXT] If user responded to a pending intent, resume the command
    auto pending = GRIM::ContextManager::getPendingIntent();
    if (pending.has_value()) {
        LOG_DEBUG("Context", "Pending intent active → " + pending->command);
        GRIM::ContextManager::clearPendingIntent();

        // Treat this response as argument to the original command
        handleCommand(pending->command + " " + line);
        return;
    }


    // === Feedback handler ===
    if (GRIM::Feedback::hasPending())
    {
        std::string originalCmd = GRIM::Feedback::getPending().value();
        if (GRIM::Feedback::processFeedbackResponse(originalCmd, line)) {
            return; // Feedback was handled, don't process as new command
        }
        // If returned false, continue processing as new command
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
    GRIM::MemoryObject contextObj;
    contextObj.id = GRIM::MemoryObject::generateUUID();
    contextObj.timestamp = std::time(nullptr);
    contextObj.source = GRIM::SourceTag::GrimInternal;      // or UserText if command came from user
    contextObj.type = GRIM::TypeTag::Command;               // identifies it as a command
    contextObj.intent = GRIM::IntentTag::Inform;          // or Query depending on use
    contextObj.context = GRIM::ContextTag::Conversation;  // most similar to “Session”
    contextObj.raw = cmdRaw;                                // actual command string
    contextObj.normalized = GRIMInput::normalizeCommand(cmdRaw);
    contextObj.confidence = 1.0f;
    contextObj.tags = {"context_command", "session"};

    GRIM::ContextManager::rememberContextObject(contextObj);
}



    std::cout << finalText << std::endl;

    if (!result.voice.empty() && result.voice.find("[TRACE]") == std::string::npos)
        Voice::speak(result.voice, result.category.empty() ? "routine" : result.category);

    // Request feedback for successful voice commands
    if (!GRIM::Feedback::isMultiCommandContext() && !GRIM::Feedback::hasPending() && 
        result.success && result.category != "conversation" && result.category != "cancellation" && 
        result.category != "banter" && GRIM::Feedback::isVoiceCommand())
    {
        std::string ask = "Was that what you wanted?";
        history.push(ask, Colors::Cyan.toUInt());
        Voice::speak(ask, "feedback");
        GRIM::Feedback::setPending(cmdRaw);
        LOG_DEBUG("Feedback", "Opened feedback prompt for command: " + cmdRaw);
    }
    else if (!result.success && result.errorCode != "ERR_UNKNOWN_CMD" && GRIM::Feedback::isVoiceCommand())
    {
        LOG_DEBUG("Feedback", "Skipping feedback for failed command: " + cmdRaw + " (" + result.errorCode + ")");
        
        // Only give "Ready" notification for actual application/system errors
        if (result.errorCode == "ERR_APP_NO_ARGUMENT" || 
            result.errorCode == "ERR_APP_LAUNCH_FAILED" ||
            result.errorCode == "ERR_FS_MISSING_DIR" ||
            result.errorCode == "ERR_FS_DIR_NOT_FOUND") {
            Voice::speak("Ready.", "routine");
        }
    }
    
    // Reset voice flag for next command
    GRIM::Feedback::setVoiceCommand(false);

    // Post-command RL feedback
    try {
        float execTime = 0.0f;
        float sentimentScore = (result.success ? 0.5f : -0.5f);
        float diversityFactor = 0.2f;
        std::string mood = GRIM::ContextManager::getCurrentMood();

        GRIM::RL::processCommandResult(result, cmdRaw, execTime, sentimentScore, diversityFactor, mood);
    } catch (const std::exception& e) {
        LOG_ERROR("RL", std::string("Post-dispatch RL feedback error: ") + e.what());
    }

    // Update context and personality
    GRIM::ContextManager::recordUsage(cmdRaw);
    GRIM::PersonalityManager::updateAfterCommand(result.success);
    GRIM::DialogueProactive::checkAfterCommand(line, result);
    GRIM::ContextManager::decayOldContext(60); // decay context older than 60s
    LOG_TRACE("HandleCommand", "END");
}
