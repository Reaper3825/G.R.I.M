#include "ai.hpp"
#include "voice/voice.hpp"
#include "voice/voice_speak.hpp"  // ✅ For Voice::speak()
#include "bootstrap/bootstrap.hpp"  // MMO orchestrator globals
#include "../MMO/Core/SessionContextManager.hpp"
#include "../MMO/Core/ToolRegistry.hpp"
#include "../MMO/Core/ActionPolicyRegistry.hpp"
#include "../MMO/Core/ModelRegistry.hpp"
#include "../MMO/Core/ToolTrainingParser.hpp"
#include "nlp/NlpAnnotation.hpp"
#include "nlp/RouterMetadataBuilder.hpp"
#include "memory/context_snapshot.hpp"
#include "memory/atomic_writer.hpp"
#include "resources.hpp"
#include "commands/commands_core.hpp"
#include "error_manager.hpp"
#include "logger.hpp"
#include "personality_manager.hpp"
#include "action_executor.hpp"  // ✅ Intelligent action execution
#include "task_planner.hpp"     // ✅ Multi-step task planning
#include "location.hpp"  // ✅ For location context
#include "helpers/grim_input.hpp"  // ✅ For parseInput
#include "grim_backend.hpp"  // ✅ Native GRIM model backend (external reference)
#include "../MMO/Backends/OllamaBackend.hpp"
#include <cpr/cpr.h>
#include <fstream>
#include <future>
#include <algorithm>

// ---------------- Globals ----------------
double g_silenceThreshold = 1e-6; // default, overridden in aiConfig
int g_silenceTimeoutMs    = 7000; // default 7 seconds
std::string g_whisperLanguage = "en";
int g_whisperMaxTokens        = 32;

// ====================================================
// Helpers: ensure voice section exists in memory
// ====================================================
nlohmann::json& voiceMemory() {
    if (!longTermMemory.contains("voice") || !longTermMemory["voice"].is_object()) {
        longTermMemory["voice"] = {
            {"corrections", nlohmann::json::object()},
            {"shortcuts", nlohmann::json::object()},
            {"usage_counts", nlohmann::json::object()},
            {"last_command", ""}
        };
    }
    return longTermMemory["voice"];
}

// =========================================================
// Memory persistence
// =========================================================
void saveMemory() {
    try {
        GRIM::AtomicWriter::writeString("memory.json", longTermMemory.dump(2));
        LOG_PHASE("Memory saved", true);
    } catch (const std::exception& e) {
        LOG_ERROR("Memory", std::string("Failed to save memory.json: ") + e.what());
        LOG_PHASE("Memory save", false);
    }
}

void loadMemory() {
    std::ifstream f("memory.json");
    if (f) {
        try {
            f >> longTermMemory;
            LOG_PHASE("Memory loaded", true);
        } catch (const std::exception& e) {
            LOG_ERROR("Memory", std::string("Failed to parse memory.json: ") + e.what());
            longTermMemory = nlohmann::json::object();
            LOG_PHASE("Memory load", false);
        }
    } else {
        LOG_DEBUG("Memory", "No memory.json found. Creating new file.");
        longTermMemory = nlohmann::json::object();
    }

    // Ensure voice structure exists
    voiceMemory();
    if (!longTermMemory.contains("voice_baseline")) {
        longTermMemory["voice_baseline"] = 0.0;
    }

    saveMemory();
}

// =========================================================
// Voice helpers
// =========================================================
void rememberCorrection(const std::string& wrong, const std::string& right) {
    voiceMemory()["corrections"][wrong] = right;
    saveMemory();
}

void rememberShortcut(const std::string& phrase, const std::string& command) {
    voiceMemory()["shortcuts"][phrase] = command;
    saveMemory();
}

void incrementUsageCount(const std::string& command) {
    auto& counts = voiceMemory()["usage_counts"];
    if (!counts.contains(command)) counts[command] = 0;
    counts[command] = counts[command].get<int>() + 1;
    saveMemory();
}

void setLastCommand(const std::string& command) {
    voiceMemory()["last_command"] = command;
    saveMemory();
}

// =========================================================
// Core async AI call — all requests go through MMO orchestrator
// =========================================================
static constexpr const char* kDefaultSessionId = "default";

void clearConversationHistory() {
    auto& scm = GRIM::MMO::SessionContextManager::instance();
    scm.clearHistory(kDefaultSessionId);
    LOG_DEBUG("AI", "Conversation history cleared");
}

// =========================================================
// callOllamaDirect — bypass orchestrator for Ollama backend
// =========================================================
static std::string callOllamaDirect(const std::string& prompt) {
    std::string url   = aiConfig.value("ollama_url", "http://127.0.0.1:11434");
    std::string model = aiConfig.value("default_model", "llama3.1:8b");

    GRIM::MMO::OllamaBackend backend(url, "ollama-direct", model);

    // Build conversation history from SessionContextManager
    auto& scm = GRIM::MMO::SessionContextManager::instance();
    auto messages = scm.getMessages(kDefaultSessionId);

    std::vector<GRIM::MMO::HistoryEntry> history;
    history.reserve(messages.size());
    for (const auto& msg : messages) {
        history.push_back({msg.role, msg.content});
    }

    GRIM::MMO::GenerationOptions opts;
    opts.timeout_ms = 60000;

    GRIM::MMO::GenerationResult gen = backend.generateWithHistory(prompt, history, opts);
    if (gen.success) {
        return gen.text;
    }

    LOG_ERROR("AI", "Ollama direct call failed: " + gen.error);
    return "[AI] Backend call failed: " + gen.error;
}

std::future<std::string> callAIAsync(const std::string& prompt) {
    return std::async(std::launch::async, [prompt]() -> std::string {
        // Check aiConfig["backend"] to decide routing
        std::string backend = aiConfig.value("backend", "auto");

        // Direct Ollama path — no orchestrator needed
        if (backend == "ollama") {
            return callOllamaDirect(prompt);
        }

        // All other backends route through MMO orchestrator
        if (!g_orchestrator) {
            throw std::runtime_error(
                "g_orchestrator is NULL — MMO must be initialized before calling AI. "
                "Check bootstrap/bootstrap.cpp bootstrapMMOLayer().");
        }

        auto& registry = GRIM::MMO::ModelRegistry::instance();
        if (!registry.isEnabled()) {
            throw std::runtime_error(
                "MMO is disabled in config but backend '" + backend
                + "' requires it. Set mmo.enabled=true in ai_config.json "
                "or set backend to 'ollama'.");
        }

        // Produce NlpAnnotation for structured routing
        auto& scm = GRIM::MMO::SessionContextManager::instance();
        GRIM::MMO::ContextSnapshotV2 snapshot = scm.snapshot("default");
        GRIM::NlpAnnotation annotation = GRIM::annotate(prompt, snapshot);

        // Build router metadata envelope
        auto& toolReg = GRIM::MMO::ToolRegistry::instance();
        GRIM::RouterMetadataBuilder builder;
        builder.setAnnotation(annotation)
               .setContextV2(snapshot)
               .setToolSummary(toolReg.generateCompactPrompt());
        GRIM::RouterMetadata meta = builder.build();

        GRIM::MMO::RequestContext ctx;
        ctx.request_id = std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
        ctx.prompt = prompt;
        ctx.metadata_json = meta.toJson().dump();
        ctx.tool_registry_version = toolReg.version();

        auto result = g_orchestrator->generate(ctx);
        if (result.success) {
            return result.response;
        }

        LOG_ERROR("AI", "MMO orchestrator failed: " + result.error);
        return "[AI] Orchestrator error: " + result.error;
    });
}

// =========================================================
// Blocking AI call → returns CommandResult (with retry)
// =========================================================
CommandResult ai_process(const std::string& input) {
    CommandResult result;
    result.category  = "routine";
    result.color     = Colors::Cyan;
    result.success   = false;
    result.errorCode = "ERR_AI_BACKEND_UNAVAILABLE";

    const int maxRetries = 2;
    std::string reply;

    for (int attempt = 1; attempt <= maxRetries; ++attempt) {
        try {
            std::string prefix = GRIM::PersonalityManager::generatePrefix();
            
            // ✅ NEW: Add location context for location-aware conversations
            std::string locationContext = "";
            std::string lowerInput = input;
            std::transform(lowerInput.begin(), lowerInput.end(), lowerInput.begin(), ::tolower);
            
            bool isLocationRelevant = (
                lowerInput.find("weather") != std::string::npos ||
                lowerInput.find("near me") != std::string::npos ||
                lowerInput.find("nearby") != std::string::npos ||
                lowerInput.find("local") != std::string::npos ||
                lowerInput.find("where am i") != std::string::npos ||
                lowerInput.find("my location") != std::string::npos ||
                lowerInput.find("around here") != std::string::npos
            );
            
            if (isLocationRelevant && (g_location.lat != 0.0 || g_location.lon != 0.0)) {
                locationContext = " [USER LOCATION: " + g_location.fullAddress() + "]";
            }
            
            auto future = callAIAsync(prefix + locationContext + " " + input);
            reply = future.get();


            if (!reply.empty() && reply.rfind("[AI] Backend call failed", 0) != 0) {
                result.success = true;
                result.errorCode = "ERR_NONE";
                break;
            }

            LOG_DEBUG("AI", "Attempt " + std::to_string(attempt) + " failed: " + reply);
        }
        catch (const std::exception& e) {
            LOG_ERROR("AI", std::string("Exception on attempt ") + std::to_string(attempt) + ": " + e.what());
        }
    }

    // Memory update
    longTermMemory["last_input"] = input;
    longTermMemory["last_reply"] = reply;
    saveMemory();

    result.message = reply.empty() ? "[AI] Failed to process request" : reply;
    result.voice   = result.message;
    return result;
}
// =========================================================
// Unified AI interpreter — determines if input is a command or conversation
// =========================================================
CommandResult ai_interpret(const std::string& input, bool allowCommands)
{
    CommandResult result;
    result.category = "ai_interpret";
    result.color = Colors::Cyan;
    result.success = false;
    result.errorCode = "ERR_AI_INTERPRET_FAIL";

    try {
        // Record user message in conversation history
        auto& scmHistory = GRIM::MMO::SessionContextManager::instance();

        // Set system prompt once per session
        if (aiConfig.contains("personality")) {
            auto personality = aiConfig["personality"];
            if (personality.value("use_custom_prompt", false)) {
                std::string sys = personality.value("custom_prompt",
                    "You are GRIM. Be brief and direct.");
                scmHistory.setSystemPrompt(kDefaultSessionId, sys);
            }
        }

        scmHistory.addMessage(kDefaultSessionId, "user", input);
        size_t max_hist = aiConfig.value("conversation_history_size", 10);
        scmHistory.trimHistory(kDefaultSessionId, max_hist);

        // --- Build context prompt with strict JSON formatting ---
        std::string personalityPrefix = GRIM::PersonalityManager::generatePrefix();
        
        // ✅ NEW: Add location context for location-aware queries
        std::string locationContext = "";
        if (g_location.lat != 0.0 || g_location.lon != 0.0) {
            locationContext = "\n[USER LOCATION: " + g_location.fullAddress() + 
                            " (lat: " + std::to_string(g_location.lat) + 
                            ", lon: " + std::to_string(g_location.lon) + ")]";
        }

        // ✅ NEW: Add available commands context for AI
        std::string availableCommands = GRIM::MMO::ToolRegistry::instance().generateCompactPrompt();
        
        std::string prompt =
            "You are GRIM. Respond with ONLY valid JSON, no markdown, no explanations.\n\n"
            + availableCommands + "\n"  // ✅ Available tools
            "Format:\n"
            "{\"intent\":\"command\",\"suggested_command\":\"command_name argument\"}\n"
            "OR\n"
            "{\"intent\":\"conversation\",\"response\":\"...\"}\n\n"
            "Rules:\n"
            "- intent must be: \"command\", \"conversation\", or \"question\"\n"
            "- If command: suggested_command MUST use a command from the list above with proper arguments\n"
            "- For actions: use action commands (open, search, etc.)\n"
            "- For questions: use information commands (system_info, list_tools, etc.) OR provide direct answer\n"
            "- If conversation/question: include \"response\"\n"
            "- Output MUST start with { and end with }\n"
            "- DO NOT invent commands - only use commands from the available list\n\n"
            + personalityPrefix + locationContext +
            "\n\nInput: \"" + input + "\"\n\n"
            "JSON response:";

        // --- Query backend via MMO Orchestrator ---
        LOG_DEBUG("AI", "ai_interpret dispatching via MMO Orchestrator");

        std::string reply;

        // Reuse the async infrastructure
        auto future = callAIAsync(prompt);
        reply = future.get();

        // --- Early validation of reply ---
        if (reply.empty() || reply.rfind("[AI] Backend call failed", 0) == 0) {
            LOG_ERROR("AI", "Backend returned error or empty response: " + reply);
            result.message = "[AI] Could not interpret input.";
            result.voice = "Sorry, I couldn't interpret that.";
            return result;
        }

        // --- Extract JSON from response (Mistral sometimes adds extra text) ---
        auto& trainingParser = GRIM::MMO::ToolTrainingParser::instance();
        auto parsed = trainingParser.parseModelOutput(reply);

        // Build training example (outcome filled in at each exit point)
        GRIM::MMO::ToolTrainingExample trainEx;
        trainEx.raw_user_input   = input;
        trainEx.raw_model_output = reply;
        trainEx.extracted_json   = parsed.extracted_json;

        if (!parsed.valid) {
            LOG_ERROR("AI", "No valid JSON in response: " + reply);
            trainEx.parsed_intent = "parse_failure";
            trainEx.outcome = GRIM::MMO::ToolTrainingExample::Outcome::ParseFailure;
            trainingParser.record(std::move(trainEx));

            result.message = "[AI] Invalid response format.";
            result.voice = "Sorry, I got a malformed response.";
            return result;
        }

        LOG_DEBUG("AI", "Extracted JSON: " + parsed.extracted_json);

        // Use parsed result
        const nlohmann::json& j = parsed.parsed;

        // --- Route based on intent ---
        std::string intent = parsed.intent;

        if (intent == "command" && allowCommands) {
            std::string suggested = j.value("suggested_command", "");
            if (!suggested.empty()) {
                LOG_DEBUG("AI", "Interpreter inferred command: " + suggested);
                
                // ✅ Check if this is a complex multi-step task
                if (GRIM::TaskPlanner::isComplexTask(suggested)) {
                    LOG_DEBUG("AI", "Detected complex task, planning execution: " + suggested);
                    
                    // Plan the task
                    auto task = GRIM::TaskPlanner::planTask(suggested, input);
                    
                    if (task && !task->steps.empty()) {
                        // Present plan to user
                        std::string planSummary = "I'll do this in " + std::to_string(task->steps.size()) + " steps. Proceed?";
                        Voice::speak(planSummary, "question");
                        
                        // TODO: Wait for user confirmation, then execute
                        // For now, return the plan
                        result.message = planSummary;
                        result.category = "task_plan";
                        result.success = true;
                        result.errorCode = "ERR_NONE";
                        return result;
                    }
                }
                
                // Gate through ActionPolicyRegistry (Training Wheels)
                LOG_DEBUG("AI", "Evaluating policy for command: " + suggested);
                
                auto [cmd, arg] = GRIMInput::parseInput(suggested);

                // Check if the tool exists in the registry before proceeding
                auto& toolRegGap = GRIM::MMO::ToolRegistry::instance();
                if (!toolRegGap.isRegistered(cmd) && !toolRegGap.resolveAlias(cmd).has_value()) {
                    // Tool not recognized — check for structured tool gap
                    std::string gap_req_id = std::to_string(
                        std::chrono::steady_clock::now().time_since_epoch().count());
                    auto gap = g_orchestrator->evaluateToolGap(cmd, gap_req_id);
                    if (gap) {
                        LOG_DEBUG("AI", "Tool gap detected for: " + cmd);
                        trainEx.parsed_intent  = "command";
                        trainEx.parsed_tool_id = cmd;
                        trainEx.parsed_args    = arg;
                        trainEx.outcome = GRIM::MMO::ToolTrainingExample::Outcome::GapDetected;
                        trainingParser.record(std::move(trainEx));

                        result.message = gap->rationale;
                        result.voice   = "I don't have a tool for that yet.";
                        result.category = "tool_gap";
                        result.success = false;
                        return result;
                    }
                }

                // Produce NlpAnnotation for confidence signals
                auto& scmPolicy = GRIM::MMO::SessionContextManager::instance();
                GRIM::MMO::ContextSnapshotV2 policySnapshot = scmPolicy.snapshot("default");
                GRIM::NlpAnnotation policyAnn = GRIM::annotate(input, policySnapshot);

                GRIM::MMO::ActionProposal proposal;
                proposal.tool_id            = cmd;
                proposal.args_json          = arg;
                proposal.router_confidence  = policyAnn.confidence_summary.overall;
                proposal.parse_certainty    = policyAnn.confidence_summary.intent_confidence;
                proposal.memory_match_quality   = 1.0f;
                proposal.referent_resolution    = policyAnn.references.empty() ? 1.0f : 0.6f;
                proposal.grounding_coverage     = policyAnn.confidence_summary.entity_confidence;
                proposal.tool_preconditions     = 1.0f;
                proposal.historical_success     = 1.0f;

                auto decision = GRIM::MMO::ActionPolicyRegistry::instance().evaluate(proposal);

                if (decision.verdict == GRIM::MMO::PolicyVerdict::Deny) {
                    LOG_DEBUG("AI", "Policy DENIED command: " + cmd + " (" + decision.reason + ")");
                    trainEx.parsed_intent    = "command";
                    trainEx.parsed_tool_id   = cmd;
                    trainEx.parsed_args      = arg;
                    trainEx.router_confidence = policyAnn.confidence_summary.overall;
                    trainEx.intent_confidence = policyAnn.confidence_summary.intent_confidence;
                    trainEx.entity_confidence = policyAnn.confidence_summary.entity_confidence;
                    trainEx.outcome = GRIM::MMO::ToolTrainingExample::Outcome::PolicyDenied;
                    trainingParser.record(std::move(trainEx));

                    result.message = "[AI] Action denied: " + decision.reason;
                    result.voice   = "I can't do that right now.";
                    result.success = false;
                    return result;
                }

                if (decision.verdict != GRIM::MMO::PolicyVerdict::Allow) {
                    // Verification required — present prompt to user
                    LOG_DEBUG("AI", "Policy VERIFY for: " + cmd + " (" + decision.reason + ")");

                    auto& sessionCtx = GRIM::MMO::SessionContextManager::instance();

                    // Record proposal so correction tuples can be collected on rejection
                    GRIM::MMO::ActionEpisode ep;
                    ep.tool_id    = cmd;
                    ep.proposed_args = arg;
                    ep.confidence = policyAnn.confidence_summary.overall;
                    ep.risk       = proposal.referent_resolution < 0.8f ? 0.5f : 0.2f;
                    ep.confirmation_requested = true;
                    sessionCtx.recordProposal("default", ep);

                    GRIM::MMO::PendingInteraction pending;
                    pending.kind             = GRIM::MMO::PendingKind::Confirmation;
                    pending.original_command = suggested;
                    pending.prompt_shown     = decision.verification_prompt;
                    pending.created_at       = std::chrono::steady_clock::now();
                    sessionCtx.setPending("default", pending);

                    trainEx.parsed_intent    = "command";
                    trainEx.parsed_tool_id   = cmd;
                    trainEx.parsed_args      = arg;
                    trainEx.router_confidence = policyAnn.confidence_summary.overall;
                    trainEx.intent_confidence = policyAnn.confidence_summary.intent_confidence;
                    trainEx.entity_confidence = policyAnn.confidence_summary.entity_confidence;
                    trainEx.outcome = GRIM::MMO::ToolTrainingExample::Outcome::PolicyVerify;
                    trainingParser.record(std::move(trainEx));

                    result.message  = decision.verification_prompt;
                    result.voice    = decision.verification_prompt;
                    result.category = "verification";
                    result.success  = true;
                    result.errorCode = "ERR_NONE";
                    return result;
                }

                // Policy allows — check for shadow mode before executing
                auto& mmoRegistry = GRIM::MMO::ModelRegistry::instance();
                if (mmoRegistry.isEnabled() && mmoRegistry.mode() == "shadow") {
                    LOG_DEBUG("AI", "[SHADOW] Would execute: " + suggested + " (suppressed)");
                    trainEx.parsed_intent  = "command";
                    trainEx.parsed_tool_id = cmd;
                    trainEx.parsed_args    = arg;
                    trainEx.outcome = GRIM::MMO::ToolTrainingExample::Outcome::ShadowSuppressed;
                    trainingParser.record(std::move(trainEx));

                    result.message = "[Shadow] I would run: " + suggested;
                    result.voice   = "";
                    result.category = "shadow_log";
                    result.success = true;
                    result.errorCode = "ERR_NONE";
                    return result;
                }

                LOG_DEBUG("AI", "Policy ALLOW — executing: " + suggested);
                
                CommandResult cmdResult;
                
                // Route action commands (click, type, press) through ActionExecutor
                if (GRIM::ActionExecutor::isActionCommand(suggested)) {
                    LOG_DEBUG("AI", "Routing through ActionExecutor: " + suggested);
                    cmdResult = GRIM::ActionExecutor::executeAction(suggested);
                } else {
                    extern CommandResult dispatchCommand(const std::string&, const std::string&);
                    cmdResult = dispatchCommand(cmd, arg);
                }
                
                // Record analytics in both registries
                auto& toolReg = GRIM::MMO::ToolRegistry::instance();
                if (cmdResult.success) {
                    toolReg.recordSuccess(cmd);
                    GRIM::MMO::ActionPolicyRegistry::instance().recordOutcome(cmd, policyAnn.confidence_summary.overall, true);
                } else {
                    toolReg.recordFailure(cmd);
                    GRIM::MMO::ActionPolicyRegistry::instance().recordOutcome(cmd, policyAnn.confidence_summary.overall, false);
                }

                // Record training example
                trainEx.parsed_intent    = "command";
                trainEx.parsed_tool_id   = cmd;
                trainEx.parsed_args      = arg;
                trainEx.router_confidence = policyAnn.confidence_summary.overall;
                trainEx.intent_confidence = policyAnn.confidence_summary.intent_confidence;
                trainEx.entity_confidence = policyAnn.confidence_summary.entity_confidence;
                trainEx.outcome = cmdResult.success
                    ? GRIM::MMO::ToolTrainingExample::Outcome::Success
                    : GRIM::MMO::ToolTrainingExample::Outcome::Failure;
                trainingParser.record(std::move(trainEx));
                
                // Mark that it was AI-inferred
                cmdResult.category = cmdResult.category.empty() ? "ai_inferred" : cmdResult.category;
                
                // Record assistant response in conversation history
                scmHistory.addMessage(kDefaultSessionId, "assistant", cmdResult.message);
                
                return cmdResult;
            }
        }
        else if (intent == "conversation" || intent == "question") {
            std::string response = j.value("response", "");
            if (!response.empty()) {
                LOG_DEBUG("AI", "Interpreter conversational reply: " + response);
                result.message = response;
                result.voice = response;
                result.category = "conversation";
                result.success = true;
                result.errorCode = "ERR_NONE";

                trainEx.parsed_intent = intent;
                trainEx.outcome = GRIM::MMO::ToolTrainingExample::Outcome::Conversation;
                trainingParser.record(std::move(trainEx));

                // Record assistant response in conversation history
                scmHistory.addMessage(kDefaultSessionId, "assistant", response);

                // store context
                longTermMemory["last_conversation"] = {
                    {"input", input},
                    {"reply", response},
                    {"timestamp", std::time(nullptr)}
                };
                saveMemory();
                return result;
            }
        }

        // --- If all else fails ---
        result.message = "[AI] I couldn't determine your intent.";
        result.voice = "I couldn't determine your intent.";
    }
    catch (const std::exception& e) {
        LOG_ERROR("AI", std::string("Exception in ai_interpret: ") + e.what());
        result.message = "[AI] Interpretation error";
        result.voice = "Something went wrong while interpreting.";
    }

    return result;
}

// =========================================================
// Warmup
// =========================================================
void warmupAI() {
    LOG_DEBUG("AI", "Warming up...");
    auto f = callAIAsync("Hello");
    f.wait();
    LOG_PHASE("AI warmup complete", true);
}
