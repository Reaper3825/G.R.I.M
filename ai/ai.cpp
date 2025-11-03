#include "ai.hpp"
#include "voice/voice.hpp"
#include "voice/voice_speak.hpp"  // ✅ For Voice::speak()
#include "resources.hpp"
#include "commands/commands_core.hpp"
#include "commands/command_registry.hpp"  // ✅ NEW: Command registry
#include "error_manager.hpp"
#include "logger.hpp"
#include "personality_manager.hpp"
#include "action_executor.hpp"  // ✅ Intelligent action execution
#include "task_planner.hpp"     // ✅ Multi-step task planning
#include "nlp/nlp.hpp"  // ✅ For NLP integration
#include "fast_classifier.hpp"  // ✅ For teaching classifier
#include "system_detect.hpp"  // ✅ For location context
#include "helpers/grim_input.hpp"  // ✅ For parseInput
#include <cpr/cpr.h>
#include <fstream>
#include <sstream>
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
        std::ofstream f("memory.json");
        if (f) {
            f << longTermMemory.dump(2);
            LOG_PHASE("Memory saved", true);
        }
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
// Backend resolver
// =========================================================
std::string resolveBackendURL() {
    std::string backend = aiConfig.value("backend", "auto");

    if (backend == "auto") {
        try {
            auto r = cpr::Get(cpr::Url{aiConfig.value("ollama_url","http://127.0.0.1:11434") + "/api/tags"},
                              cpr::Timeout{1000});
            if (r.status_code == 200) return "ollama";
        } catch (...) {}

        try {
            auto r = cpr::Get(cpr::Url{aiConfig.value("localai_url","http://127.0.0.1:8080/v1") + "/models"},
                              cpr::Timeout{1000});
            if (r.status_code == 200) return "localai";
        } catch (...) {}

        return "openai";
    }

    return backend;
}

// =========================================================
// Core async AI call
// =========================================================
// =========================================================
// Session management for KV cache reuse
// =========================================================
static std::vector<nlohmann::json> g_conversationHistory;
static std::string g_currentModel;  // Track model to detect changes
static bool g_modelWarmedUp = false;  // Track if model is loaded
static bool g_systemPromptAdded = false;  // Track if system prompt is in history

void clearConversationHistory() {
    g_conversationHistory.clear();
    g_systemPromptAdded = false;
    LOG_DEBUG("AI", "Conversation history cleared");
}

std::future<std::string> callAIAsync(const std::string& prompt) {
    return std::async(std::launch::async, [prompt]() -> std::string {
        std::string backend = resolveBackendURL();
        std::string model   = aiConfig.value("default_model", "llama3.1:8b");

        LOG_DEBUG("AI", "callAIAsync backend=" + backend + " model=" + model);

        try {
            if (backend == "ollama") {
                // ✅ LATENCY FIX: Model warmup detection
                if (g_currentModel != model) {
                    g_currentModel = model;
                    g_modelWarmedUp = false;
                    g_systemPromptAdded = false;
                    LOG_DEBUG("AI", "Model changed to " + model + ", warmup needed");
                }
                
                // ✅ LATENCY FIX: Add system prompt ONCE and keep it cached
                if (!g_systemPromptAdded && aiConfig.contains("personality")) {
                    auto personality = aiConfig["personality"];
                    if (personality.value("use_custom_prompt", false)) {
                        std::string systemPrompt = personality.value("custom_prompt", 
                            "You are GRIM. Be brief and direct.");
                        
                        // Create system message as proper JSON object
                        nlohmann::json systemMsg = nlohmann::json::object();
                        systemMsg["role"] = "system";
                        systemMsg["content"] = systemPrompt;
                        
                        g_conversationHistory.insert(g_conversationHistory.begin(), systemMsg);
                        g_systemPromptAdded = true;
                        LOG_DEBUG("AI", "System prompt cached in conversation history");
                    }
                }
                
                // Add user message to history
                nlohmann::json userMsg = nlohmann::json::object();
                userMsg["role"] = "user";
                userMsg["content"] = prompt;
                g_conversationHistory.push_back(userMsg);
                
                // Trim history based on config (keep recent context + system prompt)
                size_t maxHistory = aiConfig.value("conversation_history_size", 10);
                size_t systemMsgCount = g_systemPromptAdded ? 1 : 0;
                
                if (g_conversationHistory.size() > (maxHistory + systemMsgCount)) {
                    // Keep system message, trim from position 1 onwards
                    size_t toErase = g_conversationHistory.size() - (maxHistory + systemMsgCount);
                    g_conversationHistory.erase(
                        g_conversationHistory.begin() + systemMsgCount,
                        g_conversationHistory.begin() + systemMsgCount + toErase
                    );
                }
                
                // Build optimized request with configurable performance parameters
                nlohmann::json options = {
                    {"num_ctx", 4096},
                    {"num_predict", 512},
                    {"temperature", 0.7},
                    {"top_k", 40},
                    {"top_p", 0.9},
                    {"repeat_penalty", 1.1},
                    {"num_thread", 8},
                    {"num_gpu", 99},              // ✅ NEW: Use all available GPU layers
                    {"num_batch", 512},           // ✅ NEW: Batch size for prompt processing
                    {"use_mmap", true},           // ✅ NEW: Memory-map model for faster loading
                    {"use_mlock", false},         // ✅ NEW: Don't lock in RAM (can cause issues)
                    {"low_vram", false}           // ✅ NEW: We have enough VRAM
                };
                
                // Override with user config if present
                if (aiConfig.contains("ollama_options")) {
                    for (auto& [key, value] : aiConfig["ollama_options"].items()) {
                        options[key] = value;
                    }
                }
                
                nlohmann::json requestBody = {
                    {"model", model}, 
                    {"messages", g_conversationHistory},
                    {"stream", false},
                    {"options", options},
                    {"keep_alive", "30m"}  // ✅ NEW: Keep model loaded for 30 minutes
                };
                
                auto start = std::chrono::high_resolution_clock::now();
                
                auto resp = cpr::Post(
                    cpr::Url{ aiConfig.value("ollama_url", "http://127.0.0.1:11434") + "/api/chat" },
                    cpr::Header{{"Content-Type","application/json"}},
                    cpr::Body{ requestBody.dump() },
                    cpr::Timeout{60000}  // ✅ NEW: 60s timeout for first warmup
                );
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                
                if (resp.status_code == 200) {
                    LOG_DEBUG("AI", "Ollama response in " + std::to_string(duration) + "ms (" + 
                             std::to_string(resp.text.size()) + " bytes)");
                    
                    if (!g_modelWarmedUp) {
                        LOG_DEBUG("AI", "Model warmed up (first call took " + std::to_string(duration) + "ms)");
                        g_modelWarmedUp = true;
                    }
                    
                    auto j = nlohmann::json::parse(resp.text, nullptr, false);
                    
                    if (j.is_discarded()) {
                        LOG_ERROR("AI", "Failed to parse Ollama JSON: " + resp.text.substr(0, 500));
                        return "[AI] Backend call failed";
                    }
                    
                    std::string response = j["message"]["content"].get<std::string>();
                    
                    // Add assistant response to history for context
                    nlohmann::json assistantMsg = nlohmann::json::object();
                    assistantMsg["role"] = "assistant";
                    assistantMsg["content"] = response;
                    g_conversationHistory.push_back(assistantMsg);
                    
                    if (response.empty()) {
                        LOG_ERROR("AI", "Ollama returned empty response field. Full JSON: " + resp.text.substr(0, 500));
                        return "[AI] Backend call failed";
                    }
                    
                    return response;
                }
                
                // ✅ NEW: Log the actual error message from Ollama
                LOG_ERROR("AI", "Ollama HTTP " + std::to_string(resp.status_code) + ": " + resp.text);
                LOG_DEBUG("AI", "Request body was: " + requestBody.dump().substr(0, 500));
                return "[AI] Backend call failed";
            }
            else if (backend == "localai" || backend == "openai") {
                std::string url =
                    (backend == "localai")
                        ? aiConfig.value("localai_url","http://127.0.0.1:8080/v1") + "/chat/completions"
                        : "https://api.openai.com/v1/chat/completions";

                cpr::Header headers = {{"Content-Type","application/json"}};
                if (backend == "openai") {
                    auto apiKey = aiConfig["api_keys"].value("openai", "");
                    if (apiKey.empty()) return "[AI] Missing OpenAI API key";
                    headers["Authorization"] = "Bearer " + apiKey;
                }

                auto resp = cpr::Post(
                    cpr::Url{url},
                    headers,
                    cpr::Body{ nlohmann::json{
                        {"model", model},
                        {"messages", nlohmann::json::array({
                            {{"role","user"},{"content",prompt}}
                        })}
                    }.dump() }
                );
                if (resp.status_code == 200) {
                    auto j = nlohmann::json::parse(resp.text, nullptr, false);
                    if (j.contains("choices"))
                        return j["choices"][0]["message"]["content"].get<std::string>();
                }
            }
        }
        catch (const std::exception& e) {
            LOG_ERROR("AI", std::string("Exception: ") + e.what());
        }

        return "[AI] Backend call failed";
    });
}

// =========================================================
// Model warmup (preload and cache model in memory/VRAM)
// =========================================================
void warmupOllamaModel() {
    std::string backend = resolveBackendURL();
    if (backend != "ollama") {
        LOG_DEBUG("AI", "Skipping warmup - backend is not Ollama");
        return;
    }
    
    std::string model = aiConfig.value("default_model", "llama3.1:8b");
    LOG_DEBUG("AI", "Warming up Ollama model: " + model);
    
    try {
        // Use a minimal prompt to trigger model loading
        nlohmann::json warmupBody = {
            {"model", model},
            {"messages", nlohmann::json::array({
                {{"role", "user"}, {"content", "hi"}}
            })},
            {"stream", false},
            {"options", {
                {"num_predict", 1},  // Generate only 1 token
                {"num_ctx", 512}     // Minimal context
            }},
            {"keep_alive", "30m"}  // Keep model loaded
        };
        
        auto start = std::chrono::high_resolution_clock::now();
        
        auto resp = cpr::Post(
            cpr::Url{aiConfig.value("ollama_url", "http://127.0.0.1:11434") + "/api/chat"},
            cpr::Header{{"Content-Type", "application/json"}},
            cpr::Body{warmupBody.dump()},
            cpr::Timeout{60000}
        );
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        if (resp.status_code == 200) {
            LOG_DEBUG("AI", "Model warmup complete in " + std::to_string(duration) + "ms");
            g_currentModel = model;
            g_modelWarmedUp = true;
        } else {
            LOG_ERROR("AI", "Model warmup failed with status " + std::to_string(resp.status_code));
        }
    } catch (const std::exception& e) {
        LOG_ERROR("AI", std::string("Model warmup exception: ") + e.what());
    }
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
        std::string availableCommands = GRIM::CommandRegistry::generateCompactPrompt();
        
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

        // --- Query backend (Llama / LocalAI / OpenAI etc.) ---
        std::string backend = resolveBackendURL();
        std::string model   = aiConfig.value("default_model", "llama3.1:8b");

        LOG_DEBUG("AI", "ai_interpret backend=" + backend + " model=" + model);

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
        std::string jsonStr = reply;
        size_t jsonStart = reply.find('{');
        size_t jsonEnd = reply.rfind('}');
        
        if (jsonStart != std::string::npos && jsonEnd != std::string::npos && jsonEnd > jsonStart) {
            jsonStr = reply.substr(jsonStart, jsonEnd - jsonStart + 1);
            LOG_DEBUG("AI", "Extracted JSON: " + jsonStr);
        } else {
            LOG_ERROR("AI", "No JSON found in response: " + reply);
            result.message = "[AI] Invalid response format.";
            result.voice = "Sorry, I got a malformed response.";
            return result;
        }

        // --- Parse model response as JSON ---
        nlohmann::json j = nlohmann::json::parse(jsonStr, nullptr, false);
        
        // Validate JSON before accessing it
        if (j.is_discarded() || !j.is_object()) {
            LOG_ERROR("AI", "Interpretation failed — non-JSON response: " + reply);
            result.message = "[AI] Could not interpret input.";
            result.voice = "Sorry, I couldn't interpret that.";
            return result;
        }

        // --- Route based on intent ---
        std::string intent = j.value("intent", "conversation");

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
                
                // ✅ Check if this is a simple action command (click, type, press, etc.)
                if (GRIM::ActionExecutor::isActionCommand(suggested)) {
                    LOG_DEBUG("AI", "Detected action command, executing directly: " + suggested);
                    CommandResult actionResult = GRIM::ActionExecutor::executeAction(suggested);
                    
                    // Still teach the NLP about this pattern
                    try {
                        extern NLP g_nlp;
                        if (g_nlp.learnPattern(input, suggested)) {
                            LOG_DEBUG("AI", "✓ Taught NLP: \"" + input + "\" → " + suggested);
                        }
                    } catch (...) {}
                    
                    return actionResult;
                }
                
                // ✅ INTEGRATION #5: Teach both NLP and Fast Classifier
                try {
                    extern NLP g_nlp;
                    
                    // Teach NLP to recognize this pattern
                    if (g_nlp.learnPattern(input, suggested)) {
                        LOG_DEBUG("AI", "✓ Taught NLP: \"" + input + "\" → " + suggested);
                    }
                    
                    // Teach Fast Classifier the command words
                    std::string lowerInput = input;
                    std::transform(lowerInput.begin(), lowerInput.end(), 
                                 lowerInput.begin(), ::tolower);
                    std::string lowerSuggested = suggested;
                    std::transform(lowerSuggested.begin(), lowerSuggested.end(),
                                 lowerSuggested.begin(), ::tolower);
                    
                    // Extract verbs from the suggested command
                    std::istringstream iss(lowerSuggested);
                    std::string word;
                    while (iss >> word) {
                        // Check if this word appears in the input
                        if (lowerInput.find(word) != std::string::npos) {
                            // Common command verbs that should be boosted
                            static const std::vector<std::string> commandVerbs = {
                                "open", "close", "run", "launch", "show", "list",
                                "set", "create", "delete", "search", "find", "play",
                                "stop", "kill", "start", "restart", "shutdown"
                            };
                            
                            if (std::find(commandVerbs.begin(), commandVerbs.end(), word) != commandVerbs.end()) {
                                GRIM::FastClassifier::boostCommandWeight(word, 1.5f);
                                LOG_DEBUG("AI", "✓ Boosted Fast Classifier weight for: " + word);
                            }
                        }
                    }
                    
                } catch (const std::exception& e) {
                    LOG_ERROR("AI", std::string("Failed to teach systems: ") + e.what());
                }
                
                // ✅ NEW: Actually execute the suggested command using dispatchCommand
                LOG_DEBUG("AI", "Executing AI-suggested command: " + suggested);
                
                // Parse command and argument
                auto [cmd, arg] = GRIMInput::parseInput(suggested);
                
                // Execute the command through the normal dispatch
                extern CommandResult dispatchCommand(const std::string&, const std::string&);
                CommandResult cmdResult = dispatchCommand(cmd, arg);
                
                // Record analytics if successful
                if (cmdResult.success) {
                    GRIM::CommandRegistry::recordSuccess(cmd);
                } else {
                    GRIM::CommandRegistry::recordFailure(cmd);
                }
                
                // Mark that it was AI-inferred
                cmdResult.category = cmdResult.category.empty() ? "ai_inferred" : cmdResult.category;
                
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
// Streaming / incremental AI call
// =========================================================
void ai_process_stream(
    const std::string& input,
    nlohmann::json& memory,
    const std::function<void(const std::string&)>& callback
) {
    std::string backend = resolveBackendURL();
    std::string model   = aiConfig.value("default_model", "llama3.1:8b");

    LOG_DEBUG("AI", "ai_process_stream backend=" + backend + " model=" + model);

    bool success = false;

    try {
        if (backend == "ollama") {
            auto resp = cpr::Post(
                cpr::Url{ aiConfig.value("ollama_url", "http://127.0.0.1:11434") + "/api/generate" },
                cpr::Header{{"Content-Type","application/json"}},
                cpr::Body{ nlohmann::json{{"model", model}, {"prompt", input}, {"stream", true}}.dump() },
                cpr::Timeout{60000}
            );
            if (resp.status_code == 200) {
                std::istringstream ss(resp.text);
                std::string line;
                while (std::getline(ss, line)) {
                    if (!line.empty() && callback) callback(line + " ");
                }
                success = true;
            }
        }
        else if (backend == "localai" || backend == "openai") {
            std::string url =
                (backend == "localai")
                    ? aiConfig.value("localai_url","http://127.0.0.1:8080/v1") + "/chat/completions"
                    : "https://api.openai.com/v1/chat/completions";

            cpr::Header headers = {{"Content-Type","application/json"}};
            if (backend == "openai") {
                auto apiKey = aiConfig["api_keys"].value("openai", "");
                if (apiKey.empty()) {
                    if (callback) callback("[AI] Missing OpenAI API key\n");
                    LOG_ERROR("AI", "Missing OpenAI API key");
                    return;
                }
                headers["Authorization"] = "Bearer " + apiKey;
            }

            auto resp = cpr::Post(
                cpr::Url{url},
                headers,
                cpr::Body{ nlohmann::json{
                    {"model", model},
                    {"stream", true},
                    {"messages", nlohmann::json::array({
                        {{"role","user"},{"content",input}}
                    })}
                }.dump() },
                cpr::Timeout{60000}
            );

            if (resp.status_code == 200) {
                std::istringstream ss(resp.text);
                std::string line;
                while (std::getline(ss, line)) {
                    if (line.rfind("data:", 0) == 0) {
                        std::string chunk = line.substr(5);
                        if (chunk.find("[DONE]") != std::string::npos) break;
                        auto j = nlohmann::json::parse(chunk, nullptr, false);
                        if (!j.is_discarded() && j.contains("choices")) {
                            auto delta = j["choices"][0]["delta"];
                            if (delta.contains("content")) {
                                if (callback) callback(delta["content"].get<std::string>());
                            }
                        }
                    }
                }
                success = true;
            }
        }
    }
    catch (const std::exception& e) {
        LOG_ERROR("AI", std::string("Exception in ai_process_stream: ") + e.what());
    }

    // Memory update
    memory["last_input"] = input;
    memory["last_reply"] = success ? "[streamed reply]" : "[AI] Stream failed";
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
