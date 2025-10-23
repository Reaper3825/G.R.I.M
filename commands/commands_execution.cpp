#include "commands_execution.hpp"
#include "commands_core.hpp"
#include "logger.hpp"
#include "memory/memory_storage.hpp"
#include "memory/memory_router.hpp"
#include "ai/ai_rl.hpp"
#include "nlp/nlp.hpp"
#include "input_parser.hpp"
#include "synonyms.hpp"
#include "console_history.hpp"
#include "voice/voice_speak.hpp"
#include "helpers/color.hpp"
#include <algorithm>
#include <ctime>

extern GRIM::MemoryStorage g_memoryStorage;
extern nlohmann::json longTermMemory;
extern NLP g_nlp;
extern std::unordered_map<std::string, CommandFunc> commandMap;

#define history getConsoleHistory()

namespace GRIM {
namespace CommandExecution {

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

CommandResult tryLearnedCommand(const std::string& cmd, const std::string& arg)
{
    try {
        auto learned = g_memoryStorage.findLearnedCommand(cmd);
        if (learned.has_value()) {
            LOG_DEBUG("Dispatch", "Matched learned command: \"" + learned->raw + "\" → \"" + learned->normalized + "\"");
            return dispatchCommand(learned->normalized, arg);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Dispatch", std::string("Learned-command lookup failed: ") + e.what());
    }

    return CommandResult{false, "", "ERR_NOT_FOUND", "", "", Colors::Default};
}

CommandResult tryRLInference(const std::string& cmd, const std::string& arg)
{
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

    return CommandResult{false, "", "ERR_NOT_FOUND", "", "", Colors::Default};
}

void recordUnknownCommand(const std::string& cmd, const std::string& arg)
{
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
}

void storeLearnedCommand(const std::string& phrase, const std::string& action, float confidence)
{
    try {
        g_memoryStorage.storeLearnedCommand(phrase, action, confidence);
        g_learnedCommandMap[phrase] = action;
        commandMap[phrase] = handleLearnedCommand;
        
        LOG_DEBUG("LearnedCmd", "Stored learned command: \"" + phrase + "\" → \"" + action + "\" (confidence=" + std::to_string(confidence) + ")");
    } catch (const std::exception& e) {
        LOG_ERROR("LearnedCmd", std::string("Failed to store learned command: ") + e.what());
        throw;
    }
}

std::vector<std::pair<std::string, float>> findSimilarLearnedCommands(const std::string& input, float minSimilarity)
{
    std::vector<std::pair<std::string, float>> suggestions;
    
    try {
        auto learnedCommands = g_memoryStorage.getAllLearnedCommands();
        
        // Levenshtein distance function
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
        
        for (const auto& learned : learnedCommands) {
            int distance = levenshtein(input, learned.raw);
            float similarity = 1.0f - (static_cast<float>(distance) / 
                              std::max(input.length(), learned.raw.length()));
            
            // Weight by learned confidence
            float score = similarity * learned.confidence;
            
            if (similarity >= minSimilarity) {
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
    
    return suggestions;
}

} // namespace CommandExecution
} // namespace GRIM
