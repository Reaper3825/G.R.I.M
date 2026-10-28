#include "MMO/Core/ToolRegistry.hpp"
#include "commands_execution.hpp"
#include "commands_core.hpp"
#include "logger.hpp"
#include "memory/unified_memory.hpp"
#include "memory/memory_router.hpp"
#include "Reward_Learning/grim_rl.hpp"
#include "nlp/nlp.hpp"
#include "input_parser.hpp"
#include "synonyms.hpp"
#include "voice/voice_speak.hpp"
#include "helpers/color.hpp"
#include <algorithm>
#include <ctime>

extern GRIM::UnifiedMemoryStorage g_memoryStorage;
extern nlohmann::json longTermMemory;
extern NLP g_nlp;

namespace GRIM {
namespace CommandExecution {

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
    if (auto suggestion = GRIM::RewardLearning::suggestForUnknownCommand(cmd, arg, longTermMemory)) {
        return dispatchCommand(*suggestion, arg);
    }

    return CommandResult{false, "", "ERR_NOT_FOUND", "", "", Colors::Default};
}

void recordUnknownCommand(const std::string& cmd, const std::string& arg)
{
    try {
        GRIM::UnifiedMemoryObject unknown;
        unknown.id = GRIM::UnifiedMemoryObject::generateID();
        unknown.timestamp = static_cast<uint64_t>(std::time(nullptr));
        unknown.domain = GRIM::MemoryDomain::USER;
        unknown.type = GRIM::TypeTag::STRING;
        unknown.context = GRIM::ContextType::CONVERSATION;
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
        if (phrase == action) throw std::invalid_argument("A learned tool cannot route to itself");
        g_memoryStorage.storeLearnedCommand(phrase, action, confidence);
        GRIM::MMO::ToolDescriptor tool;
        tool.tool_id = tool.display_name = tool.usage = phrase;
        tool.provider_name = "learned";
        tool.category = "action";
        tool.description = "Learned route to " + action;
        tool.is_informational = false;
        if (!GRIM::MMO::ToolRegistry::instance().registerTool(tool,
                [action](const std::string& args) { return dispatchCommand(action, args); }, true))
            throw std::runtime_error("Learned command conflicts with an existing tool: " + phrase);
        
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
