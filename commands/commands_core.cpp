#include "commands_core.hpp"
#include "commands_memory.hpp"
#include "commands_filesystem.hpp"
#include "commands_timers.hpp"
#include "commands_interface.hpp"
#include "response_manager.hpp"
#include "console_history.hpp"
#include "voice_speak.hpp"
#include <nlohmann/json.hpp>
#include <sstream>
#include <iostream>

// ---------------- Globals ----------------
std::unordered_map<std::string, CommandFunc> commandMap;
ConsoleHistory history;
std::vector<Timer> timers;
std::filesystem::path g_currentDir;

// Externals
extern nlohmann::json longTermMemory;

// ---------------- Forward declarations ----------------
CommandResult cmdRemember(const std::string&);
CommandResult cmdRecall(const std::string&);
CommandResult cmdForget(const std::string&);
CommandResult cmdAiBackend(const std::string&);
CommandResult cmdReloadNlp(const std::string&);
CommandResult cmdShowPwd(const std::string&);
CommandResult cmdChangeDir(const std::string&);
CommandResult cmdListDir(const std::string&);
CommandResult cmdMakeDir(const std::string&);
CommandResult cmdRemoveFile(const std::string&);
CommandResult cmdSetTimer(const std::string&);
CommandResult cmdSystemInfo(const std::string&);
CommandResult cmdClean(const std::string&);
CommandResult cmdShowHelp(const std::string&);
CommandResult cmdVoice(const std::string&);
CommandResult cmdVoiceStream(const std::string&);

// ---------------- Command Map ----------------
std::unordered_map<std::string, std::function<CommandResult(const std::string&)>> commands = {
    {"remember", cmdRemember},
    {"recall",   cmdRecall},
    {"forget",   cmdForget},
    {"ai_backend", cmdAiBackend},
    {"reload_nlp", cmdReloadNlp},
    {"pwd", cmdShowPwd},
    {"cd", cmdChangeDir},
    {"ls", cmdListDir},
    {"mkdir", cmdMakeDir},
    {"rm", cmdRemoveFile},
    {"timer", cmdSetTimer},
    {"sysinfo", cmdSystemInfo},
    {"clean", cmdClean},
    {"help", cmdShowHelp},
    {"voice", cmdVoice},
    {"voice_stream", cmdVoiceStream}
};

// Initialize commandMap at startup
struct CommandMapInitializer {
    CommandMapInitializer() {
        for (auto& kv : commands) {
            commandMap[kv.first] = kv.second;
        }
    }
} commandMapInitializer;

// ---------------- Dispatcher implementations ----------------

// Main dispatcher: calls the correct command function based on intent
bool handleCommand(const Intent& intent,
                   std::string& arg,
                   std::filesystem::path& currentDir,
                   std::vector<Timer>& timersRef,
                   nlohmann::json& longTermMemory,
                   NLP& nlp,
                   ConsoleHistory& console)
{
    auto it = commandMap.find(intent.name);
    if (it != commandMap.end()) {
        CommandFunc func = it->second;
        CommandResult result = func(arg); // logic lives in the individual command file
        return result;
    } else {
        console.push("Unknown command: " + intent.name, sf::Color::Red);
        return false;
    }
}

// Parses raw input and dispatches it
bool parseAndDispatch(const std::string& input,
                      std::string& arg,
                      std::filesystem::path& currentDir,
                      std::vector<Timer>& timersRef,
                      nlohmann::json& longTermMemory,
                      ConsoleHistory& console)
{
    std::istringstream iss(input);
    std::string cmd;
    iss >> cmd;
    std::getline(iss, arg);
    if (!arg.empty() && arg[0] == ' ') arg.erase(0, 1);

    Intent intent;
    intent.name = cmd;

    NLP dummyNLP; // placeholder in case NLP parsing needed

    return handleCommand(intent, arg, currentDir, timersRef, longTermMemory, dummyNLP, console);
}
