#pragma once
#include <string>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include <functional>
#include "NLP.hpp"
#include "console_history.hpp"
#include "timer.hpp"
#include <nlohmann/json.hpp>

struct CommandContext {
    std::string& buffer;
    std::filesystem::path& currentDir;
    std::vector<Timer>& timers;
    nlohmann::json& longTermMemory;
    NLP& nlp;
    ConsoleHistory& history;
};

// Type for a command function
using CommandFunc = std::function<void(const std::vector<std::string>& args, CommandContext& ctx)>;

struct Command {
    std::string name;
    std::string description;
    CommandFunc func;
};

class CommandRegistry {
public:
    void registerCommand(const std::string& name, const std::string& description, CommandFunc func);
    void execute(const std::string& name, const std::vector<std::string>& args, CommandContext& ctx);
    void listCommands(ConsoleHistory& history) const;

private:
    std::unordered_map<std::string, Command> commands_;
};

// ✅ Unified NLP-aware handler
bool handleCommand(const Intent& intent,
                   std::string& buffer,
                   std::filesystem::path& currentDir,
                   std::vector<Timer>& timers,
                   nlohmann::json& longTermMemory,
                   NLP& nlp,
                   ConsoleHistory& history);
