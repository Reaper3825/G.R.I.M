#pragma once
#include "commands/commands_core.hpp"

// [AI] Select / show current backend
CommandResult cmdAiBackend(const std::string& arg);

// [NLP] Reload rules
CommandResult cmdReloadNlp(const std::string& arg);
