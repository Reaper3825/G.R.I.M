#pragma once
#include <string>
#include "commands/commands_core.hpp"


namespace GRIM::DialogueProactive {
    void checkAfterCommand(const std::string& input, const CommandResult& result);
}
