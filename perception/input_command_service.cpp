#include "input_command_service.hpp"
#include "commands/commands_perception.hpp"

namespace GRIM::Perception {
CommandResult ExecuteInputCommandService(GrimInputOperation operation, const std::string& arguments) {
    switch (operation) {
    case GRIM_INPUT_MOVE_MOUSE: return cmdInputMoveMouse(arguments);
    case GRIM_INPUT_CLICK: return cmdInputClick(arguments);
    case GRIM_INPUT_TYPE: return cmdInputType(arguments);
    case GRIM_INPUT_KEY: return cmdInputKey(arguments);
    case GRIM_INPUT_CLICK_ON: return cmdInputClickOn(arguments);
    default: return {false, "Unknown input operation", "ERR_INVALID_ARGS", "error", "", Colors::Red};
    }
}
}
