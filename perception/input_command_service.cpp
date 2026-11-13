#include "input_command_service.hpp"
#include "commands/commands_perception.hpp"
#include "perception_context.hpp"

namespace GRIM::Perception {
bool ExecuteInputPrimitive(const nlohmann::json& request) {
    if (!g_contextManager) return false;
    const auto op = request.at("op").get<std::string>();
    if (op == "available") return true;
    if (op == "move") g_contextManager->moveMouseTo(request.at("x").get<int>(), request.at("y").get<int>());
    else if (op == "click") g_contextManager->clickMouse(request.at("button").get<std::string>());
    else if (op == "type") g_contextManager->typeText(request.at("text").get<std::string>(), 15);
    else if (op == "key") g_contextManager->tapKey(request.at("key").get<std::string>());
    else if (op == "combo") g_contextManager->pressKeyCombo(request.at("keys").get<std::vector<std::string>>());
    else throw std::invalid_argument("Unknown input primitive");
    return true;
}
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
