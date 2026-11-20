#pragma once
#include "core/host_object_actions.hpp"

namespace GRIM::WorldLab {
// Route an explicitly allowed object action and attach the world's action identity.
class WorldToolDispatch {
public:
    static nlohmann::json tools() {
        return GRIM::HostObjectActions::tools();
    }

    static nlohmann::json execute(const std::string& command, uint64_t actionId,
                                  const std::function<bool()>& cancelled) {
        nlohmann::json output = {{"action_id", actionId}, {"status", "failed"}};
        try {
            if (command.empty() || command.size() > 512 ||
                command.find_first_of("\r\n\0", 0, 3) != std::string::npos)
                throw std::invalid_argument("Enter one object tool command");
            const auto space = command.find(' ');
            const auto id = command.substr(0, space);
            const auto first = space == std::string::npos ? space : command.find_first_not_of(' ', space);
            const auto target = first == std::string::npos ? std::string{} :
                command.substr(first, command.find_last_not_of(' ') - first + 1);
            output["tool_id"] = id;
            output.update(GRIM::HostObjectActions::execute(id, target, cancelled));
        } catch (const std::exception& error) {
            output["error"] = error.what();
            output["feedback"] = std::string("Object action failed: ") + error.what();
        }
        return output;
    }
};
}
