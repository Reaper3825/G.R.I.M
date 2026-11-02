#pragma once
#include "ui_tool_adapter.hpp"

namespace GRIM::WorldLab {
// Manual command boundary: parsing and result envelope only. The session owns its
// UI adapter; ToolRegistry owns handlers. Model policy integration is separate.
class WorldToolDispatch {
    UITools::PanelToolAdapter panels_;
public:
    void clear() { panels_.clear(); }
    // Single-flight submission is supplied by TextWorldClient, not deduplication here.
    nlohmann::json execute(const std::string& command, uint64_t actionId,
                          const std::function<bool()>& cancelled) {
        using namespace UITools;
        Json output = {{"action_id", actionId}, {"status", "failed"}};
        try {
            require(!command.empty() && command.size() <= 512 &&
                    command.find_first_of("\r\n\0", 0, 3) == std::string::npos, "Enter one UI command");
            const auto space = command.find(' ');
            const auto id = command.substr(0, space);
            auto target = space == std::string::npos ? std::string{} : command.substr(space + 1);
            while (!target.empty() && target.front() == ' ') target.erase(0, 1);
            while (!target.empty() && target.back() == ' ') target.pop_back();
            const auto& adapter = panelAdapter(id);
            output["tool_id"] = id;
            output.update(panels_.execute(adapter, target, cancelled));
            output["status"] = "completed";
        } catch (const CompletionUnknown& error) {
            output["status"] = "unknown";
            output["error"] = error.what();
            output["feedback"] = error.what();
        } catch (const std::exception& error) {
            output["error"] = error.what();
            output["feedback"] = std::string("UI action failed: ") + error.what();
        }
        return output;
    }
};
}
