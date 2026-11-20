#pragma once
#include "MMO/Core/Registries/ToolRegistry.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include "commands/command_result.hpp"
#include <chrono>
#include <functional>
#include <thread>

namespace GRIM::UI {
using Json = nlohmann::json;

struct CompletionUnknown : std::runtime_error { using std::runtime_error::runtime_error; };

inline void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

inline MMO::RegisteredTool panelTool(const std::string& id, uint64_t version) {
    auto& registry = MMO::ToolRegistry::instance();
    const auto tool = registry.getTool(id);
    require(tool && registry.version() == version && tool->tool_id == id &&
            tool->usage_type == MMO::ToolUsage::Model && tool->argument_format == "json",
            "UI tool is unavailable or changed");
    return *tool;
}
// Stateless client for the UI plugin request protocol.
class PanelRequestCompletion {
    static Json invoke(const std::string& id, const Json& args, uint64_t version) {
        auto result = MMO::ToolRegistry::instance().invoke(
            id, args.dump(), {.expected_registry_version = version, .require_canonical_model = true});
        require(!result.structuredOutput.is_null(), result.errorCode + ": " + result.message);
        auto response = std::move(result.structuredOutput);
        require(response.at("format") == "grim-ui-panel-response" && response.at("version") == 1,
                "Unsupported UI response");
        return response;
    }
public:
    static Json complete(const std::string& id, const Json& args, uint64_t version,
                         const std::function<bool()>& cancelled) {
        panelTool("ui.poll", version);
        require(!cancelled(), "UI action cancelled before submission");
        auto response = invoke(id, args, version);
        const auto requestId = response.at("request_id").get<uint64_t>();
        require(requestId != 0, "Invalid UI request identity");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        try {
            while (response.at("status") == "pending") {
                require(!cancelled() && std::chrono::steady_clock::now() < deadline, "cancelled or timed out");
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                response = invoke("ui.poll", {{"request_id", requestId}}, version);
                require(response.at("request_id") == requestId, "UI response identity mismatch");
            }
            require(response.at("status") == "completed", response.value("error", "UI request failed"));
        } catch (const std::exception& error) {
            // The panel service scopes request IDs to the calling plugin instance.
            try {
                const auto currentVersion = MMO::ToolRegistry::instance().version();
                panelTool("ui.poll", currentVersion);
                invoke("ui.poll", {{"request_id", requestId}, {"cancel", true}}, currentVersion);
            } catch (...) {}
            throw CompletionUnknown(std::string("UI completion unknown: ") + error.what() + "; inspect before retrying");
        }
        const auto result = response.at("result");
        require(result.at("success").get<bool>(), result.value("error", "UI operation failed"));
        return result;
    }
};
}
