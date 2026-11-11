#pragma once
#include "ui_tool_contract.hpp"
#include "commands/command_result.hpp"
#include <chrono>
#include <functional>
#include <thread>

namespace GRIM::WorldLab::UITools {
// Stateless request protocol. Poll/cancel stay with the submitting plugin generation.
class PanelRequestCompletion {
    static Json invoke(const std::string& id, const Json& args, uint64_t version) {
        const auto result = MMO::ToolRegistry::instance().invoke(
            id, args.dump(), {.expected_registry_version = version, .require_canonical_model = true});
        require(!result.structuredOutputJson.empty(), result.errorCode + ": " + result.message);
        auto response = Json::parse(result.structuredOutputJson);
        require(response.at("format") == "grim-ui-panel-response" && response.at("version") == 1,
                "Unsupported UI response");
        return response;
    }
public:
    static Json complete(const std::string& id, const Json& args, uint64_t version,
                         const MMO::RegisteredTool& owner, const std::function<bool()>& cancelled) {
        auto poll = descriptor("ui.poll", version);
        require(sameProvider(poll, owner), "UI poll provider mismatch");
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
        } catch (const std::exception& error) {
            // Reap even after unrelated metadata changes, but never poll a replacement owner.
            try {
                const auto currentVersion = MMO::ToolRegistry::instance().version();
                const auto current = descriptor("ui.poll", currentVersion);
                if (sameProvider(current, owner))
                    invoke("ui.poll", {{"request_id", requestId}, {"cancel", true}}, currentVersion);
            } catch (...) {}
            throw CompletionUnknown(std::string("UI completion unknown: ") + error.what() + "; inspect before retrying");
        }
        require(response.at("status") == "completed", response.value("error", "UI request failed"));
        const auto result = response.at("result");
        require(result.at("success").get<bool>(), result.value("error", "UI operation failed"));
        return result;
    }
};
}
