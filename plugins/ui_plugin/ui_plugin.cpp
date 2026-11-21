#include <chrono>
#include <string_view>
#include <thread>
#include "core/plugin_api.hpp"
#include "core/plugin_tool_result.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace {
grim::PluginHost host = {};
GrimResult deliver(GrimToolResultWriter* writer, bool success, const std::string& message,
                   const char* error, const std::string& json = {}) {
    return grim::completeToolResult(writer, success, message.c_str(), error, "ui",
                                    nullptr,
                                    json.empty() ? nullptr : json.c_str());
}
nlohmann::json panelRequest(GrimUIPanelOperation operation, const char* key, uint64_t id) {
    const char* raw = host->ui_panel_service(host.instance(), operation, key, id);
    if (!raw) throw std::runtime_error("UI panel service unavailable or request denied");
    struct Release {
        const char* text;
        ~Release() { host->string_free(host.instance(), text); }
    } release{raw};
    auto response = nlohmann::json::parse(raw);
    if (response.at("format") != "grim-ui-panel-response" || response.at("version") != 1)
        throw std::runtime_error("Unsupported UI service response");
    return response;
}

GrimResult completePanelRequest(GrimToolResultWriter* writer, nlohmann::json response) {
    if (response.at("status") == "pending") {
        const auto id = response.at("request_id").get<uint64_t>();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        try {
            do {
                if (std::chrono::steady_clock::now() >= deadline)
                    throw std::runtime_error("UI request timed out");
                response = panelRequest(GRIM_UI_POLL, nullptr, id);
                if (response.at("request_id") != id)
                    throw std::runtime_error("UI request identity mismatch");
                if (response.at("status") == "pending") std::this_thread::sleep_for(std::chrono::milliseconds(20));
            } while (response.at("status") == "pending");
            if (response.at("status") != "completed")
                throw std::runtime_error(response.value("error", "UI request failed"));
        } catch (const std::exception& error) {
            try { panelRequest(GRIM_UI_CANCEL, nullptr, id); } catch (...) {}
            return deliver(writer, false, error.what(), "ERR_TOOL_COMPLETION_UNKNOWN");
        }
    }
    if (response.at("status") != "completed")
        return deliver(writer, false, response.value("error", "UI request failed"), "ERR_UI_REQUEST");
    const auto& result = response.at("result");
    const bool success = result.at("success").get<bool>();
    const auto payload = result.dump();
    return deliver(writer, success, payload, success ? "ERR_NONE" : "ERR_UI_OPERATION", payload);
}

template<GrimUIPanelOperation Operation>
GrimResult call(const GrimToolCall* request, GrimToolResultWriter* writer, void*) {
    try {
        const std::string textArgs = request->arguments_size
            ? std::string(request->arguments, request->arguments_size) : std::string{};
        const auto args = Operation == GRIM_UI_LIST && textArgs.find_first_not_of(" \t\r\n") == std::string::npos
            ? nlohmann::json::object() : nlohmann::json::parse(textArgs);
        if (!args.is_object()) throw std::invalid_argument("Expected a JSON object");
        std::string key;
        uint64_t id = 0;
        bool cancel = false;
        if constexpr (Operation == GRIM_UI_LIST) {
            if (!args.empty()) throw std::invalid_argument("ui.list takes an empty object");
        } else if constexpr (Operation == GRIM_UI_POLL) {
            cancel = args.value("cancel", false);
            if (args.size() != (args.contains("cancel") ? 2 : 1) || !args.contains("request_id") || !args["request_id"].is_number_unsigned())
                throw std::invalid_argument("Expected a positive integer request_id");
            id = args["request_id"].get<uint64_t>();
            if (!id) throw std::invalid_argument("request_id must be positive");
        } else {
            const bool guarded = (Operation == GRIM_UI_SHOW || Operation == GRIM_UI_HIDE) && args.contains("expected_runtime_id");
            if (args.size() != (guarded ? 2 : 1) || !args.contains("panel_id") || !args["panel_id"].is_string())
                throw std::invalid_argument("Expected a string panel_id from ui.list");
            if (guarded) {
                if (!args["expected_runtime_id"].is_number_unsigned() ||
                    !(id = args["expected_runtime_id"].get<uint64_t>()))
                    throw std::invalid_argument("expected_runtime_id must be a positive integer");
            }
            key = args["panel_id"].get<std::string>();
            if (key.empty() || key.size() > 256 || key.find('\0') != std::string::npos)
                throw std::invalid_argument("panel_id must contain 1-256 bytes without NUL");
        }
        const auto response = panelRequest(cancel ? GRIM_UI_CANCEL :
            (Operation == GRIM_UI_SHOW && id ? GRIM_UI_SHOW_CURRENT :
                (Operation == GRIM_UI_HIDE && id ? GRIM_UI_HIDE_CURRENT : Operation)),
            Operation == GRIM_UI_LIST || Operation == GRIM_UI_POLL ? nullptr : key.c_str(), id);
        if (request->struct_size >= sizeof(GrimToolCall) && request->wait_for_completion)
            return completePanelRequest(writer, response);
        const auto text = response.dump();
        const auto status = response.at("status").get<std::string>();
        if (status == "pending")
            return deliver(writer, true, "UI request " + std::to_string(response.at("request_id").get<uint64_t>()) +
                " is pending; use ui.poll after UI processing. The operation is not yet complete.", "ERR_NONE", text);
        if (status == "completed") {
            const bool success = response.at("result").at("success").get<bool>();
            return deliver(writer, success, response.at("result").dump(), success ? "ERR_NONE" : "ERR_UI_OPERATION", text);
        }
        return deliver(writer, false, response.value("error", "Unknown UI request error"), "ERR_UI_REQUEST", text);
    } catch (const nlohmann::json::exception& error) {
        return deliver(writer, false, error.what(), "ERR_INVALID_ARGS");
    } catch (const std::invalid_argument& error) {
        return deliver(writer, false, error.what(), "ERR_INVALID_ARGS");
    } catch (const std::exception& error) {
        return deliver(writer, false, error.what(), "ERR_UI_SERVICE");
    }
}
}

extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* table, GrimPluginHandle instance) {
    const grim::PluginHost api{table, instance};
    if (!api || api->api_version < GRIM_API_VERSION || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR ||
        !api->ui_panel_service) return GRIM_ERROR_API_VERSION_MISMATCH;
    host = api;
    return GRIM_OK;
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_dispatch_tool(
    const char* tool_id, const GrimToolCall* request, GrimToolResultWriter* writer) {
    if (!tool_id) return GRIM_ERROR_INVALID_PARAM;
    const std::string_view id(tool_id);
    if (id == "ui.list") return call<GRIM_UI_LIST>(request, writer, nullptr);
    if (id == "ui.inspect") return call<GRIM_UI_INSPECT>(request, writer, nullptr);
    if (id == "ui.show") return call<GRIM_UI_SHOW>(request, writer, nullptr);
    if (id == "ui.hide") return call<GRIM_UI_HIDE>(request, writer, nullptr);
    if (id == "ui.bring_to_front") return call<GRIM_UI_BRING_TO_FRONT>(request, writer, nullptr);
    if (id == "ui.poll") return call<GRIM_UI_POLL>(request, writer, nullptr);
    return GRIM_ERROR_NOT_FOUND;
}

extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() { host = {}; }
