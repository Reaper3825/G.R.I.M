#include "core/plugin_api.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace {
const GrimPluginAPI* host = nullptr;
GrimResult deliver(GrimToolResultWriter* writer, bool success, const std::string& message,
                   const char* error, const std::string& json = {}) {
    GrimToolResult result;
    result.success = success; result.message = message.c_str();
    result.error_code = error; result.category = "ui";
    result.color_rgb = success ? 0x64FFFF : 0xFF6464;
    result.structured_output_json = json.empty() ? nullptr : json.c_str();
    return writer->complete(writer, &result);
}
template<GrimUIPanelOperation Operation>
GrimResult call(const GrimToolCall* request, GrimToolResultWriter* writer, void*) {
    try {
        const auto args = nlohmann::json::parse(std::string(request->arguments, request->arguments_size));
        if (!args.is_object()) throw std::invalid_argument("Expected a JSON object");
        std::string key;
        uint64_t id = 0;
        if constexpr (Operation == GRIM_UI_LIST) {
            if (!args.empty()) throw std::invalid_argument("ui.list takes an empty object");
        } else if constexpr (Operation == GRIM_UI_POLL) {
            if (args.size() != 1 || !args.contains("request_id") || !args["request_id"].is_number_unsigned())
                throw std::invalid_argument("Expected a positive integer request_id");
            id = args["request_id"].get<uint64_t>();
            if (!id) throw std::invalid_argument("request_id must be positive");
        } else {
            if (args.size() != 1 || !args.contains("panel_id") || !args["panel_id"].is_string())
                throw std::invalid_argument("Expected a string panel_id from ui.list");
            key = args["panel_id"].get<std::string>();
            if (key.empty() || key.size() > 256 || key.find('\0') != std::string::npos)
                throw std::invalid_argument("panel_id must contain 1-256 bytes without NUL");
        }
        const char* raw = host->ui_panel_service(host, Operation,
            Operation == GRIM_UI_LIST || Operation == GRIM_UI_POLL ? nullptr : key.c_str(), id);
        if (!raw) return deliver(writer, false, "UI panel service unavailable or request denied", "ERR_UI_SERVICE");
        std::string text;
        { struct Release { const char* raw; ~Release() { host->string_free(host, raw); } } release{raw}; text = raw; }
        const auto response = nlohmann::json::parse(text);
        if (response.at("format") != "grim-ui-panel-response" || response.at("version") != 1)
            throw std::runtime_error("Unsupported UI service response");
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

extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    static const GrimPluginInfo info{GRIM_API_VERSION, "UI Plugin", "1.0.0", "GRIM Project",
        "Discover, inspect, and control panels by stable identity with asynchronous completion", GRIM_PERM_UI};
    return &info;
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || api->api_version < GRIM_API_VERSION || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR ||
        !api->ui_panel_service) return GRIM_ERROR_API_VERSION_MISMATCH;
    host = api;
    GrimToolRegistration tools[6];
    const char* ids[] = {"ui.list", "ui.inspect", "ui.show", "ui.hide", "ui.bring_to_front", "ui.poll"};
    const char* descriptions[] = {
        "Queue discovery of instantiated panels and their stable IDs; poll for completion",
        "Queue a panel state snapshot by stable ID; poll for completion",
        "Queue showing a panel and bringing it forward; poll to verify completion",
        "Queue hiding a panel; poll to verify completion",
        "Queue bringing a panel forward without changing visibility; poll to verify completion",
        "Read an owned UI request status without blocking; a completed result is consumed"};
    const char* usages[] = {"ui.list {}", "ui.inspect {\"panel_id\":\"settings\"}",
        "ui.show {\"panel_id\":\"settings\"}", "ui.hide {\"panel_id\":\"settings\"}",
        "ui.bring_to_front {\"panel_id\":\"console\"}", "ui.poll {\"request_id\":1}"};
    tools[0].handler = call<GRIM_UI_LIST>; tools[1].handler = call<GRIM_UI_INSPECT>;
    tools[2].handler = call<GRIM_UI_SHOW>; tools[3].handler = call<GRIM_UI_HIDE>;
    tools[4].handler = call<GRIM_UI_BRING_TO_FRONT>; tools[5].handler = call<GRIM_UI_POLL>;
    GrimToolParameter parameters[6];
    const char* preconditions[] = {"Host UI initialized and processing frames"};
    const char* tags[] = {"ui", "panels"};
    for (size_t i = 0; i < 6; ++i) {
        auto& d = tools[i].descriptor;
        d.tool_id = d.display_name = ids[i]; d.version = "1.0.0";
        d.description = descriptions[i]; d.usage = usages[i]; d.category = "ui";
        d.argument_format = "json"; d.required_permissions = GRIM_PERM_UI;
        d.is_informational = i < 2 || i == 5; d.needs_confirmation = !d.is_informational;
        d.examples = {&usages[i], 1}; d.preconditions = {preconditions, 1}; d.capability_tags = {tags, 2};
        if (i == 0) {
            d.argument_schema_json = R"({"type":"object","properties":{},"additionalProperties":false})";
        } else {
            auto& p = parameters[i];
            p.name = i == 5 ? "request_id" : "panel_id";
            p.type = i == 5 ? "integer" : "string"; p.required = 1;
            p.description = i == 5 ? "Positive request ID returned to this plugin instance" : "Exact stable panel key from ui.list";
            d.parameters = &p; d.parameter_count = 1;
            d.argument_schema_json = i == 5
                ? R"({"type":"object","properties":{"request_id":{"type":"integer","minimum":1}},"required":["request_id"],"additionalProperties":false})"
                : R"({"type":"object","properties":{"panel_id":{"type":"string","minLength":1,"maxLength":256}},"required":["panel_id"],"additionalProperties":false})";
        }
    }
    return api->register_tools(api, tools, 6);
}
extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() { host = nullptr; }
