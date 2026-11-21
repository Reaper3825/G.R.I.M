#include "core/plugin_tool_result.hpp"
#include "application_tools.hpp"
#include <nlohmann/json.hpp>
#include <string>

GrimToolReply dispatchApplicationTool(std::string_view id, const GrimToolCall* call, GRIM::Applications::Catalog& catalog) {
    if (id != "applications.list" && id != "applications.refresh" &&
        id != "applications.inspect" && id != "applications.launch")
        return {GRIM_ERROR_NOT_FOUND};
    const std::string operation(id.substr(std::string_view("applications.").size()));
    bool success = false;
    std::string message;
    auto payload = nlohmann::json::object();
    try {
        if (!call || call->arguments_size > 65536 || (call->arguments_size && !call->arguments))
            throw std::invalid_argument("Invalid application arguments");
        const auto args = nlohmann::json::parse(std::string(call->arguments ? call->arguments : "", call->arguments_size));
        payload = nlohmann::json{{"format", "grim-application-result"}, {"version", 2},
            {"success", true}, {"value", catalog.request(operation, args)}};
        success = payload.at("success").get<bool>();
        if (!success) message = payload.value("error", "Application operation failed");
        else if (operation == "list" || operation == "refresh") {
            const auto& catalog = payload.at("value");
            message = "Applications: " + std::to_string(catalog.at("applications").size()) + " of " + catalog.at("total").dump();
            if (!catalog.at("next_offset").is_null()) message += "; next applications.list offset: " + catalog.at("next_offset").dump();
            if (!catalog.at("complete").get<bool>()) message += " (discovery incomplete; see backend errors)";
            for (const auto& app : catalog.at("applications"))
                message += "\n" + app.at("name").get<std::string>() + " (" + app.at("resource_id").get<std::string>() + ")";
        } else message = operation == "launch" ? "Application launch requested." : payload.at("value").at("application").dump(2);
    } catch (const std::exception& e) {
        success = false; message = e.what();
        payload = nlohmann::json{{"format", "grim-application-result"}, {"version", 2},
            {"success", false}, {"error", message}};
    }
    return grim::toolReply(success, message.c_str(),
                                    success ? "ERR_NONE" : "ERR_APPLICATION_SERVICE",
                                    success ? "application" : "error", std::move(payload));
}
