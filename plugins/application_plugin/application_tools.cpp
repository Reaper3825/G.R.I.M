#include "application_tools.hpp"
#include <nlohmann/json.hpp>
#include <string>

GrimResult dispatchApplicationTool(std::string_view id, const GrimToolCall* call,
                                   GrimToolResultWriter* writer, GRIM::Applications::Catalog& catalog) {
    if (id != "applications.list" && id != "applications.refresh" &&
        id != "applications.inspect" && id != "applications.launch")
        return GRIM_ERROR_NOT_FOUND;
    const std::string operation(id.substr(std::string_view("applications.").size()));
    GrimToolResult result;
    std::string message, output;
    try {
        if (!call || call->arguments_size > 65536 || (call->arguments_size && !call->arguments))
            throw std::invalid_argument("Invalid application arguments");
        const auto args = nlohmann::json::parse(std::string(call->arguments ? call->arguments : "", call->arguments_size));
        const auto value = nlohmann::json{{"format", "grim-application-result"}, {"version", 1},
            {"success", true}, {"value", catalog.request(operation, args)}};
        result.success = value.at("success").get<bool>();
        output = value.dump();
        if (!result.success) message = value.value("error", "Application operation failed");
        else if (operation == "list" || operation == "refresh") {
            const auto& catalog = value.at("value");
            message = "Applications: " + std::to_string(catalog.at("applications").size()) + " of " + catalog.at("total").dump();
            if (!catalog.at("next_offset").is_null()) message += "; next applications.list offset: " + catalog.at("next_offset").dump();
            if (!catalog.at("complete").get<bool>()) message += " (discovery incomplete; see backend errors)";
            for (const auto& app : catalog.at("applications"))
                message += "\n" + app.at("name").get<std::string>() + " (" + app.at("resource_id").get<std::string>() + ")";
        } else message = operation == "launch" ? "Application launch requested." : value.at("value").at("application").dump(2);
    } catch (const std::exception& e) {
        result.success = false; message = e.what();
        output = nlohmann::json{{"format", "grim-application-result"}, {"version", 1},
            {"success", false}, {"error", message}}.dump();
    }
    result.message = message.c_str(); result.structured_output_json = output.empty() ? nullptr : output.c_str();
    result.error_code = result.success ? "ERR_NONE" : "ERR_APPLICATION_SERVICE";
    result.category = result.success ? "application" : "error";
    return writer->complete(writer, &result);
}
