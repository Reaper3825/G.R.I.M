#include "application_tools.hpp"
#include <nlohmann/json.hpp>
#include <string>

namespace {
struct Context { GRIM::Applications::Catalog* catalog = nullptr; const char* operation = nullptr; };
Context contexts[4];
GrimResult invoke(const GrimToolCall* call, GrimToolResultWriter* writer, void* context) {
    const auto& selected = *static_cast<Context*>(context);
    const std::string operation = selected.operation;
    GrimToolResult result;
    std::string message, output;
    try {
        if (!call || call->arguments_size > 65536 || (call->arguments_size && !call->arguments))
            throw std::invalid_argument("Invalid application arguments");
        const auto args = nlohmann::json::parse(std::string(call->arguments ? call->arguments : "", call->arguments_size));
        const auto value = nlohmann::json{{"format", "grim-application-result"}, {"version", 1},
            {"success", true}, {"value", selected.catalog->request(operation, args)}};
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
}
GrimResult registerApplicationTools(const GrimPluginAPI* api, GRIM::Applications::Catalog& catalog) {
    GrimToolRegistration tools[4];
    const char* ids[] = {"applications.list", "applications.refresh", "applications.inspect", "applications.launch"};
    const char* operations[] = {"list", "refresh", "inspect", "launch"};
    const char* descriptions[] = {"List application resources discovered on this system", "Refresh application discovery and live resources",
        "Inspect an application resource and its launch identity", "Launch a discovered application using its current generation"};
    const char* usages[] = {"applications.list", "applications.refresh", "applications.inspect {t}", "applications.launch {t}"};
    const GrimToolBinding binding[] = {{"t", "t"}};
    GrimToolParameter parameters[3];
    GrimToolParameter pagination[2];
    pagination[0].name = "offset"; pagination[1].name = "limit";
    for (auto& parameter : pagination) { parameter.type = "integer"; parameter.description = "Optional catalog page offset or maximum entries (1-50)"; }
    const char* names[] = {"resource_id", "provider_instance", "runtime_id"};
    for (size_t i = 0; i < 3; ++i) {
        parameters[i].name = names[i]; parameters[i].type = i == 2 ? "integer" : "string";
        parameters[i].description = "Exact identity from the current discovered application resource";
        parameters[i].required = 1;
    }
    for (size_t i = 0; i < 4; ++i) {
        auto& t = tools[i]; auto& d = t.descriptor;
        contexts[i] = {&catalog, operations[i]};
        t.handler = invoke; t.user_data = &contexts[i]; t.short_description = descriptions[i];
        d.tool_id = d.display_name = ids[i]; d.version = "1.0.0"; d.description = descriptions[i];
        d.usage = usages[i]; d.category = "application"; d.argument_format = "json";
        d.required_permissions = GRIM_PERM_SYSTEM | GRIM_PERM_FILESYSTEM | (i == 3 ? GRIM_PERM_PROCESS : 0);
        d.is_informational = i != 3; d.needs_confirmation = i == 3;
        if (i == 0) {
            d.parameters = pagination; d.parameter_count = 2;
            d.argument_schema_json = R"({"type":"object","properties":{"offset":{"type":"integer","minimum":0},"limit":{"type":"integer","minimum":1,"maximum":50}},"additionalProperties":false})";
        }
        else if (i == 1) d.argument_schema_json = R"({"type":"object","properties":{},"additionalProperties":false})";
        else {
            t.bindings = binding; t.binding_count = 1;
            d.parameters = parameters; d.parameter_count = i == 3 ? 3 : 1;
            d.argument_schema_json = i == 2
                ? R"({"type":"object","properties":{"resource_id":{"type":"string"}},"required":["resource_id"],"additionalProperties":false})"
                : R"({"type":"object","properties":{"resource_id":{"type":"string"},"provider_instance":{"type":"string"},"runtime_id":{"type":"integer","minimum":1}},"required":["resource_id","provider_instance","runtime_id"],"additionalProperties":false})";
        }
    }
    return api->register_tools(api, tools, 4);
}
