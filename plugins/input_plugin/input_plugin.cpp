#include "core/plugin_api.hpp"
#include "input_commands.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <stdexcept>

namespace {
const GrimPluginAPI* host = nullptr;
struct UnknownInputException {};
template<GrimInputOperation Operation>
GrimResult call(const GrimToolCall* request, GrimToolResultWriter* writer, void*) {
    if constexpr (Operation != GRIM_INPUT_CLICK_ON) {
        CommandResult value;
        if (request->arguments_size > 65536 || (request->arguments_size && !request->arguments)) {
            value = {false, "Host input service unavailable or denied", "ERR_INPUT_SERVICE", "error", "", {255,100,100}};
        } else try {
            const std::string argument = request->arguments_size ? std::string(request->arguments, request->arguments_size) : std::string{};
            if constexpr (Operation == GRIM_INPUT_MOVE_MOUSE) value = inputCommands::cmdInputMoveMouse(argument);
            if constexpr (Operation == GRIM_INPUT_CLICK) value = inputCommands::cmdInputClick(argument);
            if constexpr (Operation == GRIM_INPUT_TYPE) value = inputCommands::cmdInputType(argument);
            if constexpr (Operation == GRIM_INPUT_KEY) value = inputCommands::cmdInputKey(argument);
        } catch (const inputCommands::ServiceUnavailable&) {
            value = {false, "Host input service unavailable or denied", "ERR_INPUT_SERVICE", "error", "", {255,100,100}};
        }
        GrimToolResult result;
        result.success = value.success; result.message = value.message.c_str();
        result.error_code = value.errorCode.c_str(); result.category = value.category.c_str(); result.voice = value.voice.c_str();
        result.color_rgb = (value.color.r << 16) | (value.color.g << 8) | value.color.b;
        return writer->complete(writer, &result);
    }
    const char* raw = host->input_service(host, Operation, request->arguments, request->arguments_size);
    if (!raw) {
        GrimToolResult result;
        result.message = "Host input service unavailable or denied";
        result.error_code = "ERR_INPUT_SERVICE"; result.category = "error"; result.color_rgb = 0xFF6464;
        return writer->complete(writer, &result);
    }
    std::string text;
    { struct Release { const char* raw; ~Release() { host->string_free(host, raw); } } release{raw}; text = raw; }
    const auto response = nlohmann::json::parse(text);
    if (response.at("format") != "grim-input-result" || response.at("version") != 1)
        return GRIM_ERROR_UNSUPPORTED;
    if (response.contains("exception")) {
        if (response.at("exception").is_string()) throw std::runtime_error(response.at("exception").get<std::string>());
        throw UnknownInputException{};
    }
    const auto& value = response.at("result");
    const auto message = value.at("message").get<std::string>();
    const auto error = value.at("error_code").get<std::string>();
    const auto category = value.at("category").get<std::string>();
    const auto voice = value.at("voice").get<std::string>();
    const auto structured = value.at("structured_output_json").get<std::string>();
    GrimToolResult result;
    result.success = value.at("success").get<bool>(); result.message = message.c_str();
    result.error_code = error.c_str(); result.category = category.c_str(); result.voice = voice.c_str();
    result.structured_output_json = structured.empty() ? nullptr : structured.c_str();
    result.color_rgb = value.at("color_rgb").get<uint32_t>();
    return writer->complete(writer, &result);
}
}

extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    static const GrimPluginInfo info{GRIM_API_VERSION, "Input Plugin", "1.0.0", "GRIM Project",
        "Mouse, keyboard and grounded target input through host-owned input services", GRIM_PERM_INPUT | GRIM_PERM_SYSTEM};
    return &info;
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || api->api_version < GRIM_API_VERSION || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR ||
        !api->input_service || !api->input_primitive) return GRIM_ERROR_API_VERSION_MISMATCH;
    host = api;
    inputCommands::host = api;
    GrimToolRegistration tools[5];
    tools[0].handler = call<GRIM_INPUT_MOVE_MOUSE>; tools[1].handler = call<GRIM_INPUT_CLICK>;
    tools[2].handler = call<GRIM_INPUT_TYPE>; tools[3].handler = call<GRIM_INPUT_KEY>;
    tools[4].handler = call<GRIM_INPUT_CLICK_ON>;
    const char* ids[] = {"move_mouse", "click", "type", "key", "click_on"};
    const char* descriptions[] = {"Move the pointer to physical desktop coordinates using the existing host input interface",
        "Click a mouse button at the current pointer position using the existing host input interface",
        "Type raw text into the focused application using the existing host input interface",
        "Tap a key or plus-separated key combination using the existing host input interface",
        "Send checked click input to a unique exact OCR label or native UI control in the active window"};
    const char* usages[] = {"move_mouse <x> <y>", "click [left|right|middle]", "type <text>",
        "key <key|key+key>", "click_on {t}"};
    const GrimToolBinding targetBindings[] = {{"t", "t"}};
    const char* summaries[] = {"Move the pointer to specified desktop coordinates",
        "Click a mouse button at the current position",
        "Type supplied text into the focused application",
        "Press a key or combination of keys",
        "Click a uniquely identified visible interface element"};
    tools[4].bindings = targetBindings; tools[4].binding_count = 1;
    const char* names[] = {"coordinates", "button", "text", "keys", "target"};
    const char* parameterDescriptions[] = {"Two signed integers, comma- or whitespace-separated; uses existing integer-prefix parsing",
        "Button name; omitted means left; retains existing host interpretation",
        "Raw text, including surrounding whitespace; host types with a 15 ms delay",
        "Host key name, or a plus-separated combination; retains existing host key handling",
        "Exact OCR region text (default/text prefix), or native control name, automation ID, or role (object prefix)"};
    const char* grounded[] = {"Local Windows foreground window and fresh OCR/native control bounds"};
    const char* tags[] = {"input", "computer_control"};
    GrimToolParameter parameters[5];
    for (size_t i = 0; i < 5; ++i) {
        tools[i].usage_type = GRIM_TOOL_USAGE_DEBUG;
        tools[i].short_description = summaries[i];
        auto& d = tools[i].descriptor;
        d.tool_id = d.display_name = ids[i]; d.version = "1.0.0"; d.description = descriptions[i];
        d.usage = usages[i]; d.category = "action"; d.argument_format = "raw_text";
        d.required_permissions = GRIM_PERM_INPUT | (i == 4 ? GRIM_PERM_SYSTEM : 0);
        d.is_informational = 0; d.needs_confirmation = 1;
        parameters[i].name = names[i]; parameters[i].type = "string";
        parameters[i].description = parameterDescriptions[i]; parameters[i].required = i != 1;
        d.parameters = &parameters[i]; d.parameter_count = 1;
        d.capability_tags = {tags, 2};
        if (i == 4) d.preconditions = {grounded, 1};
    }
    return api->register_tools(api, tools, 5);
}
extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() { host = nullptr; inputCommands::host = nullptr; }
