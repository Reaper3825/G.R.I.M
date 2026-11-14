#include "core/plugin_api.hpp"
#include "input_commands.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <stdexcept>
#include <mutex>

namespace {
enum GrimInputOperation { GRIM_INPUT_MOVE_MOUSE, GRIM_INPUT_CLICK, GRIM_INPUT_TYPE, GRIM_INPUT_KEY };
std::mutex operationMutex;
const GrimPluginAPI* host = nullptr;
template<GrimInputOperation Operation>
GrimResult call(const GrimToolCall* request, GrimToolResultWriter* writer, void*) {
    std::lock_guard<std::mutex> lock(operationMutex);
    CommandResult value;
    if (!host || !host->has_permission(host, GRIM_PERM_INPUT) || !host->is_provider_current(host))
        return GRIM_ERROR_PERMISSION_DENIED;
    if (request->arguments_size > 65536 || (request->arguments_size && !request->arguments)) {
        value = {false, "Input request unavailable or invalid", "ERR_INPUT_SERVICE", "error", "", {255,100,100}};
    } else try {
        const std::string argument = request->arguments_size ? std::string(request->arguments, request->arguments_size) : std::string{};
        if constexpr (Operation == GRIM_INPUT_MOVE_MOUSE) value = inputCommands::cmdInputMoveMouse(argument);
        if constexpr (Operation == GRIM_INPUT_CLICK) value = inputCommands::cmdInputClick(argument);
        if constexpr (Operation == GRIM_INPUT_TYPE) value = inputCommands::cmdInputType(argument);
        if constexpr (Operation == GRIM_INPUT_KEY) value = inputCommands::cmdInputKey(argument);
    } catch (const inputCommands::ServiceUnavailable&) {
        value = {false, "Input request unavailable or invalid", "ERR_INPUT_SERVICE", "error", "", {255,100,100}};
    }
    GrimToolResult result;
    result.success = value.success; result.message = value.message.c_str();
    result.error_code = value.errorCode.c_str(); result.category = value.category.c_str(); result.voice = value.voice.c_str();
    result.color_rgb = (value.color.r << 16) | (value.color.g << 8) | value.color.b;
    return writer->complete(writer, &result);
}
}

extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    static const GrimPluginInfo info{GRIM_API_VERSION, "Input Plugin", "1.0.0", "GRIM Project",
        "Mouse and keyboard input through provider-owned device primitives", GRIM_PERM_INPUT};
    return &info;
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || api->api_version < GRIM_API_VERSION || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR ||
        !api->has_permission || !api->is_provider_current) return GRIM_ERROR_API_VERSION_MISMATCH;
    host = api;
    GrimToolRegistration tools[4];
    tools[0].handler = call<GRIM_INPUT_MOVE_MOUSE>; tools[1].handler = call<GRIM_INPUT_CLICK>;
    tools[2].handler = call<GRIM_INPUT_TYPE>; tools[3].handler = call<GRIM_INPUT_KEY>;
    const char* ids[] = {"move_mouse", "click", "type", "key"};
    const char* descriptions[] = {"Move the pointer to physical desktop coordinates using the provider-owned native backend",
        "Click a mouse button at the current pointer position using the provider-owned native backend",
        "Type raw text into the focused application using the provider-owned native backend",
        "Tap a key or plus-separated key combination using the provider-owned native backend"};
    const char* usages[] = {"move_mouse <x> <y>", "click [left|right|middle]", "type <text>",
        "key <key|key+key>"};
    const char* summaries[] = {"Move the pointer to specified desktop coordinates",
        "Click a mouse button at the current position",
        "Type supplied text into the focused application",
        "Press a key or combination of keys"};
    const char* names[] = {"coordinates", "button", "text", "keys"};
    const char* parameterDescriptions[] = {"Two signed integers, comma- or whitespace-separated; uses existing integer-prefix parsing",
        "Button name; omitted means left; retains existing native interpretation",
        "Raw text, including surrounding whitespace; provider types with a 15 ms delay",
        "Native key name, or a plus-separated combination; retains existing native key handling"};
    const char* tags[] = {"input", "computer_control"};
    GrimToolParameter parameters[4];
    for (size_t i = 0; i < 4; ++i) {
        tools[i].usage_type = GRIM_TOOL_USAGE_DEBUG;
        tools[i].short_description = summaries[i];
        auto& d = tools[i].descriptor;
        d.tool_id = d.display_name = ids[i]; d.version = "1.0.0"; d.description = descriptions[i];
        d.usage = usages[i]; d.category = "action"; d.argument_format = "raw_text";
        d.required_permissions = GRIM_PERM_INPUT;
        d.is_informational = 0; d.needs_confirmation = 1;
        parameters[i].name = names[i]; parameters[i].type = "string";
        parameters[i].description = parameterDescriptions[i]; parameters[i].required = i != 1;
        d.parameters = &parameters[i]; d.parameter_count = 1;
        d.capability_tags = {tags, 2};
    }
    return api->register_tools(api, tools, 4);
}
extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() { host = nullptr; }
