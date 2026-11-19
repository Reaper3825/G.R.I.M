#include "core/plugin_tool_result.hpp"
#include <string_view>
#include "core/plugin_api.hpp"
#include "input_commands.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <stdexcept>
#include <mutex>

namespace {
enum GrimInputOperation { GRIM_INPUT_MOVE_MOUSE, GRIM_INPUT_CLICK, GRIM_INPUT_TYPE, GRIM_INPUT_KEY };
std::mutex operationMutex;
grim::PluginHost host = {};
template<GrimInputOperation Operation>
GrimResult call(const GrimToolCall* request, GrimToolResultWriter* writer, void*) {
    std::lock_guard<std::mutex> lock(operationMutex);
    CommandResult value;
    if (!host || !host->has_permission(host.instance(), GRIM_PERM_INPUT))
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
    return grim::completeToolResult(writer, value);
}
}

extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* table, GrimPluginHandle instance) {
    const grim::PluginHost api{table, instance};
    if (!api || api->api_version < GRIM_API_VERSION || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR ||
        !api->has_permission) return GRIM_ERROR_API_VERSION_MISMATCH;
    host = api;
    return GRIM_OK;
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_dispatch_tool(
    const char* tool_id, const GrimToolCall* request, GrimToolResultWriter* writer) {
    if (!tool_id) return GRIM_ERROR_INVALID_PARAM;
    const std::string_view id(tool_id);
    if (id == "move_mouse") return call<GRIM_INPUT_MOVE_MOUSE>(request, writer, nullptr);
    if (id == "click") return call<GRIM_INPUT_CLICK>(request, writer, nullptr);
    if (id == "type") return call<GRIM_INPUT_TYPE>(request, writer, nullptr);
    if (id == "key") return call<GRIM_INPUT_KEY>(request, writer, nullptr);
    return GRIM_ERROR_NOT_FOUND;
}

extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() { host = {}; }
