#include "core/plugin_api.hpp"
#include <string>
#include <stdexcept>

namespace {
const GrimPluginAPI* host = nullptr;

GrimResult finish(GrimToolResultWriter* writer, bool success, const std::string& message,
                  const char* error, const char* category, const std::string& voice, uint32_t color) {
    GrimToolResult result;
    result.success = success;
    result.message = message.c_str();
    result.error_code = error;
    result.category = category;
    result.voice = voice.c_str();
    result.color_rgb = color;
    return writer->complete(writer, &result);
}
std::string errorMessage(const char* code) {
    const auto* text = host->get_user_error_message(host, code);
    if (!text) throw std::runtime_error("Host error-message lookup failed");
    const std::string result(text);
    host->string_free(host, text);
    return result;
}
std::string argument(const GrimToolCall* call) {
    return std::string(call->arguments, call->arguments_size);
}
GrimResult openApplication(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    // Preserve open's original untrimmed argument semantics.
    const auto target = argument(call);
    if (target.empty()) return finish(writer, false, errorMessage("ERR_APP_NO_ARGUMENT"),
        "ERR_APP_NO_ARGUMENT", "error", "Missing application name", 0xFF5050);
#ifdef _WIN32
    int64_t nativeStatus = 0;
    if (host->open_target(host, target.c_str(), &nativeStatus) != GRIM_OK || nativeStatus <= 32)
        return finish(writer, false, errorMessage("ERR_APP_LAUNCH_FAILED") + ": " + target,
            "ERR_APP_LAUNCH_FAILED", "error", "Failed to open " + target, 0xFF5050);
    return finish(writer, true, "[App] Launched: " + target,
        "ERR_NONE", "routine", "Opening " + target, 0x64FF64);
#else
    return finish(writer, true, "[App] (Stub) Would open: " + target,
        "ERR_NONE", "routine", "Opening " + target, 0x64FF64);
#endif
}
GrimResult searchWeb(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    const auto raw = argument(call);
    const auto start = raw.find_first_not_of(" \t\n\r");
    const auto query = start == std::string::npos ? std::string{} :
        raw.substr(start, raw.find_last_not_of(" \t\n\r") - start + 1);
    if (query.empty()) return finish(writer, false, errorMessage("ERR_WEB_NO_ARGUMENT"),
        "ERR_WEB_NO_ARGUMENT", "error", "No search query", 0xFF5050);
#ifdef _WIN32
    // Preserve the original URL construction; no new encoding or query rewriting.
    const auto url = "https://www.google.com/search?q=" + query;
    int64_t nativeStatus = 0;
    if (host->open_target(host, url.c_str(), &nativeStatus) != GRIM_OK || nativeStatus <= 32)
        return finish(writer, false, errorMessage("ERR_WEB_OPEN_FAILED") + ": " + query,
            "ERR_WEB_OPEN_FAILED", "error", "Web search failed", 0xFF5050);
    return finish(writer, true, "[Web] Searching: " + query,
        "ERR_NONE", "routine", "Searching web for " + query, 0x64FFFF);
#else
    return finish(writer, true, "[Web] (Stub) Would search for: " + query,
        "ERR_NONE", "routine", "Searching web for " + query, 0x64FFFF);
#endif
}
}

extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    static const GrimPluginInfo info{GRIM_API_VERSION, "Application Plugin", "1.0.0",
        "GRIM Project", "Application/file opening and browser search", GRIM_PERM_PROCESS};
    return &info;
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION ||
        !api->get_user_error_message || !api->open_target) return GRIM_ERROR_API_VERSION_MISMATCH;
    host = api;
    GrimToolRegistration tools[2];
    tools[0].handler = openApplication;
    tools[1].handler = searchWeb;
    const char* ids[] = {"open", "search"};
    const char* descriptions[] = {"Open an application or file", "Search the web for information"};
    const char* usages[] = {"open <application-or-file>", "search <query>"};
    const char* examples[] = {"open notepad", "search weather tomorrow"};
    GrimToolParameter parameters[2];
    for (size_t i = 0; i < 2; ++i) {
        auto& d = tools[i].descriptor;
        d.tool_id = d.display_name = ids[i];
        d.version = "1.0.0";
        d.description = descriptions[i];
        d.usage = usages[i];
        d.category = i == 0 ? "action" : "information";
        d.argument_format = "raw_text";
        d.required_permissions = GRIM_PERM_PROCESS;
        d.is_informational = i == 1;
        d.needs_confirmation = i == 0;
        d.examples = {&examples[i], 1};
        parameters[i].name = i == 0 ? "target" : "query";
        parameters[i].type = "string";
        parameters[i].description = i == 0 ? "Application or file passed unchanged to the OS opener" :
            "Search text; surrounding whitespace is trimmed before opening the original Google search URL";
        parameters[i].required = 1;
        d.parameters = &parameters[i];
        d.parameter_count = 1;
    }
    return api->register_tools(api, tools, 2);
}
extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() { host = nullptr; }
