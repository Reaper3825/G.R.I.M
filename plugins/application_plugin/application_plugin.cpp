#include "core/plugin_api.hpp"
#include "core/plugin_tool_result.hpp"
#include "application_tools.hpp"
#include "shell_open.hpp"
#include "user_aliases.hpp"
#include <memory>
#include <string>
#include <stdexcept>

namespace {
grim::PluginHost host = {};
std::unique_ptr<GRIM::Applications::Catalog> catalog;
std::shared_ptr<GRIM::Applications::UserAliases> userAliases;

std::string errorMessage(const char* code) {
    const auto* text = host->get_user_error_message(host.instance(), code);
    if (!text) throw std::runtime_error("Host error-message lookup failed");
    const std::string result(text);
    host->string_free(host.instance(), text);
    return result;
}
std::string argument(const GrimToolCall* call) {
    return std::string(call->arguments, call->arguments_size);
}
GrimToolReply openApplication(const GrimToolCall* call, void*) {
    // Preserve open's original untrimmed argument semantics.
    const auto target = argument(call);
    if (target.empty()) return grim::toolReply(false, errorMessage("ERR_APP_NO_ARGUMENT"),
        "ERR_APP_NO_ARGUMENT", "error");
#ifdef _WIN32
    if (shellOpenTarget(userAliases->resolve(target)) <= 32)
        return grim::toolReply(false, errorMessage("ERR_APP_LAUNCH_FAILED") + ": " + target,
            "ERR_APP_LAUNCH_FAILED", "error");
    return grim::toolReply(true, "[App] Launched: " + target,
        "ERR_NONE", "routine");
#else
    return grim::toolReply(true, "[App] (Stub) Would open: " + target,
        "ERR_NONE", "routine");
#endif
}
GrimToolReply searchWeb(const GrimToolCall* call, void*) {
    const auto raw = argument(call);
    const auto start = raw.find_first_not_of(" \t\n\r");
    const auto query = start == std::string::npos ? std::string{} :
        raw.substr(start, raw.find_last_not_of(" \t\n\r") - start + 1);
    if (query.empty()) return grim::toolReply(false, errorMessage("ERR_WEB_NO_ARGUMENT"),
        "ERR_WEB_NO_ARGUMENT", "error");
#ifdef _WIN32
    // Preserve the original URL construction; no new encoding or query rewriting.
    const auto url = "https://www.google.com/search?q=" + query;
    if (shellOpenTarget(url) <= 32)
        return grim::toolReply(false, errorMessage("ERR_WEB_OPEN_FAILED") + ": " + query,
            "ERR_WEB_OPEN_FAILED", "error");
    return grim::toolReply(true, "[Web] Searching: " + query,
        "ERR_NONE", "routine");
#else
    return grim::toolReply(true, "[Web] (Stub) Would search for: " + query,
        "ERR_NONE", "routine");
#endif
}
}

extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* table, GrimPluginHandle instance) {
    const grim::PluginHost api{table, instance};
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION ||
        !api->get_user_error_message || !api->replace_resources) return GRIM_ERROR_API_VERSION_MISMATCH;
    host = api;
    try {
        userAliases = std::make_shared<GRIM::Applications::UserAliases>(GRIM::Applications::UserAliases::defaultDirectory());
        auto backends = GRIM::Applications::systemBackends([store=userAliases] { return store->reload(); });
        catalog = std::make_unique<GRIM::Applications::Catalog>(std::move(backends),
            GRIM::Applications::launchNative,
            [api](const auto& resources) {
                const auto json = nlohmann::json(resources).dump();
                return api->replace_resources(api.instance(), "application", json.data(), json.size()) == GRIM_OK;
            });
        // Initial discovery publishes under the loaded provider.
        catalog->request("refresh", nlohmann::json::object());
    } catch (...) { catalog.reset(); userAliases.reset(); return GRIM_ERROR_UNKNOWN; }

    return GRIM_OK;
}

extern "C" PLUGIN_EXPORT GrimToolReply grim_plugin_dispatch_tool(
    const char* tool_id, const GrimToolCall* call) {
    const std::string_view id(tool_id);
    if (id == "open") return openApplication(call, nullptr);
    if (id == "search") return searchWeb(call, nullptr);
    if (!catalog) return {GRIM_ERROR_PLUGIN_NOT_LOADED};
    return dispatchApplicationTool(id, call, *catalog);
}

extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() { catalog.reset(); userAliases.reset(); host = {}; }
