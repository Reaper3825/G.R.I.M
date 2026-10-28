// Exercise the real API implementation and registry with host service sinks.
#include "core/plugin_api_impl.hpp"
#include "commands/commands_core.hpp"
#include "MMO/Core/HardwareInventory.hpp"
#include "MMO/Core/ToolRegistry.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

GRIM::MMO::HardwareInventory g_hardwareInventory;
static std::vector<std::string> logs;
void logDebug(const std::string&, const std::string& message) { logs.push_back(message); }
void logError(const std::string&, const std::string& message) { logs.push_back(message); }
void notifyPopupActivity() {}
void showPopup() {}
namespace Voice {
void speak(const std::string&, const std::string&) {}
bool isSpeaking() { return false; }
}
std::string normalizeWord(const std::string& word) { return word; }
namespace GRIMInput {
std::pair<std::string, std::string> parseInput(const std::string& text) {
    const auto space = text.find(' ');
    return {text.substr(0, space), space == std::string::npos ? "" : text.substr(space + 1)};
}
}
void ensureCorePluginsRegistered() {}
CommandResult cmdOpenApp(const std::string&) { return CommandResult{}; }

namespace {
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
GrimCommandResult identify(const char*, void* data) {
    const auto* api = static_cast<const GrimPluginAPI*>(data);
    api->log(api, GRIM_LOG_INFO, "callback");
    return {api->has_permission(api, GRIM_PERM_MEMORY), "ok", "", "test", 0};
}
void observe(const char*, const void*, size_t, void* data) {
    const auto* api = static_cast<const GrimPluginAPI*>(data);
    api->log(api, GRIM_LOG_INFO, "event");
}
void tick(uint64_t, void* data) { ++*static_cast<int*>(data); }
void checkName(const GrimPluginAPI* api, const char* expected) {
    const auto* name = api->get_plugin_name(api);
    check(std::string(name) == expected, "Plugin identity changed after another load");
    api->string_free(api, name);
}
}

int main() try {
    auto* alpha = PluginAPI::createPluginAPI("alpha", "alpha.dll", GRIM_PERM_MEMORY);
    check(alpha->register_command(alpha, "alpha_tool", identify, alpha) == GRIM_OK, "Alpha registration failed");
    auto* beta = PluginAPI::createPluginAPI("beta", "beta.dll", GRIM_PERM_UI);
    check(alpha != beta, "Plugins share an API table");
    checkName(alpha, "alpha");
    checkName(beta, "beta");
    check(alpha->has_permission(alpha, GRIM_PERM_MEMORY), "Alpha lost permissions");
    check(!beta->has_permission(beta, GRIM_PERM_MEMORY), "Beta inherited Alpha permissions");
    check(!alpha->has_permission(alpha, static_cast<GrimPermission>(GRIM_PERM_MEMORY | GRIM_PERM_UI)), "Combined permissions partially accepted");
    check(beta->store_memory(beta, "key", "value", "test") == GRIM_ERROR_PERMISSION_DENIED, "API used another owner's permissions");
    check(dispatchCommand("alpha_tool", "").success, "Command callback used wrong owner");
    check(logs.back() == "[Plugin: alpha] callback", "Callback logging used wrong owner");
    check(alpha->register_command(alpha, "alpha_later", identify, alpha) == GRIM_OK, "Later registration failed");
    auto tool = GRIM::MMO::ToolRegistry::instance().getTool("alpha_later");
    check(tool && tool->provider_name == "alpha" && tool->permission_bits == GRIM_PERM_MEMORY, "Later registration used last loaded plugin");
    check(beta->unregister_command(beta, "alpha_tool") == GRIM_ERROR_PERMISSION_DENIED, "Plugin removed another owner's tool");
    GRIM::MMO::ToolRegistry::instance().registerSimple("builtin", "Builtin test", "test",
        [](const std::string&) { return CommandResult{}; });
    check(beta->register_command(beta, "builtin", identify, beta) == GRIM_ERROR_ALREADY_EXISTS, "Plugin overwrote builtin");
    check(beta->register_command(beta, "", identify, beta) == GRIM_ERROR_INVALID_PARAM, "Empty tool name accepted");

    auto event = alpha->subscribe_event(alpha, "test", observe, alpha);
    check(beta->unsubscribe_event(beta, event) == GRIM_ERROR_PERMISSION_DENIED, "Plugin removed another owner's event");
    beta->emit_event(beta, "test", nullptr, 0);
    check(logs.back() == "[Plugin: alpha] event", "Event callback used wrong owner");
    int alphaTicks = 0, betaTicks = 0;
    auto timer = alpha->create_timer(alpha, 0, false, tick, &alphaTicks);
    check(beta->cancel_timer(beta, timer) == GRIM_ERROR_PERMISSION_DENIED, "Plugin cancelled another owner's timer");
    beta->create_timer(beta, 0, false, tick, &betaTicks);
    bool duplicateRejected = false;
    try { PluginAPI::createPluginAPI("alpha", "duplicate.dll", 0); }
    catch (const std::invalid_argument&) { duplicateRejected = true; }
    check(duplicateRejected, "Duplicate identity replaced a retained API");

    PluginAPI::cleanupPluginContext("alpha");
    check(!GRIM::MMO::ToolRegistry::instance().isRegistered("alpha_tool") && !GRIM::MMO::ToolRegistry::instance().getTool("alpha_later"), "Owner cleanup retained tools");
    PluginAPI::processTimers();
    check(alphaTicks == 0 && betaTicks == 1, "Owner cleanup retained timer or removed another owner's timer");
    checkName(beta, "beta");
    const auto before = logs.size();
    beta->emit_event(beta, "test", nullptr, 0);
    check(logs.size() == before, "Owner cleanup retained event callback");
    alpha = PluginAPI::createPluginAPI("alpha", "new-alpha.dll", 0);
    checkName(alpha, "alpha");
    check(!alpha->has_permission(alpha, GRIM_PERM_MEMORY), "Reload inherited old permissions");
    PluginAPI::cleanupPluginContext("alpha");
    PluginAPI::cleanupPluginContext("beta");
    std::cout << "Plugin API ownership tests passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
