#include "core/plugin_manager.hpp"
#include "core/plugin_api_impl.hpp"
#include "core/shell_open.hpp"
#include "MMO/Core/ToolRegistry.hpp"
#include "error_manager.hpp"
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <stdexcept>

// Only the OS primitive and configurable error catalog are replaced. The DLL,
// loader, API permission/ownership handling, registry, and result writer are real.
namespace {
std::string lastTarget, errorPrefix = "configured:";
int64_t nativeStatus = 33;
int launches = 0;
}
int64_t shellOpenTarget(const std::string& target) { lastTarget = target; ++launches; return nativeStatus; }
std::string ErrorManager::getUserMessage(const std::string& code) { return errorPrefix + code; }

// v3.1 consumes two trailing reserved pointers, preserving v3.0 table size.
static_assert(sizeof(GrimPluginAPI) == offsetof(GrimPluginAPI, get_user_error_message) + 32 * sizeof(void*));

void runApplicationPluginTests(const std::filesystem::path& pluginDirectory) {
    namespace fs = std::filesystem;
    const auto check = [](bool ok, const char* message) { if (!ok) throw std::runtime_error(message); };
    auto& registry = GRIM::MMO::ToolRegistry::instance();
    const auto baseline = registry.toolCount();
    auto* denied = PluginAPI::createPluginAPI("launch-denied", "", 0);
    int64_t status = -1;
    const auto beforeDenied = launches;
    check(denied->open_target(denied, "fixture", &status) == GRIM_ERROR_PERMISSION_DENIED && status == 0 && launches == beforeDenied,
          "open_target bypassed PROCESS permission");
    check(denied->launch_app(denied, "fixture") == GRIM_ERROR_PERMISSION_DENIED, "Legacy launch permission changed");
    const char* error = denied->get_user_error_message(denied, "ERR_APP_NO_ARGUMENT");
    check(error && std::string(error) == "configured:ERR_APP_NO_ARGUMENT", "Host error lookup not copied");
    denied->string_free(denied, error);
    check(denied->get_user_error_message(denied, nullptr) == nullptr, "Null error code accepted");
    PluginAPI::cleanupPluginContext(denied);
    auto* allowed = PluginAPI::createPluginAPI("launch-allowed", "", GRIM_PERM_PROCESS);
    check(allowed->open_target(allowed, nullptr, &status) == GRIM_ERROR_INVALID_PARAM &&
          allowed->open_target(allowed, "fixture", nullptr) == GRIM_ERROR_INVALID_PARAM, "Invalid native launch parameters accepted");
    nativeStatus = 32;
    check(allowed->launch_app(allowed, "fixture") == GRIM_ERROR_UNKNOWN, "Legacy launch failure changed");
    nativeStatus = 33;
    check(allowed->launch_app(allowed, "fixture") == GRIM_OK, "Legacy launch success changed");
    PluginAPI::cleanupPluginContext(allowed);

    const auto folder = fs::temp_directory_path() / ("grim-application-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    check(fs::create_directory(folder), "Cannot create application test folder");
    struct Cleanup {
        fs::path folder;
        ~Cleanup() {
            PluginManager::unloadAll();
            std::error_code error;
            fs::remove(folder / "application_plugin.dll", error);
            fs::remove(folder, error);
        }
    } cleanup{folder};
    fs::copy_file(pluginDirectory / "application_plugin.dll", folder / "application_plugin.dll");
    PluginManager::initialize(folder.string());
    check(registry.toolCount() == baseline + 2, "Application plugin did not publish two tools");
    for (const auto* id : {"open", "search"}) {
        const auto tool = registry.getTool(id);
        const bool search = std::string(id) == "search";
        check(tool && tool->provider_name == "Application Plugin" && tool->provider_type == GRIM::MMO::ToolProviderType::Plugin &&
              tool->permission_bits == GRIM_PERM_PROCESS && tool->is_informational == search && tool->needs_confirmation != search &&
              tool->parameters.size() == 1 && tool->parameters[0].required && tool->examples.size() == 1,
              "Application registration metadata changed policy or lacks documentation");
    }
    const auto result = [&](const char* id, const std::string& args) { return registry.invoke(id, args); };
    const auto matches = [&](const CommandResult& r, bool ok, const std::string& message, const char* code,
                             const char* category, const std::string& voice, Color color) {
        check(r.success == ok && r.message == message && r.errorCode == code && r.category == category &&
              r.voice == voice && r.color.toUInt() == color.toUInt() && r.structuredOutputJson.empty(),
              "Application result differs from former built-in behavior");
    };
    const auto beforeEmpty = launches;
    matches(result("open", ""), false, "configured:ERR_APP_NO_ARGUMENT", "ERR_APP_NO_ARGUMENT", "error", "Missing application name", Colors::Red);
    matches(result("search", " \t\r\n"), false, "configured:ERR_WEB_NO_ARGUMENT", "ERR_WEB_NO_ARGUMENT", "error", "No search query", Colors::Red);
    check(launches == beforeEmpty, "Empty input attempted a launch");
    matches(result("open", "  C:/test folder/app.exe  "), true, "[App] Launched:   C:/test folder/app.exe  ",
        "ERR_NONE", "routine", "Opening   C:/test folder/app.exe  ", Colors::Green);
    check(lastTarget == "  C:/test folder/app.exe  ", "open trimmed or rewrote its argument");
    matches(result("search", "  Mixed Case & stuff\t"), true, "[Web] Searching: Mixed Case & stuff",
        "ERR_NONE", "routine", "Searching web for Mixed Case & stuff", Colors::Cyan);
    check(lastTarget == "https://www.google.com/search?q=Mixed Case & stuff", "Search URL or trimming changed");
    nativeStatus = 32;
    errorPrefix = "updated:";
    matches(result("open", "missing"), false, "updated:ERR_APP_LAUNCH_FAILED: missing", "ERR_APP_LAUNCH_FAILED", "error", "Failed to open missing", Colors::Red);
    matches(result("search", "  query  "), false, "updated:ERR_WEB_OPEN_FAILED: query", "ERR_WEB_OPEN_FAILED", "error", "Web search failed", Colors::Red);
    nativeStatus = 33;
    errorPrefix = "configured:";
    fs::remove(folder / "application_plugin.dll");
    PluginManager::checkForHotReload();
    check(registry.toolCount() == baseline && !registry.isRegistered("open") && !registry.isRegistered("search"), "DLL removal retained application tools");
    std::cout << "Application plugin exact results, native launch boundaries, permissions, and removal tests passed\n";
}
