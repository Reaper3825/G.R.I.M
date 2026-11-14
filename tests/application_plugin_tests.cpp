#include "core/plugin_manager.hpp"
#include "core/plugin_api_impl.hpp"
#include "MMO/Core/Registries/ToolRegistry.hpp"
#include "error_manager.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cstdlib>

// Native discovery and launch fixtures live inside the DLL. The host supplies no application implementation.
namespace { std::string errorPrefix = "configured:"; }
std::string ErrorManager::getUserMessage(const std::string& code) { return errorPrefix + code; }

void runApplicationPluginTests(const std::filesystem::path& pluginDirectory) {
    namespace fs = std::filesystem;
    const auto check = [](bool ok, const char* message) { if (!ok) throw std::runtime_error(message); };
    auto& registry = GRIM::MMO::ToolRegistry::instance();
    const auto baseline = registry.toolCount();
    const auto folder = fs::temp_directory_path() / ("grim-application-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    check(fs::create_directory(folder), "Cannot create application test folder");
    const char* previousConfig=std::getenv("GRIM_APPLICATION_CONFIG_DIR");
    const bool hadConfig=previousConfig!=nullptr;
    const std::string oldConfig=previousConfig?previousConfig:"";
    _putenv_s("GRIM_APPLICATION_CONFIG_DIR",folder.string().c_str());
    struct RestoreConfig { bool had;std::string value;~RestoreConfig(){_putenv_s("GRIM_APPLICATION_CONFIG_DIR",had?value.c_str():"");} } restore{hadConfig,oldConfig};
    struct Cleanup {
        fs::path folder;
        ~Cleanup() {
            PluginManager::unloadAll();
            std::error_code error;
            fs::remove(folder / "application_plugin.dll", error);
            fs::remove(folder / "application_aliases.json", error);
            fs::remove(folder, error);
        }
    } cleanup{folder};
    fs::copy_file(pluginDirectory / "application_plugin.dll", folder / "application_plugin.dll");
    PluginManager::initialize(folder.string());
    check(registry.toolCount() == baseline + 6, "Application plugin did not publish six tools");
    auto catalogResult = registry.invoke("applications.list", "{}");
    check(catalogResult.success, "Application list failed");
    auto apps = nlohmann::json::parse(catalogResult.structuredOutputJson).at("value").at("applications");
    check(apps.size() == 1 && GRIM::MMO::ProviderRegistry::instance().snapshot().resourceCount() >= 1, "Application resources not published at plugin commit");
    { std::ofstream config(folder/"application_aliases.json"); config << R"({"version":1,"aliases":{"editor":"  C:/test folder/app.exe  "}})"; }
    check(registry.invoke("applications.refresh","{}").success,"User alias refresh failed");
    check(registry.invoke("open","EDITOR").success,"Plugin did not resolve a user alias before native launch");
    const auto app = apps.at(0);
    const auto applicationId = app.at("resource_id");
    check(registry.invoke("applications.inspect", nlohmann::json{{"resource_id", applicationId}}.dump()).success, "Application inspection failed");
    auto launchArgs = nlohmann::json{{"resource_id", applicationId}, {"provider_instance", app.at("provider_instance")}, {"runtime_id", app.at("runtime_id")}};
    check(registry.invoke("applications.launch", launchArgs.dump()).success, "Plugin application launch failed");
    launchArgs["provider_instance"] = "retired";
    check(!registry.invoke("applications.launch", launchArgs.dump()).success, "Stale application reached launcher");
    check(registry.invoke("applications.refresh", "{}").success, "Application refresh failed");
    check(!registry.invoke("applications.inspect", "{}").success, "Malformed inspect accepted");
    for (const auto* id : {"open", "search"}) {
        const auto tool = registry.getTool(id);
        const bool search = std::string(id) == "search";
        check(tool && tool->provider.name == "Application Plugin" && tool->provider.type == GRIM::MMO::ToolProviderType::Plugin &&
              tool->permission_bits == GRIM_PERM_PROCESS && tool->is_informational == search && tool->needs_confirmation != search &&
              tool->parameters.size() == 1 && tool->parameters[0].required,
              "Application registration metadata changed policy or lacks documentation");
    }
    const auto result = [&](const char* id, const std::string& args) { return registry.invoke(id, args); };
    const auto matches = [&](const CommandResult& r, bool ok, const std::string& message, const char* code,
                             const char* category, const std::string& voice, Color color) {
        check(r.success == ok && r.message == message && r.errorCode == code && r.category == category &&
              r.voice == voice && r.color.toUInt() == color.toUInt() && r.structuredOutputJson.empty(),
              "Application result differs from former built-in behavior");
    };
    matches(result("open", ""), false, "configured:ERR_APP_NO_ARGUMENT", "ERR_APP_NO_ARGUMENT", "error", "Missing application name", Colors::Red);
    matches(result("search", " \t\r\n"), false, "configured:ERR_WEB_NO_ARGUMENT", "ERR_WEB_NO_ARGUMENT", "error", "No search query", Colors::Red);
    matches(result("open", "  C:/test folder/app.exe  "), true, "[App] Launched:   C:/test folder/app.exe  ",
        "ERR_NONE", "routine", "Opening   C:/test folder/app.exe  ", Colors::Green);
    matches(result("search", "  Mixed Case & stuff\t"), true, "[Web] Searching: Mixed Case & stuff",
        "ERR_NONE", "routine", "Searching web for Mixed Case & stuff", Colors::Cyan);
    errorPrefix = "updated:";
    matches(result("open", "missing"), false, "updated:ERR_APP_LAUNCH_FAILED: missing", "ERR_APP_LAUNCH_FAILED", "error", "Failed to open missing", Colors::Red);
    matches(result("search", "  query  "), false, "updated:ERR_WEB_OPEN_FAILED: query", "ERR_WEB_OPEN_FAILED", "error", "Web search failed", Colors::Red);
    errorPrefix = "configured:";
    // Reload the same implementation into a new DLL generation; old observed identities must fail.
    fs::last_write_time(folder / "application_plugin.dll",
        fs::last_write_time(folder / "application_plugin.dll") + std::chrono::seconds(2));
    PluginManager::checkForHotReload();
    const auto reloaded = nlohmann::json::parse(registry.invoke("applications.list", "{}").structuredOutputJson)
        .at("value").at("applications").at(0);
    check(reloaded.at("provider_instance") != app.at("provider_instance"), "Application DLL did not reload");
    launchArgs["provider_instance"] = app.at("provider_instance");
    check(!registry.invoke("applications.launch", launchArgs.dump()).success, "Old observation launched after reload");
    launchArgs["provider_instance"] = reloaded.at("provider_instance");
    launchArgs["runtime_id"] = reloaded.at("runtime_id");
    check(registry.invoke("applications.launch", launchArgs.dump()).success, "Replacement lost its own catalog");
    fs::remove(folder / "application_plugin.dll");
    PluginManager::checkForHotReload();
    for (const auto& group : GRIM::MMO::ProviderRegistry::instance().snapshot().providers)
        check(group.provider.name != "Application Plugin", "DLL removal retained the application provider");
    check(registry.toolCount() == baseline && !registry.isRegistered("open") && !registry.isRegistered("search"), "DLL removal retained application tools");
    std::cout << "Application plugin results, DLL-owned native handling, reload identity and removal tests passed\n";
}
