#include "core/plugin_manager.hpp"
#include "plugin_deployment_fixture.hpp"
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
            PluginManager::shutdownBatch();
            std::error_code error;
            fs::remove(folder / "application_plugin/application_plugin.dll", error);
            fs::remove(folder / "application_plugin/.deployment", error);
            fs::remove(folder / "application_plugin/application_plugin.tools.json", error);
            fs::remove(folder / "application_aliases.json", error);
            fs::remove(folder / "application_plugin", error);
            fs::remove(folder, error);
        }
    } cleanup{folder};
    fs::create_directory(folder / "application_plugin");
    fs::copy_file(pluginDirectory / "application_plugin.dll", folder / "application_plugin/application_plugin.dll");
    fs::copy_file(pluginDirectory / "application_plugin.tools.json", folder / "application_plugin/application_plugin.tools.json");
    writePluginFixtureRevision(folder / "application_plugin");
    PluginManager::loadBatch(folder.string());
    check(registry.toolCount() == baseline + 6, "Application plugin did not publish six tools");
    auto catalogResult = registry.invoke("applications.list", "{}");
    check(catalogResult.at("status") == "completed", "Application list failed");
    auto apps = catalogResult.at("payload").at("value").at("applications");
    check(apps.size() == 1 && GRIM::MMO::ProviderRegistry::instance().snapshot().resourceCount() >= 1, "Application resources not published at plugin commit");
    { std::ofstream config(folder/"application_aliases.json"); config << R"({"version":1,"aliases":{"editor":"  C:/test folder/app.exe  "}})"; }
    check(registry.invoke("applications.refresh","{}").at("status") == "completed","User alias refresh failed");
    check(registry.invoke("open","EDITOR").at("status") == "completed","Plugin did not resolve a user alias before native launch");
    const auto app = apps.at(0);
    const auto applicationId = app.at("resource_id");
    const auto originalTool = registry.getTool("applications.list");
    check(originalTool.has_value(), "Application list tool missing");
    const auto snapshot = GRIM::MMO::ProviderRegistry::instance().snapshot();
    bool foundPublishedDescriptor = false;
    for (const auto& group : snapshot.providers) {
        if (group.provider.instance != originalTool->provider.instance) continue;
        for (const auto& resource : group.resources)
            if (app == nlohmann::json(resource)) foundPublishedDescriptor = true;
    }
    check(foundPublishedDescriptor, "Catalog list did not return the plain published descriptor");
    const auto inspect = registry.invoke("applications.inspect", nlohmann::json{{"resource_id", applicationId}}.dump());
    check(inspect.at("status") == "completed" && inspect.at("payload").at("value").at("application") == app,
          "Application inspection did not return the plain descriptor");
    const auto launchArgs = nlohmann::json{{"resource_id", applicationId}, {"runtime_id", app.at("runtime_id")}};
    check(registry.invoke("applications.launch", launchArgs.dump()).at("status") == "completed", "Plugin application launch failed");
    check(registry.invoke("applications.refresh", "{}").at("status") == "completed", "Application refresh failed");
    check(registry.invoke("applications.inspect", "{}").at("status") == "failed", "Malformed inspect accepted");
    for (const auto* id : {"open", "search"}) {
        const auto tool = registry.getTool(id);
        const bool search = std::string(id) == "search";
        check(tool && tool->provider.name == "Application Plugin" && tool->provider.type == GRIM::MMO::ToolProviderType::Plugin &&
              tool->is_informational == search && tool->needs_confirmation != search &&
              tool->parameters.size() == 1 && tool->parameters[0].required,
              "Application registration metadata changed policy or lacks documentation");
    }
    const auto result = [&](const char* id, const std::string& args) { return registry.invoke(id, args); };
    const auto matches = [&](const nlohmann::json& r, bool ok, const std::string& message, const char* code,
                             const char* category) {
        check((r.at("status") == "completed") == ok && r.at("message") == message && r.at("error_code") == code && r.at("category") == category &&
              r.at("payload").empty(),
              "Application result differs from former built-in behavior");
    };
    matches(result("open", ""), false, "configured:ERR_APP_NO_ARGUMENT", "ERR_APP_NO_ARGUMENT", "error");
    matches(result("search", " \t\r\n"), false, "configured:ERR_WEB_NO_ARGUMENT", "ERR_WEB_NO_ARGUMENT", "error");
    matches(result("open", "  C:/test folder/app.exe  "), true, "[App] Launched:   C:/test folder/app.exe  ",
        "ERR_NONE", "routine");
    matches(result("search", "  Mixed Case & stuff\t"), true, "[Web] Searching: Mixed Case & stuff",
        "ERR_NONE", "routine");
    errorPrefix = "updated:";
    matches(result("open", "missing"), false, "updated:ERR_APP_LAUNCH_FAILED: missing", "ERR_APP_LAUNCH_FAILED", "error");
    matches(result("search", "  query  "), false, "updated:ERR_WEB_OPEN_FAILED: query", "ERR_WEB_OPEN_FAILED", "error");
    errorPrefix = "configured:";
    PluginManager::shutdownBatch();
    // Start a new host lifecycle and launch using the matching application IDs.
    PluginManager::loadBatch(folder);
    const auto reloaded = registry.invoke("applications.list", "{}").at("payload")
        .at("value").at("applications").at(0);
    const auto reloadedTool = registry.getTool("applications.list");
    check(reloadedTool && reloadedTool->provider.instance != originalTool->provider.instance, "Application DLL did not reload");
    check(reloaded.at("resource_id") == applicationId && reloaded.at("runtime_id") == app.at("runtime_id"),
          "Reload fixture did not retain matching application IDs");
    check(registry.invoke("applications.launch", launchArgs.dump()).at("status") == "completed", "Matching application IDs failed after reload");
    PluginManager::shutdownBatch();
    fs::remove(folder / "application_plugin/application_plugin.dll");
    fs::remove(folder / "application_plugin/.deployment");
    fs::remove(folder / "application_plugin/application_plugin.tools.json");
    fs::remove(folder / "application_plugin");
    for (const auto& group : GRIM::MMO::ProviderRegistry::instance().snapshot().providers)
        check(group.provider.name != "Application Plugin", "DLL removal retained the application provider");
    check(registry.toolCount() == baseline && !registry.isRegistered("open") && !registry.isRegistered("search"), "DLL removal retained application tools");
    std::cout << "Application plugin results, DLL-owned native handling, reload identity and removal tests passed\n";
}
