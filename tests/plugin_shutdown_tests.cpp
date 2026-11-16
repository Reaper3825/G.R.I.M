// Host-only loader lifecycle test with tiny fixture DLLs and API service sinks.
#include "core/plugin_manager.hpp"
#include "core/plugin_api_impl.hpp"
#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>
#include <memory>
#include <unordered_map>
#include <fstream>
#ifdef _WIN32
#include <windows.h>
#endif

namespace {
std::set<std::string> active;
std::vector<std::string> events;
std::vector<std::string> errors;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void fixtureLog(const GrimPluginAPI* self, GrimLogLevel, const char* name) {
    check(self && *static_cast<const std::string*>(self->host_context) == name, "Wrong API owner during shutdown");
    check(active.count(name) == 1, "Plugin shutdown ran after its context was removed");
    events.push_back(std::string("shutdown:") + name);
}
}
void logDebug(const std::string&, const std::string&) {}
void logError(const std::string&, const std::string& message) { errors.push_back(message); }
namespace PluginAPI {
struct TestAPI { std::string name; GrimPluginAPI api{}; };
static std::unordered_map<std::string, std::unique_ptr<TestAPI>> apis;
GrimPluginAPI* createPluginAPI(const std::string& name, const std::string&, uint32_t, bool, const GrimPluginAPI*) {
    check(active.insert(name).second, "Plugin loaded more than once");
    auto instance = std::make_unique<TestAPI>();
    instance->name = name;
    instance->api.host_context = &instance->name;
    instance->api.api_version = GRIM_API_VERSION;
    instance->api.log = fixtureLog;
    auto* result = &instance->api;
    apis.emplace(name, std::move(instance));
    return result;
}
std::vector<GRIM::MMO::ToolRegistration> readToolManifest(
    const std::filesystem::path&, const GrimPluginAPI*, const char*, GrimToolDispatch) { return {}; }
bool commitPluginGeneration(const GrimPluginAPI*, const GrimPluginAPI*,
                            const std::vector<GRIM::MMO::ToolRegistration>&) { return true; }
void beginPluginShutdown(const GrimPluginAPI*) {}
void cleanupPluginContext(const GrimPluginAPI* api) {
    const auto name = *static_cast<const std::string*>(api->host_context);
    check(active.erase(name) == 1, "Cleanup used DLL filename instead of declared plugin identity");
    check(name == "Declared-failed" || std::find(events.begin(), events.end(), "shutdown:" + name) != events.end(), "Cleanup ran before shutdown");
    events.push_back("cleanup:" + name);
    apis.erase(name);
}
}
int main(int argc, char** argv) try {
    if (argc == 4 && std::string(argv[3]) == "--missing-toolkit") {
        PluginManager::initialize(argv[1]);
        PluginManager::unloadAll();
        check(active.empty() && errors.size() == 2, "Missing toolkit did not reject both fixture loads");
        for (const auto& error : errors)
            check(error.find("Plugin dependency directory must exist and be absolute:") != std::string::npos,
                  "Missing toolkit directory did not produce an explicit error");
        std::cout << "Missing toolkit rejection tests passed\n";
        return 0;
    }
    check(argc == 3, "Expected fixture and rejected-fixture directories");
    PluginManager::initialize(argv[1]);
    check(active.size() == 2, "Both fixture plugins must load");
    PluginManager::unloadAll();
    check(active.empty() && events.size() == 4, "Shutdown did not clean up every plugin");
    PluginManager::unloadAll();
    check(events.size() == 4, "Repeated shutdown called unloaded plugins");
    PluginManager::initialize(argv[1]);
    check(active.size() == 2, "Shutdown retained stale loaded-plugin entries");
    PluginManager::unloadAll();
    check(active.empty() && events.size() == 8, "Reloaded fixtures did not shut down");
    check(errors.empty(), "Valid plugins reported load errors");
    PluginManager::initialize(argv[2]);
    check(active.empty() && PluginAPI::apis.empty(), "Failed init or incompatible ABI retained an API instance");
    check(events.size() == 10 && events.back() == "cleanup:Declared-failed", "Failed init was not cleaned up");
    check(errors.size() == 2, "Failed init and incompatible ABI were not both rejected");
    PluginManager::unloadAll();
    check(events.size() == 10, "Rejected plugin retained a loaded handle");

    const auto invalid = std::filesystem::path(argv[2]) / "invalid.dll";
    {
        std::ofstream file(invalid, std::ios::binary);
        file << "not a native library";
        check(file.good(), "Cannot write invalid-library fixture");
    }
    const auto invalidSidecar = invalid.parent_path() / "invalid.tools.json";
    std::ofstream(invalidSidecar) << "{}";
    errors.clear();
    PluginManager::initialize(argv[2]);
    PluginManager::unloadAll();
    std::filesystem::remove(invalid);
    std::filesystem::remove(invalidSidecar);
    const auto rejection = std::find_if(errors.begin(), errors.end(), [](const std::string& error) {
        return error.find("Rejected invalid.dll: Cannot load native library") != std::string::npos;
    });
    check(rejection != errors.end(), "Invalid native library did not report its load failure");
    check(rejection->find("grim-plugin-") != std::string::npos, "Load failure omitted the attempted native path");
#ifdef _WIN32
    check(rejection->find("Win32 error 193") != std::string::npos, "Load failure lost the original Win32 error");
#endif

#if defined(_WIN32) && defined(GRIM_PLUGIN_CUDA_BIN_DIR)
    check(GetModuleHandleW(L"grim_test_dependency.dll") == nullptr, "Dependency survived plugin unload");
    auto dependency = LoadLibraryExW(L"grim_test_dependency.dll", nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (dependency) FreeLibrary(dependency);
    check(!dependency, "Toolkit DLL search registration leaked beyond native load");
#endif

    std::cout << "Plugin shutdown tests passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
