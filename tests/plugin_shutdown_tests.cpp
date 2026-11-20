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
std::unordered_map<GrimPluginHandle, std::string> owners;
void fixtureLog(GrimPluginHandle instance, GrimLogLevel, const char* name) {
    check(owners.at(instance) == name, "Wrong API owner during shutdown");
    check(active.count(name) == 1, "Plugin shutdown ran after its context was removed");
    events.push_back(std::string("shutdown:") + name);
}
}
void logDebug(const std::string&, const std::string&) {}
void logError(const std::string&, const std::string& message) { errors.push_back(message); }
namespace PluginAPI {
struct LoadedProvider { std::string name; GrimPluginHandle handle; };
std::shared_ptr<LoadedProvider> registerInstance(GRIM::MMO::ProviderDescriptor provider) {
    static GrimPluginHandle next = 1;
    auto owner = std::make_shared<LoadedProvider>(LoadedProvider{std::move(provider.name), next++});
    owners.emplace(owner->handle, owner->name);
    return owner;
}
GrimPluginHandle instanceHandle(const LoadedProvider& provider) { return provider.handle; }
const GrimPluginAPI& hostAPI() {
    static const GrimPluginAPI api{.api_version = GRIM_API_VERSION, .log = fixtureLog};
    return api;
}
void validateManifestTools(const nlohmann::json&, const GRIM::MMO::ProviderDescriptor&) {}
void registerProvider(LoadedProvider& provider, const nlohmann::json&, GrimToolDispatch) {
    check(active.insert(provider.name).second, "Plugin loaded more than once");
}
void enableDispatch(LoadedProvider&) {}
void retireProvider(LoadedProvider&) {}
void releaseInstance(LoadedProvider& provider) {
    const auto& name = provider.name;
    check(active.erase(name) == 1, "Cleanup used DLL filename instead of declared plugin identity");
    check(name == "Declared-failed" || std::find(events.begin(), events.end(), "shutdown:" + name) != events.end(), "Cleanup ran before shutdown");
    events.push_back("cleanup:" + name);
    owners.erase(provider.handle);
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
    check(active.empty() && owners.empty(), "Failed init or incompatible ABI retained an API instance");
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
    std::ofstream(invalidSidecar) << R"({"schema_version":2,"api_version":)" << GRIM_API_VERSION
        << R"(,"provider":"Invalid fixture","version":"1.0","tools":[]})";
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
