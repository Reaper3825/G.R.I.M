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
const GrimPluginAPI& hostAPI() {
    static const GrimPluginAPI api{.api_version = GRIM_API_VERSION, .log = fixtureLog};
    return api;
}
struct FixtureModule {
    PluginModule plugin;
    GrimPluginHandle handle;
};
static std::vector<FixtureModule> modules;

static void stopModule(FixtureModule& fixture) {
    const auto& name = fixture.plugin.descriptor.name;
    fixture.plugin.shutdown();
    check(active.erase(name) == 1, "Cleanup used DLL filename instead of declared plugin identity");
    events.push_back("cleanup:" + name);
    owners.erase(fixture.handle);
}

std::vector<PluginLoadOutcome> ProviderBatch::load(const std::vector<std::filesystem::path>& libraries,
                                                  OpenPlugin open) {
    static GrimPluginHandle next = 1;
    std::vector<PluginLoadOutcome> results;
    for (const auto& library : libraries) {
        PluginLoadOutcome outcome{library, PluginLoadStatus::InvalidBundle, {}};
        try {
            FixtureModule fixture{open(library, outcome.status), next++};
            const auto& name = fixture.plugin.descriptor.name;
            owners.emplace(fixture.handle, name);
            check(active.insert(name).second, "Plugin loaded more than once");
            if (fixture.plugin.init(&hostAPI(), fixture.handle) == GRIM_OK) {
                outcome.status = PluginLoadStatus::Loaded;
                modules.push_back(std::move(fixture));
            } else {
                stopModule(fixture);
                outcome.status = PluginLoadStatus::InitializationFailed;
                outcome.detail = "Plugin initialization failed";
            }
        } catch (const std::exception& error) {
            outcome.detail = error.what();
        }
        results.push_back(std::move(outcome));
    }
    return results;
}

std::size_t ProviderBatch::shutdown() {
    const auto count = modules.size();
    while (!modules.empty()) {
        stopModule(modules.back());
        modules.pop_back();
    }
    return count;
}

}
int main(int argc, char** argv) try {
    if (argc == 4 && std::string(argv[3]) == "--missing-toolkit") {
        PluginManager::loadBatch(argv[1]);
        PluginManager::shutdownBatch();
        check(active.empty() && errors.size() == 2, "Missing toolkit did not reject both fixture loads");
        for (const auto& error : errors)
            check(error.find("Plugin dependency directory must exist and be absolute:") != std::string::npos,
                  "Missing toolkit directory did not produce an explicit error");
        std::cout << "Missing toolkit rejection tests passed\n";
        return 0;
    }
    check(argc == 3, "Expected fixture and rejected-fixture directories");
    PluginManager::loadBatch(argv[1]);
    check(active.size() == 2, "Both fixture plugins must load");
    PluginManager::shutdownBatch();
    check(active.empty() && events.size() == 4, "Shutdown did not clean up every plugin");
    PluginManager::shutdownBatch();
    check(events.size() == 4, "Repeated shutdown called unloaded plugins");
    PluginManager::loadBatch(argv[1]);
    check(active.size() == 2, "Shutdown retained stale loaded-plugin entries");
    PluginManager::shutdownBatch();
    check(active.empty() && events.size() == 8, "Reloaded fixtures did not shut down");
    check(errors.empty(), "Valid plugins reported load errors");
    PluginManager::loadBatch(argv[2]);
    check(active.empty() && owners.empty(), "Failed init or incompatible ABI retained an API instance");
    check(events.size() == 10 && events.back() == "cleanup:Declared-failed", "Failed init was not cleaned up");
    check(errors.size() == 2, "Failed init and incompatible ABI were not both rejected");
    PluginManager::shutdownBatch();
    check(events.size() == 10, "Rejected plugin retained a loaded handle");

    const auto invalid = std::filesystem::path(argv[2]) / "invalid" / "invalid.dll";
    std::filesystem::create_directory(invalid.parent_path());
    {
        std::ofstream file(invalid, std::ios::binary);
        file << "not a native library";
        check(file.good(), "Cannot write invalid-library fixture");
    }
    const auto invalidSidecar = invalid.parent_path() / "invalid.tools.json";
    std::ofstream(invalidSidecar) << R"({"schema_version":2,"api_version":)" << GRIM_API_VERSION
        << R"(,"provider":"Invalid fixture","version":"1.0","tools":[]})";
    errors.clear();
    PluginManager::loadBatch(argv[2]);
    PluginManager::shutdownBatch();
    std::filesystem::remove(invalid);
    std::filesystem::remove(invalidSidecar);
    std::filesystem::remove(invalid.parent_path());
    const auto rejection = std::find_if(errors.begin(), errors.end(), [](const std::string& error) {
        return error.find("Rejected invalid.dll: Cannot load native library") != std::string::npos;
    });
    check(rejection != errors.end(), "Invalid native library did not report its load failure");
    check(rejection->find(invalid.string()) != std::string::npos ||
          rejection->find(std::filesystem::absolute(invalid).string()) != std::string::npos, "Load failure omitted the attempted native path");
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
