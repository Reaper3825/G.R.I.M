// Exercise the production loader and registry with native fixture DLLs.
#include "core/plugin_manager.hpp"
#include "core/plugin_api_impl.hpp"
#include "core/plugin_notifications.hpp"
#include "error_manager.hpp"
#include "helpers/grim_input.hpp"
#include "logger.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <fstream>
#ifdef _WIN32
#include <windows.h>
#endif

namespace {
std::vector<std::string> shutdowns;
std::vector<std::string> errors;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
}
void logDebug(const std::string& tag, const std::string& message) {
    if (tag == "Plugin") shutdowns.push_back(message);
}
void logError(const std::string&, const std::string& message) { errors.push_back(message); }

// These host services are outside the loader fixtures' scope.
void publishPluginNotification(const std::string&, const std::string&) {}
std::string ErrorManager::getUserMessage(const std::string& code) { return code; }
namespace GRIMInput {
std::pair<std::string, std::string> parseInput(const std::string&) { return {}; }
}
namespace GRIM::UI {
void registerProviderPanelLease(GrimPluginHandle, const std::shared_ptr<ProviderPanelLease>&) {}
void retireProviderPanels(const std::shared_ptr<ProviderPanelLease>&) {}
PanelServiceRequest submitPanelRequest(GrimUIPanelOperation, const std::string&, uint64_t) {
    throw std::logic_error("Shutdown fixtures have no native UI");
}
}

int main(int argc, char** argv) try {
    auto& registry = GRIM::MMO::ProviderRegistry::instance();
    if (argc == 4 && std::string(argv[3]) == "--missing-toolkit") {
        const auto result = PluginManager::loadBatch(argv[1]);
        check(result.plugins.size() == 2 && std::all_of(result.plugins.begin(), result.plugins.end(), [](const auto& plugin) {
            return plugin.status == PluginLoadStatus::NativeLoadFailed;
        }), "Missing toolkit did not reject both fixture loads");
        check(PluginManager::shutdownBatch() == 0 && registry.snapshot().providers.empty() &&
              shutdowns.empty() && errors.size() == 2, "Missing toolkit left a provider or omitted an error");
        for (const auto& error : errors)
            check(error.find("Plugin dependency directory must exist and be absolute:") != std::string::npos,
                  "Missing toolkit directory did not produce an explicit error");
        std::cout << "Missing toolkit rejection tests passed\n";
        return 0;
    }
    check(argc == 3, "Expected fixture and rejected-fixture directories");
    const std::vector<std::string> expectedShutdowns{
        "[Plugin: Declared-beta] Declared-beta", "[Plugin: Declared-alpha] Declared-alpha"};
    for (int cycle = 0; cycle < 2; ++cycle) {
        shutdowns.clear();
        const auto result = PluginManager::loadBatch(argv[1]);
        const auto snapshot = registry.snapshot();
        check(result.status == PluginBatchResult::Status::Completed && result.plugins.size() == 2 &&
              snapshot.providers.size() == 2 && snapshot.providers[0].provider.name == "Declared-alpha" &&
              snapshot.providers[1].provider.name == "Declared-beta", "Both declared fixture providers must load");
        const auto native = registry.readyNativeProviders();
        check(native.size() == 2, "Both fixture handles must be ready");
        check(PluginManager::shutdownBatch() == 2 && registry.snapshot().providers.empty(),
              "Shutdown did not remove both providers");
        for (const auto& provider : native)
            check(!registry.nativeProvider(provider->handle), "Shutdown retained a fixture handle");
        check(shutdowns == expectedShutdowns, "Shutdown did not call both plugins in reverse load order with their API owners");
        check(PluginManager::shutdownBatch() == 0 && shutdowns == expectedShutdowns,
              "Repeated shutdown called unloaded plugins");
    }
    check(errors.empty(), "Valid plugins reported load errors");
    shutdowns.clear();
    const auto rejected = PluginManager::loadBatch(argv[2]);
    check(rejected.plugins.size() == 2 && rejected.plugins[0].status == PluginLoadStatus::InitializationFailed &&
          rejected.plugins[1].status == PluginLoadStatus::InvalidBundle, "Expected init and ABI rejections");
    check(registry.snapshot().providers.empty(), "A rejected provider was published");
    check(shutdowns == std::vector<std::string>{"[Plugin: Declared-failed] Declared-failed"},
          "Failed init did not run shutdown with its API owner");
    check(errors.size() == 2, "Failed init and incompatible ABI were not both rejected");
    check(PluginManager::shutdownBatch() == 0 && shutdowns.size() == 1, "Rejected plugin retained a loaded handle");

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
