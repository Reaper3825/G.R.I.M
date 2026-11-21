#include "plugin_manager.hpp"
#include "plugin_api_impl.hpp"
#include "plugin_manifest.hpp"
#include "logger.hpp"
#include <algorithm>
#include <fstream>
#include <optional>
#include <system_error>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {
#ifdef _WIN32
constexpr const char* pluginExtension = ".dll";
#elif __APPLE__
constexpr const char* pluginExtension = ".dylib";
#else
constexpr const char* pluginExtension = ".so";
#endif

std::filesystem::path toolManifestPath(std::filesystem::path library) {
    return library.replace_extension(".tools.json");
}

#ifdef _WIN32
class ScopedDllDirectory {
public:
    explicit ScopedDllDirectory(const std::filesystem::path& path) : path_(path) {
        if (!path_.is_absolute() || !std::filesystem::is_directory(path_))
            throw std::runtime_error("Plugin dependency directory must exist and be absolute: " + path_.string());
        cookie_ = AddDllDirectory(path_.c_str());
        if (!cookie_) {
            const auto error = GetLastError();
            throw std::system_error(error, std::system_category(),
                "Cannot add plugin dependency directory " + path_.string());
        }
    }
    ~ScopedDllDirectory() {
        if (!RemoveDllDirectory(cookie_)) {
            const auto error = GetLastError();
            LOG_ERROR("PluginManager", "Cannot remove plugin dependency directory " +
                path_.string() + " (Win32 error " + std::to_string(error) + ")");
        }
    }
    ScopedDllDirectory(const ScopedDllDirectory&) = delete;
    ScopedDllDirectory& operator=(const ScopedDllDirectory&) = delete;
private:
    std::filesystem::path path_;
    DLL_DIRECTORY_COOKIE cookie_;
};
#endif

void* loadLib(const std::filesystem::path& library) {
#ifdef _WIN32
    // Register the private dependencies directory for the duration of the native load.
    const auto sourceFolder = library.parent_path();
    const auto dependencyPath = sourceFolder / "dependencies";
    std::optional<ScopedDllDirectory> pluginDependencies;
    if (std::filesystem::is_directory(dependencyPath))
        pluginDependencies.emplace(dependencyPath);
#ifdef GRIM_PLUGIN_CUDA_BIN_DIR
    ScopedDllDirectory cudaDirectory(GRIM_PLUGIN_CUDA_BIN_DIR);
#endif
    auto module = LoadLibraryExW(library.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
    const auto loadError = GetLastError();
    if (!module) {
        throw std::system_error(loadError, std::system_category(),
            "Cannot load native library " + library.string() +
            " (Win32 error " + std::to_string(loadError) +
            "; dependency directory " + sourceFolder.string() + ")");
    }
    return module;
#else
    auto* module = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!module) throw std::runtime_error("Cannot load native library " + library.string() + ": " + dlerror());
    return module;
#endif
}
void unloadLib(void* handle) {
#ifdef _WIN32
    if (handle) FreeLibrary(static_cast<HMODULE>(handle));
#else
    if (handle) dlclose(handle);
#endif
}
void* getSymbol(void* handle, const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
    return dlsym(handle, name);
#endif
}

// The deployment publisher marks incomplete bundles before replacing files.
std::string deploymentRevision(const std::filesystem::path& directory) {
    const auto marker = directory / ".deployment";
    std::ifstream input(marker);
    if (!input) throw std::runtime_error("Cannot read plugin deployment marker: " + marker.string());
    std::string revision;
    if (!std::getline(input, revision) || revision.empty() || revision == "pending")
        throw std::runtime_error("Plugin deployment is incomplete");
    return revision;
}

PluginAPI::PluginModule openPlugin(const std::filesystem::path& library, PluginLoadStatus& failure) {
    const auto revision = deploymentRevision(library.parent_path());
    auto manifest = std::ifstream(toolManifestPath(library), std::ios::binary);
    if (!manifest) throw std::runtime_error("Cannot open tool manifest: " + toolManifestPath(library).string());
    const auto data = nlohmann::json::parse(manifest);
    PluginAPI::PluginModule plugin;
    plugin.descriptor = PluginAPI::readManifestProvider(data, library.string());
    plugin.tools = data.at("tools");

    failure = PluginLoadStatus::NativeLoadFailed;
    plugin.library = {loadLib(library), unloadLib};
    plugin.init = reinterpret_cast<decltype(plugin.init)>(getSymbol(plugin.library.get(), "grim_plugin_init"));
    plugin.shutdown = reinterpret_cast<decltype(plugin.shutdown)>(getSymbol(plugin.library.get(), "grim_plugin_shutdown"));
    if (!plugin.init || !plugin.shutdown) throw std::runtime_error("Missing plugin entry points");
    plugin.dispatch = reinterpret_cast<GrimToolDispatch>(getSymbol(plugin.library.get(), "grim_plugin_dispatch_tool"));

    failure = PluginLoadStatus::InvalidBundle;
    if (deploymentRevision(library.parent_path()) != revision)
        throw std::runtime_error("Plugin deployment changed during startup");
    return plugin;
}

enum class Lifecycle { Idle, Starting, Started, Stopping };
Lifecycle lifecycle = Lifecycle::Idle;
}

PluginBatchResult PluginManager::loadBatch(const std::filesystem::path& folder) {
    using Status = PluginBatchResult::Status;
    if (lifecycle != Lifecycle::Idle) return {Status::AlreadyStarted, {}};

    std::vector<std::filesystem::path> libraries;
    try {
        const auto root = std::filesystem::absolute(folder).lexically_normal();
        if (std::filesystem::exists(root)) {
            for (const auto& entry : std::filesystem::directory_iterator(root)) {
                const auto name = entry.path().filename().string();
                if (name == "core_plugin" || !entry.is_directory()) continue;
                const auto library = entry.path() / (name + pluginExtension);
                if (std::filesystem::is_regular_file(library)) libraries.push_back(library);
            }
        }
        std::sort(libraries.begin(), libraries.end());
    } catch (const std::filesystem::filesystem_error& error) {
        LOG_ERROR("PluginManager", std::string("Plugin discovery failed: ") + error.what());
        return {Status::DiscoveryFailed, {}};
    }

    lifecycle = Lifecycle::Starting;
    std::vector<PluginLoadOutcome> outcomes;
    try {
        outcomes = PluginAPI::ProviderBatch::load(libraries, openPlugin);
    } catch (...) {
        lifecycle = Lifecycle::Idle;
        throw;
    }
    lifecycle = Lifecycle::Started;
    for (const auto& outcome : outcomes) {
        const auto name = outcome.library.filename().string();
        if (outcome.status == PluginLoadStatus::Loaded) LOG_DEBUG("PluginManager", "Loaded plugin: " + name);
        else LOG_ERROR("PluginManager", "Rejected " + name + ": " + outcome.detail);
    }
    return {Status::Completed, std::move(outcomes)};
}

std::size_t PluginManager::shutdownBatch() {
    if (lifecycle != Lifecycle::Started) return 0;
    lifecycle = Lifecycle::Stopping;
    try {
        const auto stopped = PluginAPI::ProviderBatch::shutdown();
        lifecycle = Lifecycle::Idle;
        return stopped;
    } catch (...) {
        lifecycle = Lifecycle::Started;
        throw;
    }
}
