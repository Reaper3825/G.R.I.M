#include "plugin_manager.hpp"
#include "plugin_api_impl.hpp"
#include "plugin_manifest.hpp"
#include "logger.hpp"
#include <algorithm>
#include <fstream>
#include <optional>
#include <random>
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

}

std::optional<PluginManager::Fingerprint> PluginManager::fingerprint(const std::filesystem::path& path) {
    try {
        const auto time = std::filesystem::last_write_time(path);
        const auto size = std::filesystem::file_size(path);
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            LOG_DEBUG("PluginManager", "Cannot read plugin for fingerprint: " + path.string());
            return std::nullopt;
        }
        uint64_t hash = 14695981039346656037ULL;
        char buffer[65536];
        while (file.read(buffer, sizeof(buffer)) || file.gcount()) {
            for (std::streamsize i = 0; i < file.gcount(); ++i) {
                hash ^= static_cast<unsigned char>(buffer[i]);
                hash *= 1099511628211ULL;
            }
        }
        if (file.bad() || time != std::filesystem::last_write_time(path) ||
            size != std::filesystem::file_size(path)) {
            LOG_DEBUG("PluginManager", "Plugin changed or could not be read: " + path.string());
            return std::nullopt;
        }
        return Fingerprint{time, size, hash};
    } catch (const std::filesystem::filesystem_error& error) {
        LOG_DEBUG("PluginManager", std::string("Cannot fingerprint plugin: ") + error.what());
        return std::nullopt;
    }
}

std::optional<PluginManager::PluginFingerprint> PluginManager::fingerprintBundle(
    const std::filesystem::path& library) {
    const auto marker = library.parent_path() / ".deployment";
    // Manual deployments may omit the marker. CMake replaces it atomically
    // before and after publishing any changed files.
    const auto readDeployment = [&]() -> std::optional<std::string> {
        std::error_code error;
        if (!std::filesystem::exists(marker, error))
            return error ? std::nullopt : std::optional<std::string>{""};
        std::ifstream input(marker);
        std::string revision;
        if (!std::getline(input, revision) || revision.empty() || revision == "pending")
            return std::nullopt;
        return revision;
    };
    const auto deployment = readDeployment();
    if (!deployment) return std::nullopt;
    const auto sidecar = toolManifestPath(library);
    std::error_code error;
    if (!std::filesystem::is_regular_file(library, error) ||
        !std::filesystem::is_regular_file(sidecar, error)) return std::nullopt;
    const auto binary = fingerprint(library);
    const auto manifest = fingerprint(sidecar);
    if (!binary || !manifest || readDeployment() != deployment) return std::nullopt;
    return PluginFingerprint{*binary, *manifest, *deployment};
}

namespace {
std::filesystem::path createCacheDirectory() {
    std::random_device random;
    for (int attempt = 0; attempt < 32; ++attempt) {
        auto path = std::filesystem::temp_directory_path() /
            ("grim-plugin-" + std::to_string(random()) + "-" + std::to_string(random()));
        if (std::filesystem::create_directory(path)) return path;
    }
    throw std::runtime_error("Cannot create private plugin load directory");
}
void removeCache(const std::filesystem::path& directory) {
    if (directory.empty()) return;
    // This exact directory was exclusively created by createCacheDirectory.
    std::error_code error;
    std::filesystem::remove_all(directory, error);
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

void* loadLib(const std::filesystem::path& copy, const std::filesystem::path& sourceFolder) {
#ifdef _WIN32
    // Only register explicit directories for the duration of the native load.
    ScopedDllDirectory sourceDirectory(sourceFolder);
    const auto dependencyPath = copy.parent_path() / "dependencies";
    std::optional<ScopedDllDirectory> pluginDependencies;
    if (std::filesystem::is_directory(dependencyPath))
        pluginDependencies.emplace(dependencyPath);
#ifdef GRIM_PLUGIN_CUDA_BIN_DIR
    ScopedDllDirectory cudaDirectory(GRIM_PLUGIN_CUDA_BIN_DIR);
#endif
    auto module = LoadLibraryExW(copy.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
    const auto loadError = GetLastError();
    if (!module) {
        throw std::system_error(loadError, std::system_category(),
            "Cannot load native library " + copy.string() +
            " (Win32 error " + std::to_string(loadError) +
            "; dependency directory " + sourceFolder.string() + ")");
    }
    return module;
#else
    auto* module = dlopen(copy.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!module) throw std::runtime_error("Cannot load native library " + copy.string() + ": " + dlerror());
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
}

void PluginManager::initialize(const std::string& folder) {
    const auto requested = std::filesystem::absolute(folder).lexically_normal();
    if (!pluginDir.empty() && pluginDir != requested) unloadAll();
    pluginDir = requested;
    checkForHotReload();
    nextPoll = std::chrono::steady_clock::now() + std::chrono::seconds(1);
}
void PluginManager::requestReload(GrimPluginHandle instance) {
    if (!instance) return;
    std::lock_guard lock(requestMutex);
    reloadRequests.insert(instance);
}
void PluginManager::tick() {
    if (pluginDir.empty()) return;
    bool requested;
    {
        std::lock_guard lock(requestMutex);
        requested = !reloadRequests.empty();
    }
    const auto now = std::chrono::steady_clock::now();
    if (!requested && now < nextPoll) return;
    nextPoll = now + std::chrono::seconds(1);
    checkForHotReload();
}
void PluginManager::checkForHotReload() {
    if (pluginDir.empty()) return;
    try {
        std::vector<std::filesystem::path> files;
        std::error_code error;
        const bool exists = std::filesystem::exists(pluginDir, error);
        if (error) throw std::filesystem::filesystem_error("Inspect plugins folder", pluginDir, error);
        if (exists) {
            for (const auto& entry : std::filesystem::directory_iterator(pluginDir)) {
                const auto folder = entry.path();
                const auto name = folder.filename();
                if (name != "core_plugin" && entry.is_directory())
                    files.push_back(folder / (name.string() + pluginExtension));
            }
        }
        // Complete the directory scan before removing anything. An access error
        // is not evidence that a loaded plugin was deleted.
        std::sort(files.begin(), files.end());
        std::unordered_set<std::string> present;
        for (const auto& path : files) present.insert(path.parent_path().filename().string());
        std::vector<std::string> deleted;
        for (const auto& [name, plugin] : loadedPlugins)
            if (!present.count(name)) deleted.push_back(name);
        for (const auto& name : deleted) unloadPlugin(name);
        for (auto it = attempted.begin(); it != attempted.end();) {
            if (!present.count(it->first)) it = attempted.erase(it);
            else ++it;
        }
        std::unordered_set<GrimPluginHandle> requests;
        {
            std::lock_guard lock(requestMutex);
            requests.swap(reloadRequests);
        }
        for (const auto& path : files) {
            const auto name = path.parent_path().filename().string();
            const auto current = fingerprintBundle(path);
            if (!current) continue; // Wait for a complete, readable deployment.
            const auto old = loadedPlugins.find(name);
            const bool requested = old != loadedPlugins.end() && requests.count(PluginAPI::instanceHandle(*old->second.provider));
            if (!requested && old != loadedPlugins.end() && old->second.fingerprint == *current) continue;
            if (!requested && attempted.count(name) && attempted.at(name).fingerprint == *current &&
                attempted.at(name).surfaceRevision == surfaceRevision) continue;
            // Retry a rejection when another plugin changes the available surface:
            // its previous identity/alias collision may have disappeared.
            attempted[name] = {*current, surfaceRevision};
            loadPlugin(path, *current);
        }
    } catch (const std::exception& error) {
        LOG_ERROR("PluginManager", std::string("Plugin reconciliation failed: ") + error.what());
    }
}
void PluginManager::unloadAll() {
    pluginDir.clear(); // A later frame must not rediscover plugins during teardown.
    while (!loadedPlugins.empty()) unloadPlugin(loadedPlugins.begin()->first);
    attempted.clear();
    std::lock_guard lock(requestMutex);
    reloadRequests.clear();
}
bool PluginManager::loadPlugin(const std::filesystem::path& path, const PluginFingerprint& stamp) {
    const auto name = path.parent_path().filename().string();
    LoadedPlugin candidate;
    try {
        candidate.cacheDirectory = createCacheDirectory();
        const auto copy = candidate.cacheDirectory / path.filename();
        std::filesystem::copy_file(path, copy);
        const auto manifest = toolManifestPath(path);
        const auto manifestCopy = toolManifestPath(copy);
        std::filesystem::copy_file(manifest, manifestCopy);
        const auto copied = fingerprint(copy);
        const auto copiedManifest = fingerprint(manifestCopy);
        const auto dependencies = path.parent_path() / "dependencies";
        if (std::filesystem::is_directory(dependencies)) {
            const auto cacheDependencies = candidate.cacheDirectory / "dependencies";
            std::filesystem::create_directory(cacheDependencies);
            for (const auto& dependency : std::filesystem::directory_iterator(dependencies)) {
                if (dependency.is_regular_file() && dependency.path().extension() != ".pending")
                    std::filesystem::copy_file(dependency.path(), cacheDependencies / dependency.path().filename());
            }
        }
        if (fingerprintBundle(path) != stamp ||
            !copied || copied->hash != stamp.library.hash ||
            !copiedManifest || copiedManifest->hash != stamp.manifest.hash) {
            attempted.erase(name); // Retry once a complete write is visible.
            removeCache(candidate.cacheDirectory);
            return false;
        }
        std::ifstream manifestInput(manifestCopy, std::ios::binary);
        if (!manifestInput) throw std::runtime_error("Cannot open tool manifest: " + manifestCopy.string());
        const auto manifestData = nlohmann::json::parse(manifestInput);
        manifestInput.close();
        auto provider = PluginAPI::readManifestProvider(manifestData, path.string());
        PluginAPI::validateManifestTools(manifestData.at("tools"), provider);
        const auto previous = loadedPlugins.find(name);
        if (previous != loadedPlugins.end() && previous->second.providerName != provider.name)
            throw std::invalid_argument("Reload must retain the provider name");
        candidate.providerName = provider.name;
        unloadPlugin(name);
        candidate.handle = loadLib(copy, path.parent_path());
        using Init = decltype(&grim_plugin_init);
        auto init = reinterpret_cast<Init>(getSymbol(candidate.handle, "grim_plugin_init"));
        if (!init || !getSymbol(candidate.handle, "grim_plugin_shutdown"))
            throw std::runtime_error("Missing plugin entry points");

        candidate.provider = PluginAPI::registerInstance(std::move(provider));
        const auto dispatch = reinterpret_cast<GrimToolDispatch>(getSymbol(candidate.handle, "grim_plugin_dispatch_tool"));
        PluginAPI::registerProvider(*candidate.provider, manifestData.at("tools"), dispatch);
        candidate.initialized = true;
        if (init(&PluginAPI::hostAPI(), PluginAPI::instanceHandle(*candidate.provider)) != GRIM_OK)
            throw std::runtime_error("Plugin initialization failed");

        candidate.fingerprint = stamp;
        auto [entry, inserted] = loadedPlugins.try_emplace(name);
        if (!inserted) throw std::logic_error("Plugin is already loaded");
        PluginAPI::enableDispatch(*candidate.provider);
        entry->second = std::move(candidate);
        candidate.handle = nullptr;
        ++surfaceRevision;
        LOG_DEBUG("PluginManager", "Loaded plugin: " + name);
        return true;
    } catch (const std::exception& error) {
        LOG_ERROR("PluginManager", "Rejected " + path.filename().string() + ": " + error.what());
    } catch (...) {
        LOG_ERROR("PluginManager", "Exception initializing " + name);
    }
    retire(candidate);
    attempted[name] = {stamp, surfaceRevision};
    return false;
}

void PluginManager::retire(LoadedPlugin& plugin) {
    if (plugin.provider) {
        PluginAPI::retireProvider(*plugin.provider);
        if (plugin.initialized) {
            try {
                using Shutdown = void (*)();
                reinterpret_cast<Shutdown>(getSymbol(plugin.handle, "grim_plugin_shutdown"))();
            } catch (...) { LOG_ERROR("PluginManager", "Plugin shutdown raised an exception"); }
        }
        {
            std::lock_guard lock(requestMutex);
            reloadRequests.erase(PluginAPI::instanceHandle(*plugin.provider));
        }
        PluginAPI::releaseInstance(*plugin.provider);
    }
    unloadLib(plugin.handle);
    removeCache(plugin.cacheDirectory);
}

void PluginManager::unloadPlugin(const std::string& name) {
    const auto found = loadedPlugins.find(name);
    if (found == loadedPlugins.end()) return;
    retire(found->second);
    loadedPlugins.erase(found);
    ++surfaceRevision;
}
