#include "plugin_manager.hpp"
#include "plugin_api_impl.hpp"
#include "logger.hpp"
#include <algorithm>
#include <fstream>
#include <optional>
#include <random>
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

std::optional<std::string> fingerprint(const std::filesystem::path& path) {
    try {
        const auto time = std::filesystem::last_write_time(path);
        const auto size = std::filesystem::file_size(path);
        std::ifstream file(path, std::ios::binary);
        if (!file) return std::nullopt;
        uint64_t hash = 14695981039346656037ULL;
        char buffer[65536];
        while (file.read(buffer, sizeof(buffer)) || file.gcount()) {
            for (std::streamsize i = 0; i < file.gcount(); ++i) {
                hash ^= static_cast<unsigned char>(buffer[i]);
                hash *= 1099511628211ULL;
            }
        }
        if (file.bad() || time != std::filesystem::last_write_time(path) ||
            size != std::filesystem::file_size(path)) return std::nullopt;
        return std::to_string(time.time_since_epoch().count()) + ":" +
               std::to_string(size) + ":" + std::to_string(hash);
    } catch (...) { return std::nullopt; }
}
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
void* loadLib(const std::filesystem::path& copy, const std::filesystem::path& sourceFolder) {
#ifdef _WIN32
    // Search the source directory for companion libraries while loading the
    // private copy. No persistent change to the application's DLL search path.
    auto cookie = AddDllDirectory(sourceFolder.c_str());
    auto module = LoadLibraryExW(copy.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
    if (cookie) RemoveDllDirectory(cookie);
    return module;
#else
    return dlopen(copy.c_str(), RTLD_NOW | RTLD_LOCAL);
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
void PluginManager::requestReload(const GrimPluginAPI* api) {
    if (!api) return;
    std::lock_guard lock(requestMutex);
    reloadRequests.insert(api);
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
                if (entry.is_regular_file() && entry.path().extension() == pluginExtension &&
                    entry.path().stem() != "core_plugin") files.push_back(entry.path());
            }
        }
        // Complete the directory scan before removing anything. An access error
        // is not evidence that a loaded plugin was deleted.
        std::sort(files.begin(), files.end());
        std::unordered_set<std::string> present;
        for (const auto& path : files) present.insert(path.filename().string());
        std::vector<std::string> deleted;
        for (const auto& [name, plugin] : loadedPlugins)
            if (!present.count(name)) deleted.push_back(name);
        for (const auto& name : deleted) unloadPlugin(name);
        for (auto it = attempted.begin(); it != attempted.end();) {
            if (!present.count(it->first)) it = attempted.erase(it);
            else ++it;
        }
        std::unordered_set<const GrimPluginAPI*> requests;
        {
            std::lock_guard lock(requestMutex);
            requests.swap(reloadRequests);
        }
        for (const auto& path : files) {
            const auto name = path.filename().string();
            const auto current = fingerprint(path);
            if (!current) continue; // Still being written or temporarily unreadable.
            const auto old = loadedPlugins.find(name);
            const bool requested = old != loadedPlugins.end() && requests.count(old->second.api);
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
bool PluginManager::loadPlugin(const std::filesystem::path& path, const std::string& stamp) {
    const auto name = path.filename().string();
    auto previous = loadedPlugins.find(name);
    const auto* oldApi = previous == loadedPlugins.end() ? nullptr : previous->second.api;
    LoadedPlugin candidate;
    bool initialized = false;
    try {
        candidate.cacheDirectory = createCacheDirectory();
        const auto copy = candidate.cacheDirectory / path.filename();
        std::filesystem::copy_file(path, copy);
        const auto copied = fingerprint(copy);
        if (fingerprint(path) != std::optional<std::string>(stamp) || !copied ||
            copied->substr(copied->rfind(':')) != stamp.substr(stamp.rfind(':'))) {
            attempted.erase(name); // Retry once a complete write is visible.
            removeCache(candidate.cacheDirectory);
            return false;
        }
        candidate.handle = loadLib(copy, path.parent_path());
        if (!candidate.handle) throw std::runtime_error("Cannot load native library");
        using GetInfo = const GrimPluginInfo* (*)();
        using Init = GrimResult (*)(const GrimPluginAPI*);
        auto getInfo = reinterpret_cast<GetInfo>(getSymbol(candidate.handle, "grim_plugin_get_info"));
        auto init = reinterpret_cast<Init>(getSymbol(candidate.handle, "grim_plugin_init"));
        if (!getInfo || !init || !getSymbol(candidate.handle, "grim_plugin_shutdown"))
            throw std::runtime_error("Missing plugin entry points");
        const auto* info = getInfo();
        if (!info || (info->api_version >> 16) != GRIM_API_VERSION_MAJOR ||
            info->api_version > GRIM_API_VERSION) throw std::runtime_error("Plugin API version mismatch");
        if (!info->name || !*info->name || !info->version || !*info->version)
            throw std::runtime_error("Invalid plugin identity");
        candidate.declaredName = info->name;
        if (oldApi && candidate.declaredName != previous->second.declaredName)
            throw std::runtime_error("Replacement must retain provider identity");
        candidate.api = PluginAPI::createPluginAPI(info->name, path.string(),
            info->required_permissions, true, oldApi);
        initialized = true; // Shutdown must tolerate partially completed init.
        if (init(candidate.api) != GRIM_OK || !PluginAPI::commitPluginTools(candidate.api, oldApi))
            throw std::runtime_error("Initialization or tool publication rejected");
        candidate.fingerprint = stamp;
    } catch (const std::exception& error) {
        LOG_ERROR("PluginManager", "Rejected " + name + ": " + error.what());
        if (initialized) retire(candidate);
        else { unloadLib(candidate.handle); removeCache(candidate.cacheDirectory); }
        return false;
    } catch (...) {
        LOG_ERROR("PluginManager", "Exception initializing " + name);
        if (initialized) retire(candidate);
        else { unloadLib(candidate.handle); removeCache(candidate.cacheDirectory); }
        return false;
    }
    // The replacement is committed. Old-generation cleanup cannot remove its
    // tools or resources because ownership includes a generation identifier.
    if (previous != loadedPlugins.end()) {
        retire(previous->second);
        previous->second = std::move(candidate);
    } else loadedPlugins.emplace(name, std::move(candidate));
    ++surfaceRevision;
    LOG_DEBUG("PluginManager", "Loaded plugin: " + name);
    return true;
}
void PluginManager::retire(LoadedPlugin& plugin) {
    PluginAPI::beginPluginShutdown(plugin.api);
    try {
        using Shutdown = void (*)();
        reinterpret_cast<Shutdown>(getSymbol(plugin.handle, "grim_plugin_shutdown"))();
    } catch (...) { LOG_ERROR("PluginManager", "Plugin shutdown raised an exception"); }
    {
        std::lock_guard lock(requestMutex);
        reloadRequests.erase(plugin.api);
    }
    PluginAPI::cleanupPluginContext(plugin.api);
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
