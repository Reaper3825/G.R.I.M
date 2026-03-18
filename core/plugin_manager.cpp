#include "plugin_manager.hpp"
#include "plugin_api.hpp"
#include "plugin_api_impl.hpp"
#include "logger.hpp"
#include <filesystem>
#include <thread>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

using namespace std::chrono_literals;

namespace {

#ifdef _WIN32
const char* kPluginExt = ".dll";
#elif __APPLE__
const char* kPluginExt = ".dylib";
#else
const char* kPluginExt = ".so";
#endif

void* loadLib(const std::filesystem::path& path) {
#ifdef _WIN32
    return LoadLibraryA(path.string().c_str());
#else
    return dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
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

} // namespace

void PluginManager::initialize(const std::string& folder) {
    pluginDir = folder;

    if (!std::filesystem::exists(pluginDir)) {
        LOG_DEBUG("PluginManager", "No plugins directory found.");
        return;
    }

    LOG_DEBUG("PluginManager", "Scanning for plugins in: " + pluginDir.string());
    for (auto& entry : std::filesystem::directory_iterator(pluginDir)) {
        if (entry.path().extension() != kPluginExt)
            continue;

        auto name = entry.path().stem().string();
        if (name == "core_plugin")
            continue;

        loadPlugin(entry.path());
    }
}

void PluginManager::checkForHotReload() {
    for (auto& entry : std::filesystem::directory_iterator(pluginDir)) {
        if (entry.path().stem().string() == "core_plugin")
            continue;

        if (entry.path().extension() != kPluginExt)
            continue;

        std::string name = entry.path().filename().string();
        auto modTime = std::filesystem::last_write_time(entry.path());

        auto it = loadedPlugins.find(name);
        if (it == loadedPlugins.end()) {
            loadPlugin(entry.path());
            continue;
        }

        if (modTime > it->second.lastWrite) {
            LOG_DEBUG("PluginManager", "Detected change in " + name + " — reloading...");
            unloadPlugin(name);
            loadPlugin(entry.path());
        }
    }
}

void PluginManager::unloadAll() {
    for (auto it = loadedPlugins.begin(); it != loadedPlugins.end();) {
        unloadPlugin(it->first);
        it = loadedPlugins.erase(it);
    }
}

void PluginManager::loadPlugin(const std::filesystem::path& path) {
    std::string name = path.filename().string();
    void* lib = loadLib(path);
    if (!lib) {
        LOG_ERROR("PluginManager", "Failed to load plugin: " + name);
        return;
    }

    using GetInfoFunc = const GrimPluginInfo* (*)();
    using InitFunc = GrimResult (*)(const GrimPluginAPI*);

    auto getInfoFunc = reinterpret_cast<GetInfoFunc>(getSymbol(lib, "grim_plugin_get_info"));
    auto initFunc = reinterpret_cast<InitFunc>(getSymbol(lib, "grim_plugin_init"));

    if (!getInfoFunc || !initFunc) {
        LOG_ERROR("PluginManager", "Plugin missing required functions: " + name);
        unloadLib(lib);
        return;
    }

    try {
        const GrimPluginInfo* info = getInfoFunc();
        if (!info) {
            LOG_ERROR("PluginManager", "Plugin get_info returned null: " + name);
            unloadLib(lib);
            return;
        }

        if (info->api_version > GRIM_API_VERSION) {
            LOG_ERROR("PluginManager", "Plugin API version mismatch: " + name);
            unloadLib(lib);
            return;
        }

        LOG_DEBUG("PluginManager", "Loading plugin: " + std::string(info->name) + " v" + info->version);

        GrimPluginAPI* api = PluginAPI::createPluginAPI(
            info->name,
            path.string(),
            info->required_permissions
        );

        GrimResult result = initFunc(api);
        if (result != GRIM_OK) {
            LOG_ERROR("PluginManager", "Plugin init failed: " + name);
            unloadLib(lib);
            return;
        }

        LoadedPlugin lp;
        lp.handle = lib;
        lp.lastWrite = std::filesystem::last_write_time(path);
        loadedPlugins[name] = lp;

        LOG_DEBUG("PluginManager", "Loaded plugin: " + std::string(info->name));
    } catch (...) {
        LOG_ERROR("PluginManager", "Exception loading " + name);
        unloadLib(lib);
    }
}

void PluginManager::unloadPlugin(const std::string& name) {
    auto it = loadedPlugins.find(name);
    if (it == loadedPlugins.end())
        return;

    LOG_DEBUG("PluginManager", "Unloading plugin: " + name);

    try {
        using ShutdownFunc = void (*)();
        auto shutdownFunc = reinterpret_cast<ShutdownFunc>(
            getSymbol(it->second.handle, "grim_plugin_shutdown"));

        if (shutdownFunc) {
            shutdownFunc();
        }

        PluginAPI::cleanupPluginContext(name);
        unloadLib(it->second.handle);
    } catch (...) {
        LOG_ERROR("PluginManager", "Error while unloading " + name);
    }
}
