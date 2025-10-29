#include "plugin_manager.hpp"
#include "plugin_api.hpp"
#include "plugin_api_impl.hpp"
#include "logger.hpp"
#include <filesystem>
#include <thread>
#include <chrono>

using namespace std::chrono_literals;

void PluginManager::initialize(const std::string& folder) {
    pluginDir = folder;

    if (!std::filesystem::exists(pluginDir)) {
        LOG_DEBUG("PluginManager", "No plugins directory found.");
        return;
    }

    LOG_DEBUG("PluginManager", "Scanning for plugins in: " + pluginDir.string());
    for (auto& entry : std::filesystem::directory_iterator(pluginDir)) {
    if (entry.path().extension() != ".dll")
        continue;

    auto name = entry.path().stem().string();
    if (name == "core_plugin")
        continue; // skip built-in core

    loadPlugin(entry.path());
}

}

void PluginManager::checkForHotReload() {
    for (auto& entry : std::filesystem::directory_iterator(pluginDir)) {
            if (entry.path().stem().string() == "core_plugin")
        continue; // skip built-in core

        if (entry.path().extension() != ".dll")
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
    HMODULE lib = LoadLibraryA(path.string().c_str());
    if (!lib) {
        LOG_ERROR("PluginManager", "Failed to load plugin: " + name);
        return;
    }

    // Use new plugin API
    using GetInfoFunc = const GrimPluginInfo* (*)();
    using InitFunc = GrimResult (*)(const GrimPluginAPI*);
    
    auto getInfoFunc = reinterpret_cast<GetInfoFunc>(GetProcAddress(lib, "grim_plugin_get_info"));
    auto initFunc = reinterpret_cast<InitFunc>(GetProcAddress(lib, "grim_plugin_init"));
    
    if (!getInfoFunc || !initFunc) {
        LOG_ERROR("PluginManager", "Plugin missing required functions: " + name);
        FreeLibrary(lib);
        return;
    }

    try {
        // Get plugin info
        const GrimPluginInfo* info = getInfoFunc();
        if (!info) {
            LOG_ERROR("PluginManager", "Plugin get_info returned null: " + name);
            FreeLibrary(lib);
            return;
        }
        
        // Check API version compatibility
        if (info->api_version > GRIM_API_VERSION) {
            LOG_ERROR("PluginManager", "Plugin API version mismatch: " + name);
            FreeLibrary(lib);
            return;
        }
        
        LOG_DEBUG("PluginManager", "Loading plugin: " + std::string(info->name) + " v" + info->version);
        
        // Create API table with requested permissions
        GrimPluginAPI* api = PluginAPI::createPluginAPI(
            info->name,
            path.string(),
            info->required_permissions
        );
        
        // Initialize plugin
        GrimResult result = initFunc(api);
        if (result != GRIM_OK) {
            LOG_ERROR("PluginManager", "Plugin init failed: " + name);
            FreeLibrary(lib);
            return;
        }
        
        // Store plugin info
        LoadedPlugin lp;
        lp.handle = lib;
        lp.lastWrite = std::filesystem::last_write_time(path);
        loadedPlugins[name] = lp;

        LOG_DEBUG("PluginManager", "Loaded plugin: " + std::string(info->name));
    } catch (...) {
        LOG_ERROR("PluginManager", "Exception loading " + name);
        FreeLibrary(lib);
    }
}

void PluginManager::unloadPlugin(const std::string& name) {
    auto it = loadedPlugins.find(name);
    if (it == loadedPlugins.end())
        return;

    LOG_DEBUG("PluginManager", "Unloading plugin: " + name);

    try {
        // Call plugin shutdown
        using ShutdownFunc = void (*)();
        auto shutdownFunc = reinterpret_cast<ShutdownFunc>(
            GetProcAddress(it->second.handle, "grim_plugin_shutdown"));
        
        if (shutdownFunc) {
            shutdownFunc();
        }
        
        // Clean up plugin resources
        PluginAPI::cleanupPluginContext(name);
        
        FreeLibrary(it->second.handle);
    } catch (...) {
        LOG_ERROR("PluginManager", "Error while unloading " + name);
    }
}
