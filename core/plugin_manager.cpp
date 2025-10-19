#include "plugin_manager.hpp"
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

    using RegisterFunc = void (*)();
    auto registerFunc = reinterpret_cast<RegisterFunc>(
        GetProcAddress(lib, "registerGrimPlugin"));

    if (!registerFunc) {
        LOG_ERROR("PluginManager", "registerGrimPlugin() not found in " + name);
        FreeLibrary(lib);
        return;
    }

    try {
        registerFunc();
        LoadedPlugin lp;
        lp.handle = lib;
        lp.lastWrite = std::filesystem::last_write_time(path);

        // NOTE: For true deregistration tracking later, you can modify
        // registerPluginCommands() to record the names per plugin.
        // For now, we can reinitialize by clearing on unload.
        loadedPlugins[name] = lp;

        LOG_DEBUG("PluginManager", "Loaded plugin: " + name);
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
        for (const auto& cmdName : it->second.commands)
            commandMap.erase(cmdName);
        FreeLibrary(it->second.handle);
    } catch (...) {
        LOG_ERROR("PluginManager", "Error while unloading " + name);
    }
}
