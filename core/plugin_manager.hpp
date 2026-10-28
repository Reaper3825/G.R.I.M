#pragma once
#include <string>
#include <unordered_map>
#include <filesystem>
#include "grim_platform.h"

// ====================================================
// Plugin Manager (manages DLL hot-reload plugins)
// ====================================================
class PluginManager {
public:
    static void initialize(const std::string& folder = "plugins");
    static void checkForHotReload();
    static void unloadAll();

private:
    struct LoadedPlugin {
        void* handle;  // HMODULE on Windows, void* from dlopen on Unix
        std::filesystem::file_time_type lastWrite;
        std::string declaredName; // Same identity used by API registration/cleanup.
    };

    static inline std::filesystem::path pluginDir;
    static inline std::unordered_map<std::string, LoadedPlugin> loadedPlugins;

    static void loadPlugin(const std::filesystem::path& path);
    static void unloadPlugin(const std::string& name);
};
