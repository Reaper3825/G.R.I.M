#pragma once
#include <string>
#include <unordered_map>
#include <filesystem>
#include <windows.h>
#include <vector>
#include "commands_core.hpp"

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
        HMODULE handle;
        std::filesystem::file_time_type lastWrite;
        std::vector<std::string> commands;
    };

    static inline std::filesystem::path pluginDir;
    static inline std::unordered_map<std::string, LoadedPlugin> loadedPlugins;

    static void loadPlugin(const std::filesystem::path& path);
    static void unloadPlugin(const std::string& name);
};
