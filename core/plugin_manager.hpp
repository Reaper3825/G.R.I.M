#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include "plugin_api.hpp"

// Lifecycle and reconciliation run on the host frame thread. Reload requests
// are queued so a plugin never unloads itself from inside its own callback.
class PluginManager {
public:
    static void initialize(const std::string& folder = "plugins");
    static void checkForHotReload();
    static void tick();
    static void requestReload(const GrimPluginAPI* api);
    static void unloadAll();

private:
    struct Fingerprint {
        std::filesystem::file_time_type modified;
        std::uintmax_t size = 0;
        std::uint64_t hash = 0;

        bool operator==(const Fingerprint&) const = default;
    };
    struct LoadedPlugin {
        void* handle = nullptr;
        GrimPluginAPI* api = nullptr;
        std::string declaredName;
        Fingerprint fingerprint;
        std::filesystem::path cacheDirectory;
    };
    static inline std::filesystem::path pluginDir;
    static inline std::unordered_map<std::string, LoadedPlugin> loadedPlugins;
    struct Attempt { Fingerprint fingerprint; uint64_t surfaceRevision = 0; };
    static inline std::unordered_map<std::string, Attempt> attempted;
    static inline uint64_t surfaceRevision = 0;
    static inline std::mutex requestMutex;
    static inline std::unordered_set<const GrimPluginAPI*> reloadRequests;
    static inline std::chrono::steady_clock::time_point nextPoll;
    static std::optional<Fingerprint> fingerprint(const std::filesystem::path& path);
    static bool loadPlugin(const std::filesystem::path& path, const Fingerprint& fingerprint);
    static void retire(LoadedPlugin& plugin);
    static void unloadPlugin(const std::string& name);
};
