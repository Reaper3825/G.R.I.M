#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <memory>
#include "plugin_api.hpp"

namespace PluginAPI { struct LoadedProvider; }

// Lifecycle and reconciliation run on the host frame thread. Reload requests
// are queued so a plugin never unloads itself from inside its own callback.
class PluginManager {
public:
    static void initialize(const std::string& folder = "plugins");
    static void checkForHotReload();
    static void tick();
    static void requestReload(GrimPluginHandle instance);
    static void unloadAll();

private:
    struct Fingerprint {
        std::filesystem::file_time_type modified;
        std::uintmax_t size = 0;
        std::uint64_t hash = 0;

        bool operator==(const Fingerprint&) const = default;
    };
    struct PluginFingerprint {
        Fingerprint library;
        Fingerprint manifest;
        bool operator==(const PluginFingerprint&) const = default;
    };
    struct LoadedPlugin {
        void* handle = nullptr;
        std::shared_ptr<PluginAPI::LoadedProvider> provider;
        bool initialized = false;
        std::string providerName;
        PluginFingerprint fingerprint;
        std::filesystem::path cacheDirectory;
    };
    static inline std::filesystem::path pluginDir;
    static inline std::unordered_map<std::string, LoadedPlugin> loadedPlugins;
    struct Attempt { PluginFingerprint fingerprint; uint64_t surfaceRevision = 0; };
    static inline std::unordered_map<std::string, Attempt> attempted;
    static inline uint64_t surfaceRevision = 0;
    static inline std::mutex requestMutex;
    static inline std::unordered_set<GrimPluginHandle> reloadRequests;
    static inline std::chrono::steady_clock::time_point nextPoll;
    static std::optional<Fingerprint> fingerprint(const std::filesystem::path& path);
    static bool loadPlugin(const std::filesystem::path& path, const PluginFingerprint& fingerprint);
    static void retire(LoadedPlugin& plugin);
    static void unloadPlugin(const std::string& name);
};
