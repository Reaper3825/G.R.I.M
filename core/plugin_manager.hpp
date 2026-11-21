#pragma once
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

enum class PluginLoadStatus {
    Loaded, InvalidBundle, NativeLoadFailed, RegistrationRejected, InitializationFailed
};

struct PluginLoadOutcome {
    std::filesystem::path library;
    PluginLoadStatus status;
    std::string detail;
};

struct PluginBatchResult {
    enum class Status { Completed, AlreadyStarted, DiscoveryFailed };
    Status status;
    std::vector<PluginLoadOutcome> plugins;
};

// Startup and shutdown are serialized by the application lifecycle thread.
class PluginManager {
public:
    static PluginBatchResult loadBatch(const std::filesystem::path& folder = "deployed_plugins");
    static std::size_t shutdownBatch();
};
