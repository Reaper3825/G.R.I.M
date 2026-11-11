#pragma once
#include "MMO/Core/ProviderRegistry.hpp"
#include <functional>
#include <mutex>
#include <map>
#include <algorithm>

namespace GRIM::Applications {
struct Application {
    std::string identityNamespace, nativeId, name;
    std::string launchKind, launchTarget, icon;
    bool operator==(const Application&) const = default;
};
// Backends are host-owned. A failed scan must throw: its last good snapshot is retained.
struct Backend {
    std::string name;
    std::function<std::vector<Application>()> discover;
};
using Launcher = std::function<void(const Application&)>;
std::vector<Backend> systemBackends();
Backend nativeBackend();
void launchNative(const Application& application);

// Serializes refresh and launch, owns copied data only, never plugin callbacks.
class Catalog {
public:
    Catalog(MMO::ProviderDescriptor owner, std::vector<Backend> backends, Launcher launcher,
            MMO::ProviderRegistry& registry = MMO::ProviderRegistry::instance());
    nlohmann::json request(const std::string& operation, const nlohmann::json& arguments);
private:
    struct Entry { Application app; std::vector<std::string> sources; uint64_t runtime = 0; };
    void refresh();
    nlohmann::json snapshot(size_t offset = 0, size_t limit = 25) const;
    MMO::ResourceDescriptor resource(const std::string& id, const Entry& entry) const;
    MMO::ProviderDescriptor owner_;
    std::vector<Backend> backends_;
    Launcher launcher_;
    MMO::ProviderRegistry& registry_;
    std::mutex mutex_;
    std::map<std::string, std::vector<Application>> backendSnapshots_;
    std::map<std::string, Entry> entries_;
    nlohmann::json errors_ = nlohmann::json::array();
    uint64_t nextRuntime_ = 1;
};
}
