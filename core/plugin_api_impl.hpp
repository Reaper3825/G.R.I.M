#pragma once
#include "plugin_api.hpp"
#include "plugin_manager.hpp"
#include "../MMO/Core/Registries/ProviderRegistry.hpp"
#include <nlohmann/json.hpp>

namespace PluginAPI {

struct PluginModule {
    GRIM::MMO::ProviderDescriptor descriptor;
    nlohmann::json tools;
    std::unique_ptr<void, void (*)(void*)> library{nullptr, +[](void*) {}};
    decltype(&grim_plugin_init) init = nullptr;
    decltype(&grim_plugin_shutdown) shutdown = nullptr;
    GrimToolDispatch dispatch = nullptr;
};

using OpenPlugin = PluginModule (*)(const std::filesystem::path&, PluginLoadStatus& failure);

class ProviderBatch {
public:
    static std::vector<PluginLoadOutcome> load(const std::vector<std::filesystem::path>& libraries, OpenPlugin open);
    static std::size_t shutdown();

#ifdef GRIM_PLUGIN_BATCH_TESTS
    static GrimPluginHandle loadFixture(const GRIM::MMO::ProviderDescriptor& descriptor,
                                       const nlohmann::json& tools, GrimToolDispatch dispatch);
    static void stopFixture(GrimPluginHandle handle);
#endif

private:
    using Registry = GRIM::MMO::ProviderRegistry;
    using Record = Registry::Record;
    static Record stage(PluginModule plugin);
    static void initialize(const Record& record, decltype(&grim_plugin_init) init,
                           decltype(&grim_plugin_shutdown) shutdown);
    static void stop(const std::vector<Record>& records);
};

const GrimPluginAPI& hostAPI();
void dispatchSystemTick(double deltaSeconds, double offsetSeconds);

} // namespace PluginAPI
