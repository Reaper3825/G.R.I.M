#pragma once
#include "core/plugin_api_impl.hpp"
#include <nlohmann/json.hpp>
#include <utility>

class LoadedProviderFixture {
public:
    explicit LoadedProviderFixture(GRIM::MMO::ProviderDescriptor descriptor)
        : provider_(PluginAPI::registerInstance(std::move(descriptor))) {}
    LoadedProviderFixture(const LoadedProviderFixture&) = delete;
    LoadedProviderFixture& operator=(const LoadedProviderFixture&) = delete;
    ~LoadedProviderFixture() { unload(); }
    void load(const nlohmann::json& tools = nlohmann::json::array(), GrimToolDispatch dispatch = nullptr) {
        PluginAPI::registerProvider(*provider_, tools, dispatch);
        PluginAPI::enableDispatch(*provider_);
    }
    void unload() {
        if (!provider_) return;
        PluginAPI::retireProvider(*provider_);
        PluginAPI::releaseInstance(*provider_);
        provider_.reset();
    }
    GrimPluginHandle handle() const { return PluginAPI::instanceHandle(*provider_); }
    const GrimPluginAPI* operator->() const { return &PluginAPI::hostAPI(); }

private:
    std::shared_ptr<PluginAPI::LoadedProvider> provider_;
};
