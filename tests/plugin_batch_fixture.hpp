#pragma once
#include "core/plugin_api_impl.hpp"
#include <utility>

class PluginBatchFixture {
public:
    explicit PluginBatchFixture(GRIM::MMO::ProviderDescriptor descriptor)
        : descriptor_(std::move(descriptor)) {}
    PluginBatchFixture(const PluginBatchFixture&) = delete;
    PluginBatchFixture& operator=(const PluginBatchFixture&) = delete;
    ~PluginBatchFixture() { unload(); }
    void load(const nlohmann::json& tools = nlohmann::json::array(), GrimToolDispatch dispatch = nullptr) {
        handle_ = PluginAPI::ProviderBatch::loadFixture(descriptor_, tools, dispatch);
    }
    void unload() {
        if (handle_) PluginAPI::ProviderBatch::stopFixture(std::exchange(handle_, 0));
    }
    GrimPluginHandle handle() const { return handle_; }
    const GrimPluginAPI* operator->() const { return &PluginAPI::hostAPI(); }

private:
    GRIM::MMO::ProviderDescriptor descriptor_;
    GrimPluginHandle handle_ = 0;
};
