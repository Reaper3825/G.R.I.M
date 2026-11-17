#pragma once
#include "core/plugin_api.hpp"
#include <atomic>
#include <future>
#include <memory>
#include <string>
#include <string_view>
#include <mutex>
#include <vector>
#include <utility>
#include <map>
#include <nlohmann/json.hpp>

class UIPanel;

namespace GRIM::UI {
// Host-owned generation bookkeeping; native panel objects remain owned by the DLL.
struct ProviderPanelLease {
    std::atomic<bool> retired{false};
    std::string owner;
    std::vector<std::pair<std::string, uint64_t>> panels;
    bool uiPermitted = false;
};
void registerProviderPanelLease(GrimPluginHandle instance, const std::shared_ptr<ProviderPanelLease>& lease);
void attachNativePanel(UIPanel* panel, GrimPluginHandle instance);
void retireProviderPanels(const std::shared_ptr<ProviderPanelLease>& lease);

// Host-only state: neither a plugin callback nor plugin-owned data enters the UI queue.
struct PanelServiceRequest {
    std::future<nlohmann::json> result;
    std::shared_ptr<std::atomic<bool>> cancelled;
};
PanelServiceRequest submitPanelRequest(GrimUIPanelOperation operation, const std::string& key,
                                       uint64_t expectedRuntimeId = 0);
}
