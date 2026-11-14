#pragma once
#include "core/plugin_api.hpp"
#include <atomic>
#include <future>
#include <memory>
#include <string>
#include <mutex>
#include <vector>
#include <utility>
#include <map>
#include <nlohmann/json.hpp>

namespace GRIM::UI {
// Host-owned lease survives DLL unload and queued work. No plugin callbacks.
struct ProviderPanelLease {
    std::mutex mutex;
    std::atomic<bool> retired{false};
    std::string owner;
    std::string providerName;
    std::vector<std::pair<std::string, uint64_t>> panels;
    std::map<std::string, uint64_t> definitionRevisions;
    std::mutex eventsMutex;
    std::vector<nlohmann::json> events;
};
void activateProviderPanels(const std::shared_ptr<ProviderPanelLease>& lease,
                            std::vector<std::string> implementations,
                            std::vector<nlohmann::json> definitions = {});
void retireProviderPanels(const std::shared_ptr<ProviderPanelLease>& lease);
void updateProviderPanel(const std::shared_ptr<ProviderPanelLease>& lease, nlohmann::json definition);

// Host-only state: neither a plugin callback nor plugin-owned data enters the UI queue.
struct PanelServiceRequest {
    std::future<nlohmann::json> result;
    std::shared_ptr<std::atomic<bool>> cancelled;
};
PanelServiceRequest submitPanelRequest(GrimUIPanelOperation operation, const std::string& key,
                                       uint64_t expectedRuntimeId = 0);
}
