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

class UIPanel;

namespace GRIM::UI {
// Host-owned generation bookkeeping; native panel objects remain owned by the DLL.
struct ProviderPanelLease {
    std::mutex mutex;
    std::atomic<bool> retired{false};
    std::string owner;
    std::string providerName;
    std::vector<std::pair<std::string, uint64_t>> panels;
    std::vector<UIPanel*> nativePanels;
    bool registrationFailed = false;
};
using ProviderInit = GrimResult (*)(const GrimPluginAPI*);
GrimResult initializeProviderPanels(const std::shared_ptr<ProviderPanelLease>& lease,
                                   bool permitted, ProviderInit init, const GrimPluginAPI* api);
bool stageNativePanel(UIPanel* panel);
void forgetNativePanel(UIPanel* panel);
bool validateProviderPanels(const std::shared_ptr<ProviderPanelLease>& lease,
                            const std::shared_ptr<ProviderPanelLease>& replacing = {});
void activateProviderPanels(const std::shared_ptr<ProviderPanelLease>& lease,
                            std::vector<std::string> implementations);
void retireProviderPanels(const std::shared_ptr<ProviderPanelLease>& lease);

// Host-only state: neither a plugin callback nor plugin-owned data enters the UI queue.
struct PanelServiceRequest {
    std::future<nlohmann::json> result;
    std::shared_ptr<std::atomic<bool>> cancelled;
};
PanelServiceRequest submitPanelRequest(GrimUIPanelOperation operation, const std::string& key,
                                       uint64_t expectedRuntimeId = 0);
}
