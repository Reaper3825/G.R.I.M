#pragma once
#include "core/plugin_api.hpp"
#include <atomic>
#include <future>
#include <memory>
#include <string>
#include <nlohmann/json.hpp>

namespace GRIM::UI {
// Host-only state: neither a plugin callback nor plugin-owned data enters the UI queue.
struct PanelServiceRequest {
    std::future<nlohmann::json> result;
    std::shared_ptr<std::atomic<bool>> cancelled;
};
PanelServiceRequest submitPanelRequest(GrimUIPanelOperation operation, const std::string& key);
}
