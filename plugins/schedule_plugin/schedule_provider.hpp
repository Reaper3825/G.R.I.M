#pragma once
#include "core/plugin_api.hpp"
#include "scheduling_service.hpp"
#include <chrono>
namespace grim::scheduling {
class Provider {
    static constexpr auto pollInterval = std::chrono::seconds(1);
    const GrimPluginAPI* api;
    SchedulingService service;
    std::mutex operationMutex;
    std::chrono::steady_clock::time_point nextPoll;
    void publish(bool initial = false) {
        auto state = nlohmann::json{{"items",nlohmann::json::array()},{"ready",false}};
        auto delivery = nlohmann::json{{"persistent",true},{"active",!initial},{"unload_policy","suspend_until_reload"}};
        if (!initial) {
            try {
                const auto snapshot = service.resourceState();
                state = {{"items", snapshot.at("items")}, {"total",snapshot.at("total")}, {"ready",true}};
                delivery["last_poll_utc_ms"] = snapshot.at("last_poll_utc_ms");
                delivery["error"] = snapshot.at("delivery_error");
            } catch (const std::exception& e) { state["error"] = e.what(); }
        }
        const auto text = nlohmann::json::array({
            {{"resource_id","items"},{"kind","schedule.state"},{"name","Scheduled items"},{"runtime_id",uint64_t{1}},{"state",state}},
            {{"resource_id","delivery"},{"kind","schedule.state"},{"name","Schedule delivery"},{"runtime_id",uint64_t{1}},{"state",delivery}}
        }).dump();
        if (api->replace_resources(api,"schedule.state",text.data(),text.size()) != GRIM_OK)
            throw std::runtime_error("Schedule resource publication rejected");
    }
    static void tick(double, double, void* value) noexcept {
        auto& provider = *static_cast<Provider*>(value);
        const auto now = std::chrono::steady_clock::now();
        if (now < provider.nextPoll) return;
        provider.nextPoll = now + pollInterval;
        try { provider.poll(scheduleNow()); }
        catch (...) {} // Provider reports storage errors in its status/resources on the next successful poll.
    }
public:
    explicit Provider(const GrimPluginAPI* value, std::filesystem::path path = {}) : api(value), service(std::move(path)) {}
    void prepare() {
        // Candidate generations do not touch storage or deliver notifications before publication.
        publish(true);
        nextPoll = std::chrono::steady_clock::now() + pollInterval;
        if (api->subscribe_system_tick(api, tick, this) != GRIM_OK)
            throw std::runtime_error("Cannot subscribe to system ticks for schedule polling");
    }
    nlohmann::json execute(const std::string& op,const nlohmann::json& args,int64_t now = scheduleNow()) {
        std::lock_guard lock(operationMutex);
        auto result = service.execute(op,args,now);
        try { publish(); } catch (const std::exception& e) { result["resource_error"] = e.what(); }
        return result;
    }
    void poll(int64_t now) {
        std::lock_guard lock(operationMutex);
        for (const auto& text : service.takeDue(now)) api->show_notification(api,"Schedule",text.c_str());
        publish();
    }
};
}
