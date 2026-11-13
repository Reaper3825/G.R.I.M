#pragma once
#include "core/plugin_api.hpp"
#include <chrono>
#include <mutex>
#include <vector>
#include <nlohmann/json.hpp>
namespace grim::timers {
class Provider {
    struct Timer { uint64_t id; std::chrono::steady_clock::time_point expiry; std::string message; };
    const GrimPluginAPI* api;
    std::mutex mutex;
    std::vector<Timer> pending;
    uint64_t nextId = 1;
    GrimTimerHandle poller = nullptr;
    void publish(const std::vector<Timer>& values) {
        auto items = nlohmann::json::array();
        for (const auto& timer : values) items.push_back({{"id",timer.id},{"message",timer.message},
            {"deadline_steady_ms",std::chrono::duration_cast<std::chrono::milliseconds>(timer.expiry.time_since_epoch()).count()}});
        const auto text = nlohmann::json::array({{{"resource_id","queue"},{"kind","timers.queue"},
            {"name","Countdown timers"},{"runtime_id",uint64_t{1}},
            {"state",{{"items",items},{"pending",values.size()},{"persistent",false},{"unload_policy","cancel"}}}}}).dump();
        if (api->replace_resources(api,"timers.queue",text.data(),text.size()) != GRIM_OK)
            throw std::runtime_error("Timer resource publication rejected");
    }
    static void tick(uint64_t, void* state) noexcept {
        try { static_cast<Provider*>(state)->poll(); }
        catch (...) {} // Keep due work queued if publication fails; never unwind into the frame loop.
    }
public:
    explicit Provider(const GrimPluginAPI* value): api(value) {}
    ~Provider() { if (poller) api->cancel_timer(api,poller); }
    void prepare() {
        publish(pending);
        poller = api->create_timer(api,100,true,tick,this);
        if (!poller) throw std::runtime_error("Cannot schedule timer provider polling");
    }
    void add(int seconds, std::string message) {
        std::lock_guard lock(mutex);
        if (pending.size() >= 10000) throw std::runtime_error("Timer queue is full");
        auto next = pending;
        next.push_back({nextId,std::chrono::steady_clock::now()+std::chrono::seconds(seconds),std::move(message)});
        publish(next); pending.swap(next); ++nextId;
    }
    void poll(std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now()) {
        std::lock_guard lock(mutex);
        std::vector<Timer> next; std::vector<std::string> due;
        for (const auto& timer : pending) {
            if (now >= timer.expiry) due.push_back(timer.message); else next.push_back(timer);
        }
        if (due.empty()) return;
        publish(next); pending.swap(next);
        for (const auto& message : due) api->show_notification(api,"Timer",message.c_str());
    }
};
}
