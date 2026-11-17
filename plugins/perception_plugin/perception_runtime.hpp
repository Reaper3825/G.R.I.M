#pragma once
#include "core/plugin_api.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <mutex>
#include <string>
namespace perceptionPlugin {
void configureOperator(const std::string&, const nlohmann::json&);
class Runtime {
public:
    explicit Runtime(grim::PluginHost host);
    void start();
    ~Runtime();
    nlohmann::json request(const std::string&, const nlohmann::json&);
private:
    static constexpr auto pollInterval = std::chrono::milliseconds(33);
    static void tick(double, double, void*);
    void pump();
    void stopPhysical();
    void publish();
    nlohmann::json snapshot();
    grim::PluginHost api;
    std::chrono::steady_clock::time_point nextPoll;
    std::mutex mutex;
    bool physical = false;
    std::string lastError;
    uint64_t ticks = 0;
};
}
