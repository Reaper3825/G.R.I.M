#pragma once
#include "core/plugin_api.hpp"
#include <nlohmann/json.hpp>
#include <mutex>
#include <string>
namespace perceptionPlugin {
void configureOperator(const std::string&, const nlohmann::json&);
class Runtime {
public:
    explicit Runtime(const GrimPluginAPI*);
    ~Runtime();
    nlohmann::json request(const std::string&, const nlohmann::json&);
private:
    static void tick(uint64_t, void*);
    void pump();
    void stopPhysical();
    void publish();
    nlohmann::json snapshot();
    const GrimPluginAPI* api;
    GrimTimerHandle timer = nullptr;
    std::mutex mutex;
    bool physical = false;
    std::string lastError;
    uint64_t ticks = 0;
};
}
