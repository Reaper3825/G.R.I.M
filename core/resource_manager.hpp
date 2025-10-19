#pragma once
#include <nlohmann/json.hpp>

namespace GRIM {

struct ResourceSnapshot {
    float cpuUsage = 0.0f;
    float ramUsageMB = 0.0f;
    float ramTotalMB = 0.0f;
};

class ResourceManager {
public:
    static void init();
    static void update();
    static ResourceSnapshot getSnapshot();
    static nlohmann::json toJSON();

private:
    static void sampleCPU();
    static void sampleMemory();
};

} // namespace GRIM
