#include "resource_manager.hpp"
#include "logger.hpp"

#ifdef _WIN32
#include "grim_platform.h"
#include <pdh.h>
#pragma comment(lib, "pdh.lib")
#endif

namespace GRIM {

static ResourceSnapshot g_snapshot;
static bool initialized = false;

#ifdef _WIN32
static PDH_HQUERY cpuQuery;
static PDH_HCOUNTER cpuTotal;
#endif

void ResourceManager::init() {
#ifdef _WIN32
    if (PdhOpenQuery(nullptr, 0, &cpuQuery) == ERROR_SUCCESS) {
        PdhAddEnglishCounter(cpuQuery, "\\Processor(_Total)\\% Processor Time", 0, &cpuTotal);
        PdhCollectQueryData(cpuQuery);
        initialized = true;
        LOG_DEBUG("ResourceManager", "Initialized PDH CPU counters");
    } else {
        LOG_ERROR("ResourceManager", "Failed to initialize PDH CPU query");
    }
#endif
}

void ResourceManager::update() {
#ifdef _WIN32
    if (!initialized) return;
    sampleCPU();
    sampleMemory();
#endif
}

void ResourceManager::sampleCPU() {
#ifdef _WIN32
    PdhCollectQueryData(cpuQuery);
    PDH_FMT_COUNTERVALUE val;
    PdhGetFormattedCounterValue(cpuTotal, PDH_FMT_DOUBLE, nullptr, &val);
    g_snapshot.cpuUsage = static_cast<float>(val.doubleValue);
#endif
}

void ResourceManager::sampleMemory() {
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    g_snapshot.ramUsageMB =
        static_cast<float>((memInfo.ullTotalPhys - memInfo.ullAvailPhys) / (1024.0 * 1024.0));
    g_snapshot.ramTotalMB =
        static_cast<float>(memInfo.ullTotalPhys / (1024.0 * 1024.0));
#endif
}

ResourceSnapshot ResourceManager::getSnapshot() {
    return g_snapshot;
}

nlohmann::json ResourceManager::toJSON() {
    return {
        {"cpu_usage_percent", g_snapshot.cpuUsage},
        {"ram_used_mb", g_snapshot.ramUsageMB},
        {"ram_total_mb", g_snapshot.ramTotalMB}
    };
}

} // namespace GRIM
