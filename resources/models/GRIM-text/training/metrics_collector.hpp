#pragma once

#include <vector>
#include <iostream>
#include <flatbuffers/flatbuffers.h>
#include "training_logs_generated.h"

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <nvml.h>
#endif

using namespace GRIM::Training;

//======================================================//
//  System Metrics Collector
//======================================================//

class MetricsCollector {
public:
    MetricsCollector() {
#ifdef USE_CUDA
        nvmlInit();
        nvmlDeviceGetHandleByIndex(0, &nvml_device_);
#endif
    }
    
    ~MetricsCollector() {
#ifdef USE_CUDA
        nvmlShutdown();
#endif
    }
    
    void collectGPUMetrics(flatbuffers::FlatBufferBuilder& builder, 
    std::vector<flatbuffers::Offset<GPUMetrics>>& gpu_metrics) {
#ifdef USE_CUDA
        nvmlUtilization_t utilization;
        nvmlMemory_t memory;
        unsigned int temp, power, sm_clock, mem_clock;
        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
        
        nvmlDeviceGetUtilizationRates(nvml_device_, &utilization);
        nvmlDeviceGetMemoryInfo(nvml_device_, &memory);
        nvmlDeviceGetTemperature(nvml_device_, NVML_TEMPERATURE_GPU, &temp);
        nvmlDeviceGetPowerUsage(nvml_device_, &power);
        nvmlDeviceGetClockInfo(nvml_device_, NVML_CLOCK_SM, &sm_clock);
      nvmlDeviceGetClockInfo(nvml_device_, NVML_CLOCK_MEM, &mem_clock);
 nvmlDeviceGetName(nvml_device_, name, NVML_DEVICE_NAME_BUFFER_SIZE);

        size_t free_mem = 0, total_mem = 0;
        cudaError_t mem_err = cudaMemGetInfo(&free_mem, &total_mem);
        if (mem_err != cudaSuccess) {
    std::cerr << "[GPU Monitor] Warning: cudaMemGetInfo failed: " 
     << cudaGetErrorString(mem_err) << std::endl;
 free_mem = (memory.total - memory.used) * 1024 * 1024;
    total_mem = memory.total * 1024 * 1024;
        }
        
        auto gpu_name = builder.CreateString(name);
    auto metrics = CreateGPUMetrics(builder,
       0,  // device_id
     gpu_name,
            static_cast<float>(utilization.gpu),
     memory.used / (1024 * 1024),
            memory.total / (1024 * 1024),
        (total_mem - free_mem) / (1024 * 1024),
            total_mem / (1024 * 1024),
            static_cast<float>(temp),
   static_cast<float>(power) / 1000.0f,
            sm_clock,
            mem_clock
    );
        
        gpu_metrics.push_back(metrics);
#endif
    }
    
    void collectSystemMetrics(uint64_t& ram_used_mb, uint64_t& ram_total_mb, float& cpu_percent) {
#ifdef _WIN32
        MEMORYSTATUSEX mem_info;
        mem_info.dwLength = sizeof(MEMORYSTATUSEX);
     GlobalMemoryStatusEx(&mem_info);
        
     ram_total_mb = mem_info.ullTotalPhys / (1024 * 1024);
   ram_used_mb = (mem_info.ullTotalPhys - mem_info.ullAvailPhys) / (1024 * 1024);
        cpu_percent = static_cast<float>(mem_info.dwMemoryLoad);
#else
        ram_used_mb = 0;
        ram_total_mb = 0;
        cpu_percent = 0.0f;
#endif
    }
    
private:
#ifdef USE_CUDA
    nvmlDevice_t nvml_device_;
#endif
};
