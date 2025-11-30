# Resource Monitor Usage Example

The `ResourceMonitor` is now a modular singleton that can be used anywhere in your application.

## Basic Usage

```cpp
#include "hardware/resource_values.hpp"

// In your main initialization (do this once at startup)
ResourceMonitor::getInstance().initialize();

// In your update loop (call periodically, e.g., every 0.5s)
ResourceMonitor::getInstance().update();

// Get resource values anywhere in your code
float cpu = ResourceMonitor::getInstance().getCpuUsage();
float memory = ResourceMonitor::getInstance().getMemoryUsage();
float gpu = ResourceMonitor::getInstance().getGpuUsage();

// Or get everything at once
ResourceUsage usage = ResourceMonitor::getInstance().getCurrentUsage();
std::cout << "CPU: " << usage.cpuUsage << "%" << std::endl;
std::cout << "Memory: " << usage.memoryUsage << "%" << std::endl;
std::cout << "GPU: " << usage.gpuUsage << "%" << std::endl;
```

## Example: Adding Resource Monitoring to Another UI Panel

```cpp
// In your other panel's update method
void MyOtherPanel::update(float dt) {
    // Get current resource values
    ResourceUsage usage = ResourceMonitor::getInstance().getCurrentUsage();
    
    // Display them however you want
    drawText("CPU: " + std::to_string(usage.cpuUsage) + "%");
    drawText("Memory: " + std::to_string(usage.memoryUsage) + "%");
    drawText("GPU: " + std::to_string(usage.gpuUsage) + "%");
}
```

## Example: Logging Resource Usage

```cpp
void logResourceUsage() {
    ResourceUsage usage = ResourceMonitor::getInstance().getCurrentUsage();
    LOG_DEBUG("Resources", 
        "CPU: " + std::to_string(usage.cpuUsage) + "% | " +
        "Memory: " + std::to_string(usage.memoryUsage) + "% | " +
        "GPU: " + std::to_string(usage.gpuUsage) + "%"
    );
}
```

## Thread Safety

The `ResourceMonitor` is thread-safe. You can call any getter method from any thread without worrying about race conditions.

## Performance

- The monitor has built-in throttling to avoid excessive overhead
- `update()` will skip if called too frequently (< 100ms between calls)
- All getters are fast O(1) lookups with minimal locking
