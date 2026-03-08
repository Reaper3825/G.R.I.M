// Multi-Model Orchestration (MMO) - Resource Signal
// Long-lived sampler thread that periodically publishes a ResourceSnapshot.
// Consumers read the latest snapshot atomically; they never probe hardware themselves.
//======================================================//
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace GRIM::MMO {

// =========================================================
// Per-GPU live utilisation (sampled, not static)
// =========================================================
struct GPULiveState {
    int   device_index       = -1;
    float utilization_pct    = 0.0f;  // 0-100
    long  vram_used_mb       = 0;
    long  vram_free_mb       = 0;
};

// =========================================================
// Pressure state (shared vocabulary for coordinator)
// =========================================================
enum class PressureState : uint8_t {
    Healthy   = 0,
    Pressured = 1,
    Critical  = 2
};

// =========================================================
// Immutable snapshot published by the sampler loop
// =========================================================
struct ResourceSnapshot {
    std::chrono::steady_clock::time_point timestamp{};

    // CPU
    float cpu_utilization_pct = 0.0f;  // 0-100

    // RAM
    long  ram_used_mb      = 0;
    long  ram_available_mb = 0;

    // GPU (per-device)
    std::vector<GPULiveState> gpus;

    // Overall pressure (derived from thresholds)
    PressureState pressure = PressureState::Healthy;
};

// =========================================================
// Configuration for the sampler loop
// =========================================================
struct ResourceSignalConfig {
    int  poll_interval_ms          = 500;   // idle/default
    int  pressured_poll_interval_ms = 150;  // when pressured
    int  critical_poll_interval_ms  = 100;  // when critical

    // Pressure thresholds
    float cpu_pressure_pct         = 85.0f;
    float cpu_critical_pct         = 95.0f;
    float ram_pressure_pct         = 80.0f;
    float ram_critical_pct         = 92.0f;
    float vram_pressure_pct        = 85.0f;
    float vram_critical_pct        = 95.0f;
};

// =========================================================
// ResourceSignal — the sampler service
//
// Usage:
//   ResourceSignal signal;
//   signal.start(config, inventory_gpu_count);
//   ...
//   ResourceSnapshot snap = signal.latest();
//   ...
//   signal.stop();
// =========================================================
class ResourceSignal {
public:
    ResourceSignal() = default;
    ~ResourceSignal();

    // Non-copyable, non-movable (owns a thread)
    ResourceSignal(const ResourceSignal&)            = delete;
    ResourceSignal& operator=(const ResourceSignal&) = delete;

    // Start the sampler thread. gpu_count comes from HardwareInventory.
    void start(const ResourceSignalConfig& cfg, int gpu_count);

    // Stop the sampler thread (blocks until joined).
    void stop();

    // Read the latest snapshot. Lock-free on the read path.
    ResourceSnapshot latest() const;

    // Force an immediate update (useful after model load/unload).
    void forceUpdate();

    bool isRunning() const { return running_.load(std::memory_order_relaxed); }

private:
    void samplerLoop();

    // Sampling implementations (platform-specific in .cpp)
    void sampleCPU(ResourceSnapshot& snap);
    void sampleRAM(ResourceSnapshot& snap);
    void sampleGPUs(ResourceSnapshot& snap);
    PressureState derivePressure(const ResourceSnapshot& snap) const;

    int currentPollMs() const;

    ResourceSignalConfig config_{};
    int gpu_count_ = 0;

    // Shared snapshot (written by sampler thread, read by consumers)
    mutable std::mutex snapshot_mutex_;
    ResourceSnapshot   snapshot_;

    // Sampler thread
    std::atomic<bool> running_{false};
    std::atomic<bool> force_update_{false};
    std::thread       sampler_thread_;

    // CPU sampling state (Windows: previous idle/kernel/user ticks)
#ifdef _WIN32
    uint64_t prev_idle_   = 0;
    uint64_t prev_kernel_ = 0;
    uint64_t prev_user_   = 0;
#endif
};

} // namespace GRIM::MMO
