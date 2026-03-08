// Multi-Model Orchestration (MMO) - Model Loader
// Resource-aware model lifecycle state machine with use-degrading.
//
// State machine:
//   Unloaded → Loading → Loaded → InUse → Idle → EvictEligible → Unloading → Unloaded
//
// Use-degrading: the more a model is used, the longer it stays resident
// before becoming evict-eligible. Idle TTL grows with cumulative use_count.
//
// All resource decisions go through ResourceCoordinator — the loader
// never probes hardware independently.
//======================================================//
#pragma once

#include "ModelRegistry.hpp"
#include "ResourceCoordinator.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

namespace GRIM::MMO {

// =========================================================
// Residency state — per the plan's lifecycle diagram
// =========================================================
enum class ResidencyState : uint8_t {
    Unloaded      = 0,
    Loading       = 1,
    Loaded        = 2,
    InUse         = 3,
    Idle          = 4,
    EvictEligible = 5,
    Unloading     = 6
};

const char* residencyStateToString(ResidencyState s);

// =========================================================
// Per-model runtime record (internal to ModelLoader)
// =========================================================
struct ModelSlot {
    std::string    model_id;
    ResidencyState state       = ResidencyState::Unloaded;
    int            use_count   = 0;   // cumulative requests served
    int            gpu_device  = 0;

    std::chrono::steady_clock::time_point loaded_time{};
    std::chrono::steady_clock::time_point last_use_time{};
    std::chrono::steady_clock::time_point idle_since{};
};

// =========================================================
// Loader configuration (from ai_config.json → mmo.model_loader)
// =========================================================
struct ModelLoaderConfig {
    int  load_timeout_ms   = 30000;   // max wait for a model to become ready
    int  idle_ttl_ms       = 60000;   // base idle time before evict-eligible
    int  hot_ttl_cap_ms    = 300000;  // max idle TTL even for heavily used models
    int  use_degrade_step_ms = 5000;  // extra TTL per use_count
};

// =========================================================
// Load result — returned by ensureLoaded()
// =========================================================
enum class LoadResult : uint8_t {
    Ok                    = 0,
    AlreadyLoaded         = 1,
    LoadedAfterEviction   = 2,
    Deferred              = 3,
    Unavailable           = 4   // ERR_MMO_MODEL_UNAVAILABLE
};

const char* loadResultToString(LoadResult r);

// =========================================================
// ModelLoader
//
// Usage:
//   ModelLoader loader(registry, coordinator, config);
//   auto result = loader.ensureLoaded("medical-7b");
//   if (result == LoadResult::Ok || result == LoadResult::AlreadyLoaded) {
//       loader.markInUse("medical-7b");
//       // ... serve request ...
//       loader.markIdle("medical-7b");
//   }
//
// The loader serializes all operations under its own mutex.
// It delegates resource admission to ResourceCoordinator.
// =========================================================
class ModelLoader {
public:
    ModelLoader(const ModelRegistry& registry,
                ResourceCoordinator& coordinator,
                const ModelLoaderConfig& config);

    // Ensure a model is loaded and ready.
    // Submits a resource claim, handles eviction if needed,
    // and transitions the model through the state machine.
    // Returns Unavailable on failure — never silently reroutes.
    LoadResult ensureLoaded(const std::string& model_id);

    // Mark a loaded model as actively serving a request.
    // Prevents eviction. Throws if model is not in Loaded/Idle/EvictEligible state.
    void markInUse(const std::string& model_id);

    // Mark a model as no longer serving a request.
    // Starts the idle/use-degrading timer.
    void markIdle(const std::string& model_id);

    // Explicitly unload a model (shutdown or forced eviction).
    // Throws if model is InUse.
    void unload(const std::string& model_id);

    // Tick idle timers — call periodically (e.g. from resource signal loop
    // or a dedicated maintenance timer). Transitions Idle → EvictEligible
    // when the use-degraded TTL has elapsed.
    void tickIdleTimers();

    // Query the current state of a model.
    ResidencyState getState(const std::string& model_id) const;

    // Query the slot record for a model (or nullptr if unknown).
    const ModelSlot* getSlot(const std::string& model_id) const;

    // How many models are currently loaded (Loading + Loaded + InUse + Idle + EvictEligible).
    int loadedCount() const;

    // Compute the effective idle TTL for a model based on use-degrading.
    int effectiveIdleTtlMs(const ModelSlot& slot) const;

    // Unload all models (shutdown).
    void unloadAll();

    // Register a callback invoked when a model needs its backend process
    // started or stopped. The ModelLoader does not own process management
    // directly — it delegates to whatever process manager the body provides.
    using StartCallback = std::function<bool(const ModelInfo& model)>;
    using StopCallback  = std::function<void(const ModelInfo& model)>;

    void setStartCallback(StartCallback cb);
    void setStopCallback(StopCallback cb);

private:
    ModelSlot& getOrCreateSlot(const std::string& model_id);
    void evictTargets(const std::vector<std::string>& targets);

    const ModelRegistry&   registry_;
    ResourceCoordinator&   coordinator_;
    ModelLoaderConfig      config_;

    mutable std::mutex     mutex_;
    std::unordered_map<std::string, ModelSlot> slots_;

    StartCallback start_cb_;
    StopCallback  stop_cb_;
};

} // namespace GRIM::MMO
