// Multi-Model Orchestration (MMO) - Resource Coordinator
// Single admission/reservation authority for all resource-consuming operations.
// Model loading, tool/plugin loading, perception capture, server startup —
// all must go through here instead of probing hardware independently.
//======================================================//
#pragma once

#include "HardwareInventory.hpp"
#include "ResourceSignal.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace GRIM::MMO {

// =========================================================
// Resource claim — submitted by consumers before starting work
// =========================================================
enum class ClaimKind : uint8_t {
    ModelLoad      = 0,
    ModelResident   = 1,
    ToolLoad       = 2,
    ProcessStart   = 3,
    PerceptionJob  = 4
};

struct ResourceClaim {
    std::string consumer_id;        // unique id of the requester
    ClaimKind   kind          = ClaimKind::ModelLoad;
    long        ram_mb        = 0;  // estimated RAM needed
    long        vram_mb       = 0;  // estimated VRAM needed
    int         preferred_gpu = 0;  // device index (-1 = any)
    int         priority      = 0;  // higher = more important
    bool        can_defer     = true;
    bool        can_evict     = false;  // may evict idle holders
};

// =========================================================
// Resource decision — returned by the coordinator
// =========================================================
enum class DecisionAction : uint8_t {
    Allow          = 0,
    Defer          = 1,
    Throttle       = 2,
    EvictThenAllow = 3,
    Deny           = 4
};

struct ResourceDecision {
    DecisionAction action       = DecisionAction::Deny;
    std::string    reason;
    int            retry_after_ms = 0;
    std::vector<std::string> eviction_targets;  // consumer_ids to evict
};

// =========================================================
// Active holder — tracked by the coordinator
// =========================================================
struct ActiveHolder {
    std::string consumer_id;
    ClaimKind   kind          = ClaimKind::ModelLoad;
    long        ram_mb        = 0;
    long        vram_mb       = 0;
    int         gpu_device    = 0;
    int         priority      = 0;
    bool        in_use        = false;  // actively serving a request
    std::chrono::steady_clock::time_point acquired_time{};
};

// =========================================================
// Coordinator configuration
// =========================================================
struct CoordinatorConfig {
    long  ram_reserve_mb   = 512;   // keep this much RAM free
    long  vram_reserve_mb  = 256;   // keep this much VRAM free per GPU
    int   max_loaded_models = 4;
    int   defer_retry_ms   = 500;
};

// =========================================================
// ResourceCoordinator
//
// Usage:
//   ResourceCoordinator coord(inventory, signal, config);
//   auto decision = coord.requestClaim(claim);
//   if (decision.action == DecisionAction::Allow) { ... load ... }
//   coord.releaseClaim("my_consumer_id");
// =========================================================
class ResourceCoordinator {
public:
    ResourceCoordinator(const HardwareInventory& inventory,
                        ResourceSignal& signal,
                        const CoordinatorConfig& config);

    // Submit a resource claim. Returns a decision.
    ResourceDecision requestClaim(const ResourceClaim& claim);

    // Release a held claim (after unload/stop/completion).
    void releaseClaim(const std::string& consumer_id);

    // Mark a holder as actively in-use (prevents eviction).
    void markInUse(const std::string& consumer_id, bool in_use);

    // Query current holders.
    std::vector<ActiveHolder> getHolders() const;

    // How many models are currently loaded.
    int loadedModelCount() const;

private:
    bool hasSufficientRAM(long needed_mb, const ResourceSnapshot& snap) const;
    bool hasSufficientVRAM(long needed_mb, int gpu_device, const ResourceSnapshot& snap) const;
    std::vector<std::string> findEvictableHolders(long vram_needed, int gpu_device) const;

    const HardwareInventory& inventory_;
    ResourceSignal&          signal_;
    CoordinatorConfig        config_;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, ActiveHolder> holders_;
};

} // namespace GRIM::MMO
