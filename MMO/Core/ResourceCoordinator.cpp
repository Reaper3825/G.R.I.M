// ResourceCoordinator implementation
//======================================================//
#include "ResourceCoordinator.hpp"
#include "../../logger.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace GRIM::MMO {

ResourceCoordinator::ResourceCoordinator(
    const HardwareInventory& inventory,
    ResourceSignal& signal,
    const CoordinatorConfig& config)
    : inventory_(inventory)
    , signal_(signal)
    , config_(config) {}

ResourceDecision ResourceCoordinator::requestClaim(const ResourceClaim& claim) {
    if (claim.consumer_id.empty()) {
        throw std::runtime_error("ResourceClaim.consumer_id is empty — caller MUST provide a unique id");
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Reject duplicate claims
    if (holders_.count(claim.consumer_id)) {
        return { DecisionAction::Deny, "consumer_id already holds a claim: " + claim.consumer_id, 0, {} };
    }

    signal_.forceUpdate();
    ResourceSnapshot snap = signal_.latest();

    // ── Model count limit ──
    if (claim.kind == ClaimKind::ModelLoad || claim.kind == ClaimKind::ModelResident) {
        int model_count = 0;
        for (auto& [id, h] : holders_) {
            if (h.kind == ClaimKind::ModelLoad || h.kind == ClaimKind::ModelResident)
                ++model_count;
        }
        if (model_count >= config_.max_loaded_models) {
            if (claim.can_evict) {
                auto targets = findEvictableHolders(claim.vram_mb, claim.preferred_gpu);
                if (!targets.empty()) {
                    return { DecisionAction::EvictThenAllow,
                             "model limit reached; evicting idle models",
                             0, targets };
                }
            }
            if (claim.can_defer) {
                return { DecisionAction::Defer,
                         "model limit reached (" + std::to_string(config_.max_loaded_models) + ")",
                         config_.defer_retry_ms, {} };
            }
            return { DecisionAction::Deny, "model limit reached and claim cannot defer or evict", 0, {} };
        }
    }

    // ── RAM check ──
    if (claim.ram_mb > 0 && !hasSufficientRAM(claim.ram_mb, snap)) {
        if (claim.can_evict) {
            auto targets = findEvictableHolders(0, -1);
            if (!targets.empty()) {
                return { DecisionAction::EvictThenAllow, "insufficient RAM; evicting idle holders", 0, targets };
            }
        }
        if (claim.can_defer) {
            return { DecisionAction::Defer,
                     "insufficient RAM (need " + std::to_string(claim.ram_mb) + " MB)",
                     config_.defer_retry_ms, {} };
        }
        return { DecisionAction::Deny, "insufficient RAM and cannot defer or evict", 0, {} };
    }

    // ── VRAM check ──
    if (claim.vram_mb > 0 && !hasSufficientVRAM(claim.vram_mb, claim.preferred_gpu, snap)) {
        if (claim.can_evict) {
            auto targets = findEvictableHolders(claim.vram_mb, claim.preferred_gpu);
            if (!targets.empty()) {
                return { DecisionAction::EvictThenAllow, "insufficient VRAM; evicting idle holders", 0, targets };
            }
        }
        if (claim.can_defer) {
            return { DecisionAction::Defer,
                     "insufficient VRAM (need " + std::to_string(claim.vram_mb) + " MB on GPU "
                         + std::to_string(claim.preferred_gpu) + ")",
                     config_.defer_retry_ms, {} };
        }
        return { DecisionAction::Deny, "insufficient VRAM and cannot defer or evict", 0, {} };
    }

    // ── Pressure-based throttling ──
    if (snap.pressure == PressureState::Critical) {
        if (claim.can_defer) {
            return { DecisionAction::Defer, "system under critical pressure", config_.defer_retry_ms, {} };
        }
        return { DecisionAction::Throttle, "system under critical pressure, proceeding throttled", 0, {} };
    }

    // ── Admitted — record the holder ──
    ActiveHolder holder;
    holder.consumer_id  = claim.consumer_id;
    holder.kind         = claim.kind;
    holder.ram_mb       = claim.ram_mb;
    holder.vram_mb      = claim.vram_mb;
    holder.gpu_device   = claim.preferred_gpu;
    holder.priority     = claim.priority;
    holder.in_use       = false;
    holder.acquired_time = std::chrono::steady_clock::now();
    holders_[claim.consumer_id] = holder;

    LOG_DEBUG("MMO_COORD", "Claim admitted: " + claim.consumer_id
              + " (RAM=" + std::to_string(claim.ram_mb)
              + " MB, VRAM=" + std::to_string(claim.vram_mb) + " MB)");

    return { DecisionAction::Allow, "admitted", 0, {} };
}

void ResourceCoordinator::releaseClaim(const std::string& consumer_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = holders_.find(consumer_id);
    if (it == holders_.end()) {
        LOG_DEBUG("MMO_COORD", "releaseClaim: no active holder for " + consumer_id);
        return;
    }
    LOG_DEBUG("MMO_COORD", "Claim released: " + consumer_id);
    holders_.erase(it);
}

void ResourceCoordinator::markInUse(const std::string& consumer_id, bool in_use) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = holders_.find(consumer_id);
    if (it == holders_.end()) {
        throw std::runtime_error("markInUse: no active holder for " + consumer_id);
    }
    it->second.in_use = in_use;
}

std::vector<ActiveHolder> ResourceCoordinator::getHolders() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<ActiveHolder> result;
    result.reserve(holders_.size());
    for (auto& [id, h] : holders_) {
        result.push_back(h);
    }
    return result;
}

int ResourceCoordinator::loadedModelCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int count = 0;
    for (auto& [id, h] : holders_) {
        if (h.kind == ClaimKind::ModelLoad || h.kind == ClaimKind::ModelResident)
            ++count;
    }
    return count;
}

// ── Private helpers ──

bool ResourceCoordinator::hasSufficientRAM(long needed_mb, const ResourceSnapshot& snap) const {
    long available = snap.ram_available_mb;
    long reserved  = config_.ram_reserve_mb;
    // Also subtract RAM committed to existing holders
    for (auto& [id, h] : holders_) {
        available -= h.ram_mb;
    }
    return (available - reserved) >= needed_mb;
}

bool ResourceCoordinator::hasSufficientVRAM(long needed_mb, int gpu_device, const ResourceSnapshot& snap) const {
    // Find the matching GPU in snapshot
    for (auto& g : snap.gpus) {
        if (g.device_index == gpu_device) {
            long available = g.vram_free_mb;
            long reserved  = config_.vram_reserve_mb;
            // Subtract VRAM committed to existing holders on this GPU
            for (auto& [id, h] : holders_) {
                if (h.gpu_device == gpu_device)
                    available -= h.vram_mb;
            }
            return (available - reserved) >= needed_mb;
        }
    }
    // No GPU data available — if claim needs VRAM but no GPU snapshot exists, deny
    return needed_mb <= 0;
}

std::vector<std::string> ResourceCoordinator::findEvictableHolders(long vram_needed, int gpu_device) const {
    // Collect idle (not in_use) holders on the target GPU, sorted by priority ascending
    std::vector<const ActiveHolder*> candidates;
    for (auto& [id, h] : holders_) {
        if (h.in_use) continue;
        if (gpu_device >= 0 && h.gpu_device != gpu_device) continue;
        candidates.push_back(&h);
    }

    // Sort: lowest priority first, then oldest first
    std::sort(candidates.begin(), candidates.end(),
        [](const ActiveHolder* a, const ActiveHolder* b) {
            if (a->priority != b->priority) return a->priority < b->priority;
            return a->acquired_time < b->acquired_time;
        });

    std::vector<std::string> targets;
    long reclaimed = 0;
    for (auto* c : candidates) {
        targets.push_back(c->consumer_id);
        reclaimed += (gpu_device >= 0) ? c->vram_mb : c->ram_mb;
        if (vram_needed > 0 && reclaimed >= vram_needed) break;
    }
    return targets;
}

} // namespace GRIM::MMO
