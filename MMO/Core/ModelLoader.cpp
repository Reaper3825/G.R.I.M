// Multi-Model Orchestration (MMO) - Model Loader
// See ModelLoader.hpp for interface documentation.
//======================================================//
#include "ModelLoader.hpp"

#include "../../logger.hpp"

#include <algorithm>
#include <stdexcept>

namespace GRIM::MMO {

// =========================================================
// String conversions
// =========================================================

const char* residencyStateToString(ResidencyState s) {
    switch (s) {
        case ResidencyState::Unloaded:      return "Unloaded";
        case ResidencyState::Loading:       return "Loading";
        case ResidencyState::Loaded:        return "Loaded";
        case ResidencyState::InUse:         return "InUse";
        case ResidencyState::Idle:          return "Idle";
        case ResidencyState::EvictEligible: return "EvictEligible";
        case ResidencyState::Unloading:     return "Unloading";
    }
    return "Unknown";
}

const char* loadResultToString(LoadResult r) {
    switch (r) {
        case LoadResult::Ok:                  return "Ok";
        case LoadResult::AlreadyLoaded:       return "AlreadyLoaded";
        case LoadResult::LoadedAfterEviction: return "LoadedAfterEviction";
        case LoadResult::Deferred:            return "Deferred";
        case LoadResult::Unavailable:         return "Unavailable";
    }
    return "Unknown";
}

// =========================================================
// Constructor
// =========================================================

ModelLoader::ModelLoader(const ModelRegistry& registry,
                         ResourceCoordinator& coordinator,
                         const ModelLoaderConfig& config)
    : registry_(registry)
    , coordinator_(coordinator)
    , config_(config) {}

// =========================================================
// ensureLoaded — the main entry point
// =========================================================

LoadResult ModelLoader::ensureLoaded(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Model must exist in registry
    const ModelInfo* info = registry_.getModelById(model_id);
    if (!info) {
        throw std::runtime_error(
            "ModelLoader::ensureLoaded: model '" + model_id
            + "' not found in ModelRegistry");
    }

    ModelSlot& slot = getOrCreateSlot(model_id);

    // Already in a usable state?
    if (slot.state == ResidencyState::InUse
        || slot.state == ResidencyState::Loaded
        || slot.state == ResidencyState::Idle
        || slot.state == ResidencyState::EvictEligible) {
        LOG_DEBUG("MMO_LOADER", "Model '" + model_id + "' already loaded (state="
                  + residencyStateToString(slot.state) + ")");
        return LoadResult::AlreadyLoaded;
    }

    // Cannot load if already in a transitional state
    if (slot.state == ResidencyState::Loading
        || slot.state == ResidencyState::Unloading) {
        LOG_DEBUG("MMO_LOADER", "Model '" + model_id + "' is in transitional state "
                  + residencyStateToString(slot.state) + " — returning Deferred");
        return LoadResult::Deferred;
    }

    // ── Submit resource claim ──
    ResourceClaim claim;
    claim.consumer_id  = "model:" + model_id;
    claim.kind         = ClaimKind::ModelLoad;
    claim.ram_mb       = info->estimated_ram_mb;
    claim.vram_mb      = info->estimated_vram_mb;
    claim.preferred_gpu = slot.gpu_device;
    claim.priority     = 0;
    claim.can_defer    = true;
    claim.can_evict    = true;

    ResourceDecision decision = coordinator_.requestClaim(claim);

    switch (decision.action) {
        case DecisionAction::Deny:
            LOG_DEBUG("MMO_LOADER", "Claim denied for '" + model_id + "': " + decision.reason);
            return LoadResult::Unavailable;

        case DecisionAction::Defer:
            LOG_DEBUG("MMO_LOADER", "Claim deferred for '" + model_id + "': " + decision.reason);
            return LoadResult::Deferred;

        case DecisionAction::Throttle:
            LOG_DEBUG("MMO_LOADER", "Claim throttled for '" + model_id + "': " + decision.reason);
            return LoadResult::Deferred;

        case DecisionAction::EvictThenAllow:
            LOG_DEBUG("MMO_LOADER", "Evicting " + std::to_string(decision.eviction_targets.size())
                      + " holders before loading '" + model_id + "'");
            evictTargets(decision.eviction_targets);
            break;

        case DecisionAction::Allow:
            break;
    }

    // ── Transition: Unloaded → Loading ──
    slot.state = ResidencyState::Loading;

    bool load_ok = false;
    if (start_cb_) {
        load_ok = start_cb_(*info);
    } else {
        // No start callback registered — treat as immediate success.
        // The actual process management is plugged in by the body.
        load_ok = true;
    }

    if (!load_ok) {
        slot.state = ResidencyState::Unloaded;
        coordinator_.releaseClaim("model:" + model_id);
        LOG_DEBUG("MMO_LOADER", "Start callback failed for '" + model_id
                  + "' — returning Unavailable");
        return LoadResult::Unavailable;
    }

    // ── Transition: Loading → Loaded ──
    auto now = std::chrono::steady_clock::now();
    slot.state       = ResidencyState::Loaded;
    slot.loaded_time = now;
    slot.last_use_time = now;

    // Upgrade claim from ModelLoad to ModelResident
    coordinator_.releaseClaim("model:" + model_id);
    ResourceClaim resident_claim;
    resident_claim.consumer_id  = "model:" + model_id;
    resident_claim.kind         = ClaimKind::ModelResident;
    resident_claim.ram_mb       = info->estimated_ram_mb;
    resident_claim.vram_mb      = info->estimated_vram_mb;
    resident_claim.preferred_gpu = slot.gpu_device;
    resident_claim.priority     = 0;
    resident_claim.can_defer    = false;
    resident_claim.can_evict    = false;
    coordinator_.requestClaim(resident_claim);

    bool evicted_first = (decision.action == DecisionAction::EvictThenAllow);
    LOG_DEBUG("MMO_LOADER", "Model '" + model_id + "' loaded successfully"
              + (evicted_first ? " (after eviction)" : ""));

    return evicted_first ? LoadResult::LoadedAfterEviction : LoadResult::Ok;
}

// =========================================================
// markInUse / markIdle
// =========================================================

void ModelLoader::markInUse(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = slots_.find(model_id);
    if (it == slots_.end()) {
        throw std::runtime_error(
            "ModelLoader::markInUse: model '" + model_id + "' has no slot");
    }

    ModelSlot& slot = it->second;

    if (slot.state != ResidencyState::Loaded
        && slot.state != ResidencyState::Idle
        && slot.state != ResidencyState::EvictEligible) {
        throw std::runtime_error(
            "ModelLoader::markInUse: model '" + model_id + "' is in state "
            + residencyStateToString(slot.state)
            + " — must be Loaded, Idle, or EvictEligible");
    }

    slot.state = ResidencyState::InUse;
    slot.use_count++;
    slot.last_use_time = std::chrono::steady_clock::now();

    coordinator_.markInUse("model:" + model_id, true);
}

void ModelLoader::markIdle(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = slots_.find(model_id);
    if (it == slots_.end()) {
        throw std::runtime_error(
            "ModelLoader::markIdle: model '" + model_id + "' has no slot");
    }

    ModelSlot& slot = it->second;

    if (slot.state != ResidencyState::InUse) {
        throw std::runtime_error(
            "ModelLoader::markIdle: model '" + model_id + "' is in state "
            + residencyStateToString(slot.state) + " — must be InUse");
    }

    auto now = std::chrono::steady_clock::now();
    slot.state        = ResidencyState::Idle;
    slot.idle_since   = now;
    slot.last_use_time = now;

    coordinator_.markInUse("model:" + model_id, false);
}

// =========================================================
// unload — explicit teardown
// =========================================================

void ModelLoader::unload(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = slots_.find(model_id);
    if (it == slots_.end()) return;

    ModelSlot& slot = it->second;

    if (slot.state == ResidencyState::Unloaded) return;

    if (slot.state == ResidencyState::InUse) {
        throw std::runtime_error(
            "ModelLoader::unload: model '" + model_id
            + "' is InUse — cannot evict an in-use model");
    }

    slot.state = ResidencyState::Unloading;

    const ModelInfo* info = registry_.getModelById(model_id);
    if (info && stop_cb_) {
        stop_cb_(*info);
    }

    coordinator_.releaseClaim("model:" + model_id);
    slot.state = ResidencyState::Unloaded;

    LOG_DEBUG("MMO_LOADER", "Model '" + model_id + "' unloaded");
}

// =========================================================
// tickIdleTimers — Idle → EvictEligible transition
// =========================================================

void ModelLoader::tickIdleTimers() {
    std::lock_guard<std::mutex> lock(mutex_);

    auto now = std::chrono::steady_clock::now();

    for (auto& [id, slot] : slots_) {
        if (slot.state != ResidencyState::Idle) continue;

        int ttl_ms = effectiveIdleTtlMs(slot);
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - slot.idle_since).count();

        if (elapsed >= ttl_ms) {
            slot.state = ResidencyState::EvictEligible;
            LOG_DEBUG("MMO_LOADER", "Model '" + id + "' → EvictEligible (idle "
                      + std::to_string(elapsed) + "ms, TTL=" + std::to_string(ttl_ms) + "ms)");
        }
    }
}

// =========================================================
// Queries
// =========================================================

ResidencyState ModelLoader::getState(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = slots_.find(model_id);
    if (it == slots_.end()) return ResidencyState::Unloaded;
    return it->second.state;
}

const ModelSlot* ModelLoader::getSlot(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = slots_.find(model_id);
    if (it == slots_.end()) return nullptr;
    return &it->second;
}

int ModelLoader::loadedCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int count = 0;
    for (const auto& [id, slot] : slots_) {
        if (slot.state != ResidencyState::Unloaded
            && slot.state != ResidencyState::Unloading) {
            ++count;
        }
    }
    return count;
}

int ModelLoader::effectiveIdleTtlMs(const ModelSlot& slot) const {
    // Use-degrading: more uses → longer idle TTL
    int ttl = config_.idle_ttl_ms + (slot.use_count * config_.use_degrade_step_ms);
    return std::min(ttl, config_.hot_ttl_cap_ms);
}

// =========================================================
// unloadAll
// =========================================================

void ModelLoader::unloadAll() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& [id, slot] : slots_) {
        if (slot.state == ResidencyState::Unloaded) continue;

        if (slot.state == ResidencyState::InUse) {
            // Force transition out of InUse for shutdown
            coordinator_.markInUse("model:" + id, false);
        }

        slot.state = ResidencyState::Unloading;

        const ModelInfo* info = registry_.getModelById(id);
        if (info && stop_cb_) {
            stop_cb_(*info);
        }

        coordinator_.releaseClaim("model:" + id);
        slot.state = ResidencyState::Unloaded;

        LOG_DEBUG("MMO_LOADER", "Model '" + id + "' unloaded (shutdown)");
    }
}

// =========================================================
// Callbacks
// =========================================================

void ModelLoader::setStartCallback(StartCallback cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    start_cb_ = std::move(cb);
}

void ModelLoader::setStopCallback(StopCallback cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_cb_ = std::move(cb);
}

// =========================================================
// Private helpers
// =========================================================

ModelSlot& ModelLoader::getOrCreateSlot(const std::string& model_id) {
    auto it = slots_.find(model_id);
    if (it != slots_.end()) return it->second;

    ModelSlot slot;
    slot.model_id = model_id;
    auto [inserted, _] = slots_.emplace(model_id, std::move(slot));
    return inserted->second;
}

void ModelLoader::evictTargets(const std::vector<std::string>& targets) {
    for (const auto& consumer_id : targets) {
        // consumer_id format is "model:<model_id>"
        if (consumer_id.rfind("model:", 0) != 0) continue;

        std::string mid = consumer_id.substr(6);
        auto it = slots_.find(mid);
        if (it == slots_.end()) continue;

        ModelSlot& slot = it->second;

        // Never evict InUse models — the coordinator should not have
        // returned them, but enforce here as a safety check.
        if (slot.state == ResidencyState::InUse) continue;

        if (slot.state == ResidencyState::Unloaded
            || slot.state == ResidencyState::Unloading) continue;

        slot.state = ResidencyState::Unloading;

        const ModelInfo* info = registry_.getModelById(mid);
        if (info && stop_cb_) {
            stop_cb_(*info);
        }

        coordinator_.releaseClaim(consumer_id);
        slot.state = ResidencyState::Unloaded;

        LOG_DEBUG("MMO_LOADER", "Evicted model '" + mid + "'");
    }
}

} // namespace GRIM::MMO
