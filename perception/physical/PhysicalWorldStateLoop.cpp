#include "PhysicalWorldStateLoop.hpp"

#include "PhysicalPerceptionPrimitiveBus.hpp"
#include "PhysicalSpatialGroundingBus.hpp"
#include "PhysicalWorldStateBus.hpp"
#include "PhysicalWorldStateLogTag.hpp"
#include "logger.hpp"

#include <mutex>
#include <stdexcept>
#include <string>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

struct PhysicalWorldStateState {
    std::mutex                                          mutex;
    bool                                                initialized   = false;
    bool                                                shutting_down = false;

    PhysicalWorldStateBuilderConfig                     cfg{};
    std::string                                         last_error_reason;
    uint64_t                                            tick_count          = 0;
    uint64_t                                            processed_count     = 0;
    uint64_t                                            last_seen_perc_ctr  = 0;
    uint64_t                                            last_seen_ground_ctr = 0;

    PhysicalPerceptionPrimitiveBus::ResultsView         perc_view;
    PhysicalSpatialGroundingBus::ResultsView            ground_view;
};

PhysicalWorldStateState& GetState() {
    static PhysicalWorldStateState s;
    return s;
}

void LazyInitLocked(PhysicalWorldStateState& s) {
    if (s.initialized) return;
    LOG_DEBUG(PHYSICAL_WORLD_STATE_LOG_TAG,
              "TickPhysicalWorldState: first call — running lazy init "
              "(builder cfg defaults)");
    // Validate defaults eagerly so a bad default would throw at startup,
    // not on the first Tick that happens to receive matching inputs.
    try {
        PhysicalWorldStateSnapshot probe;
        // We deliberately pass empty inputs; the builder will throw only on
        // bad cfg. Empty perc/ground produce an empty snapshot but
        // source_frame_counter==0 so we never publish it here.
        BuildPhysicalWorldStateSnapshot(
            PhysicalPerceptionPrimitiveResults{},
            PhysicalSpatialGroundingResults{},
            s.cfg,
            probe);
    } catch (const std::exception& e) {
        s.last_error_reason = std::string("LazyInit: default builder cfg failed validation: ")
                              + e.what();
        LOG_ERROR(PHYSICAL_WORLD_STATE_LOG_TAG, s.last_error_reason);
        // Leave initialized=true so we don't retry every frame. Subsequent
        // Tick calls will short-circuit on the bad cfg until the UI calls
        // RequestConfigurePhysicalWorldStateBuilder with a valid config.
    }
    s.initialized = true;
}

} // anonymous namespace

void TickPhysicalWorldState() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    if (s.shutting_down) return;
    LazyInitLocked(s);
    ++s.tick_count;

    const bool perc_advanced = PhysicalPerceptionPrimitiveBus::Instance()
        .PullLatestPhysicalPerceptionResultsView(s.perc_view, s.last_seen_perc_ctr);
    const bool ground_advanced = PhysicalSpatialGroundingBus::Instance()
        .PullLatestPhysicalSpatialGroundingResultsView(s.ground_view, s.last_seen_ground_ctr);

    if (!perc_advanced && !ground_advanced) return;

    // Need both populated AND on the same source frame to produce a
    // numerically coherent snapshot. The Stage-3 loop already enforces that
    // ground.source_frame_counter == perc.source_frame_counter at publish
    // time; we re-verify rather than trust it.
    const uint64_t perc_ctr   = s.perc_view.results.source_frame_counter;
    const uint64_t ground_ctr = s.ground_view.results.source_frame_counter;
    if (perc_ctr == 0) return;
    if (ground_ctr == 0) {
        // Ground hasn't produced anything yet — still emit a snapshot from
        // perception alone so the model sees identity + 2D state even
        // before depth comes online. has_depth will be false on every entity.
    } else if (perc_ctr != ground_ctr) {
        // Mismatched buses; wait until the slower one catches up.
        return;
    }

    PhysicalWorldStateSnapshot snapshot;
    try {
        BuildPhysicalWorldStateSnapshot(
            s.perc_view.results,
            (ground_ctr == perc_ctr) ? s.ground_view.results
                                     : PhysicalSpatialGroundingResults{},
            s.cfg,
            snapshot);
    } catch (const std::exception& e) {
        s.last_error_reason = std::string("BuildPhysicalWorldStateSnapshot threw: ") + e.what();
        LOG_ERROR(PHYSICAL_WORLD_STATE_LOG_TAG, s.last_error_reason);
        return;
    }

    if (snapshot.source_frame_counter == 0) {
        // Should be impossible — perc_ctr was verified non-zero above.
        // Refuse to publish so consumers never see a ghost snapshot.
        const std::string reason =
            "TickPhysicalWorldState: built snapshot has source_frame_counter==0 "
            "after non-zero perc_ctr — internal inconsistency";
        s.last_error_reason = reason;
        LOG_ERROR(PHYSICAL_WORLD_STATE_LOG_TAG, reason);
        return;
    }

    try {
        PhysicalWorldStateBus::Instance().PublishPhysicalWorldStateSnapshotToBus(snapshot);
        ++s.processed_count;
        s.last_error_reason.clear();
    } catch (const std::exception& e) {
        s.last_error_reason = std::string("PublishPhysicalWorldStateSnapshotToBus failed: ")
                              + e.what();
        LOG_ERROR(PHYSICAL_WORLD_STATE_LOG_TAG, s.last_error_reason);
    }
}

void ShutdownPhysicalWorldState() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.shutting_down = true;
    PhysicalWorldStateBus::Instance().ResetPhysicalWorldStateBus();
    s.initialized           = false;
    s.tick_count            = 0;
    s.processed_count       = 0;
    s.last_seen_perc_ctr    = 0;
    s.last_seen_ground_ctr  = 0;
    LOG_DEBUG(PHYSICAL_WORLD_STATE_LOG_TAG, "ShutdownPhysicalWorldState: complete");
}

bool IsPhysicalWorldStateRunning() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.initialized && !s.shutting_down;
}

std::string GetLastPhysicalWorldStateError() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.last_error_reason;
}

uint64_t GetPhysicalWorldStateTickCount() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.tick_count;
}

uint64_t GetPhysicalWorldStateProcessedCount() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.processed_count;
}

PhysicalWorldStateBuilderConfig GetPhysicalWorldStateBuilderConfigSnapshot() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.cfg;
}

void RequestConfigurePhysicalWorldStateBuilder(
    const PhysicalWorldStateBuilderConfig& cfg)
{
    // Validate OUTSIDE the lock (BuildPhysicalWorldStateSnapshot validates
    // via ValidateBuilderConfig at the top — we trigger it with empty inputs
    // so a bad cfg throws BEFORE we touch s.cfg). This guarantees the lock
    // is never held across a throw.
    PhysicalWorldStateSnapshot probe;
    BuildPhysicalWorldStateSnapshot(
        PhysicalPerceptionPrimitiveResults{},
        PhysicalSpatialGroundingResults{},
        cfg,
        probe);

    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.cfg = cfg;
    s.last_error_reason.clear();
    LOG_DEBUG(PHYSICAL_WORLD_STATE_LOG_TAG,
              "RequestConfigurePhysicalWorldStateBuilder: cfg accepted");
}

}}} // namespace GRIM::Perception::Physical
