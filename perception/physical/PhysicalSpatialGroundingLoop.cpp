#include "PhysicalSpatialGroundingLoop.hpp"

#include "PhysicalFrameBus.hpp"
#include "PhysicalPerceptionPrimitiveBus.hpp"
#include "PhysicalSpatialGroundingBus.hpp"
#include "PhysicalSpatialGroundingLogTag.hpp"
#include "logger.hpp"

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

// All process-wide Stage-3 state lives here. Encapsulated in an anonymous
// namespace so no other TU can reach in.
struct PhysicalSpatialGroundingState {
    std::mutex                                          mutex;
    bool                                                initialized   = false;
    bool                                                shutting_down = false;

    PhysicalSpatialGroundingEnableFlags                 enable_flags{};

    // Owned operators — constructed lazily on first Tick.
    std::unique_ptr<PhysicalMonocularDepthEstimator>    depth_estimator;
    std::unique_ptr<PhysicalSpatialGrounder>            spatial_grounder;

    // Pending configs deposited via Request* before lazy init. Applied at
    // the first Tick.
    std::unique_ptr<PhysicalMonocularDepthEstimatorConfig>  pending_depth_cfg;
    std::unique_ptr<PhysicalSpatialGrounderConfig>          pending_ground_cfg;

    std::string  last_error_reason;
    uint64_t     tick_count          = 0;
    uint64_t     processed_count     = 0;
    uint64_t     last_seen_frame_ctr = 0;       // PhysicalFrameBus counter
    uint64_t     last_seen_perc_ctr  = 0;       // PhysicalPerceptionPrimitiveBus counter

    // Pre-allocated views so we don't reallocate every Tick.
    PhysicalFrameBus::FrameView                  frame_view;
    PhysicalPerceptionPrimitiveBus::ResultsView  perc_view;
};

PhysicalSpatialGroundingState& GetState() {
    static PhysicalSpatialGroundingState s;
    return s;
}

void LazyInitLocked(PhysicalSpatialGroundingState& s) {
    if (s.initialized) return;
    LOG_DEBUG(PHYSICAL_SPATIAL_GROUND_LOG_TAG,
              "TickPhysicalSpatialGrounding: first call — running lazy init");
    s.depth_estimator  = std::make_unique<PhysicalMonocularDepthEstimator>();
    s.spatial_grounder = std::make_unique<PhysicalSpatialGrounder>();

    auto apply = [&](auto& pending, auto loader, const char* name) {
        if (!pending) return;
        try {
            loader(*pending);
        } catch (const std::exception& e) {
            s.last_error_reason = std::string("LazyInit: pending config for ") + name
                                  + " failed: " + e.what();
            LOG_ERROR(PHYSICAL_SPATIAL_GROUND_LOG_TAG, s.last_error_reason);
        }
        pending.reset();
    };
    apply(s.pending_depth_cfg,
          [&](auto& c){ s.depth_estimator->LoadOnnxModelIntoPhysicalMonocularDepthEstimator(c); },
          "PhysicalMonocularDepthEstimator");

    // Grounder has no model file. If no pending config, install defaults so
    // it transitions to ModelLoaded — fusion runs as soon as a depth map is
    // available.
    if (s.pending_ground_cfg) {
        apply(s.pending_ground_cfg,
              [&](auto& c){ s.spatial_grounder->ConfigurePhysicalSpatialGrounder(c); },
              "PhysicalSpatialGrounder");
    } else {
        try {
            s.spatial_grounder->ConfigurePhysicalSpatialGrounder(PhysicalSpatialGrounderConfig{});
        } catch (const std::exception& e) {
            s.last_error_reason = std::string("LazyInit: default PhysicalSpatialGrounder config failed: ") + e.what();
            LOG_ERROR(PHYSICAL_SPATIAL_GROUND_LOG_TAG, s.last_error_reason);
        }
    }

    s.initialized = true;
}

} // anonymous namespace

void TickPhysicalSpatialGrounding() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    if (s.shutting_down) return;
    LazyInitLocked(s);
    ++s.tick_count;

    // We need BOTH the source frame (for depth inference) AND the matching
    // tracker output (for fusion). Either advancing on its own is not
    // enough — we wait for both, then verify they share a frame counter so
    // the fused result is coherent.
    const bool frame_advanced = PhysicalFrameBus::Instance().PullLatestFrameView(
        s.frame_view, s.last_seen_frame_ctr);
    const bool perc_advanced  = PhysicalPerceptionPrimitiveBus::Instance()
        .PullLatestPhysicalPerceptionResultsView(s.perc_view, s.last_seen_perc_ctr);

    if (!frame_advanced && !perc_advanced) return;

    // If only one advanced we cannot guarantee correlated inputs. Wait.
    if (s.frame_view.frame_counter == 0 || s.perc_view.results.source_frame_counter == 0) return;
    if (s.frame_view.frame_counter != s.perc_view.results.source_frame_counter) return;

    if (s.frame_view.model_image.empty()) {
        s.last_error_reason = "TickPhysicalSpatialGrounding: pulled frame has empty model_image";
        LOG_ERROR(PHYSICAL_SPATIAL_GROUND_LOG_TAG, s.last_error_reason);
        return;
    }

    PhysicalSpatialGroundingResults results;
    results.source_frame_counter              = s.frame_view.frame_counter;
    results.source_perception_results_counter = s.perc_view.results.source_frame_counter;
    results.model_image_width                 = s.frame_view.metadata.model_width;
    results.model_image_height                = s.frame_view.metadata.model_height;
    results.raw_image_width                   = s.frame_view.metadata.raw_width;
    results.raw_image_height                  = s.frame_view.metadata.raw_height;
    results.raw_to_model                      = s.frame_view.metadata.raw_to_model;
    if (results.model_image_width  <= 0) results.model_image_width  = s.frame_view.model_image.cols;
    if (results.model_image_height <= 0) results.model_image_height = s.frame_view.model_image.rows;
    if (results.raw_image_width    <= 0) results.raw_image_width    = s.frame_view.raw_image.cols;
    if (results.raw_image_height   <= 0) results.raw_image_height   = s.frame_view.raw_image.rows;

    // ── Depth estimation ───────────────────────────────────────────────
    if (s.enable_flags.depth_estimator && s.depth_estimator) {
        s.depth_estimator->RouteFrameToPhysicalMonocularDepthEstimator(
            s.frame_view.model_image,
            results.depth_map,
            results.depth_estimator_state,
            results.depth_estimator_last_error,
            results.last_depth_inference_ms);
        results.depth_inference_count = s.depth_estimator->GetPhysicalMonocularDepthEstimatorInferenceCount();
    } else {
        results.depth_estimator_state      = PhysicalImageOperatorState::NoModelConfigured;
        results.depth_estimator_last_error = "depth_estimator disabled by enable_flags";
    }

    // ── Spatial grounding ──────────────────────────────────────────────
    if (s.enable_flags.spatial_grounder && s.spatial_grounder) {
        s.spatial_grounder->RouteDepthAndTracksToPhysicalSpatialGrounder(
            results.depth_map,
            s.perc_view.results.entity_tracker,
            results.model_image_width,
            results.model_image_height,
            results.grounded_entities,
            results.grounder_state,
            results.grounder_last_error,
            results.last_grounding_ms);
        results.grounding_count = s.spatial_grounder->GetPhysicalSpatialGrounderRunCount();
    } else {
        results.grounder_state      = PhysicalImageOperatorState::NoModelConfigured;
        results.grounder_last_error = "spatial_grounder disabled by enable_flags";
    }

    try {
        PhysicalSpatialGroundingBus::Instance()
            .PublishPhysicalSpatialGroundingResultsToBus(results);
        ++s.processed_count;
        s.last_error_reason.clear();
    } catch (const std::exception& e) {
        s.last_error_reason = std::string("PublishPhysicalSpatialGroundingResultsToBus failed: ") + e.what();
        LOG_ERROR(PHYSICAL_SPATIAL_GROUND_LOG_TAG, s.last_error_reason);
    }
}

void ShutdownPhysicalSpatialGrounding() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.shutting_down = true;
    if (s.depth_estimator)  s.depth_estimator->ResetPhysicalMonocularDepthEstimator();
    if (s.spatial_grounder) s.spatial_grounder->ResetPhysicalSpatialGrounder();
    s.depth_estimator.reset();
    s.spatial_grounder.reset();
    s.pending_depth_cfg.reset();
    s.pending_ground_cfg.reset();
    PhysicalSpatialGroundingBus::Instance().ResetPhysicalSpatialGroundingBus();
    s.initialized          = false;
    s.tick_count           = 0;
    s.processed_count      = 0;
    s.last_seen_frame_ctr  = 0;
    s.last_seen_perc_ctr   = 0;
    LOG_DEBUG(PHYSICAL_SPATIAL_GROUND_LOG_TAG, "ShutdownPhysicalSpatialGrounding: complete");
}

bool IsPhysicalSpatialGroundingRunning() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.initialized && !s.shutting_down;
}

PhysicalSpatialGroundingEnableFlags GetPhysicalSpatialGroundingEnableFlags() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.enable_flags;
}

std::string GetLastPhysicalSpatialGroundingError() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.last_error_reason;
}

uint64_t GetPhysicalSpatialGroundingTickCount() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.tick_count;
}

uint64_t GetPhysicalSpatialGroundingProcessedCount() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.processed_count;
}

void RequestSetPhysicalSpatialGroundingEnableFlags(
    const PhysicalSpatialGroundingEnableFlags& flags)
{
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.enable_flags = flags;
}

void RequestConfigurePhysicalMonocularDepthEstimator(
    const PhysicalMonocularDepthEstimatorConfig& cfg)
{
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    if (!s.initialized) {
        s.pending_depth_cfg = std::make_unique<PhysicalMonocularDepthEstimatorConfig>(cfg);
        return;
    }
    if (!s.depth_estimator) {
        // Should not happen — initialized ⇒ depth_estimator was constructed.
        // Rule 20: surface this loudly rather than silently dropping the cfg.
        const std::string reason =
            "RequestConfigurePhysicalMonocularDepthEstimator: depth_estimator is null after init";
        s.last_error_reason = reason;
        LOG_ERROR(PHYSICAL_SPATIAL_GROUND_LOG_TAG, reason);
        throw std::runtime_error(reason);
    }
    s.depth_estimator->LoadOnnxModelIntoPhysicalMonocularDepthEstimator(cfg);
}

void RequestConfigurePhysicalSpatialGrounder(
    const PhysicalSpatialGrounderConfig& cfg)
{
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    if (!s.initialized) {
        s.pending_ground_cfg = std::make_unique<PhysicalSpatialGrounderConfig>(cfg);
        return;
    }
    if (!s.spatial_grounder) {
        const std::string reason =
            "RequestConfigurePhysicalSpatialGrounder: spatial_grounder is null after init";
        s.last_error_reason = reason;
        LOG_ERROR(PHYSICAL_SPATIAL_GROUND_LOG_TAG, reason);
        throw std::runtime_error(reason);
    }
    s.spatial_grounder->ConfigurePhysicalSpatialGrounder(cfg);
}

}}} // namespace GRIM::Perception::Physical
