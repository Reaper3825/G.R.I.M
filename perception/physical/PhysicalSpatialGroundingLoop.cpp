#include "PhysicalSpatialGroundingLoop.hpp"

#include "PhysicalFrameBus.hpp"
#include "PhysicalLatestTickWorker.hpp"
#include "PhysicalPerceptionPrimitiveBus.hpp"
#include "PhysicalSpatialGroundingBus.hpp"
#include "PhysicalSpatialGroundingLogTag.hpp"
#include "logger.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

double PhysicalSpatialElapsedMsSince(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

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
    std::atomic<uint64_t> tick_count      {0};
    std::atomic<uint64_t> processed_count {0};
    uint64_t     last_seen_frame_ctr = 0;       // PhysicalFrameBus counter
    uint64_t     last_seen_perc_ctr  = 0;       // PhysicalPerceptionPrimitiveBus counter

    // Pre-allocated views so we don't reallocate every Tick.
    PhysicalFrameBus::FrameView                  frame_view;
    PhysicalPerceptionPrimitiveBus::ResultsView  perc_view;

    // Cadence + cache state for the depth estimator. Defaults preserve
    // every-frame inference exactly (min_period_ms=0, reuse_on_stable_scene=false).
    // Loud-failure: when the per-frame scene-stability signal is invalid
    // we always re-run depth and report cache_reason="no_signal".
    PhysicalOperatorCadenceConfig cached_depth_cadence{};
    PhysicalDepthMap              cached_depth_map{};
    PhysicalImageOperatorState    cached_depth_state =
        PhysicalImageOperatorState::NoModelConfigured;
    double                        cached_last_depth_inference_ms = 0.0;
    uint64_t                      cached_depth_inference_count   = 0;
    uint64_t                      last_depth_run_steady_ns       = 0;
    uint64_t                      last_depth_fresh_frame_counter = 0;
};

PhysicalSpatialGroundingState& GetState() {
    static PhysicalSpatialGroundingState s;
    return s;
}

PhysicalLatestTickWorker& GetSpatialGroundingWorker() {
    static PhysicalLatestTickWorker worker;
    return worker;
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

namespace {

void RunPhysicalSpatialGroundingOnce() {
    const auto tick_start = std::chrono::steady_clock::now();
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    if (s.shutting_down) return;
    LazyInitLocked(s);
    ++s.tick_count;

    // Stage 2 pins the exact immutable source frame inside its result. Use
    // that frame for depth inference so asynchronous workers cannot combine
    // tracker output from one frame with pixels from a newer frame.
    const auto perc_pull_start = std::chrono::steady_clock::now();
    const bool perc_advanced  = PhysicalPerceptionPrimitiveBus::Instance()
        .PullLatestPhysicalPerceptionResultsView(s.perc_view, s.last_seen_perc_ctr);
    const double perc_pull_ms = PhysicalSpatialElapsedMsSince(perc_pull_start);

    if (!perc_advanced) return;

    const auto frame_pull_start = std::chrono::steady_clock::now();
    s.frame_view = s.perc_view.results.source_frame;
    const double frame_pull_ms = PhysicalSpatialElapsedMsSince(frame_pull_start);
    if (!s.frame_view.packet
        || s.frame_view.frame_counter == 0
        || s.frame_view.frame_counter != s.perc_view.results.source_frame_counter) {
        s.last_error_reason =
            "TickPhysicalSpatialGrounding: Stage-2 result has no matching pinned source frame";
        LOG_ERROR(PHYSICAL_SPATIAL_GROUND_LOG_TAG, s.last_error_reason);
        return;
    }
    s.last_seen_frame_ctr = s.frame_view.frame_counter;

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
    results.frame_bus_pull_ms                 = frame_pull_ms;
    results.perception_bus_pull_ms            = perc_pull_ms;
    if (results.model_image_width  <= 0) results.model_image_width  = s.frame_view.model_image.cols;
    if (results.model_image_height <= 0) results.model_image_height = s.frame_view.model_image.rows;
    if (results.raw_image_width    <= 0) results.raw_image_width    = s.frame_view.raw_image.cols;
    if (results.raw_image_height   <= 0) results.raw_image_height   = s.frame_view.raw_image.rows;

    // ── Depth estimation (cadence-gated) ──────────────────────
    if (s.enable_flags.depth_estimator && s.depth_estimator) {
        const auto depth_start = std::chrono::steady_clock::now();
        const PhysicalSceneStability& scene = s.frame_view.metadata.scene_stability;
        const uint64_t now_steady_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        const uint64_t frame_ctr = results.source_frame_counter;

        const bool has_cached =
            s.cached_depth_state == PhysicalImageOperatorState::ModelLoaded &&
            s.last_depth_fresh_frame_counter != 0;

        // Inline cadence decision (mirrors PhysicalPerceptionPrimitivesLoop's
        // DecidePhysicalCadence helper). Kept local so the two loops remain
        // independent translation units — avoids creating a shared header for
        // a 20-line policy that may diverge per loop in the future.
        const char* cache_reason = "";
        bool        run_now      = true;

        if (!has_cached) {
            run_now = true; cache_reason = "";
        } else if (!scene.valid) {
            run_now = true; cache_reason = "no_signal";
        } else {
            const bool stable_gate  = s.cached_depth_cadence.reuse_on_stable_scene && scene.is_stable;
            bool       cadence_gate = false;
            if (s.cached_depth_cadence.min_period_ms > 0 && s.last_depth_run_steady_ns != 0) {
                const uint64_t min_period_ns =
                    static_cast<uint64_t>(s.cached_depth_cadence.min_period_ms) * 1'000'000ULL;
                cadence_gate = (now_steady_ns - s.last_depth_run_steady_ns) < min_period_ns;
            }
            if (stable_gate && cadence_gate) { run_now = false; cache_reason = "stable_and_cadence"; }
            else if (stable_gate)            { run_now = false; cache_reason = "stable_scene";       }
            else if (cadence_gate)           { run_now = false; cache_reason = "cadence_floor";      }
            else                             { run_now = true;  cache_reason = "";                   }
        }

        if (run_now) {
            s.depth_estimator->RouteFrameToPhysicalMonocularDepthEstimator(
                s.frame_view.model_image,
                results.depth_map,
                results.depth_estimator_state,
                results.depth_estimator_last_error,
                results.last_depth_inference_ms);
            results.depth_inference_count =
                s.depth_estimator->GetPhysicalMonocularDepthEstimatorInferenceCount();
            results.depth_cache_status.cache_hit        = false;
            results.depth_cache_status.cache_age_frames = 0;
            results.depth_cache_status.cache_reason     = cache_reason;
            // Cache only successful runs.
            if (results.depth_estimator_state == PhysicalImageOperatorState::ModelLoaded) {
                s.cached_depth_map               = results.depth_map;
                s.cached_depth_state             = results.depth_estimator_state;
                s.cached_last_depth_inference_ms = results.last_depth_inference_ms;
                s.cached_depth_inference_count   = results.depth_inference_count;
                s.last_depth_run_steady_ns       = now_steady_ns;
                s.last_depth_fresh_frame_counter = frame_ctr;
            }
        } else {
            // Reuse cached depth map and provenance verbatim. Consumers MUST
            // inspect depth_cache_status before correlating timing.
            results.depth_map                  = s.cached_depth_map;
            results.depth_estimator_state      = s.cached_depth_state;
            results.depth_estimator_last_error.clear();
            results.last_depth_inference_ms    = s.cached_last_depth_inference_ms;
            results.depth_inference_count      = s.cached_depth_inference_count;
            results.depth_cache_status.cache_hit        = true;
            results.depth_cache_status.cache_age_frames =
                frame_ctr > s.last_depth_fresh_frame_counter
                    ? static_cast<uint32_t>(frame_ctr - s.last_depth_fresh_frame_counter)
                    : 0u;
            results.depth_cache_status.cache_reason     = cache_reason;
        }
        results.depth_wall_ms = PhysicalSpatialElapsedMsSince(depth_start);
    } else {
        results.depth_estimator_state      = PhysicalImageOperatorState::NoModelConfigured;
        results.depth_estimator_last_error = "depth_estimator disabled by enable_flags";
    }

    // ── Spatial grounding ──────────────────────────────────────────────
    if (s.enable_flags.spatial_grounder && s.spatial_grounder) {
        const auto grounder_start = std::chrono::steady_clock::now();
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
        results.grounder_wall_ms = PhysicalSpatialElapsedMsSince(grounder_start);
    } else {
        results.grounder_state      = PhysicalImageOperatorState::NoModelConfigured;
        results.grounder_last_error = "spatial_grounder disabled by enable_flags";
    }
    results.tick_total_ms = PhysicalSpatialElapsedMsSince(tick_start);

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

} // anonymous namespace

void TickPhysicalSpatialGrounding() {
    auto& worker = GetSpatialGroundingWorker();
    worker.Start([] {
        try {
            RunPhysicalSpatialGroundingOnce();
        } catch (const std::exception& e) {
            auto& s = GetState();
            std::lock_guard<std::mutex> lk(s.mutex);
            s.last_error_reason = std::string("spatial-grounding worker threw: ") + e.what();
            LOG_ERROR(PHYSICAL_SPATIAL_GROUND_LOG_TAG, s.last_error_reason);
        } catch (...) {
            auto& s = GetState();
            std::lock_guard<std::mutex> lk(s.mutex);
            s.last_error_reason = "spatial-grounding worker threw a non-standard exception";
            LOG_ERROR(PHYSICAL_SPATIAL_GROUND_LOG_TAG, s.last_error_reason);
        }
    });
    worker.RequestLatest();
}

void ShutdownPhysicalSpatialGrounding() {
    GetSpatialGroundingWorker().Stop();
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
    s.cached_depth_map     = {};
    s.cached_depth_state   = PhysicalImageOperatorState::NoModelConfigured;
    s.cached_last_depth_inference_ms = 0.0;
    s.cached_depth_inference_count   = 0;
    s.last_depth_run_steady_ns       = 0;
    s.last_depth_fresh_frame_counter = 0;
    LOG_DEBUG(PHYSICAL_SPATIAL_GROUND_LOG_TAG, "ShutdownPhysicalSpatialGrounding: complete");
}

bool IsPhysicalSpatialGroundingRunning() {
    auto& s = GetState();
    std::unique_lock<std::mutex> lk(s.mutex, std::try_to_lock);
    if (!lk.owns_lock()) return GetSpatialGroundingWorker().IsStarted();
    return s.initialized && !s.shutting_down;
}

PhysicalSpatialGroundingEnableFlags GetPhysicalSpatialGroundingEnableFlags() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.enable_flags;
}

std::string GetLastPhysicalSpatialGroundingError() {
    auto& s = GetState();
    std::unique_lock<std::mutex> lk(s.mutex, std::try_to_lock);
    if (!lk.owns_lock()) return {};
    return s.last_error_reason;
}

uint64_t GetPhysicalSpatialGroundingTickCount() {
    return GetState().tick_count.load();
}

uint64_t GetPhysicalSpatialGroundingProcessedCount() {
    return GetState().processed_count.load();
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
    s.cached_depth_cadence = cfg.cadence;
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
