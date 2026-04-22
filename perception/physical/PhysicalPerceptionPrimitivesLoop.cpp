#include "PhysicalPerceptionPrimitivesLoop.hpp"

#include "PhysicalFrameBus.hpp"
#include "PhysicalPerceptionPrimitiveBus.hpp"
#include "PhysicalPerceptionPrimitivesLogTag.hpp"
#include "logger.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

// All process-wide state for the Stage-2 subsystem lives here. Encapsulated
// in an anonymous namespace so no other TU can reach in.
struct PhysicalPerceptionPrimitivesState {
    std::mutex                                       mutex;
    bool                                             initialized   = false;
    bool                                             shutting_down = false;

    PhysicalPerceptionPrimitivesEnableFlags          enable_flags{};

    // Owned operators — constructed lazily on first Tick.
    std::unique_ptr<PhysicalObjectDetector>          object_detector;
    std::unique_ptr<PhysicalSemanticSegmenter>       semantic_segmenter;
    std::unique_ptr<PhysicalInstanceSegmenter>       instance_segmenter;
    std::unique_ptr<PhysicalImageClassifier>         image_classifier;
    std::unique_ptr<PhysicalPoseKeypointEstimator>   pose_estimator;
    std::unique_ptr<PhysicalSceneTextReader>         scene_text_reader;
    std::unique_ptr<PhysicalFacialExpressionDetector> facial_expression_detector;
    std::unique_ptr<PhysicalEntityTracker>           entity_tracker;
    std::unique_ptr<PhysicalClassPolicy>             class_policy;

    // Pending configs deposited via Request* before lazy init. Applied at
    // the first Tick. After init these are emptied; subsequent Request*
    // calls operate directly on the operator instances.
    std::unique_ptr<PhysicalObjectDetectorConfig>          pending_obj_cfg;
    std::unique_ptr<PhysicalSemanticSegmenterConfig>       pending_seg_cfg;
    std::unique_ptr<PhysicalInstanceSegmenterConfig>       pending_inst_seg_cfg;
    std::unique_ptr<PhysicalImageClassifierConfig>         pending_cls_cfg;
    std::unique_ptr<PhysicalPoseKeypointEstimatorConfig>   pending_pose_cfg;
    std::unique_ptr<PhysicalSceneTextReaderConfig>         pending_text_cfg;
    std::unique_ptr<PhysicalFacialExpressionDetectorConfig> pending_face_cfg;
    std::unique_ptr<PhysicalEntityTrackerConfig>            pending_track_cfg;
    std::unique_ptr<PhysicalClassPolicyConfig>              pending_class_policy_cfg;

    std::string  last_error_reason;
    uint64_t     tick_count           = 0;
    uint64_t     processed_count      = 0;
    uint64_t     last_seen_frame_ctr  = 0;

    // Pre-allocated FrameView so we don't reallocate cv::Mats every Tick.
    PhysicalFrameBus::FrameView frame_view;

    // ── Per-operator cadence + cache state ───────────────────────────────
    // Each cache-aware operator gets:
    //   * cached_*_cadence — copy of the cadence config last requested.
    //     Loop owns this (rather than calling back into the operator) so
    //     the gating decision is local and the operators stay pure.
    //   * cached_*_output  — the most recent FRESH (i.e. cache_hit==false)
    //     result produced by RouteFrameTo*. Reused verbatim on cache hits.
    //   * last_*_run_steady_ns / last_*_fresh_frame_counter — provenance
    //     for cadence floor + cache age.
    // The entity_tracker is intentionally excluded — it is cheap and runs
    // in lock-step with whatever detections it is fed.
    PhysicalOperatorCadenceConfig cached_obj_cadence{};
    PhysicalObjectDetectorOutput  cached_obj_output{};
    uint64_t last_obj_run_steady_ns       = 0;
    uint64_t last_obj_fresh_frame_counter = 0;

    PhysicalOperatorCadenceConfig    cached_seg_cadence{};
    PhysicalSemanticSegmenterOutput  cached_seg_output{};
    uint64_t last_seg_run_steady_ns       = 0;
    uint64_t last_seg_fresh_frame_counter = 0;

    PhysicalOperatorCadenceConfig    cached_inst_seg_cadence{};
    PhysicalInstanceSegmenterOutput  cached_inst_seg_output{};
    uint64_t last_inst_seg_run_steady_ns       = 0;
    uint64_t last_inst_seg_fresh_frame_counter = 0;

    PhysicalOperatorCadenceConfig  cached_cls_cadence{};
    PhysicalImageClassifierOutput  cached_cls_output{};
    uint64_t last_cls_run_steady_ns       = 0;
    uint64_t last_cls_fresh_frame_counter = 0;

    PhysicalOperatorCadenceConfig         cached_pose_cadence{};
    PhysicalPoseKeypointEstimatorOutput   cached_pose_output{};
    uint64_t last_pose_run_steady_ns       = 0;
    uint64_t last_pose_fresh_frame_counter = 0;

    PhysicalOperatorCadenceConfig    cached_text_cadence{};
    PhysicalSceneTextReaderOutput    cached_text_output{};
    uint64_t last_text_run_steady_ns       = 0;
    uint64_t last_text_fresh_frame_counter = 0;

    PhysicalOperatorCadenceConfig            cached_face_cadence{};
    PhysicalFacialExpressionDetectorOutput   cached_face_output{};
    uint64_t last_face_run_steady_ns       = 0;
    uint64_t last_face_fresh_frame_counter = 0;
};

PhysicalPerceptionPrimitivesState& GetState() {
    static PhysicalPerceptionPrimitivesState s;
    return s;
}

// ── Cadence gate ─────────────────────────────────────────────────────────
// Returns the decision for one operator for the current frame.
//
// Rule 20 contract:
//   * If the scene-stability signal is INVALID (cond. failed / first frame),
//     we ALWAYS run inference and surface "no_signal" so the caller can see
//     why the cache wasn't consulted. We never silently fall back.
//   * Defaults (`min_period_ms=0`, `reuse_on_stable_scene=false`) preserve
//     every-frame inference exactly. The gate is a strict opt-in.
enum class PhysicalCadenceDecision : uint8_t {
    Run          = 0,   // run inference, populate fresh result
    ReuseStable  = 1,   // scene unchanged → reuse cached result
    ReuseCadence = 2,   // min-period not yet elapsed → reuse cached result
    ReuseBoth    = 3,   // both gates active simultaneously
    NoSignal     = 4    // scene_stability invalid → forced run
};

inline const char* DescribePhysicalCadenceDecision(PhysicalCadenceDecision d) {
    switch (d) {
        case PhysicalCadenceDecision::Run:          return "";
        case PhysicalCadenceDecision::ReuseStable:  return "stable_scene";
        case PhysicalCadenceDecision::ReuseCadence: return "cadence_floor";
        case PhysicalCadenceDecision::ReuseBoth:    return "stable_and_cadence";
        case PhysicalCadenceDecision::NoSignal:     return "no_signal";
    }
    return "invalid";
}

PhysicalCadenceDecision DecidePhysicalCadence(
    const PhysicalOperatorCadenceConfig& cfg,
    const PhysicalSceneStability&        scene,
    bool                                 has_cached_result,
    uint64_t                             last_run_steady_ns,
    uint64_t                             now_steady_ns)
{
    // No cached fresh output → must run regardless of cadence config.
    if (!has_cached_result) return PhysicalCadenceDecision::Run;

    // Loud-failure path: if the scene-stability signal is invalid we cannot
    // safely consult either gate. Run and report.
    if (!scene.valid) return PhysicalCadenceDecision::NoSignal;

    const bool stable_gate_active  = cfg.reuse_on_stable_scene && scene.is_stable;
    bool       cadence_gate_active = false;
    if (cfg.min_period_ms > 0 && last_run_steady_ns != 0) {
        const uint64_t min_period_ns =
            static_cast<uint64_t>(cfg.min_period_ms) * 1'000'000ULL;
        cadence_gate_active = (now_steady_ns - last_run_steady_ns) < min_period_ns;
    }

    if (stable_gate_active && cadence_gate_active) return PhysicalCadenceDecision::ReuseBoth;
    if (stable_gate_active)                        return PhysicalCadenceDecision::ReuseStable;
    if (cadence_gate_active)                       return PhysicalCadenceDecision::ReuseCadence;
    return PhysicalCadenceDecision::Run;
}

// Stamp the cache_status fields on a result envelope. `cache_age_frames` is
// 0 for fresh runs, or `current_frame - last_fresh_frame` otherwise.
void StampPhysicalCacheStatus(PhysicalCacheStatus& status,
                              PhysicalCadenceDecision decision,
                              uint64_t current_frame_counter,
                              uint64_t last_fresh_frame_counter)
{
    const bool is_reuse =
        decision == PhysicalCadenceDecision::ReuseStable  ||
        decision == PhysicalCadenceDecision::ReuseCadence ||
        decision == PhysicalCadenceDecision::ReuseBoth;
    status.cache_hit        = is_reuse;
    status.cache_age_frames = is_reuse
        ? (current_frame_counter > last_fresh_frame_counter
              ? static_cast<uint32_t>(current_frame_counter - last_fresh_frame_counter)
              : 0u)
        : 0u;
    status.cache_reason = DescribePhysicalCadenceDecision(decision);
}

void LazyInitLocked(PhysicalPerceptionPrimitivesState& s) {
    if (s.initialized) return;
    LOG_DEBUG(PHYSICAL_PERC_PRIM_LOG_TAG,
              "TickPhysicalPerceptionPrimitives: first call — running lazy init");
    s.object_detector    = std::make_unique<PhysicalObjectDetector>();
    s.semantic_segmenter = std::make_unique<PhysicalSemanticSegmenter>();
    s.instance_segmenter = std::make_unique<PhysicalInstanceSegmenter>();
    s.image_classifier   = std::make_unique<PhysicalImageClassifier>();
    s.pose_estimator     = std::make_unique<PhysicalPoseKeypointEstimator>();
    s.scene_text_reader  = std::make_unique<PhysicalSceneTextReader>();
    s.facial_expression_detector = std::make_unique<PhysicalFacialExpressionDetector>();
    s.entity_tracker     = std::make_unique<PhysicalEntityTracker>();
    s.class_policy       = std::make_unique<PhysicalClassPolicy>();

    // Apply any pending configs. Failures are loud but do not abort init —
    // the operator simply ends up in ModelLoadFailed state and the UI
    // surfaces the reason.
    auto apply = [&](auto& pending, auto loader, const char* name) {
        if (!pending) return;
        try {
            loader(*pending);
        } catch (const std::exception& e) {
            s.last_error_reason = std::string("LazyInit: pending config for ") + name
                                  + " failed: " + e.what();
            LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, s.last_error_reason);
        }
        pending.reset();
    };
    apply(s.pending_obj_cfg,
          [&](auto& c){ s.object_detector->LoadOnnxModelIntoPhysicalObjectDetector(c); },
          "PhysicalObjectDetector");
    apply(s.pending_seg_cfg,
          [&](auto& c){ s.semantic_segmenter->LoadOnnxModelIntoPhysicalSemanticSegmenter(c); },
          "PhysicalSemanticSegmenter");
    apply(s.pending_inst_seg_cfg,
          [&](auto& c){ s.instance_segmenter->LoadOnnxModelsIntoPhysicalInstanceSegmenter(c); },
          "PhysicalInstanceSegmenter");
    apply(s.pending_cls_cfg,
          [&](auto& c){ s.image_classifier->LoadOnnxModelIntoPhysicalImageClassifier(c); },
          "PhysicalImageClassifier");
    apply(s.pending_pose_cfg,
          [&](auto& c){ s.pose_estimator->LoadOnnxModelIntoPhysicalPoseKeypointEstimator(c); },
          "PhysicalPoseKeypointEstimator");
    apply(s.pending_text_cfg,
          [&](auto& c){ s.scene_text_reader->LoadOnnxModelsIntoPhysicalSceneTextReader(c); },
          "PhysicalSceneTextReader");
    apply(s.pending_face_cfg,
          [&](auto& c){ s.facial_expression_detector->LoadOnnxModelsIntoPhysicalFacialExpressionDetector(c); },
          "PhysicalFacialExpressionDetector");
    // Tracker has no model file. If no pending config was supplied, install
    // the default config so the operator transitions to ModelLoaded and
    // starts running on first frame. This matches the user-facing contract:
    // "on first update it should check if this is running" — the tracker
    // is always running once the subsystem is up.
    if (s.pending_track_cfg) {
        apply(s.pending_track_cfg,
              [&](auto& c){ s.entity_tracker->ConfigurePhysicalEntityTracker(c); },
              "PhysicalEntityTracker");
    } else {
        try {
            s.entity_tracker->ConfigurePhysicalEntityTracker(PhysicalEntityTrackerConfig{});
        } catch (const std::exception& e) {
            s.last_error_reason = std::string("LazyInit: default PhysicalEntityTracker config failed: ") + e.what();
            LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, s.last_error_reason);
        }
    }

    // ClassPolicy mirrors EntityTracker: no model file. Default-construct
    // installs an empty rule set (pass-through) so it transitions to
    // ModelLoaded and the loop's Apply* call is always safe.
    if (s.pending_class_policy_cfg) {
        apply(s.pending_class_policy_cfg,
              [&](auto& c){ s.class_policy->ConfigurePhysicalClassPolicy(c); },
              "PhysicalClassPolicy");
    } else {
        try {
            s.class_policy->ConfigurePhysicalClassPolicy(PhysicalClassPolicyConfig{});
        } catch (const std::exception& e) {
            s.last_error_reason = std::string("LazyInit: default PhysicalClassPolicy config failed: ") + e.what();
            LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, s.last_error_reason);
        }
    }

    s.initialized = true;
}

} // anonymous namespace

void TickPhysicalPerceptionPrimitives() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    if (s.shutting_down) return;
    LazyInitLocked(s);
    ++s.tick_count;

    // Pull the latest frame from Stage-1's bus. If nothing new, return.
    if (!PhysicalFrameBus::Instance().PullLatestFrameView(s.frame_view, s.last_seen_frame_ctr)) {
        return;
    }
    if (s.frame_view.model_image.empty()) {
        s.last_error_reason = "TickPhysicalPerceptionPrimitives: pulled frame has empty model_image";
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, s.last_error_reason);
        return;
    }

    PhysicalPerceptionPrimitiveResults results;
    results.source_frame_counter = s.frame_view.frame_counter;
    results.model_image_width    = s.frame_view.metadata.model_width;
    results.model_image_height   = s.frame_view.metadata.model_height;
    results.raw_image_width      = s.frame_view.metadata.raw_width;
    results.raw_image_height     = s.frame_view.metadata.raw_height;
    results.raw_to_model         = s.frame_view.metadata.raw_to_model;

    // Belt-and-braces — frame metadata SHOULD have these, but guarantee
    // non-zero before we hand them to the bus (which will throw on zero).
    if (results.model_image_width  <= 0) results.model_image_width  = s.frame_view.model_image.cols;
    if (results.model_image_height <= 0) results.model_image_height = s.frame_view.model_image.rows;
    if (results.raw_image_width    <= 0) results.raw_image_width    = s.frame_view.raw_image.cols;
    if (results.raw_image_height   <= 0) results.raw_image_height   = s.frame_view.raw_image.rows;

    // Per-frame scene-stability signal (computed once by the conditioner).
    // Loop-local cadence gates consult this; operators stay pure.
    const PhysicalSceneStability& scene = s.frame_view.metadata.scene_stability;
    const uint64_t now_steady_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    const uint64_t frame_ctr = results.source_frame_counter;

    // Each operator runs only if its enable flag is on; the operator itself
    // gates on its state, so calling RouteFrameTo* when NoModelConfigured
    // is cheap and just fills the output envelope with state info.
    if (s.enable_flags.object_detector && s.object_detector) {
        const bool has_cached =
            s.cached_obj_output.state == PhysicalImageOperatorState::ModelLoaded &&
            s.last_obj_fresh_frame_counter != 0;
        const auto decision = DecidePhysicalCadence(
            s.cached_obj_cadence, scene, has_cached,
            s.last_obj_run_steady_ns, now_steady_ns);
        if (decision == PhysicalCadenceDecision::Run ||
            decision == PhysicalCadenceDecision::NoSignal) {
            s.object_detector->RouteFrameToPhysicalObjectDetector(
                s.frame_view.model_image,
                results.raw_to_model,
                results.raw_image_width,
                results.raw_image_height,
                frame_ctr,
                results.object_detector);
            StampPhysicalCacheStatus(results.object_detector.cache_status,
                                     decision, frame_ctr, frame_ctr);
            if (results.object_detector.state == PhysicalImageOperatorState::ModelLoaded) {
                s.cached_obj_output            = results.object_detector;
                s.last_obj_run_steady_ns       = now_steady_ns;
                s.last_obj_fresh_frame_counter = frame_ctr;
            }
        } else {
            results.object_detector = s.cached_obj_output;
            StampPhysicalCacheStatus(results.object_detector.cache_status,
                                     decision, frame_ctr,
                                     s.last_obj_fresh_frame_counter);
        }
    }
    if (s.enable_flags.semantic_segmenter && s.semantic_segmenter) {
        const bool has_cached =
            s.cached_seg_output.state == PhysicalImageOperatorState::ModelLoaded &&
            s.last_seg_fresh_frame_counter != 0;
        const auto decision = DecidePhysicalCadence(
            s.cached_seg_cadence, scene, has_cached,
            s.last_seg_run_steady_ns, now_steady_ns);
        if (decision == PhysicalCadenceDecision::Run ||
            decision == PhysicalCadenceDecision::NoSignal) {
            s.semantic_segmenter->RouteFrameToPhysicalSemanticSegmenter(
                s.frame_view.model_image,
                frame_ctr,
                results.semantic_segmenter);
            StampPhysicalCacheStatus(results.semantic_segmenter.cache_status,
                                     decision, frame_ctr, frame_ctr);
            if (results.semantic_segmenter.state == PhysicalImageOperatorState::ModelLoaded) {
                s.cached_seg_output            = results.semantic_segmenter;
                s.last_seg_run_steady_ns       = now_steady_ns;
                s.last_seg_fresh_frame_counter = frame_ctr;
            }
        } else {
            results.semantic_segmenter = s.cached_seg_output;
            StampPhysicalCacheStatus(results.semantic_segmenter.cache_status,
                                     decision, frame_ctr,
                                     s.last_seg_fresh_frame_counter);
        }
    }
    if (s.enable_flags.image_classifier && s.image_classifier) {
        const bool has_cached =
            s.cached_cls_output.state == PhysicalImageOperatorState::ModelLoaded &&
            s.last_cls_fresh_frame_counter != 0;
        const auto decision = DecidePhysicalCadence(
            s.cached_cls_cadence, scene, has_cached,
            s.last_cls_run_steady_ns, now_steady_ns);
        if (decision == PhysicalCadenceDecision::Run ||
            decision == PhysicalCadenceDecision::NoSignal) {
            // Pick the highest-confidence "person" detection from this
            // frame's detector output. Whole-frame ImageNet classification
            // is dominated by background; classifying the person crop
            // gives meaningful clothing/activity labels instead of
            // "cloak/poncho/abaya" garbage from the wall behind them.
            const cv::Mat& full = s.frame_view.model_image;
            cv::Mat        cls_input = full;        // default = whole frame
            const PhysicalObjectDetection* best_person = nullptr;
            if (s.enable_flags.object_detector &&
                results.object_detector.state == PhysicalImageOperatorState::ModelLoaded) {
                for (const auto& d : results.object_detector.detections) {
                    if (d.class_label != "person") continue;
                    if (d.model_box.width <= 1.0f || d.model_box.height <= 1.0f) continue;
                    if (!best_person || d.confidence > best_person->confidence) {
                        best_person = &d;
                    }
                }
            }
            if (best_person) {
                // Pad bbox by 15% on each side so we capture the head/feet
                // and a bit of context (improves classifier robustness),
                // then clamp to image bounds.
                const float pad_x = best_person->model_box.width  * 0.15f;
                const float pad_y = best_person->model_box.height * 0.15f;
                float x0 = best_person->model_box.x - pad_x;
                float y0 = best_person->model_box.y - pad_y;
                float x1 = best_person->model_box.x + best_person->model_box.width  + pad_x;
                float y1 = best_person->model_box.y + best_person->model_box.height + pad_y;
                x0 = std::max(0.0f, x0);
                y0 = std::max(0.0f, y0);
                x1 = std::min(static_cast<float>(full.cols), x1);
                y1 = std::min(static_cast<float>(full.rows), y1);
                const int ix0 = static_cast<int>(std::floor(x0));
                const int iy0 = static_cast<int>(std::floor(y0));
                const int iw  = static_cast<int>(std::ceil (x1 - x0));
                const int ih  = static_cast<int>(std::ceil (y1 - y0));
                if (iw >= 8 && ih >= 8 &&
                    ix0 >= 0 && iy0 >= 0 &&
                    ix0 + iw <= full.cols && iy0 + ih <= full.rows) {
                    // No-copy ROI view; the classifier resizes internally.
                    cls_input = full(cv::Rect(ix0, iy0, iw, ih));
                }
            }
            s.image_classifier->RouteFrameToPhysicalImageClassifier(
                cls_input,
                frame_ctr,
                results.image_classifier);
            StampPhysicalCacheStatus(results.image_classifier.cache_status,
                                     decision, frame_ctr, frame_ctr);
            if (results.image_classifier.state == PhysicalImageOperatorState::ModelLoaded) {
                s.cached_cls_output            = results.image_classifier;
                s.last_cls_run_steady_ns       = now_steady_ns;
                s.last_cls_fresh_frame_counter = frame_ctr;
            }
        } else {
            results.image_classifier = s.cached_cls_output;
            StampPhysicalCacheStatus(results.image_classifier.cache_status,
                                     decision, frame_ctr,
                                     s.last_cls_fresh_frame_counter);
        }
    }
    if (s.enable_flags.pose_estimator && s.pose_estimator) {
        const bool has_cached =
            s.cached_pose_output.state == PhysicalImageOperatorState::ModelLoaded &&
            s.last_pose_fresh_frame_counter != 0;
        const auto decision = DecidePhysicalCadence(
            s.cached_pose_cadence, scene, has_cached,
            s.last_pose_run_steady_ns, now_steady_ns);
        if (decision == PhysicalCadenceDecision::Run ||
            decision == PhysicalCadenceDecision::NoSignal) {
            s.pose_estimator->RouteFrameToPhysicalPoseKeypointEstimator(
                s.frame_view.model_image,
                results.raw_to_model,
                results.raw_image_width,
                results.raw_image_height,
                frame_ctr,
                results.pose_estimator);
            StampPhysicalCacheStatus(results.pose_estimator.cache_status,
                                     decision, frame_ctr, frame_ctr);
            if (results.pose_estimator.state == PhysicalImageOperatorState::ModelLoaded) {
                s.cached_pose_output            = results.pose_estimator;
                s.last_pose_run_steady_ns       = now_steady_ns;
                s.last_pose_fresh_frame_counter = frame_ctr;
            }
        } else {
            results.pose_estimator = s.cached_pose_output;
            StampPhysicalCacheStatus(results.pose_estimator.cache_status,
                                     decision, frame_ctr,
                                     s.last_pose_fresh_frame_counter);
        }
    }
    if (s.enable_flags.scene_text_reader && s.scene_text_reader) {
        const bool has_cached =
            s.cached_text_output.state == PhysicalImageOperatorState::ModelLoaded &&
            s.last_text_fresh_frame_counter != 0;
        const auto decision = DecidePhysicalCadence(
            s.cached_text_cadence, scene, has_cached,
            s.last_text_run_steady_ns, now_steady_ns);
        if (decision == PhysicalCadenceDecision::Run ||
            decision == PhysicalCadenceDecision::NoSignal) {
            s.scene_text_reader->RouteFrameToPhysicalSceneTextReader(
                s.frame_view.model_image,
                results.raw_to_model,
                results.raw_image_width,
                results.raw_image_height,
                frame_ctr,
                results.scene_text_reader);
            StampPhysicalCacheStatus(results.scene_text_reader.cache_status,
                                     decision, frame_ctr, frame_ctr);
            if (results.scene_text_reader.state == PhysicalImageOperatorState::ModelLoaded) {
                s.cached_text_output            = results.scene_text_reader;
                s.last_text_run_steady_ns       = now_steady_ns;
                s.last_text_fresh_frame_counter = frame_ctr;
            }
        } else {
            results.scene_text_reader = s.cached_text_output;
            StampPhysicalCacheStatus(results.scene_text_reader.cache_status,
                                     decision, frame_ctr,
                                     s.last_text_fresh_frame_counter);
        }
    }
    if (s.enable_flags.facial_expression_detector && s.facial_expression_detector) {
        const bool has_cached =
            s.cached_face_output.state == PhysicalImageOperatorState::ModelLoaded &&
            s.last_face_fresh_frame_counter != 0;
        const auto decision = DecidePhysicalCadence(
            s.cached_face_cadence, scene, has_cached,
            s.last_face_run_steady_ns, now_steady_ns);
        if (decision == PhysicalCadenceDecision::Run ||
            decision == PhysicalCadenceDecision::NoSignal) {
            s.facial_expression_detector->RouteFrameToPhysicalFacialExpressionDetector(
                s.frame_view.model_image,
                results.raw_to_model,
                results.raw_image_width,
                results.raw_image_height,
                frame_ctr,
                results.facial_expression_detector);
            StampPhysicalCacheStatus(results.facial_expression_detector.cache_status,
                                     decision, frame_ctr, frame_ctr);
            if (results.facial_expression_detector.state == PhysicalImageOperatorState::ModelLoaded) {
                s.cached_face_output            = results.facial_expression_detector;
                s.last_face_run_steady_ns       = now_steady_ns;
                s.last_face_fresh_frame_counter = frame_ctr;
            }
        } else {
            results.facial_expression_detector = s.cached_face_output;
            StampPhysicalCacheStatus(results.facial_expression_detector.cache_status,
                                     decision, frame_ctr,
                                     s.last_face_fresh_frame_counter);
        }
    }

    // Entity tracker runs LAST among the operators because it consumes
    // the object detector's output as its input. It is given the same
    // raw_to_model + raw dims so its raw-space track boxes are coherent
    // with everything else in the result envelope. The tracker is never
    // cadence-gated — it is cheap and downstream consumers expect it to
    // smooth/extrapolate every frame.
    if (s.enable_flags.entity_tracker && s.entity_tracker) {
        s.entity_tracker->RouteDetectionsToPhysicalEntityTracker(
            results.object_detector.detections,
            results.raw_to_model,
            results.raw_image_width,
            results.raw_image_height,
            frame_ctr,
            results.entity_tracker);
    }

    // Instance segmenter (SAM 2) consumes the object detector's boxes as
    // prompts, so it MUST run after the detector populated
    // results.object_detector.detections. Order vs entity_tracker is
    // independent (no shared mutable state) but we keep the segmenter last
    // because its encoder pass is the most expensive operator on the bus.
    if (s.enable_flags.instance_segmenter && s.instance_segmenter) {
        const bool has_cached =
            s.cached_inst_seg_output.state == PhysicalImageOperatorState::ModelLoaded &&
            s.last_inst_seg_fresh_frame_counter != 0;
        const auto decision = DecidePhysicalCadence(
            s.cached_inst_seg_cadence, scene, has_cached,
            s.last_inst_seg_run_steady_ns, now_steady_ns);
        if (decision == PhysicalCadenceDecision::Run ||
            decision == PhysicalCadenceDecision::NoSignal) {
            s.instance_segmenter->RouteFrameAndDetectionsToPhysicalInstanceSegmenter(
                s.frame_view.model_image,
                results.object_detector.detections,
                frame_ctr,
                results.instance_segmenter);
            StampPhysicalCacheStatus(results.instance_segmenter.cache_status,
                                     decision, frame_ctr, frame_ctr);
            if (results.instance_segmenter.state == PhysicalImageOperatorState::ModelLoaded) {
                s.cached_inst_seg_output            = results.instance_segmenter;
                s.last_inst_seg_run_steady_ns       = now_steady_ns;
                s.last_inst_seg_fresh_frame_counter = frame_ctr;
            }
        } else {
            results.instance_segmenter = s.cached_inst_seg_output;
            StampPhysicalCacheStatus(results.instance_segmenter.cache_status,
                                     decision, frame_ctr,
                                     s.last_inst_seg_fresh_frame_counter);
        }
    }

    // Class policy runs LAST. It mutates results.object_detector.detections,
    // results.entity_tracker.tracks, results.instance_segmenter
    // .segmentation.instances, and results.image_classifier.top_k IN PLACE
    // (relabel by canonical, drop by confidence floor / priority cutoff)
    // and writes the ranked summary into results.class_policy. The single
    // Apply* call is the integration point Stage-2 exposes to whatever
    // builds the model context matrix downstream.
    if (s.enable_flags.class_policy && s.class_policy) {
        s.class_policy->ApplyPhysicalClassPolicyToPerceptionResults(
            results, results.class_policy);
    }

    try {
        PhysicalPerceptionPrimitiveBus::Instance().PublishPhysicalPerceptionResultsToBus(results);
        ++s.processed_count;
        s.last_error_reason.clear();
    } catch (const std::exception& e) {
        s.last_error_reason = std::string("PublishPhysicalPerceptionResultsToBus failed: ") + e.what();
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, s.last_error_reason);
    }
}

void ShutdownPhysicalPerceptionPrimitives() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.shutting_down = true;
    if (s.object_detector)    s.object_detector->ResetPhysicalObjectDetector();
    if (s.semantic_segmenter) s.semantic_segmenter->ResetPhysicalSemanticSegmenter();
    if (s.instance_segmenter) s.instance_segmenter->ResetPhysicalInstanceSegmenter();
    if (s.image_classifier)   s.image_classifier->ResetPhysicalImageClassifier();
    if (s.pose_estimator)     s.pose_estimator->ResetPhysicalPoseKeypointEstimator();
    if (s.scene_text_reader)  s.scene_text_reader->ResetPhysicalSceneTextReader();
    if (s.facial_expression_detector) s.facial_expression_detector->ResetPhysicalFacialExpressionDetector();
    if (s.entity_tracker)     s.entity_tracker->ResetPhysicalEntityTracker();
    if (s.class_policy)       s.class_policy->ResetPhysicalClassPolicy();
    s.object_detector.reset();
    s.semantic_segmenter.reset();
    s.instance_segmenter.reset();
    s.image_classifier.reset();
    s.pose_estimator.reset();
    s.scene_text_reader.reset();
    s.facial_expression_detector.reset();
    s.entity_tracker.reset();
    s.class_policy.reset();
    s.pending_obj_cfg.reset();
    s.pending_seg_cfg.reset();
    s.pending_inst_seg_cfg.reset();
    s.pending_cls_cfg.reset();
    s.pending_pose_cfg.reset();
    s.pending_text_cfg.reset();
    s.pending_face_cfg.reset();
    s.pending_track_cfg.reset();
    s.pending_class_policy_cfg.reset();
    PhysicalPerceptionPrimitiveBus::Instance().ResetPhysicalPerceptionPrimitiveBus();
    s.initialized          = false;
    s.tick_count           = 0;
    s.processed_count      = 0;
    s.last_seen_frame_ctr  = 0;
    // Drop cached cadence outputs so a re-init starts fresh (defaults).
    s.cached_obj_output            = {};
    s.cached_seg_output            = {};
    s.cached_inst_seg_output       = {};
    s.cached_cls_output            = {};
    s.cached_pose_output           = {};
    s.cached_text_output           = {};
    s.cached_face_output           = {};
    s.last_obj_run_steady_ns = s.last_obj_fresh_frame_counter = 0;
    s.last_seg_run_steady_ns = s.last_seg_fresh_frame_counter = 0;
    s.last_inst_seg_run_steady_ns = s.last_inst_seg_fresh_frame_counter = 0;
    s.last_cls_run_steady_ns = s.last_cls_fresh_frame_counter = 0;
    s.last_pose_run_steady_ns = s.last_pose_fresh_frame_counter = 0;
    s.last_text_run_steady_ns = s.last_text_fresh_frame_counter = 0;
    s.last_face_run_steady_ns = s.last_face_fresh_frame_counter = 0;
    LOG_DEBUG(PHYSICAL_PERC_PRIM_LOG_TAG, "ShutdownPhysicalPerceptionPrimitives: complete");
}

bool IsPhysicalPerceptionPrimitivesRunning() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.initialized && !s.shutting_down;
}

PhysicalPerceptionPrimitivesEnableFlags GetPhysicalPerceptionPrimitivesEnableFlags() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.enable_flags;
}

std::string GetLastPhysicalPerceptionPrimitivesError() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.last_error_reason;
}

uint64_t GetPhysicalPerceptionPrimitivesTickCount() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.tick_count;
}

uint64_t GetPhysicalPerceptionPrimitivesProcessedCount() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.processed_count;
}

void RequestSetPhysicalPerceptionPrimitivesEnableFlags(
    const PhysicalPerceptionPrimitivesEnableFlags& flags)
{
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.enable_flags = flags;
    LOG_DEBUG(PHYSICAL_PERC_PRIM_LOG_TAG,
              std::string("RequestSetPhysicalPerceptionPrimitivesEnableFlags: ")
              + "obj="  + (flags.object_detector    ? "1" : "0")
              + " seg=" + (flags.semantic_segmenter ? "1" : "0")
              + " cls=" + (flags.image_classifier   ? "1" : "0")
              + " pose="+ (flags.pose_estimator     ? "1" : "0")
              + " text="+ (flags.scene_text_reader  ? "1" : "0")
              + " face="+ (flags.facial_expression_detector ? "1" : "0")
              + " track="+ (flags.entity_tracker ? "1" : "0")
              + " inst_seg="+ (flags.instance_segmenter ? "1" : "0")
              + " class_policy="+ (flags.class_policy ? "1" : "0"));
}

// Each Request* either applies immediately (init done) or stages the cfg
// for the first Tick. Errors propagate to the caller via re-throw.

void RequestConfigurePhysicalObjectDetector(const PhysicalObjectDetectorConfig& cfg) {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.cached_obj_cadence = cfg.cadence;
    if (!s.initialized) {
        s.pending_obj_cfg = std::make_unique<PhysicalObjectDetectorConfig>(cfg);
        return;
    }
    if (!s.object_detector) {
        throw std::runtime_error(
            "RequestConfigurePhysicalObjectDetector: subsystem initialised but "
            "object_detector is null — likely after ShutdownPhysicalPerceptionPrimitives()");
    }
    s.object_detector->LoadOnnxModelIntoPhysicalObjectDetector(cfg);
}

void RequestConfigurePhysicalSemanticSegmenter(const PhysicalSemanticSegmenterConfig& cfg) {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.cached_seg_cadence = cfg.cadence;
    if (!s.initialized) {
        s.pending_seg_cfg = std::make_unique<PhysicalSemanticSegmenterConfig>(cfg);
        return;
    }
    if (!s.semantic_segmenter) {
        throw std::runtime_error("RequestConfigurePhysicalSemanticSegmenter: semantic_segmenter is null");
    }
    s.semantic_segmenter->LoadOnnxModelIntoPhysicalSemanticSegmenter(cfg);
}

void RequestConfigurePhysicalImageClassifier(const PhysicalImageClassifierConfig& cfg) {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.cached_cls_cadence = cfg.cadence;
    if (!s.initialized) {
        s.pending_cls_cfg = std::make_unique<PhysicalImageClassifierConfig>(cfg);
        return;
    }
    if (!s.image_classifier) {
        throw std::runtime_error("RequestConfigurePhysicalImageClassifier: image_classifier is null");
    }
    s.image_classifier->LoadOnnxModelIntoPhysicalImageClassifier(cfg);
}

void RequestConfigurePhysicalPoseKeypointEstimator(const PhysicalPoseKeypointEstimatorConfig& cfg) {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.cached_pose_cadence = cfg.cadence;
    if (!s.initialized) {
        s.pending_pose_cfg = std::make_unique<PhysicalPoseKeypointEstimatorConfig>(cfg);
        return;
    }
    if (!s.pose_estimator) {
        throw std::runtime_error("RequestConfigurePhysicalPoseKeypointEstimator: pose_estimator is null");
    }
    s.pose_estimator->LoadOnnxModelIntoPhysicalPoseKeypointEstimator(cfg);
}

void RequestConfigurePhysicalSceneTextReader(const PhysicalSceneTextReaderConfig& cfg) {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.cached_text_cadence = cfg.cadence;
    if (!s.initialized) {
        s.pending_text_cfg = std::make_unique<PhysicalSceneTextReaderConfig>(cfg);
        return;
    }
    if (!s.scene_text_reader) {
        throw std::runtime_error("RequestConfigurePhysicalSceneTextReader: scene_text_reader is null");
    }
    s.scene_text_reader->LoadOnnxModelsIntoPhysicalSceneTextReader(cfg);
}

void RequestConfigurePhysicalFacialExpressionDetector(const PhysicalFacialExpressionDetectorConfig& cfg) {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.cached_face_cadence = cfg.cadence;
    if (!s.initialized) {
        s.pending_face_cfg = std::make_unique<PhysicalFacialExpressionDetectorConfig>(cfg);
        return;
    }
    if (!s.facial_expression_detector) {
        throw std::runtime_error("RequestConfigurePhysicalFacialExpressionDetector: facial_expression_detector is null");
    }
    s.facial_expression_detector->LoadOnnxModelsIntoPhysicalFacialExpressionDetector(cfg);
}

void RequestConfigurePhysicalEntityTracker(const PhysicalEntityTrackerConfig& cfg) {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    if (!s.initialized) {
        s.pending_track_cfg = std::make_unique<PhysicalEntityTrackerConfig>(cfg);
        return;
    }
    if (!s.entity_tracker) {
        throw std::runtime_error(
            "RequestConfigurePhysicalEntityTracker: subsystem initialised but "
            "entity_tracker is null \u2014 likely after ShutdownPhysicalPerceptionPrimitives()");
    }
    s.entity_tracker->ConfigurePhysicalEntityTracker(cfg);
}

void RequestConfigurePhysicalInstanceSegmenter(const PhysicalInstanceSegmenterConfig& cfg) {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.cached_inst_seg_cadence = cfg.cadence;
    if (!s.initialized) {
        s.pending_inst_seg_cfg = std::make_unique<PhysicalInstanceSegmenterConfig>(cfg);
        return;
    }
    if (!s.instance_segmenter) {
        throw std::runtime_error(
            "RequestConfigurePhysicalInstanceSegmenter: subsystem initialised but "
            "instance_segmenter is null \xE2\x80\x94 likely after ShutdownPhysicalPerceptionPrimitives()");
    }
    s.instance_segmenter->LoadOnnxModelsIntoPhysicalInstanceSegmenter(cfg);
}

void RequestConfigurePhysicalClassPolicy(const PhysicalClassPolicyConfig& cfg) {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    if (!s.initialized) {
        s.pending_class_policy_cfg = std::make_unique<PhysicalClassPolicyConfig>(cfg);
        return;
    }
    if (!s.class_policy) {
        throw std::runtime_error(
            "RequestConfigurePhysicalClassPolicy: subsystem initialised but "
            "class_policy is null \u2014 likely after ShutdownPhysicalPerceptionPrimitives()");
    }
    s.class_policy->ConfigurePhysicalClassPolicy(cfg);
}

}}} // namespace
