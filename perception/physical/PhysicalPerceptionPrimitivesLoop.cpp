#include "PhysicalPerceptionPrimitivesLoop.hpp"

#include "PhysicalFrameBus.hpp"
#include "PhysicalPerceptionPrimitiveBus.hpp"
#include "PhysicalPerceptionPrimitivesLogTag.hpp"
#include "logger.hpp"

#include <atomic>
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
    std::unique_ptr<PhysicalImageClassifier>         image_classifier;
    std::unique_ptr<PhysicalPoseKeypointEstimator>   pose_estimator;
    std::unique_ptr<PhysicalSceneTextReader>         scene_text_reader;

    // Pending configs deposited via Request* before lazy init. Applied at
    // the first Tick. After init these are emptied; subsequent Request*
    // calls operate directly on the operator instances.
    std::unique_ptr<PhysicalObjectDetectorConfig>          pending_obj_cfg;
    std::unique_ptr<PhysicalSemanticSegmenterConfig>       pending_seg_cfg;
    std::unique_ptr<PhysicalImageClassifierConfig>         pending_cls_cfg;
    std::unique_ptr<PhysicalPoseKeypointEstimatorConfig>   pending_pose_cfg;
    std::unique_ptr<PhysicalSceneTextReaderConfig>         pending_text_cfg;

    std::string  last_error_reason;
    uint64_t     tick_count           = 0;
    uint64_t     processed_count      = 0;
    uint64_t     last_seen_frame_ctr  = 0;

    // Pre-allocated FrameView so we don't reallocate cv::Mats every Tick.
    PhysicalFrameBus::FrameView frame_view;
};

PhysicalPerceptionPrimitivesState& GetState() {
    static PhysicalPerceptionPrimitivesState s;
    return s;
}

void LazyInitLocked(PhysicalPerceptionPrimitivesState& s) {
    if (s.initialized) return;
    LOG_DEBUG(PHYSICAL_PERC_PRIM_LOG_TAG,
              "TickPhysicalPerceptionPrimitives: first call — running lazy init");
    s.object_detector    = std::make_unique<PhysicalObjectDetector>();
    s.semantic_segmenter = std::make_unique<PhysicalSemanticSegmenter>();
    s.image_classifier   = std::make_unique<PhysicalImageClassifier>();
    s.pose_estimator     = std::make_unique<PhysicalPoseKeypointEstimator>();
    s.scene_text_reader  = std::make_unique<PhysicalSceneTextReader>();

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
    apply(s.pending_cls_cfg,
          [&](auto& c){ s.image_classifier->LoadOnnxModelIntoPhysicalImageClassifier(c); },
          "PhysicalImageClassifier");
    apply(s.pending_pose_cfg,
          [&](auto& c){ s.pose_estimator->LoadOnnxModelIntoPhysicalPoseKeypointEstimator(c); },
          "PhysicalPoseKeypointEstimator");
    apply(s.pending_text_cfg,
          [&](auto& c){ s.scene_text_reader->LoadOnnxModelsIntoPhysicalSceneTextReader(c); },
          "PhysicalSceneTextReader");

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

    // Each operator runs only if its enable flag is on; the operator itself
    // gates on its state, so calling RouteFrameTo* when NoModelConfigured
    // is cheap and just fills the output envelope with state info.
    if (s.enable_flags.object_detector && s.object_detector) {
        s.object_detector->RouteFrameToPhysicalObjectDetector(
            s.frame_view.model_image,
            results.raw_to_model,
            results.raw_image_width,
            results.raw_image_height,
            results.source_frame_counter,
            results.object_detector);
    }
    if (s.enable_flags.semantic_segmenter && s.semantic_segmenter) {
        s.semantic_segmenter->RouteFrameToPhysicalSemanticSegmenter(
            s.frame_view.model_image,
            results.source_frame_counter,
            results.semantic_segmenter);
    }
    if (s.enable_flags.image_classifier && s.image_classifier) {
        s.image_classifier->RouteFrameToPhysicalImageClassifier(
            s.frame_view.model_image,
            results.source_frame_counter,
            results.image_classifier);
    }
    if (s.enable_flags.pose_estimator && s.pose_estimator) {
        s.pose_estimator->RouteFrameToPhysicalPoseKeypointEstimator(
            s.frame_view.model_image,
            results.raw_to_model,
            results.raw_image_width,
            results.raw_image_height,
            results.source_frame_counter,
            results.pose_estimator);
    }
    if (s.enable_flags.scene_text_reader && s.scene_text_reader) {
        s.scene_text_reader->RouteFrameToPhysicalSceneTextReader(
            s.frame_view.model_image,
            results.raw_to_model,
            results.raw_image_width,
            results.raw_image_height,
            results.source_frame_counter,
            results.scene_text_reader);
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
    if (s.image_classifier)   s.image_classifier->ResetPhysicalImageClassifier();
    if (s.pose_estimator)     s.pose_estimator->ResetPhysicalPoseKeypointEstimator();
    if (s.scene_text_reader)  s.scene_text_reader->ResetPhysicalSceneTextReader();
    s.object_detector.reset();
    s.semantic_segmenter.reset();
    s.image_classifier.reset();
    s.pose_estimator.reset();
    s.scene_text_reader.reset();
    s.pending_obj_cfg.reset();
    s.pending_seg_cfg.reset();
    s.pending_cls_cfg.reset();
    s.pending_pose_cfg.reset();
    s.pending_text_cfg.reset();
    PhysicalPerceptionPrimitiveBus::Instance().ResetPhysicalPerceptionPrimitiveBus();
    s.initialized          = false;
    s.tick_count           = 0;
    s.processed_count      = 0;
    s.last_seen_frame_ctr  = 0;
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
              + " text="+ (flags.scene_text_reader  ? "1" : "0"));
}

// Each Request* either applies immediately (init done) or stages the cfg
// for the first Tick. Errors propagate to the caller via re-throw.

void RequestConfigurePhysicalObjectDetector(const PhysicalObjectDetectorConfig& cfg) {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
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
    if (!s.initialized) {
        s.pending_text_cfg = std::make_unique<PhysicalSceneTextReaderConfig>(cfg);
        return;
    }
    if (!s.scene_text_reader) {
        throw std::runtime_error("RequestConfigurePhysicalSceneTextReader: scene_text_reader is null");
    }
    s.scene_text_reader->LoadOnnxModelsIntoPhysicalSceneTextReader(cfg);
}

}}} // namespace
