#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include <opencv2/core.hpp>

#include "PhysicalFrameConditioner.hpp"  // for PhysicalSignalRawToModelTransform
#include "PhysicalSceneStability.hpp"    // for PhysicalSceneStability

namespace GRIM { namespace Perception { namespace Physical {

// Per-frame provenance carried alongside the pixel data on the bus.
// Vision-side consumers MUST treat these as authoritative — never re-derive
// scale/offset from raw_image vs model_image dimensions because letterbox
// padding makes the relationship non-trivial.
struct PhysicalFrameMetadata {
    uint64_t                          capture_steady_ns = 0;  // monotonic clock
    uint64_t                          publish_steady_ns = 0;  // monotonic clock
    uint64_t                          capture_wall_ns   = 0;  // wall clock
    int                               raw_width    = 0;
    int                               raw_height   = 0;
    int                               model_width  = 0;
    int                               model_height = 0;
    PhysicalSignalRawToModelTransform raw_to_model{};
    std::string                       color_space_label;       // "BGR8_SRGB" / "GRAY8_SRGB"
    std::string                       pipeline_summary;
    double                            applied_exposure_gain = 1.0;

    // Per-frame scene-change signal computed by PhysicalFrameConditioner.
    // Consumed by every cache-aware operator gate so the test runs exactly
    // once per frame (not once per operator).
    PhysicalSceneStability            scene_stability{};

    // Producer/bus timing telemetry, in milliseconds. `conditioning_*` is
    // copied from PhysicalFrameConditioner. `frame_bus_publish_copy_ms` is
    // stamped by PublishPhysicalFrameToBus after cloning into the immutable
    // latest-frame packet; `frame_bus_pull_copy_ms` is stamped per consumer
    // pull and should stay near-zero because the pull path shares cv::Mat
    // headers instead of copying pixel buffers.
    double                            conditioning_total_ms           = 0.0;
    double                            conditioning_quality_gate_ms    = 0.0;
    double                            conditioning_stabilization_ms   = 0.0;
    double                            conditioning_denoise_ms         = 0.0;
    double                            conditioning_exposure_ms        = 0.0;
    double                            conditioning_deblur_ms          = 0.0;
    double                            conditioning_resize_ms          = 0.0;
    double                            conditioning_color_convert_ms   = 0.0;
    double                            conditioning_scene_stability_ms = 0.0;
    double                            frame_bus_publish_copy_ms       = 0.0;
    double                            frame_bus_pull_copy_ms          = 0.0;
};

// Immutable latest-frame packet owned by the bus and shared by consumers.
// The cv::Mat headers in FrameView are shallow views into this packet; keeping
// `packet` alive inside FrameView pins the underlying pixel buffers until the
// consumer moves to a newer frame. Consumers MUST treat raw_image/model_image
// as read-only; clone locally before mutation.
struct PhysicalFramePacket {
    cv::Mat                              raw_image;
    cv::Mat                              model_image;
    uint64_t                             frame_counter = 0;
    std::chrono::steady_clock::time_point published_at{};
    std::string                          source_url;
    std::string                          source_label;
    PhysicalFrameMetadata                metadata{};
};

// Latest-frame slot. Single producer (PhysicalEnvironmentLoop), multiple
// consumers (UI panel today; model context matrix in later stages).
//
// This is the OFFICIAL handoff point between the camera subsystem and the
// rest of GRIM. When future stages add `RouteFrameToContextMatrix(...)`,
// they read from here.
class PhysicalFrameBus {
public:
    struct FrameView {
        std::shared_ptr<const PhysicalFramePacket> packet;           // pins shared pixel buffers
        cv::Mat                              raw_image;        // BGR8 shared view from camera
        cv::Mat                              model_image;      // BGR8 shared view after conditioning
        cv::Mat                              image;            // alias of model_image for existing consumers
        uint64_t                             frame_counter = 0;
        std::chrono::steady_clock::time_point published_at{};
        std::string                          source_url;
        std::string                          source_label;
        PhysicalFrameMetadata                metadata{};
    };

    static PhysicalFrameBus& Instance();

    // Producer side. Throws if either image is empty (Rule 20).
    void PublishPhysicalFrameToBus(const cv::Mat& raw_image,
                                   const cv::Mat& model_image,
                                   uint64_t frame_counter,
                                   const std::string& source_url,
                                   const std::string& source_label,
                                   const PhysicalFrameMetadata& metadata);

    // Consumer side. Returns true if a frame is available AND its counter is
    // newer than `last_seen_counter`. Updates `last_seen_counter` in that case.
    bool PullLatestFrameView(FrameView& out, uint64_t& last_seen_counter) const;

    // True if any frame has ever been published.
    bool HasEverPublishedFrame() const;

    // Clear the bus (e.g. when the active source changes).
    void ResetPhysicalFrameBus();

private:
    PhysicalFrameBus() = default;
    PhysicalFrameBus(const PhysicalFrameBus&)            = delete;
    PhysicalFrameBus& operator=(const PhysicalFrameBus&) = delete;

    mutable std::mutex                            mutex_;
    std::shared_ptr<const PhysicalFramePacket>    latest_packet_;
    bool                                          ever_published_ = false;
};

}}} // namespace GRIM::Perception::Physical
