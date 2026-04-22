#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>

#include <opencv2/core.hpp>

#include "PhysicalFrameConditioner.hpp"  // for PhysicalSignalRawToModelTransform

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
        cv::Mat                              raw_image;        // BGR8 copy from camera
        cv::Mat                              model_image;      // BGR8 copy after conditioning
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
    cv::Mat                                       latest_raw_image_;
    cv::Mat                                       latest_model_image_;
    uint64_t                                      latest_counter_ = 0;
    std::chrono::steady_clock::time_point         latest_time_{};
    std::string                                   latest_url_;
    std::string                                   latest_label_;
    PhysicalFrameMetadata                         latest_metadata_{};
    bool                                          ever_published_ = false;
};

}}} // namespace GRIM::Perception::Physical
