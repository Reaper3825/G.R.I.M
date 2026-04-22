#include "PhysicalFrameBus.hpp"

#include <stdexcept>

namespace GRIM { namespace Perception { namespace Physical {

PhysicalFrameBus& PhysicalFrameBus::Instance() {
    static PhysicalFrameBus inst;
    return inst;
}

void PhysicalFrameBus::PublishPhysicalFrameToBus(const cv::Mat& raw_image,
                                                 const cv::Mat& model_image,
                                                 uint64_t frame_counter,
                                                 const std::string& source_url,
                                                 const std::string& source_label,
                                                 const PhysicalFrameMetadata& metadata) {
    if (raw_image.empty()) {
        throw std::runtime_error(
            "PhysicalFrameBus::PublishPhysicalFrameToBus: raw_image is empty — "
            "producer MUST NOT publish empty raw frames");
    }
    if (model_image.empty()) {
        throw std::runtime_error(
            "PhysicalFrameBus::PublishPhysicalFrameToBus: model_image is empty — "
            "producer MUST NOT publish empty model frames");
    }
    std::lock_guard<std::mutex> lk(mutex_);
    raw_image.copyTo(latest_raw_image_);
    model_image.copyTo(latest_model_image_);
    latest_counter_  = frame_counter;
    latest_time_     = std::chrono::steady_clock::now();
    latest_url_      = source_url;
    latest_label_    = source_label;
    latest_metadata_ = metadata;
    ever_published_  = true;
}

bool PhysicalFrameBus::PullLatestFrameView(FrameView& out,
                                            uint64_t& last_seen_counter) const {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!ever_published_) return false;
    if (latest_counter_ == last_seen_counter) return false;
    latest_raw_image_.copyTo(out.raw_image);
    latest_model_image_.copyTo(out.model_image);
    out.model_image.copyTo(out.image);
    out.frame_counter = latest_counter_;
    out.published_at  = latest_time_;
    out.source_url    = latest_url_;
    out.source_label  = latest_label_;
    out.metadata      = latest_metadata_;
    last_seen_counter = latest_counter_;
    return true;
}

bool PhysicalFrameBus::HasEverPublishedFrame() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return ever_published_;
}

void PhysicalFrameBus::ResetPhysicalFrameBus() {
    std::lock_guard<std::mutex> lk(mutex_);
    latest_raw_image_.release();
    latest_model_image_.release();
    latest_counter_  = 0;
    latest_time_     = {};
    latest_url_.clear();
    latest_label_.clear();
    latest_metadata_ = PhysicalFrameMetadata{};
    ever_published_  = false;
}

}}} // namespace
