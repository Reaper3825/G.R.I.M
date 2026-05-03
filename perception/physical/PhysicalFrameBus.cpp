#include "PhysicalFrameBus.hpp"

#include <chrono>
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
    const auto t0 = std::chrono::steady_clock::now();
    PhysicalFrameMetadata stamped_metadata = metadata;

    auto packet = std::make_shared<PhysicalFramePacket>();
    raw_image.copyTo(packet->raw_image);
    model_image.copyTo(packet->model_image);
    std::lock_guard<std::mutex> lk(mutex_);
    stamped_metadata.frame_bus_publish_copy_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
    packet->frame_counter = frame_counter;
    packet->published_at  = std::chrono::steady_clock::now();
    packet->source_url    = source_url;
    packet->source_label  = source_label;
    packet->metadata      = stamped_metadata;
    latest_packet_        = std::move(packet);
    ever_published_       = true;
}

bool PhysicalFrameBus::PullLatestFrameView(FrameView& out,
                                            uint64_t& last_seen_counter) const {
    const auto t0 = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(mutex_);
    if (!ever_published_) return false;
    if (!latest_packet_) {
        throw std::runtime_error(
            "PhysicalFrameBus::PullLatestFrameView: ever_published=true but latest_packet_ is NULL");
    }
    if (latest_packet_->frame_counter == last_seen_counter) return false;

    out.packet        = latest_packet_;
    out.raw_image     = latest_packet_->raw_image;
    out.model_image   = latest_packet_->model_image;
    out.image         = out.model_image;
    out.frame_counter = latest_packet_->frame_counter;
    out.published_at  = latest_packet_->published_at;
    out.source_url    = latest_packet_->source_url;
    out.source_label  = latest_packet_->source_label;
    out.metadata      = latest_packet_->metadata;
    out.metadata.frame_bus_pull_copy_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
    last_seen_counter = latest_packet_->frame_counter;
    return true;
}

bool PhysicalFrameBus::HasEverPublishedFrame() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return ever_published_;
}

void PhysicalFrameBus::ResetPhysicalFrameBus() {
    std::lock_guard<std::mutex> lk(mutex_);
    latest_packet_.reset();
    ever_published_ = false;
}

}}} // namespace
