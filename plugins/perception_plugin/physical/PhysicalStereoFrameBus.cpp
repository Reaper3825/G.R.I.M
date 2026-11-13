#include "PhysicalStereoFrameBus.hpp"

#include <stdexcept>

namespace GRIM { namespace Perception { namespace Physical {

PhysicalStereoFrameBus& PhysicalStereoFrameBus::Instance() {
    static PhysicalStereoFrameBus bus;
    return bus;
}

void PhysicalStereoFrameBus::PublishPhysicalStereoFramePairToBus(
    const PhysicalStereoFramePair& pair,
    const PhysicalStereoCaptureConfig& config)
{
    if (pair.left_image.empty() || pair.right_image.empty()) {
        throw std::runtime_error(
            "PhysicalStereoFrameBus::PublishPhysicalStereoFramePairToBus: both images are required");
    }
    if (pair.pair_counter == 0 || pair.left_capture_steady_ns == 0
        || pair.right_capture_steady_ns == 0) {
        throw std::runtime_error(
            "PhysicalStereoFrameBus::PublishPhysicalStereoFramePairToBus: pair provenance is incomplete");
    }

    auto packet = std::make_shared<PhysicalStereoFramePacket>();
    pair.left_image.copyTo(packet->left_image);
    pair.right_image.copyTo(packet->right_image);
    packet->pair_counter            = pair.pair_counter;
    packet->left_frame_counter      = pair.left_frame_counter;
    packet->right_frame_counter     = pair.right_frame_counter;
    packet->left_capture_steady_ns  = pair.left_capture_steady_ns;
    packet->right_capture_steady_ns = pair.right_capture_steady_ns;
    packet->signed_skew_ms          = pair.signed_skew_ms;
    packet->left_url                = config.left_url;
    packet->left_label              = config.left_label;
    packet->right_url               = config.right_url;
    packet->right_label             = config.right_label;
    packet->published_at            = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lk(mutex_);
    latest_packet_ = std::move(packet);
    ever_published_ = true;
}

bool PhysicalStereoFrameBus::PullLatestPhysicalStereoFrameView(
    FrameView& out,
    uint64_t& last_seen_pair_counter) const
{
    std::lock_guard<std::mutex> lk(mutex_);
    if (!ever_published_) return false;
    if (!latest_packet_) {
        throw std::runtime_error(
            "PhysicalStereoFrameBus::PullLatestPhysicalStereoFrameView: published state has no packet");
    }
    if (latest_packet_->pair_counter == last_seen_pair_counter) return false;
    out.packet                  = latest_packet_;
    out.left_image              = latest_packet_->left_image;
    out.right_image             = latest_packet_->right_image;
    out.pair_counter            = latest_packet_->pair_counter;
    out.left_frame_counter      = latest_packet_->left_frame_counter;
    out.right_frame_counter     = latest_packet_->right_frame_counter;
    out.left_capture_steady_ns  = latest_packet_->left_capture_steady_ns;
    out.right_capture_steady_ns = latest_packet_->right_capture_steady_ns;
    out.signed_skew_ms          = latest_packet_->signed_skew_ms;
    out.left_url                = latest_packet_->left_url;
    out.left_label              = latest_packet_->left_label;
    out.right_url               = latest_packet_->right_url;
    out.right_label             = latest_packet_->right_label;
    last_seen_pair_counter      = latest_packet_->pair_counter;
    return true;
}

bool PhysicalStereoFrameBus::HasEverPublishedStereoFrame() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return ever_published_;
}

void PhysicalStereoFrameBus::ResetPhysicalStereoFrameBus() {
    std::lock_guard<std::mutex> lk(mutex_);
    latest_packet_.reset();
    ever_published_ = false;
}

}}} // namespace GRIM::Perception::Physical