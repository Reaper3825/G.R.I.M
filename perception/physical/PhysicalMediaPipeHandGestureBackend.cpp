#include "PhysicalHandGestureBackend.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <string>
#include <utility>

#if defined(GRIM_HAS_MEDIAPIPE_TASKS_C)
#include "mediapipe/tasks/c/core/common.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/gesture_recognizer/gesture_recognizer.h"
#endif

namespace GRIM { namespace Perception { namespace Physical {

namespace {

uint64_t SteadyNowNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

#if defined(GRIM_HAS_MEDIAPIPE_TASKS_C)

std::string TakeMediaPipeError(char*& error_message) {
    std::string value = error_message ? error_message : "MediaPipe returned no detail";
    if (error_message) {
        MpErrorFree(error_message);
        error_message = nullptr;
    }
    return value;
}

std::string StatusError(const char* operation, MpStatus status, char*& detail) {
    std::ostringstream out;
    out << operation << " failed (MpStatus " << static_cast<int>(status)
        << "): " << TakeMediaPipeError(detail);
    return out.str();
}

const char* CategoryLabel(const Category& category) {
    if (category.category_name && category.category_name[0] != '\0') {
        return category.category_name;
    }
    if (category.display_name && category.display_name[0] != '\0') {
        return category.display_name;
    }
    return "";
}

class PhysicalMediaPipeHandGestureBackend final
    : public IPhysicalHandGestureBackend {
public:
    ~PhysicalMediaPipeHandGestureBackend() override { Shutdown(); }

    bool Initialize(const PhysicalHandGestureConfig& config,
                    std::string& error) override
    {
        Shutdown();
        if (config.model_path.empty()) {
            error = "No local MediaPipe gesture model is configured";
            return false;
        }
        if (config.model_path.find("://") != std::string::npos) {
            error = "MediaPipe model_path must be a local filesystem path; URIs are rejected";
            return false;
        }
        std::error_code fs_error;
        if (!std::filesystem::is_regular_file(config.model_path, fs_error)) {
            error = "Local MediaPipe gesture model is missing: " + config.model_path;
            return false;
        }

        GestureRecognizerOptions options{};
        options.base_options.model_asset_buffer = nullptr;
        options.base_options.model_asset_buffer_count = 0;
        options.base_options.model_asset_path = config.model_path.c_str();
        options.base_options.delegate = CPU;
        options.base_options.host_environment = HOST_ENVIRONMENT_UNKNOWN;
#if defined(_WIN32)
        options.base_options.host_system = HOST_SYSTEM_WINDOWS;
#elif defined(__APPLE__)
        options.base_options.host_system = HOST_SYSTEM_MAC;
#elif defined(__linux__)
        options.base_options.host_system = HOST_SYSTEM_LINUX;
#else
        options.base_options.host_system = HOST_SYSTEM_UNKNOWN;
#endif
        options.base_options.host_version = nullptr;
        // Deliberately null. A GRIM-approved MediaPipe build must compile out
        // usage logging; it must never be given a CA bundle or network route.
        options.base_options.ca_bundle_path = nullptr;
        options.running_mode = VIDEO;
        options.num_hands = std::clamp(config.max_hands, 1, 4);
        options.min_hand_detection_confidence =
            std::clamp(config.min_hand_detection_confidence, 0.0f, 1.0f);
        options.min_hand_presence_confidence =
            std::clamp(config.min_hand_presence_confidence, 0.0f, 1.0f);
        options.min_tracking_confidence =
            std::clamp(config.min_tracking_confidence, 0.0f, 1.0f);
        options.canned_gestures_classifier_options.max_results = 1;
        options.canned_gestures_classifier_options.score_threshold =
            std::clamp(config.min_gesture_confidence, 0.0f, 1.0f);
        options.custom_gestures_classifier_options.max_results = 1;
        options.custom_gestures_classifier_options.score_threshold =
            std::clamp(config.min_gesture_confidence, 0.0f, 1.0f);
        options.result_callback = nullptr;

        char* detail = nullptr;
        const MpStatus status =
            MpGestureRecognizerCreate(&options, &recognizer_, &detail);
        if (status != kMpOk || !recognizer_) {
            error = StatusError("MpGestureRecognizerCreate", status, detail);
            recognizer_ = nullptr;
            return false;
        }
        if (detail) MpErrorFree(detail);
        max_hands_ = options.num_hands;
        last_timestamp_ms_ = -1;
        return true;
    }

    bool Process(const PhysicalHandGestureFrame& frame,
                 std::vector<PhysicalHandObservation>& observations,
                 double& inference_ms,
                 std::string& error) override
    {
        observations.clear();
        inference_ms = 0.0;
        if (!recognizer_) {
            error = "MediaPipe gesture recognizer is not initialized";
            return false;
        }
        if (!frame.rgb_data || frame.width <= 0 || frame.height <= 0 ||
            frame.byte_count != frame.width * frame.height * 3) {
            error = "Gesture input must be a packed RGB8 image";
            return false;
        }

        MpImagePtr image = nullptr;
        char* detail = nullptr;
        MpStatus status = MpImageCreateFromUint8Data(
            kMpImageFormatSrgb, frame.width, frame.height, frame.rgb_data,
            frame.byte_count, &image, &detail);
        if (status != kMpOk || !image) {
            error = StatusError("MpImageCreateFromUint8Data", status, detail);
            return false;
        }
        if (detail) {
            MpErrorFree(detail);
            detail = nullptr;
        }

        int64_t timestamp_ms = static_cast<int64_t>(
            frame.source_capture_steady_ns / 1000000ULL);
        if (timestamp_ms <= last_timestamp_ms_) timestamp_ms = last_timestamp_ms_ + 1;

        GestureRecognizerResult result{};
        const auto started = std::chrono::steady_clock::now();
        status = MpGestureRecognizerRecognizeForVideo(
            recognizer_, image, nullptr, timestamp_ms, &result, &detail);
        inference_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        MpImageFree(image);

        if (status != kMpOk) {
            error = StatusError("MpGestureRecognizerRecognizeForVideo", status, detail);
            return false;
        }
        if (detail) MpErrorFree(detail);
        last_timestamp_ms_ = timestamp_ms;

        const uint32_t hand_count = result.hand_landmarks
            ? std::min<uint32_t>(result.hand_landmarks_count,
                                 static_cast<uint32_t>(max_hands_))
            : 0;
        observations.reserve(hand_count);
        for (uint32_t hand_index = 0; hand_index < hand_count; ++hand_index) {
            PhysicalHandObservation hand;
            hand.source_frame_counter = frame.source_frame_counter;
            hand.source_capture_steady_ns = frame.source_capture_steady_ns;
            hand.result_steady_ns = SteadyNowNs();
            hand.raw_image_width = frame.width;
            hand.raw_image_height = frame.height;

            if (result.handedness && hand_index < result.handedness_count) {
                const Categories& handedness = result.handedness[hand_index];
                if (handedness.categories_count > 0 && handedness.categories) {
                    const Category& top = handedness.categories[0];
                    const std::string label = CategoryLabel(top);
                    hand.handedness_confidence = top.score;
                    if (label == "Left" || label == "left") {
                        hand.handedness = PhysicalHandedness::Left;
                    } else if (label == "Right" || label == "right") {
                        hand.handedness = PhysicalHandedness::Right;
                    }
                }
            }

            if (result.gestures && hand_index < result.gestures_count) {
                const Categories& gestures = result.gestures[hand_index];
                if (gestures.categories_count > 0 && gestures.categories) {
                    const Category& top = gestures.categories[0];
                    hand.gesture_label = CategoryLabel(top);
                    hand.gesture_confidence = top.score;
                }
            }

            const NormalizedLandmarks& normalized =
                result.hand_landmarks[hand_index];
            const uint32_t landmark_count = normalized.landmarks
                ? std::min<uint32_t>(21, normalized.landmarks_count) : 0;
            hand.landmark_count = landmark_count;
            for (uint32_t i = 0; i < landmark_count; ++i) {
                const NormalizedLandmark& source = normalized.landmarks[i];
                auto& target = hand.landmarks[i];
                target.normalized_x = source.x;
                target.normalized_y = source.y;
                target.normalized_z = source.z;
                target.raw_pixel_x = source.x * static_cast<float>(frame.width);
                target.raw_pixel_y = source.y * static_cast<float>(frame.height);
            }

            if (result.hand_world_landmarks &&
                hand_index < result.hand_world_landmarks_count) {
                const Landmarks& world = result.hand_world_landmarks[hand_index];
                const uint32_t world_count = world.landmarks
                    ? std::min<uint32_t>(landmark_count, world.landmarks_count) : 0;
                for (uint32_t i = 0; i < world_count; ++i) {
                    auto& target = hand.landmarks[i];
                    target.world_x_m = world.landmarks[i].x;
                    target.world_y_m = world.landmarks[i].y;
                    target.world_z_m = world.landmarks[i].z;
                    target.has_world = true;
                }
            }
            observations.push_back(std::move(hand));
        }
        MpGestureRecognizerCloseResult(&result);
        return true;
    }

    void Shutdown() noexcept override {
        if (!recognizer_) return;
        char* detail = nullptr;
        (void)MpGestureRecognizerClose(recognizer_, &detail);
        if (detail) MpErrorFree(detail);
        recognizer_ = nullptr;
        last_timestamp_ms_ = -1;
        max_hands_ = 1;
    }

    const char* BackendName() const noexcept override {
        return "MediaPipe Tasks C GestureRecognizer";
    }
    const char* BackendVersion() const noexcept override {
#if defined(GRIM_MEDIAPIPE_VERSION)
        return GRIM_MEDIAPIPE_VERSION;
#else
        return "unversioned";
#endif
    }
    bool IsCompiled() const noexcept override { return true; }
    bool TelemetryDisabled() const noexcept override {
#if defined(GRIM_MEDIAPIPE_TELEMETRY_DISABLED)
        return true;
#else
        return false;
#endif
    }

private:
    MpGestureRecognizerPtr recognizer_ = nullptr;
    int64_t last_timestamp_ms_ = -1;
    int max_hands_ = 1;
};

#else

class PhysicalMediaPipeHandGestureBackend final
    : public IPhysicalHandGestureBackend {
public:
    bool Initialize(const PhysicalHandGestureConfig&, std::string& error) override {
        error = "MediaPipe hand gestures were not compiled into this GRIM build";
        return false;
    }
    bool Process(const PhysicalHandGestureFrame&,
                 std::vector<PhysicalHandObservation>&,
                 double&, std::string& error) override {
        error = "MediaPipe hand gestures were not compiled into this GRIM build";
        return false;
    }
    void Shutdown() noexcept override {}
    const char* BackendName() const noexcept override {
        return "MediaPipe Tasks C GestureRecognizer";
    }
    const char* BackendVersion() const noexcept override { return "not compiled"; }
    bool IsCompiled() const noexcept override { return false; }
    bool TelemetryDisabled() const noexcept override { return true; }
};

#endif

} // namespace

std::unique_ptr<IPhysicalHandGestureBackend>
CreatePhysicalMediaPipeHandGestureBackend() {
    return std::make_unique<PhysicalMediaPipeHandGestureBackend>();
}

}}} // namespace GRIM::Perception::Physical
