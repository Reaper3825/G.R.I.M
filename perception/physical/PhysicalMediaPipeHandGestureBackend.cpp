#include "PhysicalHandGestureBackend.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if defined(GRIM_HAS_MEDIAPIPE_TASKS_C)
#include "mediapipe/tasks/c/core/common.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/gesture_recognizer/gesture_recognizer.h"

#if defined(_WIN32)
#include <Windows.h>
#elif defined(__APPLE__)
#include <dlfcn.h>
#include <mach-o/dyld.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif
#endif

namespace GRIM { namespace Perception { namespace Physical {

namespace {

uint64_t SteadyNowNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

#if defined(GRIM_HAS_MEDIAPIPE_TASKS_C)

class MediaPipeTasksCApi {
public:
    ~MediaPipeTasksCApi() { Unload(); }

    MediaPipeTasksCApi(const MediaPipeTasksCApi&) = delete;
    MediaPipeTasksCApi& operator=(const MediaPipeTasksCApi&) = delete;
    MediaPipeTasksCApi() = default;

    bool Load(std::string& error) {
        Unload();

        std::vector<std::filesystem::path> candidates;
#if defined(GRIM_MEDIAPIPE_RUNTIME_LIBRARY_PATH)
        candidates.emplace_back(GRIM_MEDIAPIPE_RUNTIME_LIBRARY_PATH);
#endif
        const auto executable_dir = ExecutableDirectory();
        if (!executable_dir.empty()) {
            candidates.emplace_back(executable_dir / LibraryFileName());
        }
        candidates.emplace_back(LibraryFileName());

        std::ostringstream failures;
        for (const auto& candidate : candidates) {
            if (candidate.empty()) continue;
            module_ = OpenModule(candidate);
            if (module_) {
                loaded_path_ = candidate.string();
                break;
            }
            if (failures.tellp() > 0) failures << "; ";
            failures << candidate.string();
        }
        if (!module_) {
            error = "Unable to load the local MediaPipe Tasks C runtime. Tried: " +
                    failures.str() + ". " + LastModuleError();
            return false;
        }

        bool complete = true;
        complete &= Resolve(error_free, "MpErrorFree", error);
        complete &= Resolve(image_create_from_uint8_data,
                            "MpImageCreateFromUint8Data", error);
        complete &= Resolve(image_free, "MpImageFree", error);
        complete &= Resolve(gesture_recognizer_create,
                            "MpGestureRecognizerCreate", error);
        complete &= Resolve(gesture_recognizer_recognize_for_video,
                            "MpGestureRecognizerRecognizeForVideo", error);
        complete &= Resolve(gesture_recognizer_close_result,
                            "MpGestureRecognizerCloseResult", error);
        complete &= Resolve(gesture_recognizer_close,
                            "MpGestureRecognizerClose", error);
        if (!complete) {
            Unload();
            return false;
        }
        return true;
    }

    void Unload() noexcept {
        error_free = nullptr;
        image_create_from_uint8_data = nullptr;
        image_free = nullptr;
        gesture_recognizer_create = nullptr;
        gesture_recognizer_recognize_for_video = nullptr;
        gesture_recognizer_close_result = nullptr;
        gesture_recognizer_close = nullptr;
        loaded_path_.clear();
        if (!module_) return;
#if defined(_WIN32)
        FreeLibrary(module_);
#else
        dlclose(module_);
#endif
        module_ = nullptr;
    }

    decltype(&MpErrorFree) error_free = nullptr;
    decltype(&MpImageCreateFromUint8Data) image_create_from_uint8_data = nullptr;
    decltype(&MpImageFree) image_free = nullptr;
    decltype(&MpGestureRecognizerCreate) gesture_recognizer_create = nullptr;
    decltype(&MpGestureRecognizerRecognizeForVideo)
        gesture_recognizer_recognize_for_video = nullptr;
    decltype(&MpGestureRecognizerCloseResult)
        gesture_recognizer_close_result = nullptr;
    decltype(&MpGestureRecognizerClose) gesture_recognizer_close = nullptr;

private:
#if defined(_WIN32)
    using ModuleHandle = HMODULE;
#else
    using ModuleHandle = void*;
#endif

    static const char* LibraryFileName() noexcept {
#if defined(_WIN32)
        return "libmediapipe.dll";
#elif defined(__APPLE__)
        return "libmediapipe.dylib";
#else
        return "libmediapipe.so";
#endif
    }

    static std::filesystem::path ExecutableDirectory() {
#if defined(_WIN32)
        std::wstring path(32768, L'\0');
        const DWORD count = GetModuleFileNameW(
            nullptr, path.data(), static_cast<DWORD>(path.size()));
        if (count == 0 || count >= path.size()) return {};
        path.resize(count);
        return std::filesystem::path(path).parent_path();
#elif defined(__APPLE__)
        uint32_t size = 0;
        (void)_NSGetExecutablePath(nullptr, &size);
        if (size == 0) return {};
        std::vector<char> path(size);
        if (_NSGetExecutablePath(path.data(), &size) != 0) return {};
        return std::filesystem::weakly_canonical(path.data()).parent_path();
#else
        std::vector<char> path(4096, '\0');
        const ssize_t count = readlink("/proc/self/exe", path.data(),
                                       path.size() - 1);
        if (count <= 0) return {};
        path[static_cast<size_t>(count)] = '\0';
        return std::filesystem::path(path.data()).parent_path();
#endif
    }

    static ModuleHandle OpenModule(const std::filesystem::path& path) {
#if defined(_WIN32)
        return LoadLibraryW(path.wstring().c_str());
#else
        return dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    }

    static std::string LastModuleError() {
#if defined(_WIN32)
        const DWORD code = GetLastError();
        char* message = nullptr;
        const DWORD length = FormatMessageA(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                FORMAT_MESSAGE_IGNORE_INSERTS,
            nullptr, code, 0, reinterpret_cast<char*>(&message), 0, nullptr);
        std::string result = length && message
            ? std::string(message, length)
            : "Windows loader error " + std::to_string(code);
        if (message) LocalFree(message);
        return result;
#else
        const char* detail = dlerror();
        return detail ? detail : "dynamic loader returned no detail";
#endif
    }

    void* FindSymbol(const char* name) const noexcept {
#if defined(_WIN32)
        return reinterpret_cast<void*>(GetProcAddress(module_, name));
#else
        return dlsym(module_, name);
#endif
    }

    template <typename Function>
    bool Resolve(Function& function, const char* name, std::string& error) {
        function = reinterpret_cast<Function>(FindSymbol(name));
        if (function) return true;
        if (!error.empty()) error += "; ";
        error += "MediaPipe runtime is missing required symbol ";
        error += name;
        return false;
    }

    ModuleHandle module_ = nullptr;
    std::string loaded_path_;
};

std::string TakeMediaPipeError(MediaPipeTasksCApi& api,
                               char*& error_message) {
    std::string value = error_message ? error_message : "MediaPipe returned no detail";
    if (error_message) {
        api.error_free(error_message);
        error_message = nullptr;
    }
    return value;
}

std::string StatusError(MediaPipeTasksCApi& api, const char* operation,
                        MpStatus status, char*& detail) {
    std::ostringstream out;
    out << operation << " failed (MpStatus " << static_cast<int>(status)
        << "): " << TakeMediaPipeError(api, detail);
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
        if (!api_.Load(error)) return false;

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
        const MpStatus status = api_.gesture_recognizer_create(
            &options, &recognizer_, &detail);
        if (status != kMpOk || !recognizer_) {
            error = StatusError(api_, "MpGestureRecognizerCreate", status, detail);
            recognizer_ = nullptr;
            api_.Unload();
            return false;
        }
        if (detail) api_.error_free(detail);
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
        MpStatus status = api_.image_create_from_uint8_data(
            kMpImageFormatSrgb, frame.width, frame.height, frame.rgb_data,
            frame.byte_count, &image, &detail);
        if (status != kMpOk || !image) {
            error = StatusError(api_, "MpImageCreateFromUint8Data", status, detail);
            return false;
        }
        if (detail) {
            api_.error_free(detail);
            detail = nullptr;
        }

        int64_t timestamp_ms = static_cast<int64_t>(
            frame.source_capture_steady_ns / 1000000ULL);
        if (timestamp_ms <= last_timestamp_ms_) timestamp_ms = last_timestamp_ms_ + 1;

        GestureRecognizerResult result{};
        const auto started = std::chrono::steady_clock::now();
        status = api_.gesture_recognizer_recognize_for_video(
            recognizer_, image, nullptr, timestamp_ms, &result, &detail);
        inference_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        api_.image_free(image);

        if (status != kMpOk) {
            error = StatusError(
                api_, "MpGestureRecognizerRecognizeForVideo", status, detail);
            return false;
        }
        if (detail) api_.error_free(detail);
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
        api_.gesture_recognizer_close_result(&result);
        return true;
    }

    void Shutdown() noexcept override {
        if (recognizer_) {
            char* detail = nullptr;
            (void)api_.gesture_recognizer_close(recognizer_, &detail);
            if (detail) api_.error_free(detail);
            recognizer_ = nullptr;
        }
        api_.Unload();
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
    MediaPipeTasksCApi api_;
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
