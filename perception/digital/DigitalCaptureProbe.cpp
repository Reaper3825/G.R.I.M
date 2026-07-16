#include "DigitalCaptureProbe.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "DigitalCaptureSource.hpp"
#include "DigitalContextProjector.hpp"
#include "DigitalFrameBus.hpp"
#include "DigitalPerceptionPrimitiveBus.hpp"
#include "MMO/Core/SessionContextManager.hpp"

#ifdef _WIN32
#include "core/grim_platform.h"
#ifndef WDA_EXCLUDEFROMCAPTURE
#define WDA_EXCLUDEFROMCAPTURE 0x00000011
#endif
#endif

namespace GRIM { namespace Perception { namespace Digital {

namespace {

struct ProbeOptions {
    int duration_seconds = 5;
    int interval_ms = 1000;
    DigitalCaptureRequest request{};
    std::filesystem::path output = "data/digital_capture_probe.png";
};

bool ParseInt(const std::string& value, int& out) {
    try {
        std::size_t used = 0;
        const int parsed = std::stoi(value, &used);
        if (used != value.size()) return false;
        out = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool ParseMode(const std::string& value, DigitalCaptureMode& mode) {
    if (value == "active-monitor") mode = DigitalCaptureMode::ActiveMonitor;
    else if (value == "monitor") mode = DigitalCaptureMode::Monitor;
    else if (value == "active-window") mode = DigitalCaptureMode::ActiveWindow;
    else if (value == "virtual-desktop") mode = DigitalCaptureMode::VirtualDesktop;
    else return false;
    return true;
}

bool ParseOptions(int argc, char* argv[], ProbeOptions& options, std::string& error) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                error = std::string(name) + " requires a value";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--digital-capture-probe") continue;
        if (arg == "--duration") {
            const char* value = require_value("--duration");
            if (!value || !ParseInt(value, options.duration_seconds) ||
                options.duration_seconds < 1 || options.duration_seconds > 600) {
                if (error.empty()) error = "--duration must be between 1 and 600 seconds";
                return false;
            }
        } else if (arg == "--interval-ms") {
            const char* value = require_value("--interval-ms");
            if (!value || !ParseInt(value, options.interval_ms) ||
                options.interval_ms < 50 || options.interval_ms > 60000) {
                if (error.empty()) error = "--interval-ms must be between 50 and 60000";
                return false;
            }
        } else if (arg == "--mode") {
            const char* value = require_value("--mode");
            if (!value || !ParseMode(value, options.request.mode)) {
                if (error.empty()) error = "invalid --mode";
                return false;
            }
        } else if (arg == "--monitor-index") {
            const char* value = require_value("--monitor-index");
            if (!value || !ParseInt(value, options.request.monitor_index)) {
                if (error.empty()) error = "invalid --monitor-index";
                return false;
            }
            options.request.mode = DigitalCaptureMode::Monitor;
        } else if (arg == "--output") {
            const char* value = require_value("--output");
            if (!value) return false;
            options.output = value;
        } else {
            error = "unknown probe argument: " + arg;
            return false;
        }
    }
    return true;
}

std::uint64_t Fingerprint(const cv::Mat& image) {
    if (image.empty()) return 0;
    const std::uint8_t* bytes = image.ptr<std::uint8_t>();
    const std::size_t total = image.total() * image.elemSize();
    const std::size_t stride = std::max<std::size_t>(1, total / 65536);
    std::uint64_t hash = 1469598103934665603ull;
    for (std::size_t i = 0; i < total; i += stride) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    hash ^= static_cast<std::uint64_t>(image.cols);
    hash *= 1099511628211ull;
    hash ^= static_cast<std::uint64_t>(image.rows);
    return hash;
}

double ChangeScore(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || b.empty() || a.size() != b.size() || a.type() != b.type()) return 1.0;
    cv::Mat small_a;
    cv::Mat small_b;
    cv::resize(a, small_a, cv::Size(160, 90), 0.0, 0.0, cv::INTER_AREA);
    cv::resize(b, small_b, cv::Size(160, 90), 0.0, 0.0, cv::INTER_AREA);
    cv::Mat diff;
    cv::absdiff(small_a, small_b, diff);
    const cv::Scalar mean = cv::mean(diff);
    return (mean[0] + mean[1] + mean[2]) / (3.0 * 255.0);
}

bool HasPixelVariation(const cv::Mat& image) {
    if (image.empty()) return false;
    cv::Scalar mean;
    cv::Scalar deviation;
    cv::meanStdDev(image, mean, deviation);
    return deviation[0] > 0.5 || deviation[1] > 0.5 || deviation[2] > 0.5;
}

bool RunContractTests(std::string& error) {
    cv::Mat dark(8, 8, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat light(8, 8, CV_8UC3, cv::Scalar(255, 255, 255));
    if (Fingerprint(dark) == 0 || Fingerprint(dark) == Fingerprint(light)) {
        error = "frame fingerprint contract failed";
        return false;
    }
    if (ChangeScore(dark, dark) != 0.0 || ChangeScore(dark, light) < 0.99) {
        error = "change-score contract failed";
        return false;
    }

    DigitalFrameBus::Instance().Reset();
    DigitalCaptureResult success;
    success.image = light.clone();
    success.metadata.status = DigitalCaptureStatus::Ok;
    success.metadata.backend = "contract-test";
    success.metadata.monitor_id = "synthetic-monitor";
    success.metadata.active_window_title = "Synthetic Window";
    success.metadata.active_process_name = "synthetic.exe";
    success.metadata.capture_wall_ns = 1;
    DigitalFrameBus::Instance().Publish(success, 1);
    success.image.setTo(cv::Scalar(0, 0, 0));

    DigitalFrameBus::FrameView view;
    std::uint64_t cursor = 0;
    if (!DigitalFrameBus::Instance().PullLatest(view, cursor) ||
        view.frame_counter != 1 || cv::mean(view.image)[0] < 254.0) {
        error = "frame bus immutability contract failed";
        return false;
    }
    if (DigitalFrameBus::Instance().PullLatest(view, cursor)) {
        error = "frame bus cursor contract failed";
        return false;
    }

    DigitalPerceptionPrimitiveBus::Instance().Reset();
    DigitalPerceptionPrimitiveSnapshot primitives;
    primitives.source_frame_counter = 1;
    primitives.ocr.status = DigitalPrimitiveStatus::Ok;
    primitives.ocr.provider = "contract-ocr";
    primitives.ocr.full_text = "Synthetic Text";
    primitives.ocr.regions.push_back({{1, 2, 3, 4}, "Synthetic Text", 0.99f});
    primitives.automation.status = DigitalPrimitiveStatus::Ok;
    primitives.automation.provider = "contract-automation";
    primitives.automation.target_matches_capture = true;
    DigitalUiElement synthetic_button;
    synthetic_button.desktop_rect = {10, 20, 30, 40};
    synthetic_button.name = "Continue";
    synthetic_button.role = "button";
    primitives.automation.elements.push_back(std::move(synthetic_button));
    DigitalPerceptionPrimitiveBus::Instance().Publish(primitives);

    DigitalPerceptionPrimitiveBus::SnapshotView primitive_view;
    std::uint64_t primitive_cursor = 0;
    if (!DigitalPerceptionPrimitiveBus::Instance().PullLatest(
            primitive_view, primitive_cursor) || !primitive_view.snapshot ||
        primitive_view.snapshot->ocr.full_text != "Synthetic Text" ||
        primitive_cursor != 1) {
        error = "digital primitive bus contract failed";
        return false;
    }
    if (DigitalPerceptionPrimitiveBus::Instance().PullLatest(
            primitive_view, primitive_cursor)) {
        error = "digital primitive bus cursor contract failed";
        return false;
    }

    ShutdownDigitalContextProjector();
    TickDigitalContextProjector();
    const auto snapshot = ::GRIM::MMO::SessionContextManager::instance().snapshot("default");
    if (snapshot.visual_context.digital.active_window != "Synthetic Window" ||
        snapshot.visual_context.digital.capture_status != "ok" ||
        snapshot.visual_context.digital.provenance_frame_counter != 1 ||
        snapshot.visual_context.digital.ocr_text != "Synthetic Text" ||
        snapshot.visual_context.digital.ui_elements.empty() ||
        snapshot.visual_context.digital.preferred_grounding_source !=
            "contract-automation" ||
        snapshot.visual_context.digital.primitive_provenance_frame_counter != 1) {
        error = "digital reasoning-context projection contract failed";
        return false;
    }
    DigitalFrameBus::Instance().Reset();
    DigitalPerceptionPrimitiveBus::Instance().Reset();
    ShutdownDigitalContextProjector();
    return true;
}

#ifdef _WIN32
DWORD GdiResourceCount() {
    return GetGuiResources(GetCurrentProcess(), GR_GDIOBJECTS);
}

bool VerifyWindowExclusionApi(std::string& error) {
    static const wchar_t* class_name = L"GRIMDigitalCaptureProbeWindow";
    WNDCLASSW wc{};
    wc.lpfnWndProc = DefWindowProcW;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = class_name;
    const ATOM atom = RegisterClassW(&wc);
    if (!atom && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
        error = "failed to register exclusion probe window";
        return false;
    }
    HWND window = CreateWindowExW(0, class_name, L"GRIM capture exclusion probe",
                                  WS_POPUP, 0, 0, 32, 32, nullptr, nullptr,
                                  wc.hInstance, nullptr);
    if (!window) {
        error = "failed to create exclusion probe window";
        return false;
    }
    const bool set = SetDigitalCaptureExcludedWindow(window, &error);
    DWORD affinity = 0;
    const bool read = GetWindowDisplayAffinity(window, &affinity) != FALSE;
    DestroyWindow(window);
    if (!set || !read || affinity != WDA_EXCLUDEFROMCAPTURE) {
        if (error.empty()) error = "excluded-window affinity did not round-trip";
        return false;
    }
    return true;
}
#else
unsigned long GdiResourceCount() { return 0; }
bool VerifyWindowExclusionApi(std::string&) { return true; }
#endif

std::filesystem::path WithSuffix(const std::filesystem::path& path,
                                 const std::string& suffix) {
    return path.parent_path() /
        (path.stem().string() + suffix + path.extension().string());
}

int RunProbe(const ProbeOptions& options) {
    std::string contract_error;
    if (!RunContractTests(contract_error)) {
        std::cerr << "[FAIL] " << contract_error << '\n';
        return 2;
    }
    std::cout << "[PASS] deterministic frame, bus, and reasoning projection contracts\n";

    if (!VerifyWindowExclusionApi(contract_error)) {
        std::cerr << "[FAIL] capture exclusion: " << contract_error << '\n';
        return 3;
    }
    std::cout << "[PASS] GRIM-owned window exclusion affinity\n";

    auto source = CreatePlatformDigitalCaptureSource();
    if (!source) {
        std::cerr << "[FAIL] capture source factory returned null\n";
        return 4;
    }

    const auto monitors = source->EnumerateMonitors();
    if (monitors.empty()) {
        std::cerr << "[FAIL] no active monitors were enumerated\n";
        return 5;
    }
    bool has_primary = false;
    bool has_negative_origin = false;
    bool has_scaled_monitor = false;
    for (std::size_t i = 0; i < monitors.size(); ++i) {
        const auto& monitor = monitors[i];
        if (monitor.index != static_cast<int>(i) || !monitor.desktop_rect.IsValid() ||
            monitor.id.empty() || monitor.dpi_x == 0 || monitor.dpi_y == 0 ||
            std::fabs(monitor.scale_factor - static_cast<float>(monitor.dpi_x) / 96.0f) > 0.02f) {
            std::cerr << "[FAIL] invalid monitor descriptor at index " << i << '\n';
            return 6;
        }
        has_primary = has_primary || monitor.is_primary;
        has_negative_origin = has_negative_origin || monitor.desktop_rect.x < 0 ||
                              monitor.desktop_rect.y < 0;
        has_scaled_monitor = has_scaled_monitor || monitor.dpi_x != 96 || monitor.dpi_y != 96;
        std::cout << "monitor=" << monitor.index << " id=" << monitor.id
                  << " rect=" << monitor.desktop_rect.x << ',' << monitor.desktop_rect.y
                  << ' ' << monitor.desktop_rect.width << 'x' << monitor.desktop_rect.height
                  << " dpi=" << monitor.dpi_x << 'x' << monitor.dpi_y
                  << " scale=" << monitor.scale_factor
                  << (monitor.is_primary ? " primary" : "") << '\n';
    }
    if (!has_primary) {
        std::cerr << "[FAIL] monitor enumeration did not identify a primary monitor\n";
        return 7;
    }
    std::cout << "[PASS] monitor enumeration; negative-origin="
              << (has_negative_origin ? "exercised" : "not-present")
              << " dpi-scaling=" << (has_scaled_monitor ? "exercised" : "96-dpi-only") << '\n';

    DigitalCaptureRequest invalid;
    invalid.mode = DigitalCaptureMode::Monitor;
    invalid.monitor_index = static_cast<int>(monitors.size());
    if (source->Capture(invalid).metadata.status != DigitalCaptureStatus::InvalidRequest) {
        std::cerr << "[FAIL] invalid monitor request did not fail explicitly\n";
        return 8;
    }

    const int iterations = std::max(1, (options.duration_seconds * 1000 +
                                        options.interval_ms - 1) / options.interval_ms);
    const auto first_path = WithSuffix(options.output, "_first");
    const auto last_path = WithSuffix(options.output, "_last");
    if (!options.output.parent_path().empty()) {
        std::filesystem::create_directories(options.output.parent_path());
    }

    const auto gdi_before = GdiResourceCount();
    cv::Mat previous;
    std::uint64_t previous_fingerprint = 0;
    int changed_frames = 0;
    double max_change = 0.0;
    double total_capture_ms = 0.0;

    for (int i = 0; i < iterations; ++i) {
        DigitalCaptureResult result = source->Capture(options.request);
        if (!result.Succeeded()) {
            std::cerr << "[FAIL] live capture " << i << " status="
                      << ToString(result.metadata.status) << " error="
                      << result.metadata.error << '\n';
            return 9;
        }
        if (result.image.type() != CV_8UC3 ||
            result.image.cols != result.metadata.source_rect.width ||
            result.image.rows != result.metadata.source_rect.height ||
            !HasPixelVariation(result.image)) {
            std::cerr << "[FAIL] live frame contract at capture " << i << '\n';
            return 10;
        }

        const std::uint64_t fingerprint = Fingerprint(result.image);
        if (i == 0) {
            if (!cv::imwrite(first_path.string(), result.image)) {
                std::cerr << "[FAIL] could not write " << first_path << '\n';
                return 11;
            }
        } else {
            const double change = ChangeScore(previous, result.image);
            max_change = std::max(max_change, change);
            if (fingerprint != previous_fingerprint && change > 0.0001) ++changed_frames;
        }
        total_capture_ms += result.metadata.capture_duration_ms;
        previous = result.image;
        previous_fingerprint = fingerprint;

        std::cout << "capture=" << (i + 1) << '/' << iterations
                  << " mode=" << ToString(result.metadata.mode)
                  << " size=" << result.image.cols << 'x' << result.image.rows
                  << " monitor=" << result.metadata.monitor_id
                  << " ms=" << result.metadata.capture_duration_ms << '\n';

        if (i + 1 < iterations) {
            std::this_thread::sleep_for(std::chrono::milliseconds(options.interval_ms));
        }
    }

    if (!cv::imwrite(last_path.string(), previous)) {
        std::cerr << "[FAIL] could not write " << last_path << '\n';
        return 12;
    }
    const auto gdi_after = GdiResourceCount();
    if (gdi_after > gdi_before + 1) {
        std::cerr << "[FAIL] GDI resource count grew from " << gdi_before
                  << " to " << gdi_after << '\n';
        return 13;
    }

    std::cout << "[PASS] live stability captures=" << iterations
              << " failures=0 changed=" << changed_frames
              << " max-change=" << max_change
              << " avg-ms=" << (total_capture_ms / iterations)
              << " gdi-before=" << gdi_before << " gdi-after=" << gdi_after << '\n';
    std::cout << "first-frame=" << first_path.string() << '\n';
    std::cout << "last-frame=" << last_path.string() << '\n';
    return 0;
}

} // namespace

bool TryRunDigitalCaptureProbe(int argc, char* argv[], int& exit_code) {
    bool requested = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--digital-capture-probe") {
            requested = true;
            break;
        }
    }
    if (!requested) return false;

    ProbeOptions options;
    std::string error;
    if (!ParseOptions(argc, argv, options, error)) {
        std::cerr << "digital capture probe: " << error << '\n';
        std::cerr << "usage: GRIM --digital-capture-probe [--duration 60] "
                     "[--interval-ms 1000] [--mode active-monitor|monitor|active-window|virtual-desktop] "
                     "[--monitor-index N] [--output path.png]\n";
        exit_code = 64;
        return true;
    }

    exit_code = RunProbe(options);
    return true;
}

}}} // namespace GRIM::Perception::Digital
