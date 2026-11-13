#include "DigitalCaptureSource.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <sstream>
#include <utility>

#include <opencv2/imgproc.hpp>

#ifdef _WIN32
#include "core/grim_platform.h"
#include <dwmapi.h>
#include <psapi.h>
#include <ShellScalingApi.h>
#pragma comment(lib, "Dwmapi.lib")
#pragma comment(lib, "Shcore.lib")

#ifndef WDA_EXCLUDEFROMCAPTURE
#define WDA_EXCLUDEFROMCAPTURE 0x00000011
#endif
#endif

namespace GRIM { namespace Perception { namespace Digital {

namespace {

std::uint64_t SteadyNowNs() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

std::uint64_t WallNowNs() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
}

#ifdef _WIN32

std::string Win32ErrorMessage(DWORD code) {
    if (code == ERROR_SUCCESS) return {};
    LPWSTR raw = nullptr;
    const DWORD chars = FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPWSTR>(&raw), 0, nullptr);
    if (chars == 0 || raw == nullptr) {
        return "Win32 error " + std::to_string(code);
    }

    const int bytes = WideCharToMultiByte(CP_UTF8, 0, raw, static_cast<int>(chars),
                                          nullptr, 0, nullptr, nullptr);
    std::string result;
    if (bytes > 0) {
        result.resize(static_cast<std::size_t>(bytes));
        WideCharToMultiByte(CP_UTF8, 0, raw, static_cast<int>(chars),
                            result.data(), bytes, nullptr, nullptr);
        while (!result.empty() &&
               (result.back() == '\r' || result.back() == '\n' || result.back() == ' ')) {
            result.pop_back();
        }
    }
    LocalFree(raw);
    return result.empty() ? ("Win32 error " + std::to_string(code)) : result;
}

std::string Utf8FromWide(const wchar_t* value) {
    if (value == nullptr || *value == L'\0') return {};
    const int chars = static_cast<int>(wcslen(value));
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, value, chars, nullptr, 0, nullptr, nullptr);
    if (bytes <= 0) return {};
    std::string result(static_cast<std::size_t>(bytes), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value, chars, result.data(), bytes, nullptr, nullptr);
    return result;
}

DigitalRect RectFromWin32(const RECT& rect) {
    return {rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top};
}

DigitalRect IntersectRects(const DigitalRect& a, const DigitalRect& b) {
    const int left = std::max(a.x, b.x);
    const int top = std::max(a.y, b.y);
    const int right = std::min(a.x + a.width, b.x + b.width);
    const int bottom = std::min(a.y + a.height, b.y + b.height);
    return {left, top, std::max(0, right - left), std::max(0, bottom - top)};
}

struct NativeMonitor {
    HMONITOR handle = nullptr;
    DigitalMonitorDescriptor descriptor{};
};

BOOL CALLBACK CollectMonitor(HMONITOR monitor, HDC, LPRECT, LPARAM user_data) {
    auto* monitors = reinterpret_cast<std::vector<NativeMonitor>*>(user_data);
    MONITORINFOEXW info{};
    info.cbSize = sizeof(info);
    if (!GetMonitorInfoW(monitor, &info)) return TRUE;

    NativeMonitor native;
    native.handle = monitor;
    native.descriptor.index = static_cast<int>(monitors->size());
    native.descriptor.id = Utf8FromWide(info.szDevice);
    native.descriptor.desktop_rect = RectFromWin32(info.rcMonitor);
    native.descriptor.work_rect = RectFromWin32(info.rcWork);
    native.descriptor.is_primary = (info.dwFlags & MONITORINFOF_PRIMARY) != 0;

    UINT dpi_x = 96;
    UINT dpi_y = 96;
    if (SUCCEEDED(GetDpiForMonitor(monitor, MDT_EFFECTIVE_DPI, &dpi_x, &dpi_y))) {
        native.descriptor.dpi_x = dpi_x;
        native.descriptor.dpi_y = dpi_y;
        native.descriptor.scale_factor = static_cast<float>(dpi_x) / 96.0f;
    }

    DISPLAY_DEVICEW device{};
    device.cb = sizeof(device);
    if (EnumDisplayDevicesW(info.szDevice, 0, &device, 0)) {
        native.descriptor.friendly_name = Utf8FromWide(device.DeviceString);
    }
    if (native.descriptor.friendly_name.empty()) {
        native.descriptor.friendly_name = native.descriptor.id;
    }

    monitors->push_back(std::move(native));
    return TRUE;
}

std::vector<NativeMonitor> EnumerateNativeMonitors() {
    std::vector<NativeMonitor> monitors;
    if (!EnumDisplayMonitors(nullptr, nullptr, CollectMonitor,
                             reinterpret_cast<LPARAM>(&monitors))) {
        return {};
    }
    std::sort(monitors.begin(), monitors.end(), [](const NativeMonitor& a, const NativeMonitor& b) {
        const auto& ar = a.descriptor.desktop_rect;
        const auto& br = b.descriptor.desktop_rect;
        if (ar.x != br.x) return ar.x < br.x;
        if (ar.y != br.y) return ar.y < br.y;
        return a.descriptor.id < b.descriptor.id;
    });
    for (std::size_t i = 0; i < monitors.size(); ++i) {
        monitors[i].descriptor.index = static_cast<int>(i);
    }
    return monitors;
}

DigitalRect VirtualDesktopBounds(const std::vector<NativeMonitor>& monitors) {
    if (monitors.empty()) return {};
    int left = monitors.front().descriptor.desktop_rect.x;
    int top = monitors.front().descriptor.desktop_rect.y;
    int right = left + monitors.front().descriptor.desktop_rect.width;
    int bottom = top + monitors.front().descriptor.desktop_rect.height;
    for (const auto& monitor : monitors) {
        const auto& r = monitor.descriptor.desktop_rect;
        left = std::min(left, r.x);
        top = std::min(top, r.y);
        right = std::max(right, r.x + r.width);
        bottom = std::max(bottom, r.y + r.height);
    }
    return {left, top, right - left, bottom - top};
}

std::string ActiveWindowTitle(HWND window) {
    const int length = GetWindowTextLengthW(window);
    if (length <= 0) return {};
    std::wstring title(static_cast<std::size_t>(length) + 1, L'\0');
    const int copied = GetWindowTextW(window, title.data(), length + 1);
    if (copied <= 0) return {};
    title.resize(static_cast<std::size_t>(copied));
    return Utf8FromWide(title.c_str());
}

std::string ActiveProcessName(HWND window) {
    DWORD process_id = 0;
    GetWindowThreadProcessId(window, &process_id);
    if (process_id == 0) return {};

    HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, process_id);
    if (!process) return {};

    std::wstring path(32768, L'\0');
    DWORD length = static_cast<DWORD>(path.size());
    std::string result;
    if (QueryFullProcessImageNameW(process, 0, path.data(), &length) && length > 0) {
        path.resize(length);
        result = Utf8FromWide(std::filesystem::path(path).filename().c_str());
    }
    CloseHandle(process);
    return result;
}

DigitalRect WindowBounds(HWND window) {
    RECT rect{};
    if (SUCCEEDED(DwmGetWindowAttribute(window, DWMWA_EXTENDED_FRAME_BOUNDS,
                                        &rect, sizeof(rect)))) {
        return RectFromWin32(rect);
    }
    if (GetWindowRect(window, &rect)) return RectFromWin32(rect);
    return {};
}

class Win32DigitalCaptureSource final : public DigitalCaptureSource {
public:
    std::string BackendName() const override { return "win32-gdi-dib"; }

    std::vector<DigitalMonitorDescriptor> EnumerateMonitors() override {
        const auto native = EnumerateNativeMonitors();
        std::vector<DigitalMonitorDescriptor> result;
        result.reserve(native.size());
        for (const auto& monitor : native) result.push_back(monitor.descriptor);
        return result;
    }

    DigitalCaptureResult Capture(const DigitalCaptureRequest& request) override {
        const auto started = std::chrono::steady_clock::now();
        DigitalCaptureResult result;
        auto& meta = result.metadata;
        meta.backend = BackendName();
        meta.source_device_id = "local";
        meta.source_platform = "windows";
        meta.source_transport = "native";
        meta.mode = request.mode;
        meta.capture_steady_ns = SteadyNowNs();
        meta.capture_wall_ns = WallNowNs();

        const auto monitors = EnumerateNativeMonitors();
        if (monitors.empty()) {
            meta.status = DigitalCaptureStatus::NoDisplays;
            meta.error = "EnumDisplayMonitors returned no active displays";
            StampDuration(started, meta);
            return result;
        }

        const HWND active_window = GetForegroundWindow();
        if (active_window) {
            meta.active_window_title = ActiveWindowTitle(active_window);
            meta.active_process_name = ActiveProcessName(active_window);
            meta.active_window_rect = WindowBounds(active_window);
        }

        DigitalRect source_rect{};
        const NativeMonitor* selected_monitor = nullptr;
        switch (request.mode) {
            case DigitalCaptureMode::ActiveMonitor: {
                HMONITOR handle = active_window
                    ? MonitorFromWindow(active_window, MONITOR_DEFAULTTOPRIMARY)
                    : MonitorFromPoint(POINT{0, 0}, MONITOR_DEFAULTTOPRIMARY);
                const auto it = std::find_if(monitors.begin(), monitors.end(),
                    [handle](const NativeMonitor& monitor) { return monitor.handle == handle; });
                selected_monitor = it != monitors.end() ? &*it : &monitors.front();
                source_rect = selected_monitor->descriptor.desktop_rect;
                break;
            }
            case DigitalCaptureMode::Monitor:
                if (request.monitor_index < 0 ||
                    request.monitor_index >= static_cast<int>(monitors.size())) {
                    meta.status = DigitalCaptureStatus::InvalidRequest;
                    meta.error = "monitor index " + std::to_string(request.monitor_index) +
                                 " is outside [0, " + std::to_string(monitors.size()) + ")";
                    StampDuration(started, meta);
                    return result;
                }
                selected_monitor = &monitors[static_cast<std::size_t>(request.monitor_index)];
                source_rect = selected_monitor->descriptor.desktop_rect;
                break;
            case DigitalCaptureMode::ActiveWindow:
                if (!active_window || !IsWindowVisible(active_window)) {
                    meta.status = DigitalCaptureStatus::SourceUnavailable;
                    meta.error = "there is no visible foreground window";
                    StampDuration(started, meta);
                    return result;
                }
                source_rect = IntersectRects(meta.active_window_rect,
                                             VirtualDesktopBounds(monitors));
                if (HMONITOR handle = MonitorFromWindow(active_window, MONITOR_DEFAULTTONEAREST)) {
                    const auto it = std::find_if(monitors.begin(), monitors.end(),
                        [handle](const NativeMonitor& monitor) { return monitor.handle == handle; });
                    if (it != monitors.end()) selected_monitor = &*it;
                }
                break;
            case DigitalCaptureMode::VirtualDesktop:
                source_rect = VirtualDesktopBounds(monitors);
                break;
        }

        if (!source_rect.IsValid()) {
            meta.status = DigitalCaptureStatus::SourceUnavailable;
            meta.error = "capture source has an empty desktop rectangle";
            StampDuration(started, meta);
            return result;
        }

        meta.source_rect = source_rect;
        if (selected_monitor) {
            meta.monitor_index = selected_monitor->descriptor.index;
            meta.monitor_id = selected_monitor->descriptor.id;
            meta.dpi_x = selected_monitor->descriptor.dpi_x;
            meta.dpi_y = selected_monitor->descriptor.dpi_y;
            meta.scale_factor = selected_monitor->descriptor.scale_factor;
        } else {
            meta.monitor_index = -1;
            meta.monitor_id = "virtual-desktop";
        }

        DWORD error_code = ERROR_SUCCESS;
        result.image = CaptureDesktopRect(source_rect, request.include_layered_windows,
                                          error_code);
        if (result.image.empty()) {
            meta.status = error_code == ERROR_ACCESS_DENIED
                ? DigitalCaptureStatus::PermissionDenied
                : DigitalCaptureStatus::CaptureFailed;
            meta.error = "desktop capture failed: " + Win32ErrorMessage(error_code);
        } else {
            meta.status = DigitalCaptureStatus::Ok;
        }
        StampDuration(started, meta);
        return result;
    }

private:
    static void StampDuration(std::chrono::steady_clock::time_point started,
                              DigitalCaptureMetadata& meta) {
        meta.capture_duration_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
    }

    static cv::Mat CaptureDesktopRect(const DigitalRect& rect,
                                      bool include_layered_windows,
                                      DWORD& error_code) {
        error_code = ERROR_SUCCESS;
        HDC screen = GetDC(nullptr);
        if (!screen) {
            error_code = GetLastError();
            return {};
        }
        HDC memory = CreateCompatibleDC(screen);
        if (!memory) {
            error_code = GetLastError();
            ReleaseDC(nullptr, screen);
            return {};
        }

        BITMAPINFO bitmap_info{};
        bitmap_info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bitmap_info.bmiHeader.biWidth = rect.width;
        bitmap_info.bmiHeader.biHeight = -rect.height;
        bitmap_info.bmiHeader.biPlanes = 1;
        bitmap_info.bmiHeader.biBitCount = 32;
        bitmap_info.bmiHeader.biCompression = BI_RGB;

        void* pixels = nullptr;
        HBITMAP bitmap = CreateDIBSection(screen, &bitmap_info, DIB_RGB_COLORS,
                                          &pixels, nullptr, 0);
        if (!bitmap || !pixels) {
            error_code = GetLastError();
            if (bitmap) DeleteObject(bitmap);
            DeleteDC(memory);
            ReleaseDC(nullptr, screen);
            return {};
        }

        HGDIOBJ previous = SelectObject(memory, bitmap);
        if (!previous || previous == HGDI_ERROR) {
            error_code = GetLastError();
            DeleteObject(bitmap);
            DeleteDC(memory);
            ReleaseDC(nullptr, screen);
            return {};
        }

        const DWORD raster_operation = SRCCOPY |
            (include_layered_windows ? CAPTUREBLT : 0);
        const BOOL copied = BitBlt(memory, 0, 0, rect.width, rect.height,
                                   screen, rect.x, rect.y, raster_operation);
        if (!copied) {
            error_code = GetLastError();
            SelectObject(memory, previous);
            DeleteObject(bitmap);
            DeleteDC(memory);
            ReleaseDC(nullptr, screen);
            return {};
        }
        GdiFlush();

        cv::Mat bgr;
        try {
            cv::Mat bgra(rect.height, rect.width, CV_8UC4, pixels);
            cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
        } catch (const cv::Exception&) {
            error_code = ERROR_INVALID_DATA;
            bgr.release();
        } catch (const std::exception&) {
            error_code = ERROR_OUTOFMEMORY;
            bgr.release();
        }

        SelectObject(memory, previous);
        DeleteObject(bitmap);
        DeleteDC(memory);
        ReleaseDC(nullptr, screen);
        return bgr;
    }
};

#else

class UnsupportedDigitalCaptureSource final : public DigitalCaptureSource {
public:
    std::string BackendName() const override { return "unsupported"; }
    std::vector<DigitalMonitorDescriptor> EnumerateMonitors() override { return {}; }
    DigitalCaptureResult Capture(const DigitalCaptureRequest& request) override {
        DigitalCaptureResult result;
        result.metadata.mode = request.mode;
        result.metadata.backend = BackendName();
        result.metadata.source_device_id = "local";
#if defined(__APPLE__)
        result.metadata.source_platform = "macos";
#elif defined(__linux__)
        result.metadata.source_platform = "linux";
#else
        result.metadata.source_platform = "unknown";
#endif
        result.metadata.source_transport = "native";
        result.metadata.status = DigitalCaptureStatus::Unsupported;
        result.metadata.error = "digital capture is not implemented on this platform";
        result.metadata.capture_steady_ns = SteadyNowNs();
        result.metadata.capture_wall_ns = WallNowNs();
        return result;
    }
};

#endif

} // namespace

std::unique_ptr<DigitalCaptureSource> CreatePlatformDigitalCaptureSource() {
#ifdef _WIN32
    return std::make_unique<Win32DigitalCaptureSource>();
#else
    return std::make_unique<UnsupportedDigitalCaptureSource>();
#endif
}

bool EnsureDigitalCaptureDpiAwareness(std::string* error) {
#ifdef _WIN32
    if (SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) {
        if (error) error->clear();
        return true;
    }
    const DWORD code = GetLastError();
    // ERROR_ACCESS_DENIED means a manifest or earlier startup code already
    // selected the process awareness. Coordinates are stable; do not downgrade.
    if (code == ERROR_ACCESS_DENIED) {
        if (error) error->clear();
        return true;
    }
    if (error) *error = Win32ErrorMessage(code);
    return false;
#else
    if (error) error->clear();
    return true;
#endif
}

bool SetDigitalCaptureExcludedWindow(void* native_window, std::string* error) {
#ifdef _WIN32
    if (!native_window || !IsWindow(static_cast<HWND>(native_window))) {
        if (error) *error = "native window is null or invalid";
        return false;
    }
    if (SetWindowDisplayAffinity(static_cast<HWND>(native_window),
                                 WDA_EXCLUDEFROMCAPTURE)) {
        if (error) error->clear();
        return true;
    }
    const DWORD code = GetLastError();
    if (error) *error = Win32ErrorMessage(code);
    return false;
#else
    (void)native_window;
    if (error) error->clear();
    return true;
#endif
}

}}} // namespace GRIM::Perception::Digital
