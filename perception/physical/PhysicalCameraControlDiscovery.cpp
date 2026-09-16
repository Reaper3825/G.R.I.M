#include "PhysicalCameraControlDiscovery.hpp"

#include <string>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <dshow.h>
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "oleaut32.lib")
#pragma comment(lib, "strmiids.lib")
#elif defined(__linux__)
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace GRIM { namespace Perception { namespace Physical {

#if defined(_WIN32)
namespace {

template <typename T>
void ReleaseCom(T*& value) {
    if (value) {
        value->Release();
        value = nullptr;
    }
}

PhysicalCameraControlRange CameraControlRange(
    IAMCameraControl* control,
    long property) {
    PhysicalCameraControlRange result;
    if (!control) return result;
    long minimum = 0, maximum = 0, step = 0, default_value = 0, flags = 0;
    if (SUCCEEDED(control->GetRange(
            property, &minimum, &maximum, &step, &default_value, &flags))) {
        result.available = minimum < maximum;
        result.minimum = static_cast<double>(minimum);
        result.maximum = static_cast<double>(maximum);
        result.step = static_cast<double>(step);
        result.default_value = static_cast<double>(default_value);
    }
    return result;
}

PhysicalCameraControlRange VideoProcAmpRange(
    IAMVideoProcAmp* control,
    long property) {
    PhysicalCameraControlRange result;
    if (!control) return result;
    long minimum = 0, maximum = 0, step = 0, default_value = 0, flags = 0;
    if (SUCCEEDED(control->GetRange(
            property, &minimum, &maximum, &step, &default_value, &flags))) {
        result.available = minimum < maximum;
        result.minimum = static_cast<double>(minimum);
        result.maximum = static_cast<double>(maximum);
        result.step = static_cast<double>(step);
        result.default_value = static_cast<double>(default_value);
    }
    return result;
}

} // namespace
#endif

PhysicalCameraControlProfile DiscoverPhysicalCameraControlProfile(
    int device_index,
    int requested_opencv_backend) {
    PhysicalCameraControlProfile profile;
    if (device_index < 0) {
        profile.reason = "device index is negative";
        return profile;
    }

#if defined(_WIN32)
    (void)requested_opencv_backend;
    const HRESULT init_result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool uninitialize = SUCCEEDED(init_result);
    if (FAILED(init_result) && init_result != RPC_E_CHANGED_MODE) {
        profile.reason = "COM initialization failed";
        return profile;
    }

    ICreateDevEnum* device_enum = nullptr;
    IEnumMoniker* moniker_enum = nullptr;
    IMoniker* selected_moniker = nullptr;
    IBaseFilter* filter = nullptr;
    IAMCameraControl* camera_control = nullptr;
    IAMVideoProcAmp* video_control = nullptr;

    HRESULT hr = CoCreateInstance(
        CLSID_SystemDeviceEnum, nullptr, CLSCTX_INPROC_SERVER,
        IID_ICreateDevEnum, reinterpret_cast<void**>(&device_enum));
    if (SUCCEEDED(hr)) {
        hr = device_enum->CreateClassEnumerator(
            CLSID_VideoInputDeviceCategory, &moniker_enum, 0);
    }
    if (hr == S_FALSE) hr = E_FAIL;
    if (SUCCEEDED(hr)) {
        ULONG fetched = 0;
        int index = 0;
        IMoniker* moniker = nullptr;
        while (moniker_enum->Next(1, &moniker, &fetched) == S_OK) {
            if (index == device_index) {
                selected_moniker = moniker;
                break;
            }
            moniker->Release();
            ++index;
        }
        if (!selected_moniker) hr = E_FAIL;
    }
    if (SUCCEEDED(hr)) {
        hr = selected_moniker->BindToObject(
            nullptr, nullptr, IID_IBaseFilter,
            reinterpret_cast<void**>(&filter));
    }
    if (SUCCEEDED(hr)) {
        filter->QueryInterface(
            IID_IAMCameraControl,
            reinterpret_cast<void**>(&camera_control));
        filter->QueryInterface(
            IID_IAMVideoProcAmp,
            reinterpret_cast<void**>(&video_control));
        profile.exposure = CameraControlRange(
            camera_control, CameraControl_Exposure);
        profile.gain = VideoProcAmpRange(video_control, VideoProcAmp_Gain);
        profile.manual_auto_exposure_value_available = true;
        profile.manual_auto_exposure_value = 0.25;
        profile.provider = "DirectShow IAMCameraControl/IAMVideoProcAmp";
        profile.reason = profile.SupportsMotionExposure()
            ? "exposure and gain ranges discovered"
            : "camera does not expose both exposure and gain ranges";
    } else {
        profile.reason =
            "DirectShow device enumeration could not match this camera index";
    }

    ReleaseCom(video_control);
    ReleaseCom(camera_control);
    ReleaseCom(filter);
    ReleaseCom(selected_moniker);
    ReleaseCom(moniker_enum);
    ReleaseCom(device_enum);
    if (uninitialize) CoUninitialize();
    return profile;
#elif defined(__linux__)
    (void)requested_opencv_backend;
    const std::string path = "/dev/video" + std::to_string(device_index);
    const int fd = ::open(path.c_str(), O_RDWR | O_NONBLOCK);
    if (fd < 0) {
        profile.reason = "cannot open " + path + ": " + std::strerror(errno);
        return profile;
    }
    auto query = [&](uint32_t id) {
        PhysicalCameraControlRange range;
        v4l2_queryctrl control{};
        control.id = id;
        if (::ioctl(fd, VIDIOC_QUERYCTRL, &control) == 0
            && (control.flags & V4L2_CTRL_FLAG_DISABLED) == 0
            && control.minimum < control.maximum) {
            range.available = true;
            range.minimum = control.minimum;
            range.maximum = control.maximum;
            range.step = control.step;
            range.default_value = control.default_value;
        }
        return range;
    };
    profile.exposure = query(V4L2_CID_EXPOSURE_ABSOLUTE);
    profile.gain = query(V4L2_CID_GAIN);
    ::close(fd);
    profile.manual_auto_exposure_value_available = true;
    profile.manual_auto_exposure_value = 1.0; // V4L2_EXPOSURE_MANUAL
    profile.provider = "V4L2 VIDIOC_QUERYCTRL";
    profile.reason = profile.SupportsMotionExposure()
        ? "exposure and gain ranges discovered"
        : "camera does not expose both exposure and gain ranges";
    return profile;
#else
    (void)requested_opencv_backend;
    profile.provider = "unavailable";
    profile.reason =
        "native exposure/gain range discovery is not implemented on this platform";
    return profile;
#endif
}

}}} // namespace GRIM::Perception::Physical
