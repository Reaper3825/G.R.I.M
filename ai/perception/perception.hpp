#pragma once
#include <string>
#include <memory>
#include <vector>
#include <opencv2/core.hpp>

struct Percept {
    std::string label;
    float confidence;
    cv::Rect boundingBox;
    std::string source;
    double timestamp;
};

class IPerceptionSource {
public:
    virtual ~IPerceptionSource() = default;

    // Human-readable name for debugging (e.g. "BGFX", "Desktop", "Camera0")
    virtual std::string name() const = 0;

    // Capture one frame (synchronous or queued)
    virtual bool captureFrame(cv::Mat& outFrame) = 0;

    // Optional preprocessing (downscale, color convert, etc.)
    virtual void preprocess(cv::Mat& frame) {}

    // Optional: GPU handle access for zero-copy interop (future)
    virtual void* getNativeHandle() { return nullptr; }
};
