#pragma once

#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include "DigitalPerceptionPrimitiveTypes.hpp"

namespace GRIM { namespace Perception { namespace Digital {

// Cross-platform contract. Device-specific implementations may replace the
// local Tesseract provider without changing the primitive loop or consumers.
class DigitalOcrProvider {
public:
    virtual ~DigitalOcrProvider() = default;
    virtual const char* ProviderName() const noexcept = 0;
    virtual DigitalOcrResult Recognize(const cv::Mat& bgr_frame,
                                       unsigned int dpi_x,
                                       unsigned int dpi_y) = 0;
};

std::unique_ptr<DigitalOcrProvider> CreateDefaultDigitalOcrProvider();

}}} // namespace GRIM::Perception::Digital
