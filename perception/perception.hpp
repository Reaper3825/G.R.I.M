#pragma once
#include <string>
#include "perception_queries.hpp"
#include <opencv2/core.hpp> // ✅ For cv::Mat

namespace GRIM {
namespace Perception {

// Initialize perception subsystem
void init();

// Shutdown perception subsystem
void shutdown();

// Perform OCR on a preprocessed cv::Mat image (for enhanced OCR)
std::string readTextFromImage(const cv::Mat& image);

} // namespace Perception
} // namespace GRIM
