#pragma once
#include <string>
#include <opencv2/core.hpp> // ✅ For cv::Mat

namespace GRIM {
namespace Perception {

// Initialize perception subsystem
void init();

// Shutdown perception subsystem
void shutdown();

// Check if perception is available
bool isAvailable();

// Capture and analyze current screen
std::string analyzeScreen();

// Perform OCR on current screen or region
std::string readText(int x = -1, int y = -1, int width = -1, int height = -1);

// Perform OCR on a preprocessed cv::Mat image (for enhanced OCR)
std::string readTextFromImage(const cv::Mat& image);

// Detect objects in current view
std::string detectObjects();

} // namespace Perception
} // namespace GRIM
