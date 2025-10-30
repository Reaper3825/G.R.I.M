#pragma once
#include <string>

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

// Detect objects in current view
std::string detectObjects();

} // namespace Perception
} // namespace GRIM
