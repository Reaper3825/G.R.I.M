#pragma once
#include <string>

// Lightweight declarations for the existing host observation services. Capture
// and live context remain in the host; no OpenCV object crosses the plugin API.
namespace GRIM::Perception {
bool isAvailable();
std::string analyzeScreen();
std::string readText(int x = -1, int y = -1, int width = -1, int height = -1);
std::string detectObjects();
}
