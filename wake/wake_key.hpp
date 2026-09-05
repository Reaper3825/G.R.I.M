#pragma once
#include <string>

namespace WakeKey {

void start();

void stop();
bool isRunning();
void update();

// External local controllers (for example physical hand gestures) use the
// same wake/capture path as the configured key binding.
bool requestWake(const std::string& source);

} // namespace WakeKey
