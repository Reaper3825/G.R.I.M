#pragma once
#include <string>
// Copies display data onto the UI queue. Never retains plugin callbacks or pointers.
void publishPluginNotification(const std::string& title, const std::string& message);
