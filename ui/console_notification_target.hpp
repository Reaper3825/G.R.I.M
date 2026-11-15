#pragma once
#include "core/host_exports.hpp"
#include <string>

// Host notification delivery uses this interface, never the plugin's concrete
// panel type. Calls are synchronous on the UI thread after a fresh panel lookup.
class GRIM_HOST_API ConsoleNotificationTarget {
public:
    virtual ~ConsoleNotificationTarget() = default;
    virtual void appendNotification(const std::string& text) = 0;
};
