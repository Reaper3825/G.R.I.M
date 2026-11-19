#pragma once
#include "core/plugin_api.hpp"
#include <stdexcept>

namespace grim::runtime {
inline grim::PluginHost host = {};
inline std::string take(const char* text) {
    if (!text) throw std::runtime_error("Host runtime service unavailable");
    struct Release { const char* text; ~Release() { host->string_free(host.instance(), text); } } release{text};
    return text;
}
}
