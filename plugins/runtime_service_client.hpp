#pragma once
#include "core/plugin_api.hpp"
#include "commands/command_result.hpp"
#include <stdexcept>

namespace grim::runtime {
inline grim::PluginHost host = {};
inline std::string take(const char* text) {
    if (!text) throw std::runtime_error("Host runtime service unavailable");
    struct Release { const char* text; ~Release() { host->string_free(host.instance(), text); } } release{text};
    return text;
}
inline GrimResult deliver(GrimToolResultWriter* writer, const CommandResult& value) {
    GrimToolResult result;
    result.success = value.success;
    result.message = value.message.c_str();
    result.error_code = value.errorCode.c_str();
    result.category = value.category.c_str();
    result.voice = value.voice.c_str();
    result.color_rgb = (value.color.r << 16) | (value.color.g << 8) | value.color.b;
    return writer->complete(writer, &result);
}
}
