#pragma once
#include "core/plugin_api.hpp"
#include "commands/command_result.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace grim::runtime {
inline const GrimPluginAPI* host = nullptr;
struct UnknownServiceFailure {};
inline std::string take(const char* text) {
    if (!text) throw std::runtime_error("Host runtime service unavailable");
    struct Release { const char* text; ~Release() { host->string_free(host, text); } } release{text};
    return text;
}
inline nlohmann::json response(const char* text) {
    const auto value = nlohmann::json::parse(take(text));
    if (value.at("format") != "grim-runtime-service" || value.at("version") != 1)
        throw std::runtime_error("Unsupported host runtime service response");
    if (!value.at("error").is_null()) {
        if (value.at("error").at("kind") == "exception")
            throw std::runtime_error(value.at("error").at("message").get<std::string>());
        throw UnknownServiceFailure{};
    }
    return value.at("value");
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
