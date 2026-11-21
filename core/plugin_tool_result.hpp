#pragma once

#include "plugin_api.hpp"
#include "commands/command_result.hpp"
#include "MMO/Core/ToolResultJson.hpp"
#include <algorithm>
#include <new>

namespace grim {

// Called in the provider so allocation and release use the same runtime.
inline GrimToolReply toolReply(const nlohmann::json& result) noexcept {
    try {
        if (!result.is_object() || !result.at("payload").is_object())
            return {GRIM_ERROR_INVALID_PARAM};
        const auto text = result.dump();
        auto* bytes = new char[text.size()];
        std::copy(text.begin(), text.end(), bytes);
        return {GRIM_OK, bytes, text.size(), [](const char* data) noexcept { delete[] data; }};
    } catch (const std::bad_alloc&) {
        return {GRIM_ERROR_OUT_OF_MEMORY};
    } catch (...) {
        return {GRIM_ERROR_INVALID_PARAM};
    }
}

inline GrimToolReply toolReply(bool success, std::string_view message,
                               std::string_view errorCode, std::string_view category,
                               nlohmann::json payload = nlohmann::json::object()) {
    return toolReply(GRIM::MMO::makeToolResult(success, message, errorCode, category,
                                             std::move(payload)));
}

inline GrimToolReply toolReply(const CommandResult& value) {
    return toolReply(value.success, value.message, value.errorCode, value.category,
                     value.structuredOutput);
}

}
