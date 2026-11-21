#pragma once

#include "plugin_api.hpp"
#include "commands/command_result.hpp"

namespace grim {

inline GrimResult completeToolResult(GrimToolResultWriter* writer, bool success,
                                     const char* message, const char* errorCode, const char* category,
                                     const char* voice = nullptr,
                                     const char* structuredJson = nullptr) {
    GrimToolResult result;
    result.success = success;
    result.message = message;
    result.error_code = errorCode;
    result.category = category;
    result.voice = voice;
    result.structured_output_json = structuredJson;
    return writer->complete(writer, &result);
}

inline GrimResult completeToolResult(GrimToolResultWriter* writer, const CommandResult& value) {
    const auto payloadText = value.structuredOutput.dump();
    return completeToolResult(writer, value.success, value.message.c_str(), value.errorCode.c_str(),
                              value.category.c_str(), value.voice.c_str(),
                              payloadText.c_str());
}

}
