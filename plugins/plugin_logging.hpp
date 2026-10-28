#pragma once
#include "core/plugin_api.hpp"
#include <string>
namespace grim::plugin {
void setLoggingAPI(const GrimPluginAPI* api);
void log(GrimLogLevel level, const std::string& tag, const std::string& message);
}
#ifdef LOG_DEBUG
#undef LOG_DEBUG
#endif
#ifdef LOG_ERROR
#undef LOG_ERROR
#endif
#define LOG_DEBUG(tag, message) grim::plugin::log(GRIM_LOG_DEBUG, tag, message)
#define LOG_ERROR(tag, message) grim::plugin::log(GRIM_LOG_ERROR, tag, message)
