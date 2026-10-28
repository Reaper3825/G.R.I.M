#include "plugin_logging.hpp"
#include <atomic>
namespace grim::plugin {
namespace { std::atomic<const GrimPluginAPI*> loggingAPI{nullptr}; }
void setLoggingAPI(const GrimPluginAPI* api) { loggingAPI.store(api); }
void log(GrimLogLevel level, const std::string& tag, const std::string& message) {
    if (const auto* api = loggingAPI.load()) {
        const auto text = "[" + tag + "] " + message;
        api->log(api, level, text.c_str());
    }
}
}
