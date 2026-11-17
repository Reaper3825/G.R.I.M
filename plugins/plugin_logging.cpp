#include "plugin_logging.hpp"
#include <mutex>

namespace grim::plugin {
namespace {
std::mutex loggingMutex;
PluginHost loggingAPI;
}

void setLoggingAPI(PluginHost api) {
    std::lock_guard lock(loggingMutex);
    loggingAPI = api;
}

void log(GrimLogLevel level, const std::string& tag, const std::string& message) {
    PluginHost api;
    {
        std::lock_guard lock(loggingMutex);
        api = loggingAPI;
    }
    if (api) {
        const auto text = "[" + tag + "] " + message;
        api->log(api.instance(), level, text.c_str());
    }
}
}
