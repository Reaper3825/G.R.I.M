// Compiled into the test DLL, never into the host. No real processes are launched.
#include "plugins/application_plugin/application_catalog.hpp"
#include "plugins/application_plugin/shell_open.hpp"
namespace GRIM::Applications {
Backend nativeBackend() { return {"fixture", [] {
    return std::vector<Application>{{"fixture", "editor", "Test editor", "fixture", "editor", ""}};
}}; }
Backend portableBackend() { return {"portable-fixture", [] { return std::vector<Application>{}; }}; }
void launchNative(const Application& app) {
    if (app.nativeId != "editor" || app.launchTarget != "editor") throw std::runtime_error("Wrong plugin launch target");
}
}
int64_t shellOpenTarget(const std::string& target) {
    if (target == "missing" || target == "https://www.google.com/search?q=query") return 32;
    if (target == "  C:/test folder/app.exe  " || target == "https://www.google.com/search?q=Mixed Case & stuff") return 33;
    throw std::runtime_error("Plugin altered the native launch target or launched empty input");
}
