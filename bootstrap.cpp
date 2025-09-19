#include "bootstrap.hpp"
#include "bootstrap_config.hpp"
#include "resources.hpp"
#include "console_history.hpp"
#include "aliases.hpp"
#include "system_detect.hpp"

// Define global system info
SystemInfo g_systemInfo;

void runBootstrapChecks(int argc, char** argv) {
    grimLog("[GRIM] Startup begin");

    // Centralized config/memory bootstrap
    bootstrap_config::initAll();

    // Aliases system (cache only at bootstrap)
    grimLog("[aliases] Bootstrap: initializing (cache only, no scan)");
    aliases::init();
    grimLog("[aliases] Bootstrap: init finished");

    // Fonts
    std::string fontPath = findAnyFontInResources(argc, argv, &history);
    if (!fontPath.empty())
        grimLog("[Config] Font found: " + fontPath);
    else
        grimLog("[Config] No system font found, UI may render incorrectly");

    grimLog("[GRIM] ---- Bootstrap Complete ----");
}
