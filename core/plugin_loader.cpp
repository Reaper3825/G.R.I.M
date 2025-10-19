#include "plugin.hpp"
#include "logger.hpp"
#include <filesystem>
#include <windows.h>




// ====================================================
// Runtime Plugin Loader
// ====================================================
void loadExternalPlugins() {
    namespace fs = std::filesystem;
    const fs::path pluginDir = "plugins";

    if (!fs::exists(pluginDir)) {
        LOG_ERROR("[PluginLoader]", "No plugins directory found.");
        return;
    }

    for (const auto& entry : fs::directory_iterator(pluginDir)) {
        if (!entry.is_regular_file())
            continue;

        auto path = entry.path();
        if (path.extension() != ".dll")
            continue;

        LOG_DEBUG("[PluginLoader]", "Loading external plugin: " + path.string());

        HMODULE lib = LoadLibraryA(path.string().c_str());
        if (!lib) {
            LOG_ERROR("[PluginLoader]", "Failed to load " + path.string());
            continue;
        }

        using RegisterFunc = void (*)();
        auto registerFunc = reinterpret_cast<RegisterFunc>(
            GetProcAddress(lib, "registerGrimPlugin"));

        if (!registerFunc) {
            LOG_ERROR("PluginLoader", "Missing registerGrimPlugin() export in " + path.string());
            FreeLibrary(lib);
            continue;
        }

        try {
            registerFunc();
            LOG_DEBUG("PluginLoader", "Successfully registered plugin: " + path.filename().string());
        } catch (...) {
            LOG_ERROR("PluginLoader", "Exception while registering " + path.string());
        }
    }
}
