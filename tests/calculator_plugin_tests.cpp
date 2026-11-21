#include "core/plugin_api.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

using Json = nlohmann::json;
static GrimToolDispatch dispatch;
static void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

static void validateManifest(const std::filesystem::path& path) {
    std::ifstream input(path);
    check(static_cast<bool>(input), "Missing calculator manifest");
    const auto entries = Json::parse(input).at("tools");
    check(entries.size() == 1, "Expected one calculator tool");
    const auto& entry = entries[0];
    check(entry.value("usage_type", 1) == 1 && entry.at("is_informational") == true &&
          entry.at("needs_confirmation") == false, "Invalid tool policy");
    check(entry.at("parameters").size() == 1 && entry.at("parameters")[0].at("required") == true &&
          entry.at("parameters")[0].at("name") == "expression", "Missing expression parameter");
}

int main(int argc, char** argv) {
    try {
        check(argc == 2, "Pass the built calculator plugin path");
#ifdef _WIN32
        const auto library = LoadLibraryA(argv[1]);
        check(library != nullptr, "Could not load calculator plugin DLL");
        const auto symbol = [library](const char* name) { return GetProcAddress(library, name); };
#else
        const auto library = dlopen(argv[1], RTLD_NOW);
        check(library != nullptr, "Could not load calculator plugin library");
        const auto symbol = [library](const char* name) { return dlsym(library, name); };
#endif
        const auto pluginInit = reinterpret_cast<GrimResult (*)(const GrimPluginAPI*, GrimPluginHandle)>(symbol("grim_plugin_init"));
        const auto pluginShutdown = reinterpret_cast<void (*)()>(symbol("grim_plugin_shutdown"));
        check(pluginInit && pluginShutdown, "Missing native plugin exports");
        check(pluginInit(nullptr, 0) == GRIM_ERROR_API_VERSION_MISMATCH, "Null API accepted");
        GrimPluginAPI api{};
        api.api_version = GRIM_API_VERSION - 0x100;
        check(pluginInit(&api, 1) == GRIM_ERROR_API_VERSION_MISMATCH, "Old API accepted");
        api.api_version = GRIM_API_VERSION;
        dispatch = reinterpret_cast<GrimToolDispatch>(symbol("grim_plugin_dispatch_tool"));
        check(dispatch != nullptr, "Missing dispatcher export");
        check(pluginInit(&api, 1) == GRIM_OK, "Plugin initialization failed");
        validateManifest(std::filesystem::path(argv[1]).replace_extension(".tools.json"));
        const GrimToolCall bad{.arguments_size = 1};
        check(dispatch("calculate", nullptr).delivery == GRIM_ERROR_INVALID_PARAM, "Null call accepted");
        check(dispatch("calculate", &bad).delivery == GRIM_ERROR_INVALID_PARAM, "Missing arguments accepted");
        check(dispatch("missing", &bad).delivery == GRIM_ERROR_NOT_FOUND, "Unknown tool accepted");
        pluginShutdown();
#ifdef _WIN32
        FreeLibrary(library);
#else
        dlclose(library);
#endif
        std::cout << "Calculator plugin ABI and manifest tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
