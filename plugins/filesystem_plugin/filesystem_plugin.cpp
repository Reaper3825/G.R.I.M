#include "core/plugin_tool_registration.hpp"
// Filesystem provider owns operations and captures its base directory at load.
#include "core/plugin_api.hpp"
#include <algorithm>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
enum Operation { Ls, Mkdir, Rm };
struct Provider;
struct OperationContext { Provider* provider; Operation operation; };
struct Provider {
    const GrimPluginAPI* api;
    const fs::path currentDirectory = fs::current_path();
    OperationContext operations[3] = {{this, Ls}, {this, Mkdir}, {this, Rm}};
    explicit Provider(const GrimPluginAPI* value) : api(value) {}
    bool publishDirectory(const fs::path& directory) const {
        const auto path = directory.u8string();
        const auto payload = nlohmann::json::array({{
            {"resource_id", "working_directory"}, {"kind", "filesystem.directory"},
            {"name", "Working directory"}, {"runtime_id", uint64_t{1}},
            {"state", {{"format", "grim-filesystem-directory"}, {"version", 1},
                {"path", std::string(path.begin(), path.end())}, {"scope", "provider"}}}
        }}).dump();
        return api->replace_resources(api, "filesystem.directory", payload.data(), payload.size()) == GRIM_OK;
    }
};
std::unique_ptr<Provider> provider;

GrimResult complete(GrimToolResultWriter* writer, bool success,
                    const std::string& message, const char* error = "ERR_NONE") {
    GrimToolResult result;
    result.success = success;
    result.message = message.c_str();
    result.error_code = error;
    result.category = success ? "routine" : "error";
    result.color_rgb = success ? 0x00FFFF : 0xFF0000;
    return writer->complete(writer, &result);
}

GrimResult execute(const GrimToolCall* call, GrimToolResultWriter* writer, void* data) {
    if (!call || call->struct_size != sizeof(*call) || !writer ||
        writer->struct_size != sizeof(*writer) || !writer->complete || !data ||
        (!call->arguments && call->arguments_size)) return GRIM_ERROR_INVALID_PARAM;
    const std::string argument(call->arguments ? call->arguments : "", call->arguments_size);
    auto& context = *static_cast<OperationContext*>(data);
    const auto op = context.operation;
    const auto& currentDirectory = context.provider->currentDirectory;
    try {
        // A single raw path may contain spaces. Optional enclosing double quotes
        // are accepted; no shell expansion, flags, or multiple-path parsing.
        auto pathText = argument;
        if (pathText.size() >= 2 && pathText.front() == '"' && pathText.back() == '"')
            pathText = pathText.substr(1, pathText.size() - 2);
        if (pathText.find('\0') != std::string::npos)
            return complete(writer, false, "[FS] Path contains a NUL byte", "ERR_FS_INVALID_PATH");
        if (op == Ls) {
            std::vector<std::string> names;
            for (const auto& entry : fs::directory_iterator(currentDirectory))
                names.push_back(entry.path().filename().string());
            std::sort(names.begin(), names.end());
            std::string message = "[FS] Contents:\n";
            for (const auto& name : names) message += " - " + name + "\n";
            return complete(writer, true, message);
        }
        if (pathText.empty()) return complete(writer, false,
            "[FS] A path argument is required", "ERR_FS_NO_ARGUMENT");
        const auto path = currentDirectory / fs::path(pathText);
        if (op == Mkdir) {
            if (!fs::create_directory(path)) return complete(writer, false,
                "[FS] Failed to create directory: " + path.string(), "ERR_FS_CREATE_FAILED");
            return complete(writer, true, "[FS] Directory created: " + path.string());
        }
        // remove() deletes a file, symlink, or empty directory; never recursively.
        if (!fs::remove(path)) return complete(writer, false,
            "[FS] File not found: " + pathText, "ERR_FS_NOT_FOUND");
        return complete(writer, true, "[FS] Removed: " + path.string());
    } catch (const fs::filesystem_error& error) {
        return complete(writer, false, std::string("[FS] ") + error.what(), "ERR_FS_IO");
    } catch (const std::exception& error) {
        return complete(writer, false, std::string("[FS] ") + error.what(), "ERR_FS_INVALID_PATH");
    }
}
}

extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    static const GrimPluginInfo info{GRIM_API_VERSION, "Filesystem Plugin", "1.0.0",
        "GRIM Project", "Basic filesystem operations",
        GRIM_PERM_FILESYSTEM};
    return &info;
}

extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION ||
        !api->register_tools || !api->replace_resources)
        return GRIM_ERROR_API_VERSION_MISMATCH;
    try {
        provider = std::make_unique<Provider>(api);
        const char* ids[] = {"ls", "mkdir", "rm"};
        const char* descriptions[] = {"List entries in the filesystem plugin's working directory",
            "Create one directory; parent directories must already exist",
            "Remove one file, symlink, or empty directory; never recursively"};
        const char* usages[] = {"ls", "mkdir {c}", "rm {t}"};
        const char* summaries[] = {"List entries inside the current working directory",
            "Create a new directory within an existing parent",
            "Remove one file or empty directory without recursion"};
        GRIM::PluginToolRegistration tools[3];
        const std::map<std::string, std::string> directoryBindings = {{"c", "c"}};
        const std::map<std::string, std::string> pathBindings = {{"t", "t"}};
        tools[Mkdir].descriptor.bindings = directoryBindings;
        tools[Rm].descriptor.bindings = pathBindings;
        GRIM::MMO::ToolParameter paths[3];
        for (size_t i = 0; i < 3; ++i) {
            tools[i].descriptor.usage_type = GRIM::MMO::ToolUsage::System;
            tools[i].descriptor.short_description = summaries[i];
            tools[i].handler = execute;
            tools[i].user_data = &provider->operations[i];
            auto& d = tools[i].descriptor;
            d.tool_id = d.display_name = ids[i];
            d.version = "1.0.0";
            d.description = descriptions[i];
            d.usage = usages[i];
            d.category = "filesystem";
            d.argument_format = "raw_text";
            d.permission_bits = GRIM_PERM_FILESYSTEM;
            d.is_informational = i == Ls;
            d.needs_confirmation = !d.is_informational;
            if (i == Mkdir || i == Rm) {
                paths[i].name = "path";
                paths[i].type = "path";
                paths[i].description = "One absolute path or path relative to the plugin working directory; spaces allowed; optional enclosing double quotes";
                paths[i].required = 1;
                d.parameters = {paths[i]};
            }
        }
        const auto registered = GRIM::registerPluginTools(api, tools, 3);
        if (registered != GRIM_OK) return registered;
        return provider->publishDirectory(provider->currentDirectory) ? GRIM_OK : GRIM_ERROR_UNKNOWN;
    } catch (...) { return GRIM_ERROR_UNKNOWN; }
}

extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() { provider.reset(); }
