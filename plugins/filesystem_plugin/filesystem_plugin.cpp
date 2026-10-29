// Standalone API v3 plugin. No host globals, C++ symbols, or model dependencies.
#include "core/plugin_api.hpp"
#include <algorithm>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
fs::path currentDirectory;
std::mutex directoryMutex;
enum Operation { Pwd, Cd, Ls, Mkdir, Rm };
Operation operations[] = {Pwd, Cd, Ls, Mkdir, Rm};

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
    const auto op = *static_cast<Operation*>(data);
    std::lock_guard lock(directoryMutex);
    try {
        // A single raw path may contain spaces. Optional enclosing double quotes
        // are accepted; no shell expansion, flags, or multiple-path parsing.
        auto pathText = argument;
        if (pathText.size() >= 2 && pathText.front() == '"' && pathText.back() == '"')
            pathText = pathText.substr(1, pathText.size() - 2);
        if (pathText.find('\0') != std::string::npos)
            return complete(writer, false, "[FS] Path contains a NUL byte", "ERR_FS_INVALID_PATH");
        if (op == Pwd) return complete(writer, true, "[FS] Current directory: " + currentDirectory.string());
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
        if (op == Cd) {
            if (!fs::exists(path)) return complete(writer, false,
                "[FS] Directory does not exist: " + pathText, "ERR_FS_NOT_FOUND");
            if (!fs::is_directory(path)) return complete(writer, false,
                "[FS] Path is not a directory: " + pathText, "ERR_FS_NOT_DIRECTORY");
            currentDirectory = fs::canonical(path);
            return complete(writer, true, "[FS] Changed directory to: " + currentDirectory.string());
        }
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
        "GRIM Project", "Working-directory navigation and basic filesystem operations",
        GRIM_PERM_FILESYSTEM};
    return &info;
}

extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION)
        return GRIM_ERROR_API_VERSION_MISMATCH;
    try {
        currentDirectory = fs::current_path();
        const char* ids[] = {"pwd", "cd", "ls", "mkdir", "rm"};
        const char* descriptions[] = {"Show the filesystem plugin's working directory",
            "Change the filesystem plugin's working directory without changing the host process directory",
            "List entries in the filesystem plugin's working directory",
            "Create one directory; parent directories must already exist",
            "Remove one file, symlink, or empty directory; never recursively"};
        const char* usages[] = {"pwd", "cd <directory>", "ls", "mkdir <directory>", "rm <path>"};
        const char* examples[] = {"pwd", "cd C:/Users", "ls", "mkdir new folder", "rm old.txt"};
        GrimToolRegistration tools[5];
        GrimToolParameter paths[5];
        for (size_t i = 0; i < 5; ++i) {
            tools[i].handler = execute;
            tools[i].user_data = &operations[i];
            auto& d = tools[i].descriptor;
            d.tool_id = d.display_name = ids[i];
            d.version = "1.0.0";
            d.description = descriptions[i];
            d.usage = usages[i];
            d.category = "filesystem";
            d.argument_format = "raw_text";
            d.required_permissions = GRIM_PERM_FILESYSTEM;
            d.is_informational = i == Pwd || i == Ls;
            d.needs_confirmation = !d.is_informational;
            d.examples = {&examples[i], 1};
            if (i == Cd || i == Mkdir || i == Rm) {
                paths[i].name = "path";
                paths[i].type = "path";
                paths[i].description = "One absolute path or path relative to the plugin working directory; spaces allowed; optional enclosing double quotes";
                paths[i].required = 1;
                d.parameters = &paths[i];
                d.parameter_count = 1;
            }
        }
        return api->register_tools(api, tools, 5);
    } catch (...) { return GRIM_ERROR_UNKNOWN; }
}

extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {}
