#include <string_view>
// Filesystem provider owns operations and captures its base directory at load.
#include "core/plugin_api.hpp"
#include "core/plugin_tool_result.hpp"
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
    grim::PluginHost api;
    const fs::path currentDirectory = fs::current_path();
    OperationContext operations[3] = {{this, Ls}, {this, Mkdir}, {this, Rm}};
    explicit Provider(grim::PluginHost value) : api(value) {}
    bool publishDirectory(const fs::path& directory) const {
        const auto path = directory.u8string();
        const auto payload = nlohmann::json::array({{
            {"resource_id", "working_directory"}, {"kind", "filesystem.directory"},
            {"name", "Working directory"}, {"runtime_id", uint64_t{1}},
            {"state", {{"format", "grim-filesystem-directory"}, {"version", 1},
                {"path", std::string(path.begin(), path.end())}, {"scope", "provider"}}}
        }}).dump();
        return api->replace_resources(api.instance(), "filesystem.directory", payload.data(), payload.size()) == GRIM_OK;
    }
};
std::unique_ptr<Provider> provider;

GrimToolReply filesystemReply(bool success, std::string_view message, const char* error = "ERR_NONE") {
    return grim::toolReply(success, message, error, success ? "routine" : "error");
}

GrimToolReply execute(const GrimToolCall* call, void* data) {
    if (!call || call->struct_size != sizeof(*call) || !data ||
        (!call->arguments && call->arguments_size)) return {GRIM_ERROR_INVALID_PARAM};
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
            return filesystemReply(false, "[FS] Path contains a NUL byte", "ERR_FS_INVALID_PATH");
        if (op == Ls) {
            std::vector<std::string> names;
            for (const auto& entry : fs::directory_iterator(currentDirectory))
                names.push_back(entry.path().filename().string());
            std::sort(names.begin(), names.end());
            std::string message = "[FS] Contents:\n";
            for (const auto& name : names) message += " - " + name + "\n";
            return filesystemReply(true, message);
        }
        if (pathText.empty()) return filesystemReply(false,
            "[FS] A path argument is required", "ERR_FS_NO_ARGUMENT");
        const auto path = currentDirectory / fs::path(pathText);
        if (op == Mkdir) {
            if (!fs::create_directory(path)) return filesystemReply(false,
                "[FS] Failed to create directory: " + path.string(), "ERR_FS_CREATE_FAILED");
            return filesystemReply(true, "[FS] Directory created: " + path.string());
        }
        // remove() deletes a file, symlink, or empty directory; never recursively.
        if (!fs::remove(path)) return filesystemReply(false,
            "[FS] File not found: " + pathText, "ERR_FS_NOT_FOUND");
        return filesystemReply(true, "[FS] Removed: " + path.string());
    } catch (const fs::filesystem_error& error) {
        return filesystemReply(false, std::string("[FS] ") + error.what(), "ERR_FS_IO");
    } catch (const std::exception& error) {
        return filesystemReply(false, std::string("[FS] ") + error.what(), "ERR_FS_INVALID_PATH");
    }
}
}

extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* table, GrimPluginHandle instance) {
    const grim::PluginHost api{table, instance};
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION ||
        !api->replace_resources)
        return GRIM_ERROR_API_VERSION_MISMATCH;
    try {
        provider = std::make_unique<Provider>(api);
        return provider->publishDirectory(provider->currentDirectory) ? GRIM_OK : GRIM_ERROR_UNKNOWN;
    } catch (...) { return GRIM_ERROR_UNKNOWN; }
}

extern "C" PLUGIN_EXPORT GrimToolReply grim_plugin_dispatch_tool(
    const char* tool_id, const GrimToolCall* request) {
    if (!tool_id) return {GRIM_ERROR_INVALID_PARAM};
    const std::string_view id(tool_id);
    if (id == "ls") return execute(request, &provider->operations[Ls]);
    if (id == "mkdir") return execute(request, &provider->operations[Mkdir]);
    if (id == "rm") return execute(request, &provider->operations[Rm]);
    return {GRIM_ERROR_NOT_FOUND};
}

extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() { provider.reset(); }
