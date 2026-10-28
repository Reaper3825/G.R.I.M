// ============================================================================
// G.R.I.M Plugin API Implementation
// ============================================================================
// This file implements the host-side functions that populate the plugin API
// function table. These are the actual implementations that plugins call.
// ============================================================================

#include "plugin_api.hpp"
#include "logger.hpp"
#include "commands/commands_core.hpp"
#include "commands/commands_ai.hpp"
#include "voice/voice_speak.hpp"
#include "../MMO/Core/HardwareInventory.hpp"
#include "../MMO/Core/ToolRegistry.hpp"  // Phase 0.25: plugin↔ToolRegistry sync
#include "popup_ui/popup_ui.hpp"
#include "helpers/grim_input.hpp"  // For GRIMInput::parseInput
#include "synonyms.hpp"            // For normalizeWord() synonym resolution
#include <unordered_map>
#include <vector>
#include <fstream>
#include <cstdarg>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <thread>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <chrono>

// Platform-specific includes
#ifdef _WIN32
    #include <windows.h>  // For CreateProcess, psapi, tlhelp32, etc.
    #include <psapi.h>
    #include <tlhelp32.h>
#else
    #include <unistd.h>
    #include <sys/types.h>
    #include <sys/wait.h>
    #include <signal.h>
    #include <dirent.h>
    #include <fstream>
    #include <cstdlib>
#endif

// External globals
extern GRIM::MMO::HardwareInventory g_hardwareInventory;

namespace PluginAPI {

// ============================================================================
// Internal Data Structures
// ============================================================================

struct EventSubscription {
    std::string event_name;
    GrimEventCallback callback;
    void* user_data;
    std::string plugin_name;
};

struct PluginTimer {
    uint64_t id;
    uint64_t interval_ms;
    bool repeat;
    GrimTimerCallback callback;
    void* user_data;
    std::chrono::steady_clock::time_point next_fire;
    bool active;
    std::string plugin_name;
};

struct PluginContext {
    std::string name;
    std::string path;
    uint32_t permissions;
    GrimPluginAPI api{};
};

// Static storage
static std::unordered_map<uint64_t, EventSubscription> s_event_subscriptions;
static uint64_t s_next_event_id = 1;
static std::unordered_map<uint64_t, PluginTimer> s_timers;
static uint64_t s_next_timer_id = 1;
static std::unordered_map<std::string, std::string> s_context_storage;
static std::unordered_map<void*, std::string> s_allocated_strings;
static std::unordered_map<std::string, std::unique_ptr<PluginContext>> s_plugin_contexts;

// ============================================================================
// Helper Functions
// ============================================================================

static const PluginContext& pluginContext(const GrimPluginAPI* self) {
    if (!self || !self->host_context) {
        throw std::invalid_argument("Plugin API requires its owning API object");
    }
    return *static_cast<const PluginContext*>(self->host_context);
}

static bool checkPermission(const GrimPluginAPI* self, GrimPermission perm) {
    return (pluginContext(self).permissions & perm) == static_cast<uint32_t>(perm);
}

static const char* allocateString(const std::string& str) {
    char* buffer = new char[str.size() + 1];
    std::strcpy(buffer, str.c_str());
    s_allocated_strings[buffer] = str;
    return buffer;
}

// ============================================================================
// Logging Implementation
// ============================================================================

static void api_log(const GrimPluginAPI* self, GrimLogLevel level, const char* message) {
    std::string prefix = "[Plugin: " + pluginContext(self).name + "] ";
    std::string full_msg = prefix + message;
    
    switch (level) {
        case GRIM_LOG_DEBUG:    LOG_DEBUG("Plugin", full_msg); break;
        case GRIM_LOG_INFO: LOG_DEBUG("Plugin", full_msg); break;  // Map INFO to DEBUG
    case GRIM_LOG_WARNING:  LOG_ERROR("Plugin", full_msg); break;  // Map WARNING to ERROR
        case GRIM_LOG_ERROR:    LOG_ERROR("Plugin", full_msg); break;
        case GRIM_LOG_CRITICAL: LOG_ERROR("Plugin", full_msg); break;  // Map CRITICAL to ERROR
    }
}

static void api_log_fmt(const GrimPluginAPI* self, GrimLogLevel level, const char* format, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    api_log(self, level, buffer);
}

// ============================================================================
// Command Registration Implementation
// ============================================================================

static GrimResult api_register_command(const GrimPluginAPI* self, const char* command_name, GrimCommandHandler handler, void* user_data) {
    if (!command_name || !*command_name || !handler) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    std::string cmd_name = command_name;
    
    const auto& owner = pluginContext(self);
    GRIM::MMO::ToolDescriptor desc;
    desc.tool_id = desc.display_name = cmd_name;
    desc.provider_type = GRIM::MMO::ToolProviderType::Plugin;
    desc.provider_name = owner.name;
    desc.description = "Plugin command: " + cmd_name;
    desc.category = "plugin";
    desc.permission_bits = owner.permissions;
    desc.is_informational = false;
    auto invoke = [handler, user_data](const std::string& input) -> CommandResult {
        const auto returned = handler(input.c_str(), user_data);
        CommandResult result;
        result.success = returned.success;
        result.message = returned.message ? returned.message : "";
        result.errorCode = returned.error_code ? returned.error_code : "";
        result.category = returned.category ? returned.category : "";
        const auto rgb = returned.color_rgb;
        result.color = Color{static_cast<uint8_t>((rgb >> 16) & 0xFF),
                             static_cast<uint8_t>((rgb >> 8) & 0xFF),
                             static_cast<uint8_t>(rgb & 0xFF)};
        return result;
    };
    return GRIM::MMO::ToolRegistry::instance().registerTool(desc, std::move(invoke))
        ? GRIM_OK : GRIM_ERROR_ALREADY_EXISTS;
}

static GrimResult api_unregister_command(const GrimPluginAPI* self, const char* command_name) {
    if (!command_name || !*command_name) return GRIM_ERROR_INVALID_PARAM;
    auto& registry = GRIM::MMO::ToolRegistry::instance();
    const auto tool = registry.getTool(command_name);
    if (!tool || tool->tool_id != command_name) return GRIM_ERROR_NOT_FOUND;
    const auto& owner = pluginContext(self);
    if (tool->provider_type != GRIM::MMO::ToolProviderType::Plugin || tool->provider_name != owner.name)
        return GRIM_ERROR_PERMISSION_DENIED;
    return registry.unregisterTool(command_name, owner.name) ? GRIM_OK : GRIM_ERROR_NOT_FOUND;
}

static bool api_is_command_registered(const GrimPluginAPI* self, const char* command_name) {
    return command_name && GRIM::MMO::ToolRegistry::instance().isRegistered(command_name);
}

static GrimCommandResult api_execute_command(const GrimPluginAPI* self, const char* command_text) {
    GrimCommandResult result;
    result.success = false;
    result.message = "Not implemented";
    result.error_code = "UNSUPPORTED";
    result.category = "system";
    result.color_rgb = 0xFF0000;
    
    if (!command_text) {
        result.error_code = "INVALID_PARAM";
        result.message = "Command text is null";
        return result;
    }
    
    // Parse and execute through GRIM's system
    auto [cmd, arg] = GRIMInput::parseInput(command_text);
    CommandResult cmd_result = dispatchCommand(cmd, arg);

    result.success = cmd_result.success;
    result.message = allocateString(cmd_result.message);
    result.error_code = allocateString(cmd_result.errorCode);
    result.category = allocateString(cmd_result.category);
    result.color_rgb = (cmd_result.color.r << 16) | (cmd_result.color.g << 8) | cmd_result.color.b;
    
    return result;
}

// ============================================================================
// Event System Implementation
// ============================================================================

static GrimEventHandle api_subscribe_event(const GrimPluginAPI* self, const char* event_name, GrimEventCallback callback, void* user_data) {
    if (!event_name || !callback) {
        return nullptr;
    }
    
    uint64_t handle = s_next_event_id++;
    
    EventSubscription sub;
    sub.event_name = event_name;
    sub.callback = callback;
    sub.user_data = user_data;
    sub.plugin_name = pluginContext(self).name;
    
    s_event_subscriptions[handle] = sub;
    
    LOG_DEBUG("PluginAPI", "Event subscription: " + std::string(event_name));
    return reinterpret_cast<GrimEventHandle>(handle);
}

static GrimResult api_unsubscribe_event(const GrimPluginAPI* self, GrimEventHandle handle) {
    if (!handle) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    uint64_t id = reinterpret_cast<uint64_t>(handle);
    auto it = s_event_subscriptions.find(id);
    
    if (it == s_event_subscriptions.end()) {
        return GRIM_ERROR_NOT_FOUND;
    }
    
    if (it->second.plugin_name != pluginContext(self).name) {
        return GRIM_ERROR_PERMISSION_DENIED;
    }
    s_event_subscriptions.erase(it);
    return GRIM_OK;
}

static GrimResult api_emit_event(const GrimPluginAPI* self, const char* event_name, const void* data, size_t data_size) {
    if (!event_name) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    std::string evt_name = event_name;
    
    // Notify all subscribers
    for (const auto& [id, sub] : s_event_subscriptions) {
        if (sub.event_name == evt_name) {
            sub.callback(event_name, data, data_size, sub.user_data);
        }
    }
    
    return GRIM_OK;
}

// ============================================================================
// Memory & Context Implementation
// ============================================================================

static const char* api_get_context(const GrimPluginAPI* self, const char* key) {
    if (!key) return nullptr;
    
    auto it = s_context_storage.find(key);
    if (it == s_context_storage.end()) {
        return nullptr;
    }
    
    return allocateString(it->second);
}

static GrimResult api_set_context(const GrimPluginAPI* self, const char* key, const char* value) {
    if (!key || !value) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    s_context_storage[key] = value;
    return GRIM_OK;
}

static GrimResult api_delete_context(const GrimPluginAPI* self, const char* key) {
    if (!key) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    auto it = s_context_storage.find(key);
    if (it == s_context_storage.end()) {
        return GRIM_ERROR_NOT_FOUND;
    }
    
    s_context_storage.erase(it);
    return GRIM_OK;
}

static bool api_has_context(const GrimPluginAPI* self, const char* key) {
    if (!key) return false;
    return s_context_storage.find(key) != s_context_storage.end();
}

static GrimResult api_store_memory(const GrimPluginAPI* self, const char* key, const char* value, const char* category) {
    if (!checkPermission(self, GRIM_PERM_MEMORY)) {
        return GRIM_ERROR_PERMISSION_DENIED;
    }
    
    if (!key || !value) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    // Store in GRIM's memory system
    // TODO: Implement proper MemoryObject creation
// g_memoryStorage.store(key, value, category ? category : "plugin");
    LOG_DEBUG("PluginAPI", std::string("Memory store not yet implemented for key: ") + key);
    return GRIM_OK;
}

static const char* api_retrieve_memory(const GrimPluginAPI* self, const char* key) {
    if (!checkPermission(self, GRIM_PERM_MEMORY)) {
        return nullptr;
    }
    
    if (!key) return nullptr;
    
    // TODO: Implement proper memory retrieval
    // std::string value = g_memoryStorage.retrieve(key);
    // if (value.empty()) return nullptr;
    // return allocateString(value);
    
    LOG_DEBUG("PluginAPI", std::string("Memory retrieve not yet implemented for key: ") + key);
    return nullptr;
}

static GrimResult api_delete_memory(const GrimPluginAPI* self, const char* key) {
    if (!checkPermission(self, GRIM_PERM_MEMORY)) {
        return GRIM_ERROR_PERMISSION_DENIED;
    }
    
    if (!key) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    // TODO: Implement proper memory deletion
    // g_memoryStorage.erase(key);
    LOG_DEBUG("PluginAPI", std::string("Memory delete not yet implemented for key: ") + key);
    return GRIM_OK;
}

// ============================================================================
// NLP & AI Implementation
// ============================================================================

static GrimIntent api_classify_intent(const GrimPluginAPI* self, const char* text) {
    (void)text;
    // ABI compatibility only. The application no longer classifies raw user
    // text into host-authored intents.
    return {};
}

static const char* api_resolve_synonym(const GrimPluginAPI* self, const char* word) {
    if (!word) return nullptr;
    
    // Use GRIM's existing synonym system (synonyms.hpp)
    std::string resolved = normalizeWord(word);
    
    // If normalization returns the same word, no synonym was found
    if (resolved == word) {
 LOG_DEBUG("PluginAPI", std::string("No synonym found for: ") + word);
    }
    
    return allocateString(resolved);
}

static const char* api_get_conversation_history(const GrimPluginAPI* self, int limit) {
    // Return JSON array of recent history
    // For now, return empty array
    return allocateString("[]");
}

// ============================================================================
// Output & UI Implementation
// ============================================================================

static void api_send_response(const GrimPluginAPI* self, const char* text) {
    if (!checkPermission(self, GRIM_PERM_UI)) {
        return;
    }
    
    if (text) {
        LOG_DEBUG("Plugin", std::string(text));
        // TODO: Send to actual UI output
    }
}

static void api_send_response_colored(const GrimPluginAPI* self, const char* text, uint32_t rgb_color) {
    if (!checkPermission(self, GRIM_PERM_UI)) {
        return;
    }
    
    if (text) {
        // Extract RGB components
        uint8_t r = (rgb_color >> 16) & 0xFF;
        uint8_t g = (rgb_color >> 8) & 0xFF;
        uint8_t b = rgb_color & 0xFF;
        
        LOG_DEBUG("Plugin", std::string(text));
        // TODO: Send to actual UI output with color
    }
}

static void api_show_notification(const GrimPluginAPI* self, const char* title, const char* message) {
    if (!checkPermission(self, GRIM_PERM_UI)) {
        return;
    }
    
    if (title && message) {
        std::string full_msg = std::string(title) + ": " + message;
        LOG_DEBUG("Plugin", full_msg);
        
        // Show popup UI notification (platform-specific UI)
        #ifdef _WIN32
        notifyPopupActivity();
        showPopup();
        #else
        // On Linux, just log for now (TODO: implement notification daemon integration)
        #endif
    }
}

// ============================================================================
// Voice Implementation
// ============================================================================

static GrimResult api_queue_tts(const GrimPluginAPI* self, const char* text, const GrimVoiceParams* params) {
    if (!checkPermission(self, GRIM_PERM_VOICE)) {
        return GRIM_ERROR_PERMISSION_DENIED;
    }
    
    if (!text) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    Voice::speak(text, "plugin");
    return GRIM_OK;
}

static GrimResult api_stop_tts(const GrimPluginAPI* self) {
    if (!checkPermission(self, GRIM_PERM_VOICE)) {
        return GRIM_ERROR_PERMISSION_DENIED;
    }
  
    // TODO: Implement audio stop functionality
    // Voice::stopAudio();
    LOG_DEBUG("PluginAPI", "Audio stop not yet implemented");
    return GRIM_OK;
}

static bool api_is_speaking(const GrimPluginAPI* self) {
    if (!checkPermission(self, GRIM_PERM_VOICE)) {
        return false;
    }
    
    return Voice::isSpeaking();
}

// ============================================================================
// File System Implementation
// ============================================================================

static GrimFileHandle api_open_file(const GrimPluginAPI* self, const char* path, const char* mode) {
    if (!checkPermission(self, GRIM_PERM_FILESYSTEM)) {
        return nullptr;
    }
    
    if (!path || !mode) {
        return nullptr;
    }
    
    FILE* file = fopen(path, mode);
    return reinterpret_cast<GrimFileHandle>(file);
}

static GrimResult api_close_file(const GrimPluginAPI* self, GrimFileHandle handle) {
    if (!checkPermission(self, GRIM_PERM_FILESYSTEM)) {
        return GRIM_ERROR_PERMISSION_DENIED;
    }
    
    if (!handle) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    FILE* file = reinterpret_cast<FILE*>(handle);
    fclose(file);
    return GRIM_OK;
}

static size_t api_read_file(const GrimPluginAPI* self, GrimFileHandle handle, char* buffer, size_t size) {
    if (!checkPermission(self, GRIM_PERM_FILESYSTEM)) {
        return 0;
    }
    
    if (!handle || !buffer) {
        return 0;
    }
    
    FILE* file = reinterpret_cast<FILE*>(handle);
    return fread(buffer, 1, size, file);
}

static size_t api_write_file(const GrimPluginAPI* self, GrimFileHandle handle, const char* data, size_t size) {
    if (!checkPermission(self, GRIM_PERM_FILESYSTEM)) {
        return 0;
    }
    
    if (!handle || !data) {
        return 0;
    }
    
    FILE* file = reinterpret_cast<FILE*>(handle);
    return fwrite(data, 1, size, file);
}

static const char* api_read_file_text(const GrimPluginAPI* self, const char* path) {
    if (!checkPermission(self, GRIM_PERM_FILESYSTEM)) {
        return nullptr;
    }
    
    if (!path) {
        return nullptr;
    }
    
    std::ifstream file(path);
    if (!file.is_open()) {
        return nullptr;
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    
    return allocateString(content);
}

static GrimResult api_write_file_text(const GrimPluginAPI* self, const char* path, const char* content) {
    if (!checkPermission(self, GRIM_PERM_FILESYSTEM)) {
        return GRIM_ERROR_PERMISSION_DENIED;
    }
    
    if (!path || !content) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    std::ofstream file(path);
    if (!file.is_open()) {
        return GRIM_ERROR_UNKNOWN;
    }
    
    file << content;
    return GRIM_OK;
}

static bool api_file_exists(const GrimPluginAPI* self, const char* path) {
    if (!checkPermission(self, GRIM_PERM_FILESYSTEM)) {
        return false;
    }
    
    if (!path) {
        return false;
    }
    
    return std::filesystem::exists(path);
}

static GrimResult api_delete_file(const GrimPluginAPI* self, const char* path) {
    if (!checkPermission(self, GRIM_PERM_FILESYSTEM)) {
        return GRIM_ERROR_PERMISSION_DENIED;
    }
    
    if (!path) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    if (!std::filesystem::exists(path)) {
        return GRIM_ERROR_NOT_FOUND;
    }
    
    std::filesystem::remove(path);
    return GRIM_OK;
}

// ============================================================================
// Process Management Implementation
// ============================================================================

static GrimResult api_launch_app(const GrimPluginAPI* self, const char* app_name) {
    if (!checkPermission(self, GRIM_PERM_PROCESS)) {
        return GRIM_ERROR_PERMISSION_DENIED;
    }
    
    if (!app_name) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    // Use GRIM's existing command system
    CommandResult result = cmdOpenApp(app_name);
    
    return result.success ? GRIM_OK : GRIM_ERROR_UNKNOWN;
}

static GrimResult api_launch_process(const GrimPluginAPI* self, const char* command, const char* args) {
    if (!checkPermission(self, GRIM_PERM_PROCESS)) {
        return GRIM_ERROR_PERMISSION_DENIED;
    }
    
    if (!command) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    #ifdef _WIN32
    // Windows implementation
    std::string cmd_line = command;
    if (args && strlen(args) > 0) {
        cmd_line += " " + std::string(args);
    }
    
    STARTUPINFOA si = {};
    PROCESS_INFORMATION pi = {};
    si.cb = sizeof(si);
    
    if (!CreateProcessA(
        nullptr,
        const_cast<char*>(cmd_line.c_str()),
        nullptr,
        nullptr,
        FALSE,
        0,
        nullptr,
        nullptr,
        &si,
        &pi
    )) {
        LOG_ERROR("PluginAPI", "Failed to launch process: " + cmd_line);
        return GRIM_ERROR_UNKNOWN;
    }
    
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    
    LOG_DEBUG("PluginAPI", "Launched process: " + cmd_line);
    return GRIM_OK;
    
    #else
    // Linux implementation
    pid_t pid = fork();
    
    if (pid < 0) {
        LOG_ERROR("PluginAPI", "Failed to fork process");
        return GRIM_ERROR_UNKNOWN;
    }
    
    if (pid == 0) {
        // Child process
        if (args && strlen(args) > 0) {
            execl("/bin/sh", "sh", "-c", (std::string(command) + " " + args).c_str(), nullptr);
        } else {
            execl("/bin/sh", "sh", "-c", command, nullptr);
        }
        // If exec fails
        _exit(1);
    }
    
    // Parent process
    LOG_DEBUG("PluginAPI", std::string("Launched process: ") + command);
    return GRIM_OK;
    #endif
}

static bool api_is_process_running(const GrimPluginAPI* self, const char* process_name) {
    if (!checkPermission(self, GRIM_PERM_PROCESS)) {
        return false;
    }
    
    if (!process_name) {
        return false;
    }
    
    #ifdef _WIN32
    // Windows implementation
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
        return false;
    }
    
    PROCESSENTRY32 pe32;
    pe32.dwSize = sizeof(PROCESSENTRY32);
    
    bool found = false;
    if (Process32First(snapshot, &pe32)) {
        do {
            std::string exe_file = pe32.szExeFile;
            std::string search_name = process_name;
            
            std::transform(exe_file.begin(), exe_file.end(), exe_file.begin(), ::tolower);
            std::transform(search_name.begin(), search_name.end(), search_name.begin(), ::tolower);
            
            if (search_name.find(".exe") == std::string::npos) {
                search_name += ".exe";
            }
            
            if (exe_file == search_name) {
                found = true;
                break;
            }
        } while (Process32Next(snapshot, &pe32));
    }
    
    CloseHandle(snapshot);
    return found;
    
    #else
    // Linux implementation - check /proc
    DIR* proc_dir = opendir("/proc");
    if (!proc_dir) {
        return false;
    }
    
    bool found = false;
    struct dirent* entry;
    
    while ((entry = readdir(proc_dir)) != nullptr) {
        // Check if directory name is a number (PID)
        if (entry->d_type != DT_DIR) continue;
        
        int pid = atoi(entry->d_name);
        if (pid <= 0) continue;
        
        // Read /proc/[pid]/comm
        std::string comm_path = std::string("/proc/") + entry->d_name + "/comm";
        std::ifstream comm_file(comm_path);
        if (comm_file.is_open()) {
            std::string proc_name;
            std::getline(comm_file, proc_name);
            
            if (proc_name == process_name) {
                found = true;
                break;
            }
        }
    }
    
    closedir(proc_dir);
    return found;
    #endif
}

static GrimResult api_kill_process(const GrimPluginAPI* self, const char* process_name) {
    if (!checkPermission(self, GRIM_PERM_PROCESS)) {
        return GRIM_ERROR_PERMISSION_DENIED;
    }
    
    if (!process_name) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    #ifdef _WIN32
    // Windows implementation
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
        return GRIM_ERROR_UNKNOWN;
    }
    
    PROCESSENTRY32 pe32;
    pe32.dwSize = sizeof(PROCESSENTRY32);
    
    bool killed = false;
    if (Process32First(snapshot, &pe32)) {
        do {
            std::string exe_file = pe32.szExeFile;
            std::string search_name = process_name;
            
            std::transform(exe_file.begin(), exe_file.end(), exe_file.begin(), ::tolower);
            std::transform(search_name.begin(), search_name.end(), search_name.begin(), ::tolower);
            
            if (search_name.find(".exe") == std::string::npos) {
                search_name += ".exe";
            }
            
            if (exe_file == search_name) {
                HANDLE hProcess = OpenProcess(PROCESS_TERMINATE, FALSE, pe32.th32ProcessID);
                if (hProcess) {
                    TerminateProcess(hProcess, 0);
                    CloseHandle(hProcess);
                    killed = true;
                    LOG_DEBUG("PluginAPI", "Killed process: " + std::string(process_name));
                }
            }
        } while (Process32Next(snapshot, &pe32));
    }
    
    CloseHandle(snapshot);
    return killed ? GRIM_OK : GRIM_ERROR_NOT_FOUND;
    
    #else
    // Linux implementation - find and kill by name
    DIR* proc_dir = opendir("/proc");
    if (!proc_dir) {
        return GRIM_ERROR_UNKNOWN;
    }
    
    bool killed = false;
    struct dirent* entry;
    
    while ((entry = readdir(proc_dir)) != nullptr) {
        if (entry->d_type != DT_DIR) continue;
        
        int pid = atoi(entry->d_name);
        if (pid <= 0) continue;
        
        // Read /proc/[pid]/comm
        std::string comm_path = std::string("/proc/") + entry->d_name + "/comm";
        std::ifstream comm_file(comm_path);
        if (comm_file.is_open()) {
            std::string proc_name;
            std::getline(comm_file, proc_name);
            
            if (proc_name == process_name) {
                if (kill(pid, SIGTERM) == 0) {
                    killed = true;
                    LOG_DEBUG("PluginAPI", "Killed process: " + std::string(process_name));
                }
            }
        }
    }
    
    closedir(proc_dir);
    return killed ? GRIM_OK : GRIM_ERROR_NOT_FOUND;
    #endif
}

// ============================================================================
// System Information Implementation
// ============================================================================

static const char* api_get_system_info(const GrimPluginAPI* self, const char* key) {
    if (!checkPermission(self, GRIM_PERM_SYSTEM)) {
        return nullptr;
    }
    
    if (!key) return nullptr;
    
    std::string key_str = key;
    
    if (key_str == "os") {
        return allocateString(g_hardwareInventory.os_name);
    } else if (key_str == "cpu") {
        return allocateString(std::to_string(g_hardwareInventory.cpu_cores) + " cores");
    }
    
return nullptr;
}

static uint64_t api_get_memory_usage(const GrimPluginAPI* self) {
    if (!checkPermission(self, GRIM_PERM_SYSTEM)) {
        return 0;
    }
    
    #ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    return pmc.WorkingSetSize;
    #else
    // Linux implementation - read /proc/self/status
    std::ifstream status("/proc/self/status");
    std::string line;
    
    while (std::getline(status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            // Extract the number (in kB)
            size_t pos = line.find_first_of("0123456789");
            if (pos != std::string::npos) {
                uint64_t kb = std::stoull(line.substr(pos));
                return kb * 1024; // Convert to bytes
            }
        }
    }
    
    return 0;
    #endif
}

static float api_get_cpu_usage(const GrimPluginAPI* self) {
    if (!checkPermission(self, GRIM_PERM_SYSTEM)) {
        return 0.0f;
    }
    
    #ifdef _WIN32
    static ULARGE_INTEGER lastCPU, lastSysCPU, lastUserCPU;
    static int numProcessors = 0;
    static HANDLE process = GetCurrentProcess();
    static bool initialized = false;
    
    if (!initialized) {
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        numProcessors = sysInfo.dwNumberOfProcessors;
        
        FILETIME ftime, fsys, fuser;
        GetSystemTimeAsFileTime(&ftime);
        memcpy(&lastCPU, &ftime, sizeof(FILETIME));
        
        GetProcessTimes(process, &ftime, &ftime, &fsys, &fuser);
        memcpy(&lastSysCPU, &fsys, sizeof(FILETIME));
        memcpy(&lastUserCPU, &fuser, sizeof(FILETIME));
        
        initialized = true;
        return 0.0f;
    }
    
    FILETIME ftime, fsys, fuser;
    ULARGE_INTEGER now, sys, user;
    
    GetSystemTimeAsFileTime(&ftime);
    memcpy(&now, &ftime, sizeof(FILETIME));
    
    GetProcessTimes(process, &ftime, &ftime, &fsys, &fuser);
    memcpy(&sys, &fsys, sizeof(FILETIME));
    memcpy(&user, &fuser, sizeof(FILETIME));
    
    double percent = (sys.QuadPart - lastSysCPU.QuadPart) + (user.QuadPart - lastUserCPU.QuadPart);
    percent /= (now.QuadPart - lastCPU.QuadPart);
    percent /= numProcessors;
    
    lastCPU = now;
    lastUserCPU = user;
    lastSysCPU = sys;
    
    return static_cast<float>(percent * 100.0);
    #else
    // Linux implementation - read /proc/self/stat
    static unsigned long long last_total_time = 0;
    static unsigned long long last_process_time = 0;
    static bool initialized = false;
    
    // Get system total time from /proc/stat
    std::ifstream stat_file("/proc/stat");
    std::string line;
    std::getline(stat_file, line);
    stat_file.close();
    
    unsigned long long user, nice, system, idle;
    sscanf(line.c_str(), "cpu %llu %llu %llu %llu", &user, &nice, &system, &idle);
    unsigned long long total_time = user + nice + system + idle;
    
    // Get process time from /proc/self/stat
    std::ifstream proc_stat("/proc/self/stat");
    std::string proc_line;
    std::getline(proc_stat, proc_line);
    proc_stat.close();
    
    // Parse utime and stime (fields 14 and 15, 1-indexed)
    unsigned long long utime = 0, stime = 0;
    int field = 0;
    size_t pos = 0;
    
    // Skip pid field
    pos = proc_line.find(' ');
    // Skip comm field (enclosed in parentheses, can contain spaces)
    pos = proc_line.find(')', pos) + 1;
    
    // Parse remaining fields
    std::istringstream iss(proc_line.substr(pos));
    std::string token;
    while (iss >> token && field < 15) {
        field++;
        if (field == 12) utime = std::stoull(token);  // 14th field (0-indexed: 12)
        if (field == 13) stime = std::stoull(token);  // 15th field (0-indexed: 13)
    }
    
    unsigned long long process_time = utime + stime;
    
    if (!initialized) {
        last_total_time = total_time;
        last_process_time = process_time;
        initialized = true;
        return 0.0f;
    }
    
    unsigned long long total_delta = total_time - last_total_time;
    unsigned long long process_delta = process_time - last_process_time;
    
    float percent = 0.0f;
    if (total_delta > 0) {
        percent = static_cast<float>(process_delta) / static_cast<float>(total_delta) * 100.0f;
    }
    
    last_total_time = total_time;
    last_process_time = process_time;
    
    return percent;
    #endif
}

// ============================================================================
// Timer Implementation (Stub)
// ============================================================================

static GrimTimerHandle api_create_timer(const GrimPluginAPI* self, uint64_t interval_ms, bool repeat, GrimTimerCallback callback, void* user_data) {
    if (!callback) {
        return nullptr;
    }
    
    uint64_t id = s_next_timer_id++;
    
    PluginTimer timer;
    timer.id = id;
    timer.interval_ms = interval_ms;
    timer.repeat = repeat;
    timer.callback = callback;
    timer.user_data = user_data;
    timer.next_fire = std::chrono::steady_clock::now() + std::chrono::milliseconds(interval_ms);
    timer.active = true;
    timer.plugin_name = pluginContext(self).name;
    
    s_timers[id] = timer;
    
    return reinterpret_cast<GrimTimerHandle>(id);
}

static GrimResult api_cancel_timer(const GrimPluginAPI* self, GrimTimerHandle handle) {
    if (!handle) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    uint64_t id = reinterpret_cast<uint64_t>(handle);
    auto it = s_timers.find(id);
    
    if (it == s_timers.end()) {
        return GRIM_ERROR_NOT_FOUND;
    }
    
    if (it->second.plugin_name != pluginContext(self).name) {
        return GRIM_ERROR_PERMISSION_DENIED;
    }
    s_timers.erase(it);
    return GRIM_OK;
}

// ============================================================================
// Plugin Lifecycle Implementation
// ============================================================================

static const char* api_get_plugin_name(const GrimPluginAPI* self) {
    return allocateString(pluginContext(self).name);
}

static const char* api_get_plugin_path(const GrimPluginAPI* self) {
    return allocateString(pluginContext(self).path);
}

static uint32_t api_get_permissions(const GrimPluginAPI* self) {
    return pluginContext(self).permissions;
}

static bool api_has_permission(const GrimPluginAPI* self, GrimPermission perm) {
    return checkPermission(self, perm);
}

static void api_request_reload(const GrimPluginAPI* self) {
    LOG_DEBUG("PluginAPI", "Plugin requested reload: " + pluginContext(self).name);
    // TODO: Trigger reload
}

// ============================================================================
// Utility Implementation
// ============================================================================

static void* api_allocate_memory(const GrimPluginAPI* self, size_t size) {
    return malloc(size);
}

static void api_free_memory(const GrimPluginAPI* self, void* ptr) {
    free(ptr);
}

static const char* api_string_duplicate(const GrimPluginAPI* self, const char* str) {
    if (!str) return nullptr;
    return allocateString(str);
}

static void api_string_free(const GrimPluginAPI* self, const char* str) {
    if (!str) return;
    
    auto it = s_allocated_strings.find(const_cast<void*>(static_cast<const void*>(str)));
    if (it != s_allocated_strings.end()) {
        delete[] str;
        s_allocated_strings.erase(it);
    }
}

// ============================================================================
// API Table Construction
// ============================================================================

GrimPluginAPI* createPluginAPI(const std::string& plugin_name, const std::string& plugin_path, uint32_t permissions) {
    if (plugin_name.empty() || s_plugin_contexts.count(plugin_name)) {
        throw std::invalid_argument("Empty or duplicate plugin identity: " + plugin_name);
    }
    auto context = std::make_unique<PluginContext>();
    context->name = plugin_name;
    context->path = plugin_path;
    context->permissions = permissions;
    auto& api = context->api;
    api.host_context = context.get();
    
    // Fill in API table
    api.api_version = GRIM_API_VERSION;
    
    // Logging
    api.log = api_log;
    api.log_fmt = api_log_fmt;
    
    // Commands
    api.register_command = api_register_command;
    api.unregister_command = api_unregister_command;
    api.is_command_registered = api_is_command_registered;
    api.execute_command = api_execute_command;
    
    // Events
    api.subscribe_event = api_subscribe_event;
    api.unsubscribe_event = api_unsubscribe_event;
    api.emit_event = api_emit_event;
    
    // Memory & Context
    api.get_context = api_get_context;
    api.set_context = api_set_context;
    api.delete_context = api_delete_context;
    api.has_context = api_has_context;
    api.store_memory = api_store_memory;
    api.retrieve_memory = api_retrieve_memory;
    api.delete_memory = api_delete_memory;
    
    // NLP & AI
    api.classify_intent = api_classify_intent;
    api.resolve_synonym = api_resolve_synonym;
    api.get_conversation_history = api_get_conversation_history;
    
    // Output & UI
    api.send_response = api_send_response;
    api.send_response_colored = api_send_response_colored;
    api.show_notification = api_show_notification;
    
    // Voice
    api.queue_tts = api_queue_tts;
    api.stop_tts = api_stop_tts;
    api.is_speaking = api_is_speaking;
    
    // File System
    api.open_file = api_open_file;
    api.close_file = api_close_file;
    api.read_file = api_read_file;
    api.write_file = api_write_file;
    api.read_file_text = api_read_file_text;
    api.write_file_text = api_write_file_text;
    api.file_exists = api_file_exists;
    api.delete_file = api_delete_file;
    
    // Process Management
    api.launch_app = api_launch_app;
    api.launch_process = api_launch_process;
    api.is_process_running = api_is_process_running;
    api.kill_process = api_kill_process;
    
    // System Information
    api.get_system_info = api_get_system_info;
    api.get_memory_usage = api_get_memory_usage;
    api.get_cpu_usage = api_get_cpu_usage;
    
    // Timers
    api.create_timer = api_create_timer;
    api.cancel_timer = api_cancel_timer;
    
    // Plugin Lifecycle
    api.get_plugin_name = api_get_plugin_name;
    api.get_plugin_path = api_get_plugin_path;
    api.get_permissions = api_get_permissions;
    api.has_permission = api_has_permission;
    api.request_reload = api_request_reload;
    
    // Utilities
    api.allocate_memory = api_allocate_memory;
    api.free_memory = api_free_memory;
    api.string_duplicate = api_string_duplicate;
    api.string_free = api_string_free;
    
    auto* result = &api;
    s_plugin_contexts.emplace(plugin_name, std::move(context));
    return result;
}

void cleanupPluginContext(const std::string& plugin_name) {
    GRIM::MMO::ToolRegistry::instance().unregisterByProvider(plugin_name);

    // Remove all event subscriptions from this plugin
    for (auto it = s_event_subscriptions.begin(); it != s_event_subscriptions.end();) {
        if (it->second.plugin_name == plugin_name) {
            it = s_event_subscriptions.erase(it);
        } else {
            ++it;
        }
    }

    for (auto it = s_timers.begin(); it != s_timers.end();) {
        if (it->second.plugin_name == plugin_name) it = s_timers.erase(it);
        else ++it;
    }
    s_plugin_contexts.erase(plugin_name);
}

void processTimers() {
    auto now = std::chrono::steady_clock::now();
    
    for (auto& [id, timer] : s_timers) {
        if (!timer.active) continue;
        
        if (now >= timer.next_fire) {
            // Fire the callback
            timer.callback(timer.id, timer.user_data);
            
            if (timer.repeat) {
                // Schedule next firing
                timer.next_fire = now + std::chrono::milliseconds(timer.interval_ms);
            } else {
                // One-shot timer, deactivate
                timer.active = false;
            }
        }
    }
    
    // Clean up inactive timers
    for (auto it = s_timers.begin(); it != s_timers.end();) {
        if (!it->second.active) {
            it = s_timers.erase(it);
        } else {
            ++it;
        }
    }
}

void emitGlobalEvent(const char* event_name, const void* data, size_t data_size) {
    if (!event_name) return;
    
    std::string evt_name = event_name;
    
    // Notify all subscribers
    for (const auto& [id, sub] : s_event_subscriptions) {
        if (sub.event_name == evt_name) {
            sub.callback(event_name, data, data_size, sub.user_data);
        }
    }
}

} // namespace PluginAPI
