#pragma once

// ============================================================================
// G.R.I.M Plugin API v1.0
// ============================================================================
// This header defines the stable API interface for hot-reloadable plugins.
// Plugins should only interact with GRIM through this API to ensure forward
// compatibility and system stability.
//
// Design principles:
// - ABI-stable C interface with C++ wrappers
// - Opaque handles instead of direct pointers
// - Version negotiation for compatibility
// - Permission-based access control
// ============================================================================

#include <cstdint>
#include <cstddef>

// ============================================================================
// API Versioning
// ============================================================================
#define GRIM_API_VERSION_MAJOR 1
#define GRIM_API_VERSION_MINOR 0
#define GRIM_API_VERSION_PATCH 0

#define GRIM_API_VERSION ((GRIM_API_VERSION_MAJOR << 16) | (GRIM_API_VERSION_MINOR << 8) | GRIM_API_VERSION_PATCH)

// ============================================================================
// Platform-specific export/import macros
// ============================================================================
#if defined(_WIN32) || defined(_WIN64)
    #ifdef GRIM_BUILD_HOST
        #define GRIM_API __declspec(dllexport)
    #else
        #define GRIM_API __declspec(dllimport)
    #endif
    #define PLUGIN_EXPORT __declspec(dllexport)
#else
    #define GRIM_API __attribute__((visibility("default")))
    #define PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

// ============================================================================
// Core Types & Handles
// ============================================================================

// Opaque handles (never dereference directly in plugins)
typedef void* GrimHandle;
typedef void* GrimEventHandle;
typedef void* GrimMemoryHandle;
typedef void* GrimFileHandle;
typedef void* GrimTimerHandle;

// Result codes
typedef enum {
    GRIM_OK = 0,
    GRIM_ERROR_INVALID_PARAM = 1,
    GRIM_ERROR_NOT_FOUND = 2,
    GRIM_ERROR_PERMISSION_DENIED = 3,
    GRIM_ERROR_ALREADY_EXISTS = 4,
    GRIM_ERROR_OUT_OF_MEMORY = 5,
    GRIM_ERROR_UNSUPPORTED = 6,
    GRIM_ERROR_PLUGIN_NOT_LOADED = 7,
    GRIM_ERROR_API_VERSION_MISMATCH = 8,
    GRIM_ERROR_UNKNOWN = 99
} GrimResult;

// Log levels (matches your logger)
typedef enum {
    GRIM_LOG_DEBUG = 0,
    GRIM_LOG_INFO = 1,
    GRIM_LOG_WARNING = 2,
    GRIM_LOG_ERROR = 3,
    GRIM_LOG_CRITICAL = 4
} GrimLogLevel;

// Plugin permissions
typedef enum {
    GRIM_PERM_FILESYSTEM = (1 << 0),    // Can read/write files
    GRIM_PERM_PROCESS = (1 << 1),       // Can launch processes
    GRIM_PERM_NETWORK = (1 << 2),       // Can use network (still offline-first!)
    GRIM_PERM_MEMORY = (1 << 3),        // Can access memory system
    GRIM_PERM_VOICE = (1 << 4),         // Can use TTS/STT
    GRIM_PERM_UI = (1 << 5),            // Can show UI elements
    GRIM_PERM_SYSTEM = (1 << 6),        // Can query system info
    GRIM_PERM_ALL = 0xFFFFFFFF
} GrimPermission;

// Voice parameters
typedef struct {
    float pitch;          // 0.5 - 2.0
    float speed;          // 0.5 - 2.0
    float volume;         // 0.0 - 1.0
    const char* voice_id; // Speaker ID for multi-voice TTS
} GrimVoiceParams;

// Command result
typedef struct {
    bool success;
    const char* message;
    const char* error_code;
    const char* category;
    uint32_t color_rgb;  // RGB color for UI feedback
} GrimCommandResult;

// Intent structure (matches your Intent struct)
typedef struct {
    const char* name;
    const char* description;
    const char* category;
    bool matched;
    double confidence;
} GrimIntent;

// ============================================================================
// Callback Function Types
// ============================================================================

// Command handler: (user_input, plugin_data) -> CommandResult
typedef GrimCommandResult (*GrimCommandHandler)(const char* input, void* user_data);

// Event callback: (event_name, event_data, user_data)
typedef void (*GrimEventCallback)(const char* event_name, const void* event_data, size_t data_size, void* user_data);

// Timer callback: (timer_id, user_data)
typedef void (*GrimTimerCallback)(uint64_t timer_id, void* user_data);

// ============================================================================
// Plugin API Function Table
// ============================================================================

typedef struct GrimPluginAPI {
    // API version info
    uint32_t api_version;
    
    // ========================================================================
    // Logging Functions
    // ========================================================================
    void (*log)(GrimLogLevel level, const char* message);
    void (*log_fmt)(GrimLogLevel level, const char* format, ...);
    
    // ========================================================================
    // Command Registration
    // ========================================================================
    GrimResult (*register_command)(const char* command_name, GrimCommandHandler handler, void* user_data);
    GrimResult (*unregister_command)(const char* command_name);
    bool (*is_command_registered)(const char* command_name);
    
    // Execute a command programmatically
    GrimCommandResult (*execute_command)(const char* command_text);
    
    // ========================================================================
    // Event System
    // ========================================================================
    GrimEventHandle (*subscribe_event)(const char* event_name, GrimEventCallback callback, void* user_data);
    GrimResult (*unsubscribe_event)(GrimEventHandle handle);
    GrimResult (*emit_event)(const char* event_name, const void* data, size_t data_size);
    
    // Common events:
    // - "command_executed" - fired after any command runs
    // - "user_input" - raw user text input
    // - "plugin_loaded" / "plugin_unloaded"
    // - "system_shutdown" - clean up resources
    
    // ========================================================================
    // Memory & Context API
    // ========================================================================
    const char* (*get_context)(const char* key);
    GrimResult (*set_context)(const char* key, const char* value);
    GrimResult (*delete_context)(const char* key);
    bool (*has_context)(const char* key);
    
    // Long-term memory storage
    GrimResult (*store_memory)(const char* key, const char* value, const char* category);
    const char* (*retrieve_memory)(const char* key);
    GrimResult (*delete_memory)(const char* key);
    
    // ========================================================================
    // Language & AI Functions
    // ========================================================================
    // Deprecated ABI slot: the host returns an unmatched result and performs
    // no local intent classification.
    GrimIntent (*classify_intent)(const char* text);
    const char* (*resolve_synonym)(const char* word);
    
    // Get conversation history (returns JSON array)
    const char* (*get_conversation_history)(int limit);
    
    // ========================================================================
    // Output & UI Functions
    // ========================================================================
    void (*send_response)(const char* text);
    void (*send_response_colored)(const char* text, uint32_t rgb_color);
    void (*show_notification)(const char* title, const char* message);
    
    // Voice synthesis (requires GRIM_PERM_VOICE)
    GrimResult (*queue_tts)(const char* text, const GrimVoiceParams* params);
    GrimResult (*stop_tts)();
    bool (*is_speaking)();
    
    // ========================================================================
    // File System API (requires GRIM_PERM_FILESYSTEM)
    // ========================================================================
    GrimFileHandle (*open_file)(const char* path, const char* mode); // "r", "w", "a", etc.
    GrimResult (*close_file)(GrimFileHandle handle);
    size_t (*read_file)(GrimFileHandle handle, char* buffer, size_t size);
    size_t (*write_file)(GrimFileHandle handle, const char* data, size_t size);
    
    // High-level file operations
    const char* (*read_file_text)(const char* path);  // Returns entire file as string
    GrimResult (*write_file_text)(const char* path, const char* content);
    bool (*file_exists)(const char* path);
    GrimResult (*delete_file)(const char* path);
    
    // ========================================================================
    // Process Management (requires GRIM_PERM_PROCESS)
    // ========================================================================
    GrimResult (*launch_app)(const char* app_name);
    GrimResult (*launch_process)(const char* command, const char* args);
    bool (*is_process_running)(const char* process_name);
    GrimResult (*kill_process)(const char* process_name);
    
    // ========================================================================
    // System Information (requires GRIM_PERM_SYSTEM)
    // ========================================================================
    const char* (*get_system_info)(const char* key);  // "os", "cpu", "memory", "disk", etc.
    uint64_t (*get_memory_usage)();  // Returns bytes
    float (*get_cpu_usage)();        // Returns 0.0-100.0
    
    // ========================================================================
    // Timer & Scheduling
    // ========================================================================
    GrimTimerHandle (*create_timer)(uint64_t interval_ms, bool repeat, GrimTimerCallback callback, void* user_data);
    GrimResult (*cancel_timer)(GrimTimerHandle handle);
    
    // ========================================================================
    // Plugin Lifecycle & Metadata
    // ========================================================================
    const char* (*get_plugin_name)();
    const char* (*get_plugin_path)();
    uint32_t (*get_permissions)();  // Returns bitmask of GrimPermission
    bool (*has_permission)(GrimPermission perm);
    
    // Request plugin reload (useful for development)
    void (*request_reload)();
    
    // ========================================================================
    // Utility Functions
    // ========================================================================
    void* (*allocate_memory)(size_t size);  // Use GRIM's allocator
    void (*free_memory)(void* ptr);
    
    // String utilities (GRIM manages lifetime)
    const char* (*string_duplicate)(const char* str);
    void (*string_free)(const char* str);
    
    // Reserved for future expansion
    void* reserved[32];
    
} GrimPluginAPI;

// ============================================================================
// Plugin Descriptor (exported by plugin)
// ============================================================================

typedef struct GrimPluginInfo {
    uint32_t api_version;           // GRIM_API_VERSION this plugin expects
    const char* name;               // Plugin name
    const char* version;            // Plugin version (e.g., "1.0.0")
    const char* author;             // Plugin author
    const char* description;        // Brief description
    uint32_t required_permissions;  // Bitmask of required GrimPermission flags
} GrimPluginInfo;

// ============================================================================
// Plugin Entry Points (plugins must export these)
// ============================================================================

extern "C" {
    // Get plugin metadata
    PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info();
    
    // Initialize plugin (called on load)
    // Returns GRIM_OK on success
    PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api);
    
    // Shutdown plugin (called on unload)
    PLUGIN_EXPORT void grim_plugin_shutdown();
    
    // Optional: called when plugin is reloaded
    PLUGIN_EXPORT void grim_plugin_reload();
}

// ============================================================================
// C++ Helper Wrapper (optional convenience layer)
// ============================================================================

#ifdef __cplusplus
namespace grim {

class PluginAPI {
private:
    const GrimPluginAPI* m_api;
    
public:
    explicit PluginAPI(const GrimPluginAPI* api) : m_api(api) {}
    
    // Logging
    void log(GrimLogLevel level, const char* message) { m_api->log(level, message); }
    void debug(const char* msg) { log(GRIM_LOG_DEBUG, msg); }
    void info(const char* msg) { log(GRIM_LOG_INFO, msg); }
    void warning(const char* msg) { log(GRIM_LOG_WARNING, msg); }
    void error(const char* msg) { log(GRIM_LOG_ERROR, msg); }
    
    // Commands
    GrimResult registerCommand(const char* name, GrimCommandHandler handler, void* data = nullptr) {
        return m_api->register_command(name, handler, data);
    }
    
    GrimResult unregisterCommand(const char* name) {
        return m_api->unregister_command(name);
    }
    
    // Events
    GrimEventHandle subscribeEvent(const char* name, GrimEventCallback cb, void* data = nullptr) {
        return m_api->subscribe_event(name, cb, data);
    }
    
    GrimResult emitEvent(const char* name, const void* data = nullptr, size_t size = 0) {
        return m_api->emit_event(name, data, size);
    }
    
    // Context
    const char* getContext(const char* key) { return m_api->get_context(key); }
    GrimResult setContext(const char* key, const char* value) { return m_api->set_context(key, value); }
    
    // Output
    void sendResponse(const char* text) { m_api->send_response(text); }
    void showNotification(const char* title, const char* msg) { m_api->show_notification(title, msg); }
    
    // Voice
    GrimResult queueTTS(const char* text, const GrimVoiceParams* params = nullptr) {
        return m_api->queue_tts(text, params);
    }
    
    // File operations
    const char* readFile(const char* path) { return m_api->read_file_text(path); }
    GrimResult writeFile(const char* path, const char* content) { return m_api->write_file_text(path, content); }
    bool fileExists(const char* path) { return m_api->file_exists(path); }
    
    // System info
    const char* getSystemInfo(const char* key) { return m_api->get_system_info(key); }
    
    // Permissions
    bool hasPermission(GrimPermission perm) { return m_api->has_permission(perm); }
    
    // Raw API access
    const GrimPluginAPI* raw() const { return m_api; }
};

} // namespace grim
#endif // __cplusplus

// ============================================================================
// Example Plugin Implementation
// ============================================================================
/*

// example_plugin.cpp
#include "core/plugin_api.hpp"

static const GrimPluginAPI* g_api = nullptr;

// Command handler
static GrimCommandResult my_command_handler(const char* input, void* user_data) {
    g_api->log(GRIM_LOG_INFO, "My plugin command executed!");
    
    GrimCommandResult result;
    result.success = true;
    result.message = "Command executed successfully";
    result.error_code = nullptr;
    result.category = "custom";
    result.color_rgb = 0x00FF00; // Green
    return result;
}

// Plugin metadata
extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    static GrimPluginInfo info;
    info.api_version = GRIM_API_VERSION;
    info.name = "Example Plugin";
    info.version = "1.0.0";
    info.author = "Your Name";
    info.description = "An example plugin demonstrating the API";
    info.required_permissions = GRIM_PERM_UI | GRIM_PERM_FILESYSTEM;
    return &info;
}

// Initialize
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (api->api_version < GRIM_API_VERSION) {
        return GRIM_ERROR_API_VERSION_MISMATCH;
    }
    
    g_api = api;
    g_api->log(GRIM_LOG_INFO, "Example plugin initializing...");
    
    // Register a command
    g_api->register_command("example", my_command_handler, nullptr);
    
    return GRIM_OK;
}

// Shutdown
extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {
    if (g_api) {
        g_api->log(GRIM_LOG_INFO, "Example plugin shutting down...");
        g_api->unregister_command("example");
    }
}
*/
