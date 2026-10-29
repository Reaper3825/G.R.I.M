#pragma once

// ============================================================================
// G.R.I.M Plugin API v3.3
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
#define GRIM_API_VERSION_MAJOR 3
#define GRIM_API_VERSION_MINOR 3
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

// All pointers are borrowed for the duration of the API call. The host copies
// registration metadata and result data before returning. No STL crosses the ABI.
struct GrimStringList { const char* const* items = nullptr; size_t count = 0; };
struct GrimToolParameter {
    uint32_t struct_size = sizeof(GrimToolParameter);
    const char* name = nullptr;
    const char* type = nullptr;
    const char* description = nullptr;
    uint32_t required = 0;
};
struct GrimToolDescriptor {
    uint32_t struct_size = sizeof(GrimToolDescriptor);
    const char* tool_id = nullptr;
    const char* display_name = nullptr;
    const char* version = nullptr;
    const char* description = nullptr;
    const char* usage = nullptr;
    const char* category = nullptr;
    // raw_text or json. JSON tools require an object-valued JSON Schema.
    const char* argument_format = nullptr;
    const char* argument_schema_json = nullptr;
    const GrimToolParameter* parameters = nullptr;
    size_t parameter_count = 0;
    GrimStringList examples, aliases, keywords, capability_tags;
    GrimStringList preconditions, context_requirements, affordance_tokens;
    uint32_t required_permissions = 0;
    uint32_t is_informational = 0;
    uint32_t needs_confirmation = 0;
};
struct GrimToolCall {
    uint32_t struct_size = sizeof(GrimToolCall);
    const char* arguments = nullptr;
    size_t arguments_size = 0;
};
struct GrimToolResult {
    uint32_t struct_size = sizeof(GrimToolResult);
    uint32_t success = 0;
    const char* message = nullptr;
    const char* error_code = nullptr;
    const char* category = nullptr;
    const char* voice = nullptr;
    const char* structured_output_json = nullptr;
    uint32_t color_rgb = 0;
};
// Complete exactly once, synchronously. complete() copies every field immediately.
// The writer and call cannot be retained or used by background threads.
struct GrimToolResultWriter {
    uint32_t struct_size = sizeof(GrimToolResultWriter);
    void* host_context = nullptr;
    GrimResult (*complete)(GrimToolResultWriter*, const GrimToolResult*) = nullptr;
};
typedef GrimResult (*GrimToolHandler)(const GrimToolCall*, GrimToolResultWriter*, void* user_data);
struct GrimToolRegistration {
    uint32_t struct_size = sizeof(GrimToolRegistration);
    GrimToolDescriptor descriptor;
    GrimToolHandler handler = nullptr;
    void* user_data = nullptr; // Plugin-owned until shutdown; not freed by the host.
};

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

// Tool handlers and their results are defined above.

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

    // Host-owned identity for this loaded plugin. Pass this API object as self
    // on every call; valid until plugin shutdown and host cleanup complete.
    const void* host_context;
    
    // ========================================================================
    // Logging Functions
    // ========================================================================
    void (*log)(const GrimPluginAPI* self, GrimLogLevel level, const char* message);
    void (*log_fmt)(const GrimPluginAPI* self, GrimLogLevel level, const char* format, ...);
    
    // ========================================================================
    // Command Registration
    // ========================================================================
    // Initialization batches are staged until the loader commits a successful init.
    GrimResult (*register_tools)(const GrimPluginAPI* self, const GrimToolRegistration* tools, size_t count);
    GrimResult (*unregister_command)(const GrimPluginAPI* self, const char* command_name);
    bool (*is_command_registered)(const GrimPluginAPI* self, const char* command_name);
    
    // Execute a command programmatically
    GrimResult (*execute_command)(const GrimPluginAPI* self, const char* command_text, GrimToolResultWriter* result);
    
    // ========================================================================
    // Event System
    // ========================================================================
    GrimEventHandle (*subscribe_event)(const GrimPluginAPI* self, const char* event_name, GrimEventCallback callback, void* user_data);
    GrimResult (*unsubscribe_event)(const GrimPluginAPI* self, GrimEventHandle handle);
    GrimResult (*emit_event)(const GrimPluginAPI* self, const char* event_name, const void* data, size_t data_size);
    
    // Common events:
    // - "command_executed" - fired after any command runs
    // - "user_input" - raw user text input
    // - "plugin_loaded" / "plugin_unloaded"
    // - "system_shutdown" - clean up resources
    
    // ========================================================================
    // Memory & Context API
    // ========================================================================
    const char* (*get_context)(const GrimPluginAPI* self, const char* key);
    GrimResult (*set_context)(const GrimPluginAPI* self, const char* key, const char* value);
    GrimResult (*delete_context)(const GrimPluginAPI* self, const char* key);
    bool (*has_context)(const GrimPluginAPI* self, const char* key);
    
    // Long-term memory storage
    GrimResult (*store_memory)(const GrimPluginAPI* self, const char* key, const char* value, const char* category);
    const char* (*retrieve_memory)(const GrimPluginAPI* self, const char* key);
    GrimResult (*delete_memory)(const GrimPluginAPI* self, const char* key);
    
    // ========================================================================
    // Language & AI Functions
    // ========================================================================
    // Deprecated ABI slot: the host returns an unmatched result and performs
    // no local intent classification.
    GrimIntent (*classify_intent)(const GrimPluginAPI* self, const char* text);
    const char* (*resolve_synonym)(const GrimPluginAPI* self, const char* word);
    
    // Get conversation history (returns JSON array)
    const char* (*get_conversation_history)(const GrimPluginAPI* self, int limit);
    
    // ========================================================================
    // Output & UI Functions
    // ========================================================================
    void (*send_response)(const GrimPluginAPI* self, const char* text);
    void (*send_response_colored)(const GrimPluginAPI* self, const char* text, uint32_t rgb_color);
    void (*show_notification)(const GrimPluginAPI* self, const char* title, const char* message);
    
    // Voice synthesis (requires GRIM_PERM_VOICE)
    GrimResult (*queue_tts)(const GrimPluginAPI* self, const char* text, const GrimVoiceParams* params);
    GrimResult (*stop_tts)(const GrimPluginAPI* self);
    bool (*is_speaking)(const GrimPluginAPI* self);
    
    // ========================================================================
    // File System API (requires GRIM_PERM_FILESYSTEM)
    // ========================================================================
    GrimFileHandle (*open_file)(const GrimPluginAPI* self, const char* path, const char* mode); // "r", "w", "a", etc.
    GrimResult (*close_file)(const GrimPluginAPI* self, GrimFileHandle handle);
    size_t (*read_file)(const GrimPluginAPI* self, GrimFileHandle handle, char* buffer, size_t size);
    size_t (*write_file)(const GrimPluginAPI* self, GrimFileHandle handle, const char* data, size_t size);
    
    // High-level file operations
    const char* (*read_file_text)(const GrimPluginAPI* self, const char* path);  // Returns entire file as string
    GrimResult (*write_file_text)(const GrimPluginAPI* self, const char* path, const char* content);
    bool (*file_exists)(const GrimPluginAPI* self, const char* path);
    GrimResult (*delete_file)(const GrimPluginAPI* self, const char* path);
    
    // ========================================================================
    // Process Management (requires GRIM_PERM_PROCESS)
    // ========================================================================
    GrimResult (*launch_app)(const GrimPluginAPI* self, const char* app_name);
    GrimResult (*launch_process)(const GrimPluginAPI* self, const char* command, const char* args);
    bool (*is_process_running)(const GrimPluginAPI* self, const char* process_name);
    GrimResult (*kill_process)(const GrimPluginAPI* self, const char* process_name);
    
    // ========================================================================
    // System Information (requires GRIM_PERM_SYSTEM)
    // ========================================================================
    const char* (*get_system_info)(const GrimPluginAPI* self, const char* key);  // "os", "cpu", "memory", "disk", etc.
    uint64_t (*get_memory_usage)(const GrimPluginAPI* self);  // Returns bytes
    float (*get_cpu_usage)(const GrimPluginAPI* self);        // Returns 0.0-100.0
    
    // ========================================================================
    // Timer & Scheduling
    // ========================================================================
    GrimTimerHandle (*create_timer)(const GrimPluginAPI* self, uint64_t interval_ms, bool repeat, GrimTimerCallback callback, void* user_data);
    GrimResult (*cancel_timer)(const GrimPluginAPI* self, GrimTimerHandle handle);
    
    // ========================================================================
    // Plugin Lifecycle & Metadata
    // ========================================================================
    const char* (*get_plugin_name)(const GrimPluginAPI* self);
    const char* (*get_plugin_path)(const GrimPluginAPI* self);
    uint32_t (*get_permissions)(const GrimPluginAPI* self);  // Returns bitmask of GrimPermission
    bool (*has_permission)(const GrimPluginAPI* self, GrimPermission perm);
    
    // Request plugin reload (useful for development)
    void (*request_reload)(const GrimPluginAPI* self);
    
    // ========================================================================
    // Utility Functions
    // ========================================================================
    void* (*allocate_memory)(const GrimPluginAPI* self, size_t size);  // Use GRIM's allocator
    void (*free_memory)(const GrimPluginAPI* self, void* ptr);
    
    // String utilities (GRIM manages lifetime)
    const char* (*string_duplicate)(const GrimPluginAPI* self, const char* str);
    void (*string_free)(const GrimPluginAPI* self, const char* str);
    
    // v3.1: consumes two reserved slots; earlier fields and table size unchanged.
    // Returns a host-owned string; release with string_free. Null on failure.
    const char* (*get_user_error_message)(const GrimPluginAPI* self, const char* code);
    // Requires PROCESS. On Windows, returns the ShellExecute status unchanged;
    // native_status > 32 means success. GRIM_OK means a native status was obtained.
    // Other platforms return GRIM_ERROR_UNSUPPORTED. No command dispatch occurs.
    GrimResult (*open_target)(const GrimPluginAPI* self, const char* target, int64_t* native_status);
    // v3.2: host-owned JSON metadata snapshot, released with string_free.
    // Null tool_id returns all tools in registry order; otherwise resolves the
    // canonical ID, display name, or alias. Missing tools produce an empty array.
    // Envelope: {"format":"grim-tool-snapshot","version":1,"tools":[...]}
    // Null return means API failure. Read-only; does not dispatch or change stats.
    const char* (*get_tool_snapshot)(const GrimPluginAPI* self, const char* tool_id);
    // v3.3: application aliases (separate from ToolRegistry command aliases).
    // Host-owned version-1 JSON; release with string_free. Null return = API failure.
    // Null key lists entries in the host's existing iteration order; otherwise
    // returns the host's alias-info text, including existing missing-name behavior.
    const char* (*get_alias_snapshot)(const GrimPluginAPI* self, const char* key);
    // Requires FILESYSTEM. Calls the existing synchronous host refresh. The JSON
    // records normal return vs exception; legacy commands ignore returned status.
    const char* (*refresh_aliases)(const GrimPluginAPI* self);
    void* reserved[27];
    
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
    
    // Initialize plugin (called on load). Tools stay staged until successful return.
    // Shutdown is also called after failed init; tolerate partial initialization.
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
    void log(GrimLogLevel level, const char* message) { m_api->log(m_api, level, message); }
    void debug(const char* msg) { log(GRIM_LOG_DEBUG, msg); }
    void info(const char* msg) { log(GRIM_LOG_INFO, msg); }
    void warning(const char* msg) { log(GRIM_LOG_WARNING, msg); }
    void error(const char* msg) { log(GRIM_LOG_ERROR, msg); }
    
    // Commands
    GrimResult registerTools(const GrimToolRegistration* tools, size_t count) {
        return m_api->register_tools(m_api, tools, count);
    }

    GrimResult unregisterCommand(const char* name) {
        return m_api->unregister_command(m_api, name);
    }
    
    // Events
    GrimEventHandle subscribeEvent(const char* name, GrimEventCallback cb, void* data = nullptr) {
        return m_api->subscribe_event(m_api, name, cb, data);
    }
    
    GrimResult emitEvent(const char* name, const void* data = nullptr, size_t size = 0) {
        return m_api->emit_event(m_api, name, data, size);
    }
    
    // Context
    const char* getContext(const char* key) { return m_api->get_context(m_api, key); }
    GrimResult setContext(const char* key, const char* value) { return m_api->set_context(m_api, key, value); }
    
    // Output
    void sendResponse(const char* text) { m_api->send_response(m_api, text); }
    void showNotification(const char* title, const char* msg) { m_api->show_notification(m_api, title, msg); }
    
    // Voice
    GrimResult queueTTS(const char* text, const GrimVoiceParams* params = nullptr) {
        return m_api->queue_tts(m_api, text, params);
    }
    
    // File operations
    const char* readFile(const char* path) { return m_api->read_file_text(m_api, path); }
    GrimResult writeFile(const char* path, const char* content) { return m_api->write_file_text(m_api, path, content); }
    bool fileExists(const char* path) { return m_api->file_exists(m_api, path); }
    
    // System info
    const char* getSystemInfo(const char* key) { return m_api->get_system_info(m_api, key); }
    
    // Permissions
    bool hasPermission(GrimPermission perm) { return m_api->has_permission(m_api, perm); }
    
    // Raw API access
    const GrimPluginAPI* raw() const { return m_api; }
};

} // namespace grim
#endif // __cplusplus
