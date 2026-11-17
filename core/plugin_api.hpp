#pragma once

// ============================================================================
// G.R.I.M Plugin API v19.0
// ============================================================================
// This header defines the stable API interface for hot-reloadable plugins.
// Service plugins use this C table. Native UI plugins additionally import the
// exported C++ UI classes and require a matching host toolchain and class ABI.
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
#define GRIM_API_VERSION_MAJOR 19
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
    GRIM_PERM_INPUT = (1 << 7),         // Can send mouse/keyboard input through host services
    GRIM_PERM_ALL = 0xFFFFFFFF
} GrimPermission;


// All pointers are borrowed for the duration of the API call. The host copies
// result data before returning. No STL crosses the ABI.
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
using GrimToolDispatch = GrimResult (*)(const char* tool_id, const GrimToolCall*, GrimToolResultWriter*);

enum GrimUIPanelOperation : uint32_t {
    GRIM_UI_LIST = 0, GRIM_UI_INSPECT = 1, GRIM_UI_SHOW = 2,
    GRIM_UI_HIDE = 3, GRIM_UI_BRING_TO_FRONT = 4, GRIM_UI_POLL = 5,
    // v3.13: request_id carries the observed runtime_id; check on UI thread before showing.
    GRIM_UI_SHOW_CURRENT = 6,
    GRIM_UI_CANCEL = 7, // Cancel/release one owned request; an already applied effect is not undone.
    GRIM_UI_HIDE_CURRENT = 8 // v3.14: guarded hide, with the same arguments as SHOW_CURRENT.
};


// ============================================================================
// Callback Function Types
// ============================================================================

// Tool handlers and their results are defined above.

// Event callback: (event_name, event_data, user_data)
typedef void (*GrimEventCallback)(const char* event_name, const void* event_data, size_t data_size, void* user_data);

// System tick callback
typedef void (*GrimSystemTickCallback)(double delta_seconds, double offset_seconds, void* user_data);

// ============================================================================
// Plugin API Function Table
// ============================================================================

using GrimPluginHandle = uint64_t;

typedef struct GrimPluginAPI {
    // API version info
    uint32_t api_version;

    // ========================================================================
    // Logging Functions
    // ========================================================================
    void (*log)(GrimPluginHandle instance, GrimLogLevel level, const char* message);
    
    // ========================================================================
    // Command Registration
    // ========================================================================
    // Remove a tool owned by the current provider.
    GrimResult (*unregister_command)(GrimPluginHandle instance, const char* command_name);
    
    // Execute a command programmatically
    GrimResult (*execute_command)(GrimPluginHandle instance, const char* command_text, GrimToolResultWriter* result);
    
    // ========================================================================
    // Event System
    // ========================================================================
    GrimEventHandle (*subscribe_event)(GrimPluginHandle instance, const char* event_name, GrimEventCallback callback, void* user_data);
    GrimResult (*unsubscribe_event)(GrimPluginHandle instance, GrimEventHandle handle);
    GrimResult (*emit_event)(GrimPluginHandle instance, const char* event_name, const void* data, size_t data_size);
    
    // Common events:
    // - "command_executed" - fired after any command runs
    // - "user_input" - raw user text input
    // - "plugin_loaded" / "plugin_unloaded"
    // - "system_shutdown" - clean up resources
    
    // ========================================================================
    // Notifications
    // ========================================================================
    void (*show_notification)(GrimPluginHandle instance, const char* title, const char* message);
    
    // ========================================================================
    // Plugin Lifecycle & Metadata
    // ========================================================================
    const char* (*get_plugin_name)(GrimPluginHandle instance);
    const char* (*get_plugin_path)(GrimPluginHandle instance);
    uint32_t (*get_permissions)(GrimPluginHandle instance);  // Returns bitmask of GrimPermission
    bool (*has_permission)(GrimPluginHandle instance, GrimPermission perm);
    
    // Request plugin reload (useful for development)
    void (*request_reload)(GrimPluginHandle instance);
    
    // ========================================================================
    // Utility Functions
    // ========================================================================
    // Releases a host-owned string.
    void (*string_free)(GrimPluginHandle instance, const char* str);
    
    // Returns a host-owned string; release with string_free. Null on failure.
    const char* (*get_user_error_message)(GrimPluginHandle instance, const char* code);
    // v3.2: host-owned JSON metadata snapshot, released with string_free.
    // Null tool_id returns all tools in registry order; otherwise resolves the
    // canonical ID, display name, or alias. Missing tools produce an empty array.
    // Envelope: {"format":"grim-tool-snapshot","version":1,"tools":[...]}
    // Null return means API failure. Read-only; does not dispatch or change stats.
    const char* (*get_tool_snapshot)(GrimPluginHandle instance, const char* tool_id);
    // Manual memory services
    // (MEMORY). JSON envelope: {format:"grim-runtime-service",version:1,
    // value:...,error:null|{kind,message}}; release with string_free.
    // Null = invalid input, denied permission, or API failure. No command dispatch.
    const char* (*stage_manual_memory)(GrimPluginHandle instance, const char* raw);
    // value is an ordered array of {id,raw}; uses the existing search defaults.
    const char* (*search_memory)(GrimPluginHandle instance, const char* query);
    // value is presence bool. Retains legacy forget's check+flush, not deletion.
    const char* (*flush_memory_record_if_present)(GrimPluginHandle instance, uint64_t id);
    // v3.8: UI permission. Submits copied, stable-key panel requests or polls a
    // caller-owned request, not ownership of the target panel. Always nonblocking.
    // Host-owned grim-ui-panel-response v1
    // JSON; free with string_free. Null = invalid input, denied access, or failure.
    // Submit: request_id=0; panel_key=null for LIST, required for other operations.
    // Poll: panel_key=null, request_id>0. A completed poll consumes its result.
    // v3.13: SHOW_CURRENT uses panel_key + request_id=observed runtime identity.
    // v3.14: HIDE_CURRENT uses the same guarded identity for hiding a panel.
    // CANCEL uses panel_key=null + owned request_id; releases/cancels queued work,
    // but cannot undo an effect already applied. Table/descriptor layouts unchanged.
    const char* (*ui_panel_service)(GrimPluginHandle instance, GrimUIPanelOperation operation,
                                    const char* panel_key, uint64_t request_id);
    // v5.0: Generic provider identity (JSON name/instance/path); string_free releases it.
    const char* (*get_provider_identity)(GrimPluginHandle instance);
    // Copy an entire resource kind atomically. JSON array <=64 MiB, <=10000 entries.
    // Each entry: resource_id, kind, name, runtime_id (>0), state (object).
    // Initialization calls are serialized on the lifecycle thread (no init workers).
    // The provider must have a live registry record.
    // Empty array removes the kind. Ownership is derived from the instance handle.
    GrimResult (*replace_resources)(GrimPluginHandle instance, const char* kind,
                                   const char* resources_json, size_t size);
    // True only while this generation is the currently published provider.
    bool (*is_provider_current)(GrimPluginHandle instance);
    // v14 removes the copied-widget transport. Native panels use the exported
    // C++ UI interface and register during the owning provider's init scope.
    // Native UI plugins must match the host compiler, CRT and UI class ABI.
    void* reserved[8];

    // v15.2: One callback per generation, on the application frame thread.
    // Delta is time since the previous frame start. Offset is time since the first.
    // Both are zero on the first application frame. The host detaches the
    // callback at shutdown. Keep user_data valid until then.
    GrimResult (*subscribe_system_tick)(GrimPluginHandle instance,
                                       GrimSystemTickCallback callback, void* user_data);
    
} GrimPluginAPI;

// ============================================================================
// Plugin Entry Points (plugins must export these)
// ============================================================================

extern "C" {
    // Initialize plugin (called on load). Tool dispatch starts after successful return.
    // Shutdown is also called after failed init; tolerate partial initialization.
    // Returns GRIM_OK on success
    PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api, GrimPluginHandle instance);
    
    // Shutdown plugin (called on unload)
    PLUGIN_EXPORT void grim_plugin_shutdown();
    
}

// ============================================================================
// C++ Helper Wrapper (optional convenience layer)
// ============================================================================

#ifdef __cplusplus
namespace grim {

class PluginHost {
public:
    PluginHost() = default;
    PluginHost(const GrimPluginAPI* api, GrimPluginHandle instance) : api_(api), instance_(instance) {}
    const GrimPluginAPI* operator->() const { return api_; }
    explicit operator bool() const { return api_ != nullptr; }
    GrimPluginHandle instance() const { return instance_; }

private:
    const GrimPluginAPI* api_ = nullptr;
    GrimPluginHandle instance_ = 0;
};

} // namespace grim
#endif // __cplusplus
