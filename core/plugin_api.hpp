#pragma once

// ============================================================================
// G.R.I.M Plugin API v27.0
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
#define GRIM_API_VERSION_MAJOR 27
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

// Arguments are borrowed for the duration of the dispatch call.
struct GrimToolCall {
    uint32_t struct_size = sizeof(GrimToolCall);
    const char* arguments = nullptr;
    size_t arguments_size = 0;
    // Optional request for a final result from providers with asynchronous operations.
    uint32_t wait_for_completion = 0;
};
// A successful dispatch returns one provider-owned, UTF-8 JSON result.
// The receiver calls release(json) once while the provider is pinned.
// A dispatch failure returns a non-OK code and no buffer.
struct GrimToolReply {
    GrimResult delivery = GRIM_ERROR_UNKNOWN;
    const char* json = nullptr;
    size_t size = 0;
    void (*release)(const char*) noexcept = nullptr;
};
using GrimToolDispatch = GrimToolReply (*)(const char* tool_id, const GrimToolCall*);

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
    
    // Return the registry result as host-owned JSON. Release with string_free.
    // Null indicates an API failure.
    const char* (*execute_command)(GrimPluginHandle instance, const char* command_text);
    
    // ========================================================================
    // Notifications
    // ========================================================================
    void (*show_notification)(GrimPluginHandle instance, const char* title, const char* message);
    
    // ========================================================================
    // Plugin Lifecycle & Metadata
    // ========================================================================
    const char* (*get_plugin_name)(GrimPluginHandle instance);
    const char* (*get_plugin_path)(GrimPluginHandle instance);
    
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
    // Copy an entire resource kind atomically. JSON array <=64 MiB, <=10000 entries.
    // Each entry: resource_id, kind, name, runtime_id (>0), state (object).
    // Initialization calls are serialized on the lifecycle thread (no init workers).
    // The provider must have a live registry record.
    // Empty array removes the kind. Ownership is derived from the instance handle.
    GrimResult (*replace_resources)(GrimPluginHandle instance, const char* kind,
                                   const char* resources_json, size_t size);
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
