# G.R.I.M Plugin API Documentation

**Version**: 2.0.0

**Last Updated**: October 28, 2026

**API Version**: 2.0.0

**Implementation Status**: Beta (Core features functional, some advanced features stubbed)

---

## Table of Contents

1. [Overview](#overview)
2. [Implementation Status](#implementation-status)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
5. [Permissions System](#permissions-system)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Overview

### Explicit plugin ownership (API v2)

There is no active/current plugin global. The loader creates one host-owned
`GrimPluginAPI` table per loaded plugin. Its opaque `host_context` identifies
that plugin's name, path, and permissions. Plugins retain the table received by
`grim_plugin_init` and pass it as the first argument to every host function:

```cpp
api->register_command(api, "my_command", myCommandHandler, user_data);
api->log(api, GRIM_LOG_INFO, "message");
if (api->has_permission(api, GRIM_PERM_FILESYSTEM)) { /* ... */ }
```

A command, event, or timer callback uses its provider's retained API pointer;
callbacks do not select or change any active plugin. The C++ `grim::PluginAPI`
wrapper supplies that pointer automatically. Handler signatures and `user_data`
remain unchanged. Tool execution still occurs through explicit command dispatch.
`MMO::ToolRegistry` is the single store for both tool metadata and callable
handlers. `commandMap`, the API's `s_commands`, and the old `CommandRegistry`
staging store have been removed. Built-ins and plugin commands register directly;
bootstrap can enrich metadata with `updateMetadata` without replacing handlers.

The table remains valid through `grim_plugin_shutdown`; host cleanup then removes
owned commands, subscriptions, timers, and the API instance. It must not be used
after cleanup. Plugins must stop their own threads before shutdown returns. The
API remains serialized by the caller; this change does not add thread safety or
in-flight-call pinning for hot reload. Native libraries execute in the host
process; permission checks on host API calls are not a process sandbox.

Commands cannot replace existing command/registry names. Unregistering commands,
unsubscribing events, and cancelling timers require the registering owner. Failed
initialization removes owned API registrations before unloading the library.

**ABI migration:** API v2 changes function signatures and table layout. The loader
rejects v1 binaries and newer unsupported versions before initialization. Rebuild
plugins against `core/plugin_api.hpp`, pass the API pointer at each host call,
and check the API major version. No v1 compatibility shim uses ambient identity.
Existing plugin DLLs on disk are not rebuilt automatically.


Current lifecycle integration: application teardown stops wake input and calls
`PluginManager::unloadAll()` before destroying host services. Plugin cleanup uses
the declared plugin name recorded at load, matching registration identity.
The v2 API table intentionally breaks v1 binary compatibility. Registration,
runtime queries, aliases, dispatch, usage statistics, and provider removal use
MMO `ToolRegistry`. Legacy host registration exports are stateless adapters.
Automatic hot-reload polling and plugin timer dispatch are not currently wired
into the application loop; the historical feature tables below are not proof of
that integration.

Focused host-only loader, API ownership, and unified-registry regression tests:
`powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_plugin_shutdown.ps1`.
This also compiles the real API implementation with service sinks and checks
identity, permissions, later registration, callbacks, ownership, and cleanup
with two plugins present. The registry tests use the real command-dispatch path
and cover aliases, duplicate rejection, batch rollback, metadata enrichment,
execution status, statistics, and provider removal. No GRIM application is built
or launched. The existing Tools viewer regression is available separately through
`tests/run_ui_input_foundation.ps1 -Tools`.

Host-side registration requires a `ToolDescriptor` and a `ToolHandler` in the same
call. Descriptors live in `MMO/Core/ToolDescriptor.hpp`, independent of registry
storage. Missing handlers/providers/IDs reject registration; duplicate names or
aliases return false. Explicit replacement must retain provider identity and
provider type. Batch validation commits all entries or none. Metadata updates
preserve handlers and usage statistics. Aliases resolve to one canonical entry
for both discovery and execution.

`dispatchCommand` delegates to `ToolRegistry::invoke`, which runs the selected
handler outside the registry mutex and records the result once. Unknown and
unavailable tools fail without invoking a handler. Invocation does not infer
intent or replace the caller's argument/policy validation. Registry entries are
retained during execution, but this does not yet pin native DLL lifetime against
concurrent loader shutdown. Full plugin-supplied metadata registration and result
ownership remain separate API work.

The GRIM Plugin API enables dynamic extension of GRIM's functionality through hot-reloadable DLL plugins. Plugins can:

✅ **Register custom commands**  
✅ **Listen to system events**  
✅ **Access NLP/AI systems**  
✅ **Integrate with voice & UI**  
✅ **Query system information**  
✅ **Use GRIM's logging infrastructure**  

### Architecture

- **Host-side**: `GRIM.exe` exposes functions via `GrimPluginAPI` table
- **Plugin-side**: DLLs implement required entry points
- **Hot-reload**: Plugins can be reloaded without restarting GRIM
- **Permission-based**: Host API access checks via permission flags

---

## Implementation Status

### ✅ **Fully Implemented**

| Category | Function | Status |
|----------|----------|--------|
| **Logging** | `log()`, `log_fmt()` | ✅ Working |
| **Commands** | `register_command()`, `unregister_command()`, `execute_command()` | ✅ Working |
| **Events** | `subscribe_event()`, `unsubscribe_event()`, `emit_event()` | ✅ Working |
| **Context** | `get_context()`, `set_context()`, `delete_context()`, `has_context()` | ✅ Working |
| **NLP** | `classify_intent()`, `resolve_synonym()` | ✅ Working |
| **Output** | `send_response()`, `send_response_colored()`, `show_notification()` | ✅ Working |
| **Voice** | `speak()`, `speak_with_params()`, `is_speaking()` | ✅ Working |
| **System Info** | `get_system_info()` (os, cpu) | ✅ Working |
| **Timers** | `create_timer()`, `cancel_timer()` | ✅ Working |
| **Lifecycle** | `get_plugin_name()`, `get_plugin_path()`, `has_permission()`, `request_reload()` | ✅ Working |
| **Memory Utils** | `allocate_memory()`, `free_memory()` | ✅ Working |

### ⚠️ **Partially Implemented / Stubbed**

| Function | Status | Notes |
|----------|--------|-------|
| `store_memory()` | 🟡 Stubbed | Requires `MemoryObject` adaptation |
| `retrieve_memory()` | 🟡 Stubbed | Requires `MemoryObject` adaptation |
| `delete_memory()` | 🟡 Stubbed | Requires `MemoryObject` adaptation |
| `stop_audio()` | 🟡 Stubbed | Voice system doesn't have stop function yet |

### 📝 **Implementation Notes**

**Synonym Resolution**: Now uses GRIM's existing `synonyms.json` system via `normalizeWord()`. Converts words to their canonical forms (e.g., `"launch"` → `"open_app"`).

**Memory System**: The memory API expects simple key-value pairs, but GRIM's `MemoryStorage` uses structured `MemoryObject` types. This requires bridging logic that's not yet implemented.

**Voice Control**: The `stop_audio()` function is stubbed because the Voice namespace doesn't expose an audio stop mechanism yet.

---

## Quick Start

### 1. Create Plugin Structure

```cpp
// my_plugin.cpp
#include "core/plugin_api.hpp"
#include <string>

// Global API pointer
static const GrimPluginAPI* g_api = nullptr;

// Plugin metadata
extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    static GrimPluginInfo info = {
        .api_version = GRIM_API_VERSION,
      .name = "My Awesome Plugin",
      .version = "1.0.0",
        .author = "Your Name",
        .description = "Does awesome things!",
        .required_permissions = GRIM_PERM_UI | GRIM_PERM_VOICE
    };
    return &info;
}

// Command handler
static GrimCommandResult myCommandHandler(const char* input, void* user_data) {
    g_api->log(g_api, GRIM_LOG_INFO, "My command executed!");
    g_api->queue_tts(g_api, "Hello from my plugin!", nullptr);
    
    return {
        .success = true,
      .message = "Command executed successfully",
        .error_code = "OK",
     .category = "plugin",
        .color_rgb = 0x00FF00  // Green
    };
}

// Plugin initialization
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    g_api = api;
    
    // Check API version
    if ((api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION) {
        return GRIM_ERROR_API_VERSION_MISMATCH;
    }
    
  // Register command
    GrimResult res = g_api->register_command(g_api, "my_command", myCommandHandler, nullptr);
    if (res != GRIM_OK) {
   g_api->log(g_api, GRIM_LOG_ERROR, "Failed to register command");
      return res;
    }
    
    g_api->log(g_api, GRIM_LOG_INFO, "My Plugin initialized successfully!");
    return GRIM_OK;
}

// Plugin shutdown
extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {
    g_api->unregister_command(g_api, "my_command");
    g_api->log(g_api, GRIM_LOG_INFO, "My Plugin shut down");
}

// Optional: Hot-reload support
extern "C" PLUGIN_EXPORT void grim_plugin_reload() {
  g_api->log(g_api, GRIM_LOG_INFO, "My Plugin reloaded");
}
```

### 2. Build Configuration (CMake)

```cmake
add_library(my_plugin SHARED my_plugin.cpp)

target_include_directories(my_plugin PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core
)

target_link_libraries(my_plugin PRIVATE GRIM_EXPORT)

set_target_properties(my_plugin PROPERTIES
    OUTPUT_NAME "my_plugin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/plugins"
)
```

### 3. Load Plugin

Place your `my_plugin.dll` in the `plugins/` directory. GRIM will auto-discover and load it on startup.

---

## API Reference

### Logging Functions

```cpp
// Simple log message
api->log(api, GRIM_LOG_INFO, "Plugin loaded");

// Formatted log message
api->log_fmt(api, GRIM_LOG_DEBUG, "Value: %d, String: %s", 42, "test");
```

**Log Levels**:
- `GRIM_LOG_DEBUG` - Development/diagnostic info
- `GRIM_LOG_INFO` - Normal operation (⚠️ mapped to DEBUG internally)
- `GRIM_LOG_WARNING` - Recoverable issues (⚠️ mapped to ERROR internally)
- `GRIM_LOG_ERROR` - Error conditions
- `GRIM_LOG_CRITICAL` - Fatal errors (⚠️ mapped to ERROR internally)

> **Note**: GRIM's logger currently has `LOG_DEBUG`, `LOG_TRACE`, and `LOG_ERROR`. INFO/WARNING/CRITICAL are automatically mapped.

---

### Command Registration

```cpp
// Register a command
GrimResult res = api->register_command(api, "weather", weatherHandler, nullptr);

// Check if command exists
if (api->is_command_registered(api, "weather")) {
    // Already registered
}

// Unregister
api->unregister_command(api, "weather");

// Execute a command programmatically
GrimCommandResult result = api->execute_command(api, "help");
if (result.success) {
  api->log(api, GRIM_LOG_INFO, result.message);
}
```

**Command Handler Signature**:
```cpp
GrimCommandResult handler(const char* input, void* user_data);
```

---

### Event System

```cpp
// Event callback
void onEvent(const char* event_name, const void* event_data, size_t data_size, void* user_data) {
    api->log_fmt(api, GRIM_LOG_DEBUG, "Event: %s (size: %zu)", event_name, data_size);
}

// Subscribe to events
GrimEventHandle handle = api->subscribe_event(api, "command_executed", onEvent, nullptr);

// Unsubscribe
api->unsubscribe_event(api, handle);

// Emit custom events
const char* data = "custom payload";
api->emit_event(api, "my_plugin_event", data, strlen(data) + 1);
```

---

### Context Management

```cpp
// Store context
api->set_context(api, "last_query", "weather in NYC");

// Retrieve context
const char* query = api->get_context(api, "last_query");

// Check if exists
if (api->has_context(api, "last_query")) {
    // Context exists
}

// Delete context
api->delete_context(api, "last_query");
```

---

### Memory System (⚠️ Stubbed)

```cpp
// Store memory (not yet functional)
api->store_memory(api, "user_preference", "dark_mode", "settings");

// Retrieve memory (not yet functional)
const char* value = api->retrieve_memory(api, "user_preference");

// Delete memory (not yet functional)
api->delete_memory(api, "user_preference");
```

> **Status**: These functions are stubbed and return safe defaults. They log debug messages instead of performing actual operations.

---

### NLP & AI

```cpp
// Classify user intent
GrimIntent intent = api->classify_intent(api, "open chrome");
if (intent.matched) {
    api->log_fmt(api, GRIM_LOG_INFO, "Intent: %s (confidence: %.2f)", intent.name, intent.confidence);
}

// Resolve synonym
const char* canonical = api->resolve_synonym(api, "launch");  // Returns "open_app"

// Get conversation history
const char* history = api->get_conversation_history(api, 10);  // Last 10 messages
```

---

### Voice & TTS

```cpp
// Simple speech
api->queue_tts(api, "Hello, world!", nullptr);

// Advanced speech with parameters
GrimVoiceParams params = {
    .pitch = 1.0f,
    .speed = 1.2f,
    .volume = 0.8f,
    .voice_id = "default"
};
api->queue_tts(api, "Custom voice output", &params);

// Check if speaking
if (api->is_speaking(api)) {
    api->log(api, GRIM_LOG_DEBUG, "TTS is active");
}

// Stop audio (⚠️ Stubbed)
api->stop_tts(api);  // Not yet functional
```

---

### UI & Output

```cpp
// Send response to console
api->send_response(api, "Operation complete");

// Colored response
api->send_response_colored(api, "Success!", 0x00FF00);  // Green

// Show notification
api->show_notification(api, "Alert", "Task finished!");
```

---

### System Information

```cpp
// Get OS name
const char* os = api->get_system_info(api, "os");  // e.g., "Windows"

// Get CPU info
const char* cpu = api->get_system_info(api, "cpu");  // e.g., "8 cores"

// Get memory usage (bytes)
uint64_t mem = api->get_memory_usage(api);

// Get CPU usage (0-100%)
float cpu_usage = api->get_cpu_usage(api);
```

---

### Timers

```cpp
// Timer callback
void onTimer(uint64_t timer_id, void* user_data) {
    api->log_fmt(api, GRIM_LOG_DEBUG, "Timer %llu fired", timer_id);
}

// Create one-shot timer (1 second)
GrimTimerHandle timer = api->create_timer(api, 1000, false, onTimer, nullptr);

// Create repeating timer (500ms)
GrimTimerHandle repeating = api->create_timer(api, 500, true, onTimer, nullptr);

// Cancel timer
api->cancel_timer(api, timer);
```

---

### Plugin Metadata

```cpp
// Get current plugin name
const char* name = api->get_plugin_name(api);

// Get plugin DLL path
const char* path = api->get_plugin_path(api);

// Get permission bitmask
uint32_t perms = api->get_permissions(api);

// Check specific permission
if (api->has_permission(api, GRIM_PERM_FILESYSTEM)) {
    // Can access files
}

// Request hot-reload
api->request_reload(api);
```

---

## Permissions System

### Permission Flags

```cpp
typedef enum {
    GRIM_PERM_FILESYSTEM = (1 << 0),  // Read/write files
    GRIM_PERM_PROCESS    = (1 << 1),  // Launch processes
    GRIM_PERM_NETWORK    = (1 << 2),  // Use network (offline-first!)
    GRIM_PERM_MEMORY     = (1 << 3),  // Access memory system
    GRIM_PERM_VOICE= (1 << 4),  // Use TTS/STT
    GRIM_PERM_UI         = (1 << 5),  // Show UI elements
    GRIM_PERM_SYSTEM     = (1 << 6),  // Query system info
    GRIM_PERM_ALL        = 0xFFFFFFFF
} GrimPermission;
```

### Requesting Permissions

Declare required permissions in `grim_plugin_get_info()`:

```cpp
static GrimPluginInfo info = {
    .required_permissions = GRIM_PERM_UI | GRIM_PERM_VOICE | GRIM_PERM_SYSTEM
};
```

### Checking Permissions

```cpp
if (api->has_permission(api, GRIM_PERM_FILESYSTEM)) {
    // Safe to call file operations
} else {
    return GRIM_ERROR_PERMISSION_DENIED;
}
```

---

## Best Practices

### 1. Resource Cleanup
Always unregister commands and unsubscribe from events in `grim_plugin_shutdown()`:

```cpp
PLUGIN_EXPORT void grim_plugin_shutdown() {
    g_api->unregister_command(g_api, "mycommand");
    g_api->unsubscribe_event(g_api, my_event_handle);
    // Free any allocated resources
}
```

### 2. Thread Safety
The plugin API is **not** thread-safe. Only call API functions from:
- `grim_plugin_init()`
- `grim_plugin_shutdown()`
- Registered callbacks (command handlers, event handlers, timers)

### 3. Memory Management
- Use `g_api->allocate_memory(g_api)` and `g_api->free_memory(g_api)` for cross-DLL allocations
- GRIM owns strings returned by API functions - don't free them unless documented
- Use `g_api->string_duplicate(g_api)` if you need to keep a string

### 4. Permission Requests
Only request permissions you actually need. Users may deny plugins with excessive permissions.

### 5. Offline-First Design
**CRITICAL**: Most functionality should work offline. Only use network for:
- Web browser commands
- External API integrations explicitly requested by user

### 6. Logging
Use appropriate log levels:
- `DEBUG` for development/diagnostics
- `INFO` for normal operations
- `WARNING` for recoverable issues
- `ERROR` for failures
- `CRITICAL` for fatal errors

### 7. Version Checking
Always check API version compatibility:

```cpp
PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if ((api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION) {
        return GRIM_ERROR_API_VERSION_MISMATCH;
    }
    // ...
}
```

### 8. Command Naming
Use clear, descriptive command names:
- ✅ Good: `weather`, `translate`, `note save`
- ❌ Bad: `w`, `tr`, `ns`

### 9. Documentation
Document your plugin's commands and features in your `grim_plugin_get_info()` description.

### 10. Synonym Support
Leverage GRIM's synonym system for natural command variations:
```cpp
// User can say "launch chrome" or "open chrome" or "start chrome"
// All resolve to the same canonical command via synonym resolution
```

---

## Troubleshooting

### Plugin Won't Load

**Problem**: Plugin DLL not loading.

**Solutions**:
- Check API version compatibility
- Ensure all required functions are exported
- Verify dependencies are available
- Check GRIM logs for detailed error messages

### Command Not Registering

**Problem**: `register_command()` returns error.

**Solutions**:
- Command name already registered (check with `is_command_registered()`)
- Invalid command name (empty or NULL)
- Check return code and log the error

### Permission Denied Errors

**Problem**: API calls return `GRIM_ERROR_PERMISSION_DENIED`.

**Solutions**:
- Check `required_permissions` in `grim_plugin_get_info()`
- Verify permission with `has_permission()` before calling
- Request only necessary permissions

### Plugin Crashes GRIM

**Problem**: Plugin causes GRIM to crash.

**Solutions**:
- **Never** dereference opaque handles (use them only as tokens)
- Check for NULL pointers before use
- Don't call API functions from threads you create
- Clean up resources in `grim_plugin_shutdown()`

### Events Not Firing

**Problem**: Event callbacks never called.

**Solutions**:
- Verify event name is correct (case-sensitive)
- Check that handle returned by `subscribe_event()` is not NULL
- Ensure you're listening for the right event
- Check if event is actually being emitted

### Hot-Reload Not Working

**Problem**: Changes to plugin not reflected after rebuild.

**Solutions**:
- Call `g_api->request_reload(g_api)` from plugin or reload via GRIM command
- Ensure DLL is not locked by debugger
- Implement `grim_plugin_reload()` for custom reload logic

### Synonym Resolution Not Working

**Problem**: `resolve_synonym()` returns input unchanged.

**Solutions**:
- Check if synonym exists in `synonyms.json`
- Synonyms are case-insensitive (normalized to lowercase)
- Returns original word if no mapping exists (not an error)

### Memory API Functions Don't Work

**Problem**: `store_memory()`, `retrieve_memory()`, `delete_memory()` don't persist data.

**Explanation**: These functions are currently **stubbed** and log debug messages instead. The GRIM memory system uses `MemoryObject` structures that require additional bridging logic not yet implemented.

**Workaround**: Use context storage (`set_context()`/`get_context()`) for simple key-value data.

---

## Future Enhancements

Planned additions to the plugin API:

- **File System API**: Full file read/write/directory operations
- **Process Management**: Launch and monitor external processes
- **Full Memory Integration**: Bridge plugin key-value API to `MemoryObject` system
- **Audio Control**: Implement `stop_audio()` and audio playback controls
- **Advanced UI**: Custom panel creation, input dialogs, progress bars
- **Network API**: HTTP client, websockets (offline-first principles)

---

## Example Plugins

Check the `plugins/` directory for reference implementations:

- **`osint_plugin`** - OSINT scanning and analysis commands
- **`core_plugin`** - Core system commands (built directly into GRIM.exe)

---

## Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check existing plugin examples in `plugins/` directory
- Review `core/plugin_api.hpp` for full API surface
- See `core/plugin_api_impl.cpp` for implementation details

---
