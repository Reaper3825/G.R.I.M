# G.R.I.M Plugin API Documentation

**Version**: 1.0.0  
**Last Updated**: October 29, 2025  
**API Version**: 1.0.0  
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
- **Permission-based**: Sandboxed access via permission flags

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
    g_api->log(GRIM_LOG_INFO, "My command executed!");
    g_api->speak("Hello from my plugin!");
    
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
    if (api->api_version < GRIM_API_VERSION) {
        return GRIM_ERROR_API_VERSION_MISMATCH;
    }
    
  // Register command
    GrimResult res = g_api->register_command("my_command", myCommandHandler, nullptr);
    if (res != GRIM_OK) {
   g_api->log(GRIM_LOG_ERROR, "Failed to register command");
      return res;
    }
    
    g_api->log(GRIM_LOG_INFO, "My Plugin initialized successfully!");
    return GRIM_OK;
}

// Plugin shutdown
extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {
    g_api->unregister_command("my_command");
    g_api->log(GRIM_LOG_INFO, "My Plugin shut down");
}

// Optional: Hot-reload support
extern "C" PLUGIN_EXPORT void grim_plugin_reload() {
  g_api->log(GRIM_LOG_INFO, "My Plugin reloaded");
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
api->log(GRIM_LOG_INFO, "Plugin loaded");

// Formatted log message
api->log_fmt(GRIM_LOG_DEBUG, "Value: %d, String: %s", 42, "test");
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
GrimResult res = api->register_command("weather", weatherHandler, nullptr);

// Check if command exists
if (api->is_command_registered("weather")) {
    // Already registered
}

// Unregister
api->unregister_command("weather");

// Execute a command programmatically
GrimCommandResult result = api->execute_command("help");
if (result.success) {
  api->log(GRIM_LOG_INFO, result.message);
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
    api->log_fmt(GRIM_LOG_DEBUG, "Event: %s (size: %zu)", event_name, data_size);
}

// Subscribe to events
GrimEventHandle handle = api->subscribe_event("command_executed", onEvent, nullptr);

// Unsubscribe
api->unsubscribe_event(handle);

// Emit custom events
const char* data = "custom payload";
api->emit_event("my_plugin_event", data, strlen(data) + 1);
```

---

### Context Management

```cpp
// Store context
api->set_context("last_query", "weather in NYC");

// Retrieve context
const char* query = api->get_context("last_query");

// Check if exists
if (api->has_context("last_query")) {
    // Context exists
}

// Delete context
api->delete_context("last_query");
```

---

### Memory System (⚠️ Stubbed)

```cpp
// Store memory (not yet functional)
api->store_memory("user_preference", "dark_mode", "settings");

// Retrieve memory (not yet functional)
const char* value = api->retrieve_memory("user_preference");

// Delete memory (not yet functional)
api->delete_memory("user_preference");
```

> **Status**: These functions are stubbed and return safe defaults. They log debug messages instead of performing actual operations.

---

### NLP & AI

```cpp
// Classify user intent
GrimIntent intent = api->classify_intent("open chrome");
if (intent.matched) {
    api->log_fmt(GRIM_LOG_INFO, "Intent: %s (confidence: %.2f)", intent.name, intent.confidence);
}

// Resolve synonym
const char* canonical = api->resolve_synonym("launch");  // Returns "open_app"

// Get conversation history
const char* history = api->get_conversation_history(10);  // Last 10 messages
```

---

### Voice & TTS

```cpp
// Simple speech
api->speak("Hello, world!");

// Advanced speech with parameters
GrimVoiceParams params = {
    .pitch = 1.0f,
    .speed = 1.2f,
    .volume = 0.8f,
    .voice_id = "default"
};
api->speak_with_params("Custom voice output", &params);

// Check if speaking
if (api->is_speaking()) {
    api->log(GRIM_LOG_DEBUG, "TTS is active");
}

// Stop audio (⚠️ Stubbed)
api->stop_audio();  // Not yet functional
```

---

### UI & Output

```cpp
// Send response to console
api->send_response("Operation complete");

// Colored response
api->send_response_colored("Success!", 0x00FF00);  // Green

// Show notification
api->show_notification("Alert", "Task finished!", GRIM_NOTIFY_INFO);
```

---

### System Information

```cpp
// Get OS name
const char* os = api->get_system_info("os");  // e.g., "Windows"

// Get CPU info
const char* cpu = api->get_system_info("cpu");  // e.g., "8 cores"

// Get memory usage (bytes)
uint64_t mem = api->get_memory_usage();

// Get CPU usage (0-100%)
float cpu_usage = api->get_cpu_usage();
```

---

### Timers

```cpp
// Timer callback
void onTimer(uint64_t timer_id, void* user_data) {
    api->log_fmt(GRIM_LOG_DEBUG, "Timer %llu fired", timer_id);
}

// Create one-shot timer (1 second)
GrimTimerHandle timer = api->create_timer(1000, false, onTimer, nullptr);

// Create repeating timer (500ms)
GrimTimerHandle repeating = api->create_timer(500, true, onTimer, nullptr);

// Cancel timer
api->cancel_timer(timer);
```

---

### Plugin Metadata

```cpp
// Get current plugin name
const char* name = api->get_plugin_name();

// Get plugin DLL path
const char* path = api->get_plugin_path();

// Get permission bitmask
uint32_t perms = api->get_permissions();

// Check specific permission
if (api->has_permission(GRIM_PERM_FILESYSTEM)) {
    // Can access files
}

// Request hot-reload
api->request_reload();
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
if (api->has_permission(GRIM_PERM_FILESYSTEM)) {
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
    g_api->unregister_command("mycommand");
    g_api->unsubscribe_event(my_event_handle);
    // Free any allocated resources
}
```

### 2. Thread Safety
The plugin API is **not** thread-safe. Only call API functions from:
- `grim_plugin_init()`
- `grim_plugin_shutdown()`
- Registered callbacks (command handlers, event handlers, timers)

### 3. Memory Management
- Use `g_api->allocate_memory()` and `g_api->free_memory()` for cross-DLL allocations
- GRIM owns strings returned by API functions - don't free them unless documented
- Use `g_api->string_duplicate()` if you need to keep a string

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
    if (api->api_version < GRIM_API_VERSION) {
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
- Call `g_api->request_reload()` from plugin or reload via GRIM command
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
