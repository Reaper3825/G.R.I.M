# G.R.I.M Plugin API Documentation

## Table of Contents
1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [API Reference](#api-reference)
4. [Creating Your First Plugin](#creating-your-first-plugin)
5. [Example Plugins](#example-plugins)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The G.R.I.M Plugin API enables you to extend GRIM's functionality through hot-reloadable dynamic libraries (DLLs on Windows, .so on Linux). Plugins can:

- Register custom commands
- Subscribe to system events
- Access memory and context
- Use NLP and AI features
- Control voice synthesis
- Interact with the file system
- Launch processes
- Display UI notifications

### Design Principles

- **ABI Stability**: C-compatible interface ensures plugins remain compatible across updates
- **Permission-Based Security**: Plugins declare required permissions upfront
- **Offline-First**: Most APIs work offline; network access is restricted and controlled
- **Version Negotiation**: Plugins specify API version compatibility
- **Opaque Handles**: Plugins never access internal GRIM structures directly

### Current API Version

**Version**: 1.0.0 (Major.Minor.Patch)
- **Major**: 1 - Breaking changes increment this
- **Minor**: 0 - New features, backward compatible
- **Patch**: 0 - Bug fixes

---

## Getting Started

### Prerequisites

- C++20 compatible compiler
- CMake 3.22+
- GRIM development headers (`core/plugin_api.hpp`)

### Plugin Structure

Every plugin must:
1. Export `grim_plugin_get_info()` - Returns plugin metadata
2. Export `grim_plugin_init()` - Initialization entry point
3. Export `grim_plugin_shutdown()` - Cleanup entry point
4. Optionally export `grim_plugin_reload()` - Hot-reload handler

### Minimal Plugin Template

```cpp
#include "core/plugin_api.hpp"

static const GrimPluginAPI* g_api = nullptr;

extern "C" {
    PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
        static GrimPluginInfo info;
        info.api_version = GRIM_API_VERSION;
        info.name = "My Plugin";
        info.version = "1.0.0";
        info.author = "Your Name";
        info.description = "Brief description";
        info.required_permissions = GRIM_PERM_UI;
        return &info;
    }

    PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
        g_api = api;
        g_api->log(GRIM_LOG_INFO, "Plugin initialized");
        return GRIM_OK;
    }

    PLUGIN_EXPORT void grim_plugin_shutdown() {
        g_api->log(GRIM_LOG_INFO, "Plugin shutting down");
    }
}
```

---

## API Reference

### Core Types

#### Result Codes (`GrimResult`)

| Value | Description |
|-------|-------------|
| `GRIM_OK` | Operation succeeded |
| `GRIM_ERROR_INVALID_PARAM` | Invalid parameter provided |
| `GRIM_ERROR_NOT_FOUND` | Requested resource not found |
| `GRIM_ERROR_PERMISSION_DENIED` | Insufficient permissions |
| `GRIM_ERROR_ALREADY_EXISTS` | Resource already exists |
| `GRIM_ERROR_OUT_OF_MEMORY` | Memory allocation failed |
| `GRIM_ERROR_UNSUPPORTED` | Operation not supported |
| `GRIM_ERROR_PLUGIN_NOT_LOADED` | Plugin not loaded |
| `GRIM_ERROR_API_VERSION_MISMATCH` | API version incompatible |
| `GRIM_ERROR_UNKNOWN` | Unknown error |

#### Log Levels (`GrimLogLevel`)

| Level | Use Case |
|-------|----------|
| `GRIM_LOG_DEBUG` | Detailed diagnostic information |
| `GRIM_LOG_INFO` | General informational messages |
| `GRIM_LOG_WARNING` | Warning conditions |
| `GRIM_LOG_ERROR` | Error conditions |
| `GRIM_LOG_CRITICAL` | Critical failures |

#### Permissions (`GrimPermission`)

| Permission | Description |
|------------|-------------|
| `GRIM_PERM_FILESYSTEM` | Read/write files on disk |
| `GRIM_PERM_PROCESS` | Launch and manage processes |
| `GRIM_PERM_NETWORK` | Network access (still offline-first!) |
| `GRIM_PERM_MEMORY` | Access memory system |
| `GRIM_PERM_VOICE` | Use TTS/STT features |
| `GRIM_PERM_UI` | Display UI elements and notifications |
| `GRIM_PERM_SYSTEM` | Query system information |
| `GRIM_PERM_ALL` | All permissions (use sparingly) |

**Note**: Combine permissions with bitwise OR: `GRIM_PERM_UI | GRIM_PERM_FILESYSTEM`

---

### Logging Functions

#### `log(level, message)`
Write a message to GRIM's log.

**Parameters:**
- `level` (GrimLogLevel): Severity level
- `message` (const char*): Message to log

**Example:**
```cpp
g_api->log(GRIM_LOG_INFO, "Processing user request");
g_api->log(GRIM_LOG_ERROR, "Failed to open file");
```

#### `log_fmt(level, format, ...)`
Write a formatted message to the log (printf-style).

**Example:**
```cpp
g_api->log_fmt(GRIM_LOG_INFO, "Processing %d items", count);
```

---

### Command Registration

#### `register_command(name, handler, user_data)`
Register a new command that users can invoke.

**Parameters:**
- `name` (const char*): Command name (e.g., "weather", "translate")
- `handler` (GrimCommandHandler): Function to handle command
- `user_data` (void*): Optional data passed to handler

**Returns:** `GrimResult`

**Handler Signature:**
```cpp
GrimCommandResult handler(const char* input, void* user_data)
```

**Example:**
```cpp
GrimCommandResult weather_handler(const char* input, void* user_data) {
    GrimCommandResult result;
    result.success = true;
    result.message = "It's sunny and 72°F";
    result.error_code = nullptr;
    result.category = "weather";
    result.color_rgb = 0xFFFF00; // Yellow
    return result;
}

g_api->register_command("weather", weather_handler, nullptr);
```

#### `unregister_command(name)`
Remove a previously registered command.

**Returns:** `GrimResult`

#### `is_command_registered(name)`
Check if a command is already registered.

**Returns:** `bool`

#### `execute_command(command_text)`
Execute a command programmatically.

**Example:**
```cpp
GrimCommandResult result = g_api->execute_command("open notepad");
```

---

### Event System

Events allow plugins to react to system occurrences without polling.

#### Common Events

| Event Name | Description | Data |
|------------|-------------|------|
| `command_executed` | After any command runs | Command name |
| `user_input` | Raw user text input | Input text |
| `intent_classified` | NLP intent detected | Intent struct |
| `plugin_loaded` | When a plugin loads | Plugin name |
| `plugin_unloaded` | When a plugin unloads | Plugin name |
| `system_shutdown` | GRIM is shutting down | None |

#### `subscribe_event(event_name, callback, user_data)`
Subscribe to an event.

**Returns:** `GrimEventHandle` (save this to unsubscribe later)

**Callback Signature:**
```cpp
void callback(const char* event_name, const void* event_data, size_t data_size, void* user_data)
```

**Example:**
```cpp
void on_user_input(const char* event_name, const void* event_data, size_t data_size, void* user_data) {
    const char* input = static_cast<const char*>(event_data);
    g_api->log_fmt(GRIM_LOG_DEBUG, "User said: %s", input);
}

GrimEventHandle handle = g_api->subscribe_event("user_input", on_user_input, nullptr);
```

#### `unsubscribe_event(handle)`
Stop listening to an event.

#### `emit_event(event_name, data, data_size)`
Emit a custom event other plugins can listen to.

**Example:**
```cpp
const char* msg = "Custom event triggered";
g_api->emit_event("my_plugin.custom_event", msg, strlen(msg) + 1);
```

---

### Memory & Context API

#### `get_context(key)` / `set_context(key, value)`
Short-term context storage (current session).

**Example:**
```cpp
g_api->set_context("last_query", "What's the weather?");
const char* last = g_api->get_context("last_query");
```

#### `delete_context(key)` / `has_context(key)`
Remove or check for context keys.

#### `store_memory(key, value, category)` / `retrieve_memory(key)`
Long-term memory storage (persists across sessions).

**Example:**
```cpp
g_api->store_memory("user_preference.theme", "dark", "preferences");
const char* theme = g_api->retrieve_memory("user_preference.theme");
```

#### `delete_memory(key)`
Remove a memory entry.

---

### NLP & AI Functions

#### `classify_intent(text)`
Analyze text and return detected intent.

**Returns:** `GrimIntent`

**Example:**
```cpp
GrimIntent intent = g_api->classify_intent("open chrome");
if (intent.matched) {
    g_api->log_fmt(GRIM_LOG_INFO, "Intent: %s (confidence: %.2f)", 
                   intent.name, intent.confidence);
}
```

#### `resolve_synonym(word)`
Get the canonical form of a synonym.

**Example:**
```cpp
const char* canonical = g_api->resolve_synonym("browser");
// Returns "chrome" or "firefox" based on synonym mappings
```

#### `get_conversation_history(limit)`
Retrieve recent conversation (returns JSON array).

**Example:**
```cpp
const char* history = g_api->get_conversation_history(10);
// Parse JSON to analyze past interactions
```

---

### Output & UI Functions

#### `send_response(text)`
Send text response to the user.

**Example:**
```cpp
g_api->send_response("Task completed successfully!");
```

#### `send_response_colored(text, rgb_color)`
Send colored text response.

**Example:**
```cpp
g_api->send_response_colored("Error occurred!", 0xFF0000); // Red text
```

#### `show_notification(title, message)`
Display a UI notification.

**Example:**
```cpp
g_api->show_notification("Weather Update", "It's going to rain today");
```

---

### Voice Functions (Requires `GRIM_PERM_VOICE`)

#### `queue_tts(text, params)`
Queue text for speech synthesis.

**Parameters:**
- `text` (const char*): Text to speak
- `params` (const GrimVoiceParams*): Voice parameters (nullable)

**Voice Parameters:**
```cpp
GrimVoiceParams params;
params.pitch = 1.0f;     // 0.5 - 2.0 (default: 1.0)
params.speed = 1.0f;     // 0.5 - 2.0 (default: 1.0)
params.volume = 0.8f;    // 0.0 - 1.0 (default: 1.0)
params.voice_id = "p226"; // Speaker ID or nullptr for default

g_api->queue_tts("Hello there!", &params);
```

#### `stop_tts()` / `is_speaking()`
Control speech playback.

---

### File System API (Requires `GRIM_PERM_FILESYSTEM`)

#### `read_file_text(path)` / `write_file_text(path, content)`
High-level file operations.

**Example:**
```cpp
const char* content = g_api->read_file_text("C:/data/config.txt");
if (content) {
    // Process content
    g_api->string_free(content); // Free when done
}

g_api->write_file_text("C:/data/output.txt", "Hello World");
```

#### `file_exists(path)` / `delete_file(path)`
File management utilities.

#### Low-Level File I/O

For binary files or streaming, use:
- `open_file(path, mode)` - Returns `GrimFileHandle`
- `read_file(handle, buffer, size)` - Returns bytes read
- `write_file(handle, data, size)` - Returns bytes written
- `close_file(handle)` - Close file

**Example:**
```cpp
GrimFileHandle file = g_api->open_file("data.bin", "rb");
char buffer[1024];
size_t read = g_api->read_file(file, buffer, sizeof(buffer));
g_api->close_file(file);
```

---

### Process Management (Requires `GRIM_PERM_PROCESS`)

#### `launch_app(app_name)`
Launch an application by name.

**Example:**
```cpp
g_api->launch_app("notepad");
g_api->launch_app("chrome");
```

#### `launch_process(command, args)`
Launch a process with arguments.

**Example:**
```cpp
g_api->launch_process("python", "script.py --verbose");
```

#### `is_process_running(name)` / `kill_process(name)`
Process management utilities.

---

### System Information (Requires `GRIM_PERM_SYSTEM`)

#### `get_system_info(key)`
Query system information.

**Common Keys:**
- `"os"` - Operating system name
- `"cpu"` - CPU model
- `"memory"` - Total RAM
- `"disk"` - Disk space info

**Example:**
```cpp
const char* os = g_api->get_system_info("os");
g_api->log_fmt(GRIM_LOG_INFO, "Running on: %s", os);
```

#### `get_memory_usage()` / `get_cpu_usage()`
Get current resource usage.

**Example:**
```cpp
uint64_t mem_bytes = g_api->get_memory_usage();
float cpu_percent = g_api->get_cpu_usage();
```

---

### Timer & Scheduling

#### `create_timer(interval_ms, repeat, callback, user_data)`
Schedule a delayed or periodic task.

**Parameters:**
- `interval_ms` (uint64_t): Delay in milliseconds
- `repeat` (bool): True for repeating timer
- `callback` (GrimTimerCallback): Function to call
- `user_data` (void*): Custom data

**Example:**
```cpp
void timer_callback(uint64_t timer_id, void* user_data) {
    g_api->log(GRIM_LOG_INFO, "Timer fired!");
}

// One-shot timer (5 seconds)
GrimTimerHandle timer = g_api->create_timer(5000, false, timer_callback, nullptr);

// Repeating timer (every 1 second)
GrimTimerHandle repeating = g_api->create_timer(1000, true, timer_callback, nullptr);
```

#### `cancel_timer(handle)`
Stop and remove a timer.

---

### Plugin Lifecycle & Metadata

#### `get_plugin_name()` / `get_plugin_path()`
Get current plugin information.

#### `get_permissions()` / `has_permission(perm)`
Check granted permissions.

**Example:**
```cpp
if (g_api->has_permission(GRIM_PERM_FILESYSTEM)) {
    // Safe to use file operations
}
```

#### `request_reload()`
Request plugin to be reloaded (useful during development).

---

### Utility Functions

#### `allocate_memory(size)` / `free_memory(ptr)`
Use GRIM's memory allocator (recommended for cross-DLL memory).

#### `string_duplicate(str)` / `string_free(str)`
GRIM-managed string utilities.

**Example:**
```cpp
const char* copy = g_api->string_duplicate("Hello");
// Use copy...
g_api->string_free(copy);
```

---

## Creating Your First Plugin

### Step 1: Set Up CMakeLists.txt

Create `plugins/my_plugin/CMakeLists.txt`:

```cmake
project(my_plugin LANGUAGES CXX)

add_library(my_plugin SHARED
    ${CMAKE_SOURCE_DIR}/plugins/my_plugin.cpp
)

target_include_directories(my_plugin PRIVATE
    ${CMAKE_SOURCE_DIR}
)

target_compile_definitions(my_plugin PRIVATE GRIM_BUILD_PLUGIN)
target_link_libraries(my_plugin PRIVATE GRIM_EXPORT)
add_dependencies(my_plugin GRIM)

set_target_properties(my_plugin PROPERTIES
    CXX_STANDARD 20
    OUTPUT_NAME "my_plugin"
)
```

Add to main `CMakeLists.txt`:
```cmake
add_subdirectory(plugins/my_plugin)
```

### Step 2: Create Plugin Source

`plugins/my_plugin.cpp`:

```cpp
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
```

### Step 3: Build and Test

```bash
cmake --build . --config Release
```

The plugin DLL will be in `plugins/my_plugin/Release/my_plugin.dll` (Windows) or `plugins/my_plugin/my_plugin.so` (Linux).

---

## Example Plugins

### Example 1: Counter Plugin

Tracks how many times a command is used.

```cpp
#include "core/plugin_api.hpp"

static const GrimPluginAPI* g_api = nullptr;
static int counter = 0;

static GrimCommandResult count_command(const char* input, void* user_data) {
    counter++;
    
    char msg[256];
    snprintf(msg, sizeof(msg), "Counter: %d", counter);
    
    g_api->send_response(msg);
    g_api->set_context("counter_value", std::to_string(counter).c_str());
    
    GrimCommandResult result;
    result.success = true;
    result.message = msg;
    result.error_code = nullptr;
    result.category = "utility";
    result.color_rgb = 0x0088FF;
    return result;
}

extern "C" {
    PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
        static GrimPluginInfo info;
        info.api_version = GRIM_API_VERSION;
        info.name = "Counter Plugin";
        info.version = "1.0.0";
        info.author = "Example";
        info.description = "Counts command invocations";
        info.required_permissions = GRIM_PERM_UI;
        return &info;
    }

    PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
        g_api = api;
        g_api->register_command("count", count_command, nullptr);
        return GRIM_OK;
    }

    PLUGIN_EXPORT void grim_plugin_shutdown() {
        g_api->unregister_command("count");
    }
}
```

### Example 2: Note-Taking Plugin

Save and retrieve notes to files.

```cpp
#include "core/plugin_api.hpp"
#include <string>
#include <cstring>

static const GrimPluginAPI* g_api = nullptr;

static GrimCommandResult note_save(const char* input, void* user_data) {
    // Expected: "note save: <content>"
    const char* content = strstr(input, ":");
    if (!content) {
        GrimCommandResult result;
        result.success = false;
        result.message = "Usage: note save: your note content";
        result.error_code = "INVALID_FORMAT";
        result.category = "note";
        result.color_rgb = 0xFF0000;
        return result;
    }
    
    content += 1; // Skip ':'
    while (*content == ' ') content++; // Skip spaces
    
    GrimResult res = g_api->write_file_text("notes.txt", content);
    
    GrimCommandResult result;
    result.success = (res == GRIM_OK);
    result.message = result.success ? "Note saved!" : "Failed to save note";
    result.error_code = result.success ? nullptr : "FILE_ERROR";
    result.category = "note";
    result.color_rgb = result.success ? 0x00FF00 : 0xFF0000;
    return result;
}

static GrimCommandResult note_read(const char* input, void* user_data) {
    const char* content = g_api->read_file_text("notes.txt");
    
    GrimCommandResult result;
    if (content) {
        result.success = true;
        result.message = content;
        result.category = "note";
        result.color_rgb = 0x00FF00;
        // Note: content will be freed by GRIM
    } else {
        result.success = false;
        result.message = "No notes found";
        result.error_code = "NOT_FOUND";
        result.category = "note";
        result.color_rgb = 0xFFFF00;
    }
    return result;
}

extern "C" {
    PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
        static GrimPluginInfo info;
        info.api_version = GRIM_API_VERSION;
        info.name = "Notes Plugin";
        info.version = "1.0.0";
        info.author = "Example";
        info.description = "Save and retrieve notes";
        info.required_permissions = GRIM_PERM_FILESYSTEM | GRIM_PERM_UI;
        return &info;
    }

    PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
        g_api = api;
        g_api->register_command("note save", note_save, nullptr);
        g_api->register_command("note read", note_read, nullptr);
        g_api->log(GRIM_LOG_INFO, "Notes plugin loaded");
        return GRIM_OK;
    }

    PLUGIN_EXPORT void grim_plugin_shutdown() {
        g_api->unregister_command("note save");
        g_api->unregister_command("note read");
    }
}
```

### Example 3: Event Logger Plugin

Logs all system events to a file.

```cpp
#include "core/plugin_api.hpp"
#include <ctime>
#include <cstring>

static const GrimPluginAPI* g_api = nullptr;
static GrimEventHandle cmd_handle = nullptr;
static GrimEventHandle input_handle = nullptr;

static void log_event_to_file(const char* event_type, const char* data) {
    // Get timestamp
    time_t now = time(nullptr);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
    
    // Format log entry
    char entry[512];
    snprintf(entry, sizeof(entry), "[%s] %s: %s\n", timestamp, event_type, data);
    
    // Append to log file
    GrimFileHandle file = g_api->open_file("event_log.txt", "a");
    if (file) {
        g_api->write_file(file, entry, strlen(entry));
        g_api->close_file(file);
    }
}

static void on_command_executed(const char* event_name, const void* event_data, 
                                 size_t data_size, void* user_data) {
    const char* cmd = static_cast<const char*>(event_data);
    log_event_to_file("COMMAND", cmd);
}

static void on_user_input(const char* event_name, const void* event_data,
                          size_t data_size, void* user_data) {
    const char* input = static_cast<const char*>(event_data);
    log_event_to_file("INPUT", input);
}

extern "C" {
    PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
        static GrimPluginInfo info;
        info.api_version = GRIM_API_VERSION;
        info.name = "Event Logger";
        info.version = "1.0.0";
        info.author = "Example";
        info.description = "Logs all system events to file";
        info.required_permissions = GRIM_PERM_FILESYSTEM;
        return &info;
    }

    PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
        g_api = api;
        
        cmd_handle = g_api->subscribe_event("command_executed", on_command_executed, nullptr);
        input_handle = g_api->subscribe_event("user_input", on_user_input, nullptr);
        
        g_api->log(GRIM_LOG_INFO, "Event logger plugin started");
        return GRIM_OK;
    }

    PLUGIN_EXPORT void grim_plugin_shutdown() {
        if (cmd_handle) g_api->unsubscribe_event(cmd_handle);
        if (input_handle) g_api->unsubscribe_event(input_handle);
        g_api->log(GRIM_LOG_INFO, "Event logger plugin stopped");
    }
}
```

---

## Best Practices

### 1. Error Handling
Always check return codes and handle errors gracefully:

```cpp
GrimResult res = g_api->register_command("mycommand", handler, nullptr);
if (res != GRIM_OK) {
    g_api->log(GRIM_LOG_ERROR, "Failed to register command");
    return res;
}
```

### 2. Resource Cleanup
Unregister all commands and unsubscribe from all events in `grim_plugin_shutdown()`:

```cpp
PLUGIN_EXPORT void grim_plugin_shutdown() {
    g_api->unregister_command("mycommand");
    g_api->unsubscribe_event(my_event_handle);
    // Free any allocated resources
}
```

### 3. Thread Safety
The plugin API is **not** thread-safe. Only call API functions from:
- `grim_plugin_init()`
- `grim_plugin_shutdown()`
- Registered callbacks (command handlers, event handlers, timers)

### 4. Memory Management
- Use `g_api->allocate_memory()` and `g_api->free_memory()` for cross-DLL allocations
- GRIM owns strings returned by API functions - don't free them unless documented
- Use `g_api->string_duplicate()` if you need to keep a string

### 5. Permission Requests
Only request permissions you actually need. Users may deny plugins with excessive permissions.

### 6. Offline-First Design
**CRITICAL**: Most functionality should work offline. Only use network for:
- Web browser commands
- External API integrations explicitly requested by user

### 7. Logging
Use appropriate log levels:
- `DEBUG` for development/diagnostics
- `INFO` for normal operations
- `WARNING` for recoverable issues
- `ERROR` for failures
- `CRITICAL` for fatal errors

### 8. Version Checking
Always check API version compatibility:

```cpp
PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (api->api_version < GRIM_API_VERSION) {
        return GRIM_ERROR_API_VERSION_MISMATCH;
    }
    // ...
}
```

### 9. Command Naming
Use clear, descriptive command names:
- ✅ Good: `weather`, `translate`, `note save`
- ❌ Bad: `w`, `tr`, `ns`

### 10. Documentation
Document your plugin's commands and features in your `grim_plugin_get_info()` description.

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

---

## Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check existing plugin examples in `plugins/` directory
- Review `core/plugin_api.hpp` for full API surface

---

**Version**: 1.0.0  
**Last Updated**: October 29, 2025  
**API Version**: 1.0.0
