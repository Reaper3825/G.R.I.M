# Response System Improvements

This document describes the improvements made to the G.R.I.M response system.

## Overview

The response system has been enhanced with several new features to make interactions more natural, varied, and contextual. These improvements maintain the offline-first design principle while adding intelligence to response selection.

## Key Features

### 1. Response History Tracking

**Problem**: Previously, the same response variant could be selected consecutively, making interactions feel repetitive.

**Solution**: Added response history tracking that remembers the last 3 responses for each key and avoids repeating them.

**Implementation**:
```cpp
// Automatically tracked per response key
std::string resp1 = ResponseManager::get("ack_understood"); // "Got it."
std::string resp2 = ResponseManager::get("ack_understood"); // "Understood." (different!)
std::string resp3 = ResponseManager::get("ack_understood"); // "Okay." (still different!)
```

**Benefits**:
- More natural conversation flow
- Reduces perceived repetition
- Maintains randomization while ensuring variety

### 2. Contextual Greetings

**Problem**: Generic greetings didn't reflect time of day or context.

**Solution**: Added time-based greeting selection with dedicated response sets for different times of day.

**Time Ranges**:
- **Morning** (5:00 AM - 11:59 AM): "Good morning!", "Morning! Ready to assist.", etc.
- **Afternoon** (12:00 PM - 4:59 PM): "Good afternoon!", "Afternoon! How can I help?", etc.
- **Evening** (5:00 PM - 9:59 PM): "Good evening!", "Evening! What do you need?", etc.
- **Night** (10:00 PM - 4:59 AM): "Still up? I'm here if you need me.", etc.

**Usage**:
```cpp
std::string greeting = ResponseManager::getGreeting();
// Returns appropriate greeting based on current time
```

### 3. Parameter Substitution

**Problem**: Responses with dynamic content required string concatenation in command code.

**Solution**: Added template-based parameter substitution for cleaner, more maintainable response definitions.

**Usage**:
```cpp
// Define a template (or use existing responses with {placeholder} syntax)
std::unordered_map<std::string, std::string> params;
params["app"] = "Chrome";
params["action"] = "opening";

std::string response = ResponseManager::getWithParams("open_app_success", params);
// Result: "Launching Chrome" or similar, with parameters substituted
```

**Benefits**:
- Cleaner command code
- Easier to maintain response templates
- Supports multiple parameters per response

### 4. New Response Categories

Added new response categories to support common interaction patterns:

#### Greetings
- `greeting_morning` - Morning greetings
- `greeting_afternoon` - Afternoon greetings
- `greeting_evening` - Evening greetings
- `greeting_night` - Late night greetings

#### Acknowledgments
- `ack_understood` - Simple acknowledgments ("Got it.", "Understood.", etc.)
- `ack_working` - Processing indicators ("Working on it...", "Give me a moment...", etc.)
- `ack_done` - Completion confirmations ("Done.", "All set.", etc.)

### 5. Enhanced Error Messages

Added new error codes for better error handling:

- `ERR_AI_INVALID_BACKEND` - Invalid AI backend specified
- `ERR_RESPONSE_PARAM_MISSING` - Template parameter missing
- `ERR_NETWORK_UNAVAILABLE` - Network connection unavailable
- `ERR_PERMISSION_DENIED` - Permission denied for operation

Each error code includes both user-friendly and debug messages for better troubleshooting.

## API Reference

### Core Functions

#### `ResponseManager::get(const std::string& keyOrMessage)`
Retrieves a response by key, automatically selecting a variant that hasn't been used recently.

**Parameters**:
- `keyOrMessage`: Response key (e.g., "clean", "ack_done") or literal message

**Returns**: Response string with history-aware randomization

**Example**:
```cpp
std::string response = ResponseManager::get("clean");
// Returns: "History cleared." or "Console wiped clean." or similar
```

#### `ResponseManager::getWithParams(const std::string& key, const std::unordered_map<std::string, std::string>& params)`
Retrieves a response with parameter substitution.

**Parameters**:
- `key`: Response key
- `params`: Map of parameter names to values

**Returns**: Response string with parameters substituted

**Example**:
```cpp
std::unordered_map<std::string, std::string> params;
params["app"] = "Firefox";
std::string response = ResponseManager::getWithParams("open_app_success", params);
```

#### `ResponseManager::getGreeting()`
Gets a contextual greeting based on the current time of day.

**Returns**: Time-appropriate greeting string

**Example**:
```cpp
std::string greeting = ResponseManager::getGreeting();
// Returns "Good morning!" if called between 5 AM and noon
```

#### `ResponseManager::clearHistory()`
Clears the response history, allowing all variants to be selected again.

**Usage**:
```cpp
ResponseManager::clearHistory(); // Useful for testing or reset scenarios
```

#### `ResponseManager::systemMessage(const std::string& msg, const Color& color)`
Creates a system message (existing function, unchanged).

**Parameters**:
- `msg`: Message text
- `color`: Display color (default: Green)

**Returns**: CommandResult with system message configuration

## Testing

A comprehensive test suite is available via the `test_response` command:

```cpp
CommandResult cmdTestResponse(const std::string& arg);
```

**Tests include**:
1. Basic response retrieval
2. Response variety (ensures non-repetition)
3. Contextual greeting generation
4. Parameter substitution
5. Literal message passthrough
6. New acknowledgment responses

**Running Tests**:
When the project is built, you can run:
```
test_response
```

This will output a detailed test report showing pass/fail status for each feature.

## Usage Examples

### Example 1: Using Acknowledgments
```cpp
// In a command handler
std::string ack = ResponseManager::get("ack_working");
history.push(ack, Colors::Cyan.toUInt());
Voice::speak(ack, "system");

// ... do work ...

std::string done = ResponseManager::get("ack_done");
history.push(done, Colors::Green.toUInt());
Voice::speak(done, "system");
```

### Example 2: Contextual Startup
```cpp
// In startup code
std::string greeting = ResponseManager::getGreeting();
std::string startup = ResponseManager::get("startup");
std::string fullMessage = greeting + " " + startup;
Voice::speak(fullMessage, "system");
```

### Example 3: Error Handling with New Codes
```cpp
if (!networkAvailable) {
    return ErrorManager::report("ERR_NETWORK_UNAVAILABLE");
}
```

## Implementation Details

### Thread Safety

The response system is fully thread-safe:

**Response History Protection**:
```cpp
static std::mutex responseHistoryMutex;

// All access to responseHistory is protected
std::lock_guard<std::mutex> lock(responseHistoryMutex);
```

**Thread-Safe Time Conversion**:
```cpp
// Uses platform-specific thread-safe functions
#ifdef _WIN32
    localtime_s(&localTimeBuf, &now);  // Windows thread-safe variant
#else
    localtime_r(&now, &localTimeBuf);  // POSIX thread-safe variant
#endif
```

This ensures the response system can be safely called from multiple threads simultaneously, such as:
- Voice processing thread
- UI event handlers
- AI background tasks
- Multiple concurrent command handlers

### Response History Structure
```cpp
static std::unordered_map<std::string, std::deque<size_t>> responseHistory;
static std::mutex responseHistoryMutex;
static const size_t MAX_HISTORY_SIZE = 3;
```

Each response key maintains a deque of recently used variant indices. When selecting a response:
1. Collect all variants not in recent history
2. If all variants are in history, clear history and use all variants
3. Select random variant from available options
4. Add selected variant to history
5. Trim history if it exceeds MAX_HISTORY_SIZE

### Time-Based Greeting Logic
```cpp
std::time_t now = std::time(nullptr);
std::tm localTimeBuf;

// Thread-safe time conversion
#ifdef _WIN32
    localtime_s(&localTimeBuf, &now);  // Windows
#else
    localtime_r(&now, &localTimeBuf);  // POSIX
#endif

int hour = localTimeBuf.tm_hour;

if (hour >= 5 && hour < 12) -> "greeting_morning"
else if (hour >= 12 && hour < 17) -> "greeting_afternoon"
else if (hour >= 17 && hour < 22) -> "greeting_evening"
else -> "greeting_night"
```

### Parameter Substitution Algorithm
```cpp
// Find and replace all {param_name} with actual values
for (const auto& [paramName, paramValue] : params) {
    std::string placeholder = "{" + paramName + "}";
    // Replace all occurrences
    while ((pos = response.find(placeholder, pos)) != npos) {
        response.replace(pos, placeholder.length(), paramValue);
        pos += paramValue.length();
    }
}
```

## Design Decisions

### Why History Size = 3?
- Small enough to fit in memory
- Large enough to prevent immediate repetition
- Optimal for response sets of 3-4 variants (most common)
- Can be adjusted via MAX_HISTORY_SIZE constant if needed

### Why Time-Based, Not Machine Learning?
- Maintains offline-first design principle
- No training data required
- Deterministic and predictable behavior
- Fast and lightweight
- Easy to understand and debug

### Why Parameter Substitution Instead of Format Strings?
- Simple placeholder syntax: `{name}`
- Type-safe (all parameters are strings)
- Easy to extend with more complex substitution rules later
- Compatible with JSON response definitions

## Future Enhancements

Potential future improvements (not currently implemented):

1. **Response Mood System**: Adjust response tone based on user interaction patterns
2. **Custom Response Loading**: Load user-defined responses from external files
3. **Response Analytics**: Track which responses are most effective
4. **Multi-Language Support**: Select responses based on user language preference
5. **Response Chaining**: Link multiple responses for complex multi-step interactions
6. **Verbosity Levels**: Different response sets for terse/normal/verbose modes

## Migration Guide

Existing code continues to work without modification. To adopt new features:

### Before:
```cpp
std::string msg = "Opening " + appName;
history.push(msg, Colors::Green.toUInt());
```

### After (with parameter substitution):
```cpp
std::unordered_map<std::string, std::string> params;
params["app"] = appName;
std::string msg = ResponseManager::getWithParams("open_app_success", params);
history.push(msg, Colors::Green.toUInt());
```

### Before:
```cpp
std::string response = "Got it.";
```

### After (with variety):
```cpp
std::string response = ResponseManager::get("ack_understood");
// Will vary between "Got it.", "Understood.", "Okay.", etc.
```

## Backwards Compatibility

All existing functionality is preserved:
- `ResponseManager::get()` works exactly as before for existing response keys
- Literal messages are passed through unchanged
- `ResponseManager::systemMessage()` unchanged
- No breaking changes to the public API

New features are opt-in and don't affect existing command implementations.

## Performance Impact

- **Memory**: Minimal (< 1KB for history tracking)
- **CPU**: Negligible (simple random selection and string replacement)
- **Startup**: No impact (responses are static data)
- **Runtime**: Microseconds per response selection

## Summary

These improvements make the G.R.I.M response system more intelligent and natural while maintaining its lightweight, offline-first design. The changes are minimal, focused, and backwards-compatible, following the project's design principles.
