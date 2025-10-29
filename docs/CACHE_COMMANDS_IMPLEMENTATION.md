# TTS Cache Management Commands - Implementation Summary

## Issue Fixed
The "Yes?" audio file was playing twice because the actual WAV file contained the phrase spoken twice during recording. This was a data issue, not a code bug.

## Solution Implemented
Created cache management commands to allow clearing corrupted/duplicate TTS cache entries.

## New Files Created

### 1. `commands/commands_grim.hpp`
- Header file defining GRIM system management command signatures
- Exports: `cmdClearCache()` and `cmdResetCache()`

### 2. `commands/commands_grim.cpp`
- Implementation of cache management commands
- **cmdClearCache()**: Complete wipe - removes ALL cached audio files
- **cmdResetCache()**: Wipe + regenerate - removes all files AND regenerates common phrases

## Modified Files

### 1. `plugins/core_plugin.cpp`
- Added include: `#include "commands_grim.hpp"`
- Registered two new commands:
  - `grim_register_command("clear_cache", cmdClearCache)`
  - `grim_register_command("reset_cache", cmdResetCache)`

### 2. `resources/nlp_rules.json`
- Added NLP pattern for "clear_cache":
  ```json
  {
    "intent": "clear_cache",
    "description": "Clear TTS cache (wipe all cached audio files)",
    "pattern": "^(?:hey\\s+grim[, ]*|grim[, ]*|please\\s+)?(?:clear|clean|wipe|delete)\\s+(?:tts\\s+)?cache\\s*$"
  }
  ```
  
- Added NLP pattern for "reset_cache":
  ```json
  {
    "intent": "reset_cache",
    "description": "Reset TTS cache (wipe and regenerate common phrases)",
    "pattern": "^(?:hey\\s+grim[, ]*|grim[, ]*|please\\s+)?(?:reset|rebuild|regenerate|refresh)\\s+(?:tts\\s+)?cache\\s*$"
  }
  ```

## Usage

### Clear Cache (Complete Wipe)
**Voice Commands:**
- "Clear cache"
- "Clear TTS cache"
- "Wipe cache"
- "Delete cache"

**What it does:**
1. Shuts down cache system
2. Deletes cache index file
3. Removes ALL temporary `.wav` files
4. Removes ALL permanent cache `.wav` files
5. Reinitializes empty cache
6. Cache rebuilds automatically on next TTS request

**Output:** "TTS cache cleared. Removed X cached audio files. Cache will rebuild as needed."

### Reset Cache (Wipe + Regenerate)
**Voice Commands:**
- "Reset cache"
- "Reset TTS cache"
- "Rebuild cache"
- "Regenerate cache"
- "Refresh cache"

**What it does:**
1. Shuts down cache system
2. Deletes cache index file
3. Removes ALL temporary `.wav` files
4. Removes ALL permanent cache `.wav` files
5. Reinitializes empty cache
6. **Immediately regenerates common phrases** (Yes, No, Okay, etc.)

**Output:** "TTS cache reset complete. Removed X old files and regenerated common phrases."

## Common Phrases Regenerated (Reset Only)
The `reset_cache` command regenerates these frequently-used phrases:
- "Yes", "Yes?"
- "No"
- "Okay"
- "I'm listening"
- "How can I help you?"
- "Understood"
- "Processing"
- "Done"
- "Ready"
- "Error occurred"
- "Please wait"
- "Command executed"
- "I'm here"
- "Go ahead"
- "Affirmative"
- "Negative"
- "Acknowledged"

## Technical Details

### Cache Locations
- **Temp**: `D:/G.R.I.M/resources/tts_out/temp/*.wav`
- **Permanent**: `D:/G.R.I.M/resources/tts_out/cache/*.wav`
- **Index**: `D:/G.R.I.M/resources/tts_out/cache_index.json`

### Error Handling
Both commands include:
- Try-catch blocks for filesystem exceptions
- Detailed logging via LOG_DEBUG/LOG_ERROR
- User-friendly error messages
- Graceful fallback on failure

### Return Values
**Success:**
```cpp
{
    true,                              // success
    "TTS cache cleared/reset...",      // message
    "ERR_NONE",                        // errorCode
    "routine",                         // category
    "Cache cleared successfully",      // voice
    Colors::Green                      // color
}
```

**Failure:**
```cpp
{
    false,                         // success
    "Failed to clear cache: ...",  // message
    "ERR_CACHE_CLEAR_FAILED",      // errorCode
    "error",                       // category
    "Cache clear failed",          // voice
    Colors::Red                    // color
}
```

## Build Integration
- Files automatically included via `file(GLOB COMMAND_SOURCES "commands/*.cpp")`
- No CMakeLists.txt changes required
- Header automatically picked up by glob pattern
- Compiles with existing build configuration

## Testing

### To Fix Corrupted "Yes?" Audio:
1. Say "GRIM, reset cache"
2. Wait ~30-60 seconds (XTTS v2 regenerates all common phrases)
3. Say "GRIM" again
4. Should hear clean "Yes?" spoken correctly (once)

### To Quick-Wipe Cache:
1. Say "GRIM, clear cache"
2. All cache files deleted
3. Next TTS request regenerates on-demand

## Implementation Notes

### Differences Between Commands

| Feature | `clear_cache` | `reset_cache` |
|---------|---------------|---------------|
| Delete temp files | ? Yes | ? Yes |
| Delete permanent cache | ? Yes | ? Yes |
| Delete cache index | ? Yes | ? Yes |
| Regenerate common phrases | ? No | ? Yes |
| Speed | Instant | 30-60s (regeneration) |
| Use case | Quick wipe | Fix corrupted audio |

### Key Functions Used
- `Voice::TTSCache::shutdown()` - Safely closes cache system
- `Voice::TTSCache::init()` - Reinitializes empty cache
- `Voice::preCacheCommonPhrases()` - Generates 17 common phrases (reset only)

### Why Two Commands?

**`clear_cache`** - Fast cleanup for general maintenance
- No waiting for regeneration
- Good for freeing disk space
- Cache rebuilds naturally during use

**`reset_cache`** - Fix corrupted/duplicate audio
- Ensures critical phrases ("Yes?", "Ready", etc.) are clean
- Prevents double-speaking issues
- Immediate availability of common phrases

## Root Cause of Original Issue
The "Yes?" audio file contained the phrase spoken twice **in the source recording**, not due to code duplication. The cache preserved this corrupted recording. Running `reset_cache` generates fresh, clean audio.
