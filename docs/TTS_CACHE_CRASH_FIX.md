# TTS Cache Crash Fix

## ?? The Crash

**Exit Code:** `-1073741819` (0xC0000005 = Access Violation)

**When:** Playing cached TTS audio  
**Why:** Old cache entries pointed to files in `temp/` that were moved to `cache/`

### Crash Log:
```
[TTSCache] Cache HIT for: Yes?...
[Voice/Coqui] Using cached audio: D:/G.R.I.M/resources/tts_out/temp\cCkFd...wav
(CRASH - file doesn't exist)
```

## ?? Root Cause

### The Problem Sequence:
1. **Old behavior:** Files stayed in `temp/`, cache index pointed to `temp/`
2. **Code update:** Files now move from `temp/` ? `cache/`
3. **Cache index:** Still had old entries pointing to `temp/`
4. **Result:** Tried to play file that doesn't exist ? **CRASH!**

### Why It Crashed:
The audio playback system tried to:
1. Open a `.wav` file from the cached path
2. File didn't exist (was moved)
3. Invalid file handle ? Access Violation ? **CRASH**

## ? The Fix (2-Part Solution)

### Part 1: Safety Check (Code Fix)
**File:** `tts_cache.cpp`

Added file existence validation in `getCached()`:

```cpp
std::string getCached(...)
{
    auto it = g_cacheIndex.find(key);
    if (it != g_cacheIndex.end())
    {
        // ? NEW: Verify file exists
        if (fs::exists(it->second.filePath))
        {
            return it->second.filePath;  // ? Safe
        }
        else
        {
            // ? Remove invalid entry
            LOG_DEBUG("TTSCache", "Cache entry invalid (file missing), removing");
            g_cacheIndex.erase(it);
            saveIndex();
            return "";  // ? Return empty = cache miss
        }
    }
    return "";  // Cache miss
}
```

**Benefits:**
- ? No crashes from invalid paths
- ? Self-healing cache (removes bad entries)
- ? Automatic re-generation of missing files

### Part 2: Cache Reset (Manual Fix)
**Script:** `scripts/reset_tts_cache.ps1`

```powershell
# Run this to clear old cache entries
.\scripts\reset_tts_cache.ps1
```

**What it does:**
1. Backs up old cache index
2. Deletes old cache index (will rebuild)
3. Cleans temp directory
4. Cache rebuilds automatically on next use

## ?? How to Fix Your System

### Option A: Run the Reset Script (Recommended)
```powershell
cd D:\G.R.I.M
.\scripts\reset_tts_cache.ps1
```

### Option B: Manual Cleanup
```powershell
# Delete the cache index
Remove-Item "D:\G.R.I.M\resources\tts_out\cache_index.json" -Force

# Clean temp directory
Remove-Item "D:\G.R.I.M\resources\tts_out\temp\*.wav" -Force

# Restart GRIM
```

### Option C: Just Restart (Lazy Fix)
With the new code fix, you can just:
1. Restart GRIM
2. First TTS will regenerate missing files
3. Cache will auto-clean bad entries

## ?? Expected Behavior After Fix

### Before (Crash):
```
[TTSCache] Cache HIT for: Yes?...
[Voice/Coqui] Using cached audio: temp\cCkFd...wav
(CRASH - file doesn't exist) ?
```

### After (Safe):
```
[TTSCache] Cache HIT for: Yes?...
[TTSCache] Cache entry invalid (file missing), removing
[Voice/Coqui] Generating new audio...
[TTSCache] Stored PERMANENT cache: Yes?
[Voice/Audio] Playing: cache\xyz123.wav ?
```

## ?? Testing

### Test Case 1: Invalid Cache Entry
```
1. Have old cache entry pointing to temp/
2. Try to play cached phrase
3. Expected: Auto-removes bad entry, regenerates
4. Result: ? No crash, audio plays
```

### Test Case 2: Valid Cache Entry
```
1. Have valid cache entry pointing to cache/
2. Try to play cached phrase
3. Expected: Plays immediately
4. Result: ? Instant playback
```

### Test Case 3: First Run After Reset
```
1. Clear cache index
2. Start GRIM
3. Trigger TTS
4. Expected: Pre-cache runs, rebuilds cache
5. Result: ? All common phrases cached
```

## ?? Files Changed

### Code Changes:
- ? `voice/tts_cache.cpp` - Added file existence check
- ? `voice/tts_cache.hpp` - (no changes needed)

### New Files:
- ? `scripts/reset_tts_cache.ps1` - Cache reset utility

### No Changes Needed:
- `voice_speak.cpp` - Already fixed to use returned path
- `voice_speak.hpp` - Already updated

## ?? Prevention

### How to Avoid This in Future:

**1. Always validate file paths:**
```cpp
// ? DON'T:
return cachedPath;

// ? DO:
if (fs::exists(cachedPath)) {
    return cachedPath;
} else {
    handleMissing();
    return "";
}
```

**2. Handle cache migrations:**
When changing file locations:
1. Update the code to use new locations ?
2. Migrate existing cache entries ?
3. Add validation for old entries ?
4. Document the migration ?

**3. Self-healing caches:**
Caches should:
- Validate entries before use ?
- Remove invalid entries ?
- Regenerate missing data ?

## ?? Key Takeaways

### What We Learned:

**1. Migration Risk:**
When you change where files are stored, old cache indices can become invalid.

**2. Defense in Depth:**
- Code fix: Prevents crashes
- Reset script: Cleans up old data
- Documentation: Explains the issue

**3. Self-Healing Systems:**
The cache now automatically fixes itself:
- Detects missing files
- Removes bad entries
- Regenerates on demand

### Best Practices Applied:

? **Validation** - Check file exists before using  
? **Graceful Degradation** - Cache miss ? regenerate, don't crash  
? **Self-Healing** - Remove bad entries automatically  
? **Logging** - Clear messages about what's happening  
? **Recovery Tools** - Script to fix user systems  

## ?? Summary

**Problem:** Old cache entries caused crashes  
**Root Cause:** Files moved, cache index not updated  
**Code Fix:** Added file existence validation  
**User Fix:** Run reset script or just restart  
**Result:** No more crashes, self-healing cache

---

**Status:** ? Fixed  
**Build:** Successful  
**Action Required:** Run `reset_tts_cache.ps1` or restart GRIM
