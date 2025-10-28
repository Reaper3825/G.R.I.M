# TTS Cache Path Bug Fix

## ?? The Bug

When TTS generated audio, the cache system would:
1. Generate audio in `temp/` directory
2. Move it to `cache/` directory  
3. Return the **old temp path** to the player
4. Player tries to play from temp ? **File not found!** ?

### Error Log:
```
[TTSCache] Moved file: temp\1CV2Z...wav ? cache\1CV2Z...wav
[Voice/Audio] File not found: temp\1CV2Z...wav  ? BUG!
```

## ? The Fix

### Changed Files:

**1. `voice_speak.cpp` - `coquiSpeak()`**
```cpp
// OLD (wrong):
TTSCache::store(text, speaker, speed, generatedFile);
return generatedFile;  // Returns temp path ?

// NEW (correct):
std::string finalPath = TTSCache::store(text, speaker, speed, generatedFile);
return finalPath;  // Returns cache path ?
```

**2. `tts_cache.hpp` - Function signature**
```cpp
// OLD:
void store(...);

// NEW:
std::string store(...);  // Returns final cached path
```

**3. `tts_cache.cpp` - Implementation**
```cpp
std::string store(...)
{
    // ...move file logic...
    
    // OLD:
    saveIndex();
    // (no return value)
    
    // NEW:
    saveIndex();
    return destPath.string();  // Return where file actually is
}
```

## ?? Result

### Before:
```
1. Generate: temp\abc.wav
2. Move to: cache\abc.wav
3. Play from: temp\abc.wav ? (doesn't exist)
4. Error: "File not found"
```

### After:
```
1. Generate: temp\abc.wav
2. Move to: cache\abc.wav
3. Return: cache\abc.wav ?
4. Play from: cache\abc.wav ? (exists!)
5. Success! ??
```

## ?? Testing

### Test Case 1: First Synthesis
```
Input: "How are you?"
Expected: 
  - File generated in temp/
  - File moved to cache/
  - Audio plays successfully ?
```

### Test Case 2: Cached Phrase
```
Input: "Yes" (pre-cached)
Expected:
  - Retrieved from cache/
  - No temp file created
  - Audio plays immediately ?
```

### Test Case 3: System Message
```
Input: System message (permanent cache)
Expected:
  - File generated in temp/
  - File moved to cache/ (permanent)
  - Audio plays successfully ?
  - Survives cleanup
```

## ?? Impact

**Files Fixed:**
- ? `voice_speak.cpp`
- ? `tts_cache.cpp`
- ? `tts_cache.hpp`

**Bugs Fixed:**
- ? "File not found" errors after cache move
- ? Audio playback failures
- ? Wasted synthesis (file exists but can't be played)

**Performance Improvement:**
- No more failed playbacks
- Cache actually works now
- No re-synthesis due to "missing" files

## ?? How It Was Found

**User Report:**
```
[ERROR][Voice/Audio] File not found: D:/G.R.I.M/resources/tts_out/temp\1CV2Z...wav
```

**Investigation:**
```
[TTSCache] Moved file: temp\... ? cache\...  ? File moved
[Voice/Audio] File not found: temp\...       ? Looking in wrong place!
```

**Root Cause:**
`store()` function moved the file but didn't communicate the new location to the caller.

## ?? Key Lesson

**Always return the actual location of resources after moving/copying them.**

If a function moves a file, it should:
1. ? Move the file
2. ? Return the new path
3. ? Let caller use the new path

Don't assume the caller remembers where you moved it!

---

**Status:** ? Fixed  
**Build:** Successful  
**Ready for:** Testing
