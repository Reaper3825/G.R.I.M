# Animation Troubleshooting Guide

## Issue: No Animation Visible

### Quick Diagnostics

1. **Check Console Output**
   ```
   Look for: "PopupAnim: scale=X.XX, alpha=X.XX, pulse=X.XX"
   ```
   - If scale is always 1.0 ? Animation not updating
   - If scale varies ? Animation working, rendering issue

2. **Verify Voice Detection**
   ```
   Look for: "Voice started speaking" in logs
   ```
   - Should appear when GRIM speaks
   - Check `Voice::isSpeaking()` returns true

3. **Check Popup Visibility**
   ```
   Look for: "PopupUI shown" / "PopupUI hidden"
   ```
   - Popup must be visible for animations to apply
   - Auto-shows when voice starts

### Common Problems

#### Problem: Scale stuck at 1.0
**Cause**: `updateVoiceAnim()` not being called

**Solution**: Verify loop calls:
```cpp
updateVoiceAnim(g_anim, isSpeaking, dt);
```

#### Problem: Window not updating
**Cause**: `applyAnimationToWindow()` not called

**Solution**: Check:
```cpp
if (g_popupVisible && g_hwnd) {
    applyAnimationToWindow(g_hwnd, size, size, scale, alpha);
}
```

#### Problem: Popup never appears
**Cause**: Voice detection not working

**Solution**:
1. Check `Voice::isSpeaking()` implementation
2. Verify `g_isSpeaking` atomic is set in `Voice::playAudio()`
3. Test with: `Voice::speak("test", "system")`

#### Problem: Animation freezes
**Cause**: Delta time (`dt`) is zero or very small

**Solution**: Verify timing:
```cpp
auto now = std::chrono::steady_clock::now();
std::chrono::duration<float> dtSec = now - g_frameStart;
float dt = dtSec.count(); // Should be ~0.016 (60fps)
g_frameStart = now;
```

## Verification Steps

### 1. Test Animation Math
```cpp
// In popup_anim.cpp, add temporary logging:
void updateVoiceAnim(PopupAnimState& state, bool isSpeaking, float dtSeconds) {
    LOG_DEBUG("Anim", "Input: speaking=" + std::to_string(isSpeaking) + 
              ", dt=" + std::to_string(dtSeconds));
    
    // ... existing code ...
    
    LOG_DEBUG("Anim", "Output: scale=" + std::to_string(state.scale) + 
              ", pulse=" + std::to_string(state.pulse));
}
```

### 2. Test Window Updates
```cpp
// In popup_window.cpp:
void applyAnimationToWindow(HWND hwnd, int width, int height, 
                           float scale, float alpha)
{
    LOG_DEBUG("Window", "Applying: scale=" + std::to_string(scale) + 
              ", alpha=" + std::to_string(alpha));
    
    // ... existing code ...
    
    LOG_DEBUG("Window", "Updated window to " + 
              std::to_string(scaledWidth) + "x" + std::to_string(scaledHeight));
}
```

### 3. Test Voice Events
```cpp
// In voice_speak.cpp:
void playAudio(const std::string& path) {
    LOG_DEBUG("Voice", "=== PLAYBACK START ===");
    g_isSpeaking = true;
    
    // ... playback code ...
    
    while (Audio::isPlaying()) {
        LOG_DEBUG("Voice", "Still playing...");
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    g_isSpeaking = false;
    LOG_DEBUG("Voice", "=== PLAYBACK END ===");
}
```

## Expected Behavior

### Idle State
- Scale: ~1.01 (breathing animation)
- Pulse: 0.0
- Voice Intensity: 0.0
- Window: Subtle size oscillation

### Speaking State
- Scale: 0.95 - 1.05 (breathing + pulse)
- Pulse: 0.0 - 1.0 (cycling at 3 Hz)
- Voice Intensity: ~1.0
- Window: Visible pulsing effect

### Transition
- Intensity ramps up in ~100ms
- Intensity fades out in ~300ms
- Scale smoothly interpolates
- No jarring jumps

## Debug Commands

### Force Show Popup
```cpp
showPopup();
```

### Manually Trigger Animation
```cpp
g_anim.voiceIntensity = 1.0f;
updateVoiceAnim(g_anim, true, 0.016f);
```

### Check Animation State
```cpp
LOG_DEBUG("State", "alpha=" + std::to_string(g_anim.alpha) + 
          ", scale=" + std::to_string(g_anim.scale) +
          ", pulse=" + std::to_string(g_anim.pulse) +
          ", breathe=" + std::to_string(g_anim.breathe) +
          ", intensity=" + std::to_string(g_anim.voiceIntensity));
```

## Performance Metrics

### Normal Operation
- Frame time: ~1-2ms
- CPU usage: <1%
- Memory: ~256KB

### High Load
- Frame time: <5ms
- CPU usage: <5%
- Window updates: 60 FPS

### If Exceeding
- Reduce update frequency (every 2-3 frames)
- Increase sleep time in loop
- Skip updates when intensity < 0.01

## Still Not Working?

### Check Build Configuration
- C++14 mode enabled? ?
- BGFX linked? ?
- STB_IMAGE included? ?
- PortAudio linked? ?

### Verify File Paths
```
D:/G.R.I.M/resources/shaders/g_sprite_Diffuse.png
D:/G.R.I.M/resources/shaders/g_sprite_Oreo.png
```

### Test Minimal Case
```cpp
// Create simple test that just scales window:
for (float s = 0.9f; s <= 1.1f; s += 0.01f) {
    applyAnimationToWindow(hwnd, 256, 256, s, 1.0f);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
}
```

If this works ? Problem is in animation calculation  
If this doesn't work ? Problem is in window rendering

## Contact Info

For more help, check:
- `docs/VOICE_ANIMATION_SYSTEM.md` - Full system documentation
- `docs/CONSOLE_INTEGRATION.md` - Console UI integration
- GitHub Issues - Report bugs
