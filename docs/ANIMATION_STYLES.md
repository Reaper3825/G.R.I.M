# Alternative Animation Styles

## Current Style: Subtle Bounce

**Effect**: Single bounce when speech starts, then settles  
**Intensity**: ±8% scale, 20 FPS updates  
**Glow**: Subtle white edge highlight (15% brighten)

```cpp
// Single bounce on speech start
state.pulse = exp(-time * 0.5) * 0.5; // Quick decay
state.scale = 1.0 + pulse * 0.08;     // ±8% bounce
```

---

## Style Options

### 1. **Minimal Fade** (Most Subtle)
```cpp
// Only fade alpha, no scale change
void updateVoiceAnim(PopupAnimState& state, bool isSpeaking, float dtSeconds) {
    if (isSpeaking) {
        state.voiceIntensity = 1.0f;
    } else {
        state.voiceIntensity *= exp(-dtSeconds / 1.0f); // Slow fade
    }
    
    state.scale = 1.0f; // No scaling
    state.pulse = state.voiceIntensity * 0.3f; // Gentle glow only
}
```

**Result**: Just brightens slightly when speaking

---

### 2. **Smooth Wave** (Professional)
```cpp
// Gentle sine wave while speaking
void updateVoiceAnim(PopupAnimState& state, bool isSpeaking, float dtSeconds) {
    static float waveTime = 0.0f;
    
    if (isSpeaking) {
        waveTime += dtSeconds * 1.5f; // 1.5 Hz
        state.pulse = (sin(waveTime * 2.0 * M_PI) + 1.0) * 0.5; // 0-1
        state.voiceIntensity = 1.0f;
    } else {
        state.pulse *= exp(-dtSeconds / 0.5f);
        state.voiceIntensity *= exp(-dtSeconds / 0.5f);
    }
    
    // Very subtle scale
    state.scale = 1.0f + state.pulse * 0.03f * state.voiceIntensity; // ±3%
}
```

**Result**: Smooth, slow wave motion

---

### 3. **Quick Flash** (Attention-Grabbing)
```cpp
// Single quick flash on speech start
void updateVoiceAnim(PopupAnimState& state, bool isSpeaking, float dtSeconds) {
    static bool wasSpeaking = false;
    
    if (isSpeaking && !wasSpeaking) {
        state.pulse = 1.0f; // Full intensity
    }
    
    // Fast decay
    state.pulse *= exp(-dtSeconds / 0.2f); // 200ms decay
    state.voiceIntensity = isSpeaking ? 1.0f : 0.0f;
    state.scale = 1.0f; // No scale change
    
    wasSpeaking = isSpeaking;
}
```

**Result**: Quick bright flash, then normal

---

### 4. **Notification Style** (Game UI)
```cpp
// Pop in, hold, pop out
void updateVoiceAnim(PopupAnimState& state, bool isSpeaking, float dtSeconds) {
    float targetScale = isSpeaking ? 1.05f : 1.0f;
    
    // Spring interpolation
    float k = exp(-dtSeconds / 0.1f);
    state.scale = targetScale + (state.scale - targetScale) * k;
    
    state.voiceIntensity = isSpeaking ? 1.0f : 0.0f;
    state.pulse = state.voiceIntensity;
}
```

**Result**: Pops to 105% when speaking, smooth return

---

### 5. **Typing Indicator** (Chat UI)
```cpp
// Three-dot bounce pattern
void updateVoiceAnim(PopupAnimState& state, bool isSpeaking, float dtSeconds) {
    static float bounceTime = 0.0f;
    
    if (isSpeaking) {
        bounceTime += dtSeconds * 4.0f; // 4 Hz
        
        // Three bounces pattern
        float wave = sin(bounceTime * 2.0 * M_PI);
        state.pulse = max(0.0f, wave) * 0.5f; // Only positive half
        state.voiceIntensity = 1.0f;
    } else {
        state.pulse = 0.0f;
        state.voiceIntensity = 0.0f;
    }
    
    state.scale = 1.0f + state.pulse * 0.04f; // ±4% bounce
}
```

**Result**: Rhythmic bounce pattern like "..."

---

### 6. **No Animation** (Static)
```cpp
// Completely static, only show/hide
void updateVoiceAnim(PopupAnimState& state, bool isSpeaking, float dtSeconds) {
    state.scale = 1.0f;
    state.pulse = 0.0f;
    state.breathe = 0.0f;
    state.voiceIntensity = isSpeaking ? 1.0f : 0.0f;
}
```

**Result**: No animation, popup just appears/disappears

---

## Shader Alternatives

### 1. **No Glow** (Original Colors)
```glsl
void main() {
    vec4 diffuse = texture(diffuseMap, vTexCoord);
    float op = texture(opacityMap, vTexCoord).r;
    float outA = clamp(op * animAlpha, 0.0, 1.0);
    vec3 finalRGB = vec3(diffuse.b, diffuse.g, diffuse.r);
    fragColor = vec4(finalRGB, outA);
}
```

### 2. **Pulse Border** (Edge-Only)
```glsl
void main() {
    vec4 diffuse = texture(diffuseMap, vTexCoord);
    float op = texture(opacityMap, vTexCoord).r;
    
    // Highlight only edges
    float edgeMask = smoothstep(0.9, 1.0, op); // Edges only
    vec3 edgeGlow = vec3(1.0, 1.0, 1.0) * voiceIntensity * edgeMask;
    
    vec3 finalRGB = vec3(diffuse.b, diffuse.g, diffuse.r) + edgeGlow * 0.2;
    fragColor = vec4(clamp(finalRGB, 0.0, 1.0), op * animAlpha);
}
```

### 3. **Color Shift** (Hue Change)
```glsl
void main() {
    vec4 diffuse = texture(diffuseMap, vTexCoord);
    float op = texture(opacityMap, vTexCoord).r;
    
    vec3 baseRGB = vec3(diffuse.b, diffuse.g, diffuse.r);
    vec3 activeRGB = vec3(diffuse.r, diffuse.b, diffuse.g); // Shift hue
    
    vec3 finalRGB = mix(baseRGB, activeRGB, voiceIntensity * 0.3);
    fragColor = vec4(finalRGB, op * animAlpha);
}
```

---

## How to Apply

1. **Choose Animation Style**: Pick one from above
2. **Replace in `popup_anim.cpp`**: Update `updateVoiceAnim()` function
3. **Choose Shader Effect**: Pick one from shader section
4. **Replace in `popup.frag`**: Update shader `main()` function
5. **Rebuild**: `cmake --build .`

---

## Recommendations

| Use Case | Animation | Shader | Update Rate |
|----------|-----------|--------|-------------|
| Professional App | Minimal Fade | No Glow | 10 FPS |
| Game UI | Notification Style | Pulse Border | 30 FPS |
| Chat Bot | Typing Indicator | Edge-Only | 60 FPS |
| Minimal Distraction | No Animation | Original | N/A |
| Attention Signal | Quick Flash | Edge-Only | 30 FPS |

---

## Current Settings

**Animation**: Subtle Bounce (single bounce on speech start)  
**Shader**: White edge highlight (15% brighten)  
**Update Rate**: 20 FPS (every 3 frames)  
**Scale Range**: ±8% (1.0 to 1.08)

**Changes from creepy version**:
- ? Removed constant breathing
- ? Removed cyan glow
- ? Single bounce per speech event
- ? Subtle edge highlight
- ? Lower update frequency
- ? Quick decay (not endless oscillation)
