#include "popup_anim.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================
// Easing functions for smoother animation
// ===========================================================
static float easeOutCubic(float t) {
    float f = t - 1.0f;
    return f * f * f + 1.0f;
}

static float easeInOutCubic(float t) {
    if (t < 0.5f)
        return 4.0f * t * t * t;
    else {
        float f = 2.0f * t - 2.0f;
        return 0.5f * f * f * f + 1.0f;
    }
}

// ===========================================================
// Update animation state (alpha + scale) with SMOOTH EASING
// ===========================================================
void updateAnim(PopupAnimState& state, bool visible, float dtSeconds, float timeConstant) {
    // Detect visibility change
    if (visible && !state.wasShowing) {
        // Just became visible - start from scaled down and transparent
        state.alpha = 0.0f;
        state.scale = 0.8f;
    }
    state.wasShowing = visible;
    
    // Time-based exponential smoothing toward target values
    float targetAlpha = visible ? 1.f : 0.f;
    float targetScale = visible ? 1.f : 0.8f; // Scale from 80% to 100% when showing

    // Avoid divide-by-zero and clamp dt
    if (dtSeconds <= 0.f) dtSeconds = 0.016f;
    if (timeConstant <= 0.f) timeConstant = 0.08f;

    // Smoother exponential approach with easing
    float k = std::exp(-dtSeconds / timeConstant);
    
    float newAlpha = targetAlpha + (state.alpha - targetAlpha) * k;
    float newScale = targetScale + (state.scale - targetScale) * k;
    
    // Apply easing for smoother visual transition
    float alphaDelta = newAlpha - state.alpha;
    state.alpha = state.alpha + alphaDelta * easeOutCubic(1.0f - k);
    
    float scaleDelta = newScale - state.scale;
    state.scale = state.scale + scaleDelta * easeOutCubic(1.0f - k);

    // Snap small deltas to target to avoid tiny residuals
    if (std::fabs(state.alpha - targetAlpha) < 0.001f) state.alpha = targetAlpha;
    if (std::fabs(state.scale - targetScale) < 0.001f) state.scale = targetScale;

    state.showing = visible;
}

// ===========================================================
// Update voice-reactive animation (SMOOTH, NOT CREEPY)
// ===========================================================
void updateVoiceAnim(PopupAnimState& state, bool isSpeaking, float dtSeconds) {
    if (dtSeconds <= 0.f) dtSeconds = 0.016f;

    // Pulse animation when speaking (subtle bounce with smooth decay)
    static bool wasSpeaking = false;
    
    if (isSpeaking) {
        static float pulseTime = 0.0f;
        
        // Reset pulse on speech start
        if (!wasSpeaking) {
            pulseTime = 0.0f;
        }
        
        pulseTime += dtSeconds * 2.0f; // 2 Hz
        
        // Smooth decay with easing
        float decayFactor = std::exp(-pulseTime * 0.6f);
        state.pulse = easeOutCubic(decayFactor) * 0.5f;
        
        // Smooth intensity increase
        float targetIntensity = 1.0f;
        float k = std::exp(-dtSeconds / 0.15f);
        float intensityDelta = targetIntensity - state.voiceIntensity;
        state.voiceIntensity += intensityDelta * (1.0f - k) * easeInOutCubic(1.0f - k);
        
        wasSpeaking = true;
    } else {
        // Smooth fade out
        state.pulse *= std::exp(-dtSeconds / 0.3f);
        
        // Smooth intensity decrease with easing
        float targetIntensity = 0.0f;
        float k = std::exp(-dtSeconds / 0.5f);
        float intensityDelta = targetIntensity - state.voiceIntensity;
        state.voiceIntensity += intensityDelta * (1.0f - k) * easeOutCubic(1.0f - k);
        
        wasSpeaking = false;
    }

    // Subtle idle animation (very minimal shimmer)
    static float idleTime = 0.0f;
    idleTime += dtSeconds * 0.3f;
    state.breathe = (std::sin(idleTime * 2.0f * M_PI) + 1.0f) * 0.5f;

    // Apply smooth scale with easing
    float bounceScale = state.pulse * 0.08f * state.voiceIntensity;
    float targetScale = 1.0f + bounceScale;
    
    // Smooth interpolation to target scale
    float scaleDiff = targetScale - state.scale;
    state.scale += scaleDiff * 0.15f; // Smooth follow

    // Clamp scale to reasonable bounds
    if (state.scale < 0.95f) state.scale = 0.95f;
    if (state.scale > 1.08f) state.scale = 1.08f;
}
