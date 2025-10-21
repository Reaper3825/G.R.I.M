#include "popup_anim.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================
// Update animation state (alpha + scale)
// ===========================================================
void updateAnim(PopupAnimState& state, bool visible, float dtSeconds, float timeConstant) {
    // Time-based exponential smoothing toward target values. This keeps the
    // animation frame-rate independent and stable for small render targets.
    float targetAlpha = visible ? 1.f : 0.f;
    float targetScale = visible ? 1.f : 0.9f;

    // Avoid divide-by-zero and clamp dt
    if (dtSeconds <= 0.f) dtSeconds = 0.016f;
    if (timeConstant <= 0.f) timeConstant = 0.08f;

    // Exponential approach: new = target + (current - target) * exp(-dt / tau)
    float k = std::exp(-dtSeconds / timeConstant);
    state.alpha = targetAlpha + (state.alpha - targetAlpha) * k;
    state.scale = targetScale + (state.scale - targetScale) * k;

    // Snap small deltas to target to avoid tiny residuals
    if (std::fabs(state.alpha - targetAlpha) < 0.001f) state.alpha = targetAlpha;
    if (std::fabs(state.scale - targetScale) < 0.001f) state.scale = targetScale;

    state.showing = visible;
}

// ===========================================================
// Update voice-reactive animation (LESS CREEPY VERSION)
// ===========================================================
void updateVoiceAnim(PopupAnimState& state, bool isSpeaking, float dtSeconds) {
    if (dtSeconds <= 0.f) dtSeconds = 0.016f;

    // Pulse animation when speaking (subtle bounce, not breathing)
    if (isSpeaking) {
        static float pulseTime = 0.0f;
        pulseTime += dtSeconds * 2.0f; // 2 Hz - less frantic
        
        // Single bounce per speech burst (fade out naturally)
        state.pulse = std::exp(-pulseTime * 0.5f) * 0.5f; // Quick decay, max 0.5
        
        // Increase voice intensity
        float targetIntensity = 1.0f;
        float k = std::exp(-dtSeconds / 0.15f); // Smooth approach
        state.voiceIntensity = targetIntensity + (state.voiceIntensity - targetIntensity) * k;
        
        // Reset pulse time when starting speech
        static bool wasSpeaking = false;
        if (!wasSpeaking) {
            pulseTime = 0.0f;
        }
        wasSpeaking = true;
    } else {
        // Fade out pulse quickly
        state.pulse *= std::exp(-dtSeconds / 0.3f);
        
        // Decrease voice intensity
        float targetIntensity = 0.0f;
        float k = std::exp(-dtSeconds / 0.5f); // Slower fade
        state.voiceIntensity = targetIntensity + (state.voiceIntensity - targetIntensity) * k;
    }

    // Subtle idle animation (very minimal, no breathing)
    // Just a slight shimmer effect via alpha variation
    static float idleTime = 0.0f;
    idleTime += dtSeconds * 0.3f; // Very slow
    state.breathe = (std::sin(idleTime * 2.0f * M_PI) + 1.0f) * 0.5f; // 0-1 range

    // Apply combined scale effect
    // Only scale when speaking - single bounce, no idle breathing
    float bounceScale = state.pulse * 0.08f * state.voiceIntensity; // Single bounce ±8%
    state.scale = 1.0f + bounceScale;

    // Clamp scale to reasonable bounds
    if (state.scale < 0.95f) state.scale = 0.95f;
    if (state.scale > 1.08f) state.scale = 1.08f;
}
