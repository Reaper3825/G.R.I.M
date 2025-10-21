#pragma once

// ===========================================================
// Popup animation helpers
// ===========================================================

struct PopupAnimState {
    float alpha   = 0.0f;   // 0 = hidden, 1 = shown (start hidden)
    float scale   = 0.8f;   // start smaller for scale-in effect
    bool  showing = false;  // whether animation is expanding
    bool  wasShowing = false; // track previous visibility state
    
    // Voice-reactive animation
    float pulse   = 0.0f;   // pulsing effect (0-1) when speaking
    float breathe = 0.0f;   // breathing/idle animation
    float voiceIntensity = 0.0f; // 0-1 based on audio activity
};

// Update animation state toward target visibility.
// dtSeconds: time elapsed since last update (seconds).
// timeConstant: time to approach the target (seconds); smaller = faster.
void updateAnim(PopupAnimState& state, bool visible, float dtSeconds = 0.016f, float timeConstant = 0.08f);

// Update voice-reactive animations
// isSpeaking: whether GRIM is currently speaking
// dtSeconds: time delta
void updateVoiceAnim(PopupAnimState& state, bool isSpeaking, float dtSeconds = 0.016f);
