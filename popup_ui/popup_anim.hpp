#pragma once

// ===========================================================
// Popup animation helpers
// ===========================================================

struct PopupAnimState {
    float alpha   = 0.5f;   // 0 = hidden, 1 = shown
    float scale   = 0.9f;  // start slightly smaller
    bool  showing = false; // whether animation is expanding
};

// Update animation state toward target visibility.
// dtSeconds: time elapsed since last update (seconds).
// timeConstant: time to approach the target (seconds); smaller = faster.
void updateAnim(PopupAnimState& state, bool visible, float dtSeconds = 0.016f, float timeConstant = 0.08f);
