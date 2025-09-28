#include "popup_anim.hpp"
#include <cmath>

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
