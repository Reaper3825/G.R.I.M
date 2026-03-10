// EmotionPresentationController — runtime-togglable outward emotion layer.
//
// Separates internal mood state (PersonalityManager, used by routing,
// memory, reward, and action policy) from external presentation
// (UI text styling, voice tone hints, avatar expression).
//
// When disabled, all presentation queries return neutral defaults.
// Internal mood still drives routing and memory — only outward
// expression is suppressed.
//
// Thread-safe: all operations lock-free on atomics.
//======================================================//
#pragma once

#include <atomic>
#include <string>

namespace GRIM::MMO {

// =========================================================
// PresentationChannel — which output channel to affect
// =========================================================
enum class PresentationChannel : uint8_t {
    Text   = 0,  // response text styling (prefix, emoji, tone words)
    Voice  = 1,  // TTS parameters (pitch, rate, emphasis)
    Avatar = 2   // visual avatar expression (face, posture)
};

// =========================================================
// EmotionPresentation — the outward expression payload
//
// Consumers (UI, TTS, avatar renderer) read this to determine
// how to present the response. When emotion is disabled, all
// fields contain neutral/default values.
// =========================================================
struct EmotionPresentation {
    std::string mood_label;       // "calm", "curious", etc. or "neutral" when disabled
    std::string text_prefix;      // e.g. "Hmm, " for Curious mood, empty when disabled
    float       voice_pitch   = 1.0f;  // multiplier (1.0 = neutral)
    float       voice_rate    = 1.0f;  // multiplier (1.0 = neutral)
    float       voice_emphasis = 0.5f; // 0..1 range
    std::string avatar_expression;     // e.g. "smile", "frown", "neutral"
    float       intensity     = 0.0f;  // 0..1 overall expression intensity
};

// =========================================================
// EmotionPresentationController
//
// Usage:
//   auto& epc = EmotionPresentationController::instance();
//   epc.setEnabled(true);   // or false to suppress
//   auto pres = epc.currentPresentation();
//   // Use pres.text_prefix, pres.voice_pitch, etc.
//
// Internal mood comes from PersonalityManager::get().
// This controller only controls outward expression mapping.
// =========================================================
class EmotionPresentationController {
public:
    static EmotionPresentationController& instance();

    // Enable/disable outward emotion expression.
    // When disabled, currentPresentation() returns neutral defaults.
    void setEnabled(bool enabled);
    bool isEnabled() const;

    // Enable/disable individual channels.
    void setChannelEnabled(PresentationChannel channel, bool enabled);
    bool isChannelEnabled(PresentationChannel channel) const;

    // Get the current outward emotion presentation.
    // Reads PersonalityManager::get() internally.
    EmotionPresentation currentPresentation() const;

    // Get presentation for a specific mood string (for testing/preview).
    EmotionPresentation presentationForMood(const std::string& mood_label) const;

private:
    EmotionPresentationController() = default;

    // Map internal mood to outward presentation (when enabled).
    EmotionPresentation mapMood(const std::string& mood_label,
                                float energy,
                                float confidence) const;

    std::atomic<bool> enabled_{true};
    std::atomic<bool> text_enabled_{true};
    std::atomic<bool> voice_enabled_{true};
    std::atomic<bool> avatar_enabled_{true};
};

} // namespace GRIM::MMO
