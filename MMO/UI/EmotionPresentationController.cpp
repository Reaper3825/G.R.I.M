// EmotionPresentationController.cpp — maps internal mood to outward expression.
//======================================================//

#include "EmotionPresentationController.hpp"
#include "../../ai/personality_manager.hpp"

namespace GRIM::MMO {

// ─── Singleton ────────────────────────────────────────────

EmotionPresentationController& EmotionPresentationController::instance() {
    static EmotionPresentationController inst;
    return inst;
}

// ─── Enable/disable ───────────────────────────────────────

void EmotionPresentationController::setEnabled(bool enabled) {
    enabled_.store(enabled, std::memory_order_relaxed);
}

bool EmotionPresentationController::isEnabled() const {
    return enabled_.load(std::memory_order_relaxed);
}

void EmotionPresentationController::setChannelEnabled(
    PresentationChannel channel, bool enabled) {
    switch (channel) {
        case PresentationChannel::Text:   text_enabled_.store(enabled, std::memory_order_relaxed); break;
        case PresentationChannel::Voice:  voice_enabled_.store(enabled, std::memory_order_relaxed); break;
        case PresentationChannel::Avatar: avatar_enabled_.store(enabled, std::memory_order_relaxed); break;
    }
}

bool EmotionPresentationController::isChannelEnabled(
    PresentationChannel channel) const {
    switch (channel) {
        case PresentationChannel::Text:   return text_enabled_.load(std::memory_order_relaxed);
        case PresentationChannel::Voice:  return voice_enabled_.load(std::memory_order_relaxed);
        case PresentationChannel::Avatar: return avatar_enabled_.load(std::memory_order_relaxed);
    }
    return false;
}

// ─── Neutral defaults ─────────────────────────────────────

static EmotionPresentation neutralPresentation() {
    EmotionPresentation p;
    p.mood_label        = "neutral";
    p.text_prefix       = "";
    p.voice_pitch       = 1.0f;
    p.voice_rate        = 1.0f;
    p.voice_emphasis    = 0.5f;
    p.avatar_expression = "neutral";
    p.intensity         = 0.0f;
    return p;
}

// ─── Current presentation ─────────────────────────────────

EmotionPresentation EmotionPresentationController::currentPresentation() const {
    if (!enabled_.load(std::memory_order_relaxed)) {
        return neutralPresentation();
    }

    const auto& state = GRIM::PersonalityManager::get();
    std::string mood_str = GRIM::PersonalityManager::moodToString(state.mood);
    return mapMood(mood_str, state.energy, state.confidence);
}

EmotionPresentation EmotionPresentationController::presentationForMood(
    const std::string& mood_label) const {
    return mapMood(mood_label, 0.75f, 0.75f);
}

// ─── Mood → Presentation mapping ─────────────────────────

EmotionPresentation EmotionPresentationController::mapMood(
    const std::string& mood_label,
    float energy,
    float confidence) const {

    EmotionPresentation p = neutralPresentation();
    p.mood_label = mood_label;

    // Intensity scales with energy and confidence
    p.intensity = (energy + confidence) * 0.5f;

    // ─── Text channel ─────────────────────────────────────
    if (text_enabled_.load(std::memory_order_relaxed)) {
        if (mood_label == "Curious") {
            p.text_prefix = "Hmm, ";
        } else if (mood_label == "Playful") {
            p.text_prefix = "";  // playfulness shows in word choice, not prefix
        } else if (mood_label == "Irritated") {
            p.text_prefix = "";
        } else if (mood_label == "Tired") {
            p.text_prefix = "";
        } else if (mood_label == "Focused") {
            p.text_prefix = "";
        }
        // Calm = no prefix (default)
    }

    // ─── Voice channel ────────────────────────────────────
    if (voice_enabled_.load(std::memory_order_relaxed)) {
        if (mood_label == "Curious") {
            p.voice_pitch    = 1.05f;
            p.voice_rate     = 1.0f;
            p.voice_emphasis = 0.6f;
        } else if (mood_label == "Playful") {
            p.voice_pitch    = 1.08f;
            p.voice_rate     = 1.05f;
            p.voice_emphasis = 0.7f;
        } else if (mood_label == "Irritated") {
            p.voice_pitch    = 0.95f;
            p.voice_rate     = 1.1f;
            p.voice_emphasis = 0.8f;
        } else if (mood_label == "Tired") {
            p.voice_pitch    = 0.97f;
            p.voice_rate     = 0.9f;
            p.voice_emphasis = 0.3f;
        } else if (mood_label == "Focused") {
            p.voice_pitch    = 1.0f;
            p.voice_rate     = 1.02f;
            p.voice_emphasis = 0.55f;
        }
        // Calm = neutral voice (defaults)
    } else {
        p.voice_pitch    = 1.0f;
        p.voice_rate     = 1.0f;
        p.voice_emphasis = 0.5f;
    }

    // ─── Avatar channel ───────────────────────────────────
    if (avatar_enabled_.load(std::memory_order_relaxed)) {
        if (mood_label == "Curious") {
            p.avatar_expression = "curious";
        } else if (mood_label == "Playful") {
            p.avatar_expression = "smile";
        } else if (mood_label == "Irritated") {
            p.avatar_expression = "frown";
        } else if (mood_label == "Tired") {
            p.avatar_expression = "drowsy";
        } else if (mood_label == "Focused") {
            p.avatar_expression = "attentive";
        } else {
            p.avatar_expression = "neutral";
        }
    }

    return p;
}

} // namespace GRIM::MMO
