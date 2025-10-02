#include "wake_voice.hpp"
#include "wake.hpp"     // to control Wake::g_awake
#include "logger.hpp"

#include <atomic>

namespace WakeVoice {
    // You could track last state so we don't spam logs every frame
    static std::atomic<bool> lastAwake{false};

    void update() {
        // TODO: Replace with actual voice wake detection (Whisper/Coqui)
        bool detectedWakeWord = false;

        // Example: simulate wake word if GRIM is asleep
        // (later this will be replaced with actual audio analysis)
        if (!Wake::g_awake.load()) {
            // stub: you might set detectedWakeWord = true from a real recognizer
        }

        if (detectedWakeWord && !Wake::g_awake.load()) {
            Wake::g_awake.store(true);
            logDebug("WakeVoice", "Wake word detected — GRIM is now awake!");
        }

        // Optional: only log check occasionally to avoid spamming
        if (Wake::g_awake.load() != lastAwake.load()) {
            lastAwake.store(Wake::g_awake.load());
            if (lastAwake.load()) {
                logTrace("WakeVoice", "Now awake");
            } else {
                logTrace("WakeVoice", "Now asleep");
            }
        }
    }
}
