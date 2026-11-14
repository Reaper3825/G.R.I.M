#pragma once
// Apple TTS engine using AVSpeechSynthesizer (macOS 10.14+).
// Used as the platform TTS when the primary engine (Coqui) is unavailable.

#include <string>

namespace AppleTTS {

// Initialize AVSpeechSynthesizer. Returns true on success.
bool init();

// Tear down synthesizer resources.
void shutdown();

// Returns true once init() has succeeded.
bool isReady();

// Speak text synchronously — blocks until utterance finishes.
// speed: multiplier around 1.0 (maps to AVSpeechUtteranceDefaultRate).
// pitch: multiplier around 1.0.
// Returns true if speech completed normally.
bool speak(const std::string& text, float speed, float pitch);

// Returns true while an utterance is being spoken.
bool isSpeaking();

} // namespace AppleTTS
