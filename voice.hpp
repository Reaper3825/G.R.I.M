#pragma once
#include <string>

// Runs a short microphone recording (5s) and transcribes it with Whisper.
// Prints transcription to console/stdout.
int runVoiceDemo(const std::string &modelPath);
