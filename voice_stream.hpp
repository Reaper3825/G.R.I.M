#pragma once
#include <whisper.h>
#include <vector>
#include <nlohmann/json.hpp>
#include "nlp.hpp"
#include "console_history.hpp"
#include "timer.hpp"

// ---------------- Voice Stream ----------------
void runVoiceStream(whisper_context *ctx,
                    ConsoleHistory* history,
                    std::vector<Timer>& timers,
                    nlohmann::json& longTermMemory,
                    NLP& nlp);

// ---------------- Calibration ----------------
/// Calibrate the silence baseline using mic input.
/// - Samples ~2 seconds of audio.
/// - Updates g_silenceThreshold.
/// - Persists to memory.json.
void calibrateSilence();
