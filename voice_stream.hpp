#pragma once
#include <vector>
#include <string>
#include <nlohmann/json_fwd.hpp>


class ConsoleHistory;
class Timer;
class NLP;
struct whisper_context;

namespace VoiceStream {

struct State {
    struct AudioData {
        std::vector<float> buffer;
        bool ready = false;
    };
    AudioData audio;
    bool running = false;
    std::string partial;
    size_t processedSamples = 0;
    int inputDeviceIndex = -1;
};

// Global state (defined in voice_stream.cpp)
extern State g_state;

/// Start streaming transcription in a background thread.
/// Returns true if started successfully.
bool start(whisper_context* ctx,
           ConsoleHistory* history,
           std::vector<Timer>& timers,
           nlohmann::json& longTermMemory,
           NLP& nlp);

/// Stop streaming if running.
void stop();

/// Check if the stream is currently active.
bool isRunning();

/// Calibrate silence threshold from mic input
void calibrateSilence();

} // namespace VoiceStream
