#pragma once
#include <vector>
#include <string>
#include <mutex>
#include <nlohmann/json_fwd.hpp>

// Forward declarations
class ConsoleHistory;
struct Timer;
class NLP;
struct whisper_context;

namespace VoiceStream {

// ---------------- State ----------------
struct State {
    struct AudioData {
        std::mutex mtx;
        std::vector<float> buffer;
        bool ready = false;
    };

    AudioData audio;
    bool running = false;
    std::string partial;
    size_t processedSamples = 0;
    int inputDeviceIndex = -1;
};

// Global state instance
extern State g_state;

// ---------------- Control API ----------------
bool start(whisper_context* ctx,
           ConsoleHistory* history,
           std::vector<Timer>& timers,
           nlohmann::json& longTermMemory,
           NLP& nlp);

void stop();

bool isRunning();

void calibrateSilence();

} // namespace VoiceStream
