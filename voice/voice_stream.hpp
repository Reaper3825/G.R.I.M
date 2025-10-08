#pragma once
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <nlohmann/json.hpp>
#include <SFML/Graphics/Color.hpp>
#include "nlp/nlp.hpp"
#include "timer.hpp"
#include "console_history.hpp"

struct whisper_context;

namespace VoiceStream {

    struct State
    {
        struct AudioData
        {
            std::vector<float> buffer;
            bool ready = false;
            std::mutex mtx;
        } audio;

        std::atomic<bool> running{ false };   // Thread-safe running flag
        std::thread thread;                   // Joinable thread for mic loop
        int inputDeviceIndex = -1;            // Selected mic device index
        size_t processedSamples = 0;          // Track processed sample count
        std::string partial;                  // Incremental whisper text
    };

    extern State g_state;

    bool isRunning();
    bool start(whisper_context* ctx,
               ConsoleHistory* history,
               std::vector<Timer>& timers,
               nlohmann::json& longTermMemory,
               NLP& nlp);
    void stop();
    void calibrateSilence();
}

namespace Voice
{
    // One-shot speech capture after a wake event.
    // Blocks until user finishes speaking or silence timeout.
    std::string listenOnce();

    // Access whisper context (implemented in voice.cpp / ai.cpp)
    whisper_context* getWhisperContext();
}
