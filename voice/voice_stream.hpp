#pragma once
#include <string>
#include <vector>
#include <mutex>
#include <nlohmann/json.hpp>
#include <SFML/Graphics/Color.hpp>
#include "nlp/nlp.hpp"
#include "timer.hpp"
#include "console_history.hpp"

struct whisper_context;

namespace VoiceStream {
    struct State {
        struct AudioData {
            std::vector<float> buffer;
            bool ready = false;
            std::mutex mtx;
        };

        bool running = false;
        int inputDeviceIndex = -1;
        size_t processedSamples = 0;
        std::string partial;
        AudioData audio;
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

namespace Voice {
    // One-shot speech capture after a wake event.
    // Blocks until user finishes speaking or silence timeout.
    std::string listenOnce();

    // Access whisper context (implemented in voice.cpp / ai.cpp)
    whisper_context* getWhisperContext();
}
