#pragma once
#include <vector>
#include <string>
#include <atomic>
#include <nlohmann/json.hpp>  // ✅ real header, no forward declare

struct ConsoleHistory;
struct Timer;

namespace Voice {
    bool initWakeWord(const std::string& accessKey,
                      const std::string& modelPath,
                      const std::string& keywordPath);
    bool detectWakeWordFrame(const int16_t* pcm);
    void shutdownWakeWord();
}

namespace WakeVoice {
    void start(ConsoleHistory* history,
               std::vector<Timer>& timers,
               nlohmann::json& longTermMemory);
    void stop();
    bool isRunning();
}
