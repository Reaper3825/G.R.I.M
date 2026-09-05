#pragma once
#include <string>

namespace Voice {
    bool initWakeWord(const std::string& accessKey,
                      const std::string& modelPath,
                      const std::string& keywordPath);
    bool detectWakeWordFrame(const int16_t* pcm);
    void shutdownWakeWord();
}

namespace WakeVoice {
    void start();
    void stop();
    bool isRunning();
}
