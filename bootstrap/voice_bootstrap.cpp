#include "bootstrap.hpp"

#include "../logger.hpp"
#include "../voice/voice_speak.hpp"

#include <chrono>
#include <thread>

void bootstrapVoiceSubsystem()
{
    Voice::initQueue();

    constexpr auto kReadyTimeout = std::chrono::seconds(10);
    constexpr auto kPollInterval = std::chrono::milliseconds(100);
    const auto waitStart = std::chrono::steady_clock::now();
    while (!Voice::isReady()) {
        if (std::chrono::steady_clock::now() - waitStart > kReadyTimeout) {
            LOG_ERROR("Voice", "TTS not ready after 10s - continuing without voice");
            break;
        }
        std::this_thread::sleep_for(kPollInterval);
    }

    if (Voice::isReady()) {
        Voice::speak("Welcome back, Austin. Grim is online.", "system");
        LOG_PHASE("Startup greeting spoken", true);
    } else {
        LOG_DEBUG("Voice", "Skipping startup greeting (TTS unavailable)");
        LOG_PHASE("Startup greeting spoken", false);
    }

    Voice::initPreCache();
}