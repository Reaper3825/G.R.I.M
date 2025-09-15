#include "commands_voice.hpp"
#include "commands_interface.hpp"
#include "voice.hpp"
#include "voice_stream.hpp"
#include "console_history.hpp"
#include "response_manager.hpp"
#include "ai.hpp"               // aiConfig
#include "nlp.hpp"              // NLP state
#include "commands_timers.hpp"  // Timer
#include "timer.hpp"
#include "commands_core.hpp"

#include <SFML/Graphics.hpp>
#include <iostream>
#include <nlohmann/json.hpp>
#include <filesystem>

// Externals
extern nlohmann::json aiConfig;
extern nlohmann::json longTermMemory;
extern ConsoleHistory history;
extern std::vector<Timer> timers;
extern NLP g_nlp;
extern std::string g_inputBuffer;
extern std::filesystem::path g_currentDir;

CommandResult cmdVoice([[maybe_unused]] const std::string& arg) {
    std::cout << "[DEBUG][Command] Dispatch: voice\n";

    // Run voice demo with AI config and memory
    std::string transcript = Voice::runVoiceDemo(aiConfig, longTermMemory);

    if (transcript.empty()) {
        return { "[Voice] No speech detected.", false, sf::Color::Red };
    }

    // Feed transcript back into dispatcher (like typing a command)
    parseAndDispatch(transcript, g_inputBuffer, g_currentDir, timers, longTermMemory, history);

    return { "[Voice] Processed: " + transcript, true, sf::Color::Cyan };
}

CommandResult cmdVoiceStream([[maybe_unused]] const std::string& arg) {
    std::cout << "[DEBUG][Command] Dispatch: voice_stream\n";

    if (VoiceStream::start(Voice::g_state.ctx, &history, timers, longTermMemory, g_nlp)) {
        return { "[Voice] Streaming started.", true, sf::Color::Green };
    } else {
        return { "[Voice] Failed to start streaming.", false, sf::Color::Red };
    }
}
