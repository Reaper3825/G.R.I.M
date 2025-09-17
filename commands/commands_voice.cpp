#include "commands_voice.hpp"
#include "response_manager.hpp"
#include "error_manager.hpp"
#include "voice.hpp"
#include "voice_stream.hpp"
#include "commands_core.hpp"

#include <SFML/Graphics.hpp>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <string>

// Externals
extern nlohmann::json aiConfig;
extern nlohmann::json longTermMemory;
extern std::vector<Timer> timers;
extern NLP g_nlp;
extern std::string g_inputBuffer;
extern std::filesystem::path g_currentDir;

// ------------------------------------------------------------
// [Voice] One-shot voice command
// ------------------------------------------------------------
CommandResult cmdVoice([[maybe_unused]] const std::string& arg) {
    // 🔹 Run Whisper transcription
    std::string transcript = Voice::runVoiceDemo(aiConfig, longTermMemory);

    if (transcript.empty()) {
        return {
            ErrorManager::getUserMessage("ERR_VOICE_NO_SPEECH"),
            false,
            sf::Color::Red,
            "ERR_VOICE_NO_SPEECH",
            "No speech detected",
            "error"
        };
    }

    // 🔹 Inject transcript back into GRIM as if user typed it
    handleCommand(transcript);

    // 🔹 Show transcript in history (cyan) without re-speaking
    return {
        "> " + transcript,         // mimic console input style
        true,
        sf::Color::Cyan,
        "ERR_NONE",
        "Voice command processed",
        "routine"
    };
}

// ------------------------------------------------------------
// [Voice] Continuous streaming mode
// ------------------------------------------------------------
CommandResult cmdVoiceStream([[maybe_unused]] const std::string& arg) {
    if (!Voice::g_state.ctx) {
        return {
            ErrorManager::getUserMessage("ERR_VOICE_NO_CONTEXT"),
            false,
            sf::Color::Red,
            "ERR_VOICE_NO_CONTEXT",
            "Voice context missing",
            "error"
        };
    }

    if (VoiceStream::start(Voice::g_state.ctx, nullptr, timers, longTermMemory, g_nlp)) {
        return {
            "[Voice] Streaming started.",
            true,
            sf::Color::Green,
            "ERR_NONE",
            "Voice streaming started",
            "routine"
        };
    } else {
        return {
            ErrorManager::getUserMessage("ERR_VOICE_STREAM_FAIL"),
            false,
            sf::Color::Red,
            "ERR_VOICE_STREAM_FAIL",
            "Voice streaming failed",
            "error"
        };
    }
}
