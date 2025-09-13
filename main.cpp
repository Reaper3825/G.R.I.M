#include "commands.hpp"
#include "aliases.hpp"
#include "synonyms.hpp"
#include "ai.hpp"
#include "voice.hpp"
#include "voice_stream.hpp"
#include "resources.hpp"
#include "nlp.hpp"
#include "nlp_rules.hpp"
#include "console_history.hpp"
#include "ui_helpers.hpp"
#include "ui_draw.hpp"
#include "ui_events.hpp"

#include <iostream>
#include <filesystem>
#include <string>
#include <SFML/Graphics.hpp>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#include <psapi.h>
#include <winternl.h>
#endif

namespace fs = std::filesystem;

// 🔹 Global UI textbox (for rendering only)
sf::Text g_ui_textbox;

// 🔹 Global raw input buffer (for editing)
std::string g_inputBuffer;

// 🔹 Global AI state (defined in ai.cpp, declared extern in resources.hpp)
extern nlohmann::json longTermMemory;
extern nlohmann::json aiConfig;

// ------------------------------------------------------------
// Utility: lowercase + strip punctuation for NLP normalization
// ------------------------------------------------------------
static std::string normalizeInput(const std::string& input) {
    std::string out;
    out.reserve(input.size());

    for (char c : input) {
        if (!std::ispunct(static_cast<unsigned char>(c))) {
            out.push_back(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    return out;
}

// ---------------- Helpers ----------------
static void parseAndDispatch(const std::string& text,
                             std::string& buffer,
                             fs::path& currentDir,
                             std::vector<Timer>& timers,
                             nlohmann::json& longTermMemory,
                             ConsoleHistory& history) {
    // ✅ Normalize text before NLP
    std::string cleaned = normalizeInput(text);

    Intent intent = g_nlp.parse(cleaned);   // use global g_nlp
    if (intent.matched) {
        handleCommand(intent, buffer, currentDir, timers, longTermMemory, g_nlp, history);
    } else {
        history.push("[NLP] No intent matched for input: '" + cleaned +
                     "' (rules loaded=" + std::to_string(g_nlp.rule_count()) + ")",
                     sf::Color::Red);
    }
}

// ============================================================
// Entry Point
// ============================================================
int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    std::cout << "[DEBUG] GRIM startup begin\n";

    // ---------------- Bootstrap ----------------
    loadMemory();
    loadAIConfig("ai_config.json");
    VoiceStream::calibrateSilence();

    ConsoleHistory history;

    if (!loadNlpRules("nlp_rules.json")) {
        std::cerr << "[ERROR] Failed to load NLP rules\n";
    } else {
        std::cout << "[NLP] Rules loaded\n";
    }

    if (!loadSynonyms("synonyms.json")) {
        std::cerr << "[ERROR] Failed to load synonyms\n";
    }

    loadAliases("app_aliases.json");

    // ---------------- SFML Setup ----------------
    sf::RenderWindow window(sf::VideoMode(1280, 720), "GRIM Console");
    window.setFramerateLimit(60);

    sf::Font font;
    if (!font.loadFromFile(getResourcePath() + "/DejaVuMathTeXGyre.ttf")) {
        std::cerr << "[ERROR] Could not load font\n";
        return 1;
    }

    g_ui_textbox.setFont(font);
    g_ui_textbox.setCharacterSize(20);
    g_ui_textbox.setFillColor(sf::Color::White);

    // ---------------- Voice Init ----------------
    std::string voiceErr;
    if (!Voice::initWhisper("small", &voiceErr)) {
        std::cerr << "[Voice] Failed to initialize Whisper: " << voiceErr << "\n";
    }

    // ---------------- Audible Greeting ----------------
    history.push("[GRIM] Startup complete. Hello, Austin — I am online.", sf::Color::Cyan);
    Voice::speakText("Startup complete. Hello Austin, I am online and ready.", true);

    // ---------------- Main Loop ----------------
    fs::path currentDir = fs::current_path();
    std::vector<Timer> timers;
    float scrollOffset = 0.0f; // required for drawUI()

    std::cout << "[DEBUG] GRIM startup complete, entering main loop\n";

    while (window.isOpen()) {
        // ✅ processEvents uses g_nlp internally
        processEvents(window, g_inputBuffer, currentDir, timers, longTermMemory, history);

        // Sync buffer → render text
        g_ui_textbox.setString(g_inputBuffer);

        // Draw UI
        window.clear(sf::Color::Black);
        drawUI(window, font, history, g_inputBuffer, true, scrollOffset);
        window.display();

        // Example voice demo trigger (replace with hotkey/command)
        if (false) { // placeholder condition
            std::string transcript = Voice::runVoiceDemo(longTermMemory);

            // ✅ Log both raw and normalized versions
            std::string cleaned = normalizeInput(transcript);
            history.push("[Voice Transcript RAW] " + transcript, sf::Color::Yellow);
            history.push("[Voice Transcript CLEAN] " + cleaned, sf::Color::Green);

            // ✅ Dispatch cleaned transcript to NLP
            parseAndDispatch(cleaned, g_inputBuffer, currentDir, timers, longTermMemory, history);
        }

        // Example continuous stream trigger (placeholder)
        if (false) {
            if (!VoiceStream::isRunning()) {
                VoiceStream::start(Voice::g_state.ctx, &history, timers, longTermMemory, g_nlp);
            } else {
                VoiceStream::stop();
            }
        }
    }

    // ✅ Persist learned memory on exit
    saveMemory();
    return 0;
}
