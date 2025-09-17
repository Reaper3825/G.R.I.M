#include "commands/commands_core.hpp"
#include "commands/commands_helpers.hpp"

#include "voice.hpp"        // Whisper / STT
#include "voice_speak.hpp"  // TTS
#include "voice_stream.hpp"

#include "resources.hpp"
#include "console_history.hpp"
#include "ui_helpers.hpp"
#include "ui_draw.hpp"
#include "ui_events.hpp"
#include "error_manager.hpp"
#include "bootstrap.hpp"
#include "aliases.hpp"      // 🔹 added here

#include <iostream>
#include <filesystem>
#include <string>
#include <SFML/Graphics.hpp>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <windows.h>
#undef ERROR   // ⚡ fix Logger::Level::ERROR conflict
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

// ============================================================
// Entry Point
// ============================================================
int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    // ---------------- Bootstrap ----------------
    ErrorManager::load("errors.json");
    Logger::init("grim.log");
    runBootstrapChecks(argc, argv);
    std::cout << "[DEBUG] Bootstrap complete\n";

    // 🔹 Aliases init (just loads JSON, no scan yet)
    aliases::init();

    // ---------------- SFML Setup ----------------
    sf::RenderWindow window(sf::VideoMode(512, 768), "GRIM Console");
    window.setFramerateLimit(60);

    sf::Font font;
    std::string fontPath = getResourcePath() + "/DejaVuMathTeXGyre.ttf";
    if (!font.loadFromFile(fontPath)) {
        std::cerr << "[ERROR] Could not load font: " << fontPath << "\n";
        return 1;
    }
    std::cout << "[DEBUG] Font loaded OK\n";

    g_ui_textbox.setFont(font);
    g_ui_textbox.setCharacterSize(20);
    g_ui_textbox.setFillColor(sf::Color::White);

    // ---------------- Voice Init ----------------
    std::string voiceErr;
    if (!Voice::initWhisper("small", &voiceErr)) {
        Logger::log(Logger::Level::ERROR, "Failed to initialize Whisper: " + voiceErr);
    } else {
        std::cout << "[DEBUG] Whisper initialized OK\n";
    }

    // ---------------- Audible Greeting ----------------
    history.push("[GRIM] Startup complete. Hello, Austin — I am online.", sf::Color::Cyan);
    Voice::speakText("Startup complete. Hello Austin, I am online and ready.", true);

    Logger::log(Logger::Level::INFO, "GRIM startup complete, entering main loop");
    std::cout << "[DEBUG] Entering main loop\n";

    // 🔹 Kick off background app scan *after* greeting
    aliases::refreshAsync();

    // ---------------- Main Loop ----------------
    fs::path currentDir = fs::current_path();
    std::vector<Timer> timers;
    float scrollOffset = 0.0f;

    while (window.isOpen()) {
        // Process UI + events
        processEvents(window, g_inputBuffer, currentDir, timers, longTermMemory, history);

        // Sync buffer → render text
        g_ui_textbox.setString(g_inputBuffer);

        // Draw UI
        window.clear(sf::Color::Black);
        drawUI(window, font, history, g_inputBuffer, true, scrollOffset);
        window.display();

        // ✅ Voice demo hotkey (V)
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::V)) {
            std::string transcript = Voice::runVoiceDemo(aiConfig, longTermMemory);
            if (!transcript.empty()) {
                handleCommand(transcript);
            }
        }

        // ✅ Example continuous stream trigger (placeholder)
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
    std::cout << "[DEBUG] Main exited cleanly\n";
    return 0;
}
