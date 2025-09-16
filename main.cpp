#include "commands/commands_core.hpp"
#include "commands/commands_helpers.hpp"
#include "voice.hpp"
#include "voice_stream.hpp"
#include "resources.hpp"
#include "console_history.hpp"
#include "ui_helpers.hpp"
#include "ui_draw.hpp"
#include "ui_events.hpp"
#include "error_manager.hpp"
#include "bootstrap.hpp"

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

    std::cout << "[GRIM] Startup begin\n";

    // ---------------- Bootstrap ----------------
    ErrorManager::load("errors.json");   // ✅ corrected
    Logger::init("grim.log");

    runBootstrapChecks(argc, argv);

    // ---------------- SFML Setup ----------------
    sf::RenderWindow window(sf::VideoMode(512, 768), "GRIM Console");
    window.setFramerateLimit(60);

    sf::Font font;
    if (!font.loadFromFile(getResourcePath() + "/DejaVuMathTeXGyre.ttf")) {
        Logger::log(Logger::Level::ERROR, "Could not load font");
        return 1;
    }

    g_ui_textbox.setFont(font);
    g_ui_textbox.setCharacterSize(20);
    g_ui_textbox.setFillColor(sf::Color::White);

    // ---------------- Voice Init ----------------
    std::string voiceErr;
    if (!Voice::initWhisper("small", &voiceErr)) {
        Logger::log(Logger::Level::ERROR, "Failed to initialize Whisper: " + voiceErr);
    }

    // ---------------- Audible Greeting ----------------
    history.push("[GRIM] Startup complete. Hello, Austin — I am online.", sf::Color::Cyan);
    Voice::speakText("Startup complete. Hello Austin, I am online and ready.", true);

    // ---------------- Main Loop ----------------
    fs::path currentDir = fs::current_path();
    std::vector<Timer> timers;
    float scrollOffset = 0.0f; // required for drawUI()

    Logger::log(Logger::Level::INFO, "GRIM startup complete, entering main loop");

    while (window.isOpen()) {
        // ✅ Process UI + events
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
            handleCommand(transcript); // unified flow
        }

        // Example continuous stream trigger (placeholder for future)
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
