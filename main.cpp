#include "commands/commands_core.hpp"
#include "commands/commands_helpers.hpp"

#include "voice.hpp"
#include "voice_speak.hpp"
#include "voice_stream.hpp"
#include "response_manager.hpp"

#include "resources.hpp"
#include "console_history.hpp"
#include "ui_helpers.hpp"
#include "ui_draw.hpp"
#include "ui_events.hpp"
#include "error_manager.hpp"
#include "bootstrap.hpp"
#include "aliases.hpp"

#include <iostream>
#include <filesystem>
#include <string>
#include <SFML/Graphics.hpp>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <windows.h>
#undef ERROR
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

// 🔹 Global voice hotkey
sf::Keyboard::Key g_voiceHotkey = sf::Keyboard::Num0;

// ------------------------------------------------------------
// Helper: initialize hotkey from config
// ------------------------------------------------------------
static void initHotkey() {
    std::string hotkeyStr = aiConfig["voice"].value("hotkey", "0");

    if (hotkeyStr == "F1") {
        g_voiceHotkey = sf::Keyboard::F1;
    } else if (hotkeyStr == "0") {
        g_voiceHotkey = sf::Keyboard::Num0;
    } else if (hotkeyStr == "L") {
        g_voiceHotkey = sf::Keyboard::L;
    } else {
        // default fallback
        g_voiceHotkey = sf::Keyboard::Num0;
    }

    std::cout << "[DEBUG] Voice hotkey set to: " << hotkeyStr << "\n";
}

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

    // ---------------- Voice (Coqui / Local TTS) ----------------
    if (aiConfig.contains("voice") &&
        aiConfig["voice"].value("engine", "sapi") == "coqui")
    {
        if (Voice::initTTS()) {
            std::cout << "[Voice] Coqui TTS bridge initialized\n";
        } else {
            std::cerr << "[Voice] Failed to initialize Coqui TTS bridge\n";
        }
    }

    // ---------------- Greeting ----------------
    ResponseManager::systemMessage(
        "[GRIM] Startup complete. Hello, Austin — I am online.",
        sf::Color::Green
    );

    Logger::log(Logger::Level::INFO, "GRIM startup complete, entering main loop");
    std::cout << "[DEBUG] Entering main loop\n";

    // 🔹 Initialize hotkey from config
    initHotkey();

    // 🔹 Kick off background app scan *after* greeting
    aliases::refreshAsync();

    // ---------------- Main Loop ----------------
    fs::path currentDir = fs::current_path();
    std::vector<Timer> timers;
    float scrollOffset = 0.0f;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            // Handle UI + keyboard events
            if (!processEvents(window, g_inputBuffer, currentDir, timers, longTermMemory, history)) {
                break; // window closed
            }

            // ✅ Voice hotkey (configurable, variable-driven)
            if (event.type == sf::Event::KeyPressed &&
                event.key.code == g_voiceHotkey)
            {
                std::cout << "[Voice] Hotkey pressed → capturing voice\n";
                std::string transcript = Voice::runVoiceDemo(aiConfig, longTermMemory);
                if (!transcript.empty()) {
                    handleCommand(transcript); // 🔹 same flow as typed input
                }
            }
        }

        // Sync buffer → render text
        g_ui_textbox.setString(g_inputBuffer);

        // Draw UI
        window.clear(sf::Color::Black);
        drawUI(window, font, history, g_inputBuffer, true, scrollOffset);
        window.display();

        // ✅ Example continuous stream trigger (placeholder)
        if (false) {
            if (!VoiceStream::isRunning()) {
                VoiceStream::start(Voice::g_state.ctx, &history, timers, longTermMemory, g_nlp);
            } else {
                VoiceStream::stop();
            }
        }
    }

    // ---------------- Shutdown ----------------
    Voice::shutdownTTS();   // 🔹 Clean up Coqui bridge
    saveMemory();           // 🔹 Persist learned memory

    std::cout << "[DEBUG] Main exited cleanly\n";
    return 0;
}
