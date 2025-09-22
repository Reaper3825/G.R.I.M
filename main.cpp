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

// These are NOT in pch.hpp and must stay
#ifdef _WIN32
  #include <shellapi.h>
  #include <psapi.h>
  #include <winternl.h>
  #undef ERROR
#endif

namespace fs = std::filesystem;

// 🔹 Global dummy font (required for sf::Text ctor in SFML 3)
sf::Font g_dummyFont;

// 🔹 Global UI textbox (SFML 3: ctor = (font, string, size))
sf::Text g_ui_textbox(g_dummyFont, "", 20);

// 🔹 Global raw input buffer (for editing)
std::string g_inputBuffer;

// 🔹 Global AI state (defined in ai.cpp, declared extern in resources.hpp)
extern nlohmann::json longTermMemory;
extern nlohmann::json aiConfig;

// 🔹 Global voice hotkey (SFML 3 scoped enums)
sf::Keyboard::Key g_voiceHotkey = sf::Keyboard::Key::Num0;

// ------------------------------------------------------------
// Helper: initialize hotkey from config
// ------------------------------------------------------------
static void initHotkey() {
    std::string hotkeyStr = aiConfig["voice"].value("hotkey", "0");

    if (hotkeyStr == "F1") {
        g_voiceHotkey = sf::Keyboard::Key::F1;
    } else if (hotkeyStr == "0") {
        g_voiceHotkey = sf::Keyboard::Key::Num0;
    } else if (hotkeyStr == "L") {
        g_voiceHotkey = sf::Keyboard::Key::L;
    } else {
        // default fallback
        g_voiceHotkey = sf::Keyboard::Key::Num0;
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

    // ---------------- SFML Setup ----------------
    sf::RenderWindow window(sf::VideoMode({512u, 768u}), "GRIM Console");
    window.setFramerateLimit(60);

    sf::Font font;
    fs::path fontPath;

#ifdef GRIM_FONT_PATH
    fontPath = fs::path(GRIM_FONT_PATH);
    std::cout << "[DEBUG] Trying font path (GRIM_FONT_PATH): " << fontPath.string() << "\n";
#else
    fontPath = fs::path(getResourcePath()) / "DejaVuMathTeXGyre.ttf";
    std::cout << "[DEBUG] Trying font path (resource): " << fontPath.string() << "\n";
#endif

    if (!font.openFromFile(fontPath.string())) {
        std::cerr << "[ERROR] Could not load font at " << fontPath.string() << "\n";

        // Fallback to bundled resource
        fs::path fallback = fs::path(getResourcePath()) / "DejaVuMathTeXGyre.ttf";
        std::cout << "[DEBUG] Trying fallback font: " << fallback.string() << "\n";

        if (!font.openFromFile(fallback.string())) {
            std::cerr << "[ERROR] Could not load fallback font: " << fallback.string() << "\n";

            // Final fallback → system Arial
            fs::path sysFont = "C:/Windows/Fonts/arial.ttf";
            std::cout << "[DEBUG] Trying system font: " << sysFont.string() << "\n";

            if (!font.openFromFile(sysFont.string())) {
                std::cerr << "[FATAL] Could not load any font (last attempt: " << sysFont.string() << ")\n";
                return 1;
            }
        }
    }

    std::cout << "[DEBUG] Font loaded OK\n";

    // 🔹 Apply the real font now that it’s loaded
    g_ui_textbox.setFont(font);
    g_ui_textbox.setString("");
    g_ui_textbox.setCharacterSize(20);
    g_ui_textbox.setFillColor(sf::Color::White);

    // ---------------- Voice (Coqui / Local TTS) ----------------
    if (aiConfig.contains("voice")) {
        auto voiceCfg = aiConfig["voice"];
        std::string engine = voiceCfg.value("engine", std::string("sapi"));
        std::cout << "[DEBUG] Voice engine in config: " << engine << std::endl;

        bool needsCoqui = (engine == "coqui");

        // Check rules — if *any* rule requires Coqui, we also need to init
        if (voiceCfg.contains("rules")) {
            for (auto& [k, v] : voiceCfg["rules"].items()) {
                if (v.get<std::string>() == "coqui") {
                    needsCoqui = true;
                    break;
                }
            }
        }

        if (needsCoqui) {
            if (Voice::initTTS()) {
                std::cout << "[Voice] Coqui TTS bridge initialized (speaker="
                          << voiceCfg.value("speaker", "p225")
                          << ", model="
                          << voiceCfg.value("local_engine", "unknown")
                          << ")\n";
            } else {
                std::cerr << "[Voice] Failed to initialize Coqui TTS bridge\n";
            }
        } else {
            std::cout << "[Voice] Skipping Coqui init (engine=" << engine << ")\n";
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
    std::vector<Timer> localTimers;
    float scrollOffset = 0.0f;

    while (window.isOpen()) {
        // 🔹 SFML 3 style: pollEvent() returns optional
        while (auto evOpt = window.pollEvent()) {
            const sf::Event& ev = *evOpt;

            // Handle UI + keyboard events
            if (!processEvents(window, g_inputBuffer, currentDir, localTimers, longTermMemory, history)) {
                break; // window closed
            }

            // ✅ Voice hotkey (configurable, variable-driven)
            if (const auto* key = ev.getIf<sf::Event::KeyPressed>()) {
                if (key->code == g_voiceHotkey) {
                    std::cout << "[Voice] Hotkey pressed → capturing voice\n";
                    std::string transcript = Voice::runVoiceDemo(aiConfig, longTermMemory);
                    if (!transcript.empty()) {
                        handleCommand(transcript); // 🔹 same flow as typed input
                    }
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
                VoiceStream::start(Voice::g_state.ctx, &history, localTimers, longTermMemory, g_nlp);
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
