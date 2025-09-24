#include "commands/commands_core.hpp"
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
#include "popup_ui/popup_ui.hpp"

#include <SFML/Graphics.hpp>
#include <iostream>
#include <thread>
#include <atomic>
#include <filesystem>

#ifdef _WIN32
  #include <shellapi.h>
  #include <psapi.h>
  #include <winternl.h>
  #undef ERROR
#endif

namespace fs = std::filesystem;

// 🔹 Global dummy font (required for sf::Text ctor in SFML 3)
sf::Font g_dummyFont;

// 🔹 Global UI textbox + raw buffer (declared extern in ui_helpers.hpp)
sf::Text g_ui_textbox(g_dummyFont, "", 20);
std::string g_inputBuffer;

// ============================================================
// Main entry point
// ============================================================
int main(int argc, char* argv[]) {
    std::cout << "[GRIM] Startup begin\n";

    // 🔹 Bootstrap configuration and resources
    runBootstrapChecks(argc, argv);

    // 🔹 Load dummy font (needed for sf::Text even if unused)
    fs::path fontPath = fs::path(getResourcePath()) / "DejaVuMathTeXGyre.ttf";
    if (!g_dummyFont.openFromFile(fontPath.string())) {
        std::cerr << "[Config] ERROR: Could not load dummy font\n";
    } else {
        std::cout << "[Config] Font loaded OK\n";
    }

    // 🔹 Initialize subsystems
    if (!Voice::initTTS()) {
        std::cerr << "[Voice] ERROR: Failed to initialize TTS\n";
    }
    aliases::init();

    // 🔹 Startup greeting
    Voice::speak("Welcome back, Austin. Grim is online.", "system");

    std::cout << "[GRIM] Startup complete, entering main loop\n";

    // ============================================================
    // Launch popup UI in background thread
    // ============================================================
    std::thread([]() {
        runPopupUI(128, 128); // ✅ 2-argument version only
    }).detach();

    // ============================================================
    // Console input loop
    // ============================================================
    std::string line;
    while (true) {
        std::cout << "> ";
        if (!std::getline(std::cin, line)) {
            break; // EOF / Ctrl+D
        }

        if (line == "quit" || line == "exit") {
            std::cout << "[GRIM] Shutdown requested\n";
            break;
        }

        // Handle command via unified command system
        handleCommand(line);
    }

    // ============================================================
    // Shutdown cleanup
    // ============================================================
    Voice::shutdownTTS();
    std::cout << "[GRIM] Shutdown complete\n";
    return 0;
}
