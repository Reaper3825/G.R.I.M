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
#include "logger.hpp"

#include <SFML/Graphics.hpp>
#include <thread>
#include <atomic>
#include <filesystem>
#include <chrono>

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
    LOG_PHASE("Startup begin", true);

    // 🔹 Bootstrap configuration and resources (includes TTS init)
    runBootstrapChecks(argc, argv);
    LOG_PHASE("Bootstrap checks complete", true);

    // 🔹 Load dummy font (needed for sf::Text even if unused)
    fs::path fontPath = fs::path(getResourcePath()) / "DejaVuMathTeXGyre.ttf";
    if (!g_dummyFont.openFromFile(fontPath.string())) {
        LOG_ERROR("Config", "Could not load dummy font: " + fontPath.string());
        LOG_PHASE("Font load", false);
    } else {
        LOG_DEBUG("Config", "Loaded dummy font: " + fontPath.string());
        LOG_PHASE("Font load", true);
    }

    // 🔹 Aliases
    aliases::init();
    LOG_PHASE("Aliases initialized", true);

    // ============================================================
    // Wait for TTS bridge to be ready before greeting
    // ============================================================
    if (!Voice::isReady()) {
        LOG_DEBUG("Voice", "Waiting for TTS bridge to be ready...");
        while (!Voice::isReady()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    // 🔹 Startup greeting
    Voice::speak("Welcome back, Austin. Grim is online.", "system");
    LOG_PHASE("Startup greeting spoken", true);

    LOG_PHASE("Startup complete, entering main loop", true);

    // ============================================================
    // Launch popup UI in background thread
    // ============================================================
    sf::VideoMode desktop = sf::VideoMode::getDesktopMode();
    unsigned monWidth  = desktop.size.x;
    unsigned monHeight = desktop.size.y;

    std::thread([monWidth, monHeight]() {
        LOG_DEBUG("PopupUI", "Launching with size = " +
                             std::to_string(monWidth) + "x" +
                             std::to_string(monHeight));
        runPopupUI(monWidth, monHeight);
    }).detach();
    LOG_PHASE("Popup UI launched", true);

    // ============================================================
    // Console input loop
    // ============================================================
    std::string line;
    while (true) {
        std::cout << "> "; // REPL prompt stays visible
        if (!std::getline(std::cin, line)) {
            break; // EOF / Ctrl+D
        }

        // 🔹 Skip empty input (avoid NLP spam)
        if (line.empty()) {
            continue;
        }

        if (line == "quit" || line == "exit") {
            LOG_PHASE("Shutdown requested", true);
            break;
        }

        LOG_TRACE("Console", "Dispatching command: " + line);
        handleCommand(line);
    }

    // ============================================================
    // Shutdown cleanup
    // ============================================================
    Voice::shutdownTTS();
    LOG_PHASE("Shutdown complete", true);
    return 0;
}
