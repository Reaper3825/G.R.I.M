#include "wake_key.hpp"
#include "logger.hpp"
#include "voice/voice.hpp"
#include "ui/ui_helpers.hpp"
#include "ui/ui_root.hpp"
#include "commands/commands_core.hpp"
#include "ai/ai.hpp"
#include "nlp/nlp.hpp"
#include "console_history.hpp"
#include "resources.hpp"
#include "helpers/key.hpp"
#include "popup_ui/popup_ui.hpp"
#include "bootstrap/bootstrap_config.hpp"

#include <thread>
#include "core/grim_platform.h"

namespace WakeKey {

static bool g_running = false;
static bool g_listening = false;
static KeyCode g_hotkey = KeyCode::RCtrl; // Default: Right Ctrl
static KeyCode g_consoleToggleKey = KeyCode::Grave; // Default: ~ / ` key

// -----------------------------------------------------------
// Voice command handler
// -----------------------------------------------------------
static void handleVoiceCommand(ConsoleHistory* history,
                               std::vector<Timer>& timers,
                               nlohmann::json& longTermMemory,
                               NLP& nlp)
{
    if (g_listening) return;
    g_listening = true;

    LOG_DEBUG("WakeKey", "Wake key pressed — capturing voice command.");
    notifyPopupActivity();

    // Capture speech → text
    std::string transcript = Voice::runVoiceDemo(aiConfig, longTermMemory);
    LOG_DEBUG("WakeKey", "Captured voice transcript: " + transcript);

    // Send transcript through the central pipeline
    if (!transcript.empty()) {
        handleCommand(transcript); // 🧠 All NLP + voice output handled here
    }

    g_listening = false;
}



// -----------------------------------------------------------
// Start / Stop
// -----------------------------------------------------------
void start(ConsoleHistory* history,
           std::vector<Timer>& timers,
           nlohmann::json& longTermMemory,
           NLP& nlp)
{
    if (g_running) {
        LOG_DEBUG("WakeKey", "WakeKey system already running.");
        return;
    }

    LOG_DEBUG("WakeKey", "Initializing Key system hook...");
    Key::initialize();

    // Register callback for wake key press (voice command)
    Key::onPress(g_hotkey, [=, &timers, &longTermMemory, &nlp](KeyCode code) {
        LOG_DEBUG("WakeKey", "Wake key detected via Key system.");
        handleVoiceCommand(history, timers, longTermMemory, nlp);
    });

    // Register callback for console toggle key (grave/tilde) - Toggle OVERLAY console
    Key::onPress(g_consoleToggleKey, [](KeyCode code) {
        LOG_DEBUG("WakeKey", "Console toggle key pressed (Grave/~) - Toggling overlay console");
        
        // Toggle the overlay console panel instead of Win32 console
        auto consolePanel = UIRoot::get().getPanel("Console");
        if (consolePanel) {
            bool newState = !consolePanel->isVisible();
            LOG_DEBUG("WakeKey", "Found Console panel, current state: " + std::string(consolePanel->isVisible() ? "visible" : "hidden"));
            UIRoot::get().setVisible("Console", newState);
            LOG_DEBUG("WakeKey", std::string("Overlay console set to ") + (newState ? "VISIBLE" : "HIDDEN"));
        } else {
            LOG_ERROR("WakeKey", "Console panel not found in UIRoot - checking all panels...");
            // Debug: List all available panels
            auto settingsPanel = UIRoot::get().getPanel("Settings");
            auto grimSettings = UIRoot::get().getPanel("GRIM Settings");
            LOG_DEBUG("WakeKey", "Settings panel exists: " + std::string(settingsPanel ? "YES" : "NO"));
            LOG_DEBUG("WakeKey", "GRIM Settings panel exists: " + std::string(grimSettings ? "YES" : "NO"));
        }
    });

    g_running = true;
    LOG_PHASE("WakeKey system active (overlay console toggle)", true);
}

void stop()
{
    if (!g_running) return;

    LOG_DEBUG("WakeKey", "Stopping WakeKey system...");
    Key::shutdown();
    g_running = false;

    LOG_PHASE("WakeKey system stopped", true);
}

bool isRunning()
{
    return g_running;
}

} // namespace WakeKey
