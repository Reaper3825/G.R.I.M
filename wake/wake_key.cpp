#include "wake_key.hpp"
#include "logger.hpp"
#include "voice/voice.hpp"
#include "ui/ui_helpers.hpp"
#include "commands/commands_core.hpp"
#include "ai/ai.hpp"
#include "nlp/nlp.hpp"
#include "console_history.hpp"
#include "resources.hpp"
#include "helpers/key.hpp"
#include "popup_ui/popup_ui.hpp"
#include "bootstrap/bootstrap_config.hpp"

#include <thread>
#include <windows.h>

namespace WakeKey {

static bool g_running = false;
static bool g_listening = false;
static KeyCode g_hotkey = KeyCode::RCtrl; // Default: Right Ctrl

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

    // Register callback for wake key press
    Key::onPress(g_hotkey, [=, &timers, &longTermMemory, &nlp](KeyCode code) {
        LOG_DEBUG("WakeKey", "Wake key detected via Key system.");
        handleVoiceCommand(history, timers, longTermMemory, nlp);
    });

    g_running = true;
    LOG_PHASE("WakeKey system active", true);
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
