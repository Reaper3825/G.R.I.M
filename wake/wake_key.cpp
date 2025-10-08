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


    // Perform voice capture
    std::string transcript = Voice::runVoiceDemo(aiConfig, longTermMemory);
    LOG_DEBUG("WakeKey", "Captured voice transcript: " + transcript);

    if (!transcript.empty()) {
        Intent intent = nlp.parse(transcript);

        if (intent.matched) {
            LOG_DEBUG("WakeKey", "Dispatching recognized command: " + intent.name);
            handleCommand(transcript);
        } else {
            std::string fullReply;
            ai_process_stream(
                transcript,
                longTermMemory,
                [&](const std::string& chunk) {
                    fullReply += chunk;
                    ui_set_textbox(fullReply);
                    LOG_DEBUG("WakeKey/AI", "Chunk: " + chunk);
                });

            history->push("[AI] " + fullReply, sf::Color::Green);
        }
    } else {
        LOG_DEBUG("WakeKey", "No transcript captured (silence or timeout).");
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
