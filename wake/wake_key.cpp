#include "wake_key.hpp"
#include "wake.hpp"
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
#include "core/input/InputBindings.hpp"
#include "popup_ui/popup_ui.hpp"
#include "bootstrap/bootstrap_config.hpp"

#include <thread>
#include "core/grim_platform.h"

namespace WakeKey {

static bool g_running = false;
static bool g_listening = false;
static ConsoleHistory* g_history = nullptr;
static std::vector<Timer>* g_timers = nullptr;
static nlohmann::json* g_longTermMemory = nullptr;
static NLP* g_nlp = nullptr;

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

    g_history = history;
    g_timers = &timers;
    g_longTermMemory = &longTermMemory;
    g_nlp = &nlp;

    g_running = true;
    LOG_PHASE("WakeKey system active (configurable global binding)", true);
}

void update()
{
    if (!g_running || g_listening || !g_timers || !g_longTermMemory || !g_nlp)
        return;

    if (GRIM::InputBindings::wasPressed("wake_voice")) {
        LOG_DEBUG("WakeKey", "Configurable wake binding detected.");
        handleVoiceCommand(g_history, *g_timers, *g_longTermMemory, *g_nlp);
    }
}

bool requestWake(const std::string& source)
{
    if (!g_running || g_listening || !g_timers || !g_longTermMemory || !g_nlp) {
        LOG_DEBUG("WakeKey", "External wake request rejected: " + source);
        return false;
    }

    Wake::WakeEvent event;
    event.stimulant = Wake::Stimulant::Motion;
    event.source = source;
    event.intensity = 1.0f;
    event.priority = 5;
    event.timestamp = std::chrono::steady_clock::now();
    event.payload = "voice_capture";
    Wake::triggerWake(event);
    LOG_DEBUG("WakeKey", "External wake request accepted: " + source);
    handleVoiceCommand(g_history, *g_timers, *g_longTermMemory, *g_nlp);
    return true;
}

void stop()
{
    if (!g_running) return;

    LOG_DEBUG("WakeKey", "Stopping WakeKey system...");
    Key::shutdown();
    g_history = nullptr;
    g_timers = nullptr;
    g_longTermMemory = nullptr;
    g_nlp = nullptr;
    g_running = false;

    LOG_PHASE("WakeKey system stopped", true);
}

bool isRunning()
{
    return g_running;
}

} // namespace WakeKey
