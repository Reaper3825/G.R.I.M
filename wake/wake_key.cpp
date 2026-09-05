#include "wake_key.hpp"
#include "wake.hpp"
#include "logger.hpp"
#include "voice/voice.hpp"
#include "ui/ui_helpers.hpp"
#include "ui/ui_root.hpp"
#include "ai/ai.hpp"
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

// -----------------------------------------------------------
// Wake-triggered voice capture
// -----------------------------------------------------------
static void captureAndProcessVoiceInput()
{
    if (g_listening) return;
    g_listening = true;

    LOG_DEBUG("WakeKey", "Capturing voice input.");
    notifyPopupActivity();

    // Capture speech → text
    std::string transcript = Voice::captureAndTranscribeSpeech(aiConfig);
    LOG_DEBUG("WakeKey", "Captured voice transcript: " + transcript);

    // Send transcript through the same raw conversation path as typed input.
    if (!transcript.empty()) {
        ai_process(transcript);
    }

    g_listening = false;
}



// -----------------------------------------------------------
// Start / Stop
// -----------------------------------------------------------
void start()
{
    if (g_running) {
        LOG_DEBUG("WakeKey", "WakeKey system already running.");
        return;
    }

    LOG_DEBUG("WakeKey", "Initializing Key system hook...");
    Key::initialize();

    g_running = true;
    LOG_PHASE("WakeKey system active (configurable global binding)", true);
}

void update()
{
    if (!g_running || g_listening)
        return;

    if (GRIM::InputBindings::wasPressed("wake_voice")) {
        LOG_DEBUG("WakeKey", "Configurable wake binding detected.");
        captureAndProcessVoiceInput();
    }
}

bool requestWake(const std::string& source)
{
    if (!g_running || g_listening) {
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
    captureAndProcessVoiceInput();
    return true;
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
