#include "bootstrap.hpp"
#include "bootstrap_config.hpp"
#include "resources.hpp"
#include "console_history.hpp"
#include "aliases.hpp"
#include "system_detect.hpp"
#include "voice/voice_speak.hpp"
#include "device_setups/audio_devices.hpp"
#include "logger.hpp"

// Define global system info
SystemInfo g_systemInfo;

void runBootstrapChecks(int argc, char** argv) {
    // ============================================================
    // Bootstrap start
    // ============================================================
    promptForAudioDevice();
    LOG_PHASE("Bootstrap begin", true);

    // ============================================================
    // Centralized config/memory bootstrap
    // ============================================================
    beginPhaseGroup();
    bootstrap_config::initAll();
    endPhaseGroup();
    LOG_PHASE("Configs initialized", true);

    // ============================================================
    // Aliases system (cache only at bootstrap)
    // ============================================================
    beginPhaseGroup();
    aliases::init();
    endPhaseGroup();
    LOG_PHASE("Aliases bootstrap finished", true);

    // ============================================================
    // Fonts
    // ============================================================
    std::string fontPath = findAnyFontInResources(argc, argv, &history);
    if (!fontPath.empty()) {
        LOG_PHASE("Font search", true);
        LOG_DEBUG("Config", "Font found: " + fontPath);
    } else {
        LOG_ERROR("Config", "No system font found, UI may render incorrectly");
        LOG_PHASE("Font search", false);
    }

    // ============================================================
    // System Detection
    // ============================================================
    g_systemInfo = detectSystem();
    logSystemInfo(g_systemInfo);
    LOG_PHASE("System detection", true);

    // ============================================================
    // Voice system (Coqui bridge)
    // ============================================================
    auto voiceCfg = aiConfig["voice"];
    bool needsCoqui = (voiceCfg.value("engine", "sapi") == "coqui");

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
        LOG_PHASE("Coqui TTS init", true);
        LOG_DEBUG("Voice", "Initializing Coqui TTS bridge...");
        if (!Voice::initTTS()) {
            LOG_ERROR("Voice", "Failed to initialize Coqui bridge");
            LOG_PHASE("Coqui TTS init", false);
        } else {
            LOG_PHASE("Coqui TTS init", true);
            LOG_DEBUG("Voice",
                "Coqui TTS initialized (speaker=" +
                voiceCfg.value("speaker", "p225") +
                ", model=" +
                voiceCfg.value("local_engine", "unknown") + ")");
        }
    } else {
        LOG_PHASE("Coqui TTS skipped", true);
        LOG_DEBUG("Voice", "Skipping Coqui init (engine=sapi only)");
    }

    // ============================================================
    // Bootstrap complete
    // ============================================================
    LOG_PHASE("Bootstrap complete", true);
}
