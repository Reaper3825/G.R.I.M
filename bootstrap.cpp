#include "bootstrap.hpp"
#include "bootstrap_config.hpp"
#include "resources.hpp"
#include "console_history.hpp"
#include "aliases.hpp"
#include "system_detect.hpp"
#include "voice_speak.hpp"   // ✅ add this for Voice::initTTS

// Define global system info
SystemInfo g_systemInfo;

void runBootstrapChecks(int argc, char** argv) {
    grimLog("[GRIM] Startup begin");

    // Centralized config/memory bootstrap
    bootstrap_config::initAll();

    // Aliases system (cache only at bootstrap)
    grimLog("[aliases] Bootstrap: initializing (cache only, no scan)");
    aliases::init();
    grimLog("[aliases] Bootstrap: init finished");

    // Fonts
    std::string fontPath = findAnyFontInResources(argc, argv, &history);
    if (!fontPath.empty())
        grimLog("[Config] Font found: " + fontPath);
    else
        grimLog("[Config] No system font found, UI may render incorrectly");

    // ==============================
    // Voice system (Coqui bridge)
    // ==============================
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
        grimLog("[Voice] Initializing Coqui TTS bridge...");
        if (!Voice::initTTS()) {
            grimLog("[Voice] ERROR: Failed to initialize Coqui bridge");
        } else {
            grimLog("[Voice] Coqui TTS initialized (speaker=" +
                    voiceCfg.value("speaker", "p225") +
                    ", model=" +
                    voiceCfg.value("local_engine", "unknown") + ")");
        }
    } else {
        grimLog("[Voice] Skipping Coqui init (engine=sapi only)");
    }

    grimLog("[GRIM] ---- Bootstrap Complete ----");
}
