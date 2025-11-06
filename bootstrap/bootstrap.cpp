#include "bootstrap.hpp"
#include "bootstrap_config.hpp"
#include "resources.hpp"
#include "console_history.hpp"
#include "aliases.hpp"
#include "system_detect.hpp"
#include "voice/voice_speak.hpp"
#include "device_setups/audio_devices.hpp"
#include "logger.hpp"
#include "voice/voice.hpp"
#include "ai/ai_rl.hpp"
#include "ai/ai.hpp"  // ✅ NEW: For warmupOllamaModel()
#include "ai/grim_text_server_manager.hpp"  // ✅ NEW: For GRIM-text server

#include <filesystem>    // ✅ for fs::path
#include <whisper.h>     // ✅ for whisper_context + init functions

// Define global system info
SystemInfo g_systemInfo;
namespace fs = std::filesystem; // ✅ define filesystem alias

void runBootstrapChecks(int argc, char** argv) {
    // ============================================================
    // Bootstrap start
    // ============================================================


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
                "Coqui XTTS v2 initialized (speaker=" +
                voiceCfg.value("speaker", "default") +
                ", language=" +
                voiceCfg.value("language", "en") + ")");
        }
    } else {
        LOG_PHASE("Coqui TTS skipped", true);
        LOG_DEBUG("Voice", "Skipping Coqui init (engine=sapi only)");
    }
    // ============================================================
    // Reinforcement Learning Bridge (RL)
    // ============================================================
    LOG_PHASE("RL Bridge init", true);
    LOG_DEBUG("RL", "Starting rl_bridge.py via unified BridgeManager...");
    if (!GRIM::RL::init()) {
        LOG_ERROR("RL", "Failed to start rl_bridge.py");
        LOG_PHASE("RL Bridge init", false);
    } else {
        LOG_PHASE("RL Bridge init", true);
        LOG_DEBUG("RL", "rl_bridge.py initialized and ready");
    }

    // ============================================================
    // GRIM-text Server Startup (if using native backend)
    // ============================================================
    if (aiConfig.value("backend", "auto") == "grim_native") {
        LOG_PHASE("GRIM-text server startup", true);
        LOG_DEBUG("Bootstrap", "Starting GRIM-text server...");
        
        if (GRIM::startGRIMTextServer()) {
            LOG_PHASE("GRIM-text server startup", true);
            LOG_DEBUG("Bootstrap", "GRIM-text server running at " + 
                     aiConfig.value("grim_text_url", "http://127.0.0.1:11435"));
        } else {
            LOG_ERROR("Bootstrap", "Failed to start GRIM-text server");
            LOG_PHASE("GRIM-text server startup", false);
        }
    }
    
    // ============================================================
    // AI Model Warmup (preload into VRAM for instant responses)
    // ============================================================
    if (aiConfig.value("backend", "ollama") == "ollama") {
        LOG_PHASE("AI model warmup begin", true);
        warmupOllamaModel();
        LOG_PHASE("AI model warmup complete", true);
    }

    // ============================================================
    // Preload Whisper STT model (warm-up only, no live capture)
    // ============================================================
    LOG_PHASE("Whisper preload begin", true);

    try {
        if (Voice::getWhisperContext() == nullptr)
        {
            // Resolve Whisper model from config
            std::string modelName = "ggml-base.en.bin";
            if (aiConfig.contains("whisper") && aiConfig["whisper"].contains("whisper_model"))
                modelName = aiConfig["whisper"].value("whisper_model", modelName);

            fs::path modelPath = fs::path(getResourcePath()) / "models" / modelName;
            LOG_DEBUG("Voice", "Preloading Whisper model: " + modelPath.string());

            if (!fs::exists(modelPath))
            {
                LOG_ERROR("Voice", "Whisper model missing: " + modelPath.string());
                LOG_PHASE("Whisper preload", false);
            }
            else
            {
                whisper_context_params wparams = whisper_context_default_params();
                whisper_context* ctx = whisper_init_from_file_with_params(modelPath.string().c_str(), wparams);

                if (!ctx)
                {
                    LOG_ERROR("Voice", "Failed to initialize Whisper context from: " + modelPath.string());
                    LOG_PHASE("Whisper preload", false);
                }
                else
                {
                    Voice::setWhisperContext(ctx);
                    LOG_PHASE("Whisper preload complete", true);
                    LOG_DEBUG("Voice", "Whisper model preloaded successfully (no listening loop)");
                }
            }
        }
        else
        {
            LOG_DEBUG("Voice", "Whisper already loaded, skipping preload");
            LOG_PHASE("Whisper preload complete", true);
        }
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Bootstrap", std::string("Whisper preload failed: ") + e.what());
        LOG_PHASE("Whisper preload", false);
    }

    // ============================================================
    // Bootstrap complete
    // ============================================================
    LOG_PHASE("Bootstrap complete", true);
}
