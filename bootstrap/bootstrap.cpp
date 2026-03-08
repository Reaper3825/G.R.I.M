#include "bootstrap.hpp"
#include "bootstrap_config.hpp"
#include "resources.hpp"
#include "console_history.hpp"
#include "aliases.hpp"
#include "voice/voice_speak.hpp"
#include "device_setups/audio_devices.hpp"
#include "logger.hpp"
#include "voice/voice.hpp"
#include "ai/ai_rl.hpp"
#include "ai/ai.hpp"  // For warmupOllamaModel()
#include "ai/grim_text_server_manager.hpp"  // For GRIM-text server
#include "../MMO/Core/HardwareInventory.hpp"
#include "../MMO/Core/ResourceSignal.hpp"
#include "../MMO/Core/ResourceCoordinator.hpp"
#include "../MMO/Core/ModelRegistry.hpp"
#include "../MMO/Core/ModelLoader.hpp"
#include "../MMO/Core/Orchestrator.hpp"
#include "../MMO/Core/SessionContextManager.hpp"
#include "../MMO/Core/ToolRegistry.hpp"
#include "../MMO/Core/ActionPolicyRegistry.hpp"
#include "../MMO/Backends/GrimNativeBackend.hpp"
#include "../MMO/Backends/OllamaBackend.hpp"
#include "../MMO/Backends/LlamaCppBackend.hpp"
#include "../MMO/Backends/ExternalBackend.hpp"
#include "../commands/command_registry.hpp"
#include "../core/plugin_manager.hpp"
#include "../location.hpp"

#include <filesystem>
#include <whisper.h>

// Global resource layer (replaces old g_systemInfo)
GRIM::MMO::HardwareInventory g_hardwareInventory;
GRIM::MMO::ResourceSignal*   g_resourceSignal      = nullptr;
GRIM::MMO::ResourceCoordinator* g_resourceCoordinator = nullptr;

// Global MMO orchestration layer
GRIM::MMO::ModelLoader*    g_modelLoader    = nullptr;
GRIM::MMO::Orchestrator*   g_orchestrator   = nullptr;
GRIM::MMO::MemoryFacade*   g_memoryFacade   = nullptr;

// Idle-tick background thread for ModelLoader TTL management
static std::atomic<bool> s_idleTickStop{false};
static std::thread       s_idleTickThread;

void stopMMOIdleTick() {
    s_idleTickStop.store(true, std::memory_order_relaxed);
}

namespace fs = std::filesystem;

// ================================================================
// Phase 1: Config and static data bootstrap
// ================================================================
static void bootstrapConfigAndStatics(int argc, char** argv) {
    beginPhaseGroup();
    bootstrap_config::initAll();
    endPhaseGroup();
    LOG_PHASE("Configs initialized", true);

    beginPhaseGroup();
    aliases::init();
    endPhaseGroup();
    LOG_PHASE("Aliases bootstrap finished", true);

    std::string fontPath = findAnyFontInResources(argc, argv, &history);
    if (!fontPath.empty()) {
        LOG_PHASE("Font search", true);
        LOG_DEBUG("Config", "Font found: " + fontPath);
    } else {
        LOG_ERROR("Config", "No system font found, UI may render incorrectly");
        LOG_PHASE("Font search", false);
    }
}

// ================================================================
// Phase 2: Hardware inventory and resource infrastructure
// ================================================================
static void bootstrapHardwareAndResources() {
    g_hardwareInventory = GRIM::MMO::detectHardware();
    GRIM::MMO::logHardwareInventory(g_hardwareInventory);
    LOG_PHASE("Hardware inventory", true);

    g_wifiConnected = detectWifiConnected();
    if (g_wifiConnected) {
        fetchLocationByIP(g_location);
    }
    LOG_PHASE("Location detect", true);

    {
        GRIM::MMO::ResourceSignalConfig sigCfg;
        g_resourceSignal = new GRIM::MMO::ResourceSignal();
        g_resourceSignal->start(sigCfg, g_hardwareInventory.gpu_count);
        LOG_PHASE("Resource signal started", true);
    }

    {
        GRIM::MMO::CoordinatorConfig coordCfg;
        g_resourceCoordinator = new GRIM::MMO::ResourceCoordinator(
            g_hardwareInventory, *g_resourceSignal, coordCfg);
        LOG_PHASE("Resource coordinator ready", true);
    }
}

// ================================================================
// Phase 3: Subsystem initialization (voice, RL, server)
// ================================================================
static void bootstrapSubsystems() {
    auto voiceCfg = aiConfig["voice"];
    bool needsCoqui = (voiceCfg.value("engine", "sapi") == "coqui");

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

    LOG_DEBUG("RL", "Starting rl_bridge.py via unified BridgeManager...");
    if (!GRIM::RL::init()) {
        LOG_ERROR("RL", "Failed to start rl_bridge.py");
        LOG_PHASE("RL Bridge init", false);
    } else {
        LOG_PHASE("RL Bridge init", true);
        LOG_DEBUG("RL", "rl_bridge.py initialized and ready");
    }

    if (aiConfig.value("backend", "auto") == "grim_native") {
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
}

// ================================================================
// Backend factory — creates IGenerationBackend from ModelInfo
// ================================================================
static std::unique_ptr<GRIM::MMO::IGenerationBackend>
createBackendForModel(const GRIM::MMO::ModelInfo& info) {
    switch (info.backend_type) {
    case GRIM::MMO::BackendType::GrimTextServer:
        return std::make_unique<GRIM::MMO::GrimNativeBackend>(
            info.url, info.id);

    case GRIM::MMO::BackendType::Ollama: {
        // For Ollama, the model_path stores the Ollama model name
        std::string ollama_model = info.model_path;
        if (ollama_model.empty()) {
            ollama_model = info.name;
        }
        return std::make_unique<GRIM::MMO::OllamaBackend>(
            info.url, info.id, ollama_model);
    }

    case GRIM::MMO::BackendType::LlamaCpp: {
        // For llama.cpp, model_path stores the model name for the server
        std::string llama_model = info.model_path;
        if (llama_model.empty()) {
            llama_model = info.name;
        }
        return std::make_unique<GRIM::MMO::LlamaCppBackend>(
            info.url, info.id, llama_model);
    }

    case GRIM::MMO::BackendType::External: {
        // For external backends, model_path stores the endpoint path
        std::string endpoint_path = info.model_path;
        if (endpoint_path.empty()) {
            endpoint_path = "/v1/chat/completions";
        }
        return std::make_unique<GRIM::MMO::ExternalBackend>(
            info.url, info.id, endpoint_path);
    }
    }

    return nullptr;
}

// ================================================================
// Phase 4: MMO layer registration
// ================================================================
static void bootstrapMMOLayer() {
    // ModelRegistry
    auto& registry = GRIM::MMO::ModelRegistry::instance();
    registry.loadFromConfig(aiConfig);
    LOG_PHASE("MMO model registry loaded", true);

    if (registry.isEnabled()) {
        LOG_DEBUG("Bootstrap", "MMO enabled (mode=" + registry.mode() + ")");

        // ModelLoader — load config from ai_config.json → mmo.model_loader
        GRIM::MMO::ModelLoaderConfig loaderCfg;
        if (aiConfig.contains("mmo") && aiConfig["mmo"].contains("model_loader")) {
            auto& ml = aiConfig["mmo"]["model_loader"];
            loaderCfg.load_timeout_ms     = ml.value("load_timeout_ms", 30000);
            loaderCfg.idle_ttl_ms         = ml.value("idle_ttl_ms", 60000);
            loaderCfg.hot_ttl_cap_ms      = ml.value("hot_ttl_cap_ms", 300000);
            loaderCfg.use_degrade_step_ms = ml.value("use_degrade_step_ms", 5000);
        }
        g_modelLoader = new GRIM::MMO::ModelLoader(
            registry, *g_resourceCoordinator, loaderCfg);

        g_modelLoader->setStartCallback([](const GRIM::MMO::ModelInfo& info) -> bool {
            if (info.backend_type == GRIM::MMO::BackendType::GrimTextServer) {
                auto& mgr = GRIM::GRIMTextServerManager::getInstance();
                mgr.setServerURL(info.url);
                return mgr.start();
            }
            return true;
        });
        g_modelLoader->setStopCallback([](const GRIM::MMO::ModelInfo& info) {
            if (info.backend_type == GRIM::MMO::BackendType::GrimTextServer) {
                GRIM::stopGRIMTextServer();
            }
        });

        // Orchestrator — load config from ai_config.json → mmo.orchestrator
        GRIM::MMO::OrchestratorConfig orchCfg;
        if (aiConfig.contains("mmo") && aiConfig["mmo"].contains("orchestrator")) {
            auto& oc = aiConfig["mmo"]["orchestrator"];
            orchCfg.route_timeout_ms            = oc.value("route_timeout_ms", 10000);
            orchCfg.generate_timeout_ms         = oc.value("generate_timeout_ms", 30000);
            orchCfg.synthesize_timeout_ms       = oc.value("synthesize_timeout_ms", 10000);
            orchCfg.max_submodels_per_request   = oc.value("max_submodels_per_request", 1);
        }
        g_orchestrator = new GRIM::MMO::Orchestrator(
            registry, *g_modelLoader, orchCfg);
        LOG_PHASE("MMO orchestrator ready", true);

        // Register backends for all configured models
        auto registerBackendsForModels = [&]() {
            // Router backend
            const auto* router = registry.getRouter();
            if (router) {
                auto backend = createBackendForModel(*router);
                if (backend) {
                    g_orchestrator->registerBackend(router->id, std::move(backend));
                }
            }

            // Sub-model backends
            for (const auto& sub : registry.getSubModels()) {
                auto backend = createBackendForModel(*sub);
                if (backend) {
                    g_orchestrator->registerBackend(sub->id, std::move(backend));
                }
            }
        };
        registerBackendsForModels();
        LOG_PHASE("MMO backends registered", true);

        // Background idle-tick thread
        s_idleTickStop = false;
        s_idleTickThread = std::thread([] {
            while (!s_idleTickStop.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                if (g_modelLoader && !s_idleTickStop.load(std::memory_order_relaxed)) {
                    g_modelLoader->tickIdleTimers();
                }
            }
        });
        s_idleTickThread.detach();
    } else {
        LOG_DEBUG("Bootstrap", "MMO disabled in config");
    }

    // ToolRegistry — seed from CommandRegistry
    {
        auto& toolReg = GRIM::MMO::ToolRegistry::instance();
        auto legacyTools = GRIM::CommandRegistry::getAllTools();
        for (const auto& legacy : legacyTools) {
            GRIM::MMO::ToolDescriptor desc;
            desc.tool_id        = legacy.name;
            desc.display_name   = legacy.name;
            desc.provider_type  = legacy.fromPlugin
                ? GRIM::MMO::ToolProviderType::Plugin
                : GRIM::MMO::ToolProviderType::Builtin;
            desc.provider_name  = legacy.fromPlugin ? legacy.pluginName : "builtin";
            desc.description    = legacy.description;
            desc.usage          = legacy.usage;
            desc.category       = legacy.category;
            desc.is_informational   = legacy.isInformational;
            desc.needs_confirmation = legacy.needsConfirmation;
            desc.aliases        = legacy.aliases;
            desc.examples       = legacy.examples;
            desc.keywords       = legacy.keywords;
            for (const auto& p : legacy.parameters) {
                GRIM::MMO::ToolParameter tp;
                tp.name        = p.name;
                tp.type        = p.type;
                tp.description = p.description;
                tp.required    = p.required;
                desc.parameters.push_back(tp);
            }
            desc.usage_count   = legacy.usageCount;
            desc.success_rate  = legacy.successRate;
            toolReg.registerTool(desc);
        }
        LOG_PHASE("MMO ToolRegistry loaded (" +
                  std::to_string(toolReg.toolCount()) + " tools)", true);
    }

    // ActionPolicyRegistry (Training Wheels gate)
    {
        GRIM::MMO::ActionPolicyConfig policyCfg;
        if (aiConfig.contains("training_wheels")) {
            auto& tw = aiConfig["training_wheels"];
            policyCfg.enabled               = tw.value("enabled", true);
            policyCfg.risk_threshold        = tw.value("risk_threshold", 0.6f);
            policyCfg.min_confidence_floor  = tw.value("min_confidence_floor", 0.3f);
            policyCfg.uncalibrated_router_conf = tw.value("uncalibrated_router_confidence", 0.5f);
            policyCfg.calibration_min_samples  = tw.value("calibration_min_samples", 10);

            if (tw.contains("per_category_thresholds")) {
                for (auto& [cat, val] : tw["per_category_thresholds"].items()) {
                    policyCfg.per_category_thresholds[cat] = val.get<float>();
                }
            }
        }
        auto& policyReg = GRIM::MMO::ActionPolicyRegistry::instance();
        policyReg.configure(policyCfg);
        LOG_PHASE("MMO ActionPolicy configured (enabled=" +
                  std::string(policyCfg.enabled ? "true" : "false") + ")", true);
    }

    // SessionContextManager (singleton auto-constructed)
    LOG_PHASE("MMO SessionContextManager ready", true);
}

// ================================================================
// Phase 5: Warmup and preload scheduling
// ================================================================
static void bootstrapWarmup() {
    if (aiConfig.value("backend", "ollama") == "ollama") {
        LOG_PHASE("AI model warmup begin", true);
        warmupOllamaModel();
        LOG_PHASE("AI model warmup complete", true);
    }

    LOG_PHASE("Whisper preload begin", true);
    try {
        if (Voice::getWhisperContext() == nullptr) {
            std::string modelName = "ggml-base.en.bin";
            if (aiConfig.contains("whisper") && aiConfig["whisper"].contains("whisper_model"))
                modelName = aiConfig["whisper"].value("whisper_model", modelName);

            fs::path modelPath = fs::path(getResourcePath()) / "models" / modelName;
            LOG_DEBUG("Voice", "Preloading Whisper model: " + modelPath.string());

            if (!fs::exists(modelPath)) {
                LOG_ERROR("Voice", "Whisper model missing: " + modelPath.string());
                LOG_PHASE("Whisper preload", false);
            } else {
                whisper_context_params wparams = whisper_context_default_params();
                whisper_context* ctx = whisper_init_from_file_with_params(
                    modelPath.string().c_str(), wparams);

                if (!ctx) {
                    LOG_ERROR("Voice", "Failed to initialize Whisper context from: " +
                             modelPath.string());
                    LOG_PHASE("Whisper preload", false);
                } else {
                    Voice::setWhisperContext(ctx);
                    LOG_PHASE("Whisper preload complete", true);
                    LOG_DEBUG("Voice", "Whisper model preloaded successfully (no listening loop)");
                }
            }
        } else {
            LOG_DEBUG("Voice", "Whisper already loaded, skipping preload");
            LOG_PHASE("Whisper preload complete", true);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Bootstrap", std::string("Whisper preload failed: ") + e.what());
        LOG_PHASE("Whisper preload", false);
    }
}

// ================================================================
// Public entry point — orchestrates all phases
// ================================================================
void runBootstrapChecks(int argc, char** argv) {
    bootstrapConfigAndStatics(argc, argv);     // Phase 1: config, aliases, fonts
    bootstrapHardwareAndResources();            // Phase 2: inventory, signal, coordinator
    bootstrapSubsystems();                      // Phase 3: voice, RL, server
    bootstrapMMOLayer();                        // Phase 4: registry, loader, orchestrator, tools, policy
    bootstrapWarmup();                          // Phase 5: model warmup, Whisper preload
    LOG_PHASE("Bootstrap complete", true);
}
