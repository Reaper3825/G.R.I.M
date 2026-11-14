#include "bootstrap.hpp"
#include "dataset_bootstrap.hpp"
#include "dataset_services_bootstrap.hpp"
#include "../core/grim_text_server_api.hpp"
#include "bootstrap_config.hpp"
#include "resources.hpp"
#include "console_history.hpp"
#include "device_setups/audio_devices.hpp"
#include "logger.hpp"
#include "ai/ai.hpp"
#include "ai/grim_text_server_manager.hpp"  // For GRIM-text server
#include "../MMO/Core/HardwareInventory.hpp"
#include "../MMO/Core/ResourceSignal.hpp"
#include "../MMO/Core/ResourceCoordinator.hpp"
#include "../MMO/Core/Registries/ModelRegistry.hpp"
#include "../MMO/Core/ModelLoader.hpp"
#include "../MMO/Core/ProcessManager.hpp"
#include "../MMO/Core/Orchestrator.hpp"
#include "../MMO/Core/SessionContextManager.hpp"
#include "../MMO/Core/Registries/ToolRegistry.hpp"
#include "../MMO/Core/ActionPolicyRegistry.hpp"
#include "../MMO/Backends/GrimNativeBackend.hpp"
#include "../MMO/Backends/OllamaBackend.hpp"
#include "../MMO/Backends/LlamaCppBackend.hpp"
#include "../MMO/Backends/ExternalBackend.hpp"
#include "../commands/commands_core.hpp"
#include "../core/plugin_manager.hpp"
#include "../location.hpp"

#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <sstream>
#include <vector>

// Global resource layer (replaces old g_systemInfo)
GRIM::MMO::HardwareInventory g_hardwareInventory;
GRIM::MMO::ResourceSignal*   g_resourceSignal      = nullptr;
GRIM::MMO::ResourceCoordinator* g_resourceCoordinator = nullptr;

// Global MMO orchestration layer
GRIM::MMO::ModelLoader*      g_modelLoader      = nullptr;
GRIM::MMO::ProcessManager*   g_processManager   = nullptr;
GRIM::MMO::Orchestrator*     g_orchestrator     = nullptr;
GRIM::MMO::MemoryFacade*     g_memoryFacade     = nullptr;

// Idle-tick background thread for ModelLoader TTL management
static std::atomic<bool> s_idleTickStop{false};
static std::thread       s_idleTickThread;

void stopMMOIdleTick() {
    s_idleTickStop.store(true, std::memory_order_relaxed);
}

namespace fs = std::filesystem;

static void bootstrapConfigAndStatics() {
    beginPhaseGroup();
    bootstrap_config::initAll();
    endPhaseGroup();
    LOG_PHASE("Configs initialized", true);


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
    if (aiConfig.value("backend", "auto") == "grim_native") {
        LOG_DEBUG("Bootstrap", "Starting GRIM-text server...");
        if (GRIM::startGRIMTextServer()) {
            LOG_PHASE("GRIM-text server startup", true);
            LOG_DEBUG("Bootstrap", "GRIM-text server running at " +
                     aiConfig.value("grim_text_url", GRIM::ServerAPI::defaultPublicURL()));
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

    case GRIM::MMO::BackendType::InProcessVision:
        // Vision sub-models do not expose a text-generation backend;
        // they are dispatched through PhysicalPerceptionPrimitivesLoop.
        // Return nullptr so the caller skips registerBackend().
        return nullptr;

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

    {
        LOG_DEBUG("Bootstrap", "MMO layer init (mode=" + registry.mode() + ", enabled=" + std::string(registry.isEnabled() ? "true" : "false") + ")");

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

        // Model-keyed process manager — owns one OS process per model
        g_processManager = new GRIM::MMO::ProcessManager();

        g_modelLoader->setStartCallback([](const GRIM::MMO::ModelInfo& info) -> bool {
            if (!g_processManager) return false;
            return g_processManager->start(info);
        });
        g_modelLoader->setStopCallback([](const GRIM::MMO::ModelInfo& info) {
            if (g_processManager) {
                g_processManager->stop(info.id);
            }
        });

        // Orchestrator — load config from ai_config.json → mmo.orchestrator
        GRIM::MMO::OrchestratorConfig orchCfg;
        if (aiConfig.contains("mmo") && aiConfig["mmo"].contains("orchestrator")) {
            auto& oc = aiConfig["mmo"]["orchestrator"];
            orchCfg.route_timeout_ms            = oc.value("route_timeout_ms", 600000);
            orchCfg.generate_timeout_ms         = oc.value("generate_timeout_ms", 600000);
            orchCfg.synthesize_timeout_ms       = oc.value("synthesize_timeout_ms", 600000);
            orchCfg.max_submodels_per_request   = oc.value("max_submodels_per_request", 1);
            orchCfg.correction_output_path      = oc.value("correction_output_path", "correction_tuples.jsonl");
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
                // Vision sub-models are wired into PhysicalPerceptionPrimitivesLoop
                // below — they have no HTTP generation backend.
                if (sub->kind == GRIM::MMO::ModelKind::Vision) {
                    continue;
                }
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
    }

    LOG_PHASE("MMO ToolRegistry loaded (" +
              std::to_string(GRIM::MMO::ToolRegistry::instance().toolCount()) + " tools)", true);

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

    // UISurfaceRegistry (singleton auto-constructed)
    // Touch the singleton so it's alive before any UI tools reference it.
    (void)GRIM::MMO::UISurfaceRegistry::instance();
    LOG_PHASE("MMO UISurfaceRegistry ready", true);
}

// ================================================================
// Phase 5: Warmup and preload scheduling
// ================================================================
// ================================================================
// Public entry point — orchestrates all phases
// ================================================================
void runBootstrapChecks(int argc, char** argv) {
    (void)argc;
    (void)argv;
    bootstrapConfigAndStatics();               // Phase 1: config and aliases
    bootstrapDatasetSubsystem();               // Source catalog available before runtime/UI
    bootstrapDatasetServices();                // Host-owned authoring and collection services
    bootstrapHardwareAndResources();            // Phase 2: inventory, signal, coordinator
    bootstrapSubsystems();                      // Phase 3: voice, RL, server
    bootstrapMMOLayer();                        // Phase 4: registry, loader, orchestrator, tools, policy
    LOG_PHASE("Bootstrap complete", true);
}
