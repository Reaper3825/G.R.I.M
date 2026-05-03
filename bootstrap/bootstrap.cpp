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
#include "ai/ai.hpp"
#include "ai/grim_text_server_manager.hpp"  // For GRIM-text server
#include "../MMO/Core/HardwareInventory.hpp"
#include "../MMO/Core/ResourceSignal.hpp"
#include "../MMO/Core/ResourceCoordinator.hpp"
#include "../MMO/Core/ModelRegistry.hpp"
#include "../MMO/Core/ModelLoader.hpp"
#include "../MMO/Core/ProcessManager.hpp"
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
#include "../perception/physical/PhysicalPerceptionPrimitivesLoop.hpp"
#include "../perception/physical/PhysicalSpatialGroundingLoop.hpp"
#include "../perception/physical/PhysicalMonocularDepthEstimator.hpp"

#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <vector>
#include <opencv2/dnn.hpp>
#include <whisper.h>

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

static std::string ResolveConfigPathAgainstGrimRoot(const std::string& path) {
    if (path.empty()) {
        return {};
    }
    fs::path p(path);
    if (p.is_absolute()) {
        return p.lexically_normal().string();
    }
    fs::path resolved = fs::path(getGrimRootDir()) / p;
    return resolved.lexically_normal().string();
}

static int ParsePhysicalDnnBackendId(const std::string& model_id, const std::string& value) {
    if (value == "opencv") {
        return cv::dnn::DNN_BACKEND_OPENCV;
    }
    if (value == "cuda") {
        return cv::dnn::DNN_BACKEND_CUDA;
    }
    if (value == "default") {
        return cv::dnn::DNN_BACKEND_DEFAULT;
    }
    throw std::runtime_error(
        "vision sub-model '" + model_id + "': invalid dnn_backend='" + value
        + "' (expected 'opencv', 'cuda', or 'default')");
}

static int ParsePhysicalDnnTargetId(const std::string& model_id, const std::string& value) {
    if (value == "cpu") {
        return cv::dnn::DNN_TARGET_CPU;
    }
    if (value == "opencl") {
        return cv::dnn::DNN_TARGET_OPENCL;
    }
    if (value == "opencl_fp16") {
        return cv::dnn::DNN_TARGET_OPENCL_FP16;
    }
    if (value == "cuda") {
        return cv::dnn::DNN_TARGET_CUDA;
    }
    if (value == "cuda_fp16") {
        return cv::dnn::DNN_TARGET_CUDA_FP16;
    }
    throw std::runtime_error(
        "vision sub-model '" + model_id + "': invalid dnn_target='" + value
        + "' (expected 'cpu', 'opencl', 'opencl_fp16', 'cuda', or 'cuda_fp16')");
}

template <typename Config>
static void ApplyPhysicalDnnExecutionPolicy(Config& cfg, const GRIM::MMO::ModelInfo& model) {
    if (!model.vision.dnn_backend.empty()) {
        cfg.dnn_backend_id = ParsePhysicalDnnBackendId(model.id, model.vision.dnn_backend);
    }
    if (!model.vision.dnn_target.empty()) {
        cfg.dnn_target_id = ParsePhysicalDnnTargetId(model.id, model.vision.dnn_target);
    }
}

struct RequiredPhysicalVisionPath {
    std::string label;
    fs::path    path;
};

static void AddRequiredPhysicalVisionPath(
    std::vector<RequiredPhysicalVisionPath>& out,
    const std::string& label,
    const std::string& config_path)
{
    if (config_path.empty()) {
        return;
    }
    out.push_back({ label, fs::path(ResolveConfigPathAgainstGrimRoot(config_path)) });
}

static std::vector<RequiredPhysicalVisionPath> CollectRequiredPhysicalVisionPaths(
    GRIM::MMO::ModelRegistry& registry)
{
    std::vector<RequiredPhysicalVisionPath> out;
    for (const auto* vm : registry.getVisionSubModels()) {
        if (!vm) {
            throw std::runtime_error(
                "CollectRequiredPhysicalVisionPaths: registry returned NULL vision model");
        }
        if (!vm->enabled) {
            continue;
        }

        const std::string prefix = vm->id + ": ";
        AddRequiredPhysicalVisionPath(out, prefix + "model_path", vm->model_path);
        AddRequiredPhysicalVisionPath(out, prefix + "class_names_path", vm->vision.class_names_path);
        AddRequiredPhysicalVisionPath(out, prefix + "text_embeddings_path", vm->vision.text_embeddings_path);
        AddRequiredPhysicalVisionPath(out, prefix + "recogniser_onnx_path", vm->vision.recogniser_onnx_path);
        AddRequiredPhysicalVisionPath(out, prefix + "recogniser_charset_path", vm->vision.recogniser_charset_path);
        AddRequiredPhysicalVisionPath(out, prefix + "expression_classifier_onnx_path", vm->vision.expression_classifier_onnx_path);
        AddRequiredPhysicalVisionPath(out, prefix + "expression_classifier_class_names_path", vm->vision.expression_classifier_class_names_path);
        AddRequiredPhysicalVisionPath(out, prefix + "instance_seg_decoder_onnx_path", vm->vision.instance_seg_decoder_onnx_path);
    }
    return out;
}

static std::vector<RequiredPhysicalVisionPath> FindMissingPhysicalVisionPaths(
    const std::vector<RequiredPhysicalVisionPath>& required)
{
    std::vector<RequiredPhysicalVisionPath> missing;
    for (const auto& item : required) {
        std::error_code ec;
        if (!fs::exists(item.path, ec) || !fs::is_regular_file(item.path, ec) || fs::file_size(item.path, ec) == 0) {
            missing.push_back(item);
        }
    }
    return missing;
}

static std::string FormatMissingPhysicalVisionPaths(
    const std::vector<RequiredPhysicalVisionPath>& missing)
{
    std::ostringstream oss;
    for (const auto& item : missing) {
        oss << "\n  - " << item.label << " -> " << item.path.string();
    }
    return oss.str();
}

#ifdef _WIN32
static std::string QuoteWindowsCommandArg(const fs::path& path) {
    std::string s = path.string();
    std::string quoted;
    quoted.reserve(s.size() + 2);
    quoted.push_back('"');
    for (char ch : s) {
        if (ch == '"') {
            quoted += "\\\"";
        } else {
            quoted.push_back(ch);
        }
    }
    quoted.push_back('"');
    return quoted;
}
#endif

static void EnsurePhysicalVisionModelsAvailable(GRIM::MMO::ModelRegistry& registry) {
    const auto required = CollectRequiredPhysicalVisionPaths(registry);
    auto missing = FindMissingPhysicalVisionPaths(required);
    if (missing.empty()) {
        LOG_DEBUG("Bootstrap", "Physical vision model assets present");
        LOG_PHASE("Physical vision model assets", true);
        return;
    }

    LOG_ERROR("Bootstrap", "Physical vision model assets missing:" + FormatMissingPhysicalVisionPaths(missing));

#ifdef _WIN32
    const fs::path setup_script = fs::path(getGrimRootDir()) / "scripts" / "setup_windows_physical_vision.ps1";
    if (!fs::exists(setup_script)) {
        throw std::runtime_error(
            "EnsurePhysicalVisionModelsAvailable: setup script missing: " + setup_script.string());
    }

    LOG_DEBUG("Bootstrap", "Running Windows physical vision setup script: " + setup_script.string());
    LOG_PHASE("Physical vision model auto-setup start", true);

    const std::string command =
        "powershell -NoProfile -ExecutionPolicy Bypass -File "
        + QuoteWindowsCommandArg(setup_script);
    const int rc = std::system(command.c_str());
    if (rc != 0) {
        throw std::runtime_error(
            "EnsurePhysicalVisionModelsAvailable: setup script failed with exit code "
            + std::to_string(rc) + ": " + setup_script.string());
    }

    missing = FindMissingPhysicalVisionPaths(required);
    if (!missing.empty()) {
        throw std::runtime_error(
            "EnsurePhysicalVisionModelsAvailable: setup script completed but required files are still missing:"
            + FormatMissingPhysicalVisionPaths(missing));
    }

    LOG_PHASE("Physical vision model auto-setup complete", true);
#else
    throw std::runtime_error(
        "EnsurePhysicalVisionModelsAvailable: physical vision model assets are missing and automatic startup download is only wired for Windows:"
        + FormatMissingPhysicalVisionPaths(missing));
#endif
}

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
    EnsurePhysicalVisionModelsAvailable(registry);

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
            orchCfg.route_timeout_ms            = oc.value("route_timeout_ms", 10000);
            orchCfg.generate_timeout_ms         = oc.value("generate_timeout_ms", 30000);
            orchCfg.synthesize_timeout_ms       = oc.value("synthesize_timeout_ms", 10000);
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

        // ----------------------------------------------------------------
        // Wire vision sub-models into PhysicalPerceptionPrimitivesLoop.
        // The loop is the in-process host for the OpenCV cv::dnn vision
        // operators; each registered vision sub-model becomes the active
        // ONNX for one operator slot.
        // ----------------------------------------------------------------
        namespace PE = GRIM::Perception::Physical;
        for (const auto& vm : registry.getVisionSubModels()) {
            if (!vm->enabled) {
                LOG_PHASE(std::string("MMO vision sub-model skipped: ") + vm->id
                          + " (enabled=false)", true);
                continue;
            }
            try {
                using Op = GRIM::MMO::VisionOperatorKind;
                switch (vm->vision.operator_kind) {
                case Op::ObjectDetector: {
                    PE::PhysicalObjectDetectorConfig c;
                    c.onnx_model_path  = ResolveConfigPathAgainstGrimRoot(vm->model_path);
                    c.class_names_path = ResolveConfigPathAgainstGrimRoot(vm->vision.class_names_path);
                    ApplyPhysicalDnnExecutionPolicy(c, *vm);
                    if (vm->vision.input_width  > 0) c.input_width  = vm->vision.input_width;
                    if (vm->vision.input_height > 0) c.input_height = vm->vision.input_height;
                    if (vm->vision.confidence_threshold > 0.0f)
                        c.confidence_threshold = vm->vision.confidence_threshold;
                    if (vm->vision.iou_threshold > 0.0f)
                        c.iou_threshold = vm->vision.iou_threshold;
                    // Cadence: detector feeds the tracker, so we want it
                    // fresh on motion but reusable on a truly stable scene.
                    // 33 ms floor caps it at ~30 fps even when the camera
                    // pipeline runs faster.
                    c.cadence.reuse_on_stable_scene = true;
                    c.cadence.min_period_ms         = 33;
                    PE::RequestConfigurePhysicalObjectDetector(c);
                    break;
                }
                case Op::SemanticSegmenter: {
                    PE::PhysicalSemanticSegmenterConfig c;
                    c.onnx_model_path  = ResolveConfigPathAgainstGrimRoot(vm->model_path);
                    c.class_names_path = ResolveConfigPathAgainstGrimRoot(vm->vision.class_names_path);
                    ApplyPhysicalDnnExecutionPolicy(c, *vm);
                    if (vm->vision.input_width  > 0) c.input_width  = vm->vision.input_width;
                    if (vm->vision.input_height > 0) c.input_height = vm->vision.input_height;
                    // Semantic segmentation is dense + expensive. Hard cap
                    // at ~10 fps and skip entirely on a stable scene.
                    c.cadence.reuse_on_stable_scene = true;
                    c.cadence.min_period_ms         = 100;
                    PE::RequestConfigurePhysicalSemanticSegmenter(c);
                    break;
                }
                case Op::ImageClassifier: {
                    PE::PhysicalImageClassifierConfig c;
                    c.onnx_model_path      = ResolveConfigPathAgainstGrimRoot(vm->model_path);
                    c.class_names_path     = ResolveConfigPathAgainstGrimRoot(vm->vision.class_names_path);
                    c.text_embeddings_path = ResolveConfigPathAgainstGrimRoot(vm->vision.text_embeddings_path);
                    if (vm->vision.input_width  > 0) c.input_width  = vm->vision.input_width;
                    if (vm->vision.input_height > 0) c.input_height = vm->vision.input_height;
                    if (vm->vision.top_k > 0) c.top_k = vm->vision.top_k;
                    // Whole-image classification — semantically a scene
                    // label. Cheap to reuse, ~5 fps refresh is plenty.
                    c.cadence.reuse_on_stable_scene = true;
                    c.cadence.min_period_ms         = 200;
                    PE::RequestConfigurePhysicalImageClassifier(c);
                    break;
                }
                case Op::PoseEstimator: {
                    PE::PhysicalPoseKeypointEstimatorConfig c;
                    c.onnx_model_path  = ResolveConfigPathAgainstGrimRoot(vm->model_path);
                    c.joint_names_path = ResolveConfigPathAgainstGrimRoot(vm->vision.class_names_path);
                    ApplyPhysicalDnnExecutionPolicy(c, *vm);
                    if (vm->vision.input_width  > 0) c.input_width  = vm->vision.input_width;
                    if (vm->vision.input_height > 0) c.input_height = vm->vision.input_height;
                    if (vm->vision.min_keypoint_confidence > 0.0f)
                        c.min_keypoint_confidence = vm->vision.min_keypoint_confidence;
                    // Output-format selection. Empty / "heatmap" → default Heatmap.
                    // "yolo_anchor" → YOLOv8-pose decode.
                    if (vm->vision.pose_output_format == "yolo_anchor") {
                        c.output_format = PE::PhysicalPoseKeypointEstimatorConfig::OutputFormat::YoloAnchor;
                    } else if (vm->vision.pose_output_format.empty() ||
                               vm->vision.pose_output_format == "heatmap") {
                        c.output_format = PE::PhysicalPoseKeypointEstimatorConfig::OutputFormat::Heatmap;
                    } else {
                        throw std::runtime_error(
                            "vision sub-model '" + vm->id + "': unknown pose_output_format='"
                            + vm->vision.pose_output_format + "' (expected 'heatmap' or 'yolo_anchor')");
                    }
                    if (vm->vision.num_keypoints > 0) c.num_keypoints = vm->vision.num_keypoints;
                    if (vm->vision.confidence_threshold > 0.0f)
                        c.person_confidence_threshold = vm->vision.confidence_threshold;
                    if (vm->vision.iou_threshold > 0.0f)
                        c.nms_iou_threshold = vm->vision.iou_threshold;
                    // Pose tracks fast human motion. Reuse only on a truly
                    // stable scene; otherwise hold a 30 fps floor.
                    c.cadence.reuse_on_stable_scene = true;
                    c.cadence.min_period_ms         = 33;
                    PE::RequestConfigurePhysicalPoseKeypointEstimator(c);
                    break;
                }
                case Op::SceneTextReader: {
                    PE::PhysicalSceneTextReaderConfig c;
                    c.detector_onnx_path        = ResolveConfigPathAgainstGrimRoot(vm->model_path);
                    c.recogniser_onnx_path      = ResolveConfigPathAgainstGrimRoot(vm->vision.recogniser_onnx_path);
                    c.recogniser_charset_path   = ResolveConfigPathAgainstGrimRoot(vm->vision.recogniser_charset_path);
                    c.recogniser_input_grayscale = vm->vision.recogniser_input_grayscale;
                    ApplyPhysicalDnnExecutionPolicy(c, *vm);
                    if (vm->vision.input_width  > 0) c.detector_input_width  = vm->vision.input_width;
                    if (vm->vision.input_height > 0) c.detector_input_height = vm->vision.input_height;
                    // Scene text is among the slowest-changing signals in
                    // a typical scene. ~2 fps + reuse on stable scene.
                    c.cadence.reuse_on_stable_scene = true;
                    c.cadence.min_period_ms         = 500;
                    PE::RequestConfigurePhysicalSceneTextReader(c);
                    break;
                }
                case Op::FacialExpressionDetector: {
                    PE::PhysicalFacialExpressionDetectorConfig c;
                    c.detector_onnx_path        = ResolveConfigPathAgainstGrimRoot(vm->model_path);
                    c.classifier_onnx_path      = ResolveConfigPathAgainstGrimRoot(vm->vision.expression_classifier_onnx_path);
                    c.classifier_class_names_path = ResolveConfigPathAgainstGrimRoot(vm->vision.expression_classifier_class_names_path);
                    ApplyPhysicalDnnExecutionPolicy(c, *vm);
                    if (vm->vision.input_width  > 0) c.detector_input_width  = vm->vision.input_width;
                    if (vm->vision.input_height > 0) c.detector_input_height = vm->vision.input_height;
                    if (vm->vision.confidence_threshold > 0.0f)
                        c.detector_score_threshold = vm->vision.confidence_threshold;
                    if (vm->vision.iou_threshold > 0.0f)
                        c.detector_nms_threshold = vm->vision.iou_threshold;
                    if (vm->vision.expression_classifier_input_width  > 0)
                        c.classifier_input_width  = vm->vision.expression_classifier_input_width;
                    if (vm->vision.expression_classifier_input_height > 0)
                        c.classifier_input_height = vm->vision.expression_classifier_input_height;
                    c.classifier_input_grayscale = vm->vision.expression_classifier_input_grayscale;
                    // Expressions change rapidly even when the user is
                    // physically still (eyebrows, mouth, blinks). Do NOT
                    // reuse on stable scene — always run fresh — and hold
                    // a 30 fps floor so the signal is effectively per-frame
                    // at typical webcam rates.
                    c.cadence.reuse_on_stable_scene = false;
                    c.cadence.min_period_ms         = 33;
                    PE::RequestConfigurePhysicalFacialExpressionDetector(c);
                    break;
                }
                case Op::MonocularDepthEstimator: {
                    PE::PhysicalMonocularDepthEstimatorConfig c;
                    c.onnx_model_path = ResolveConfigPathAgainstGrimRoot(vm->model_path);
                    ApplyPhysicalDnnExecutionPolicy(c, *vm);
                    if (vm->vision.input_width  > 0) c.input_width  = vm->vision.input_width;
                    if (vm->vision.input_height > 0) c.input_height = vm->vision.input_height;
                    c.input_scale = static_cast<float>(vm->vision.depth_input_scale);
                    c.input_mean  = cv::Scalar(vm->vision.depth_input_mean_r,
                                               vm->vision.depth_input_mean_g,
                                               vm->vision.depth_input_mean_b);
                    c.input_std   = cv::Scalar(vm->vision.depth_input_std_r,
                                               vm->vision.depth_input_std_g,
                                               vm->vision.depth_input_std_b);
                    c.swap_rb              = vm->vision.depth_swap_rb;
                    c.output_is_disparity  = vm->vision.depth_output_is_disparity;
                    c.metric_scale_meters  = vm->vision.depth_metric_scale_meters;
                    c.metric_epsilon       = static_cast<float>(vm->vision.depth_metric_epsilon);
                    // Depth dominates Stage-3 latency. Cap at ~10 fps and
                    // reuse on a stable scene; spatial grounding fuses the
                    // cached map with fresh tracker boxes every frame.
                    c.cadence.reuse_on_stable_scene = true;
                    c.cadence.min_period_ms         = 100;
                    PE::RequestConfigurePhysicalMonocularDepthEstimator(c);
                    break;
                }
                case Op::InstanceSegmenter: {
                    PE::PhysicalInstanceSegmenterConfig c;
                    c.encoder_onnx_path = ResolveConfigPathAgainstGrimRoot(vm->model_path);
                    c.decoder_onnx_path = ResolveConfigPathAgainstGrimRoot(vm->vision.instance_seg_decoder_onnx_path);
                    if (!vm->vision.instance_seg_encoder_input_name.empty())
                        c.encoder_input_name = vm->vision.instance_seg_encoder_input_name;
                    if (!vm->vision.instance_seg_encoder_output_image_embed_name.empty())
                        c.encoder_output_image_embed_name = vm->vision.instance_seg_encoder_output_image_embed_name;
                    if (!vm->vision.instance_seg_encoder_output_high_res_feats_0_name.empty())
                        c.encoder_output_high_res_feats_0_name = vm->vision.instance_seg_encoder_output_high_res_feats_0_name;
                    if (!vm->vision.instance_seg_encoder_output_high_res_feats_1_name.empty())
                        c.encoder_output_high_res_feats_1_name = vm->vision.instance_seg_encoder_output_high_res_feats_1_name;
                    if (!vm->vision.instance_seg_decoder_input_image_embed_name.empty())
                        c.decoder_input_image_embed_name = vm->vision.instance_seg_decoder_input_image_embed_name;
                    if (!vm->vision.instance_seg_decoder_input_high_res_feats_0_name.empty())
                        c.decoder_input_high_res_feats_0_name = vm->vision.instance_seg_decoder_input_high_res_feats_0_name;
                    if (!vm->vision.instance_seg_decoder_input_high_res_feats_1_name.empty())
                        c.decoder_input_high_res_feats_1_name = vm->vision.instance_seg_decoder_input_high_res_feats_1_name;
                    if (!vm->vision.instance_seg_decoder_input_point_coords_name.empty())
                        c.decoder_input_point_coords_name = vm->vision.instance_seg_decoder_input_point_coords_name;
                    if (!vm->vision.instance_seg_decoder_input_point_labels_name.empty())
                        c.decoder_input_point_labels_name = vm->vision.instance_seg_decoder_input_point_labels_name;
                    if (!vm->vision.instance_seg_decoder_input_mask_input_name.empty())
                        c.decoder_input_mask_input_name = vm->vision.instance_seg_decoder_input_mask_input_name;
                    if (!vm->vision.instance_seg_decoder_input_has_mask_input_name.empty())
                        c.decoder_input_has_mask_input_name = vm->vision.instance_seg_decoder_input_has_mask_input_name;
                    if (!vm->vision.instance_seg_decoder_output_masks_name.empty())
                        c.decoder_output_masks_name = vm->vision.instance_seg_decoder_output_masks_name;
                    if (!vm->vision.instance_seg_decoder_output_iou_predictions_name.empty())
                        c.decoder_output_iou_predictions_name = vm->vision.instance_seg_decoder_output_iou_predictions_name;
                    if (vm->vision.input_width  > 0) c.encoder_input_width  = vm->vision.input_width;
                    if (vm->vision.input_height > 0) c.encoder_input_height = vm->vision.input_height;
                    if (vm->vision.instance_seg_max_prompts_per_frame > 0)
                        c.max_prompts_per_frame = vm->vision.instance_seg_max_prompts_per_frame;
                    if (vm->vision.instance_seg_min_prompt_confidence > 0.0f)
                        c.min_prompt_confidence = vm->vision.instance_seg_min_prompt_confidence;
                    // SAM-2 encoder is the heaviest operator on the bus.
                    // Cap at ~5 fps and reuse on stable scene; the
                    // mask-decoder pass with fresh prompts is still cheap.
                    c.cadence.reuse_on_stable_scene = true;
                    c.cadence.min_period_ms         = 200;
                    PE::RequestConfigurePhysicalInstanceSegmenter(c);
                    break;
                }
                case Op::Unknown:
                    throw std::runtime_error(
                        "vision sub-model '" + vm->id + "' has unknown operator kind");
                }
                LOG_PHASE(std::string("MMO vision sub-model wired: ") + vm->id
                          + " → " + GRIM::MMO::ModelRegistry::visionOperatorKindToString(
                                        vm->vision.operator_kind),
                          true);
            } catch (const std::exception& e) {
                LOG_PHASE(std::string("MMO vision sub-model wiring failed: ")
                          + vm->id + " — " + e.what(), false);
            }
        }

        // ----------------------------------------------------------------
        // Class-policy: collapse the YOLO/COCO classes that the detector
        // routinely confuses on the same physical object (chair vs couch
        // is the canonical case — back-of-couch fragments produce two
        // tracks) and rank the household classes the assistant cares
        // about so the model-context summary is stable.
        //
        // post_merge_nms_iou=0.55 then collapses surviving overlapping
        // tracks/dets/masks that share the post-merge canonical label.
        // ----------------------------------------------------------------
        try {
            PE::PhysicalClassPolicyConfig pol;
            pol.merge_rules = {
                // COCO furniture confusions on the same physical surface.
                { /*canonical=*/"couch",  /*sources=*/{"chair", "bench"} },
                // COCO display devices.
                { /*canonical=*/"screen", /*sources=*/{"tv", "laptop"} },
            };
            pol.priority_rules = {
                { /*canonical=*/"person", /*rank=*/1, /*floor=*/0.0f },
                { /*canonical=*/"couch",  /*rank=*/2, /*floor=*/0.0f },
                { /*canonical=*/"screen", /*rank=*/3, /*floor=*/0.0f },
            };
            pol.default_priority_rank    = 100;
            // CLIP zero-shot already gives well-calibrated scores against a
            // curated prompt list — no need for a global confidence floor.
            pol.default_confidence_floor = 0.0f;
            pol.emit_only_top_rank       = 0;     // keep everything in summary
            // Cross-class de-dup using max(IoU, IoMin) — collapses small
            // ghost coasted tracks fully contained inside a confirmed track.
            pol.post_merge_nms_iou       = 0.40f;
            PE::RequestConfigurePhysicalClassPolicy(pol);
            LOG_PHASE("Class policy configured (chair→couch, tv/laptop→screen, "
                      "max(IoU,IoMin)=0.40)", true);
        } catch (const std::exception& e) {
            LOG_PHASE(std::string("Class policy configuration failed — ") + e.what(), false);
        }

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

    // UISurfaceRegistry (singleton auto-constructed)
    // Touch the singleton so it's alive before any UI tools reference it.
    (void)GRIM::MMO::UISurfaceRegistry::instance();
    LOG_PHASE("MMO UISurfaceRegistry ready", true);

    // Register UI tools in ToolRegistry so the model knows these capabilities exist
    {
        auto& toolReg = GRIM::MMO::ToolRegistry::instance();

        GRIM::MMO::ToolDescriptor createSurface;
        createSurface.tool_id        = "ui.create_surface";
        createSurface.display_name   = "Create UI Surface";
        createSurface.provider_type  = GRIM::MMO::ToolProviderType::Builtin;
        createSurface.provider_name  = "builtin";
        createSurface.description    = "Create a new UI surface (panel, popup, modal, toast, tool window, or inspector)";
        createSurface.category       = "ui";
        createSurface.is_informational = false;
        createSurface.needs_confirmation = true;
        createSurface.parameters = {
            {"surface_id", "string", "Unique identifier for the surface", true},
            {"kind", "string", "Surface type: overlay_panel, popup, modal, toast, tool_window, inspector", true},
            {"title", "string", "Display title", true}
        };
        createSurface.capability_tags = {"ui", "create_surface", "display"};
        createSurface.keywords = {"panel", "popup", "modal", "toast", "window", "surface", "ui"};
        toolReg.registerTool(createSurface);

        GRIM::MMO::ToolDescriptor showSurface;
        showSurface.tool_id        = "ui.show_surface";
        showSurface.display_name   = "Show UI Surface";
        showSurface.provider_type  = GRIM::MMO::ToolProviderType::Builtin;
        showSurface.provider_name  = "builtin";
        showSurface.description    = "Show a previously created UI surface";
        showSurface.category       = "ui";
        showSurface.is_informational = false;
        showSurface.parameters = {
            {"surface_id", "string", "ID of the surface to show", true}
        };
        showSurface.capability_tags = {"ui", "show_surface"};
        toolReg.registerTool(showSurface);

        GRIM::MMO::ToolDescriptor hideSurface;
        hideSurface.tool_id        = "ui.hide_surface";
        hideSurface.display_name   = "Hide UI Surface";
        hideSurface.provider_type  = GRIM::MMO::ToolProviderType::Builtin;
        hideSurface.provider_name  = "builtin";
        hideSurface.description    = "Hide a visible UI surface without destroying it";
        hideSurface.category       = "ui";
        hideSurface.is_informational = false;
        hideSurface.parameters = {
            {"surface_id", "string", "ID of the surface to hide", true}
        };
        hideSurface.capability_tags = {"ui", "hide_surface"};
        toolReg.registerTool(hideSurface);

        GRIM::MMO::ToolDescriptor destroySurface;
        destroySurface.tool_id        = "ui.destroy_surface";
        destroySurface.display_name   = "Destroy UI Surface";
        destroySurface.provider_type  = GRIM::MMO::ToolProviderType::Builtin;
        destroySurface.provider_name  = "builtin";
        destroySurface.description    = "Permanently remove a UI surface";
        destroySurface.category       = "ui";
        destroySurface.is_informational = false;
        destroySurface.needs_confirmation = true;
        destroySurface.parameters = {
            {"surface_id", "string", "ID of the surface to destroy", true}
        };
        destroySurface.capability_tags = {"ui", "destroy_surface"};
        toolReg.registerTool(destroySurface);

        LOG_PHASE("MMO UI tools registered (4 surface tools)", true);
    }
}

// ================================================================
// Phase 5: Warmup and preload scheduling
// ================================================================
static void bootstrapWarmup() {
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
