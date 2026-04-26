//======================================================//
//  Phase1_Startup.cu
//  Configuration loading, model initialization, data loading
//======================================================//
//
//  IMPLEMENTATION
//  ==============
//  This file implements all startup operations:
//  1. Configuration loading from ai_config.json
//  2. Path validation and directory creation
//  3. Logging infrastructure initialization
//  4. Tokenizer loading
//  5. Training data loading with sliding windows
//  6. Model initialization and weight seeding
//  7. Optimizer and gradient controller setup
//
//  Author: Austin Wadkins
//  Date: December 2025
//  Version: 1.0.0
//======================================================//

#include "Phase1_Startup.hpp"
#include "../OptimizerCheckpoint.hpp"
#include "ConfigDump.hpp"
#include "Startup/SlidingWindow.hpp"
#include "Startup/ClassBalancedWeights.hpp"
#include "Startup/InitFacts.hpp"
#include "Startup/Capacity/CapacityStem.hpp"
// HyperParameters_GPU.hpp is the single entry point; it defines
// GRIM_HP_GPU_DEFINED_TRAINING_STRUCTS and transitively includes
// control/ai_config_paths.hpp in the correct order.
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"

#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/StreamController/StreamController_GPU.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/CudaAllocUtils.hpp"
#include "../../Shared/Telemetry/TelemetryLatticeConfig_FromHyperParams.hpp"
#include "../../Shared/Telemetry/TelemetryControlConfig_FromHyperParams.hpp"

using GRIM::CudaAlloc::cudaMallocOrThrow;

#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
// curand.h moved to Startup/Rng.cu
// Xavier.hpp removed - weights initialized via Tensor::xavier_uniform_() with Philox PRNG (Issue #107)
#endif

namespace fs = std::filesystem;

// Module logging aliases
using GRIM::Logging::ModuleId;
using GRIM::Logging::EmitModuleInfo;
using GRIM::Logging::EmitModuleError;

namespace GRIMText::Training {

//======================================================//
//  String Utilities
//======================================================//

//======================================================//
//  Internal Helper Implementations
//======================================================//

namespace Internal {

//======================================================//
//  loadConfiguration
//
//  Thin wrapper over GRIM::HyperParameters::loadStartupConfig().
//  All ai_config.json parsing, validation, derivation, and CLI
//  override handling lives in HyperParameters_GPU.hpp (single owner).
//  This wrapper only performs the two training-side side effects:
//    - GRIM::Tokenizer::configureTokenLayout()
//    - GRIM::Logging::InitLogRecorder() + ConfigureLayerLogging()
//  These deliberately stay outside HyperParameters to preserve the
//  Shared → Training layering rule.
//======================================================//

StartupConfig loadConfiguration(int argc, char** argv) {
    // Centralized load + validate + derive (single source of truth).
    StartupConfig config = GRIM::HyperParameters::loadStartupConfig(argc, argv);

    // Tokenizer global token-layout setup (Tokenizer subsystem side effect).
    GRIM::Tokenizer::configureTokenLayout(GRIM::Tokenizer::kAtomTypeCount);

    // Log recorder bootstrap (training log-subsystem side effect).
    if (config.hyperparameters.log_recorder.enabled) {
        GRIM::Logging::InitLogRecorder(config.paths.log_dir);

        const auto& layers = config.hyperparameters.log_recorder.layers;
        GRIM::Logging::ConfigureLayerLogging(
            config.hyperparameters.log_recorder.enabled,
            layers.embedding,
            layers.rms_norm,
            layers.attention,
            layers.feed_forward,
            layers.residual,
            layers.encoding,
            layers.serialization,
            layers.execution_block);
    } else {
        GRIM::Logging::ConfigureLayerLogging(
            false, false, false, false, false, false, false, false, false);
    }

    return config;
}

void validatePaths(const PathConfig& paths) {
    if (!fs::exists(paths.vocab_path)) {
        throw std::runtime_error("Vocabulary file does not exist: " + paths.vocab_path);
    }
    if (!fs::exists(paths.data_path)) {
        throw std::runtime_error("Training data file does not exist: " + paths.data_path);
    }
    if (paths.output_model_path.empty()) {
        throw std::runtime_error("Output model path not configured");
    }
    if (paths.checkpoint_dir.empty()) {
        throw std::runtime_error("Checkpoint directory not configured");
    }
    if (paths.log_dir.empty()) {
        throw std::runtime_error("Log directory not configured");
    }
    
    // Create directories
    auto model_parent = fs::path(paths.output_model_path).parent_path();
    if (!model_parent.empty())
        fs::create_directories(model_parent);
    fs::create_directories(paths.checkpoint_dir);
    fs::create_directories(paths.log_dir);
}

// initializeLogging moved to Startup/Logging.cu

GRIM::Tokenizer::UniByte initializeTokenizer(
    const std::string& vocab_path,
    const GRIM::Config::TokenizerConfig& tok_config,
    const GRIM::Config::TrainingHyperparameters& hyperparameters,
    TrainingLogger& logger) {
    
    logger.log("Loading tokenizer configuration...");
    
    GRIM::Tokenizer::UniByteConfig cfg;
    // Rule 20: No fallback - vocab_size must be in config. Actual vocab comes from .grmt file later.
    if (tok_config.vocab_size <= 0) {
        throw std::runtime_error("FATAL: tokenizer.vocab_size not configured in ai_config.json");
    }
    // Use all tokenizer config values from hyperparameters (single source of truth)
    cfg.target_vocab_size = tok_config.vocab_size;
    cfg.character_coverage = GRIM::HyperParameters::TOKENIZER_CHARACTER_COVERAGE;
    cfg.enable_scratch_block_reasoning = hyperparameters.tokenizer_enable_scratch_block_reasoning;
    cfg.detect_numbers = hyperparameters.tokenizer_detect_numbers;
    cfg.enable_byte_fallback = tok_config.enable_byte_fallback;
    cfg.prefer_gpu = GRIM::HyperParameters::TOKENIZER_PREFER_GPU;
    
    GRIM::Tokenizer::UniByte tokenizer(cfg);
    if (!tokenizer.load(vocab_path)) {
        throw std::runtime_error("Failed to load vocabulary: " + vocab_path);
    }

    // Vocab size is reported by DataStatsDump (tokenizer_vocab_size).
    // The GRMT is the ground truth — it was written with totalVocabSize() at encode time.
    // Phase1 must match that exactly.  Post-load capping is wrong: it reduces totalVocabSize()
    // below what the GRMT recorded, guaranteeing a mismatch.  If you want fewer tokens,
    // lower vocab_size in ai_config.json and let the DataLoader retrain the tokenizer.
    return tokenizer;
}

SequenceData loadTrainingData(
    const std::string& data_path,
    int max_seq_len,
    int min_seq_valid_tokens,
    int sliding_window_stride,
    bool add_bos_token,
    bool add_eos_token,
    const GRIM::Tokenizer::UniByte& tokenizer,
    TrainingLogger& logger) {
    
    SequenceData data;

    GRMTDataLoader loader;
    if (!loader.load(data_path)) {
        throw std::runtime_error("Failed to load training data");
    }

    // Store vocab size from training data for later validation
    data.vocab_size = loader.vocabSize();
    
    auto all_sequences = loader.getSequences();

    const int bos_id = tokenizer.bosId();
    const int eos_id = tokenizer.eosId();

    // Split train/val
    size_t val_size = all_sequences.size() / 10;
    data.train_seqs.assign(all_sequences.begin() + val_size, all_sequences.end());
    data.val_seqs.assign(all_sequences.begin(), all_sequences.begin() + val_size);

    // Apply sliding windows. Implementation lives in
    // Phases/Startup/SlidingWindow.cu — see that file for the full
    // padding-ownership / final-target / overlap-masking contracts.
    // applySlidingWindows owns the entire post-load shaping pipeline:
    // BOS/EOS injection, windowing, overlong filtering, and short-sequence
    // filtering. Callers don't bracket or filter sequences themselves.
    applySlidingWindows(data.train_seqs, "train",
                        max_seq_len, sliding_window_stride, min_seq_valid_tokens,
                        add_bos_token, add_eos_token, bos_id, eos_id, logger);
    applySlidingWindows(data.val_seqs, "val",
                        max_seq_len, sliding_window_stride, min_seq_valid_tokens,
                        add_bos_token, add_eos_token, bos_id, eos_id, logger);

    // Build views and catalogs
    data.train_views.reserve(data.train_seqs.size());
    for (uint32_t i = 0; i < data.train_seqs.size(); ++i) {
        data.train_views.push_back(&data.train_seqs[i]);
        const uint32_t len = static_cast<uint32_t>(data.train_seqs[i].token_ids.size());
        data.train_catalog.add(len, len, 0, 0, 0);
    }
    
    data.val_views.reserve(data.val_seqs.size());
    for (uint32_t i = 0; i < data.val_seqs.size(); ++i) {
        data.val_views.push_back(&data.val_seqs[i]);
        const uint32_t len = static_cast<uint32_t>(data.val_seqs[i].token_ids.size());
        data.val_catalog.add(len, len, 0, 0, 0);
    }

    // Train/val sequence counts are reported by DataStatsDump.
    return data;
}

std::unique_ptr<GRIM::LanguageModel> initializeModel(
    const StartupConfig& config,
    const RunCapacity& run_capacity,
    uint32_t vocab_size,
    uint64_t xavier_seed,
    TrainingLogger& logger,
    std::string& loaded_checkpoint_path) {
    loaded_checkpoint_path.clear();
    
    logger.log("Initializing model with xavier_seed=" + std::to_string(xavier_seed) + "...");
    
    const auto& arch = config.hyperparameters.architecture;
    const auto& hp = config.hyperparameters;
    
    GRIM::HyperParameters::LanguageModelConfig model_config;
    // Slice-copy the full LanguageModelConfig from hyperparameters.architecture
    // (Phase B: TrainingHyperparameters::architecture is now LanguageModelConfig,
    // so this single assignment carries ALL model architecture + feature fields:
    // d_model/heads/layers, ScratchBlock, ExecutionBlock, LM-head centering,
    // LayerScale, QK-norm, hardcoded-hidden, MTP, etc.). Per-field copies
    // deleted — Rule 20 (no dual-source state).
    model_config = arch;

    // Phase1-specific overrides (not in hyperparameters.architecture)
    model_config.max_seq_len = config.max_seq_len;  // StartupConfig's derived value
    model_config.vocab_size = vocab_size;
    model_config.vocab_path = config.paths.vocab_path;
    model_config.infer_vocab_from_file = true;
    model_config.causal_mask = true;
    model_config.use_pre_norm = true;
    model_config.fuse_qkv = true;
    model_config.use_bias = true;

    // Compute and validate derived hyperparameter values
    model_config.computeDerivedValues();

    // Hyperparameter narration is the exclusive responsibility of
    // dumpAllHyperparameters() (ConfigDump.cu), which has already run in
    // executePhase1(). Local logging here is limited to validations
    // (throw on misconfiguration) and abnormal-mode warnings.

    if (model_config.structured_ce_enabled && model_config.structured_ce_weight <= 0.0f) {
        throw std::runtime_error("structured_ce_enabled=true but structured_ce_weight=" +
                                 std::to_string(model_config.structured_ce_weight) + " (must be > 0)");
    }

    if (model_config.mtp_enabled && (model_config.mtp_k <= 0 || model_config.mtp_alpha <= 0.0f)) {
        throw std::runtime_error("multi_token_prediction: when enabled, k and alpha must be > 0 (k=" +
            std::to_string(model_config.mtp_k) + " alpha=" + std::to_string(model_config.mtp_alpha) + ")");
    }

    if (model_config.hardcoded_hidden_pattern != GRIM::HyperParameters::LanguageModelConfig::HardcodedPattern::DISABLED) {
        logger.log("⚠️  HARDCODED HIDDEN STATES DIAGNOSTIC ENABLED: pattern=" + std::to_string(static_cast<int>(model_config.hardcoded_hidden_pattern)) +
                  ", log_every_n=" + std::to_string(model_config.hardcoded_log_every_n_batches));
        logger.log("⚠️  Encoder output will be REPLACED with synthetic patterns - this is a DIAGNOSTIC MODE ONLY!");
    }

    // Cache sizing is authored by the post-policy capacity stem (RunCapacity).
    // Rule 20: no local re-derivation here (prevents drift vs scheduler/payload).
    model_config.max_cached_batch = static_cast<int>(run_capacity.batch_rows);
    model_config.max_cached_seq_len = run_capacity.seq_cap;
    model_config.max_tokens_per_batch = static_cast<int>(run_capacity.max_tokens_per_batch);

    // STEP 0: Initialize CUDA device context FIRST (REQUIRED before any stream/allocation operations)
    // This ensures CUDA driver is loaded and device context exists before StreamController creates streams.
    // MUST be done exactly once at the start of training, not in StreamController or initGPU.
    {
        logger.log("Initializing CUDA device context...");
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("FATAL: cudaSetDevice(0) failed: ") + cudaGetErrorString(err));
        }
        
        // Force context creation by triggering a no-op CUDA operation
        err = cudaFree(0);
        if (err != cudaSuccess && err != cudaErrorInvalidValue) {
            throw std::runtime_error(std::string("FATAL: CUDA context creation failed: ") + cudaGetErrorString(err));
        }
        
        // Verify device is accessible
        int device;
        err = cudaGetDevice(&device);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("FATAL: cudaGetDevice() failed: ") + cudaGetErrorString(err));
        }
        
        cudaDeviceProp props;
        err = cudaGetDeviceProperties(&props, device);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("FATAL: cudaGetDeviceProperties() failed: ") + cudaGetErrorString(err));
        }
        
        logger.log("✓ CUDA device initialized: " + std::string(props.name));
    }
    
    // StreamController is owned by TrainingState 
    // Initialize stream_ctrl after CUDA context exists
    
    auto model = std::make_unique<GRIM::LanguageModel>(model_config);
    
    // ═══════════════════════════════════════════════════════════════
    // CORRUPTION DIAGNOSTIC: Check batch_prep vector integrity after each step
    // STEP 1: Initialize just the stream controller part of TrainingState
    // This creates CUDA streams (CUDA context must exist from STEP 0)
    {
        GRIM::StreamControllerConfig stream_config;
        stream_config.verbose = true;
        
        if (!model->getTrainingState().stream_ctrl.initialize(stream_config)) {
            throw std::runtime_error("FATAL: Failed to initialize StreamController");
        }
        logger.log("✓ StreamController initialized");
    }
    
    // STEP 2: Initialize cuBLAS handle (needed by encoder layers in initGPU)
    logger.log("Initializing cuBLAS handle...");
    model->initCuBLASHandle();
    logger.log("✓ cuBLAS handle initialized with Tensor Core acceleration");
    
    // STEP 2.5: Initialize RoPE BEFORE encoder construction (CRITICAL!)
    logger.log("Initializing RoPE (required before encoder construction)...");
    model->initPBM();
    logger.log("✓ RoPE initialized");
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 2.75: Store Autograd Seed (Session 7: TrainingTensors deleted)
    // Weight init seed is stored on TrainingState.weight_init_seed.
    // Pattern B layers use it for layer-specific weight seeding.
    // layer-specific weight seeding. All weight allocation is now done by
    // Pattern B layers (EmbeddingLayer, EncodingLayer, LMHeadLayer, ScratchBlockLayer).
    // ═══════════════════════════════════════════════════════════════
    {
        logger.log("Storing autograd weight init seed...");
        
        // CRITICAL: Use xavier_seed (derived init_seed from RNG context), not config.hyperparameters.seed
        // The xavier_seed is the computed hierarchical seed from initializeRNG()
        model->getTrainingState().initializeAutogradSeed(xavier_seed);
        logger.log("✓ Autograd seed stored (weight_init_seed=" + std::to_string(xavier_seed) + ")");
    }
    
    // STEP 3: Initialize GPU encoder (uses cuBLAS handle from step 2, seed from step 2.75)
    // NOTE: initGPU() creates Pattern B layers which self-allocate their weights
    logger.log("Initializing GPU encoder...");
    model->initGPU();
    logger.log("✓ GPU encoder fully initialized");
    
    // STEP 4: Finish TrainingState initialization (grad buffers, activation caches)
    // NOTE: Layers already set up in step 3, this just does the rest
    logger.log("Initializing TrainingState (grad buffers, activation caches)...");
    model->initTrainingState();
    logger.log("✓ TrainingState fully initialized");

    // Build parameter groups for optimizer and grad-norm (required even when logit_update_trace is off).
    // Otherwise Phase2 allocateGradNormScratch gets max_groups=0 and fails.
    // Init-time structural invariants (tie_embeddings pointer/grad equality,
    // optimizer group accounting) are verified by verifyAndDumpInitFacts() in
    // executePhase1, after this function returns. See Startup/InitFacts.cu —
    // contract is fail-loud throws on violation, dual-channel emission to
    // telemetry stream slots 48-54 and init_facts_<session>.csv on success.
    model->buildParameterGroups();

    // UpdateProbe subsystem deleted (Rule 26): probe was instrumentation that
    // was never wired to actually populate the weights/grads sample buffers,
    // so the consumer in Phase2 never fired. logit_update_trace_enabled still
    // controls the PostLoss cached_logits trace below.

    // Configure scratch blocks (scratch_blocks_enabled / scratch_num_blocks are
    // reported by ConfigDump; no local echo).
    if (model->isScratchPoolInitialized()) {
        model->configureScratchPool(config.hyperparameters.scratch_blocks_enabled);
    }

    // Build LossOptions inline from hyperparameters (single source of truth lives in
    // ai_config_paths.hpp::TrainingHyperparameters; LossOptions is the consumer-side
    // type defined by Shared/Loss/LossContext/LossContext.hpp).
    GRIM::LossContext::LossOptions loss_opts{};
    {
        const auto& hp = config.hyperparameters;
        loss_opts.label_smoothing_enabled    = hp.loss_label_smoothing_enabled;
        loss_opts.label_smoothing_epsilon    = hp.loss_label_smoothing_epsilon;
        loss_opts.focal_enabled              = hp.loss_focal_enabled;
        loss_opts.focal_gamma                = hp.loss_focal_gamma;
        loss_opts.focal_alpha                = hp.loss_focal_alpha;
        loss_opts.preference_enabled         = hp.loss_preference_enabled;
        loss_opts.preference_beta            = hp.loss_preference_beta;
        loss_opts.distillation_enabled       = hp.loss_distillation_enabled;
        loss_opts.distillation_temperature   = hp.loss_distillation_temperature;
        loss_opts.distillation_lambda        = hp.loss_distillation_lambda;
        loss_opts.masking_enabled            = hp.loss_masking_enabled;
        loss_opts.masking_tag                = hp.loss_masking_tag;
        loss_opts.entropy_reg_enabled        = hp.loss_entropy_reg_enabled;
        loss_opts.entropy_reg_lambda         = hp.loss_entropy_reg_lambda;
        loss_opts.class_balanced_enabled     = hp.loss_class_balanced_enabled;
        loss_opts.class_balanced_beta        = hp.loss_class_balanced_beta;
    }
    model->setLossOptions(loss_opts);

    // Loss options (label_smoothing / focal / distillation / preference /
    // entropy_reg / class_balanced + their scalars) are reported in the
    // "Loss options" section of ConfigDump. No local echo.

#ifdef USE_CUDA
    // Verify encoder weights are initialized (Pattern B layers handle Xavier init via Tensor::xavier_uniform_)
    {
        auto* gpu_encoder = &model->getGpuEncoder();
        const auto& cfg = model->getConfig();
        for (int layer = 0; layer < cfg.num_layers; ++layer) {
            auto* enc = gpu_encoder->getLayer(layer);
            if (!enc) {
                throw std::runtime_error("Encoder layer " + std::to_string(layer) + " is NULL after initGPU");
            }
        }
        logger.log("✓ All " + std::to_string(cfg.num_layers) + " encoder layers verified");
    }
#endif
    
    // Try to load checkpoint - scan newest-to-oldest and stop at first valid file.
    std::vector<std::pair<int, std::string>> checkpoint_candidates;
    if (fs::exists(config.paths.checkpoint_dir) && fs::is_directory(config.paths.checkpoint_dir)) {
        for (const auto& entry : fs::directory_iterator(config.paths.checkpoint_dir)) {
            const auto& p = entry.path();
            if (p.extension() == ".bin" && p.stem().string().rfind("checkpoint_epoch_", 0) == 0) {
                // Extract epoch number from "checkpoint_epoch_N"
                std::string stem = p.stem().string();
                std::string epoch_str = stem.substr(std::string("checkpoint_epoch_").size());
                try {
                    int epoch = std::stoi(epoch_str);
                    checkpoint_candidates.emplace_back(epoch, p.string());
                } catch (const std::exception& e) {
                    logger.log("[WARNING] Skipping malformed checkpoint filename: " + stem + " (" + e.what() + ")");
                }
            }
        }
    }

    std::sort(checkpoint_candidates.begin(), checkpoint_candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    bool loaded_checkpoint = false;
    for (const auto& [epoch, checkpoint_path] : checkpoint_candidates) {
        logger.log("Found checkpoint candidate: " + checkpoint_path + " (epoch " + std::to_string(epoch) + ")");
        if (model->load(checkpoint_path)) {
            logger.log("✓ Loaded weights from checkpoint: " + checkpoint_path);
            loaded_checkpoint = true;
            loaded_checkpoint_path = checkpoint_path;
            break;
        }
        logger.log("⚠ Failed to load checkpoint candidate, trying older checkpoint");
    }

    if (!loaded_checkpoint) {
        if (!checkpoint_candidates.empty()) {
            logger.log("⚠ No loadable checkpoint found, starting fresh");
        } else {
            logger.log("No checkpoint found, starting fresh");
        }
    }
    
    logger.log("✓ Model initialized");
    return model;
}

void initializeOptimizer(TrainingContext& ctx) {
    auto& logger = *ctx.logging.logger;
    auto& opt = ctx.optimizer;
    auto& model = *ctx.model;
    const auto& hp = ctx.config.hyperparameters;

    logger.log("Initializing optimizer state...");

    // Optimizer state (AdamW hyperparameters are defined as constants in AdamW_Kernal_GPU.cu)
    opt.optimizer_state.step = 0;

    // Soft restart controller — loss-detection thresholds (loss_increase_threshold,
    // max_step_window) now live in GRIM::Loss::LossSignalConfig (validation_delta_threshold)
    // and are wired in Phase2 when the LossSignalBus is constructed. SoftRestartConfig
    // owns ONLY cooldown_steps now.
    GRIM::SoftRestart::SoftRestartConfig sr_cfg;
    sr_cfg.cooldown_steps = hp.soft_restart_cooldown_steps;
    opt.soft_restart_controller = GRIM::SoftRestart::SoftRestartController(sr_cfg);

    // Gradient accumulation now tracked directly via hyperparameters.
    // opt.current_micro_step is reset at start of each accumulation
    // window. batch_size and gradient_accumulation_steps are reported by
    // ConfigDump; no local echo.
    opt.current_micro_step = 0;

    // Verify ALL encoder layers initialized (prevents lazy init during forward pass)
    auto* gpu_encoder = &model.getGpuEncoder();
    for (int layer = 0; layer < model.getConfig().num_layers; ++layer) {
        if (!gpu_encoder->getLayer(layer)) {
            throw std::runtime_error("Encoder layer " + std::to_string(layer) + " not initialized - "
                                     "ensure model.initGPU() completes all layers before training");
        }
    }

    logger.log("✓ Optimizer state initialized");

    // Restore optimizer sidecar paired to the EXACT checkpoint that loaded
    // weights. Do NOT rescan the checkpoint dir here — that's how a different
    // epoch's .opt could shadow the one matching the loaded .bin.
    // loadOptimizerState() emits its own success line via EmitModuleInfo
    // (step/global_step/best_val_loss/epoch/micro_step/groups), so the
    // success branch deliberately stays silent here (Rule 20: single source
    // of truth for the restore log). Failures are tolerated — we keep the
    // freshly-initialized optimizer state and let training resume.
    if (ctx.loaded_checkpoint_path.empty()) return;

    const std::string opt_path = optimizerSidecarPath(ctx.loaded_checkpoint_path);
    if (!fs::exists(opt_path)) {
        logger.log("⚠ No optimizer sidecar for loaded checkpoint: " + ctx.loaded_checkpoint_path);
        logger.log("  Continuing with fresh optimizer state (moments reset, LR from step 0)");
        return;
    }

    logger.log("Found optimizer sidecar for loaded checkpoint: " + opt_path);
    try {
        loadOptimizerState(ctx, opt_path);
    } catch (const std::exception& e) {
        logger.log(std::string("⚠ Optimizer sidecar load failed: ") + e.what());
        logger.log("  Continuing with fresh optimizer state for checkpoint: " + ctx.loaded_checkpoint_path);
    }
}

// initializeRNG moved to Startup/Rng.cu

} // namespace Internal

//======================================================//
//  Phase1 Main Entry Point
//======================================================//

std::unique_ptr<TrainingContext> executePhase1(int argc, char** argv) {
    auto ctx = std::make_unique<TrainingContext>();
    
    EmitModuleInfo(ModuleId::Training, "========================================", 0);
    EmitModuleInfo(ModuleId::Training, "  Phase 1: Startup & Initialization", 0);
    EmitModuleInfo(ModuleId::Training, "  GRIM-text GPU Training v3.0.0", 0);
    EmitModuleInfo(ModuleId::Training, "========================================", 0);

    // 1. Load configuration
    EmitModuleInfo(ModuleId::Training, "[Phase1] Loading configuration...", 0);
    ctx->config = Internal::loadConfiguration(argc, argv);
    EmitModuleInfo(ModuleId::Training, 
        std::string("[Phase1] ✓ Configuration loaded from: ") + ctx->config.paths.config_path.string(), 0);

    // 2. Initialize logging as early as possible (before cache prep)
    if (!ctx->config.paths.log_dir.empty()) {
        fs::create_directories(ctx->config.paths.log_dir);
    }
    if (!ctx->config.paths.status_path.empty()) {
        fs::path status_parent = fs::path(ctx->config.paths.status_path).parent_path();
        if (!status_parent.empty()) {
            fs::create_directories(status_parent);
        }
    }
    EmitModuleInfo(ModuleId::Training, "[Phase1] Initializing logging...", 0);
        ctx->logging = Internal::initializeLogging(ctx->config.paths);

    // Initialize unified BatchLogTape system (sinks + global tape pointer)
    Internal::setupBatchLogTape(ctx->logging, ctx->config);

    // 2b. Tokenizer / training_data preparation is owned by the
    //      train_tokenizer subprocess that ran BEFORE Phase 1 (see
    //      Subprocess/tokenizer_subprocess.hpp). By the time we get here,
    //      the GRMT and vocab artifacts are guaranteed to exist on disk;
    //      validatePaths() below is the single place that fails loud if
    //      they don't (Rule 20: no in-Phase fallback path).

    // 3. Validate paths
    EmitModuleInfo(ModuleId::Training, "[Phase1] Validating paths...", 0);
    Internal::validatePaths(ctx->config.paths);
    EmitModuleInfo(ModuleId::Training, "[Phase1] ✓ All paths validated", 0);

    // Data paths + vocab + sequence counts are logged together with the
    // hyperparameter dump (post-harmonize) via DataStatsSnapshot below,
    // so derived ratios from HyperParameters_GPU.hpp are reflected and
    // the visual block is unified.

    // 4. Initialize tokenizer — use the tokenizer_config already captured on
    // StartupConfig during loadConfiguration(); no second snapshot read.
    const GRIM::Config::TokenizerConfig& tok_cfg = ctx->config.tokenizer_config;
    ctx->tokenizer = Internal::initializeTokenizer(ctx->config.paths.vocab_path, tok_cfg, ctx->config.hyperparameters, *ctx->logging.logger);
    
    // 5. Load training data (parses .grmt artifact written by the
    //    tokenizer subprocess; subprocess only returned vocab_size, not
    //    the sequence buffer — that has to be loaded into this process).
    ctx->data = Internal::loadTrainingData(
        ctx->config.paths.data_path,
        ctx->config.max_seq_len,
        ctx->config.hyperparameters.min_seq_valid_tokens,
        ctx->config.sliding_window_stride,
        tok_cfg.add_bos,
        tok_cfg.add_eos,
        ctx->tokenizer,
        *ctx->logging.logger);
    
    // Use vocab size extracted during data loading (no second load needed)
    ctx->config.actual_vocab_size = ctx->data.vocab_size;
    if (ctx->config.actual_vocab_size == 0) {
        throw std::runtime_error("FATAL: training data missing vocab_size; regenerate GRMT with tokenizer.totalVocabSize()");
    }
    if (ctx->config.actual_vocab_size < static_cast<uint32_t>(GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET)) {
        throw std::runtime_error("FATAL: training data vocab_size must include special+byte+atom ranges (>= " + 
            std::to_string(GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET) + ")");
    }
    
    // GRMT/vocab.bin header equality is enforced inside the tokenizer
    // subprocess: PrepareTrainingDataFromCache (DataLoader.cu ~480-538)
    // re-reads both headers and forces a full rebuild on mismatch, then writes
    // a fresh .grmt using tokenizer.totalVocabSize() (line ~942). By the time
    // we land here the two are guaranteed equal at the byte level — a second
    // runtime equality check would only fire on loader bugs, not on stale
    // artifacts. Keep this local for the DataStatsSnapshot below.
    const uint32_t tokenizer_vocab_size = ctx->tokenizer.totalVocabSize();

    // 6. Harmonize hyperparameters
    GRIM::HyperParameters::DerivationContext hp_ctx;
    hp_ctx.train_sequence_count = static_cast<int>(ctx->data.train_seqs.size());
    hp_ctx.validation_interval = ctx->config.hyperparameters.validation_interval;
    
    auto hp_logger = [&](const std::string& msg) { ctx->logging.logger->log(msg); };

    // Three primitives, in order:
    //   (A) validate inputs (fail loud)
    //   (B) compute derived schedule from (seq_count, batch_size, epochs) ratios
    //   (C) apply cross-field policy / clamps (mutates hp, logs adjustments)
    GRIM::HyperParameters::validateTrainingHyperparameters(ctx->config.hyperparameters);
    ctx->derived_schedule = GRIM::HyperParameters::computeDerivedSchedule(
        ctx->config.hyperparameters, hp_ctx);
    GRIM::HyperParameters::applyTrainingHyperparameterPolicy(
        ctx->config.hyperparameters, ctx->derived_schedule, hp_ctx, hp_logger);

    // 7. Capacity stem (single author after HP policy).
    ctx->run_capacity = deriveRunCapacityOrThrow(ctx->config);

    // Single-source hyperparameter dump (post-policy, includes derived
    // schedule values computed in HyperParameters_GPU.hpp). The
    // DataStatsSnapshot folds vocab + sequence-count info into the same
    // visual block.
    GRIMText::Training::DataStatsSnapshot data_stats;
    data_stats.data_path            = ctx->config.paths.data_path;
    data_stats.vocab_path           = ctx->config.paths.vocab_path;
    data_stats.tokenizer_vocab_size = tokenizer_vocab_size;
    data_stats.actual_vocab_size    = ctx->config.actual_vocab_size;
    data_stats.train_sequence_count = ctx->data.train_seqs.size();
    data_stats.val_sequence_count   = ctx->data.val_seqs.size();

    dumpAllHyperparameters(
        ctx->config.hyperparameters,
        &ctx->derived_schedule,
        &data_stats,
        [](const std::string& msg) { EmitModuleInfo(ModuleId::Training, msg, 0); });
    
    // 8. Initialize production-grade RNG system (BEFORE model init for Xavier seeding)
    ctx->rng = Internal::initializeRNG(ctx->config, *ctx->logging.logger);
    
    // 9. Initialize model (uses RNG init_seed for Xavier)
    //    loaded_checkpoint_path receives the path that ACTUALLY loaded
    //    so the optimizer sidecar can be matched exactly (no independent scan).
    try {
        ctx->model = Internal::initializeModel(
            ctx->config,
            ctx->run_capacity,
            ctx->config.actual_vocab_size,
            ctx->rng.init_seed,
            *ctx->logging.logger,
            ctx->loaded_checkpoint_path);
    } catch (const std::exception& e) {
        EmitModuleError(ModuleId::Training, std::string("FATAL: Model initialization failed: ") + e.what(), 0);
        throw;
    }

    // 9a. Verify init-time structural invariants and dual-emit them to
    // telemetry stream slots (48-54) and init_facts_<session>.csv.
    // Implementation lives in Phases/Startup/InitFacts.cu — throws on
    // tie_embeddings contract violations, otherwise silent in the human log.
    verifyAndDumpInitFacts(*ctx);

    // Handle save test mode
    if (ctx->config.save_test_mode) {
        ctx->logging.logger->log("========================================");
        ctx->logging.logger->log("  SAVE TEST MODE");
        ctx->logging.logger->log("========================================");
        std::string test_save_path = ctx->config.paths.checkpoint_dir + "/save_test.bin";
        ctx->logging.logger->log("Testing model->save() to: " + test_save_path);
        bool save_ok = ctx->model->save(test_save_path);
        if (save_ok) {
            EmitModuleInfo(ModuleId::Checkpoint, "✓ Save test PASSED", 0);
            if (fs::exists(test_save_path)) {
                auto file_size = fs::file_size(test_save_path);
                EmitModuleInfo(ModuleId::Checkpoint, 
                    std::string("  File size: ") + std::to_string(file_size) + " bytes", 0);
            }
        } else {
            EmitModuleError(ModuleId::Checkpoint, "✗ Save test FAILED", 0);
        }
        std::exit(save_ok ? 0 : 1);
    }
    
    // 10. Initialize optimizer (also restores optimizer sidecar paired to the
    // exact checkpoint that loaded weights — see Internal::initializeOptimizer).
    Internal::initializeOptimizer(*ctx);

    // 10a. Compute class-balanced loss weights if enabled.
    // Implementation lives in Phases/Startup/ClassBalancedWeights.cu —
    // see that file for the frequency math and logging contract.
    if (ctx->config.hyperparameters.loss_class_balanced_enabled) {
        computeAndUploadClassBalancedWeights(
            ctx->data.train_seqs,
            ctx->config.actual_vocab_size,
            ctx->config.hyperparameters.loss_class_balanced_beta,
            ctx->model->getTrainingState(),
            *ctx->logging.logger);
    }
    
    // 11. Initialize telemetry lattice
    EmitModuleInfo(ModuleId::Training, "[Phase1] Initializing telemetry lattice...", 0);
    ctx->telemetry.lattice = std::make_unique<GRIM::Telemetry::TelemetryLattice>(
        GRIM::Telemetry::makeLatticeConfigFromHyperparameters(
            ctx->config.hyperparameters,
            ctx->model->getTrainingState().stream_ctrl.getPrimaryStream()));
    const auto& lattice_cfg = ctx->telemetry.lattice->config();
    ctx->logging.logger->log("✓ Telemetry lattice: " + std::to_string(lattice_cfg.num_levels) + " levels, " + std::to_string(lattice_cfg.num_streams) + " streams, GPU-resident");

    // 11a. Initialize telemetry CSV logger (per-step measured data export)
    {
        const std::string csv_path = ctx->config.paths.log_dir + "/telemetry_" + ctx->logging.session_id + ".csv";
        ctx->telemetry.csv_logger = std::make_unique<GRIM::Telemetry::TelemetryCsvLogger>(
            csv_path, *ctx->telemetry.lattice);
        ctx->logging.logger->log("✓ Telemetry CSV logger: " + csv_path);
    }

    // 11b. Initialize telemetry control (GPU-native kernel-based control)
    EmitModuleInfo(ModuleId::Training, "[Phase1] Initializing telemetry control...", 0);
    // Rule 20: telemetry control must use the capacity stem (single author),
    // and verify the model mirrors match it (no independent derivations).
    const uint32_t token_budget = ctx->run_capacity.max_tokens_per_batch;
    const uint32_t seq_cap = ctx->run_capacity.seq_cap;
    if (token_budget == 0 || seq_cap == 0) {
        throw std::runtime_error("FATAL: RunCapacity is invalid (token_budget=" +
                                 std::to_string(token_budget) + " seq_cap=" + std::to_string(seq_cap) + ")");
    }
    if (static_cast<uint32_t>(ctx->model->getConfig().max_tokens_per_batch) != token_budget) {
        throw std::runtime_error("FATAL: model max_tokens_per_batch does not match RunCapacity (model=" +
                                 std::to_string(ctx->model->getConfig().max_tokens_per_batch) +
                                 " stem=" + std::to_string(token_budget) + ")");
    }
    ctx->telemetry.control_config = GRIM::Telemetry::makeControlConfigFromHyperparameters(
        ctx->config.hyperparameters,
        token_budget,
        seq_cap);

    // Telemetry control field values are reported by ConfigDump
    // (telemetry_* section). GPU free/total memory is reported by ConfigDump
    // (memory.gpu.* rows). No local echoes here.

    // Write initial status
    ctx->logging.status_writer->writeStatus(
        GRIMText::Control::TrainingState_Training,
        0, ctx->config.hyperparameters.epochs, 0, 0,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        "Phase 1 complete - ready for training"
    );
    
    // Initialize timing
    ctx->start_time = std::chrono::steady_clock::now();
    

    return ctx;  // Heap-allocated, no move
}

} // namespace GRIMText::Training
