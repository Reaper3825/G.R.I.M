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

// MUST be first - defines GRIM_CONFIG_AI_CONFIG_PATHS_HPP_INCLUDED
#include "../../../../../control/ai_config_paths.hpp"

#include "../../Layers/Encoding/Encoding_GPU.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/StreamController/StreamController_GPU.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Layers/GRIMTS/GRIM-TS.hpp"

#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <curand.h>
#include "../../Shared/Activations/Xavier/Xavier.hpp"
using GRIM::launchXavierInit;
#endif

using json = nlohmann::json;

// Module logging aliases
using GRIM::Logging::ModuleId;
using GRIM::Logging::EmitModuleInfo;
using GRIM::Logging::EmitModuleWarning;
using GRIM::Logging::EmitModuleError;

namespace GRIMText::Training {

//======================================================//
//  String Utilities
//======================================================//

namespace {

std::string trimCopy(const std::string& value) {
    const auto start = value.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return {};
    const auto end = value.find_last_not_of(" \t\n\r");
    return value.substr(start, end - start + 1);
}

std::string toLowerCopy(const std::string& value) {
    std::string result = value;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

GRIM::Logging::ModuleLogLevel parseModuleLogLevelString(const std::string& text) {
    // Rule 20: No fallback parameter - function throws on invalid input, parameter was unused
    const std::string normalized = toLowerCopy(trimCopy(text));
    if (normalized == "info" || normalized == "verbose" || normalized == "all") {
        return GRIM::Logging::ModuleLogLevel::Info;
    }
    if (normalized == "warn" || normalized == "warning") {
        return GRIM::Logging::ModuleLogLevel::Warning;
    }
    if (normalized == "err" || normalized == "error" || normalized == "fatal") {
        return GRIM::Logging::ModuleLogLevel::Error;
    }
    std::ostringstream oss;
    oss << "Phase1_Startup: unknown module log level '" << text << "'";
    throw std::runtime_error(oss.str());
}

void registerDefaultLoggingProfiles() {
    using namespace GRIM::Logging;
    static bool registered = false;
    if (registered) return;
    registered = true;

    RegisterModuleLogProfile("forward_pass", {
        MakeOverride(ModuleId::ForwardPass, ModuleLogLevel::Info),
        MakeOverride(ModuleId::Activations, ModuleLogLevel::Info),
        MakeOverride(ModuleId::GuessCache, ModuleLogLevel::Info),
        MakeOverride(ModuleId::DataLoader, ModuleLogLevel::Info),
    });

    RegisterModuleLogProfile("backward_pass", {
        MakeOverride(ModuleId::BackwardPass, ModuleLogLevel::Info),
        MakeOverride(ModuleId::Optimizer, ModuleLogLevel::Info),
    });

    RegisterModuleLogProfile("optimizer", {
        MakeOverride(ModuleId::Optimizer, ModuleLogLevel::Info),
        MakeOverride(ModuleId::Scheduler, ModuleLogLevel::Info),
    });

    RegisterModuleLogProfile("validation", {
        MakeOverride(ModuleId::Validation, ModuleLogLevel::Info),
        MakeOverride(ModuleId::Checkpoint, ModuleLogLevel::Info),
    });
}

} // anonymous namespace

//======================================================//
//  RNGContext Implementation
//======================================================//

RNGContext::~RNGContext() {
#ifdef USE_CUDA
    if (cuda_rng_initialized && cuda_rng_generator) {
        curandDestroyGenerator(static_cast<curandGenerator_t>(cuda_rng_generator));
        cuda_rng_generator = nullptr;
    }
#endif
}

RNGContext::RNGContext(RNGContext&& other) noexcept
    : base_seed(other.base_seed)
    , data_seed(other.data_seed)
    , init_seed(other.init_seed)
    , cuda_seed(other.cuda_seed)
    , data_rng(std::move(other.data_rng))
    , cuda_rng_generator(other.cuda_rng_generator)
    , deterministic(other.deterministic)
    , cuda_rng_initialized(other.cuda_rng_initialized)
{
    // Transfer ownership - source no longer owns CUDA generator
    other.cuda_rng_generator = nullptr;
    other.cuda_rng_initialized = false;
}

RNGContext& RNGContext::operator=(RNGContext&& other) noexcept {
    if (this != &other) {
        // Clean up our current resources
#ifdef USE_CUDA
        if (cuda_rng_initialized && cuda_rng_generator) {
            curandDestroyGenerator(static_cast<curandGenerator_t>(cuda_rng_generator));
        }
#endif
        
        // Transfer from other
        base_seed = other.base_seed;
        data_seed = other.data_seed;
        init_seed = other.init_seed;
        cuda_seed = other.cuda_seed;
        data_rng = std::move(other.data_rng);
        cuda_rng_generator = other.cuda_rng_generator;
        deterministic = other.deterministic;
        cuda_rng_initialized = other.cuda_rng_initialized;
        
        // Nullify source
        other.cuda_rng_generator = nullptr;
        other.cuda_rng_initialized = false;
    }
    return *this;
}

//======================================================//
//  PathConfig Implementation
//======================================================//

bool PathConfig::validate() const {
    return !data_path.empty() &&
           !vocab_path.empty() &&
           !output_model_path.empty() &&
           !checkpoint_dir.empty() &&
           !log_dir.empty();
}

//======================================================//
//  Internal Helper Implementations
//======================================================//

namespace Internal {

StartupConfig loadConfiguration(int argc, char** argv) {
    StartupConfig config;
    
    // Load ai_config.json snapshot
    auto snapshot = GRIM::Config::loadAiConfigSnapshot();
    if (!snapshot) {
        throw std::runtime_error("FATAL: ai_config.json not found or unreadable");
    }
    
    config.paths.config_path = snapshot->config_path;
    const json& doc = snapshot->document;
    
    // Extract GRIM-text paths
    if (!doc.contains("paths") || !doc["paths"].contains("grim_text")) {
        throw std::runtime_error("FATAL: ai_config.json missing 'paths.grim_text' section");
    }
    
    const auto& grim_text = doc["paths"]["grim_text"];
    
    // Required paths
    if (!grim_text.contains("training_data") || grim_text["training_data"].get<std::string>().empty()) {
        throw std::runtime_error("FATAL: ai_config.json missing or empty 'grim_text.training_data'");
    }
    if (!grim_text.contains("vocab") || grim_text["vocab"].get<std::string>().empty()) {
        throw std::runtime_error("FATAL: ai_config.json missing or empty 'grim_text.vocab'");
    }
    
    config.paths.data_path = grim_text["training_data"].get<std::string>();
    config.paths.vocab_path = grim_text["vocab"].get<std::string>();
    config.paths.output_model_path = grim_text.value("model", "");
    config.paths.checkpoint_dir = grim_text.value("checkpoints", "checkpoints");
    config.paths.log_dir = grim_text.value("logs", "logs");
    config.paths.status_path = grim_text.value("training_status", "temp_status.txt");
    
    // Load hyperparameters
    if (snapshot->has_training) {
        config.hyperparameters = snapshot->hyperparameters;
    }
    GRIM::Tokenizer::configureTokenLayout(GRIM::Tokenizer::kAtomTypeCount);

    // Configure LogRecorder
    if (config.hyperparameters.log_recorder.enabled) {
        GRIM::Logging::SetDefaultModuleLogLevel(
            parseModuleLogLevelString(config.hyperparameters.log_recorder.default_level));
            
        for (const auto& [module, level] : config.hyperparameters.log_recorder.modules) {
            GRIM::Logging::SetModuleLogLevel(module, parseModuleLogLevelString(level));
        }
        
        // Initialize log recorder system
        GRIM::Logging::InitLogRecorder();
        
        // Configure layer logging enables
        const auto& layers = config.hyperparameters.log_recorder.layers;
        GRIM::Logging::ConfigureLayerLogging(
            config.hyperparameters.log_recorder.enabled,
            layers.embedding,
            layers.rms_norm,
            layers.attention,
            layers.feed_forward,
            layers.residual,
            layers.encoding,
            layers.serialization);
    } else {
        // Disable all layer logging if master disabled
        GRIM::Logging::ConfigureLayerLogging(false, false, false, false, false, false, false, false);
    }
    
    // Map loss options
    config.loss_options.label_smoothing_enabled = config.hyperparameters.loss_label_smoothing_enabled;
    config.loss_options.label_smoothing_epsilon = config.hyperparameters.loss_label_smoothing_epsilon;
    config.loss_options.focal_enabled = config.hyperparameters.loss_focal_enabled;
    config.loss_options.focal_gamma = config.hyperparameters.loss_focal_gamma;
    config.loss_options.focal_alpha = config.hyperparameters.loss_focal_alpha;
    config.loss_options.preference_enabled = config.hyperparameters.loss_preference_enabled;
    config.loss_options.preference_beta = config.hyperparameters.loss_preference_beta;
    config.loss_options.distillation_enabled = config.hyperparameters.loss_distillation_enabled;
    config.loss_options.distillation_temperature = config.hyperparameters.loss_distillation_temperature;
    config.loss_options.distillation_lambda = config.hyperparameters.loss_distillation_lambda;
    config.loss_options.masking_enabled = config.hyperparameters.loss_masking_enabled;
    config.loss_options.masking_tag = config.hyperparameters.loss_masking_tag;
    // Issue #44 FIX: Entropy regularization to prevent mode collapse
    config.loss_options.entropy_reg_enabled = config.hyperparameters.loss_entropy_reg_enabled;
    config.loss_options.entropy_reg_lambda = config.hyperparameters.loss_entropy_reg_lambda;
    
    // Map stability overrides
    config.stability.enabled = config.hyperparameters.stability_overrides_enabled;
    config.stability.batch_size = config.hyperparameters.stability_override_batch_size;
    config.stability.max_seq_len = config.hyperparameters.stability_override_max_seq_len;
    config.stability.max_tokens_per_batch = config.hyperparameters.stability_override_max_tokens_per_batch;
    config.stability.clip_abs = config.hyperparameters.stability_override_clip_abs;
    config.stability.clip_norm = config.hyperparameters.stability_override_clip_norm;
    config.stability.lr_min = config.hyperparameters.stability_override_lr_min;
    
    // Map scratch block configuration
    config.scratch.enabled = config.hyperparameters.scratch_blocks_enabled;
    config.scratch.max_tokens_per_block = config.hyperparameters.scratch_max_tokens_per_block;
    config.scratch.num_blocks = config.hyperparameters.scratch_num_blocks;
    config.scratch.write_combined = config.hyperparameters.scratch_write_combined;
    
    // Map CUDA execution mode
    config.cuda_exec.single_stream_mode = config.hyperparameters.single_stream_mode;
    config.cuda_exec.disable_async_frees = config.hyperparameters.disable_async_frees;
    config.cuda_exec.synchronize_after_kernels = config.hyperparameters.synchronize_after_kernels;
    
    // Map prediction comparison
    config.pred_comparison.enabled = config.hyperparameters.prediction_comparison_enabled;
    config.pred_comparison.interval = config.hyperparameters.prediction_comparison_interval;
    config.pred_comparison.top_k = config.hyperparameters.prediction_comparison_top_k;
    config.pred_comparison.max_positions = config.hyperparameters.prediction_comparison_max_positions;
    config.pred_comparison.log_path = config.hyperparameters.prediction_comparison_log_path;
    
    // Load architecture
    if (doc.contains("training") && doc["training"].contains("config")) {
        const auto& cfg = doc["training"]["config"];
        if (cfg.contains("d_model") && cfg["d_model"].is_number())
            config.architecture.d_model = cfg["d_model"].get<int>();
        if (cfg.contains("num_layers") && cfg["num_layers"].is_number())
            config.architecture.num_layers = cfg["num_layers"].get<int>();
        if (cfg.contains("num_heads") && cfg["num_heads"].is_number())
            config.architecture.num_heads = cfg["num_heads"].get<int>();
        if (cfg.contains("num_kv_heads") && cfg["num_kv_heads"].is_number())
            config.architecture.num_kv_heads = cfg["num_kv_heads"].get<int>();
        if (cfg.contains("d_ff") && cfg["d_ff"].is_number())
            config.architecture.d_ff = cfg["d_ff"].get<int>();
        else
            config.architecture.d_ff = config.architecture.d_model * GRIM::HyperParameters::DEFAULT_D_FF_MULTIPLIER;
        if (cfg.contains("dropout_rate") && cfg["dropout_rate"].is_number())
            config.architecture.dropout_rate = cfg["dropout_rate"].get<float>();
        if (cfg.contains("attention_dropout") && cfg["attention_dropout"].is_number())
            config.architecture.attention_dropout = cfg["attention_dropout"].get<float>();
        
        // Load tie_embeddings config (affects memory layout and parameter count)
        if (cfg.contains("tie_embeddings") && cfg["tie_embeddings"].is_boolean())
            config.architecture.tie_embeddings = cfg["tie_embeddings"].get<bool>();
            
        config.force_rebuild_vocab = cfg.value("force_rebuild_vocab", false);
    }
    config.architecture.max_seq_len = config.hyperparameters.max_seq_len;
    config.architecture.validate();
    
    // Compute derived values
    // Rule 20: No fallback - max_seq_len must be configured (used for cache allocation)
    // ONLY use stability override if stability mode is actually enabled
    if (config.stability.enabled && config.stability.max_seq_len > 0) {
        config.max_seq_len = config.stability.max_seq_len;
    } else if (config.hyperparameters.max_seq_len > 0) {
        config.max_seq_len = config.hyperparameters.max_seq_len;
    } else {
        throw std::runtime_error("FATAL: max_seq_len not configured in ai_config.json (stability or hyperparameters)");
    }
    config.sliding_window_stride = std::max(1, config.max_seq_len / 2);
    
    // Apply stability overrides to batch size and LR (only if stability mode enabled)
    if (config.stability.enabled) {
        config.hyperparameters.batch_size = config.stability.batch_size;
        config.hyperparameters.dynamic_lr_min = config.stability.lr_min;
    }
 
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) config.paths.data_path = argv[++i];
        else if (arg == "--vocab" && i + 1 < argc) config.paths.vocab_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc) config.paths.output_model_path = argv[++i];
        else if (arg == "--epochs" && i + 1 < argc) config.hyperparameters.epochs = std::atoi(argv[++i]);
        else if (arg == "--batch-size" && i + 1 < argc) config.hyperparameters.batch_size = std::atoi(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) config.hyperparameters.learning_rate = std::atof(argv[++i]);
        else if (arg == "--save-test") config.save_test_mode = true;
        else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --data <path>      Training data path\n";
            std::cout << "  --vocab <path>     Vocabulary path\n";
            std::cout << "  --output <path>    Output model path\n";
            std::cout << "  --epochs <n>       Number of epochs\n";
            std::cout << "  --batch-size <n>   Batch size\n";
            std::cout << "  --lr <rate>        Learning rate\n";
            std::cout << "  --save-test        Test serialization save and exit\n";
            std::exit(0);
        }
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
    fs::create_directories(fs::path(paths.output_model_path).parent_path());
    fs::create_directories(paths.checkpoint_dir);
    fs::create_directories(paths.log_dir);
}

LoggingContext initializeLogging(const PathConfig& paths) {
    LoggingContext ctx;
    
    ctx.session_id = std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count());
    ctx.raw_log_path = paths.log_dir + "/training_" + ctx.session_id + ".log";
    
    // Bootstrap log file
    {
        std::ofstream bootstrap(ctx.raw_log_path, std::ios::app);
        if (bootstrap.is_open()) {
            auto now = std::chrono::system_clock::now();
            auto tt = std::chrono::system_clock::to_time_t(now);
            bootstrap << "[BOOT] Phase1 logging bootstrap at "
                      << std::put_time(std::localtime(&tt), "%Y-%m-%d %H:%M:%S")
                      << std::endl;
        }
    }
    
    // Set grad check log path
    GRIM::setGradCheckLogPath(ctx.raw_log_path);
    
    ctx.logger = std::make_unique<TrainingLogger>(paths.log_dir, ctx.session_id);
    ctx.metrics_collector = std::make_unique<MetricsCollector>();
    ctx.status_writer = std::make_unique<StatusFileWriter>(paths.status_path);
    
    return ctx;
}

GRIM::Tokenizer::UniByte initializeTokenizer(
    const std::string& vocab_path,
    const GRIM::Config::TokenizerConfig& tok_config,
    TrainingLogger& logger) {
    
    logger.log("Loading tokenizer configuration...");
    
    GRIM::Tokenizer::UniByteConfig cfg;
    // Rule 20: No fallback - vocab_size must be in config. Actual vocab comes from .grmt file later.
    if (tok_config.vocab_size <= 0) {
        throw std::runtime_error("FATAL: tokenizer.vocab_size not configured in ai_config.json");
    }
    cfg.target_vocab_size = tok_config.vocab_size;
    cfg.character_coverage = 0.9995f;
    cfg.enable_scratch_block_reasoning = true;
    cfg.detect_numbers = true;
    cfg.detect_urls = true;
    cfg.detect_emails = true;
    cfg.detect_paths = true;
    cfg.detect_dates = true;
    cfg.detect_code_literals = true;
    cfg.enable_byte_fallback = tok_config.enable_byte_fallback;
    cfg.prefer_gpu = true;
    
    GRIM::Tokenizer::UniByte tokenizer(cfg);
    if (!tokenizer.load(vocab_path)) {
        throw std::runtime_error("Failed to load vocabulary: " + vocab_path);
    }
    
    const int original_vocab = tokenizer.totalVocabSize();
    
    // Cap vocab size if configured (reduces loss computation time)
    if (tok_config.max_vocab_size > 0 && original_vocab > tok_config.max_vocab_size) {
        tokenizer.capVocabSize(tok_config.max_vocab_size);
        logger.log("✓ Loaded " + std::to_string(original_vocab) + " tokens, capped to " + 
                   std::to_string(tokenizer.totalVocabSize()) + " (max_vocab_size=" + 
                   std::to_string(tok_config.max_vocab_size) + ")");
    } else {
        logger.log("✓ Loaded " + std::to_string(tokenizer.totalVocabSize()) + " tokens");
    }
    
    return tokenizer;
}

SequenceData loadTrainingData(
    const std::string& data_path,
    int max_seq_len,
    int min_seq_valid_tokens,
    int sliding_window_stride,
    const GRIM::Tokenizer::UniByte& tokenizer,
    TrainingLogger& logger) {
    
    SequenceData data;
    
    logger.log("Loading training data...");
    GRMTDataLoader loader;
    if (!loader.load(data_path)) {
        throw std::runtime_error("Failed to load training data");
    }
    logger.log("✓ Loaded " + std::to_string(loader.size()) + " sequences");
    
    auto all_sequences = loader.getSequences();

    const int bos_id = tokenizer.bosId();
    const int eos_id = tokenizer.eosId();
    const auto resolveLegacyToken = [&](const char* token) -> int {
        const auto ids = tokenizer.encode(token);
        if (ids.size() == 1) {
            const int id = ids[0];
            if (tokenizer.tokenToString(id) == token) {
                return id;
            }
        }
        return -1;
    };
    const int legacy_bos_id = resolveLegacyToken("<|startoftext|>");
    const int legacy_eos_id = resolveLegacyToken("<|endoftext|>");

    size_t added_bos = 0;
    size_t added_eos = 0;
    size_t replaced_legacy_bos = 0;
    size_t replaced_legacy_eos = 0;
    size_t modified = 0;

    for (auto& seq : all_sequences) {
        if (seq.token_ids.empty()) continue;

        bool changed = false;
        if (legacy_bos_id >= 0 && bos_id >= 0 && legacy_bos_id != bos_id &&
            seq.token_ids.front() == legacy_bos_id) {
            seq.token_ids.front() = bos_id;
            replaced_legacy_bos++;
            changed = true;
        }
        if (legacy_eos_id >= 0 && eos_id >= 0 && legacy_eos_id != eos_id &&
            seq.token_ids.back() == legacy_eos_id) {
            seq.token_ids.back() = eos_id;
            replaced_legacy_eos++;
            changed = true;
        }

        const bool has_bos = (bos_id >= 0 && seq.token_ids.front() == bos_id);
        const bool has_eos = (eos_id >= 0 && seq.token_ids.back() == eos_id);
        const bool add_bos = (bos_id >= 0 && !has_bos);
        const bool add_eos = (eos_id >= 0 && !has_eos);

        if (add_bos || add_eos) {
            std::vector<int> new_ids;
            new_ids.reserve(seq.token_ids.size() + (add_bos ? 1 : 0) + (add_eos ? 1 : 0));
            std::vector<float> new_numeric_values;
            new_numeric_values.reserve(seq.token_numeric_values.size() + (add_bos ? 1 : 0) + (add_eos ? 1 : 0));
            std::vector<uint8_t> new_numeric_mask;
            new_numeric_mask.reserve(seq.token_numeric_mask.size() + (add_bos ? 1 : 0) + (add_eos ? 1 : 0));
            // GRMT v4: text features
            std::vector<uint16_t> new_text_features;
            new_text_features.reserve(seq.token_text_features.size() + (add_bos ? GRIM::Tokenizer::kTextFeatureDim : 0) + (add_eos ? GRIM::Tokenizer::kTextFeatureDim : 0));
            std::vector<uint8_t> new_text_mask;
            new_text_mask.reserve(seq.token_text_mask.size() + (add_bos ? 1 : 0) + (add_eos ? 1 : 0));
            // GRMT v5: Also shift precomputed targets when adding BOS/EOS
            std::vector<int> new_targets;
            new_targets.reserve(seq.targets.size() + (add_bos ? 1 : 0) + (add_eos ? 1 : 0));
            if (add_bos) {
                new_ids.push_back(bos_id);
                new_numeric_values.push_back(0.0f);
                new_numeric_mask.push_back(0);
                for (int i = 0; i < GRIM::Tokenizer::kTextFeatureDim; ++i) new_text_features.push_back(0);
                new_text_mask.push_back(0);
                new_targets.push_back(-1);  // BOS position has no target (masked)
                added_bos++;
            }
            new_ids.insert(new_ids.end(), seq.token_ids.begin(), seq.token_ids.end());
            new_numeric_values.insert(new_numeric_values.end(),
                                      seq.token_numeric_values.begin(),
                                      seq.token_numeric_values.end());
            new_numeric_mask.insert(new_numeric_mask.end(),
                                    seq.token_numeric_mask.begin(),
                                    seq.token_numeric_mask.end());
            new_text_features.insert(new_text_features.end(),
                                     seq.token_text_features.begin(),
                                     seq.token_text_features.end());
            new_text_mask.insert(new_text_mask.end(),
                                 seq.token_text_mask.begin(),
                                 seq.token_text_mask.end());
            // GRMT v5: Copy precomputed targets (shifted by BOS if added)
            new_targets.insert(new_targets.end(),
                               seq.targets.begin(),
                               seq.targets.end());
            if (add_eos) {
                new_ids.push_back(eos_id);
                new_numeric_values.push_back(0.0f);
                new_numeric_mask.push_back(0);
                for (int i = 0; i < GRIM::Tokenizer::kTextFeatureDim; ++i) new_text_features.push_back(0);
                new_text_mask.push_back(0);
                // GRMT v5: Last target before EOS should be -1 (don't train to predict EOS)
                if (!new_targets.empty()) {
                    new_targets.back() = -1;
                }
                new_targets.push_back(-1);  // EOS position has no target
                added_eos++;
            }
            seq.token_ids = std::move(new_ids);
            seq.token_numeric_values = std::move(new_numeric_values);
            seq.token_numeric_mask = std::move(new_numeric_mask);
            seq.token_text_features = std::move(new_text_features);
            seq.token_text_mask = std::move(new_text_mask);
            // GRMT v5: Use shifted precomputed targets instead of recomputing from scratch
            seq.targets = std::move(new_targets);
            changed = true;
            modified++;
        }
    }

    if (modified > 0) {
        logger.log("[Data] Boundary tokens normalized: added_bos=" + std::to_string(added_bos) +
                   " added_eos=" + std::to_string(added_eos) +
                   " legacy_bos=" + std::to_string(replaced_legacy_bos) +
                   " legacy_eos=" + std::to_string(replaced_legacy_eos));
    }

    // Split train/val
    size_t val_size = all_sequences.size() / 10;
    data.train_seqs.assign(all_sequences.begin() + val_size, all_sequences.end());
    data.val_seqs.assign(all_sequences.begin(), all_sequences.begin() + val_size);
    
    // Apply sliding windows
    auto applySlidingWindows = [&](std::vector<TrainingSequence>& sequences,
                                   const std::string& split_name,
                                   bool mask_window_last_token) {
        std::vector<TrainingSequence> windowed;
        windowed.reserve(sequences.size());
        
        size_t long_seq_count = 0;
        size_t generated_windows = 0;
        
        for (const auto& seq : sequences) {
            if (static_cast<int>(seq.token_ids.size()) <= max_seq_len) {
                windowed.push_back(seq);
                continue;
            }
            
            long_seq_count++;
            const size_t seq_len = seq.token_ids.size();
            size_t start = 0;
            const size_t stride = static_cast<size_t>(sliding_window_stride);
            
            while (start < seq_len) {
                size_t end = std::min(seq_len, start + static_cast<size_t>(max_seq_len));
                
                TrainingSequence window;
                window.token_ids.assign(seq.token_ids.begin() + start, seq.token_ids.begin() + end);
                window.targets.assign(seq.targets.begin() + start, seq.targets.begin() + end);
                window.token_numeric_values.assign(seq.token_numeric_values.begin() + start,
                                                  seq.token_numeric_values.begin() + end);
                window.token_numeric_mask.assign(seq.token_numeric_mask.begin() + start,
                                                 seq.token_numeric_mask.begin() + end);
                // GRMT v4: slice text features (16 values per token)
                window.token_text_features.assign(
                    seq.token_text_features.begin() + start * GRIM::Tokenizer::kTextFeatureDim,
                    seq.token_text_features.begin() + end * GRIM::Tokenizer::kTextFeatureDim);
                window.token_text_mask.assign(seq.token_text_mask.begin() + start,
                                              seq.token_text_mask.begin() + end);
                // GRMT v6: slice byte lengths
                window.token_byte_lengths.assign(seq.token_byte_lengths.begin() + start,
                                                 seq.token_byte_lengths.begin() + end);
                if (mask_window_last_token && !window.targets.empty()) {
                    window.targets.back() = -1;
                }
                windowed.push_back(std::move(window));
                generated_windows++;
                
                if (end == seq_len) break;
                start += stride;
            }
        }
        
        sequences = std::move(windowed);
        if (long_seq_count > 0) {
            logger.log("Sliding window (" + split_name + "): " +
                       std::to_string(long_seq_count) + " long sequences expanded into " +
                       std::to_string(generated_windows) + " windows");
        }
    };
    
    applySlidingWindows(data.train_seqs, "train", true);
    applySlidingWindows(data.val_seqs, "val", false);
    
    // HARD FILTER: Remove any sequences still exceeding max_seq_len after sliding window
    // This catches cached .grmt files with old sequence lengths
    auto filterOverlong = [&](std::vector<TrainingSequence>& sequences, const std::string& split_name) {
        size_t before = sequences.size();
        sequences.erase(
            std::remove_if(sequences.begin(), sequences.end(),
                [max_seq_len](const TrainingSequence& seq) {
                    return static_cast<int>(seq.token_ids.size()) > max_seq_len;
                }),
            sequences.end());
        size_t removed = before - sequences.size();
        if (removed > 0) {
            logger.log("[FILTER] " + split_name + ": Removed " + std::to_string(removed) + 
                       " sequences exceeding max_seq_len=" + std::to_string(max_seq_len));
        }
    };
    filterOverlong(data.train_seqs, "train");
    filterOverlong(data.val_seqs, "val");

    // HARD FILTER: Remove sequences with too few valid tokens after masking
    // This prevents "valid_tokens=0" errors during loss computation
    // (position 0 is always masked as BOS, final position is masked as boundary)
    auto filterShortSequences = [&](std::vector<TrainingSequence>& sequences, const std::string& split_name) {
        if (min_seq_valid_tokens <= 0) return;  // Disabled if <= 0
        size_t before = sequences.size();
        sequences.erase(
            std::remove_if(sequences.begin(), sequences.end(),
                [min_seq_valid_tokens](const TrainingSequence& seq) {
                    // Count valid targets: excludes position 0 (BOS) and final position (boundary)
                    // Mirrors the masking logic in ComputeLossBatch.cu::prepareLossBatchInputs()
                    int valid = 0;
                    for (size_t i = 1; i + 1 < seq.targets.size(); ++i) {
                        if (seq.targets[i] >= 0) valid++;
                    }
                    return valid < min_seq_valid_tokens;
                }),
            sequences.end());
        size_t removed = before - sequences.size();
        if (removed > 0) {
            logger.log("[FILTER] " + split_name + ": Removed " + std::to_string(removed) + 
                       " sequences with < " + std::to_string(min_seq_valid_tokens) + " valid tokens");
        }
    };
    filterShortSequences(data.train_seqs, "train");
    filterShortSequences(data.val_seqs, "val");
    
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
    
    logger.log("Train sequences: " + std::to_string(data.train_seqs.size()));
    logger.log("Val sequences: " + std::to_string(data.val_seqs.size()));
    
    return data;
}

std::unique_ptr<GRIM::LanguageModel> initializeModel(
    const StartupConfig& config,
    uint32_t vocab_size,
    uint64_t xavier_seed,
    TrainingLogger& logger) {
    
    logger.log("Initializing model with xavier_seed=" + std::to_string(xavier_seed) + "...");
    
    const auto& arch = config.architecture;
    const auto& hp = config.hyperparameters;
    
    GRIM::LanguageModelConfig model_config;
    model_config.vocab_size = vocab_size;
    model_config.d_model = arch.d_model;
    model_config.num_layers = arch.num_layers;
    model_config.num_heads = arch.num_heads;
    model_config.num_kv_heads = arch.num_kv_heads;  // GQA: use config value from ai_config.json
    model_config.d_ff = arch.d_ff;
    // Rule 20: No fallback - use config.max_seq_len (already validated in loadConfiguration)
    model_config.max_seq_len = config.max_seq_len;
    model_config.dropout_rate = arch.dropout_rate;
    model_config.attention_dropout = arch.attention_dropout;
    model_config.vocab_path = config.paths.vocab_path;
    model_config.infer_vocab_from_file = true;
    model_config.positional_encoding = GRIM::HyperParameters::DEFAULT_POSITIONAL_ENCODING;
    model_config.causal_mask = true;
    model_config.use_pre_norm = true;
    model_config.fuse_qkv = true;
    model_config.tie_embeddings = arch.tie_embeddings;  // Load from ai_config.json
    model_config.use_bias = true;
    model_config.use_gpu = true;
    model_config.use_flash_attention = hp.use_flash_attention;
    model_config.min_seq_len_for_flash = 512;
    logger.log("Flash attention: enabled=" + std::string(model_config.use_flash_attention ? "true" : "false") +
               ", min_seq_len=" + std::to_string(model_config.min_seq_len_for_flash));
    
    // ScratchBlock reasoning config (loaded from ai_config.json via config.hyperparameters)
    model_config.use_scratch_block = hp.scratch_block_reasoning_enabled;
    model_config.scratch_block_atom_embedding_dim = hp.scratch_block_reasoning_atom_embedding_dim;
    model_config.scratch_block_max_atoms = hp.scratch_block_reasoning_max_atoms;
    model_config.scratch_block_atom_scale = hp.scratch_block_reasoning_atom_scale;
    
    // Compute derived values (head_dim = d_model / num_heads)
    model_config.computeDerivedValues();
    
    logger.log("ScratchBlock reasoning: enabled=" + std::string(model_config.use_scratch_block ? "true" : "false") +
              ", atom_embedding_dim=" + std::to_string(model_config.scratch_block_atom_embedding_dim) +
              ", max_atoms=" + std::to_string(model_config.scratch_block_max_atoms) +
              ", atom_scale=" + std::to_string(model_config.scratch_block_atom_scale));

    model_config.numeric_head_enabled = hp.loss_numeric_head_enabled;
    model_config.numeric_head_loss_weight = hp.loss_numeric_head_weight;
    model_config.numeric_head_huber_delta = hp.loss_numeric_head_huber_delta;
    model_config.numeric_head_log_scale = hp.loss_numeric_head_log_scale;

    logger.log("Numeric head: enabled=" + std::string(model_config.numeric_head_enabled ? "true" : "false") +
              ", loss_weight=" + std::to_string(model_config.numeric_head_loss_weight) +
              ", huber_delta=" + std::to_string(model_config.numeric_head_huber_delta) +
              ", log_scale=" + std::string(model_config.numeric_head_log_scale ? "true" : "false"));
    
    // LM Head centering configuration (Issue #37 / #40)
    model_config.lm_head_center_hidden_states = hp.lm_head_center_hidden_states;
    model_config.lm_head_recenter_gradients = hp.lm_head_recenter_gradients;
    
    logger.log("LM Head centering: center_hidden_states=" + std::string(model_config.lm_head_center_hidden_states ? "true" : "false") +
              ", recenter_gradients=" + std::string(model_config.lm_head_recenter_gradients ? "true" : "false"));
    
    // Hardcoded Hidden States Diagnostic (Issue #42)
    model_config.hardcoded_hidden_pattern = static_cast<GRIM::LanguageModelConfig::HardcodedPattern>(hp.hardcoded_hidden_pattern);
    model_config.hardcoded_log_every_n_batches = hp.hardcoded_log_every_n_batches;
    
    if (model_config.hardcoded_hidden_pattern != GRIM::LanguageModelConfig::HardcodedPattern::DISABLED) {
        logger.log("⚠️  HARDCODED HIDDEN STATES DIAGNOSTIC ENABLED: pattern=" + std::to_string(static_cast<int>(model_config.hardcoded_hidden_pattern)) +
                  ", log_every_n=" + std::to_string(model_config.hardcoded_log_every_n_batches));
        logger.log("⚠️  Encoder output will be REPLACED with synthetic patterns - this is a DIAGNOSTIC MODE ONLY!");
    }
    
    // Cache sizing - Use configured batch_size directly (stability override if enabled)
    // Rule 20: No backwards derivation from token budgets - we KNOW the batch size from config
    const int actual_batch_size = config.stability.enabled 
        ? config.hyperparameters.stability_override_batch_size
        : config.hyperparameters.batch_size;
    
    // Sequence length cap from cache_limits
    const uint32_t seq_cap = std::min<uint32_t>(
        static_cast<uint32_t>(model_config.max_seq_len),
        static_cast<uint32_t>(hp.cache_max_seq_len));
    
    // Cache allocation: allocate for the ACTUAL batch size, not some derived "max"
    // No arbitrary margins - if you need more, increase batch_size in config
    model_config.max_cached_batch = actual_batch_size;
    model_config.max_cached_seq_len = seq_cap;
    
    // Token budget is just batch * seq_len (used for logits allocation limit)
    const uint32_t token_budget = static_cast<uint32_t>(actual_batch_size) * seq_cap;
    model_config.max_tokens_per_batch = static_cast<int>(token_budget);
    
    logger.log("Cache allocation: batch=" + std::to_string(actual_batch_size) + 
               ", seq_len=" + std::to_string(seq_cap) + 
               ", tokens=" + std::to_string(token_budget));
    
    // Activation quantization
    model_config.activation_quantization.enabled = hp.activation_quantization_enabled;
    model_config.activation_quantization.apply_to_embeddings = hp.activation_quantization_apply_to_embeddings;
    model_config.activation_quantization.apply_to_encoder_outputs = hp.activation_quantization_apply_to_encoder_outputs;
    model_config.activation_quantization.apply_to_layer_caches = hp.activation_quantization_apply_to_layer_caches;
    model_config.activation_quantization.apply_to_qkv_cache = hp.activation_quantization_apply_to_qkv_cache;
    model_config.activation_quantization.apply_to_logits = hp.activation_quantization_apply_to_logits;
    model_config.activation_quantization.scale = hp.activation_quantization_scale;
    model_config.activation_quantization.clip_min = hp.activation_quantization_clip_min;
    model_config.activation_quantization.clip_max = hp.activation_quantization_clip_max;
    model_config.activation_quantization.zero_point = hp.activation_quantization_zero_point;
    model_config.activation_quantization.symmetric = hp.activation_quantization_symmetric;
    
    logger.log("Model architecture: d_model=" + std::to_string(arch.d_model) + 
               ", num_layers=" + std::to_string(arch.num_layers) + 
               ", num_heads=" + std::to_string(arch.num_heads) + 
               ", d_ff=" + std::to_string(arch.d_ff) +
               ", max_seq_len=" + std::to_string(model_config.max_seq_len));
    
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
    
    // STEP 1: Initialize just the stream controller part of TrainingState
    // This creates CUDA streams (CUDA context must exist from STEP 0)
    {
        GRIM::StreamControllerConfig stream_config;
        stream_config.verbose = true;
        stream_config.create_transfer_stream = true;
        stream_config.create_auxiliary_stream = false;
        
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
    
    // STEP 3: Initialize GPU encoder (uses cuBLAS handle from step 2, RoPE from step 2.5)
    logger.log("Initializing GPU encoder...");
    model->initGPU();
    logger.log("✓ GPU encoder fully initialized");
    
    // STEP 4: Finish TrainingState initialization (grad buffers, requires embedder from initGPU)
    logger.log("Initializing TrainingState (grad buffers, requires embedder from initGPU)...");
    model->initTrainingState();
    logger.log("✓ TrainingState fully initialized");

    if (config.hyperparameters.logit_update_trace_enabled) {
        const bool tied = model->getConfig().tie_embeddings;
        const std::string group = tied ? "embedding_lm_head_tied" : "lm_head_weight";
        model->configureUpdateProbe(group);
        logger.log("[LogitTrace] update_probe enabled group='" + group + "'");
    } else {
        model->disableUpdateProbe();
    }
    
    // Configure scratch blocks
    if (model->isScratchPoolInitialized()) {
        model->configureScratchPool(config.scratch.enabled);
        if (config.scratch.enabled) {
            logger.log("✓ Scratch blocks enabled (" +
                      std::to_string(config.scratch.num_blocks) + " blocks × " +
                      std::to_string(config.scratch.max_tokens_per_block) + " tokens)");
        }
    }
    
    model->setLossOptions(config.loss_options);
    
    // Log loss configuration
    std::cout << "[LossConfig] startup: "
              << "label_smoothing=" << (config.loss_options.label_smoothing_enabled ? "ON" : "off")
              << " focal=" << (config.loss_options.focal_enabled ? "ON" : "off")
              << " distill=" << (config.loss_options.distillation_enabled ? "ON" : "off")
              << " pref=" << (config.loss_options.preference_enabled ? "ON" : "off")
              << " entropy_reg=" << (config.loss_options.entropy_reg_enabled ? "ON" : "off")
              << " ent_lambda=" << config.loss_options.entropy_reg_lambda
              << std::endl;
    
#ifdef USE_CUDA
    // Xavier initialization for encoder weights if needed
    {
        auto* gpu_encoder = &model->getGpuEncoder();
        const auto& cfg = model->getConfig();
        auto sample_rms = [&](float* ptr, size_t count) -> float {
            if (!ptr || count == 0) return 0.0f;
            std::vector<float> sample(std::min<size_t>(32, count), 0.0f);
            cudaMemcpy(sample.data(), ptr, sample.size() * sizeof(float), cudaMemcpyDeviceToHost);
            float sum_sq = 0.f;
            for (float v : sample) sum_sq += v * v;
            return std::sqrt(sum_sq / static_cast<float>(sample.size()));
        };
        
        for (int layer = 0; layer < cfg.num_layers; ++layer) {
            auto* enc = gpu_encoder->getLayer(layer);
            if (!enc) continue;
            
            float* w_qkv = enc->getAttnWqkv();
            float* w_o = enc->getAttnWo();
            float* w1 = enc->getFFNW1();
            float* w2 = enc->getFFNW2();
            
            // GQA dimension calculation using TensorContract
            const int head_dim = cfg.head_dim;  // Use pre-computed value from config
            TensorContract::GQADims gqa_dims{cfg.num_heads, cfg.num_kv_heads, head_dim};
            const int total_qkv_dim = gqa_dims.total_qkv_dim();
            const size_t qkv_size = static_cast<size_t>(total_qkv_dim) * cfg.d_model;
            
            const float qkv_std = std::sqrt(2.0f / (cfg.d_model + static_cast<float>(total_qkv_dim)));
            const float wo_std = std::sqrt(2.0f / (2.0f * cfg.d_model));
            const float w1_std = std::sqrt(2.0f / (cfg.d_model + cfg.d_ff));
            const float w2_std = std::sqrt(2.0f / (cfg.d_ff + cfg.d_model));
            
            bool reseeded = false;
            if (sample_rms(w_qkv, qkv_size) < 1e-7f && w_qkv) {
                launchXavierInit(w_qkv, static_cast<int>(qkv_size), qkv_std, xavier_seed + layer * 4 + 0, 0);
                reseeded = true;
            }
            if (sample_rms(w_o, cfg.d_model * cfg.d_model) < 1e-7f && w_o) {
                launchXavierInit(w_o, cfg.d_model * cfg.d_model, wo_std, xavier_seed + layer * 4 + 1, 0);
                reseeded = true;
            }
            if (sample_rms(w1, cfg.d_model * cfg.d_ff) < 1e-7f && w1) {
                launchXavierInit(w1, cfg.d_model * cfg.d_ff, w1_std, xavier_seed + layer * 4 + 2, 0);
                reseeded = true;
            }
            if (sample_rms(w2, cfg.d_ff * cfg.d_model) < 1e-7f && w2) {
                launchXavierInit(w2, cfg.d_ff * cfg.d_model, w2_std, xavier_seed + layer * 4 + 3, 0);
                reseeded = true;
            }
            if (reseeded) {
                logger.log("[EncoderInit] Layer " + std::to_string(layer) + " reseeded with Xavier");
            }
        }
    }
#endif
    

    logger.log("Update probe DISABLED for performance");
    
    // Try to load checkpoint
    std::string latest_checkpoint = config.paths.checkpoint_dir + "/checkpoint_epoch_1.bin";
    if (fs::exists(latest_checkpoint)) {
        logger.log("Found checkpoint: " + latest_checkpoint);
        if (model->load(latest_checkpoint)) {
            logger.log("✓ Loaded weights from checkpoint");
        } else {
            logger.log("⚠ Failed to load checkpoint, starting fresh");
        }
    } else {
        logger.log("No checkpoint found, starting fresh");
    }
    
    logger.log("✓ Model initialized");
    return model;
}

OptimizerContext initializeOptimizer(
    GRIM::LanguageModel& model,
    const StartupConfig& config,
    TrainingLogger& logger) {
    
    OptimizerContext ctx;
    const auto& hp = config.hyperparameters;
    
    logger.log("Initializing optimizer state...");
    
    // Optimizer state (AdamW hyperparameters are defined as constants in AdamW_Kernal_GPU.cu)
    ctx.optimizer_state.step = 0;
    
    // Soft restart controller
    GRIM::SoftRestart::SoftRestartConfig sr_cfg;
    sr_cfg.loss_increase_threshold = hp.soft_restart_loss_increase_threshold;
    sr_cfg.max_step_window = hp.soft_restart_max_step_window;
    sr_cfg.cooldown_steps = hp.soft_restart_cooldown_steps;
    ctx.soft_restart_controller = GRIM::SoftRestart::SoftRestartController(sr_cfg);
    
    // Dynamic LR controller
    GRIM::DynamicLR::DynamicLRConfig lr_cfg;
    lr_cfg.base_learning_rate = hp.learning_rate;
    lr_cfg.min_learning_rate = hp.dynamic_lr_min;
    lr_cfg.max_learning_rate = hp.dynamic_lr_max;
    lr_cfg.increase_factor = hp.dynamic_lr_increase_factor;
    lr_cfg.decrease_factor = hp.dynamic_lr_decrease_factor;
    lr_cfg.upper_grad_norm = hp.dynamic_lr_upper_grad_norm;
    lr_cfg.lower_grad_norm = hp.dynamic_lr_lower_grad_norm;
    lr_cfg.max_loss_jump = hp.dynamic_lr_max_loss_jump;
    lr_cfg.smoothing = hp.dynamic_lr_smoothing;
    lr_cfg.cooldown_steps = hp.dynamic_lr_cooldown_steps;
    lr_cfg.warmup_steps = hp.dynamic_lr_warmup_steps;
    lr_cfg.max_step_up_ratio = hp.dynamic_lr_max_step_up_ratio;
    lr_cfg.max_step_down_ratio = hp.dynamic_lr_max_step_down_ratio;
    lr_cfg.auto_band = hp.dynamic_lr_auto_band;
    lr_cfg.band_sigma = hp.dynamic_lr_band_sigma;
    lr_cfg.band_floor = hp.dynamic_lr_band_floor;
    lr_cfg.band_ceiling = hp.dynamic_lr_band_ceiling;
    lr_cfg.band_min_samples = hp.dynamic_lr_band_min_samples;
    lr_cfg.band_min_span = hp.dynamic_lr_band_min_span;
    lr_cfg.adaptive_smoothing = hp.dynamic_lr_adaptive_smoothing;
    lr_cfg.smoothing_min = hp.dynamic_lr_smoothing_min;
    lr_cfg.smoothing_max = hp.dynamic_lr_smoothing_max;
    lr_cfg.variance_reference = hp.dynamic_lr_variance_reference;
    lr_cfg.adaptive_cooldown = hp.dynamic_lr_adaptive_cooldown;
    lr_cfg.cooldown_min = hp.dynamic_lr_cooldown_min;
    lr_cfg.cooldown_max = hp.dynamic_lr_cooldown_max;
    lr_cfg.adaptive_loss = hp.dynamic_lr_adaptive_loss;
    lr_cfg.loss_sigma = hp.dynamic_lr_loss_sigma;
    lr_cfg.loss_min_samples = hp.dynamic_lr_loss_min_samples;
    lr_cfg.loss_floor = hp.dynamic_lr_loss_floor;
    lr_cfg.guard_logging = hp.dynamic_lr_guard_logging;
    lr_cfg.guard_floor_steps = hp.dynamic_lr_guard_floor_steps;
    lr_cfg.guard_grad_multiplier = hp.dynamic_lr_guard_grad_multiplier;
    lr_cfg.guard_loss_patience = hp.dynamic_lr_guard_loss_patience;
    lr_cfg.guard_loss_multiplier = hp.dynamic_lr_guard_loss_multiplier;
    lr_cfg.baseline_capture_steps = hp.dynamic_lr_baseline_capture_steps;
    lr_cfg.baseline_drift = hp.dynamic_lr_baseline_drift;
    lr_cfg.momentum_interval = hp.dynamic_lr_momentum_interval;
    lr_cfg.momentum_gain = hp.dynamic_lr_momentum_gain;
    lr_cfg.momentum_decay = hp.dynamic_lr_momentum_decay;
    lr_cfg.safety_interval = hp.dynamic_lr_safety_interval;
    lr_cfg.safety_gain = hp.dynamic_lr_safety_gain;
    lr_cfg.safety_scale = hp.dynamic_lr_safety_scale;
    
    ctx.dynamic_lr_controller = GRIM::DynamicLR::DynamicLRController(lr_cfg);
    ctx.dynamic_lr_controller.setRuntimeLimits(hp.dynamic_lr_min, hp.dynamic_lr_max);
    ctx.dynamic_lr_controller.setEnabled(hp.dynamic_lr_enabled);
    
    // Gradient accumulation controller
    logger.log("Initializing GradAccumulationController...");
    ctx.grad_controller = std::make_unique<GRIM::ModelGradAccumulationController>();
    
    // DEBUG: Verify pointer integrity before bindToModel
    auto& ts = model.getTrainingState();
    {
        std::ostringstream oss;
        oss << "[DEBUG Phase1] Before bindToModel:\n"
            << "  model address=" << &model << "\n"
            << "  training_state address=" << &ts
            << " (offset from model: "
            << (reinterpret_cast<const char*>(&ts) - reinterpret_cast<const char*>(&model))
            << " bytes)\n"
            << "  embedding_grads=" << static_cast<const void*>(ts.embedding_grads()) << "\n"
            << "  lm_head_weight_grads=" << static_cast<const void*>(ts.lm_head_weight_grads()) << "\n"
            << "  embedding_grad_size=" << (static_cast<std::size_t>(model.getConfig().vocab_size) * model.getConfig().d_model) << "\n"
            << "  tie_embeddings=" << (model.getConfig().tie_embeddings ? "TRUE" : "FALSE");
        EmitModuleInfo(ModuleId::Training, oss.str());
    }
    
    ctx.grad_controller->bindToModel(model);
    
    GRIM::GradAccumulationConfig grad_cfg;
    grad_cfg.accum_steps = std::max(1, hp.gradient_accumulation_steps);  // Load from config, minimum 1
    grad_cfg.stream = model.getTrainingState().stream_ctrl.getPrimaryStream();  // Use StreamController's primary stream!
    
    // CRITICAL: Verify stream is valid before configuring controller
    if (!grad_cfg.stream) {
        EmitModuleError(ModuleId::Training,
                        "[FATAL] GradAccumulationController: getPrimaryStream() returned nullptr! StreamController may not be initialized yet.");
        throw std::runtime_error("GradAccumulationController: primary stream is null");
    }
    logger.log("✓ GradAccumulationController stream validated: " + 
               std::to_string(reinterpret_cast<uintptr_t>(grad_cfg.stream)));
    grad_cfg.zero_on_window_start = true;
    grad_cfg.zero_on_optimizer_complete = true;
    grad_cfg.auto_scale_loss = true;
    grad_cfg.verbose = true;  // CRITICAL: verbose=false to prevent registration spam during forward pass
    grad_cfg.strict_mode = true;
    // Disable per-buffer RMS monitoring by default (expensive: sync per buffer).
    grad_cfg.monitor_gradients = false;
    grad_cfg.gradient_explosion_threshold = 1e6f;
    ctx.grad_controller->configure(grad_cfg);
    
    // BUG FIX: Verify ALL encoder layers initialized BEFORE bindToModel was called
    // If layers 4-5 are missing, they'll be lazily initialized during forward pass,
    // causing GradController registration during execution → crash
    auto* gpu_encoder = &model.getGpuEncoder();
    for (int layer = 0; layer < model.getConfig().num_layers; ++layer) {
        if (!gpu_encoder->getLayer(layer)) {
            EmitModuleError(ModuleId::Training,
                            "[FATAL] Layer " + std::to_string(layer) + " not initialized before bindToModel! " +
                            "This causes lazy initialization during forward pass → crash. " +
                            "Ensure model.initGPU() completes all layers before bindToModel().");
            throw std::runtime_error("Encoder layer " + std::to_string(layer) + " not initialized");
        }
    }
    
    logger.log("✓ GradAccumulationController configured:");
    logger.log("  - Accumulation steps: " + std::to_string(grad_cfg.accum_steps) + 
               " (effective batch = " + std::to_string(hp.batch_size * grad_cfg.accum_steps) + ")");
    logger.log("  - Registered " + std::to_string(ctx.grad_controller->controller().bufferCount()) + " gradient buffers");
    logger.log("  - Total memory: " + std::to_string(ctx.grad_controller->controller().totalGradientBytes() / (1024.0 * 1024.0)) + " MB");
    
    logger.log("✓ Optimizer state initialized");
    return ctx;
}

std::vector<float> computeRareTokenScores(
    const std::vector<TrainingSequence*>& train_views,
    uint32_t vocab_size,
    size_t train_count,
    TrainingLogger& logger) {
    if (vocab_size == 0) {
        throw std::runtime_error("RareTokens: vocab_size must be explicitly provided by the call site");
    }
    if (vocab_size < static_cast<uint32_t>(GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET)) {
        throw std::runtime_error("RareTokens: vocab_size must include byte+atom token ranges (>= 512)");
    }

    GRIM::RareTokens::Config rare_cfg;
    rare_cfg.vocab_size = vocab_size;
    rare_cfg.rare_threshold = std::max<uint64_t>(16, train_count / 32);
    rare_cfg.smoothing = 4.0f;
    rare_cfg.max_boost = 12.0f;
    rare_cfg.rarity_exponent = 0.5f;
    
    std::vector<const std::vector<int>*> rare_token_sequences;
    rare_token_sequences.reserve(train_views.size());
    for (auto* seq : train_views) {
        if (!seq) {
            throw std::runtime_error("RareTokens: train view pointer is null");
        }
        rare_token_sequences.push_back(&seq->token_ids);
    }
    
    auto rare_token_freqs = GRIM::RareTokens::computeFrequencies(rare_token_sequences, rare_cfg.vocab_size);
    auto rare_inv_table = GRIM::RareTokens::buildInverseFrequencyTable(rare_token_freqs, rare_cfg);
    std::vector<float> sequence_rarity = GRIM::RareTokens::scoreSequences(rare_token_sequences, rare_inv_table, rare_cfg);
    
    if (!sequence_rarity.empty()) {
        float min_r = sequence_rarity.front();
        float max_r = sequence_rarity.front();
        float mean_r = 0.0f;
        for (float r : sequence_rarity) {
            min_r = std::min(min_r, r);
            max_r = std::max(max_r, r);
            mean_r += r;
        }
        mean_r /= static_cast<float>(sequence_rarity.size());
        
        std::ostringstream msg;
        msg << "[RareTokens] rarity scores: min=" << std::fixed << std::setprecision(4)
            << min_r << " max=" << max_r << " mean=" << mean_r
            << " threshold=" << rare_cfg.rare_threshold;
        logger.log(msg.str());
    }
    
    return sequence_rarity;
}

// Issue #38 FIX: Compute per-token class weights to prevent mode collapse on frequent tokens
// Returns GPU-allocated buffer with weights indexed by token ID
// Frequent tokens (like SPACE=277) get LOWER weight to reduce their gradient contribution
float* computeAndUploadTokenWeights(
    const std::vector<TrainingSequence*>& train_views,
    uint32_t vocab_size,
    cudaStream_t stream,
    TrainingLogger& logger) {
    
    if (vocab_size == 0) {
        throw std::runtime_error("computeAndUploadTokenWeights: vocab_size must be > 0");
    }
    
    logger.log("[Issue38Fix] Computing per-token class weights to prevent mode collapse...");
    
    // Step 1: Count token frequencies
    std::vector<uint64_t> token_counts(vocab_size, 0);
    uint64_t total_tokens = 0;
    for (const auto* seq : train_views) {
        if (!seq) continue;
        for (int token_id : seq->token_ids) {
            if (token_id >= 0 && static_cast<uint32_t>(token_id) < vocab_size) {
                token_counts[static_cast<size_t>(token_id)]++;
                total_tokens++;
            }
        }
    }
    
    if (total_tokens == 0) {
        logger.log("  ⚠ WARNING: No tokens found in training data, skipping token weights");
        return nullptr;
    }
    
    // Step 2: Compute inverse frequency weights
    // Formula: weight[t] = sqrt(total_tokens / (vocab_size * freq[t] + smoothing))
    // This gives lower weight to frequent tokens, higher weight to rare tokens
    // The sqrt prevents extreme weights while still providing meaningful differentiation
    const float smoothing = 1.0f;  // Prevent division by zero
    std::vector<float> host_weights(vocab_size);
    
    float min_weight = FLT_MAX;
    float max_weight = 0.0f;
    uint32_t token_277_count = token_counts[277];
    float token_277_weight = 0.0f;
    
    for (uint32_t t = 0; t < vocab_size; ++t) {
        float freq = static_cast<float>(token_counts[t]);
        // Inverse frequency with sqrt dampening
        float weight = sqrtf(static_cast<float>(total_tokens) / 
                            (static_cast<float>(vocab_size) * freq + smoothing));
        // Cap weights to prevent extreme values
        weight = std::min(weight, 10.0f);  // Max 10x boost for very rare tokens
        weight = std::max(weight, 0.1f);   // Min 0.1x for very frequent tokens
        host_weights[t] = weight;
        
        min_weight = std::min(min_weight, weight);
        max_weight = std::max(max_weight, weight);
        
        if (t == 277) {
            token_277_weight = weight;
        }
    }
    
    // Normalize so mean weight = 1.0 (preserves overall gradient scale)
    float mean_weight = 0.0f;
    for (float w : host_weights) {
        mean_weight += w;
    }
    mean_weight /= static_cast<float>(vocab_size);
    
    for (float& w : host_weights) {
        w /= mean_weight;
    }
    token_277_weight /= mean_weight;
    min_weight /= mean_weight;
    max_weight /= mean_weight;
    
    // Log statistics
    {
        std::ostringstream msg;
        float token_277_freq_pct = 100.0f * static_cast<float>(token_277_count) / static_cast<float>(total_tokens);
        msg << "[Issue38Fix] Token weight stats:\n"
            << "  Total tokens: " << total_tokens << "\n"
            << "  Token 277 (SPACE): count=" << token_277_count 
            << " (" << std::fixed << std::setprecision(2) << token_277_freq_pct << "% of all tokens)\n"
            << "  Token 277 weight: " << std::setprecision(4) << token_277_weight 
            << " (lower = less gradient contribution)\n"
            << "  Weight range: [" << min_weight << ", " << max_weight << "]";
        logger.log(msg.str());
    }
    
    // Step 3: Upload to GPU
    float* d_token_weights = nullptr;
    size_t weights_bytes = vocab_size * sizeof(float);
    
    cudaError_t err = cudaMalloc(&d_token_weights, weights_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc failed for token_weights: ") + 
                                 cudaGetErrorString(err));
    }
    
    err = cudaMemcpyAsync(d_token_weights, host_weights.data(), weights_bytes,
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(d_token_weights);
        throw std::runtime_error(std::string("cudaMemcpy failed for token_weights: ") + 
                                 cudaGetErrorString(err));
    }
    
    logger.log("✓ Token weights uploaded to GPU (" + 
               std::to_string(weights_bytes / 1024) + " KB)");
    
    return d_token_weights;
}

RNGContext initializeRNG(
    const StartupConfig& config,
    TrainingLogger& logger) {
    
    RNGContext ctx;
    
    logger.log("Initializing production-grade RNG system...");
    
    // Seed already loaded from ai_config.json via config.hyperparameters.seed
    int64_t config_seed = config.hyperparameters.seed;
    
    // Determine base seed
    if (config_seed < 0) {
        // Non-deterministic: use high-resolution timestamp
        auto now = std::chrono::high_resolution_clock::now();
        ctx.base_seed = static_cast<uint64_t>(now.time_since_epoch().count());
        ctx.deterministic = false;
        logger.log("✓ RNG mode: NON-DETERMINISTIC (random seed from timestamp)");
    } else {
        // Deterministic: use config seed
        ctx.base_seed = static_cast<uint64_t>(config_seed);
        ctx.deterministic = true;
        logger.log("✓ RNG mode: DETERMINISTIC (seed=" + std::to_string(ctx.base_seed) + ")");
    }
    
    // Hierarchical seeding (PyTorch/JAX pattern)
    ctx.data_seed = ctx.base_seed + 0;      // Data shuffling
    ctx.init_seed = ctx.base_seed + 1000;   // Weight initialization
    ctx.cuda_seed = ctx.base_seed + 2000;   // CUDA operations
    
    // Initialize CPU RNG for data operations
    ctx.data_rng = std::mt19937_64(ctx.data_seed);
    
#ifdef USE_CUDA
    // Initialize CUDA RNG for GPU dropout/sampling
    curandGenerator_t cuda_gen;
    curandStatus_t status = curandCreateGenerator(&cuda_gen, CURAND_RNG_PSEUDO_DEFAULT);
    if (status != CURAND_STATUS_SUCCESS) {
        logger.log("⚠ WARNING: Failed to create CUDA RNG generator (status=" + 
                   std::to_string(status) + "). GPU dropout will use uncontrolled randomness.");
        ctx.cuda_rng_generator = nullptr;
        ctx.cuda_rng_initialized = false;
    } else {
        status = curandSetPseudoRandomGeneratorSeed(cuda_gen, ctx.cuda_seed);
        if (status != CURAND_STATUS_SUCCESS) {
            logger.log("⚠ WARNING: Failed to set CUDA RNG seed (status=" + 
                       std::to_string(status) + "). GPU dropout will use default seed.");
            curandDestroyGenerator(cuda_gen);
            ctx.cuda_rng_generator = nullptr;
            ctx.cuda_rng_initialized = false;
        } else {
            ctx.cuda_rng_generator = static_cast<void*>(cuda_gen);
            ctx.cuda_rng_initialized = true;
            logger.log("✓ CUDA RNG initialized (seed=" + std::to_string(ctx.cuda_seed) + ")");
        }
    }
#else
    ctx.cuda_rng_generator = nullptr;
    ctx.cuda_rng_initialized = false;
    logger.log("⚠ CUDA RNG not available (USE_CUDA not defined)");
#endif
    
    // Log seed hierarchy for reproducibility
    std::ostringstream seed_log;
    seed_log << "RNG seed hierarchy:\n"
             << "  base_seed  = " << ctx.base_seed << " (master)\n"
             << "  data_seed  = " << ctx.data_seed << " (CPU shuffling)\n"
             << "  init_seed  = " << ctx.init_seed << " (weight initialization)\n"
             << "  cuda_seed  = " << ctx.cuda_seed << " (GPU dropout/sampling)\n"
             << "To reproduce this run, set ai_config.json: training.config.seed = " << ctx.base_seed;
    logger.log(seed_log.str());
    
    logger.log("✓ RNG system initialized");
    return ctx;
}

} // namespace Internal

//======================================================//
//  Phase1 Main Entry Point
//======================================================//

std::unique_ptr<TrainingContext> executePhase1(int argc, char** argv) {
    auto ctx = std::make_unique<TrainingContext>();
    
    EmitModuleInfo(ModuleId::Training, "========================================", 0);
    EmitModuleInfo(ModuleId::Training, "  Phase 1: Startup & Initialization", 0);
    EmitModuleInfo(ModuleId::Training, "  GRIM-text GPU Training v2.0.0", 0);
    EmitModuleInfo(ModuleId::Training, "========================================", 0);
    
    // Enable verbose module logging
    GRIM::Logging::ApplyModuleLogOverrides({
        GRIM::Logging::MakeOverride(GRIM::Logging::ModuleId::ForwardPass, GRIM::Logging::ModuleLogLevel::Info),
        GRIM::Logging::MakeOverride(GRIM::Logging::ModuleId::BackwardPass, GRIM::Logging::ModuleLogLevel::Info),
        GRIM::Logging::MakeOverride(GRIM::Logging::ModuleId::Checkpoint, GRIM::Logging::ModuleLogLevel::Info)
    });
    registerDefaultLoggingProfiles();
    
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
    
    // Forward BackwardPass module logs to training logger (MUST persist!)
    ctx->logging.backward_sink = std::make_unique<GRIM::Logging::ModuleLogSink>();
    auto backward_formatter = [logger = ctx->logging.logger.get()](const GRIM::Logging::ModuleLogEvent& evt) {
        const char* level = "INFO";
        switch (evt.level) {
            case GRIM::Logging::ModuleLogLevel::Warning: level = "WARN"; break;
            case GRIM::Logging::ModuleLogLevel::Error: level = "ERR"; break;
            default: break;
        }
        std::ostringstream msg;
        msg << "[" << evt.module << "][" << level << "] " << evt.message;
        logger->log(msg.str());
    };
    if (!ctx->logging.backward_sink->bind("BackwardPass", backward_formatter)) {
        ctx->logging.logger->log("[WARNING] Failed to bind BackwardPass module logger - backward diagnostics may not appear in logs");
    }
    
    // Forward StreamController module logs to training logger (CRITICAL for init debugging!)
    ctx->logging.stream_controller_sink = std::make_unique<GRIM::Logging::ModuleLogSink>();
    auto stream_formatter = [logger = ctx->logging.logger.get()](const GRIM::Logging::ModuleLogEvent& evt) {
        const char* level = "INFO";
        switch (evt.level) {
            case GRIM::Logging::ModuleLogLevel::Warning: level = "WARN"; break;
            case GRIM::Logging::ModuleLogLevel::Error: level = "ERR"; break;
            default: break;
        }
        std::ostringstream msg;
        msg << "[" << evt.module << "][" << level << "] " << evt.message;
        logger->log(msg.str());
    };
    if (!ctx->logging.stream_controller_sink->bind("StreamController", stream_formatter)) {
        ctx->logging.logger->log("[WARNING] Failed to bind StreamController module logger - stream diagnostics may not appear in logs");
    }
    
    // Forward Checkpoint module logs to training logger (CRITICAL for save/load debugging!)
    ctx->logging.checkpoint_sink = std::make_unique<GRIM::Logging::ModuleLogSink>();
    auto checkpoint_formatter = [logger = ctx->logging.logger.get()](const GRIM::Logging::ModuleLogEvent& evt) {
        const char* level = "INFO";
        switch (evt.level) {
            case GRIM::Logging::ModuleLogLevel::Warning: level = "WARN"; break;
            case GRIM::Logging::ModuleLogLevel::Error: level = "ERR"; break;
            default: break;
        }
        std::ostringstream msg;
        msg << "[" << evt.module << "][" << level << "] " << evt.message;
        logger->log(msg.str());
    };
    if (!ctx->logging.checkpoint_sink->bind("Checkpoint", checkpoint_formatter)) {
        ctx->logging.logger->log("[WARNING] Failed to bind Checkpoint module logger - save/load diagnostics may not appear in logs");
    }

    // Forward Activations module logs to training logger (Flash Attention diagnostics)
    ctx->logging.activations_sink = std::make_unique<GRIM::Logging::ModuleLogSink>();
    auto activations_formatter = [logger = ctx->logging.logger.get()](const GRIM::Logging::ModuleLogEvent& evt) {
        const char* level = "INFO";
        switch (evt.level) {
            case GRIM::Logging::ModuleLogLevel::Warning: level = "WARN"; break;
            case GRIM::Logging::ModuleLogLevel::Error: level = "ERR"; break;
            default: break;
        }
        std::ostringstream msg;
        msg << "[" << evt.module << "][" << level << "] " << evt.message;
        logger->log(msg.str());
    };
    if (!ctx->logging.activations_sink->bind("Activations", activations_formatter)) {
        ctx->logging.logger->log("[WARNING] Failed to bind Activations module logger - Flash Attention diagnostics may not appear in logs");
    }

    // Forward GuessCache module logs to training logger (GRIM-TS cache diagnostics)
    ctx->logging.guess_cache_sink = std::make_unique<GRIM::Logging::ModuleLogSink>();
    auto guess_cache_formatter = [logger = ctx->logging.logger.get()](const GRIM::Logging::ModuleLogEvent& evt) {
        const char* level = "INFO";
        switch (evt.level) {
            case GRIM::Logging::ModuleLogLevel::Warning: level = "WARN"; break;
            case GRIM::Logging::ModuleLogLevel::Error: level = "ERR"; break;
            default: break;
        }
        std::ostringstream msg;
        msg << "[" << evt.module << "][" << level << "] " << evt.message;
        logger->log(msg.str());
    };
    if (!ctx->logging.guess_cache_sink->bind("GuessCache", guess_cache_formatter)) {
        ctx->logging.logger->log("[WARNING] Failed to bind GuessCache module logger - cache diagnostics may not appear in logs");
    }
    
    // 2b. Auto-prepare training data/vocab from merged cache when needed.
    {
        const fs::path vocab_path = ctx->config.paths.vocab_path;
        const bool data_missing = !fs::exists(ctx->config.paths.data_path);
        const bool vocab_missing = vocab_path.empty() || !fs::exists(vocab_path);

        if (ctx->config.force_rebuild_vocab || data_missing || vocab_missing) {
            if (ctx->logging.logger) {
                ctx->logging.logger->log("[Phase1] Auto-preparing training data/vocab...");
            }
            GRIM::Config::GrimTextPaths grim_paths{};
            grim_paths.training_data = ctx->config.paths.data_path;
            grim_paths.vocab = ctx->config.paths.vocab_path;

            std::string training_path = ctx->config.paths.data_path;
            std::string vocab_path_str = ctx->config.paths.vocab_path;
            bool prepared = false;
            try {
                prepared = GRIM::PrepareTrainingDataFromCache(
                    grim_paths,
                    training_path,
                    vocab_path_str,
                    ctx->config.force_rebuild_vocab,
                    ctx->config.clear_merged_cache);
            } catch (const std::exception& e) {
                if (ctx->logging.logger) {
                    ctx->logging.logger->log(std::string("[Phase1] Auto-prepare exception: ") + e.what());
                }
                throw;
            }

            if (prepared) {
                ctx->config.paths.data_path = training_path;
                ctx->config.paths.vocab_path = vocab_path_str;
                if (ctx->logging.logger) {
                    ctx->logging.logger->log("[Phase1] Auto-prepare completed.");
                }
            } else {
                if (ctx->logging.logger) {
                    ctx->logging.logger->log("[Phase1] Auto-prepare failed.");
                }
                throw std::runtime_error(
                    "FATAL: training data/vocab missing and cache preparation failed");
            }
        }
    }

    // 3. Validate paths
    EmitModuleInfo(ModuleId::Training, "[Phase1] Validating paths...", 0);
    Internal::validatePaths(ctx->config.paths);
    EmitModuleInfo(ModuleId::Training, "[Phase1] ✓ All paths validated", 0);

    EmitModuleInfo(ModuleId::Training, "Configuration:", 0);
    EmitModuleInfo(ModuleId::Training, std::string("  Data: ") + ctx->config.paths.data_path, 0);
    EmitModuleInfo(ModuleId::Training, std::string("  Vocab: ") + ctx->config.paths.vocab_path, 0);
    EmitModuleInfo(ModuleId::Training, std::string("  Epochs: ") + std::to_string(ctx->config.hyperparameters.epochs), 0);
    EmitModuleInfo(ModuleId::Training, std::string("  Batch size: ") + std::to_string(ctx->config.hyperparameters.batch_size), 0);
    EmitModuleInfo(ModuleId::Training, std::string("  Learning rate: ") + std::to_string(ctx->config.hyperparameters.learning_rate), 0);
    
    // 4. Initialize tokenizer
    GRIM::Config::TokenizerConfig tok_cfg;
    if (auto snapshot = GRIM::Config::loadAiConfigSnapshot()) {
        if (snapshot->has_tokenizer) {
            tok_cfg = snapshot->tokenizer_config;
        }
    }
    ctx->tokenizer = Internal::initializeTokenizer(ctx->config.paths.vocab_path, tok_cfg, *ctx->logging.logger);
    
    ctx->logging.logger->log("[DEBUG] Starting training data load...");
    // 5. Load training data
    ctx->data = Internal::loadTrainingData(
        ctx->config.paths.data_path,
        ctx->config.max_seq_len,
        ctx->config.hyperparameters.min_seq_valid_tokens,
        ctx->config.sliding_window_stride,
        ctx->tokenizer,
        *ctx->logging.logger);
    
    // Get vocab size from data loader (authoritative source)
    GRMTDataLoader loader;
    if (!loader.load(ctx->config.paths.data_path)) {
        throw std::runtime_error("FATAL: Failed to reload training data for vocab size");
    }
    ctx->config.actual_vocab_size = loader.vocabSize();
    if (ctx->config.actual_vocab_size == 0) {
        throw std::runtime_error("FATAL: training data missing vocab_size; regenerate GRMT with tokenizer.totalVocabSize()");
    }
    if (ctx->config.actual_vocab_size < static_cast<uint32_t>(GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET)) {
        throw std::runtime_error("FATAL: training data vocab_size must include byte+atom ranges (>= 512)");
    }
    
    // CRITICAL: Validate vocab size matches tokenizer (detects stale .grmt files after atom encoding changes)
    const uint32_t tokenizer_vocab_size = ctx->tokenizer.totalVocabSize();
    if (!loader.validateVocabSize(tokenizer_vocab_size, std::cerr)) {
        throw std::runtime_error("FATAL: Vocab size mismatch between tokenizer and training data - see error above");
    }
    ctx->logging.logger->log("✓ Vocab size validated: tokenizer and training data match (" + 
                            std::to_string(tokenizer_vocab_size) + " tokens)");
    
    ctx->logging.logger->log("✓ Vocab size from training data: " + std::to_string(ctx->config.actual_vocab_size));
    
    // 6. Compute rare token scores
    ctx->data.sequence_rarity = Internal::computeRareTokenScores(
        ctx->data.train_views,
        ctx->config.actual_vocab_size,
        ctx->data.train_seqs.size(),
        *ctx->logging.logger);

    if (!ctx->data.val_views.empty()) {
        ctx->logging.logger->log("Computing validation rarity scores...");
        ctx->data.val_sequence_rarity = Internal::computeRareTokenScores(
            ctx->data.val_views,
            ctx->config.actual_vocab_size,
            ctx->data.val_seqs.size(),
            *ctx->logging.logger);
    }
    
    // 7. Harmonize hyperparameters
    GRIM::HyperParameters::DerivationContext hp_ctx;
    hp_ctx.train_sequence_count = static_cast<int>(ctx->data.train_seqs.size());
    hp_ctx.validation_interval = ctx->config.hyperparameters.validation_interval;
    
    auto hp_logger = [&](const std::string& msg) { ctx->logging.logger->log(msg); };
    ctx->derived_schedule = GRIM::HyperParameters::harmonizeTrainingHyperparameters(
        ctx->config.hyperparameters, hp_ctx, hp_logger);
    
    // 8. Initialize production-grade RNG system (BEFORE model init for Xavier seeding)
    ctx->rng = Internal::initializeRNG(ctx->config, *ctx->logging.logger);
    
    // 9. Initialize model (uses RNG init_seed for Xavier)
    try {
        ctx->model = Internal::initializeModel(
            ctx->config,
            ctx->config.actual_vocab_size,
            ctx->rng.init_seed,
            *ctx->logging.logger);
    } catch (const std::exception& e) {
        EmitModuleError(ModuleId::Training, std::string("FATAL: Model initialization failed: ") + e.what(), 0);
        throw;
    }
    
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
    
    // 10. Initialize optimizer
    ctx->optimizer = Internal::initializeOptimizer(*ctx->model, ctx->config, *ctx->logging.logger);
    
    // Token weighting removed - standard transformers use uniform loss weights
    
    // 10c. Issue #39 FIX: Allocate and initialize logit bias buffer for output bias correction
    // This prevents tokens like SPACE from having systematically higher pre-softmax activations.
    // The bias tracks EMA of mean logit per token and is subtracted before softmax.
    {
        auto& ts = ctx->model->getTrainingState();
        const int vocab_size = static_cast<int>(ctx->config.actual_vocab_size);
        const size_t bias_bytes = vocab_size * sizeof(float);
        
        // Allocate main bias buffer (stores running EMA)
        cudaError_t err = cudaMalloc(&ts.logit_bias, bias_bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc failed for logit_bias: ") + 
                                     cudaGetErrorString(err));
        }
        // Initialize to zero (no bias correction initially - let EMA build up)
        err = cudaMemsetAsync(ts.logit_bias, 0, bias_bytes, ts.stream_ctrl.getPrimaryStream());
        if (err != cudaSuccess) {
            cudaFree(ts.logit_bias);
            ts.logit_bias = nullptr;
            throw std::runtime_error(std::string("cudaMemset failed for logit_bias: ") + 
                                     cudaGetErrorString(err));
        }
        
        // Allocate scratch buffer for batch mean computation
        err = cudaMalloc(&ts.logit_bias_update, bias_bytes);
        if (err != cudaSuccess) {
            cudaFree(ts.logit_bias);
            ts.logit_bias = nullptr;
            throw std::runtime_error(std::string("cudaMalloc failed for logit_bias_update: ") + 
                                     cudaGetErrorString(err));
        }
        
        ts.logit_bias_count = vocab_size;
        ts.logit_bias_update_step = 0;
        
        ctx->logging.logger->log("✓ Logit bias buffer allocated: " + 
                                 std::to_string(2 * bias_bytes / 1024) + " KB total (EMA + scratch)");
        ctx->logging.logger->log("  Issue #39: Output logit bias correction ENABLED");
    }
    
    // 11. Initialize telemetry lattice
    EmitModuleInfo(ModuleId::Training, "[Phase1] Initializing telemetry lattice...", 0);
    ctx->telemetry.config.num_levels = 8;  // k ∈ [0,7]: strides [1,2,4,8,16,32,64,128]
    ctx->telemetry.config.num_streams = 5; // LOSS, GRAD_NORM_MEAN, GRAD_NORM_MAX, LEARNING_RATE, TOKENS_PER_BATCH
    ctx->telemetry.config.hyperparams.beta_mu = 0.95f;
    ctx->telemetry.config.hyperparams.beta_a = 0.995f;
    ctx->telemetry.config.hyperparams.beta_delta = 0.90f;
    ctx->telemetry.config.hyperparams.beta_r = 0.85f;
    ctx->telemetry.config.hyperparams.beta_run = 0.80f;
    ctx->telemetry.config.hyperparams.beta_v = 0.90f;
    ctx->telemetry.config.hyperparams.k_out0 = 2.5f;
    ctx->telemetry.config.hyperparams.alpha_v = 1.5f;
    ctx->telemetry.config.hyperparams.epsilon = 1e-7f;
    ctx->telemetry.config.hyperparams.strict_mode = true; // Fail loud (Rule 20: no compatibility shims)
    ctx->telemetry.config.stream = ctx->model->getTrainingState().stream_ctrl.getPrimaryStream(); // Use same stream as training
    
    ctx->telemetry.lattice = GRIM::Telemetry::initTelemetryLattice(ctx->telemetry.config);
    if (!ctx->telemetry.lattice) {
        throw std::runtime_error("FATAL: Failed to initialize telemetry lattice");
    }
    ctx->logging.logger->log("✓ Telemetry lattice: 8 levels, 5 streams, GPU-resident");
    
    // 11b. Initialize telemetry control (GPU-native kernel-based control)
    EmitModuleInfo(ModuleId::Training, "[Phase1] Initializing telemetry control...", 0);
    // Rule 20: Use model's actual token budget (already computed during cache allocation)
    const uint32_t token_budget = ctx->model->getConfig().max_tokens_per_batch;
    if (token_budget == 0) {
        throw std::runtime_error("FATAL: model token_budget is 0 - cache allocation failed");
    }
    ctx->telemetry.control_config.reference_tokens = static_cast<float>(token_budget);
    ctx->telemetry.control_config.reference_seq_len = static_cast<float>(ctx->config.max_seq_len);
    ctx->logging.logger->log("[DEBUG] ctx->config.max_seq_len = " + std::to_string(ctx->config.max_seq_len));
    
    // TelemetryControl configuration - load from global config
    const auto& hp = ctx->config.hyperparameters;
    const bool telemetry_control_enabled = hp.telemetry_control_enabled;
    
    // Populate plateau noise config from TrainingParams (loaded from ai_config.json)
    ctx->telemetry.control_config.plateau_noise_enabled = hp.telemetry_plateau_noise_enabled;
    ctx->telemetry.control_config.plateau_noise_patience = hp.telemetry_plateau_noise_patience;
    ctx->telemetry.control_config.plateau_noise_variance_threshold = hp.telemetry_plateau_noise_variance_threshold;
    ctx->telemetry.control_config.plateau_noise_std = hp.telemetry_plateau_noise_std;
    ctx->telemetry.control_config.plateau_noise_proportional = hp.telemetry_plateau_noise_proportional;
    ctx->telemetry.control_config.plateau_noise_cooldown = hp.telemetry_plateau_noise_cooldown;
    ctx->telemetry.control_config.plateau_noise_max_per_epoch = hp.telemetry_plateau_noise_max_per_epoch;

    ctx->telemetry.control_config.spike_mild_threshold = hp.telemetry_spike_mild_threshold;
    ctx->telemetry.control_config.spike_moderate_threshold = hp.telemetry_spike_moderate_threshold;
    ctx->telemetry.control_config.spike_severe_threshold = hp.telemetry_spike_severe_threshold;
    ctx->telemetry.control_config.moderate_grad_scale = hp.telemetry_moderate_grad_scale;
    ctx->telemetry.control_config.moderate_cooldown_extension = hp.telemetry_moderate_cooldown_extension;
    ctx->telemetry.control_config.min_grad_for_nonzero_loss = hp.telemetry_min_grad_for_nonzero_loss;
    ctx->telemetry.control_config.loss_threshold_for_grad_check = hp.telemetry_loss_threshold_for_grad_check;
    ctx->telemetry.control_config.max_consecutive_zero_grad_steps = hp.telemetry_max_consecutive_zero_grad_steps;
    ctx->telemetry.control_config.seq_len_regime_change_threshold = hp.telemetry_seq_len_regime_change_threshold;
    ctx->telemetry.control_config.regime_change_suppression_steps = hp.telemetry_regime_change_suppression_steps;
    ctx->telemetry.control_config.volatility_damping_threshold = hp.telemetry_volatility_damping_threshold;
    ctx->telemetry.control_config.max_volatility_damping = hp.telemetry_max_volatility_damping;
    ctx->telemetry.control_config.gradient_decay_threshold = hp.telemetry_gradient_decay_threshold;
    ctx->telemetry.control_config.max_decay_boost = hp.telemetry_max_decay_boost;
    ctx->telemetry.control_config.progress_boost_threshold = hp.telemetry_progress_boost_threshold;
    ctx->telemetry.control_config.max_progress_boost = hp.telemetry_max_progress_boost;
    ctx->telemetry.control_config.outlier_frequency_trigger = hp.telemetry_outlier_frequency_trigger;
    ctx->telemetry.control_config.outlier_persistence_trigger = hp.telemetry_outlier_persistence_trigger;
    ctx->telemetry.control_config.anchor_drift_sigma_multiplier = hp.telemetry_anchor_drift_sigma_multiplier;
    ctx->telemetry.control_config.soft_restart_cooldown_steps = hp.telemetry_soft_restart_cooldown_steps;
    ctx->telemetry.control_config.warmup_steps = hp.telemetry_warmup_steps;
    ctx->telemetry.control_config.baseline_stabilization_steps = hp.telemetry_baseline_stabilization_steps;
    ctx->telemetry.control_config.verbose_logging = hp.telemetry_verbose_logging;
    ctx->telemetry.control_config.fail_loud_on_accumulation_bug = hp.telemetry_fail_loud_on_accumulation_bug;
    
    if (telemetry_control_enabled) {
        // ENABLED: Use configured thresholds for gradient spike detection and interventions
        ctx->logging.logger->log("Telemetry control: ENABLED");
        ctx->logging.logger->log("  Plateau noise: " + std::string(hp.telemetry_plateau_noise_enabled ? "ON" : "OFF") +
                                " (patience=" + std::to_string(hp.telemetry_plateau_noise_patience) +
                                ", variance_threshold=" + std::to_string(hp.telemetry_plateau_noise_variance_threshold) +
                                ", noise_std=" + std::to_string(hp.telemetry_plateau_noise_std) + ")");
    } else {
        // DISABLED: Set ALL thresholds to prevent ANY interventions
        ctx->logging.logger->log("Telemetry control: DISABLED (monitoring only)");
        ctx->telemetry.control_config.spike_severe_threshold = 1000.0f;   // Never trigger severe
        ctx->telemetry.control_config.spike_moderate_threshold = 1000.0f; // Never trigger moderate (WAS MISSING!)
        ctx->telemetry.control_config.spike_mild_threshold = 1000.0f;     // Never trigger mild
        ctx->telemetry.control_config.max_volatility_damping = 1.0f;      // No damping
        ctx->telemetry.control_config.gradient_decay_threshold = 0.0f;    // No decay detection
        ctx->telemetry.control_config.max_decay_boost = 1.0f;             // No boosting
        ctx->telemetry.control_config.plateau_noise_enabled = false;      // No plateau noise
        ctx->telemetry.control_config.moderate_grad_scale = 1.0f;         // No gradient scaling
    }
    
    ctx->telemetry.controller = std::make_unique<GRIM::Telemetry::TelemetryControl>(ctx->telemetry.control_config);
    ctx->telemetry.controller->initGPU();  // No stream caching (Rule 22)
    ctx->logging.logger->log("✓ Telemetry control: GPU-native (plateau noise injection built-in)");
    
    // 12. Check GPU memory
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    EmitModuleInfo(ModuleId::Training, 
        std::string("GPU Memory: ") + std::to_string(free_mem / (1024*1024)) + " MB free / " + 
        std::to_string(total_mem / (1024*1024)) + " MB total", 0);
    
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
