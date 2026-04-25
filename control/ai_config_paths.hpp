#pragma once

//======================================================//
// AI CONFIG ORGANIZATION
//
// This file defines the C++ structs that parse ai_config.json.
// The JSON→C++ mapping is:
//
// JSON Section                → C++ Struct
// ----------------------------------------
// paths.grim_text            → GrimTextPaths
// training.config            → TrainingHyperparameters (incl. execution_block, scratch_blocks, …)
// tokenizer                  → TokenizerConfig
// data_collection            → DataCollectionConfig
//
// RULE: All runtime defaults in TrainingHyperparameters MUST
// match compile-time defaults in HyperParameters_GPU.hpp.
// The JSON overrides those defaults at runtime.
//
// For compile-time constants (CUDA blocks, epsilons, etc.),
// see HyperParameters_GPU.hpp - DO NOT duplicate them here.
//======================================================//


#include <string>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <optional>
#include <type_traits>
#include <map>
#include <set>
#include <sstream>
#include <vector>

// TrainingHyperparameters and its loaders moved to HyperParameters_GPU.hpp
// Opt in to the host-types block (TrainingHyperparameters + JSON loaders).
#define GRIM_HP_HOST_TYPES_REQUIRED 1
// (Phase A refactor: HyperParameters_GPU.hpp is the single source of truth.)
#include "../resources/models/GRIM-text/Shared/HyperParameters/HyperParameters_GPU.hpp"


#ifdef _WIN32
    #include <windows.h>  // GetModuleFileNameW, etc.
#elif defined(__APPLE__)
    #include <mach-o/dyld.h>
#else
    #include <unistd.h>
    #include <limits.h>
#endif

namespace GRIM {
namespace Config {

/**
 * @brief Load GRIM-text paths from ai_config.json
 * 
 * This function reads the centralized ai_config.json file that is written by
 * the UI Training Panel in GRIM.exe and provides the absolute paths for:
 * - vocab: Vocabulary file path
 * - model: Model file path
 * - training_data: Training data file path (.grmt)
 * - checkpoints: Checkpoints directory path
 * - logs: Logs directory path
 * 
 * All components (training_control_server, train_gpu, data_collection_server)
 * should use this function to ensure they read the same paths that the UI writes.
 * 
 * @param configPath Path to ai_config.json (defaults to "./ai_config.json")
 * @return true if paths were loaded successfully, false otherwise
 */
struct GrimTextPaths {
    std::string vocab;
    std::string model;
    std::string training_data;
    std::string checkpoints;
    std::string collected;
    std::string directory_collection;
    std::string verified;
    std::string logs;
    std::string training_status;
    std::string merge_checkpoints_exe;
    std::string collector_log;
    std::string source_config;  // Path to source_data.json for data collection
    std::string model_store;   // Path to model store directory for per-model configs
    
    bool isValid() const {
        // At minimum, we need vocab and training_data to do any work
        return !vocab.empty() && !training_data.empty();
    }
    
    void printPaths() const {
        std::cout << "[Config] GRIM-text paths from ai_config.json:" << std::endl;
        std::cout << "  vocab: " << (vocab.empty() ? "(not set)" : vocab) << std::endl;
        std::cout << "  model: " << (model.empty() ? "(not set)" : model) << std::endl;
        std::cout << "  training_data: " << (training_data.empty() ? "(not set)" : training_data) << std::endl;
        std::cout << "  checkpoints: " << (checkpoints.empty() ? "(not set)" : checkpoints) << std::endl;
        std::cout << "  collected: " << (collected.empty() ? "(not set)" : collected) << std::endl;
        std::cout << "  directory_collection: " << (directory_collection.empty() ? "(not set)" : directory_collection) << std::endl;
        std::cout << "  verified: " << (verified.empty() ? "(not set)" : verified) << std::endl;
        std::cout << "  logs: " << (logs.empty() ? "(not set)" : logs) << std::endl;
        std::cout << "  training_status: " << (training_status.empty() ? "(not set)" : training_status) << std::endl;
        std::cout << "  merge_checkpoints_exe: " << (merge_checkpoints_exe.empty() ? "(not set)" : merge_checkpoints_exe) << std::endl;
        std::cout << "  collector_log: " << (collector_log.empty() ? "(not set)" : collector_log) << std::endl;
        std::cout << "  source_config: " << (source_config.empty() ? "(not set)" : source_config) << std::endl;
        std::cout << "  model_store: " << (model_store.empty() ? "(not set)" : model_store) << std::endl;
    }
};

/**
 * @brief Data collection configuration
 * 
 * Controls behavior of data collection pipeline (collection, verification, merging)
 */
struct DataCollectionConfig {
    bool clear_merged_cache_on_merge = false;  // Whether to clear merged_verified_cache.jsonl on merge
    int max_new_entries_per_run = 5000;        // Global limit: stop after collecting this many NEW entries
};




/**
 * @brief Tokenizer configuration from ai_config.json
 * 
 * This loads the tokenizer configuration (vocab_size, max_length, model_type, etc.)
 * from the tokenizer section of ai_config.json.
 * 
 * Note: This mirrors GRIM::TokenizerConfig from Tokenizer_GPU.hpp
 * All values MUST be loaded from ai_config.json - no hardcoded defaults
 */
struct TokenizerConfig {
    int vocab_size = 50000;  // Target vocab size for training new tokenizer (actual size comes from vocab.bin)
    int max_vocab_size = 0;  // Hard cap on loaded vocab (0 = no cap, >0 = keep top-K most frequent tokens)
    int max_length = 8192;
    int min_subword_freq = 3;  // Minimum frequency for subwords to be included in vocab
    bool prune_during_mining = false;  // Enable memory pruning during subword mining (disable if RAM is plentiful)
    bool enable_parallel_subword_mining = true;  // Parallelize subword mining during vocab training
    int subword_mining_workers = 0;  // 0 = auto, >0 fixed worker count
    size_t subword_mining_max_bytes = 0;  // 0 = use tokenizer default ceap
    std::string model_type = "unibytes";
    std::vector<std::string> special_tokens = {"<pad>", "<unk>", "<s>", "</s>"};
    bool add_bos = true;
    bool add_eos = true;
    std::string unk_token = "<unk>";
    std::string pad_token = "<pad>";
    std::string bos_token = "<s>";
    std::string eos_token = "</s>";
    bool enable_nfkc_normalization = true;
    bool enable_lowercasing = true;
    bool enable_parallel_tokenization = true;
    int parallel_threshold = 1000;
    bool enable_byte_fallback = true;
    uint32_t expected_checksum = 0;
    bool save_text_vocab = true;  // Also save human-readable .txt alongside .bin
    float vocab_score_multiplier = 1.0f;  // Multiply all vocab scores by this value on save (experiment knob)
};

struct AiConfigSnapshot {
    std::filesystem::path config_path;
    nlohmann::json document;
    GrimTextPaths grim_paths;
    TrainingHyperparameters hyperparameters;
    TokenizerConfig tokenizer_config;
    DataCollectionConfig data_collection_config;
    bool has_grim_paths = false;
    bool has_training = false;
    bool has_tokenizer = false;
    bool has_data_collection = false;
};

inline std::optional<AiConfigSnapshot> loadAiConfigSnapshot(const std::string& configPath = "ai_config.json");

namespace detail {

inline std::optional<std::filesystem::path> resolveAiConfigPath(const std::string& configPath) {
    namespace fs = std::filesystem;
    fs::path candidate(configPath);
    if (!configPath.empty() && fs::exists(candidate)) {
        return fs::absolute(candidate);
    }

    fs::path defaultCandidate = fs::current_path() / "ai_config.json";
    if (fs::exists(defaultCandidate)) {
        return defaultCandidate;
    }

    fs::path searchPath = fs::current_path();
    for (int i = 0; i < 5 && searchPath.has_parent_path(); ++i) {
        searchPath = searchPath.parent_path();
        fs::path parentCandidate = searchPath / "ai_config.json";
        if (fs::exists(parentCandidate)) {
            return parentCandidate;
        }
    }

    return std::nullopt;
}

inline std::filesystem::path resolveGrimRoot() {
    namespace fs = std::filesystem;
    
#ifdef GRIM_ROOT_DIR
    fs::path root = fs::path(GRIM_ROOT_DIR);
    if (fs::exists(root)) {
        return root;
    }
#endif

    // Try to find GRIM root by walking up from current directory or executable location
    fs::path searchPath = fs::current_path();
    
    // Also try executable directory
    fs::path exeDir;
#ifdef _WIN32
    char buffer[MAX_PATH];
    if (GetModuleFileNameA(nullptr, buffer, MAX_PATH)) {
        exeDir = fs::path(buffer).parent_path();
    }
#elif defined(__APPLE__)
    char buffer[PATH_MAX];
    uint32_t size = sizeof(buffer);
    if (_NSGetExecutablePath(buffer, &size) == 0) {
        exeDir = fs::path(buffer).parent_path();
    }
#else
    char buffer[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len > 0) {
        buffer[len] = '\0';
        exeDir = fs::path(buffer).parent_path();
    }
#endif
    
    std::vector<fs::path> searchPaths = {searchPath, exeDir};
    
    for (auto& base : searchPaths) {
        if (base.empty()) continue;
        
        fs::path probe = base;
        for (int i = 0; i < 10 && probe.has_parent_path(); ++i) {
            if (fs::exists(probe / "control") && fs::exists(probe / "resources")) {
                return probe;
            }
            if (!probe.has_parent_path()) break;
            probe = probe.parent_path();
        }
    }
    
    // Fallback: return current directory
    return fs::current_path();
}

inline bool populateGrimTextPathsFromConfig(const nlohmann::json& config, GrimTextPaths& paths) {
    if (!config.contains("paths")) {
        std::cerr << "[Config] WARNING: ai_config.json does not contain 'paths' section" << std::endl;
        return false;
    }

    const auto& pathsNode = config["paths"];
    if (!pathsNode.contains("grim_text")) {
        std::cerr << "[Config] WARNING: ai_config.json does not contain 'paths.grim_text' section" << std::endl;
        return false;
    }

    const auto& grimTextPaths = pathsNode["grim_text"];
    if (!grimTextPaths.is_object()) {
        std::cerr << "[Config] WARNING: 'paths.grim_text' is not an object" << std::endl;
        return false;
    }

    // Get GRIM root for resolving relative paths
    std::filesystem::path grimRoot = resolveGrimRoot();
    
    auto assignIfPresent = [&](const char* key, std::string& field) {
        if (grimTextPaths.contains(key) && grimTextPaths[key].is_string()) {
            std::string pathStr = grimTextPaths[key].get<std::string>();
            std::filesystem::path path(pathStr);
            
            // If path is relative, resolve it relative to GRIM root
            if (path.is_relative()) {
                field = (grimRoot / path).string();
            } else {
                // Path is already absolute, use as-is
                field = pathStr;
            }
        }
    };

    assignIfPresent("vocab", paths.vocab);
    assignIfPresent("model", paths.model);
    assignIfPresent("training_data", paths.training_data);
    assignIfPresent("checkpoints", paths.checkpoints);
    assignIfPresent("collected", paths.collected);
    assignIfPresent("directory_collection", paths.directory_collection);
    assignIfPresent("verified", paths.verified);
    assignIfPresent("logs", paths.logs);
    assignIfPresent("training_status", paths.training_status);
    assignIfPresent("merge_checkpoints_exe", paths.merge_checkpoints_exe);
    assignIfPresent("collector_log", paths.collector_log);
    assignIfPresent("source_config", paths.source_config);
    assignIfPresent("model_store", paths.model_store);

    return true;
}


/**
 * @brief Check if a nested JSON path exists
 * @param json The root JSON object
 * @param path Dot-separated path (e.g., "loss.focal.enabled")
 * @return true if path exists
 */
inline bool jsonPathExists(const nlohmann::json& json, const std::string& path) {
    const nlohmann::json* current = &json;
    std::istringstream stream(path);
    std::string segment;
    
    while (std::getline(stream, segment, '.')) {
        if (!current->is_object() || !current->contains(segment)) {
            return false;
        }
        current = &(*current)[segment];
    }
    return true;
}




// Compute formula-derived hyperparameters from base fields.
// Called AFTER applyTrainingConfigObject() so base fields are populated from JSON.
// These ALWAYS overwrite — they are mathematical derivations, not defaults.
// Rule 20: throws if any base field needed for derivation is invalid.

/// Derive warmup_steps and dependent fields once estimated_total_steps is known (Phase2).
/// Must be called after deriveComputedHyperparameters() and before the training loop.


inline bool populateDataCollectionConfigFromConfig(const nlohmann::json& config, DataCollectionConfig& dc_config) {
    if (!config.contains("data_collection")) {
        return false;
    }

    const auto& dc = config["data_collection"];
    if (!dc.is_object()) {
        return false;
    }

    assignTrainingField(dc_config.clear_merged_cache_on_merge, dc, "clear_merged_cache_on_merge");
    assignTrainingField(dc_config.max_new_entries_per_run, dc, "max_new_entries_per_run");

    return true;
}

inline bool populateTokenizerConfigFromConfig(const nlohmann::json& config, TokenizerConfig& tokenizer_config, TrainingHyperparameters& hyperparameters) {
    if (!config.contains("tokenizer")) {
        return false;
    }

    const auto& tok = config["tokenizer"];
    if (!tok.is_object()) {
        return false;
    }

    assignTrainingField(tokenizer_config.vocab_size, tok, "vocab_size");
    assignTrainingField(tokenizer_config.max_vocab_size, tok, "max_vocab_size");
    assignTrainingField(tokenizer_config.max_length, tok, "max_length");
    assignTrainingField(tokenizer_config.min_subword_freq, tok, "min_subword_freq");
    assignTrainingField(tokenizer_config.prune_during_mining, tok, "prune_during_mining");
    assignTrainingField(tokenizer_config.enable_parallel_subword_mining, tok, "enable_parallel_subword_mining");
    assignTrainingField(tokenizer_config.subword_mining_workers, tok, "subword_mining_workers");
    assignTrainingField(tokenizer_config.subword_mining_max_bytes, tok, "subword_mining_max_bytes");
    assignTrainingField(tokenizer_config.model_type, tok, "model_type");
    assignTrainingField(tokenizer_config.add_bos, tok, "add_bos");
    assignTrainingField(tokenizer_config.add_eos, tok, "add_eos");
    assignTrainingField(tokenizer_config.unk_token, tok, "unk_token");
    assignTrainingField(tokenizer_config.pad_token, tok, "pad_token");
    assignTrainingField(tokenizer_config.bos_token, tok, "bos_token");
    assignTrainingField(tokenizer_config.eos_token, tok, "eos_token");
    assignTrainingField(tokenizer_config.enable_nfkc_normalization, tok, "enable_nfkc_normalization");
    assignTrainingField(tokenizer_config.enable_lowercasing, tok, "enable_lowercasing");
    assignTrainingField(tokenizer_config.enable_parallel_tokenization, tok, "enable_parallel_tokenization");
    assignTrainingField(tokenizer_config.parallel_threshold, tok, "parallel_threshold");
    assignTrainingField(tokenizer_config.enable_byte_fallback, tok, "enable_byte_fallback");
    assignTrainingField(tokenizer_config.expected_checksum, tok, "expected_checksum");
    assignTrainingField(tokenizer_config.save_text_vocab, tok, "save_text_vocab");
    assignTrainingField(tokenizer_config.vocab_score_multiplier, tok, "vocab_score_multiplier");

    // Backward compatibility with configs that only set max_vocab_size:
    // treat it as the target vocab size when vocab_size is omitted.
    if (!tok.contains("vocab_size") &&
        tok.contains("max_vocab_size") &&
        tokenizer_config.max_vocab_size > 0) {
        tokenizer_config.vocab_size = tokenizer_config.max_vocab_size;
    }

    // Scratch block reasoning configuration - load into hyperparameters (single source of truth)
    if (tok.contains("scratch_block_reasoning") && tok["scratch_block_reasoning"].is_object()) {
        const auto& sbr = tok["scratch_block_reasoning"];
        assignTrainingField(hyperparameters.tokenizer_enable_scratch_block_reasoning, sbr, "enabled");
        assignTrainingField(hyperparameters.tokenizer_detect_numbers, sbr, "detect_numbers");
    }

    if (tok.contains("special_tokens") && tok["special_tokens"].is_array()) {
        tokenizer_config.special_tokens.clear();
        for (const auto& token : tok["special_tokens"]) {
            if (token.is_string()) {
                tokenizer_config.special_tokens.push_back(token.get<std::string>());
            }
        }
    }

    return true;
}

} // namespace detail

/// Derive warmup_steps and dependent fields once estimated_total_steps is known (Phase2).
/// Must be called after populateTrainingHyperparametersFromConfig() and before the training loop.

inline std::optional<AiConfigSnapshot> loadAiConfigSnapshot(const std::string& configPath) {
    try {
        auto resolved_path = detail::resolveAiConfigPath(configPath);
        if (!resolved_path) {
            std::cerr << "[Config] ERROR: ai_config.json not found at: " << configPath << std::endl;
            std::cerr << "[Config]        Also searched current directory and parent directories" << std::endl;
            return std::nullopt;
        }

        std::ifstream configFile(*resolved_path);
        if (!configFile.is_open()) {
            std::cerr << "[Config] ERROR: Could not open ai_config.json at: " << *resolved_path << std::endl;
            return std::nullopt;
        }

        nlohmann::json config;
        configFile >> config;

        AiConfigSnapshot snapshot;
        snapshot.config_path = *resolved_path;
        snapshot.document = std::move(config);
        snapshot.has_grim_paths = detail::populateGrimTextPathsFromConfig(snapshot.document, snapshot.grim_paths);
        snapshot.has_training = populateTrainingHyperparametersFromConfig(snapshot.document, snapshot.hyperparameters);
        snapshot.has_tokenizer = detail::populateTokenizerConfigFromConfig(snapshot.document, snapshot.tokenizer_config, snapshot.hyperparameters);
        snapshot.has_data_collection = detail::populateDataCollectionConfigFromConfig(snapshot.document, snapshot.data_collection_config);
        return snapshot;
    } catch (const std::exception& e) {
        std::cerr << "[Config] ERROR: Exception loading ai_config.json: " << e.what() << std::endl;
        return std::nullopt;
    }
}

inline bool loadGrimTextPaths(GrimTextPaths& paths, const std::string& configPath = "ai_config.json") {
    auto snapshot = loadAiConfigSnapshot(configPath);
    if (!snapshot) {
        return false;
    }

    if (!snapshot->has_grim_paths) {
        return false;
    }

    paths = snapshot->grim_paths;

    // Avoid spamming logs when polled frequently (e.g., from UI timers)
    static std::string lastPrintedSignature;
    const std::string signature =
        snapshot->config_path.string() + "|" +
        paths.vocab + "|" + paths.model + "|" + paths.training_data + "|" +
        paths.checkpoints + "|" + paths.collected + "|" + paths.directory_collection + "|" + paths.verified + "|" +
        paths.logs + "|" + paths.training_status + "|" + paths.merge_checkpoints_exe +
        "|" + paths.collector_log + "|" + paths.source_config +
        "|" + paths.model_store;

    if (signature != lastPrintedSignature) {
        std::cout << "[Config] Loading GRIM-text paths from: " << snapshot->config_path << std::endl;
        paths.printPaths();
        lastPrintedSignature = signature;
    }

    return paths.isValid();
}


inline bool loadTokenizerConfig(TokenizerConfig& config, const std::string& configPath = "ai_config.json") {
    auto snapshot = loadAiConfigSnapshot(configPath);
    if (!snapshot) {
        return false;
    }

    if (!snapshot->has_tokenizer) {
        return false;
    }

    config = snapshot->tokenizer_config;
    std::cout << "[Config] Loaded tokenizer config from: " << snapshot->config_path << std::endl;
    std::cout << "  vocab_size: " << config.vocab_size << std::endl;
    std::cout << "  max_vocab_size: " << config.max_vocab_size << std::endl;
    std::cout << "  max_length: " << config.max_length << std::endl;
    std::cout << "  enable_parallel_subword_mining: "
              << (config.enable_parallel_subword_mining ? "true" : "false") << std::endl;
    std::cout << "  subword_mining_workers: " << config.subword_mining_workers << std::endl;
    std::cout << "  subword_mining_max_bytes: " << config.subword_mining_max_bytes << std::endl;
    std::cout << "  model_type: " << config.model_type << std::endl;
    std::cout << "  add_bos: " << (config.add_bos ? "true" : "false") << std::endl;
    std::cout << "  add_eos: " << (config.add_eos ? "true" : "false") << std::endl;
    std::cout << "  enable_lowercasing: " << (config.enable_lowercasing ? "true" : "false") << std::endl;
    std::cout << "  enable_nfkc_normalization: " << (config.enable_nfkc_normalization ? "true" : "false") << std::endl;
    std::cout << "  enable_parallel_tokenization: " << (config.enable_parallel_tokenization ? "true" : "false") << std::endl;
    std::cout << "  parallel_threshold: " << config.parallel_threshold << std::endl;
    std::cout << "  special_tokens: [";
    for (size_t i = 0; i < config.special_tokens.size(); ++i) {
        std::cout << "\"" << config.special_tokens[i] << "\"";
        if (i + 1 < config.special_tokens.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    return true;
}

inline bool loadDataCollectionConfig(DataCollectionConfig& config, const std::string& configPath = "ai_config.json") {
    auto snapshot = loadAiConfigSnapshot(configPath);
    if (!snapshot) {
        return false;
    }

    if (!snapshot->has_data_collection) {
        return false;
    }

    config = snapshot->data_collection_config;
    return true;
}

/**
 * @brief Get the training status file path
 * 
 * This returns the path where train_gpu.exe writes its real-time status
 * and where training_control_server.exe reads it from.
 * 
 * Loads from ai_config.json paths.grim_text.training_status if available,
 * otherwise falls back to automatic discovery.
 * 
 * @return Absolute path to training_status.fb
 */
inline std::string getTrainingStatusFilePath() {
    // First, try to load from ai_config.json
    GrimTextPaths paths;
    if (loadGrimTextPaths(paths) && !paths.training_status.empty()) {
        return paths.training_status;
    }
    
    // Fallback: Status file lives in the training directory
    // This ensures train_gpu.exe and training_control_server.exe can both find it
    std::filesystem::path grimRoot = std::filesystem::current_path();
    
    // Walk up to find GRIM root (has both 'control' and 'resources' directories)
    for (int i = 0; i < 5 && grimRoot.has_parent_path(); ++i) {
        if (std::filesystem::exists(grimRoot / "control") && 
            std::filesystem::exists(grimRoot / "resources")) {
            break;
        }
        grimRoot = grimRoot.parent_path();
    }
    
    std::filesystem::path statusPath = grimRoot / "resources" / "models" / "GRIM-text" / "training" / "training_status.fb";
    return statusPath.string();
}

/**
 * @brief Get the checkpoints directory path
 * 
 * This returns the directory where data collection checkpoints are stored.
 * 
 * Loads from ai_config.json paths.grim_text.checkpoints if available,
 * otherwise falls back to automatic discovery.
 * 
 * @return Absolute path to checkpoints directory
 */
inline std::string getCheckpointDir() {
    // First, try to load from ai_config.json
    GrimTextPaths paths;
    if (loadGrimTextPaths(paths) && !paths.checkpoints.empty()) {
        return paths.checkpoints;
    }
    
    // Fallback: Checkpoint directory
    std::filesystem::path grimRoot = std::filesystem::current_path();
    
    // Walk up to find GRIM root (has both 'control' and 'resources' directories)
    for (int i = 0; i < 5 && grimRoot.has_parent_path(); ++i) {
        if (std::filesystem::exists(grimRoot / "control") && 
            std::filesystem::exists(grimRoot / "resources")) {
            break;
        }
        grimRoot = grimRoot.parent_path();
    }
    
    std::filesystem::path checkpointPath = grimRoot / "resources" / "models" / "GRIM-text" / "checkpoints";
    return checkpointPath.string();
}

/**
 * Get the collector log file path from ai_config.json.
 * Falls back to automatic discovery if not in config.
 */
inline std::string getCollectorLogPath() {
    // First, try to load from ai_config.json
    GrimTextPaths paths;
    if (loadGrimTextPaths(paths) && !paths.collector_log.empty()) {
        return paths.collector_log;
    }
    
    // Fallback: Collector log path
    std::filesystem::path grimRoot = std::filesystem::current_path();
    
    // Walk up to find GRIM root (has both 'control' and 'resources' directories)
    for (int i = 0; i < 5 && grimRoot.has_parent_path(); ++i) {
        if (std::filesystem::exists(grimRoot / "control") && 
            std::filesystem::exists(grimRoot / "resources")) {
            break;
        }
        grimRoot = grimRoot.parent_path();
    }
    
    std::filesystem::path logPath = grimRoot / "resources" / "models" / "GRIM-text" / "training" / "logs" / "collector.log";
    return logPath.string();
}

/**
 * Get the collected data directory from ai_config.json.
 * Falls back to automatic discovery if not in config.
 */
inline std::string getCollectedDir() {
    // First, try to load from ai_config.json
    GrimTextPaths paths;
    if (loadGrimTextPaths(paths) && !paths.collected.empty()) {
        return paths.collected;
    }
    
    // Fallback: Collected data directory
    std::filesystem::path grimRoot = std::filesystem::current_path();
    
    // Walk up to find GRIM root (has both 'control' and 'resources' directories)
    for (int i = 0; i < 5 && grimRoot.has_parent_path(); ++i) {
        if (std::filesystem::exists(grimRoot / "control") && 
            std::filesystem::exists(grimRoot / "resources")) {
            break;
        }
        grimRoot = grimRoot.parent_path();
    }
    
    std::filesystem::path collectedPath = grimRoot / "resources" / "models" / "GRIM-text" / "data" / "collected";
    return collectedPath.string();
}

/**
 * Get the verified data directory from ai_config.json.
 * Falls back to automatic discovery if not in config.
 */
inline std::string getVerifiedDir() {
    // First, try to load from ai_config.json
    GrimTextPaths paths;
    if (loadGrimTextPaths(paths) && !paths.verified.empty()) {
        return paths.verified;
    }
    
    // Fallback: Verified data directory
    std::filesystem::path grimRoot = std::filesystem::current_path();
    
    // Walk up to find GRIM root (has both 'control' and 'resources' directories)
    for (int i = 0; i < 5 && grimRoot.has_parent_path(); ++i) {
        if (std::filesystem::exists(grimRoot / "control") && 
            std::filesystem::exists(grimRoot / "resources")) {
            break;
        }
        grimRoot = grimRoot.parent_path();
    }
    
    std::filesystem::path verifiedPath = grimRoot / "resources" / "models" / "GRIM-text" / "data" / "verified";
    return verifiedPath.string();
}

/**
 * @brief Get actual vocab size from vocab.bin file
 * 
 * Loads the tokenizer and returns its actual vocabulary size.
 * This is the authoritative source for vocab size, not ai_config.json.
 * 
 * @param vocabPath Path to vocab.bin file (optional, auto-detects if not provided)
 * @return Vocab size from tokenizer, or 0 on error
 */
inline uint32_t getActualVocabSize(const std::string& vocabPath = "") {
    try {
        std::string path = vocabPath;
        
        // Auto-detect vocab path if not provided
        if (path.empty()) {
            GrimTextPaths paths;
            if (loadGrimTextPaths(paths) && !paths.vocab.empty()) {
                path = paths.vocab;
            } else {
                // Fallback
                std::filesystem::path grimRoot = std::filesystem::current_path();
                for (int i = 0; i < 5 && grimRoot.has_parent_path(); ++i) {
                    if (std::filesystem::exists(grimRoot / "control") && 
                        std::filesystem::exists(grimRoot / "resources")) {
                        break;
                    }
                    grimRoot = grimRoot.parent_path();
                }
                path = (grimRoot / "resources" / "models" / "GRIM-text" / "training" / "data" / "vocab.bin").string();
            }
        }
        
        // Load tokenizer to get actual vocab size
        // Note: This requires linking against the tokenizer library
        // For a header-only solution, we could parse the binary format directly
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            std::cerr << "[Config] Failed to open vocab file: " << path << std::endl;
            return 0;
        }
        
        // Read GMKT header format (version 2)
        char magic[4];
        file.read(magic, 4);
        if (magic[0] != 'K' || magic[1] != 'T' || magic[2] != 'M' || magic[3] != 'G') {
            std::cerr << "[Config] Invalid vocab file magic" << std::endl;
            return 0;
        }
        
        uint16_t version;
        file.read(reinterpret_cast<char*>(&version), 2);
        
        if (version >= 2) {
            // Skip checksum (4 bytes)
            file.seekg(4, std::ios::cur);
            
            // Skip config vocab_size (4 bytes) and max_length (4 bytes)
            file.seekg(8, std::ios::cur);
            
            // Skip 3 bools (3 bytes)
            file.seekg(3, std::ios::cur);
        }
        
        // Read actual vocab size (number of tokens that follow)
        uint32_t vocab_size;
        file.read(reinterpret_cast<char*>(&vocab_size), 4);
        
        return vocab_size;
        
    } catch (const std::exception& e) {
        std::cerr << "[Config] Error reading vocab size: " << e.what() << std::endl;
        return 0;
    }
}

} // namespace Config
} // namespace GRIM
